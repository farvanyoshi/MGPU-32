"""Lockstep co-simulation of mgpu_core against the Rust engine (docs/uarch/verification.md 3).

The testbench plays dispatcher and memory. The engine runs the same launch in driven mode:
for every record on the core's retire port it sets CLOCK to the record's issue clock,
steps the same warp and compares the records under rule E4 (CommitRecord.warp_key). At the
end it checks that both finished the same way and compares global memory (E1-E3).
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from mgpu import _core, hostregs as hr
from mgpu.isa.disasm import disasm
from mgpu.trace import CommitRecord, from_tuple

from memory import Memory

PKG = Path(__file__).resolve().parents[2] / "rtl/pkg/mgpu_core_pkg.sv"


def _channels() -> list[str]:
    """Retire channel names in index order (the R_* parameters of mgpu_core_pkg.sv)."""
    found = re.findall(r"\bR_(\w+)\s*=\s*(\d+)", PKG.read_text())
    return [n.lower() for n, _ in sorted(found, key=lambda x: int(x[1]))]


CHANNELS = _channels()
NRET = len(CHANNELS)


def _struct_fields(name: str) -> list[tuple[str, int]]:
    """(field, width) of a packed struct in mgpu_core_pkg.sv, most significant first."""
    text = PKG.read_text()
    body = re.search(r"typedef struct packed \{([^}]*)\}\s*" + name + r";", text).group(1)
    fields = []
    for line in body.splitlines():
        line = line.split("//")[0].strip()
        m = re.match(r"logic\s*((?:\[\d+:\d+\])*)\s*([\w\s,]+);", line)
        if not m:
            continue
        width = 1
        for hi, lo in re.findall(r"\[(\d+):(\d+)\]", m.group(1)):
            width *= int(hi) - int(lo) + 1
        for n in m.group(2).split(","):
            fields.append((n.strip(), width))
    return fields


class Struct:
    """Field decoder for a packed struct of mgpu_core_pkg.sv."""

    def __init__(self, name: str):
        self.fields = _struct_fields(name)
        self.width = sum(w for _, w in self.fields)

    def decode(self, value: int) -> dict:
        out, pos = {}, self.width
        for name, width in self.fields:
            pos -= width
            out[name] = (value >> pos) & ((1 << width) - 1)
        return out


RETIRE = Struct("retire_t")
MREQ = Struct("mreq_t")


def lanes(value: int) -> tuple:
    return tuple((value >> (32 * i)) & 0xFFFFFFFF for i in range(32))


@dataclass
class Launch:
    words: list
    entry: int = 0x1000
    grid: int = 1
    block: int = 32
    args: list = field(default_factory=list)
    memory: dict = field(default_factory=dict)      # address -> words
    mem_size: int = 1 << 20


@dataclass
class Result:
    cycles: int                  # launch cycles, as CYCLES counts them (core.md 11.6)
    retired: int
    fault: tuple | None
    records: list


def model_cycles(ln: Launch, warps: int, lanes: int = 8, sfu_lanes: int = 2,
                 t_mem: int = 100, t_seg: int = 8) -> int:
    """CYCLES of the launch under the timing model (docs/uarch/core.md 11), one core."""
    g = _core.Gpu(mem_size=ln.mem_size, max_block_dim=32 * warps)
    if ln.words:
        g.mem_write(ln.entry, b"".join((w & 0xFFFFFFFF).to_bytes(4, "little") for w in ln.words))
    for addr, words in ln.memory.items():
        g.mem_write(addr, b"".join((w & 0xFFFFFFFF).to_bytes(4, "little") for w in words))
    for off, v in {hr.ENTRY: ln.entry, hr.GRID_DIM: ln.grid, hr.BLOCK_DIM: ln.block,
                   **{hr.ARG(i): a for i, a in enumerate(ln.args)}}.items():
        g.reg_write(off, v)
    rep = g.launch_timed(cores=1, warps=warps, lanes=lanes, sfu_lanes=sfu_lanes,
                         mem_latency=t_mem, seg_cycles=t_seg)
    return rep["cycles"]


class LockstepError(AssertionError):
    pass


_clock_task = None


def ensure_clock(dut) -> None:
    """Start the clock once per cocotb test (a test's tasks end with it)."""
    global _clock_task
    if _clock_task is None or _clock_task.done():
        _clock_task = cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())


class Lockstep:
    def __init__(self, dut, launch: Launch, warps: int, t_mem: int = 100, t_seg: int = 8):
        self.dut, self.ln, self.W = dut, launch, warps
        # With one warp per block nothing depends on the schedule, so atomic values are
        # compared too; otherwise rule E4 (warp_key, C-37) applies.
        self.strict = launch.block <= 32
        self.mem = Memory(launch.mem_size, t_mem, t_seg)
        self.gpu = _core.Gpu(mem_size=launch.mem_size, max_block_dim=32 * warps)
        if launch.words:
            self._load(launch.entry, launch.words)
        for addr, words in launch.memory.items():
            self._load(addr, words)
        regs = {hr.ENTRY: launch.entry, hr.GRID_DIM: launch.grid, hr.BLOCK_DIM: launch.block}
        regs.update({hr.ARG(i): a for i, a in enumerate(launch.args)})
        for off, v in regs.items():
            self.gpu.reg_write(off, v)
        if not self.gpu.begin():
            raise ValueError("launch rejected by the engine")
        self.records: list[tuple[int, CommitRecord]] = []
        # MGPU_LOCKSTEP_DUMP=<file>: append "issue_cycle block warp pc" per retired instruction
        path = os.environ.get("MGPU_LOCKSTEP_DUMP")
        self.dump = open(path, "a") if path else None
        self.block_start = 0

    def _load(self, addr, words):
        self.mem.write_words(addr, words)
        self.gpu.mem_write(addr, b"".join((w & 0xFFFFFFFF).to_bytes(4, "little") for w in words))

    def _rtl_record(self, r: dict, block: int) -> CommitRecord:
        rec = CommitRecord(0, block, r["warp"], r["pc"], r["word"])
        if r["sw_valid"]:
            rec.swrite = (r["sw_reg"], r["sw_val"])
        if r["vw_valid"]:
            rec.vwrite = (r["vw_reg"], lanes(r["vw_val"]))
        rec.exec_after = r["exec_after"]
        space = 1 if r["mw_shared"] else 0
        for lane in range(32):
            if r["mw_lanes"] >> lane & 1:
                rec.mwrites.append((space, (r["mw_addr"] >> (32 * lane)) & 0xFFFFFFFF,
                                    (r["mw_val"] >> (32 * lane)) & 0xFFFFFFFF))
        rec.event = r["event_"]
        if rec.event == 3:
            rec.fault = (r["fault_code"], r["fault_lane"], r["fault_info"])
        return rec

    def _check(self, now: int, chan: int, r: dict, block: int) -> CommitRecord:
        rtl = self._rtl_record(r, block)
        self.gpu.set_clock(r["clock"])
        try:
            eng = from_tuple(self.gpu.step(block, rtl.warp))
        except Exception as e:
            raise LockstepError(f"cycle {now}: engine cannot step ({block}, {rtl.warp}): {e}") from e
        if self._key(rtl) != self._key(eng):
            raise LockstepError(
                f"cycle {now}, {CHANNELS[chan]} channel, block {block} warp {rtl.warp} "
                f"pc {rtl.pc:#x}: {disasm(eng.word, eng.pc)}\n"
                f"  rtl:    {self._show(rtl)}\n  engine: {self._show(eng)}")
        self.records.append((now, rtl))
        if self.dump:
            self.dump.write(f"{self.block_start + r['clock']} {block} {rtl.warp} {rtl.pc:#x}\n")
        return rtl

    def _key(self, r: CommitRecord) -> tuple:
        if self.strict:
            return (r.pc, r.word, r.swrite, r.vwrite, r.exec_after, tuple(r.mwrites), r.event, r.fault)
        return r.warp_key()

    @staticmethod
    def _show(r: CommitRecord) -> str:
        vw = r.vwrite and (r.vwrite[0], [hex(x) for x in r.vwrite[1]])
        return (f"word={r.word:#010x} swrite={r.swrite} vwrite={vw} exec={r.exec_after:#010x} "
                f"mw={r.mwrites} event={r.event} fault={r.fault}")

    async def run(self, max_cycles: int = 200_000, reset: bool = True,
                  kill_at: int | None = None) -> Result:
        """Run the launch to completion. With kill_at, abort the core in that cycle and
        return without the end-of-launch checks. reset=False keeps the core's state from a
        previous run (it must be idle)."""
        dut, ln = self.dut, self.ln
        ensure_clock(dut)
        dut.start.value = 0
        dut.kill.value = 0
        dut.halt_in.value = 0          # a lone core: no other core can fault
        dut.fault_go.value = 1
        dut.new_launch.value = 0
        dut.if_resp_valid.value = 0
        dut.rd_valid.value = 0
        dut.at_valid.value = 0
        dut.ack_valid.value = 0
        if reset:
            dut.rst_n.value = 0
            for _ in range(2):
                await RisingEdge(dut.clk)
            dut.rst_n.value = 1
        dut.mem_last.value = ln.mem_size - 4
        dut.entry.value = ln.entry
        dut.block_dim.value = ln.block
        dut.grid_dim.value = ln.grid
        dut.num_warps.value = (ln.block + 31) // 32
        dut.args.value = sum((a & 0xFFFFFFFF) << (32 * i) for i, a in enumerate(ln.args))

        next_block, block, done_blocks = 0, None, 0
        fault, end, block_start = None, None, 0
        for now in range(max_cycles):
            await FallingEdge(dut.clk)
            await Timer(1, unit="ns")
            if kill_at is not None and now == kill_at + 1:
                dut.kill.value = 0
                assert int(dut.idle.value), "core not idle after kill"
                return Result(now, len(self.records), None, self.records)
            if kill_at is not None and now == kill_at:
                dut.kill.value = 1
                dut.start.value = 0
                await RisingEdge(dut.clk)
                continue
            # ---- dispatch (core.md 11.1): cycle 1, or the first idle cycle of the core
            if int(dut.block_done.value):
                done_blocks += 1
                end = now - 1                      # the block completed in the previous cycle
                if done_blocks == ln.grid:
                    break
            start = now >= 1 and next_block < ln.grid and (int(dut.idle.value) or int(dut.block_done.value))
            dut.start.value = int(bool(start))
            if start:
                block, block_start = next_block, now
                self.block_start = now
                dut.new_launch.value = int(next_block == 0)
                dut.block_id.value = next_block
                next_block += 1
            resp = self.mem.ifetch_response(now)
            dut.if_resp_valid.value = int(resp is not None)
            if resp:
                dut.if_resp_id.value, dut.if_resp_data.value = resp
            m = self.mem
            rd, at, ack = m.read_resp.pop(now, None), m.atomic_resp.pop(now, None), m.acks.pop(now, None)
            dut.rd_valid.value = int(rd is not None)
            if rd:
                dut.rd_warp.value, dut.rd_seg.value, dut.rd_data.value = rd
            dut.at_valid.value = int(at is not None)
            if at:
                dut.at_warp.value, dut.at_seg.value, dut.at_old.value = at
            dut.ack_valid.value = int(ack is not None)
            if ack is not None:
                dut.ack_warp.value = ack
            await Timer(1, unit="ns")
            # ---- retire records, in channel order
            valid = int(dut.ret_valid.value)
            if valid:
                allrec = int(dut.ret.value)
                for ch in range(NRET):
                    if valid >> ch & 1:
                        r = RETIRE.decode(allrec >> (ch * RETIRE.width))
                        rec = self._check(now, ch, r, block)
                        if rec.fault:
                            fault = rec.fault
                            end = block_start + r["clock"] + 1     # issue cycle + 1 (11.6)
                if fault:
                    break
            req = int(dut.lsu_req.value)
            if req >> (MREQ.width - 1):                    # LSU first, then I-fetch (11.4)
                self.mem.lsu_request(now, MREQ.decode(req))
            if int(dut.if_req_valid.value):
                self.mem.ifetch_request(now, int(dut.if_req_id.value), int(dut.if_req_addr.value))
                if self.dump:
                    self.dump.write(f"F {now} {int(dut.if_req_addr.value):#x} {int(dut.if_req_id.value)}\n")
            await RisingEdge(dut.clk)
        else:
            raise LockstepError(f"no completion after {max_cycles} cycles")
        dut.start.value = 0
        self._finish(fault)
        return Result(end, len(self.records), fault, self.records)

    def _finish(self, fault) -> None:
        g = self.gpu
        if g.running:
            raise LockstepError("the core finished but the engine is still running")
        status = g.reg_read(hr.STATUS)
        want = fault[0] if fault else hr.OK
        if status != want:
            raise LockstepError(f"engine STATUS {status}, RTL {want}")
        size = self.ln.mem_size
        if bytes(self.mem.data) != g.mem_read(0, size):
            eng = g.mem_read(0, size)
            first = next(i for i in range(size) if eng[i] != self.mem.data[i])
            raise LockstepError(f"global memory differs from address {first & ~3:#x}")
