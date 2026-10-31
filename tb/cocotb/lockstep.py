"""Lockstep co-simulation of mgpu_core against the Rust engine (docs/uarch/verification.md 3).

The testbench plays dispatcher and memory. The engine runs the same launch in driven mode:
for every record on the core's retire port it sets CLOCK to the record's issue clock,
steps the same warp and compares the records under rule E4 (CommitRecord.warp_key). At the
end it checks that both finished the same way and compares global memory (E1-E3).
"""

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
NRET = 7
CHANNELS = ["salu", "valu", "fpu", "sfu", "lsu_a", "lsu_b", "fault"]


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


RETIRE_FIELDS = _struct_fields("retire_t")
RETIRE_WIDTH = sum(w for _, w in RETIRE_FIELDS)


def decode_retire(value: int) -> dict:
    out, pos = {}, RETIRE_WIDTH
    for name, width in RETIRE_FIELDS:
        pos -= width
        out[name] = (value >> pos) & ((1 << width) - 1)
    return out


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
        self.mem = Memory(launch.mem_size, t_mem, t_seg)
        self.gpu = _core.Gpu(mem_size=launch.mem_size, max_block_dim=32 * warps)
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
        if rtl.warp_key() != eng.warp_key():
            raise LockstepError(
                f"cycle {now}, {CHANNELS[chan]} channel, block {block} warp {rtl.warp} "
                f"pc {rtl.pc:#x}: {disasm(eng.word, eng.pc)}\n"
                f"  rtl:    {self._show(rtl)}\n  engine: {self._show(eng)}")
        self.records.append((now, rtl))
        return rtl

    @staticmethod
    def _show(r: CommitRecord) -> str:
        vw = r.vwrite and (r.vwrite[0], [hex(x) for x in r.vwrite[1]])
        return (f"word={r.word:#010x} swrite={r.swrite} vwrite={vw} exec={r.exec_after:#010x} "
                f"mw={r.mwrites} event={r.event} fault={r.fault}")

    async def run(self, max_cycles: int = 200_000) -> Result:
        dut, ln = self.dut, self.ln
        ensure_clock(dut)
        dut.rst_n.value = 0
        dut.start.value = 0
        dut.new_launch.value = 0
        dut.if_resp_valid.value = 0
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
                dut.new_launch.value = int(next_block == 0)
                dut.block_id.value = next_block
                next_block += 1
            resp = self.mem.ifetch_response(now)
            dut.if_resp_valid.value = int(resp is not None)
            if resp:
                dut.if_resp_id.value, dut.if_resp_data.value = resp
            await Timer(1, unit="ns")
            # ---- retire records, in channel order
            valid = int(dut.ret_valid.value)
            if valid:
                allrec = int(dut.ret.value)
                for ch in range(NRET):
                    if valid >> ch & 1:
                        r = decode_retire(allrec >> (ch * RETIRE_WIDTH))
                        rec = self._check(now, ch, r, block)
                        if rec.fault:
                            fault = rec.fault
                            end = block_start + r["clock"] + 1     # issue cycle + 1 (11.6)
                if fault:
                    break
            if int(dut.if_req_valid.value):
                self.mem.ifetch_request(now, int(dut.if_req_id.value), int(dut.if_req_addr.value))
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
