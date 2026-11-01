"""Lockstep co-simulation of the whole GPU (mgpu_top) against the Rust engine.

The testbench is the host and the memory: it loads memory, writes the launch registers and
GO through the CSR port, serves the 128-bit memory port (memory.BeatMemory), and waits for
irq_done. The engine runs the same launch in driven mode with one resident block per core;
every record on every core's retire port is checked as in lockstep.py. At the end the
testbench reads STATUS, ERR_* and CYCLES through the CSR port and compares them with the
engine and the timing model, and compares global memory (unless the launch faulted: memory
after a fault is undefined, [FLT-1])."""

from cocotb.triggers import FallingEdge, RisingEdge, Timer

from mgpu import hostregs as hr

from lockstep import NRET, RETIRE, Launch, Lockstep, LockstepError, ensure_clock, model_cycles
from memory import BeatMemory


class TopResult:
    def __init__(self, regs, cycles, retired, fault):
        self.regs, self.cycles, self.retired, self.fault = regs, cycles, retired, fault

    @property
    def status(self):
        return self.regs[hr.STATUS]


class TopLockstep(Lockstep):
    def __init__(self, dut, launch: Launch, cores: int, warps: int = 8, t_mem: int = 100,
                 lanes: int = 8, sfu_lanes: int = 2):
        self.lanes, self.sfu_lanes = lanes, sfu_lanes
        super().__init__(dut, launch, warps, t_mem, 8, cores, allow_rejected=True)

    @staticmethod
    def _make_memory(size, t_mem, t_seg):
        return BeatMemory(size, t_mem)

    async def csr(self, write, off, value=0):
        """One CSR transaction; returns the read data."""
        dut = self.dut
        await FallingEdge(dut.clk)
        dut.csr_valid.value, dut.csr_write.value = 1, int(write)
        dut.csr_addr.value, dut.csr_wdata.value = off, value & 0xFFFFFFFF
        await FallingEdge(dut.clk)
        dut.csr_valid.value = 0
        return None if write else int(dut.csr_rdata.value)

    async def run(self, max_cycles: int = 1_000_000, reset: bool = True) -> TopResult:
        dut, ln, C = self.dut, self.ln, self.C
        ensure_clock(dut)
        dut.csr_valid.value = 0
        dut.mrd_valid.value = 0
        dut.mack_valid.value = 0
        dut.mreq_ready.value = 1
        if reset:
            dut.rst_n.value = 0
            for _ in range(2):
                await RisingEdge(dut.clk)
            dut.rst_n.value = 1
        regs = {hr.ENTRY: ln.entry, hr.GRID_DIM: ln.grid, hr.BLOCK_DIM: ln.block}
        regs.update({hr.ARG(i): a for i, a in enumerate(ln.args)})
        for off, v in regs.items():
            await self.csr(True, off, v)
        # GO: the launch pulse follows in the next cycle, which is launch cycle 0
        await FallingEdge(dut.clk)
        dut.csr_valid.value, dut.csr_write.value, dut.csr_addr.value, dut.csr_wdata.value = 1, 1, hr.GO, 1
        await RisingEdge(dut.clk)
        cur_block = [None] * C
        fault = None
        for now in range(max_cycles):
            await FallingEdge(dut.clk)
            dut.csr_valid.value = 0
            ready = self.mem.drive(dut, now)
            await Timer(1, unit="ns")
            valid = int(dut.ret_valid.value)
            if valid:
                allrec = int(dut.ret.value)
                for c in range(C):
                    for ch in range(NRET):
                        k = c * NRET + ch
                        if valid >> k & 1:
                            r = RETIRE.decode(allrec >> (k * RETIRE.width))
                            rec = self._check(now, ch, r, cur_block[c])
                            if rec.fault:
                                fault = rec.fault
            starts = int(dut.u_disp.start.value)
            if starts:
                ids = int(dut.u_disp.block_id.value)
                for c in range(C):
                    if starts >> c & 1:
                        cur_block[c] = (ids >> (32 * c)) & 0xFFFFFFFF
            self.mem.observe(now, dut, ready)
            if int(dut.overflow.value):
                raise LockstepError(f"cycle {now}: memory request queue overflow")
            if int(dut.irq_done.value):
                break
            await RisingEdge(dut.clk)
        else:
            raise LockstepError(f"no end of launch after {max_cycles} cycles")
        # read back the result registers through the CSR port
        out = {}
        for off in (hr.GO, hr.STATUS, hr.ERR_PC, hr.ERR_BLOCK, hr.ERR_WARP, hr.ERR_LANE,
                    hr.ERR_INFO, hr.CYCLES):
            out[off] = await self.csr(False, off)
        self._finish_top(out, fault)
        return TopResult(out, out[hr.CYCLES], len(self.records), fault)

    def _finish_top(self, out, fault):
        g = self.gpu
        if out[hr.GO]:
            raise LockstepError("GO still reads 1 after irq_done")
        if g.running:
            raise LockstepError("the GPU finished but the engine is still running")
        for off in (hr.STATUS, hr.ERR_PC, hr.ERR_BLOCK, hr.ERR_WARP, hr.ERR_LANE, hr.ERR_INFO):
            if out[off] != g.reg_read(off):
                raise LockstepError(f"{hr.NAMES[off]}: RTL {out[off]:#x}, engine {g.reg_read(off):#x}")
        want = model_cycles(self.ln, self.W, self.lanes, self.sfu_lanes, self.mem.t_mem, 8, self.C)
        if out[hr.CYCLES] != want:
            raise LockstepError(f"CYCLES: RTL {out[hr.CYCLES]}, timing model {want}")
        if out[hr.STATUS] == hr.OK:
            size = self.ln.mem_size
            eng = g.mem_read(0, size)
            if bytes(self.mem.data) != eng:
                first = next(i for i in range(size) if eng[i] != self.mem.data[i])
                raise LockstepError(f"global memory differs from address {first & ~3:#x}")
