"""mgpu_host_regs against a reference of ISA 13.1 (the Rust engine's reg_read/reg_write
semantics plus the CSR timing of interfaces.md 1): random reads and writes, GO, and
end-of-launch reports from a stand-in dispatcher."""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from mgpu import _core, hostregs as hr

W = int(os.environ.get("W", "8"))
MEM_SIZE = int(os.environ.get("MEM_SIZE", str(1 << 20)))
WRITABLE = [hr.ENTRY, hr.GRID_DIM, hr.BLOCK_DIM] + [hr.ARG(i) for i in range(8)]
ALL = list(range(0, 0x60, 4)) + [0x60, 0x7C, 0xFC]


class Ref:
    def __init__(self):
        g = _core.Gpu(mem_size=MEM_SIZE, max_block_dim=32 * W)
        self.const = {off: g.reg_read(off) for off in (hr.ID, hr.VERSION, hr.MAX_BLOCK_DIM, hr.MEM_SIZE)}
        self.regs = {off: 0 for off in WRITABLE}
        self.status = [0] * 6
        self.cycles = 0
        self.running = False

    def read(self, off, live):
        off &= ~3
        if off in self.const:
            return self.const[off]
        if off in self.regs:
            return self.regs[off]
        if off == hr.GO:
            return int(self.running)
        if hr.STATUS <= off <= hr.ERR_INFO:
            return self.status[(off - hr.STATUS) // 4]
        if off == hr.CYCLES:
            return live if self.running else self.cycles
        return 0

    def write(self, off, value):
        """Returns True if this write starts a launch."""
        off &= ~3
        if self.running:
            return False
        if off in self.regs:
            self.regs[off] = value
        elif off == hr.GO and value & 1:
            self.running, self.status, self.cycles = True, [0] * 6, 0
            return True
        return False


@cocotb.test()
async def random_traffic(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    rng = random.Random(3)
    dut.rst_n.value = 0
    for sig in (dut.csr_valid, dut.csr_write, dut.end_valid):
        sig.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    ref = Ref()
    pending_read = None
    launch_due = None
    launches = ends = reads = 0
    for cycle in range(20000):
        await FallingEdge(dut.clk)
        # outputs registered at the last edge
        if pending_read is not None:
            assert int(dut.csr_rvalid.value) == 1
            got = int(dut.csr_rdata.value)
            assert got == pending_read[1], f"cycle {cycle}: read {pending_read[0]:#x} = {got:#x}, want {pending_read[1]:#x}"
            reads += 1
        else:
            assert int(dut.csr_rvalid.value) == 0
        assert int(dut.launch.value) == int(launch_due == cycle), f"cycle {cycle}: launch pulse"
        for name, off in (("entry", hr.ENTRY), ("grid_dim", hr.GRID_DIM), ("block_dim", hr.BLOCK_DIM)):
            assert int(getattr(dut, name).value) == ref.regs[off], name
        # inputs for this cycle
        live = rng.getrandbits(32)
        dut.live_cycles.value = live
        end = ref.running and launch_due != cycle and rng.random() < 0.05
        dut.end_valid.value = int(end)
        if end:
            st = [rng.choice([0, 1, 2, 3, 4, 5])] + [rng.getrandbits(32) for _ in range(5)]
            cyc = rng.getrandbits(32)
            dut.end_status.value = sum(v << (32 * i) for i, v in enumerate(st))
            dut.end_cycles.value = cyc
        op = rng.random()
        pending_read = None
        dut.csr_valid.value = int(op < 0.8)
        if op < 0.8:
            off = rng.choice(ALL) | rng.randrange(4)
            write = op < 0.35
            dut.csr_write.value = int(write)
            dut.csr_addr.value = off
            if write:
                value = rng.getrandbits(32) if off & ~3 != hr.GO else rng.randrange(4)
                dut.csr_wdata.value = value
                if ref.write(off, value):
                    launch_due = cycle + 1
                    launches += 1
            else:
                pending_read = (off, ref.read(off, live))
        if end:
            ref.running, ref.status, ref.cycles = False, st, cyc
            ends += 1
        await Timer(1, unit="ns")
        assert int(dut.csr_ready.value) == 1
        await RisingEdge(dut.clk)
        if end:
            await Timer(1, unit="ns")
            assert int(dut.irq_done.value) == 1
    assert launches > 50 and ends > 50 and reads > 5000, (launches, ends, reads)
    dut._log.info(f"{reads} reads checked, {launches} launches, {ends} launch ends")
