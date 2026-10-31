"""mgpu_axil_csr in front of mgpu_host_regs (mgpu_axil_probe): random AXI4-Lite reads and
writes, with AW and W offered at different times and random B/R back-pressure, against the
host-register reference of test_host_regs."""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from mgpu import hostregs as hr

import test_host_regs as thr

thr.MEM_SIZE = 1 << 20
ALL = thr.ALL


async def cycle(dut):
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)


@cocotb.test()
async def random_transactions(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    rng = random.Random(9)
    for s in ("s_awvalid", "s_wvalid", "s_bready", "s_arvalid", "s_rready", "end_valid"):
        getattr(dut, s).value = 0
    dut.s_wstrb.value = 0xF
    dut.rst_n.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)
    ref = thr.Ref()
    reads = writes = launches = 0
    for _ in range(3000):
        # a running launch may end at any time between transactions
        if ref.running and rng.random() < 0.3:
            st = [rng.randrange(6)] + [rng.getrandbits(32) for _ in range(5)]
            cyc = rng.getrandbits(32)
            dut.end_valid.value = 1
            dut.end_status.value = sum(v << (32 * i) for i, v in enumerate(st))
            dut.end_cycles.value = cyc
            await cycle(dut)
            dut.end_valid.value = 0
            ref.running, ref.status, ref.cycles = False, st, cyc
        off = rng.choice(ALL)
        if rng.random() < 0.45:
            value = rng.getrandbits(32) if off != hr.GO else rng.randrange(4)
            aw_at, w_at = rng.randrange(3), rng.randrange(3)
            t = 0
            while True:
                dut.s_awvalid.value, dut.s_awaddr.value = int(t >= aw_at), off
                dut.s_wvalid.value, dut.s_wdata.value = int(t >= w_at), value
                await Timer(1, unit="ns")
                if int(dut.s_awready.value):
                    assert int(dut.s_wready.value)
                    break
                await cycle(dut)
                t += 1
            launched = ref.write(off, value)
            await cycle(dut)
            dut.s_awvalid.value = dut.s_wvalid.value = 0
            if launched:
                assert int(dut.launch.value) == 1, "GO did not raise launch"
                launches += 1
            for _ in range(rng.randrange(3)):
                assert int(dut.s_bvalid.value) == 1
                await cycle(dut)
            dut.s_bready.value = 1
            await Timer(1, unit="ns")
            assert int(dut.s_bvalid.value) == 1 and int(dut.s_bresp.value) == 0
            await cycle(dut)
            dut.s_bready.value = 0
            writes += 1
        else:
            dut.s_arvalid.value, dut.s_araddr.value = 1, off
            await Timer(1, unit="ns")
            while not int(dut.s_arready.value):
                await cycle(dut)
                await Timer(1, unit="ns")
            want = ref.read(off, 0)
            await cycle(dut)
            dut.s_arvalid.value = 0
            while not int(dut.s_rvalid.value):
                await cycle(dut)
            for _ in range(rng.randrange(3)):
                await cycle(dut)
            dut.s_rready.value = 1
            await Timer(1, unit="ns")
            assert int(dut.s_rvalid.value) == 1 and int(dut.s_rresp.value) == 0
            got = int(dut.s_rdata.value)
            assert got == want, f"read {off:#x}: {got:#x}, want {want:#x}"
            await cycle(dut)
            dut.s_rready.value = 0
            reads += 1
    assert launches > 20 and reads > 1000
    dut._log.info(f"{reads} reads and {writes} writes through AXI4-Lite, {launches} launches")
