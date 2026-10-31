"""mgpu_srf / mgpu_vrf against a Python model: multi-port writes, zero registers, init."""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from tbutil import pack, unpack

W = 8
NR = int(os.environ.get("NR", "4"))
NW = int(os.environ.get("NW", "3"))


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.we.value = 0
    dut.init.value = 0
    await FallingEdge(dut.clk)


@cocotb.test()
async def random_ports(dut):
    vector = os.environ["RF"] == "vrf"
    width = 1024 if vector else 32
    await setup(dut)
    rng = random.Random(1)
    args = [rng.getrandbits(32) for _ in range(8)]
    exe = [rng.getrandbits(32) for _ in range(W)]
    if not vector:
        dut.init_args.value = pack(args)
        dut.init_exec.value = pack(exe)
    dut.init.value = 1
    await FallingEdge(dut.clk)
    dut.init.value = 0
    model = {}
    for w in range(W):
        for r in range(32):
            model[(w, r)] = 0 if vector else (args[r - 1] if 1 <= r <= 8 else exe[w] if r == 31 else 0)
    for _ in range(1500):
        warps = rng.sample(range(W), NW)               # distinct warps per cycle
        writes = [(rng.random() < 0.8, w, rng.randrange(32), rng.getrandbits(width)) for w in warps]
        dut.we.value = pack([int(x[0]) for x in writes], 1)
        dut.wwarp.value = pack([x[1] for x in writes], 3)
        dut.waddr.value = pack([x[2] for x in writes], 5)
        dut.wdata.value = pack([x[3] for x in writes], width)
        reads = [(rng.randrange(W), rng.randrange(32)) for _ in range(NR)]
        dut.rwarp.value = pack([x[0] for x in reads], 3)
        dut.raddr.value = pack([x[1] for x in reads], 5)
        await ReadOnly()
        got = unpack(dut.rdata.value, NR, width)
        for (w, r), g in zip(reads, got):
            assert g == (0 if r == 0 else model[(w, r)]), (w, r)
        await RisingEdge(dut.clk)
        for en, w, r, v in writes:
            if en and r:
                model[(w, r)] = v
        await FallingEdge(dut.clk)
