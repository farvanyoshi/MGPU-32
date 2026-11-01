"""mgpu_vrf (banked, RAM-based) against a model: lane-masked writes from several ports,
pass reads on every copy of every bank with one cycle of latency, never-written lanes
reading 0, and the block-start clear."""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

W, L, NC, NW = 8, int(os.environ.get("L", "8")), 4, 6
P = 32 // L


def pack(vals, width):
    return sum((v & ((1 << width) - 1)) << (width * i) for i, v in enumerate(vals))


@cocotb.test()
async def random_traffic(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    rng = random.Random(5 + L)
    dut.we.value = 0
    dut.re.value = 0
    dut.init.value = 1
    await FallingEdge(dut.clk)
    dut.init.value = 0
    model = {}                                         # (warp, reg, lane) -> value; absent = 0
    expect = None
    checked = 0
    for cyc in range(4000):
        if rng.random() < 0.003:                       # block start
            dut.init.value = 1
            dut.we.value = 0
            dut.re.value = 0
            await ReadOnly()
            await RisingEdge(dut.clk)
            model.clear()
            await FallingEdge(dut.clk)
            dut.init.value = 0
            expect = None
            continue
        # writes: distinct warps, any lanes
        warps = rng.sample(range(W), NW)
        writes = []
        for w in warps:
            en = rng.random() < 0.6
            reg = rng.randrange(1, 32)
            mask = rng.choice([0xFFFFFFFF, rng.getrandbits(32), ((1 << L) - 1) << (L * rng.randrange(P))])
            data = [rng.getrandbits(32) for _ in range(32)]
            writes.append((en, w, reg, mask, data))
        dut.we.value = pack([int(x[0]) for x in writes], 1)
        dut.wwarp.value = pack([x[1] for x in writes], 3)
        dut.wreg.value = pack([x[2] for x in writes], 5)
        dut.wmask.value = pack([x[3] for x in writes], 32)
        dut.wdata.value = sum(pack(x[4], 32) << (1024 * i) for i, x in enumerate(writes))
        # reads: some banks, a pass, a register per copy
        re = [rng.random() < 0.7 for _ in range(W)]
        rpass = [rng.randrange(P) for _ in range(W)]
        rreg = [[rng.randrange(32) for _ in range(NC)] for _ in range(W)]
        dut.re.value = pack([int(x) for x in re], 1)
        dut.rpass.value = pack(rpass, max(1, (P - 1).bit_length()))
        dut.rreg.value = sum(pack(rreg[w], 5) << (5 * NC * w) for w in range(W))
        await ReadOnly()
        # data read in the previous cycle
        if expect is not None:
            data = int(dut.rdata.value)
            for w, c, vals in expect:
                for l in range(L):
                    got = (data >> (32 * (l + L * (c + NC * w)))) & 0xFFFFFFFF
                    assert got == vals[l], f"cycle {cyc}: bank {w} copy {c} lane {l}: {got:#x} != {vals[l]:#x}"
                    checked += 1
        expect = [(w, c, [model.get((w, rreg[w][c], rpass[w] * L + l), 0) for l in range(L)])
                  for w in range(W) if re[w] for c in range(NC)]
        await RisingEdge(dut.clk)
        for en, w, reg, mask, data in writes:
            if en:
                for i in range(32):
                    if mask >> i & 1:
                        model[(w, reg, i)] = data[i]
        await FallingEdge(dut.clk)
    dut._log.info(f"L={L}: {checked} lane reads checked")
