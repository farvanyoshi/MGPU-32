"""mgpu_srf / mgpu_vrf against a Python model: random writes and reads, zero registers,
lane masks, synchronous VRF reads."""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from tbutil import pack, unpack

W = 8
L = int(os.environ.get("L", "8"))
P = 32 // L
PB = max(1, (P - 1).bit_length())


@cocotb.test()
async def srf_random(dut):
    if os.environ.get("RF") != "srf":
        return
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    rng = random.Random(1)
    model = {}
    dut.we.value = 0
    await FallingEdge(dut.clk)
    for _ in range(4000):
        w, r, v = rng.randrange(W), rng.randrange(32), rng.getrandbits(32)
        dut.we.value = 1
        dut.wwarp.value, dut.waddr.value, dut.wdata.value = w, r, v
        if r:
            model[(w, r)] = v
        reads = [(rng.randrange(W), rng.randrange(32)) for _ in range(3)]
        dut.rwarp.value = pack([x[0] for x in reads], 3)
        dut.raddr.value = pack([x[1] for x in reads], 5)
        await RisingEdge(dut.clk)
        await ReadOnly()
        got = unpack(dut.rdata.value, 3)
        for (ww, rr), g in zip(reads, got):
            assert g == (model.get((ww, rr), g) if rr else 0), (ww, rr, hex(g))
        await FallingEdge(dut.clk)


@cocotb.test()
async def vrf_random(dut):
    if os.environ.get("RF") != "vrf":
        return
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    rng = random.Random(2)
    model = {}
    dut.we.value = 0
    addr = lambda w, r, p: (w << (5 + PB)) | (r << PB) | p  # noqa: E731
    await FallingEdge(dut.clk)
    prev = None
    for _ in range(3000):
        w, r, p = rng.randrange(W), rng.randrange(32), rng.randrange(P)
        mask = rng.getrandbits(L)
        data = [rng.getrandbits(32) for _ in range(L)]
        dut.we.value = 1
        dut.waddr.value, dut.wmask.value, dut.wdata.value = addr(w, r, p), mask, pack(data)
        reads = [(rng.randrange(W), rng.randrange(32), rng.randrange(P)) for _ in range(3)]
        dut.raddr.value = pack([addr(*x) for x in reads], 5 + PB + 3)
        # expected read data: state before this cycle's write (read and write same edge)
        snap = {k: list(v) for k, v in model.items()}
        exp = [[0] * L if x[1] == 0 else snap.get(x, None) for x in reads]
        if r:
            cur = model.setdefault((w, r, p), [None] * L)
            for i in range(L):
                if mask >> i & 1:
                    cur[i] = data[i]
        await RisingEdge(dut.clk)
        await ReadOnly()
        got = [unpack(int(dut.rdata.value) >> (k * L * 32), L) for k in range(3)]
        for k in range(3):
            if exp[k] is None:
                continue
            for i in range(L):
                if exp[k][i] is not None:
                    assert got[k][i] == exp[k][i], (reads[k], i, hex(got[k][i]), hex(exp[k][i]))
        prev = got
        await FallingEdge(dut.clk)
    assert prev is not None
