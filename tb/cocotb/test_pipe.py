"""mgpu_pipe: latency, bubbles and reset."""

import os
import random

import cocotb
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from tbutil import start, stream

DEPTH = int(os.environ["DEPTH"])


@cocotb.test()
async def latency_and_bubbles(dut):
    await start(dut)
    rng = random.Random(DEPTH)
    data = [rng.getrandbits(32) for _ in range(50)]

    def apply(d, x):
        d.in_data.value = x

    for bubbles in (0, 3):
        out = await stream(dut, data, apply, lambda d: int(d.out_data.value), DEPTH, bubbles)
        assert out == data


@cocotb.test()
async def reset_clears_valid(dut):
    await start(dut)
    if DEPTH == 0:
        return
    dut.in_valid.value = 1
    dut.in_data.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.out_valid.value) == 0
