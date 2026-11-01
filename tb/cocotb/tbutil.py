"""Shared cocotb helpers: clock and reset, and a pipelined-unit stream driver.

Cycle model used by `stream`: inputs are driven at the falling edge, the rising edge
captures them, and outputs are sampled (ReadOnly) right after that rising edge. An input
driven in iteration e of a unit with latency LAT (registers) is therefore observed in
iteration e + max(LAT - 1, 0).
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


async def start(dut, cycles_in_reset: int = 2):
    """Start the clock, hold reset, and return at a falling edge with reset released."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    for _ in range(cycles_in_reset):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1


def pack(values, width: int = 32) -> int:
    """Pack lane values (lane 0 in the low bits) into one integer."""
    out = 0
    for i, v in enumerate(values):
        out |= (int(v) & ((1 << width) - 1)) << (width * i)
    return out


def unpack(value, lanes: int, width: int = 32) -> list:
    v = int(value)
    return [(v >> (width * i)) & ((1 << width) - 1) for i in range(lanes)]


async def stream(dut, items, apply, read, latency: int, bubble_every: int = 0) -> list:
    """Drive `items` (one per cycle, with an idle cycle every `bubble_every` cycles if set),
    check every output appears exactly `latency` cycles after its input, and return the
    outputs in input order."""
    due: list[int] = []
    outputs: list = []
    lag = max(latency - 1, 0)
    it, i = 0, 0
    while len(outputs) < len(items):
        if i < len(items) and not (bubble_every and it % bubble_every == bubble_every - 1):
            apply(dut, items[i])
            dut.in_valid.value = 1
            due.append(it + lag)
            i += 1
        else:
            dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.out_valid.value):
            k = len(outputs)
            assert k < len(due) and due[k] == it, f"output {k} in cycle {it}, expected {due[k] if k < len(due) else '-'}"
            outputs.append(read(dut))
        else:
            assert not due[len(outputs):] or due[len(outputs)] != it, f"output {len(outputs)} missing in cycle {it}"
        await FallingEdge(dut.clk)
        it += 1
        assert it < 4 * len(items) + latency + 16, "unit stopped producing outputs"
    dut.in_valid.value = 0
    return outputs
