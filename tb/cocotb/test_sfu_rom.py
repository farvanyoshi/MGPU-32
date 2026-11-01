"""mgpu_sfu_rom holds exactly the normative tables of spec/sfu."""

import cocotb
from cocotb.triggers import Timer

from mgpu.fp.sfu_tables import TABLES


def s(v, bits):
    return v - (1 << bits) if v >> (bits - 1) else v


@cocotb.test()
async def all_rows(dut):
    for t, name in enumerate(TABLES):
        for i, (c0, c1, c2) in enumerate(TABLES[name]):
            dut.tbl.value = t
            dut.seg.value = i
            await Timer(1, unit="ns")
            got = (int(dut.c0.value), s(int(dut.c1.value), 24), s(int(dut.c2.value), 18))
            assert got == (c0, c1, c2), (name, i, got)
