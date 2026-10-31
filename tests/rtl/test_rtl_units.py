"""Build each RTL unit with Verilator and run its cocotb testbench (tb/cocotb)."""

import pytest

from rtlutil import requires_rtl, run_cocotb, sources

pytestmark = requires_rtl


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_pipe(depth):
    run_cocotb("mgpu_pipe", sources("common/mgpu_pipe.sv"), "test_pipe",
               parameters={"DEPTH": depth}, extra_env={"DEPTH": str(depth)})


def test_decode():
    run_cocotb("mgpu_decode_probe", sources("../tb/rtl/mgpu_decode_probe.sv"), "test_decode")


def test_sfu_rom():
    run_cocotb("mgpu_sfu_rom", sources("units/mgpu_sfu_rom.sv"), "test_sfu_rom")
