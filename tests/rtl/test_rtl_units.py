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


@pytest.mark.parametrize("lanes", [8, 1])
def test_valu(lanes):
    run_cocotb("mgpu_valu", sources("common/mgpu_pipe.sv", "units/mgpu_int_lane.sv", "units/mgpu_valu.sv"),
               "test_valu", parameters={"L": lanes}, extra_env={"L": str(lanes), "LAT": "2"})


FPU_SOURCES = ("common/mgpu_pipe.sv", "units/mgpu_round.sv", "units/mgpu_fpu_lane.sv", "units/mgpu_fpu.sv")


@pytest.mark.parametrize("lanes", [8, 2])
def test_fpu(lanes):
    run_cocotb("mgpu_fpu", sources(*FPU_SOURCES), "test_fpu", parameters={"L": lanes},
               extra_env={"L": str(lanes), "LAT": "4"})
