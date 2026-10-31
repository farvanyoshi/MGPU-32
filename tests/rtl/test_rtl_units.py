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


SFU_SOURCES = ("common/mgpu_pipe.sv", "units/mgpu_round.sv", "units/mgpu_sfu_rom.sv",
               "units/mgpu_sfu_lane.sv", "units/mgpu_sfu.sv")


@pytest.mark.parametrize("lanes", [2, 1])
def test_sfu(lanes):
    run_cocotb("mgpu_sfu", sources(*SFU_SOURCES), "test_sfu", parameters={"S": lanes},
               extra_env={"S": str(lanes), "LAT": "8"})


def test_core_decode():
    run_cocotb("mgpu_core_decode_probe",
               sources("core/mgpu_decode.sv", "../tb/rtl/mgpu_core_decode_probe.sv"),
               "test_core_decode")


def test_srf():
    run_cocotb("mgpu_srf", sources("core/mgpu_srf.sv"), "test_regfiles", parameters={"NR": 5, "NW": 3},
               extra_env={"RF": "srf", "NR": "5", "NW": "3"})


def test_vrf():
    run_cocotb("mgpu_vrf", sources("core/mgpu_vrf.sv"), "test_regfiles", parameters={"NR": 4, "NW": 5},
               extra_env={"RF": "vrf", "NR": "4", "NW": "5"})
