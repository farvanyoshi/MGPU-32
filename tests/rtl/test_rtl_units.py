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


@pytest.mark.parametrize("lanes", [8, 32])
def test_vrf(lanes):
    run_cocotb("mgpu_vrf", sources("common/mgpu_ram.sv", "core/mgpu_vrf.sv"), "test_vrf",
               parameters={"L": lanes}, extra_env={"L": str(lanes)})


VEXEC_SOURCES = (FPU_SOURCES + SFU_SOURCES[2:] + ("units/mgpu_int_lane.sv", "units/mgpu_valu.sv",
                 "core/mgpu_decode.sv", "core/mgpu_vexec.sv", "../tb/rtl/mgpu_vexec_probe.sv"))


@pytest.mark.parametrize("kind,lanes,lat", [(0, 8, 2), (0, 32, 2), (1, 8, 4), (1, 4, 4), (2, 2, 8)])
def test_vexec(kind, lanes, lat):
    # the VALU and FPU take one register-file slice per cycle; the SFU narrower ones
    params = {"KIND": kind, "LANES": lanes, "LAT": lat, "L": lanes if kind < 2 else 8}
    run_cocotb("mgpu_vexec_probe", sources(*dict.fromkeys(VEXEC_SOURCES)), "test_vexec",
               parameters=params, extra_env={k: str(v) for k, v in params.items()})


@pytest.mark.parametrize("warps", [8, 3])
def test_fetch(warps):
    run_cocotb("mgpu_fetch", sources("core/mgpu_decode.sv", "core/mgpu_fetch.sv"), "test_fetch",
               parameters={"W": warps}, extra_env={"W": str(warps)})


def test_host_regs():
    run_cocotb("mgpu_host_regs", sources("top/mgpu_host_regs.sv"), "test_host_regs",
               parameters={"W": 8, "MEM_SIZE": 1 << 20}, extra_env={"W": "8", "MEM_SIZE": str(1 << 20)})


@pytest.mark.parametrize("cores", [4, 1])
def test_dispatcher(cores):
    run_cocotb("mgpu_dispatcher", sources("top/mgpu_dispatcher.sv"), "test_dispatcher",
               parameters={"C": cores}, extra_env={"C": str(cores)})


@pytest.mark.parametrize("cores,t_mem", [(4, 20), (1, 3), (2, 100)])
def test_memsys(cores, t_mem):
    run_cocotb("mgpu_memsys_probe", sources("top/mgpu_mem_arbiter.sv", "top/mgpu_atomic_unit.sv",
                                            "../tb/rtl/mgpu_memsys_probe.sv"), "test_memsys",
               parameters={"C": cores}, extra_env={"C": str(cores), "T_MEM": str(t_mem)})


def test_axil():
    run_cocotb("mgpu_axil_probe", sources("top/mgpu_axil_csr.sv", "top/mgpu_host_regs.sv",
                                          "../tb/rtl/mgpu_axil_probe.sv"), "test_axil")
