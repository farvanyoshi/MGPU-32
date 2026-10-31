"""mgpu_top in lockstep with the Rust engine, through its host and memory ports
(tb/cocotb/test_top.py)."""

import os

import pytest

from rtlutil import RTL, requires_rtl, run_cocotb, sources

pytestmark = requires_rtl
MEM = 1 << 20


def top_sources():
    files = ["common/mgpu_pipe.sv", "common/mgpu_ram.sv"]
    for d in ("units", "core", "top"):
        files += [str(p.relative_to(RTL)) for p in sorted((RTL / d).glob("*.sv"))]
    return sources(*files)


@pytest.mark.parametrize("cores", [1, 2, 4])
def test_top_lockstep(cores):
    env = {"C": str(cores), "MEM_SIZE": str(MEM),
           "MGPU_RTL_FUZZ_SEEDS": os.environ.get("MGPU_RTL_FUZZ_SEEDS", "8" if cores == 1 else "4"),
           "FUZZ_START": os.environ.get("FUZZ_START", "0")}
    run_cocotb("mgpu_top", top_sources(), "test_top",
               parameters={"C": cores, "MEM_SIZE": MEM}, extra_env=env)
