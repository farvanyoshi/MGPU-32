"""mgpu_core in lockstep with the Rust engine (tb/cocotb/test_core.py)."""

from rtlutil import RTL, requires_rtl, run_cocotb, sources

pytestmark = requires_rtl


def core_sources():
    files = ["common/mgpu_pipe.sv", "common/mgpu_ram.sv"]
    files += [str(p.relative_to(RTL)) for p in sorted((RTL / "units").glob("*.sv"))]
    files += [str(p.relative_to(RTL)) for p in sorted((RTL / "core").glob("*.sv"))]
    return sources(*files)


def test_core_lockstep():
    run_cocotb("mgpu_core", core_sources(), "test_core", parameters={"W": 8, "L": 8, "S": 2},
               extra_env={"W": "8", "L": "8", "S": "2"})


def test_core_fuzz():
    """Fuzz programs in lockstep; MGPU_RTL_FUZZ_SEEDS sets the batch size per mode."""
    import os
    env = {"W": "8", "L": "8", "S": "2",
           "MGPU_RTL_FUZZ_SEEDS": os.environ.get("MGPU_RTL_FUZZ_SEEDS", "16"),
           "FUZZ_START": os.environ.get("FUZZ_START", "0"),
           "FUZZ_MODES": os.environ.get("FUZZ_MODES", "diff,raw,eq")}
    run_cocotb("mgpu_core", core_sources(), "test_core_fuzz", parameters={"W": 8, "L": 8, "S": 2},
               extra_env=env)
