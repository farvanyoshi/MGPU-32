"""mgpu_core in lockstep with the Rust engine (tb/cocotb/test_core.py)."""

from rtlutil import RTL, requires_rtl, run_cocotb, sources

pytestmark = requires_rtl


def core_sources():
    files = ["common/mgpu_pipe.sv"]
    files += [str(p.relative_to(RTL)) for p in sorted((RTL / "units").glob("*.sv"))]
    files += [str(p.relative_to(RTL)) for p in sorted((RTL / "core").glob("*.sv"))]
    return sources(*files)


def test_core_lockstep():
    run_cocotb("mgpu_core", core_sources(), "test_core", parameters={"W": 8, "L": 8, "S": 2},
               extra_env={"W": "8", "L": "8", "S": "2"})
