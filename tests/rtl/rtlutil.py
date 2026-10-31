"""Build and run RTL units under cocotb + Verilator (pytest marker `rtl`)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RTL = ROOT / "rtl"
BUILD = ROOT / "build" / "rtl"


# The verilator wheel's verilated.mk leaves CFG_CXXFLAGS_PCH_I empty, so designs large
# enough to use a precompiled header pass its file name to g++ as an input file. A
# command-line variable in MAKEFLAGS overrides it with GCC's flag.
PIP_VERILATOR_MAKEFLAGS = "CFG_CXXFLAGS_PCH_I=-include"


def verilator_root() -> str | None:
    """VERILATOR_ROOT for the pip-bundled Verilator, or None if verilator is on PATH."""
    if shutil.which("verilator") and not os.environ.get("MGPU_FORCE_PIP_VERILATOR"):
        return None
    try:
        import verilator
    except ImportError:
        return ""
    root = os.path.dirname(verilator.__file__)
    return root if os.access(os.path.join(root, "bin", "verilator"), os.X_OK) else ""


def available() -> bool:
    try:
        import cocotb_tools  # noqa: F401
    except ImportError:
        return False
    return verilator_root() != ""


requires_rtl = pytest.mark.skipif(not available(), reason="needs verilator and cocotb (pip install -e '.[rtl]')")


def sources(*names: str) -> list[Path]:
    """Package files first, then the requested rtl/ files."""
    pkgs = [RTL / "pkg/mgpu_isa_pkg.sv", RTL / "pkg/mgpu_fp_pkg.sv", RTL / "pkg/mgpu_core_pkg.sv"]
    return [p for p in pkgs if p.exists()] + [RTL / n for n in names]


def run_cocotb(top: str, srcs: list, test_module: str, parameters: dict | None = None,
               tag: str = "", extra_env: dict | None = None) -> None:
    """Build `top` (cached per parameter set) and run tb/cocotb/<test_module>.py."""
    from cocotb_tools.runner import get_runner
    root = verilator_root()
    if root:
        os.environ["VERILATOR_ROOT"] = root
        os.environ["PATH"] = os.path.join(root, "bin") + os.pathsep + os.environ["PATH"]
        os.environ["MAKEFLAGS"] = PIP_VERILATOR_MAKEFLAGS
    params = parameters or {}
    build_dir = BUILD / (top + tag + "".join(f"_{k}{v}" for k, v in sorted(params.items())))
    runner = get_runner("verilator")
    runner.build(sources=[str(s) for s in srcs], hdl_toplevel=top, build_dir=str(build_dir),
                 parameters=params, build_args=["-Wall", "-Wno-fatal", "-O3"], always=False)
    env = dict(extra_env or {})
    # results go to the build directory so they never land in the source tree
    results = runner.test(hdl_toplevel=top, test_module=test_module, build_dir=str(build_dir),
                          test_dir=str(ROOT / "tb" / "cocotb"), extra_env=env,
                          results_xml=str(build_dir / f"results{tag}.xml"))
    from cocotb_tools.runner import get_results
    total, failed = get_results(results)
    assert failed == 0, f"{failed} of {total} cocotb tests failed for {top} (see {build_dir})"


def lint() -> subprocess.CompletedProcess:
    env = dict(os.environ)
    return subprocess.run(["make", "-C", str(RTL), "lint"], capture_output=True, text=True, env=env)
