"""Bulk FPU/SFU vectors through the Verilated units, checked against libmgpu_core.

CI size by default; scale with MGPU_RTL_BULK_N and MGPU_RTL_SWEEP_STRIDE (1 = exhaustive
per binade, about ten minutes).
"""

import os
import shutil
import subprocess

import pytest

from rtlutil import ROOT, requires_rtl

pytestmark = [requires_rtl, pytest.mark.skipif(not (shutil.which("cargo") and shutil.which("g++")),
                                               reason="needs cargo and g++")]


def make(target, **vars):
    args = ["make", "-C", str(ROOT / "tb/cpp"), target] + [f"{k}={v}" for k, v in vars.items()]
    r = subprocess.run(args, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert " 0 mismatches" in out, out[-3000:]
    return out


def test_random_vectors():
    make("bulk", N=os.environ.get("MGPU_RTL_BULK_N", "20000"), SEED=7)


def test_sfu_binade_sweep():
    make("sweep", STRIDE=os.environ.get("MGPU_RTL_SWEEP_STRIDE", "256"))
