"""Compile and run the C ABI smoke test against libmgpu_core.a."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (shutil.which("cargo") and shutil.which("gcc")), reason="needs cargo and gcc")
def test_c_smoke(tmp_path):
    subprocess.run(["cargo", "build", "--release", "-q", "-p", "mgpu-core"],
                   cwd=ROOT / "rust", check=True)
    exe = tmp_path / "smoke"
    subprocess.run(["gcc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", ROOT / "include",
                    ROOT / "tests/c/smoke.c", ROOT / "rust/target/release/libmgpu_core.a",
                    "-lpthread", "-ldl", "-lm", "-o", exe], check=True)
    r = subprocess.run([exe], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "OK", r.stdout + r.stderr
