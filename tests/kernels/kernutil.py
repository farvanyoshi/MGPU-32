"""Helpers for the kernel-suite tests (kernels/*.s)."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from mgpu.asm import assemble_file
from mgpu.device import Device, rust_available

ROOT = Path(__file__).resolve().parents[2]

# (backend, scale): the model is ~150x slower, so it runs smaller problems.
BACKENDS = [pytest.param(("model", "small"), id="model"),
            pytest.param(("rust", "large"), id="rust",
                         marks=pytest.mark.skipif(not rust_available(), reason="mgpu._core not built"))]


@lru_cache(maxsize=None)
def kernel(name: str):
    return assemble_file(ROOT / "kernels" / f"{name}.s")


def device(backend, **params) -> Device:
    return Device(backend[0], mem_size=16 << 20, **params)


def size(backend, small: int, large: int) -> int:
    return small if backend[1] == "small" else large


def f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def bits(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).view(np.uint32)
