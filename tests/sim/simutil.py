"""Shared helpers for the simulator tests."""

import struct

import pytest

from mgpu.device import Device, rust_available

from asm import assemble

CODE = 0x1000
DATA = 0x10000
MEM = 1 << 20

BACKEND_PARAMS = [pytest.param("model"),
                  pytest.param("rust", marks=pytest.mark.skipif(not rust_available(),
                                                                reason="mgpu._core not built"))]


def f2b(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def b2f(x: int) -> float:
    return struct.unpack("<f", struct.pack("<I", x))[0]


class Run:
    """Assemble `src`, run it, and keep the device for inspection."""

    def __init__(self, backend, src, grid=1, block=32, args=(), data=None, mem=MEM, **params):
        params.setdefault("keep_state", True)
        params.setdefault("trace", True)
        self.dev = Device(backend, mem_size=mem, **params)
        self.words = assemble(src, CODE)
        self.dev.write_words(CODE, self.words)
        for addr, words in (data or {}).items():
            self.dev.write_words(addr, words)
        self.res = self.dev.launch(CODE, grid, block, args)

    def s(self, r, block=0, warp=0):
        return self.dev.warp_state(block, warp)[1][r]

    def v(self, r, block=0, warp=0):
        return self.dev.warp_state(block, warp)[2][r]

    def mem(self, addr, n):
        return self.dev.read_words(addr, n)

