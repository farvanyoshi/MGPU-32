"""Front-end host API over either simulator backend.

    dev = Device(backend="rust")          # or "model" for the pure-Python reference
    dev.write_words(0x1000, program)
    res = dev.launch(entry=0x1000, grid=4, block=64, args=[n, a_ptr])
    print(res)                            # STATUS and fault details
"""

from dataclasses import dataclass

from . import hostregs as H
from .trace import CommitRecord, from_tuple

BACKENDS = ("rust", "model")


def rust_available() -> bool:
    try:
        from . import _core  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class LaunchResult:
    status: int
    err_pc: int
    err_block: int
    err_warp: int
    err_lane: int
    err_info: int
    steps: int

    @property
    def ok(self) -> bool:
        return self.status == H.OK

    def __str__(self) -> str:
        if self.ok:
            return f"ok ({self.steps} instructions)"
        name = H.ERROR_NAMES.get(self.status, f"error {self.status}")
        return (f"{name} at pc={self.err_pc:#010x} block={self.err_block} warp={self.err_warp} "
                f"lane={self.err_lane} info={self.err_info:#010x}")


class Device:
    def __init__(self, backend: str = "rust", **params):
        if backend == "model":
            from .model import Gpu
        elif backend == "rust":
            from ._core import Gpu
        else:
            raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
        self.backend = backend
        self.gpu = Gpu(**params)

    # ---- memory -----------------------------------------------------------------------------
    def write(self, addr: int, data) -> None:
        """Write bytes, a bytes-like object, or a numpy array (raw little-endian bytes)."""
        if hasattr(data, "tobytes"):
            data = data.tobytes()
        self.gpu.mem_write(addr, bytes(data))

    def read(self, addr: int, n: int) -> bytes:
        return self.gpu.mem_read(addr, n)

    def write_words(self, addr: int, words) -> None:
        self.write(addr, b"".join((w & 0xFFFFFFFF).to_bytes(4, "little") for w in words))

    def read_words(self, addr: int, n: int) -> list[int]:
        b = self.read(addr, 4 * n)
        return [int.from_bytes(b[i:i + 4], "little") for i in range(0, 4 * n, 4)]

    def read_array(self, addr: int, n: int, dtype="float32"):
        import numpy as np
        dt = np.dtype(dtype)
        return np.frombuffer(self.read(addr, n * dt.itemsize), dtype=dt).copy()

    # ---- host registers and launch -----------------------------------------------------------
    def reg_read(self, off: int) -> int:
        return self.gpu.reg_read(off)

    def reg_write(self, off: int, value: int) -> None:
        self.gpu.reg_write(off, value)

    def result(self) -> LaunchResult:
        r = self.gpu.reg_read
        return LaunchResult(r(H.STATUS), r(H.ERR_PC), r(H.ERR_BLOCK), r(H.ERR_WARP),
                            r(H.ERR_LANE), r(H.ERR_INFO), self.gpu.steps)

    def launch(self, entry: int, grid: int, block: int, args=()) -> LaunchResult:
        """Write the launch registers, set GO and run the kernel to completion."""
        args = list(args) + [0] * (8 - len(args))
        assert len(args) == 8, "at most 8 kernel arguments"
        self.reg_write(H.ENTRY, entry)
        self.reg_write(H.GRID_DIM, grid)
        self.reg_write(H.BLOCK_DIM, block)
        for n, a in enumerate(args):
            self.reg_write(H.ARG(n), a)
        self.reg_write(H.GO, 1)
        return self.result()

    # ---- inspection -----------------------------------------------------------------------------
    @property
    def trace(self) -> list[CommitRecord]:
        t = self.gpu.trace
        return [r if isinstance(r, CommitRecord) else from_tuple(r) for r in t]

    def warp_state(self, block: int, warp: int):
        """(pc, s[32], v[32][32], state); after the kernel ends this needs keep_state=True."""
        return self.gpu.warp_state(block, warp)
