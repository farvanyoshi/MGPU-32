"""Front-end host API over either simulator backend.

High level (docs/toolchain/tools.md):

    from mgpu.asm import assemble_file
    prog = assemble_file("kernels/saxpy.s")
    dev = Device()                         # Rust engine; Device("model") for the reference
    x, y = dev.upload(xs), dev.upload(ys)  # numpy arrays -> Buffers in GPU memory
    dev.run(prog, grid, block, n, 2.5, x, y)   # floats become f32 bits, Buffers addresses
    result = y.read()

Low level: write_words/read_words, reg_read/reg_write, launch(entry, grid, block, args).
"""

import struct
from dataclasses import dataclass

from . import hostregs as H
from .trace import CommitRecord, from_tuple

HEAP_START = 0x10000

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


class KernelFault(Exception):
    """A launch ended with STATUS != 0."""

    def __init__(self, result: "LaunchResult", detail: str = ""):
        self.result = result
        super().__init__(str(result) + (f"\n  {detail}" if detail else ""))


@dataclass(eq=False)
class Buffer:
    """A region of GPU global memory holding a typed array. Passing a Buffer as a kernel
    argument passes its address."""
    device: "Device"
    addr: int
    shape: tuple
    dtype: str

    @property
    def count(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        import numpy as np
        return self.count * np.dtype(self.dtype).itemsize

    def read(self):
        """Download the buffer as a numpy array of its shape and dtype."""
        return self.device.read_array(self.addr, self.count, self.dtype).reshape(self.shape)

    def write(self, array) -> None:
        import numpy as np
        a = np.ascontiguousarray(array, dtype=self.dtype)
        assert a.size == self.count, "size mismatch"
        self.device.write(self.addr, a)

    def __index__(self) -> int:
        return self.addr

    def __int__(self) -> int:
        return self.addr

    def __repr__(self) -> str:
        return f"Buffer(addr={self.addr:#x}, shape={self.shape}, dtype={self.dtype})"


def kernel_arg(a) -> int:
    """Convert a kernel argument to its 32-bit register value."""
    if isinstance(a, Buffer):
        return a.addr
    if isinstance(a, bool):
        return int(a)
    if isinstance(a, float) or type(a).__name__ in ("float32", "float64", "float16"):
        return struct.unpack("<I", struct.pack("<f", float(a)))[0]
    if hasattr(a, "__index__"):
        v = a.__index__()
        if not -(1 << 31) <= v < 1 << 32:
            raise ValueError(f"kernel argument {v} does not fit in 32 bits")
        return v & 0xFFFFFFFF
    raise TypeError(f"unsupported kernel argument {a!r} (int, float or Buffer)")


class Device:
    """`timing`: None for functional simulation, or a dict of timing-model parameters
    (cores, warps, lanes, sfu_lanes, mem_latency, seg_cycles, icache_lines; {} = defaults).
    With timing, every launch runs under the cycle-level model (Rust backend only) and
    `timing_report` holds the result."""

    def __init__(self, backend: str = "rust", timing: dict | None = None, **params):
        if backend == "model":
            from .model import Gpu
        elif backend == "rust":
            from ._core import Gpu
        else:
            raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
        self.backend = backend
        if timing is not None and backend != "rust":
            raise ValueError("the timing model needs the rust backend")
        self.timing = timing
        self.timing_report: dict | None = None
        self.gpu = Gpu(**params)
        self.mem_size = self.gpu.mem_size
        self.heap_ptr = HEAP_START
        self.loaded: dict[int, tuple] = {}      # id(program) -> (base, end, program)
        self.allocations: list[tuple] = []      # (addr, end) of live buffers

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
        if self.timing is not None:
            self.timing_report = self.gpu.launch_timed(**self.timing)
        else:
            self.reg_write(H.GO, 1)
        return self.result()

    # ---- programs, buffers and kernels ---------------------------------------------------------
    def load(self, program):
        """Write an assembled program at its base address; the heap moves above it."""
        if id(program) in self.loaded:
            return program
        lo, hi = program.base, program.end
        if hi > self.mem_size:
            raise ValueError(f"program [{lo:#x}, {hi:#x}) does not fit in {self.mem_size:#x} bytes")
        if any(lo < end and hi > addr for addr, end in self.allocations):
            raise ValueError(f"program [{lo:#x}, {hi:#x}) overlaps allocated buffers")
        self.write(program.base, program.image)
        self.loaded[id(program)] = (lo, hi, program)
        if hi > HEAP_START:
            self.heap_ptr = max(self.heap_ptr, hi)
        return program

    def alloc(self, nbytes: int, align: int = 256) -> int:
        """Reserve `nbytes` of global memory (bump allocation) and return the address."""
        addr = self.heap_ptr + (-self.heap_ptr % align)
        if addr + nbytes > self.mem_size:
            raise MemoryError(f"out of GPU memory: {nbytes} bytes at {addr:#x}, "
                              f"memory size {self.mem_size:#x}")
        self.heap_ptr = addr + nbytes
        self.allocations.append((addr, addr + nbytes))
        return addr

    def upload(self, array, dtype=None) -> Buffer:
        """Copy a numpy array (or sequence) into newly allocated GPU memory."""
        import numpy as np
        a = np.ascontiguousarray(array, dtype=dtype)
        buf = Buffer(self, self.alloc(max(a.nbytes, 4)), a.shape, a.dtype.str)
        self.write(buf.addr, a)
        return buf

    def zeros(self, shape, dtype="float32") -> Buffer:
        import numpy as np
        return self.upload(np.zeros(shape, dtype=dtype))

    def free_all(self) -> None:
        """Forget every allocation (buffers created before become invalid)."""
        self.allocations.clear()
        self.heap_ptr = max([HEAP_START] + [hi for lo, hi, _ in self.loaded.values()])

    def run(self, program, grid: int, block: int, *args, kernel: str | None = None,
            check: bool = True) -> LaunchResult:
        """Load `program` if needed and launch `kernel` (or its only kernel). Arguments may
        be ints, floats (passed as f32 bits) or Buffers (passed as addresses). Raises
        KernelFault if the kernel faults, unless check=False."""
        if isinstance(program, int):
            entry = program
        else:
            self.load(program)
            entry = program.entry(kernel)
        if len(args) > 8:
            raise ValueError("at most 8 kernel arguments")
        res = self.launch(entry, grid, block, [kernel_arg(a) for a in args])
        if check and not res.ok:
            raise KernelFault(res, self.describe_fault(res))
        return res

    def describe_fault(self, res: LaunchResult) -> str:
        from .isa import disasm
        if res.ok or res.status == H.E_LAUNCH:
            return ""
        if res.err_pc > self.mem_size - 4:
            return "instruction: <outside memory>"
        word = self.read_words(res.err_pc, 1)[0]
        return f"instruction: {disasm(word, res.err_pc)}"

    # ---- inspection -----------------------------------------------------------------------------
    @property
    def trace(self) -> list[CommitRecord]:
        t = self.gpu.trace
        return [r if isinstance(r, CommitRecord) else from_tuple(r) for r in t]

    def warp_state(self, block: int, warp: int):
        """(pc, s[32], v[32][32], state); after the kernel ends this needs keep_state=True."""
        return self.gpu.warp_state(block, warp)
