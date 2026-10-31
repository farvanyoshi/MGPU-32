# MGPU-32 tools: `mgpu-dis`, `mgpu-run` and the host API

`pip install -e .` installs three commands: `mgpu-as` (see [assembler.md](assembler.md)),
`mgpu-dis` and `mgpu-run`. All three work with both simulator backends (`rust`, the default, and
`model`). See [`docs/sim/README.md`](../sim/README.md) for the simulators themselves.

## 1. `mgpu-run`: run a kernel from the command line

```bash
python -c "import numpy as n; n.save('x.npy', n.arange(1000, dtype=n.float32)); n.save('y.npy', n.ones(1000, n.float32))"
mgpu-run kernels/saxpy.s --grid 8 --block 128 \
         --arg 1000 --arg f32:2.0 --arg in:x.npy --arg inout:y.npy --stats
```
```
saxpy @ 0x1000  grid=8 block=128 [rust]: ok (384 instructions)
instructions        384
vector instructions 288 (SIMD utilization 98.2% of 32 lanes)
...
saved y.npy (1000 x float32)
```

The program may be a `.s` file, which is assembled at `--base` (default `0x1000`), or a `.bin`/`.hex`
image loaded at `--base` and started at `--entry` (default: the base). `--kernel NAME` selects one of
several `.kernel`s.

Each `--arg` becomes the next kernel argument, s1 through s8:

| Form | Passes |
|---|---|
| `123`, `0x10`, `-1` | An integer |
| `f32:2.5` | The binary32 bits of a float |
| `in:FILE.npy` | Address of an uploaded copy of the array |
| `inout:FILE.npy` | Same as `in:`; the array is saved back to `FILE.npy` after the run |
| `out:FILE.npy:COUNT[:DTYPE]` | Address of a zeroed buffer (default dtype `float32`), saved after the run |
| `zeros:COUNT[:DTYPE]` | Address of a zeroed scratch buffer |
| `sym:NAME` | Value of a program symbol, for example a data table assembled with the kernel |

Other options:

| Option | Effect |
|---|---|
| `--backend model` | Use the pure-Python reference instead of the Rust engine |
| `--trace [N]` | Print the first N retired instructions: step, block, warp, PC, disassembly, EXEC |
| `--stats` | Instruction mix and SIMD utilization, the fraction of the 32 lanes active in vector instructions |
| `--dump ADDR:COUNT[:DTYPE]` | Print memory after the run; ADDR may be a symbol |
| `--mem-size`, `--max-steps` | Simulator parameters |

The exit status is 0 on success, 1 for usage, assembly or file errors, 2 if the kernel faulted (the
fault is printed with the disassembled instruction) and 3 on a watchdog timeout.

## 2. `mgpu-dis`: disassemble an image

```bash
mgpu-as kernels/saxpy.s -o saxpy.bin --sym saxpy.sym.json
mgpu-dis saxpy.bin --sym saxpy.sym.json          # reassemblable source, labels restored
mgpu-dis saxpy.bin --listing                     # address / word / instruction
mgpu-dis saxpy.hex                               # $readmemh images too (base from the @ line)
```

The default output always reassembles, with `mgpu-as` at the same base, to the **identical** image:
- Illegal words, illegal special-register reads, and *non-canonical* encodings (instructions with
  ignored fields set, rule DEC-2) are written as `.word`, with the decoded instruction in a comment.
- With a symbol map, branch targets are shown as label names.

## 3. Host API (Python)

```python
import numpy as np
from mgpu.asm import assemble_file
from mgpu.device import Device, KernelFault

prog = assemble_file("kernels/matmul.s")
dev = Device()                                    # Device("model") for the reference model
A = dev.upload(np.random.rand(64, 32).astype(np.float32))
B = dev.upload(np.random.rand(32, 48).astype(np.float32))
C = dev.zeros((64, 48))
dev.run(prog, 4 * 3, 256, A, B, C, 64, 48, 32)    # grid, block, then up to 8 arguments
print(C.read())                                   # numpy array, shape (64, 48)
```

| Call | What it does |
|---|---|
| `Device(backend="rust", **params)` | A simulated GPU; `params` as in `docs/sim/README.md` (`mem_size`, `trace`, …) |
| `dev.load(program)` | Writes an assembled program at its base. `run` does this automatically |
| `dev.upload(array)`, `dev.zeros(shape, dtype)` | Allocate global memory and return a `Buffer` |
| `dev.alloc(nbytes, align=256)`, `dev.free_all()` | The bump allocator behind buffers. The heap starts at `0x10000`, above any loaded program |
| `Buffer.read()`, `Buffer.write(a)`, `Buffer.addr` | Download, upload, address. A Buffer passed as a kernel argument passes its address |
| `dev.run(program, grid, block, *args, kernel=None, check=True)` | Launch. Arguments are ints, floats (as f32 bits) or Buffers. Raises `KernelFault` (with `.result` and the disassembled faulting instruction) when STATUS ≠ 0 |
| `dev.launch(entry, grid, block, args)` | Low-level launch through the host registers; returns a `LaunchResult` |
| `dev.trace`, `mgpu.stats.trace_stats(trace, block)` | Commit records, and their instruction mix and SIMD utilization (needs `trace=True`) |

A typical edit-run loop is to write a kernel in `kernels/`, try it with `mgpu-run`, and then pin
its behaviour with a test in `tests/kernels/` that compares against numpy on both backends.
