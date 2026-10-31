# MGPU-32

A 32-bit educational GPU architecture with 32-lane SIMD execution, explicit lane masks, warp
scheduling and software-managed shared memory. This repository holds a complete implementation: a
normative ISA specification, a reference simulator, a toolchain, a microarchitecture specification
and SystemVerilog RTL.

## Status

The project is built in phases. Each phase ends at a git tag (a *breakpoint*) where the work is
reviewed before continuing.

| Breakpoint | Tag | Contents | State |
|---|---|---|---|
| 1 | `bp1-isa-spec` | Normative ISA v1.0, machine-readable instruction table, bit-exact SFU | **done** |
| 2 | `bp2-simulator` | Python reference model + fast Rust simulator (PyO3 + C ABI), differential fuzzing | **done** |
| 3 | `bp3-toolchain` | Assembler, disassembler, runner, host API, kernel suite | **done** |
| 4 | `bp4-uarch` | Microarchitecture specification + cycle-level timing model | **done** |
| 5 | `bp5-rtl-units` | RTL execution units (ALU, FPU, SFU) verified with cocotb/Verilator | **done** |
| 6 | `bp6-rtl-core` | Full RTL core, lockstep trace co-simulation | planned |
| 7 | `bp7-rtl-top` / `v1.0` | Multi-core top level, host interface, synthesis report | planned |

## The architecture in one paragraph

A kernel launches over a grid of blocks; each block is split into warps of 32 lanes. A warp has one
PC and executes one instruction at a time across all 32 lanes. A 32-bit EXEC mask (scalar register
s31) decides which lanes write. Scalar instructions (`S_*`) run once per warp and ignore EXEC;
vector instructions (`V_*`) run per lane under EXEC. Divergence is handled in software by saving,
narrowing and restoring EXEC. There are 24 opcodes in four fixed 32-bit formats, IEEE-754 binary32
floats with flush-to-zero and a bit-exact special-function unit, 16 KiB of shared memory per block
with a barrier, and a weak memory model.

## Documentation

| Document | What it is |
|---|---|
| [`docs/isa/MGPU-32-ISA.md`](docs/isa/MGPU-32-ISA.md) | The normative ISA specification (start here) |
| [`docs/isa/encoding-tables.md`](docs/isa/encoding-tables.md) | Opcode map, funct tables, bit patterns (generated) |
| [`docs/isa/sfu.md`](docs/isa/sfu.md) | Bit-exact FRCP / FRSQ / FEX2 / FLG2 algorithm |
| [`docs/isa/equivalence.md`](docs/isa/equivalence.md) | What any two conforming implementations must agree on |
| [`docs/isa/clarifications.md`](docs/isa/clarifications.md) | Every change from the original design note, with rationale |
| [`docs/sim/README.md`](docs/sim/README.md) | Simulators: Python/C APIs, reference schedule, trace format, fuzzing |
| [`docs/toolchain/assembler.md`](docs/toolchain/assembler.md) | Assembly language: syntax, expressions, directives, pseudo-instructions, output formats |
| [`docs/toolchain/tools.md`](docs/toolchain/tools.md) | `mgpu-run`, `mgpu-dis` and the Python host API |
| [`kernels/README.md`](kernels/README.md) | Example kernels and what their statistics show |
| [`docs/uarch/README.md`](docs/uarch/README.md) | Reference microarchitecture: parameters, top level, core, interfaces, timing contract |
| [`docs/uarch/performance.md`](docs/uarch/performance.md) | Timing model and the design-space measurements behind the parameters |
| [`docs/uarch/verification.md`](docs/uarch/verification.md) | How the RTL will be verified against the simulators |
| [`docs/rtl/README.md`](docs/rtl/README.md) | The SystemVerilog: layout, conventions, how to lint and test, what is verified |

## Repository layout

```
spec/isa.toml              machine-readable instruction table (single source of truth)
spec/sfu/*.txt             normative SFU coefficient tables
docs/isa/, docs/sim/,      ISA specification; simulator guide; assembler and tools guides;
  docs/toolchain/,         microarchitecture specification
  docs/uarch/
kernels/                   example kernels (saxpy, reduce, matmul, conv3x3, scan, ...)
python/mgpu/               Python package: ISA tables, bit-exact float32, reference model,
                           assembler/disassembler (mgpu.asm), Device host API, mgpu-run,
                           fuzzers (python -m mgpu.fuzz)
rust/mgpu-core/            fast simulator engine and cycle-level timing model
                           (also a C library: include/mgpu_core.h)
rust/mgpu-py/              Python bindings (mgpu._core)
rtl/                       SystemVerilog (packages generated from spec/; units: VALU, FPU, SFU)
tb/                        RTL testbenches: cocotb (Python) and a Verilator C++ bulk harness
tools/                     generators (gen_isa.py, gen_sfu_tables.py), sfu_accuracy.py, bench.py,
                           sweep.py (design-space sweep with the timing model)
tests/                     pytest suite (ISA, float, simulator, assembler, kernels, fuzzing, C ABI)
```

## Getting started

You need Python 3.11 or newer and a Rust toolchain (https://rustup.rs).

```bash
pip install -e ".[test]"        # builds the Rust engine; installs mgpu-as, mgpu-dis, mgpu-run
python -m pytest                # ~720 tests; simulator and kernel tests run on both backends
```

Run a kernel from the command line:

```bash
python -c "import numpy as n; n.save('x.npy', n.arange(1000, dtype=n.float32)); n.save('y.npy', n.ones(1000, n.float32))"
mgpu-run kernels/saxpy.s --grid 8 --block 128 --arg 1000 --arg f32:2.0 --arg in:x.npy --arg inout:y.npy --stats
```

Or from Python:

```python
import numpy as np
from mgpu.asm import assemble_file
from mgpu.device import Device

dev = Device()                                   # Device("model") for the pure-Python reference
x = dev.upload(np.arange(1000, dtype=np.float32))
y = dev.upload(np.ones(1000, dtype=np.float32))
dev.run(assemble_file("kernels/saxpy.s"), 8, 128, 1000, 2.0, x, y)
print(y.read()[:4])                              # [1. 3. 5. 7.]
```

Other useful commands:

```bash
mgpu-as kernels/matmul.s --lst matmul.lst        # assemble, with a listing
mgpu-dis matmul.bin                              # back to reassemblable source
python -m mgpu.fuzz --seeds 2500                 # 10,000 random programs: model vs Rust engine
python tools/bench.py                            # simulator throughput
mgpu-run kernels/saxpy.s ... --timing --warps 8  # cycle-level timing of the reference design
python tools/sweep.py                            # design-space sweep (warps, lanes, memory latency)
pip install -e ".[rtl]" && make -C rtl lint      # RTL: Verilator + cocotb, lint
python -m pytest -m rtl                          # RTL unit testbenches (bit-exact vs the engine)
```

Generated files must stay in sync with their sources:

```bash
python tools/gen_isa.py --check
python tools/gen_sfu_tables.py --check
```

## License

MIT — see [LICENSE](LICENSE).
