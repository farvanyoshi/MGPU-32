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
| 2 | `bp2-simulator` | Python reference model + fast Rust simulator (PyO3 + C ABI), differential tests | planned |
| 3 | `bp3-toolchain` | Assembler, disassembler, runner, host API, kernel suite | planned |
| 4 | `bp4-uarch` | Microarchitecture specification + cycle-level timing model | planned |
| 5 | `bp5-rtl-units` | RTL execution units (ALU, FPU, SFU) verified with cocotb/Verilator | planned |
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

## Repository layout

```
spec/isa.toml              machine-readable instruction table (single source of truth)
spec/sfu/*.txt             normative SFU coefficient tables
docs/isa/                  ISA specification
python/mgpu/               Python package (ISA tables, bit-exact float32, SFU; simulator next)
tools/gen_isa.py           validates isa.toml, generates tables and docs/isa/encoding-tables.md
tools/gen_sfu_tables.py    regenerates the SFU tables deterministically
tools/sfu_accuracy.py      exhaustive SFU accuracy sweep
tests/                     pytest suite
```

## Checks

The suite needs Python ≥ 3.11, plus pytest and numpy.

```bash
python3 tools/gen_isa.py --check        # instruction table valid, generated files current, encodings match
python3 tools/gen_sfu_tables.py --check # SFU tables reproduce bit-for-bit
python3 tools/sfu_accuracy.py           # exhaustive SFU accuracy sweep (~2 minutes)
python3 -m pytest tests
```

## License

MIT — see [LICENSE](LICENSE).
