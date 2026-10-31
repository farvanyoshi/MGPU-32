# MGPU-32 RTL

SystemVerilog implementation of the reference microarchitecture ([`docs/uarch`](../uarch/README.md)).
The work is staged over three breakpoints:

| Breakpoint | Contents | State |
|---|---|---|
| 5a `bp5-rtl-units` | Generated ISA package and SFU ROM; integer ALU, FPU and SFU units, verified bit-exactly | **done** |
| 5b `bp6-rtl-core` | One core: warp table, scheduler, fetch/I-cache, decode, register files, LSU, barriers; lockstep co-simulation | planned |
| 5c `bp7-rtl-top` | Host registers, dispatcher, memory arbiter, atomic unit, multi-core; cycle comparison; synthesis | planned |

## 1. Tools

```bash
pip install -e ".[rtl]"        # cocotb and the verilator wheel (or install verilator yourself)
make -C rtl lint               # verilator --lint-only -Wall on every top-level unit
python -m pytest -m rtl        # build every unit with Verilator and run its cocotb testbench
```

- **Verilator.** `tools/verilator.sh` uses `verilator` from PATH if there is one. Otherwise it uses
  the binary bundled in the `verilator` pip package, with `VERILATOR_ROOT` set for you.
- **Python tests.** The RTL tests are marked `rtl`, and are skipped automatically when Verilator or
  cocotb is missing.
- **Wheel workaround.** The `verilator` wheel ships a `verilated.mk` with an empty
  `CFG_CXXFLAGS_PCH_I`. Designs large enough to use a precompiled header then fail to compile with
  "`Vtop__pch.h.fast: linker input file not found`". The test helper passes
  `MAKEFLAGS=CFG_CXXFLAGS_PCH_I=-include` (GCC's flag) automatically; do the same if you invoke the
  wheel's Verilator by hand.
- **Build cache.** Builds are cached in `build/rtl/<unit>_<parameters>/`; delete that directory to
  force a rebuild.

## 2. Layout

```
rtl/pkg/      packages: mgpu_isa_pkg (generated from spec/isa.toml), mgpu_fp_pkg
rtl/common/   generic blocks (mgpu_pipe)
rtl/units/    execution units; mgpu_sfu_rom.sv is generated from spec/sfu/
tb/cocotb/    cocotb testbenches (Python; references are mgpu._core and mgpu.model)
tb/cpp/       Verilator C++ harnesses linked against libmgpu_core.a for bulk vectors
tests/rtl/    pytest entry points
```

Generated files are never edited by hand. Regenerate them with `python tools/gen_isa.py` and
`python tools/gen_sfu_tables.py`; `--check` verifies them in CI.

## 3. Conventions

- **Language.** Synthesizable SystemVerilog-2017: `always_ff` / `always_comb`, packed types,
  `logic` only, no latches. Clock `clk`, reset `rst_n` (synchronous, active low).
- **Structure.** One module per file, named like the file.
- **Pipelines.** Units follow a valid/data pipeline convention: `in_valid` with operands goes in, and
  `out_valid` with results comes out exactly `LAT` cycles later, with a new input accepted every
  cycle. A unit's datapath is combinational, followed by `mgpu_pipe` stages that give the latency of
  [`core.md`](../uarch/core.md) §7.
- **Clean lint.** The code must lint clean under `verilator -Wall`. A waiver (`lint_off`) is allowed
  only with a comment that explains why.
- **Clock speed.** Timing closure is not a goal yet. The combinational datapaths will be split across
  their pipeline stages when the design is first synthesized for a clock target.

## 4. Units

| Module | Does | Latency | Spec |
|---|---|---|---|
| `mgpu_isa_pkg` (generated) | `op_e` (Rust `Op` order), `unit_of`, `decode_op` ([DEC-1]), SR masks | — | ISA §4–5 |
| `mgpu_fp_pkg` | FTZ, NaN/Inf/zero predicates, ordering keys | — | ISA §8 |
| `mgpu_pipe` | valid/data delay line | `DEPTH` | — |
| `mgpu_int_lane` / `mgpu_valu` | Every integer funct and immediate form, all 16 V_CMP conditions, IMAD, SEL | 2 | ISA §5.1, 5.3, 6.2–6.5 |
| `mgpu_round` | [FP-3] rounding of `(−1)^s · M · 2^k` | combinational | ISA §8 |
| `mgpu_fpu_lane` / `mgpu_fpu` | FADD, FSUB, FMUL, FFMA, FMIN, FMAX, I2F, F2I | 4 | ISA §8 |
| `mgpu_sfu_rom` (generated) | 7 × 128 coefficient rows | — | `sfu.md` |
| `mgpu_sfu_lane` / `mgpu_sfu` | FRCP, FRSQ, FEX2, FLG2 | 8 | `sfu.md` |

The unit wrappers take `L` (or `S`) lanes per cycle and are fully pipelined: a new pass every cycle,
results exactly `LAT` cycles later. The core (Phase 5b) feeds them P passes per vector instruction.

## 5. What is verified

| Check | Volume | Reference |
|---|---|---|
| `make -C rtl lint` | Every unit, `verilator -Wall`, no warnings | — |
| Decoder (`test_decode`) | Every opcode × every value of the full funct field (65,536 words) + 50,000 random words | `mgpu.isa.decode` |
| SFU ROM (`test_sfu_rom`) | All 896 rows | `spec/sfu/*.txt` |
| VALU (`test_valu`) | 60 ops × 512 lane operations, at L = 8 and L = 1 | `mgpu.model`, `mgpu.fp` |
| FPU (`test_fpu`) | 31,194 lane operations (special-value cross products, cancellations, random), at L = 8 and L = 2 | `mgpu._core` |
| SFU (`test_sfu`) | 15,180 operations (specials, every path-selecting exponent, powers of two, random), at S = 2 and S = 1 | `mgpu._core` |
| Bulk random (`tb/cpp`, `make bulk`) | 20,000 vectors per FPU op in CI; 200,000 per op locally (2 million lane results, 0 mismatches) | `libmgpu_core` (`mgpu_fp_op`) |
| SFU binade sweep (`make sweep`) | Every 256th mantissa in CI. Locally at stride 64: 3,801,088 results, 0 mismatches; `STRIDE=1` checks every mantissa (about 243 million results) | `libmgpu_core` |

The cocotb testbenches also check the pipeline: back-to-back issue, bubbles, and that each result
appears exactly `LAT` cycles after its input. Each testbench was checked to fail when a one-line
bug is planted in the unit (for example ANDN computed as AND, or an off-by-one rounding scale).

```bash
make -C tb/cpp bulk N=10000000     # 10 million random vectors per FPU op
make -C tb/cpp sweep STRIDE=1      # every mantissa of every SFU path-selecting binade
```

See [`docs/uarch/verification.md`](../uarch/verification.md) for the overall plan.
