# MGPU-32 RTL

SystemVerilog implementation of the reference microarchitecture ([`docs/uarch`](../uarch/README.md)).
The work is staged over three breakpoints:

| Breakpoint | Contents | State |
|---|---|---|
| 5a `bp5-rtl-units` | Generated ISA package and SFU ROM; integer ALU, FPU and SFU units, verified bit-exactly | in progress |
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

## 4. What is verified

Each unit's testbench streams operands through the unit (back to back, and with bubbles) and checks
two things:
1. every result is bit-identical to the reference implementation (`mgpu._core` for float and SFU
   operations, `mgpu.model` for integer operations);
2. every result appears exactly `LAT` cycles after its input.

See [`docs/uarch/verification.md`](../uarch/verification.md) for the overall plan.
