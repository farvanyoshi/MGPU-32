# MGPU-32 RTL

SystemVerilog implementation of the reference microarchitecture ([`docs/uarch`](../uarch/README.md)).
The work is staged over three breakpoints:

| Breakpoint | Contents | State |
|---|---|---|
| 5a `bp5-rtl-units` | Generated ISA package and SFU ROM; integer ALU, FPU and SFU units, verified bit-exactly | **done** |
| 5b `bp6-rtl-core` | One core: warp table, scheduler, fetch/I-cache, decode, register files, LSU, barriers; lockstep co-simulation, cycle-exact against the timing model | **done** |
| 5c `bp7-rtl-top` | Host registers, dispatcher, memory arbiter, atomic unit, multi-core; cycle comparison; synthesis | planned |

## 1. Tools

```bash
pip install -e ".[rtl]"        # cocotb and the verilator wheel (or install verilator yourself)
make -C rtl lint               # verilator --lint-only -Wall on every top-level unit
python -m pytest -m rtl        # units, then the core in lockstep with the Rust engine
MGPU_RTL_FUZZ_SEEDS=200 python -m pytest -m rtl -k fuzz   # a larger fuzz soak
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
rtl/pkg/      packages: mgpu_isa_pkg (generated from spec/isa.toml), mgpu_fp_pkg, mgpu_core_pkg
rtl/common/   generic blocks (mgpu_pipe)
rtl/units/    execution units; mgpu_sfu_rom.sv is generated from spec/sfu/
rtl/core/     the core (mgpu_core and its blocks)
tb/cocotb/    cocotb testbenches (Python; references are mgpu._core and mgpu.model);
              lockstep.py, memory.py and directed.py drive the core
tb/rtl/       small wrappers that expose internal types on plain ports for testing
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
results exactly `LAT` cycles later. The core feeds them P passes per vector instruction.

## 5. The core (`rtl/core`)

`mgpu_core` (parameters `W`, `L`, `S`, `LINES`; defaults 8, 8, 2, 16) runs one block, following
[`core.md`](../uarch/core.md):

| Module | Does |
|---|---|
| `mgpu_core_pkg` | `ctrl_t`, the issue packet `ipkt_t`, the retire record `retire_t`, the LSU types, fault codes, retire channels |
| `mgpu_decode` | Word → `ctrl_t`: illegal ([DEC-1], GETSR legality), unit, registers, extended immediate, `sb`, destinations |
| `mgpu_fetch` | Round-robin fetch, 16 × 128 B direct-mapped I-cache with overlapping fills tagged by warp slot, decode stage, per-warp instruction buffers, fetch-fault markers |
| `mgpu_srf`, `mgpu_vrf` | Register files: full-width reads, one write port per writer, block-start initialization |
| `mgpu_salu` | Scalar ALU (through `mgpu_int_lane`), branches, NOP/FENCE/EXIT/BAR |
| `mgpu_vexec` | VALU, FPU or SFU with its P-pass sequencer; merged 32-lane record on the completion cycle; V_CMP masks; V_SHFL/V_READLANE |
| `mgpu_agen` | LSU analysis at issue: lane addresses, precise fault, segment count and order, bank-conflict degree and rounds |
| `mgpu_lsu` | Segment requests, per-warp load/atomic slots, shared memory in bank rounds, shared atomics lane by lane |
| `mgpu_core` | Warp table, scheduler with per-unit free-at counters, operand read, packet pipeline, writeback, barrier and fence, stores outstanding, faults, retire port |

**Choices worth knowing:**
- **Operands at issue.** The register files are read in full (all 32 lanes) in the issue cycle, so
  the LSU's segment and conflict counts, and hence its occupancy, are known when the scheduler needs
  them. The P address passes of `core.md` §8.1 remain as LSU occupancy. The register files are flop
  arrays here; their block-RAM mapping is a Phase 5c synthesis task.
- **Retire port.** One record per completed instruction, on one channel per writer (SALU, VALU,
  FPU, SFU, four LSU channels, faults), so no channel ever needs two records in a cycle. The LSU
  channels are stores (at their last request), shared accesses and empty global accesses (after the
  occupancy), global loads, and global atomics (both at the last response + 1).
- **Faults** are detected at issue. The core stops issuing, lets the instructions in flight retire,
  then reports the fault record.
- **Memory ports** are segment wide (1024-bit data, one request per cycle, separate READ-data,
  ATOMIC and WRITE-ack response channels; [`interfaces.md`](../uarch/interfaces.md) §4.1). The
  Phase 5c arbiter converts them to the 128-bit beats of §2. Global atomics are executed by the
  memory side, in lane order (the testbench model now, the atomic unit in 5c).
- **Barriers.** A warp arrives once its BAR has completed and its stores are acknowledged. The
  barrier releases in the cycle every live, non-exited warp has arrived, and released warps may issue
  in that cycle. The barrier PC is taken when the first BAR issues, as the timing model does.

**Lockstep testbench** (`tb/cocotb/lockstep.py`): the testbench plays dispatcher and memory
(`memory.py`, the 11.4 accept rule, LSU before I-fetch within a cycle). The Rust engine runs the same
launch in driven mode. For every record on the retire port, the testbench sets the engine's CLOCK to
the record's issue clock (`Gpu.set_clock`), steps the same warp, and compares:
- rule E4 (`warp_key`) for multi-warp blocks;
- the full record, atomic values included, for single-warp blocks.

At the end it compares STATUS and all of global memory, and it compares CYCLES with `launch_timed`.

To investigate a failure, rerun one fuzz program with `MGPU_RTL_FUZZ_SEEDS=1 FUZZ_START=<seed>
FUZZ_MODES=<mode>`. `MGPU_LOCKSTEP_DUMP=<file>` logs every retired instruction's issue cycle and every
I-cache fill. The cycle comparison found two timing-model bugs this way, both fixed in the model:
- it served memory requests at issue instead of in request order;
- it prefetched the BAR again, instead of PC + 4, while a warp waited at a barrier.

## 6. What is verified

| Check | Volume | Reference |
|---|---|---|
| `make -C rtl lint` | Every unit, `verilator -Wall`, no warnings | — |
| Decoder (`test_decode`) | Every opcode × every value of the full funct field (65,536 words) + 50,000 random words | `mgpu.isa.decode` |
| SFU ROM (`test_sfu_rom`) | All 896 rows | `spec/sfu/*.txt` |
| VALU (`test_valu`) | 60 ops × 512 lane operations, at L = 8 and L = 1 | `mgpu.model`, `mgpu.fp` |
| FPU (`test_fpu`) | 31,194 lane operations (special-value cross products, cancellations, random), at L = 8 and L = 2 | `mgpu._core` |
| SFU (`test_sfu`) | 15,180 operations (specials, every path-selecting exponent, powers of two, random), at S = 2 and S = 1 | `mgpu._core` |
| Bulk random (`tb/cpp`, `make bulk`) | 20,000 vectors per FPU op in CI; 200,000 per op locally (2 million lane results, 0 mismatches) | `libmgpu_core` (`mgpu_fp_op`) |
| SFU binade sweep (`make sweep`) | Every 256th mantissa in CI. **Exhaustive locally (`STRIDE=1`): every mantissa of 29 binades, 243,269,632 results, 0 mismatches** | `libmgpu_core` |

The cocotb testbenches also check the pipeline: back-to-back issue, bubbles, and that each result
appears exactly `LAT` cycles after its input. Each testbench was checked to fail when a one-line
bug is planted in the unit (for example ANDN computed as AND, or an off-by-one rounding scale).

```bash
make -C tb/cpp bulk N=10000000     # 10 million random vectors per FPU op
make -C tb/cpp sweep STRIDE=1      # every mantissa of every SFU path-selecting binade
```

**Core (`test_core`, `test_core_fuzz`), every program in lockstep and cycle-exact against the timing
model:**

| Programs | Volume |
|---|---|
| Directed core tests | Scalar and vector chains, every scalar and vector op group, divergence and loops, GETSR (CLOCK included), illegal-word and fetch faults, global segment counts 1/2/32, store collisions, shared conflict degrees 1/2/32 and broadcast, global and shared atomics, EXEC = 0 on every LSU path, all memory fault kinds, BAR after outstanding stores, a barrier completed by EXIT, FENCE, kill and restart without reset |
| `tests/sim` directed programs (harvested by `directed.py`) | 135 programs |
| Kernel suite (`tests/kernels`, small sizes) | 30 launches, all ten kernels including the faulting `bad_barrier` |
| Random ALU-only programs | 12 per run |
| `mgpu.fuzz` diff, raw and eq programs | 16 per mode in CI. Local soaks: 200 diff + 198 raw + 200 eq programs (records), then 150 diff + 146 raw + 150 eq programs on new seeds (records and cycle count); no divergence |

Unit tests for the core's blocks:
- `test_vexec`: whole random instructions through the decoder and executor, every record field and
  the completion cycle, against the model, for five unit configurations.
- `test_fetch`: the timing model's fetch rules run in lockstep with `mgpu_fetch`.
- `test_core_decode`: `ctrl_t` fields for every instruction.

Each was checked to fail with a planted bug (EXEC ignored in the merge, a stale fill installed, CLOCK
off by one, an unsigned atomic MIN).

See [`docs/uarch/verification.md`](../uarch/verification.md) for the overall plan.
