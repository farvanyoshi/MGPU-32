# MGPU-32 simulators

Two independent implementations of the ISA live in this repository:

| Backend | Code | Purpose | Speed |
|---|---|---|---|
| `model` | `python/mgpu/model/` | Executable specification. Every instruction reads like the ISA pseudocode and cites its rule IDs. Slow and obviously correct. | ~30 K warp-instr/s |
| `rust` | `rust/mgpu-core/` | Fast engine for real workloads, the toolchain and RTL co-simulation. Exposed to Python as `mgpu._core` and to C as `include/mgpu_core.h`. | ~5 M warp-instr/s (~150 M lane-ops/s) |

Both follow the same **reference schedule** (§3) and emit the same **commit records** (§4). For
*any* program, including racy or faulting ones, they must therefore produce identical traces, memory
and host registers. The differential fuzzer checks exactly that (§6). Speeds were measured with
`tools/bench.py` (150–190× apart).

## 1. Quick start

```bash
pip install -e .            # builds the Rust extension with maturin (needs a Rust toolchain)
python -m pytest            # whole test suite
```

```python
from mgpu.device import Device
from mgpu.isa import assemble_one

dev = Device("rust")                     # or Device("model")
prog = [assemble_one(s) for s in ("V_GETSR v1, GTID", "V_SHLI v2, v1, 2",
                                   "V_ADD v2, v2, s1", "V_SW v1, 0(v2)", "EXIT")]
dev.write_words(0x1000, prog)
res = dev.launch(entry=0x1000, grid=4, block=64, args=[0x8000])
print(res)                               # ok (40 instructions)
print(dev.read_words(0x8000, 8))         # [0, 1, 2, 3, 4, 5, 6, 7]
```

The full assembler (`mgpu-as`) arrives with the toolchain at breakpoint 3.

## 2. Python API

`Device(backend="rust" | "model", **params)` accepts these parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `mem_size` | 64 MiB | Global memory size `MEM_SIZE` (multiple of 4, ≤ 2³²) |
| `max_block_dim` | 1024 | `MAX_BLOCK_DIM` (multiple of 32, 32–1024) |
| `resident_blocks` | 1 | Blocks the reference schedule keeps resident at once |
| `seed` | `None` | `None`: round-robin warp interleaving; an integer: seeded random interleaving |
| `shared_poison` | `0xDEADBEEF` | Initial value of every shared-memory word |
| `max_steps` | 50,000,000 | Watchdog; exceeding it raises `mgpu.errors.SimTimeout` |
| `trace` | `False` | Record a `CommitRecord` per instruction (`dev.trace`) |
| `keep_state` | `False` | Keep finished blocks so `dev.warp_state(block, warp)` works after the run |

Methods:

| Method | What it does |
|---|---|
| `write(addr, bytes \| ndarray)`, `write_words`, `read`, `read_words`, `read_array` | Host access to global memory |
| `reg_read(off)`, `reg_write(off, value)` | Host registers (ISA §13.1; offsets in `mgpu.hostregs`). Writing 1 to GO runs the kernel to completion |
| `launch(entry, grid, block, args)` | Writes the launch registers, sets GO, and returns a `LaunchResult` (`status`, `err_*`, `steps`, `.ok`) |
| `trace` | Commit records of the last launch |
| `warp_state(b, w)` | `(pc, s[32], v[32][32], state)` |

`dev.gpu` exposes the backend object, which adds **driven mode**:

| Call | What it does |
|---|---|
| `begin()` | Validates the launch registers and starts the launch without running it |
| `runnable()` | Lists the `(block, warp)` pairs that may retire next |
| `step(block, warp)` | Retires one instruction of that warp and returns its record |

An external scheduler (the RTL in phase 5) decides the order. Blocks become resident on first use.

## 3. Reference schedule

This schedule is not part of the ISA; any legal schedule is allowed by the spec. It is fixed here
so that both backends make *identical* choices. A new backend that wants trace-level agreement must
implement it exactly.

1. **Launch** follows ISA [HOST-3]/[HOST-4]. `CLOCK` and `CYCLES` count *steps*. A step is one
   retired or faulting instruction, numbered from 0. `CLOCK` reads the number of the step that
   executes it.
2. **Residency.** Blocks become resident in increasing `BLOCK_ID`, while fewer than
   `resident_blocks` are resident. A block stops being resident once all its warps have exited.
   Residency is refilled before the first step and after every step.
3. **Eligibility.** A warp is eligible if its block is resident and the warp is running: not exited
   and not waiting at a barrier. The eligible list is sorted by `(block, warp)`.
4. **Choice.** With no seed, the next warp is the first eligible key strictly greater than the
   previously chosen key, wrapping to the smallest. With a seed, it is `eligible[prng.below(len)]`.
5. **PRNG (xorshift64\*).**
   - State starts at `(seed XOR 0x9E3779B97F4A7C15) mod 2⁶⁴`, or `0x9E3779B97F4A7C15` if that is 0.
   - `next`: `x ^= x >> 12; x ^= x << 25; x ^= x >> 27` (all mod 2⁶⁴), store `x`, and return
     `x * 0x2545F4914F6CDD1D mod 2⁶⁴`.
   - `below(n) = (next() >> 32) mod n`.
6. **Faults.** The faulting step produces a fault record, sets STATUS and stops the kernel at once.
7. **Watchdog.** Before each step, if `steps ≥ max_steps`, the simulator raises `SimTimeout`. GO
   stays 1.

## 4. Trace format

One commit record per step (`mgpu/trace.py`, `rust/mgpu-core/src/trace.rs`, and `mgpu_record` in C):

| Field | Meaning |
|---|---|
| `step`, `block`, `warp`, `pc`, `word` | Identity of the instruction (`word` = 0 for a fetch fault) |
| `swrite` | `(reg, value)` scalar write, or none. Writes to s0 are not recorded |
| `vwrite` | `(reg, (32 lane values after the write))`, or none. Recorded for every vector instruction with rd ≠ 0, even if EXEC = 0 |
| `exec_after` | s31 after the instruction |
| `mwrites` | `[(space, addr, value)]` in program order. Space 0 is global, 1 is shared. Includes every lane of a store, colliding lanes included, and every lane of an atomic (the new value) |
| `event` | 0 none, 1 EXIT, 2 BAR (arrival), 3 fault |
| `released` | This BAR or EXIT completed a barrier |
| `fault` | `(code, lane, info)` or none. ERR_PC is the record's `pc` |

`CommitRecord.key()` is the canonical tuple used for global comparison. `warp_key()` drops the
schedule-dependent parts (step, release flag, and atomic values, per C-37) for the per-warp check
of equivalence rule E4.

## 5. Float arithmetic implementation notes

Neither backend uses host floating point. The Python ops (`mgpu/fp/ops.py`) compute exact results
with big integers and round once with `round_fixed` ([FP-3]).

The Rust engine (`fp.rs`) uses u128. To add two values whose exponents differ by more than 63 bits,
it **jams** the smaller one into a sticky bit:
- After normalization, the larger operand's lowest set bit is 63 places above the unit.
- Rounding boundaries and midpoints are therefore even multiples of that unit.
- The jammed value lies strictly inside the same open interval between such points as the exact
  value does, so the rounded result cannot change.

`tests/fp` checks both implementations against an independent `Fraction` oracle and against each
other.

## 6. Testing

| Suite | What it checks |
|---|---|
| `tests/test_isa_spec.py`, `tests/test_sfu.py` | Generated tables are current, conformance encodings, decode policy, SFU special cases and accuracy |
| `tests/fp/` | Float ops against the Fraction oracle and numpy; Rust vs Python on all special-value combinations plus ~30 k random operands per op |
| `tests/sim/` | Directed tests of every instruction group, faults, barriers, launch validation and the spec's kernels. **Every test runs on both backends** |
| `tests/fuzz/` | Random programs: model vs Rust (structured, raw words, well-defined) and schedule independence (E1/E2/E4) |
| `tests/test_capi.py` | Compiles `tests/c/smoke.c` with gcc against `libmgpu_core.a` |

### Fuzzing

```bash
python -m mgpu.fuzz --seeds 2500                  # 10,000 cases in 4 modes, uses all cores
python -m mgpu.fuzz --mode diff --start 42 --seeds 1 --show   # replay one case with its listing
MGPU_FUZZ_SEEDS=500 python -m pytest tests/fuzz   # a bigger batch inside pytest
```

| Mode | What is compared |
|---|---|
| `diff` | Random structured programs: all 101 instructions, nested loops, divergent EXEC regions, BAR, faults, illegal words, races. Model vs Rust trace, memory, registers and timeout |
| `raw` | Mostly-random instruction words. Exercises the decoder and fault paths |
| `eq-diff` | Well-defined programs, model vs Rust |
| `eq` | Well-defined programs under 6 schedules (resident blocks × seeds × shared poison). Checks final memory, STATUS and per-warp streams |

The generators are deterministic per seed, so a failing seed is a complete reproducer.

## 7. C interface

`include/mgpu_core.h` declares the C API. Link with `rust/target/release/libmgpu_core.a`
(`cargo build --release -p mgpu-core`) and `-lpthread -ldl -lm`.
- Lifecycle: `mgpu_create`, `mgpu_destroy`.
- Host side: `mgpu_mem_write`/`mgpu_mem_read` and `mgpu_reg_write`/`mgpu_reg_read`. Writing GO runs
  the kernel.
- Driven mode: `mgpu_begin`, `mgpu_runnable`, and `mgpu_step`, which fills an `mgpu_record`.

`tests/c/smoke.c` is a complete example.

## 8. Code map

```
python/mgpu/fp/        f32.py (round_fixed), ops.py (arithmetic), sfu.py, testvals.py
python/mgpu/model/     state.py (warps, blocks, BAR), execute.py (semantics), gpu.py (host, schedule)
python/mgpu/sched.py   reference schedule + PRNG        python/mgpu/trace.py  commit records
python/mgpu/device.py  front end                         python/mgpu/fuzz/     generators, runners, CLI
rust/mgpu-core/src/    fp.rs sfu.rs decode.rs gpu.rs sched.rs trace.rs capi.rs (+ generated isa_gen.rs, sfu_tables.rs)
rust/mgpu-py/src/      PyO3 bindings
```
