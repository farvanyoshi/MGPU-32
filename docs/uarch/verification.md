# RTL verification plan

The RTL is correct when it satisfies [`equivalence.md`](../isa/equivalence.md) against the reference
simulators on every test program. Its timing should also match the timing contract
([`core.md`](core.md) §11) closely enough that the performance model stays trustworthy. Each layer
below reuses something that already exists.

## 1. Tools

- **Verilator.** Lint and simulation.
- **cocotb.** Python testbenches: they import `mgpu` directly, so every reference is the real golden
  model.
- **C++ Verilator harness.** For long lockstep runs, linking `libmgpu_core.a` through
  `include/mgpu_core.h`.
- **Yosys.** Synthesis statistics (cells, flops, memories).
- **CI.** Lint, unit tests and the lockstep regression run on every push.

## 2. Unit level (breakpoint 5)

Each execution unit gets a cocotb test that drives random and corner-case operands and compares every
result bit with the Rust engine:

| Unit | Reference | Stimulus |
|---|---|---|
| Integer ALU lane | `mgpu.model.execute.int_op` / V_CMP conditions | All functs × random operands × edge values (0, ±1, INT_MIN/MAX, shifts ≥ 32) |
| FPU (FADD/FSUB/FMUL/FFMA/FMIN/FMAX/I2F/F2I/compares) | `mgpu._core.fp_batch` | The special-value cross product of `mgpu.fp.testvals`, cancellation pairs, flush boundaries, ≥ 10⁶ random vectors per op in the C++ harness |
| SFU | `mgpu._core.fp_batch` (frcp/frsq/fex2/flg2) | Every mantissa for exponents −1, 0, 1 (exhaustive per binade), plus special cases |
| Decoder | `mgpu.isa.decode` | All 2³² words through the C++ harness. Legal/illegal classification and every field must match |
| Shared memory + conflict logic | `docs/uarch/core.md` §8.3 | Random lane addresses; conflict degree checked against the model's `conflict_degree` |
| Coalescer | The model's `segments()` | Random lane addresses and EXEC masks |

The SFU ROMs and the decoder tables are generated from `spec/` by the same generators as the
simulators (`tools/gen_isa.py`, `tools/gen_sfu_tables.py`). Their freshness is checked in CI.

## 3. Core level: lockstep co-simulation (breakpoint 6)

*Status: done. See [`docs/rtl/README.md`](../rtl/README.md) §5–6 for what runs and the volumes.*

The core runs a real program while the reference engine runs in **driven mode**:

1. The testbench loads the same memory image into the RTL's memory model and into `mgpu._core.Gpu`,
   and starts both launches. The engine side uses `begin()`.
2. Whenever the RTL **retires** an instruction of warp (b, w), the testbench calls
   `gpu.step(b, w)`. That executes the same instruction in the engine, and the testbench compares the
   commit records under the per-warp rule E4: PC, word, register writes, EXEC, memory writes, events
   and faults.
   - The RTL exposes a *retire port*: one record per retired instruction, carrying exactly the
     `mgpu_record` fields.
3. At the end, it compares STATUS and the ERR_* registers (E2/E3) and all of global memory (E1).
4. It also compares the launch's cycle count with `launch_timed` under the same parameters. On a
   single core the RTL matches the timing model exactly on every program run so far, so the cycle
   comparison of §4 is already exact at core level.

With one warp per block nothing can depend on the schedule, and the testbench compares atomic
values as well. Multi-warp programs from the fuzzer are restricted to eq mode (well-defined
programs), because for a racy program the engine executes each instruction at its retire cycle
while the RTL's memory sees it at request time.

Because the engine follows whatever warp order the RTL chooses, the comparison is exact even for
programs whose results depend on the schedule.

**Program sources:**
- the directed tests of `tests/sim`, already written in assembly
- the kernel suite
- `mgpu.fuzz` diff-mode and eq-mode programs (thousands, with faults and illegal words included)

## 4. Top level (breakpoint 7)

- **Host interface:** a CSR-port test of every register in ISA §13.1, including writes ignored while
  running, launch validation (STATUS 5 and 2, GRID_DIM = 0) and GO/irq timing.
- **Multi-core:** dispatch order, abort with several cores faulting, and atomics from several cores
  hitting one word.
- **Cycle comparison:** for every kernel and micro-benchmark, the RTL's `CYCLES` against
  `launch_timed` with the same parameters. The target is an exact match on the micro-benchmarks of
  `tests/test_timing.py`, and a documented, bounded difference on the kernels. Any difference is
  either fixed in the RTL or written into `core.md` §11 and the model.

## 5. Coverage goals

| Coverage | Goal |
|---|---|
| Opcodes and functs | Every instruction issued at least once, with EXEC = 0, partial, and full |
| Faults | Every error code, every ERR field, a fault on every pass of a vector access |
| Scheduler | Every warp slot issuing; each unit-busy situation; barrier release by BAR and by EXIT |
| LSU | Segment counts 1, 2 and 32; bank conflict degrees 1, 2 and 32; store collisions; atomic collisions |
| I-cache | Hit, miss, a miss while a fill is pending, conflict eviction |

Coverage is collected with Verilator's line and toggle coverage plus functional counters in the
cocotb testbenches.
