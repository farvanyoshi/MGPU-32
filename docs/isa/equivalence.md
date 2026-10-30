# Implementation equivalence (normative)

MGPU-32 is specified so that any two conforming implementations behave identically on every
**well-defined** program. Examples of implementations are the Python reference model, the Rust
simulator, the RTL, or someone else's emulator. This document defines "well-defined" and "identically"
precisely. It is the contract that the test suites and the RTL co-simulation check.

## 1. Sources of variation the ISA allows

Only these things may legitimately differ between conforming implementations:

1. **Implementation parameters**: `MEM_SIZE`, `MAX_BLOCK_DIM` and the value of `CLOCK` (§1.2, [SR-3]).
2. **Scheduling**: the order and concurrency in which blocks run, and how instructions of different
   warps interleave.
3. **The contents of uninitialized shared memory** ([C-21]).
4. **Which fault is reported** when more than one warp faults ([FLT-2]), and global memory contents
   after a fault ([FLT-1]).
5. **`CYCLES`** and any other performance-related value.

Everything else is fixed by the specification: every instruction's result bits, including all
floating-point and SFU results, and every fault condition.

## 2. Well-defined programs

A launch (memory image, ENTRY, GRID_DIM, BLOCK_DIM and ARG0–ARG7) is **well-defined** if, for *every*
schedule permitted by §2 and §12 of the ISA, all of the following hold:

- **W1 – No data races.** Consider two accesses to the same word in the same memory space, where at
  least one is a write, they come from different warps, and they are not both atomics. Then they must
  be ordered: by a BAR both warps take part in (same block), or by a kernel boundary. For accesses from
  different blocks, the program's result must not depend on their order. FENCE/flag protocols are
  allowed only if the final result is the same for every interleaving.
- **W2 – No uninitialized shared reads.** Every shared-memory word is written, in the current block,
  before it is read.
- **W3 – No CLOCK dependence.** No value that affects memory, control flow or a fault is derived from
  `CLOCK`.
- **W4 – No dependence on atomic order across warps.** Return values of atomics that race with other
  warps must not affect memory, control flow or faults. (Final memory values of racing `ADD`, `MIN`,
  `MAX`, `AND` and `OR` atomics are order-independent and therefore allowed; racing `XCHG` final values
  are not.)
- **W5 – No self-modifying code** ([MEM-2]).
- **W6 – Fits the implementation.** `BLOCK_DIM ≤ MAX_BLOCK_DIM`, and every address the program
  touches is below `MEM_SIZE`, for both implementations being compared.

## 3. What must be identical

For a well-defined launch, two conforming implementations must agree on:

- **E1 – Final global memory.** Every byte of global memory below the smaller of the two `MEM_SIZE`s,
  after GO returns to 0.
- **E2 – Completion status.** STATUS.
- **E3 – Fault details.** If STATUS ≠ 0 and, in every schedule, exactly one warp raises a fault (for
  example, a single-warp kernel), then ERR_PC, ERR_BLOCK, ERR_WARP, ERR_LANE and ERR_INFO also agree.
  If several warps may fault, only "STATUS ≠ 0" is guaranteed (E2), and E1 does not apply.
- **E4 – Per-warp retired-instruction stream.** For each (block, warp), the sequence of retired
  instructions is the same: PC, instruction word, the registers written with their values (all 32
  lanes for vector writes, with inactive lanes unchanged), EXEC after the instruction, and the memory
  words written. The order *between* warps may differ; the order *within* a warp may not.

E4 is stronger than E1 and is what the RTL co-simulation checks, in lockstep against the reference
simulator. Because it is per-warp, it does not depend on scheduling.

## 4. Implications for implementers *(informative)*

- A cycle-accurate implementation may reorder, overlap or speculate as much as it likes, provided
  each warp's architecturally visible stream matches E4 and the memory model of §12 is respected.
- A functional simulator may run blocks one at a time in increasing BLOCK_ID order and interleave
  warps round-robin, as the reference model does. For a well-defined program, it yields the same E1–E4
  as any other legal schedule.
- To test a program against W1–W4, run it on the reference model under several schedules: different
  warp-interleaving seeds and block orders, and different shared-memory poison values. The program is
  well-defined only if the results are identical every time.
