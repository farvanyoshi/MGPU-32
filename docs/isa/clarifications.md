# Clarifications to the original MGPU-32 design note

This log records every place where the original note (*"MGPU-32: A Minimal RISC GPU ISA"*, Oct 6,
2026, 27 pages) was ambiguous, under-specified or self-contradictory, together with the rule adopted
in [`MGPU-32-ISA.md`](MGPU-32-ISA.md) v1.0. Section numbers like "§7.6" refer to the **original
note**. Rule IDs like [DEC-1] refer to the v1.0 spec.

**Status column:**
- **Decided**: chosen together with the project owners during planning.
- **Proposed**: chosen while writing v1.0 and open to veto at breakpoint 1.
- **Confirmed**: the original was already precise and was checked, for example by re-deriving the
  worked encodings.

## Decisions made during planning

| ID | Original | Issue | Resolution | Status |
|---|---|---|---|---|
| C-01 | §6.3 | SFU ops were "approximations accurate to at least 22 bits", so two correct implementations could return different bits. | The bit-exact table-plus-quadratic algorithm in [`sfu.md`](sfu.md), with normative tables in `spec/sfu/`. Measured worst case: 0.767 ULP (23.80 bits). | Decided |
| C-02 | §7.10 | BAR waits for every *non-exited* warp, which means a block can never deadlock, yet the note under it says skipping a BAR deadlocks and asks for detection ("all live warps blocked", error 4). | [BAR-1..4]: release when every non-exited warp waits, but all waiting warps must be at the same BAR PC; otherwise error 4 "barrier mismatch". Exited warps are excused. | Decided |

## Encoding and decoding

| ID | Original | Issue | Resolution | Status |
|---|---|---|---|---|
| C-03 | §6 | Only reserved *opcodes* were said to be illegal; unused funct values and unused fields were unspecified. | [DEC-1]: opcode and the full funct field must name a listed instruction. [DEC-2]: every other unused field is ignored on decode and written as 0 by assemblers. Ignoring rather than rejecting keeps decoders small and is equally deterministic. | Proposed |
| C-04 | §6.2 | POPC/FF1 have Vector = "no", but the consequence was not stated. | V_ALU funct 0x11/0x12 is an illegal instruction. | Proposed |
| C-05 | §7.9 vs §5, §6.1 | §7.9 writes `S_BZ sa, label`, suggesting the ra field, but the opcode map lists "s (tested)" under rd, and the worked encoding `S_BNZ s12, -5 = 0x8580FFFB` puts 12 in rd. | [ENC-4]: the tested register is in **rd**, and ra is ignored. | Confirmed |
| C-06 | §5 | The sb bit sits at bit 10 in R format but bit 5 in R4; the note's table showed this but never said it. | Stated explicitly in §4. sb is meaningful only for `vb\|sb` operands, and ignored elsewhere, including shift-immediates. | Confirmed |
| C-07 | §7.6 | READLANE's rb is a scalar, but it was unclear whether sb must be set. SHFL's rb was written `vb` only. | READLANE always reads rb as a scalar and ignores sb. SHFL accepts `vb\|sb`; a broadcast lane index gives a lane broadcast. | Proposed |
| C-08 | §6.3 | "SFU ops ignore rb", but I2F and F2I are also unary. | All six unary V_FPU ops (FRCP, FRSQ, FEX2, FLG2, I2F, F2I) ignore rb and sb. | Proposed |
| C-09 | §4.2, §7.5 | Reading a *reserved* SR ID was unspecified. | [SR-1]: illegal instruction (error 1), for both S_GETSR and V_GETSR. [DEC-4]: raised whatever EXEC is. | Proposed |
| C-10 | §6.1 | "24 of 64 opcodes are used." | Re-counted: 14 + 7 + 3 = 24. All six worked encodings re-derived bit-for-bit and kept as conformance vectors in `spec/isa.toml`. | Confirmed |

## Determinism within one instruction

| ID | Original | Issue | Resolution | Status |
|---|---|---|---|---|
| C-11 | §7.7, §9 | Colliding V_SW/V_STS lanes: "one value wins; which one is unspecified". | [MEM-4]: the highest-numbered active lane wins. This is still a valid reading of the original rule, but deterministic. | Proposed |
| C-12 | §7.8 | Atomic lanes are applied "in unspecified order", so the old values returned within one instruction were nondeterministic. | Ascending lane order (§6.9). Results across warps still depend on timing (see equivalence.md). | Proposed |
| C-13 | — | Overlapping source and destination registers (for example `V_SHFL v1, v1, v2`, or V_ATOM with vd = va) were not addressed. | [SEM-1]: every source in every lane is read before any destination is written. | Proposed |
| C-14 | §7.1 | EXEC timing was stated only for scalar writes to s31. | [EX-3]: an instruction always uses EXEC as it was before the instruction. Any write to s31 (V_CMP, V_READLANE, S_GETSR, S_LW, …) takes effect from the next instruction. | Proposed |

## Memory and faults

| ID | Original | Issue | Resolution | Status |
|---|---|---|---|---|
| C-15 | §7.7 | Misaligned or out-of-range addresses were "undefined behavior (an implementation may halt the warp)". | They **must** fault: error 2 (misaligned) or error 3 (out of range). Each lane is checked misalignment first, then range, in ascending lane order. Faults are precise: the instruction has no effect ([MEM-3]). | Proposed |
| C-16 | §4.3 | "Out of range" had no definition, because global memory size was not fixed. | Implementation parameter `MEM_SIZE`, readable by the host; shared memory size is 16384. | Proposed |
| C-17 | §10.1 | "STATUS: error code plus the faulting block, warp, and PC" did not say how three values fit in one register, and gave no register offsets. | Full host register map (§13.1). STATUS holds the code, and ERR_PC, ERR_BLOCK, ERR_WARP carry the rest. Added ERR_LANE and ERR_INFO (instruction word, faulting address, or barrier PC) for debugging, plus ID, VERSION, MAX_BLOCK_DIM, MEM_SIZE and CYCLES. | Proposed |
| C-18 | §10.1 | Effect of a fault on the rest of the kernel was unstated. | [FLT-1]: the kernel aborts. Global memory after an abort is undefined. If several warps fault concurrently, one fault is reported ([FLT-2]). | Proposed |
| C-19 | §7.9 | Running off the end of memory, or branching outside it, was unspecified. | [MEM-7]: fetching a PC > `MEM_SIZE − 4` raises error 3. Branch arithmetic wraps mod 2³² ([ENC-5]). | Proposed |
| C-20 | — | Writing to code while it runs was unaddressed. | [MEM-2]: undefined behaviour. | Proposed |
| C-21 | §4.3, §10.2 | Shared memory is "undefined" at block start. | Unchanged. Reading it before writing it is outside the equivalence guarantee. The reference model fills it with a configurable poison pattern to expose such bugs. | Confirmed |

## Launch and host interface

| ID | Original | Issue | Resolution | Status |
|---|---|---|---|---|
| C-22 | §3 vs §12 | Blocks may have up to 1024 threads (32 warps) and must sit on one core, but the reference core has only 8 warp slots, so it could not run a 1024-thread block. | `MAX_BLOCK_DIM` is an implementation parameter (32–1024, a multiple of 32), readable by the host. A larger `BLOCK_DIM` fails the launch with new **error 5** (invalid launch). The reference model uses 1024. | Proposed |
| C-23 | §10.1 | Launch edge cases were unspecified: `GRID_DIM = 0`, `BLOCK_DIM = 0`, misaligned `ENTRY`, writing GO or the parameter registers while a kernel runs. | [HOST-1..5]: validation order is BLOCK_DIM → ENTRY → GRID_DIM. Writes while running are ignored, and GO cannot cancel a kernel. | Proposed |
| C-24 | §4.2 | `CLOCK` makes results implementation-dependent. | [SR-3]: implementation-defined but monotonic per warp, and excluded from the equivalence guarantee. | Proposed |
| C-25 | §3, §7.10 | Concurrent residency of a block's warps was implied by BAR but never stated. | [EX-1]: all warps of a block are resident together. | Proposed |

## Floating point

| ID | Original | Issue | Resolution | Status |
|---|---|---|---|---|
| C-26 | §7.3 | Denormal flushing was stated for FPU arithmetic only; compares, FMIN/FMAX and F2I were unclear. | [FP-1]: every float instruction flushes its inputs. Integer XOR/ANDN, used for FNEG/FABS, never flush. | Proposed |
| C-27 | §7.3 | Output flushing did not say whether tininess is detected before or after rounding. | [FP-3]: after rounding, with an unbounded exponent. The flushed zero keeps the sign of the exact result. | Proposed |
| C-28 | §7.3 | Invalid operations give the canonical NaN, but whether a NaN *input* propagates its payload was unstated. | [FP-2]: every NaN result is exactly `0x7FC00000`. | Proposed |
| C-29 | §6.3 | FMIN/FMAX order of −0 and +0, and the both-NaN case, were unspecified. | [FP-7]: −0 < +0; NaN, NaN → canonical NaN. | Proposed |
| C-30 | §7.4 | FFMA: the sign of an exact-zero result and the infinity/invalid cases were unspecified. | [FP-6], following IEEE-754 roundTiesToEven conventions. | Proposed |
| C-31 | §6.3 | F2I of ±∞ was not listed. | [FP-10]: saturates to `0x7FFFFFFF` / `0x80000000`. | Proposed |

## Pseudo-instructions and assembly

| ID | Original | Issue | Resolution | Status |
|---|---|---|---|---|
| C-32 | §7.11 | `V_NOT vd, va` expanded to `V_XOR vd, va, sX` "with sX = 0xFFFFFFFF", but did not say which register sX is or who loads it. | Expand to `V_SUB vd, v0, va ; V_ADDI vd, vd, -1` (`~x = −x − 1`). This needs no scratch register and is correct even when vd = va. | Proposed |
| C-33 | §7.11 | `V_FNEG` had the same problem. | The scratch register is now an explicit operand: `V_FNEG vd, va, st`, which overwrites `st`. Added the matching `V_FABS vd, va, st`. | Proposed |
| C-34 | §7.11 | `V_FDIV vd, va, vb` used an unnamed temporary `vt`. | Explicit operand: `V_FDIV vd, va, vb, vt`. | Proposed |
| C-35 | §7.11 | `S_LI` could be read as "shortest form". | Always exactly two instructions, `S_LUI` then `S_ORI`, so code layout never depends on a value. | Proposed |
| C-36 | §10.4 | Case sensitivity, the `exec` alias, SR names as operands and numeric branch offsets were not addressed. | §14 of v1.0. The full grammar is defined at breakpoint 3 (toolchain). | Proposed |

## Not part of the ISA

- The original's §2 (VLIW/SIMD/SIMT discussion) and the closing "questions worth deciding" are design
  rationale; they are summarized in §1 of v1.0.
- The original's §12 (reference microarchitecture) is not normative and becomes `docs/uarch/` at
  breakpoint 4. Its "detect deadlock (all live warps blocked)" requirement is subsumed by C-02: under
  [BAR-3] a barrier alone cannot deadlock.
