# MGPU-32 Instruction Set Architecture — Version 1.0

**Status:** normative. This document supersedes the original design note *"MGPU-32: A Minimal RISC
GPU ISA"* (Oct 6, 2026). It keeps that note's architecture and adds every rule needed for two
independent implementations to produce identical results. Each place where the original was ambiguous
or self-contradictory is logged, with its resolution, in [`clarifications.md`](clarifications.md).

Companion documents:

| Document | Contents |
|---|---|
| [`encoding-tables.md`](encoding-tables.md) | Generated opcode map, funct tables and bit patterns |
| [`sfu.md`](sfu.md) | Bit-exact algorithm for FRCP, FRSQ, FEX2 and FLG2 |
| [`equivalence.md`](equivalence.md) | Exactly what two conforming implementations must agree on |
| [`../../spec/isa.toml`](../../spec/isa.toml) | Machine-readable instruction table (normative) |
| [`../../spec/sfu/`](../../spec/sfu) | SFU coefficient tables (normative) |

**Conventions.** "Must" and "must not" are requirements on implementations or programs, as the
sentence says. Rules carry identifiers such as **[DEC-1]** so tests and other documents can cite
them. Sections marked *(informative)* explain things but add no requirements. Hex numbers are written
`0x…`, and bit ranges `[hi:lo]` are inclusive.

---

## 1. Overview

MGPU-32 is a 32-bit, load/store, fixed-width RISC instruction set for a small GPU. One instruction
is fetched per **warp** (32 threads) and executed on all 32 **lanes** at once, under a 32-bit
active-lane mask called **EXEC**. Scalar and vector instructions are explicit, in the style of
AMD GCN/RDNA. Divergence is handled by software writing EXEC; there is no hardware reconvergence stack.

> A GPU core is a scalar RISC CPU whose register file has a second half that is 32 lanes wide, plus
> a bitmask that decides which lanes are allowed to write.

Design goals, in priority order: (1) simple to reason about, (2) simple hardware, (3) real enough to
run real kernels (SAXPY, reductions, matrix multiply, image filters), (4) room to grow (opcodes
0x30–0x3F are reserved for extensions).

### 1.1 Fixed parameters

| Parameter | Value |
|---|---|
| Word size | 32 bits |
| Instruction size | 32 bits, 4-byte aligned, little-endian |
| Warp width | 32 lanes |
| Vector registers | 32 per lane (v0–v31); v0 reads as zero |
| Scalar registers | 32 per warp (s0–s31); s0 reads as zero; s31 is EXEC |
| Address spaces | 32-bit byte-addressed global memory; 16 KiB shared memory per block |
| Number formats | 32-bit two's-complement integers; IEEE-754 binary32 floats (with the rules of §8) |

### 1.2 Implementation parameters

These values may differ between conforming implementations. Each one is readable by the host (§13).

| Parameter | Meaning | Constraint |
|---|---|---|
| `MEM_SIZE` | Bytes of global memory; valid addresses are `0 … MEM_SIZE-1` | Multiple of 4, 4 ≤ `MEM_SIZE` ≤ 2³² |
| `MAX_BLOCK_DIM` | Largest supported `BLOCK_DIM` | Multiple of 32, 32 ≤ `MAX_BLOCK_DIM` ≤ 1024 |
| `CLOCK` | Value of the CLOCK special register | See [SR-3] |

---

## 2. Execution model

A program (a **kernel**) is launched over a **grid** of **blocks**. Each block is split into **warps**
of 32 **lanes**. Each warp has exactly one program counter, and EXEC decides which of its lanes take
part in each vector instruction.

- **Grid:** `GRID_DIM` blocks, numbered `0 … GRID_DIM-1`. Blocks are independent. They may run in any
  order, on any core, concurrently or not, and are never guaranteed to be resident at the same time.
- **Block:** `BLOCK_DIM` threads (1 … `MAX_BLOCK_DIM`). **[EX-1]** All warps of a block are resident
  at the same time, on the same core, share one 16 KiB shared memory, and can synchronize with `BAR`.
- **Warp:** 32 consecutive threads of a block. Warp *w* holds threads 32*w* … 32*w*+31. The warp is
  the unit of fetch, decode and scheduling. A block has `NUM_WARPS = ceil(BLOCK_DIM / 32)` warps.
- **Lane:** one thread in a warp, with lane IDs 0–31. A lane owns one copy of every vector register.

The ISA is one-dimensional (one block ID, one thread ID). Code that works on 2-D images computes x
and y from the 1-D index.

**[EX-2] One PC per warp.** All lanes of a warp are always at the same instruction. There is no
per-lane PC and no hardware reconvergence.

**[EX-3] EXEC.** EXEC is scalar register s31. Bit *i* = 1 means lane *i* is *active*.
- A **vector** instruction (exec = `mask` in `isa.toml`) affects only active lanes. Inactive lanes
  keep their old register values, perform no memory access, and cannot fault.
- A **scalar** instruction (exec = `ignore`) ignores EXEC and always executes once per warp. That
  includes branches, so even with EXEC = 0 a warp can still run the scalar code that restores EXEC.
- The EXEC value an instruction uses is the value of s31 *before* that instruction executes. Writing
  s31 (with any scalar instruction, or `V_CMP`, `V_READLANE` or `S_GETSR`) changes EXEC from the
  next instruction onward.

**[EX-4] Warp lifetime.** A warp starts at the kernel entry PC with the state given in §13.3. It runs
until it executes `EXIT`, which frees its resources. A block finishes when all of its warps have
exited, and a kernel finishes when all of its blocks have finished, or when it is aborted by a fault
(§11).

**[EX-5] Sequential semantics within a warp.** Each warp behaves as though it executes its
instructions one at a time, in program order. Each instruction reads all of its source operands
(registers, EXEC, and the memory it loads) before it writes any destination (§6.1).

---

## 3. Architectural state

Each warp sees a PC, 32 scalar registers (one copy per warp), 32 vector registers (one copy per
lane), a set of read-only special registers, and two memory spaces.

### 3.1 Registers

| Name | Count | Width | Copies | Notes |
|---|---|---|---|---|
| PC | 1 | 32 | per warp | Byte address of the current instruction; always a multiple of 4 |
| s0 | 1 | 32 | per warp | Always reads 0; writes are discarded |
| s1–s30 | 30 | 32 | per warp | General purpose; s1–s8 hold kernel arguments at launch |
| s31 (EXEC) | 1 | 32 | per warp | Active-lane mask; read and written like any scalar register |
| v0 | 1 | 32 | per lane | Always reads 0; writes are discarded |
| v1–v31 | 31 | 32 | per lane | General purpose |

Registers are untyped bit containers. Integer instructions treat them as two's complement and float
instructions as binary32. Moving a value between the two interpretations is free.

In assembly, `exec` is an alias for `s31`.

### 3.2 Special registers (read-only)

Special registers are read with `S_GETSR` (into a scalar register) or `V_GETSR` (into a vector
register). The SR ID goes in the `ra` field.

| ID | Name | Kind | Value |
|---|---|---|---|
| 0 | `BLOCK_ID` | uniform | Index of this block in the grid |
| 1 | `BLOCK_DIM` | uniform | Threads per block |
| 2 | `GRID_DIM` | uniform | Blocks in the grid |
| 3 | `WARP_ID` | uniform | Index of this warp within its block |
| 4 | `NUM_WARPS` | uniform | Warps per block, `ceil(BLOCK_DIM / 32)` |
| 5 | `CLOCK` | uniform | Implementation-defined cycle counter, see [SR-3] |
| 16 | `TID` | per lane | `32 × WARP_ID + LANE_ID` |
| 17 | `LANE_ID` | per lane | `0 … 31` |
| 18 | `GTID` | per lane | `u32(BLOCK_ID × BLOCK_DIM + TID)` |

- **[SR-1]** IDs 6–15 and 19–31 are reserved. Reading a reserved ID with either instruction is an
  illegal instruction (error 1).
- **[SR-2]** `S_GETSR` on a per-lane SR (ID ≥ 16) is an illegal instruction. `V_GETSR` on a uniform
  SR writes the same value to every active lane.
- **[SR-3]** `CLOCK` returns the low 32 bits of a counter whose rate and starting value are
  implementation-defined. Two reads by the same warp, in program order, return non-decreasing values,
  modulo wrap-around. Programs whose results depend on `CLOCK` are outside the equivalence guarantee.
- `TID`, `LANE_ID` and `GTID` are defined for every lane, including lanes ≥ `BLOCK_DIM` in the last
  warp, although those lanes start inactive.

### 3.3 Memory spaces

| Space | Size | Scope | Accessed by | Notes |
|---|---|---|---|---|
| Global | `MEM_SIZE` bytes | whole GPU and host | S_LW, S_SW, V_LW, V_SW, V_ATOM, instruction fetch | Byte-addressed, little-endian. Holds kernel code, inputs and outputs |
| Shared | 16384 bytes | one block | V_LDS, V_STS, V_ATOM (`.S`) | Separate address space starting at 0. Contents undefined at block start and lost at block end |
| Registers | §3.1 | one warp | all ALU instructions | Not addressable |

**[MEM-1]** Every data access is a 32-bit word at a 4-byte-aligned address. A word at address *a*
consists of bytes *a* (least significant) through *a*+3. Byte and halfword access is a listed extension.

**[MEM-2]** Instructions are fetched from global memory. A program must not write to the bytes of
any instruction that a running kernel may execute; doing so is undefined behaviour.

---

## 4. Instruction formats and encoding

Every instruction is 32 bits, with a 6-bit opcode in [31:26]. There are four formats:

| Format | [31:26] | [25:21] | [20:16] | [15:11] | [10:6] | [5] | [4:0] |
|---|---|---|---|---|---|---|---|
| **R** | opcode | rd | ra | rb | sb [10], funct[9:6] | funct[5] | funct[4:0] |
| **R4** | opcode | rd | ra | rb | rc | sb | funct |
| **I** | opcode | rd | ra | imm16 [15:0] | | | |
| **J** | opcode | offset26 [25:0] | | | | | |

- **R** (register-register): `opcode[6] rd[5] ra[5] rb[5] sb[1] funct[10]`. Used by most ALU work.
- **R4** (three sources): `opcode[6] rd[5] ra[5] rb[5] rc[5] sb[1] funct[5]`. Used only by V_FFMA,
  V_IMAD and V_SEL. **Note:** sb is bit 10 in R format but bit 5 in R4.
- **I** (immediate): `opcode[6] rd[5] ra[5] imm16[16]`. Used by immediate ALU ops, loads, stores and
  conditional branches.
- **J** (jump): `opcode[6] offset26[26]`. Used only by S_J.

**Register fields.** Each register field is 5 bits (0–31). Whether a field names a scalar or a vector
register is fixed by the instruction (the operand tokens in `isa.toml`). The assembler checks this;
the hardware does not need to.

**[ENC-1] The sb (scalar-broadcast) bit.** In an instruction whose rb operand is `vb|sb`, sb = 1 means
rb names a *scalar* register whose value is used by all 32 lanes, and sb = 0 means rb names a vector
register. The assembler sets sb automatically when an s-register is written in the rb position, for
example `V_ADD v3, v2, s3`. In every other instruction, sb is ignored ([DEC-2]).

**[ENC-2] Immediates.** imm16 is sign-extended (`sext16`) for S_ADDI, V_ADDI, every load and store,
and the conditional branches. It is zero-extended (`zext16`) for S_ANDI, S_ORI and V_ANDI. S_LUI
places imm16 in bits [31:16]. Together, `S_LUI` and `S_ORI` build any 32-bit constant with no
sign-extension surprises. Shift-immediate instructions (`*_SHLI`, `*_SHRI`, `*_SRAI`) take their
shift amount, 0–31, from the 5-bit rb field.

**[ENC-3] Store data.** Stores have no destination, so the rd field holds the register whose value is
stored, as in `V_SW vt, imm(va)`.

**[ENC-4] Conditional-branch register.** S_BZ and S_BNZ test the scalar register in the **rd** field.
The ra field is ignored.

**[ENC-5] Branch targets.** Offsets count instructions, not bytes, and are relative to the next
instruction. Arithmetic wraps modulo 2³²:

```
target = u32(PC + 4 + (sext(offset) << 2))
```

This gives ±32 K instructions for conditional branches (imm16) and ±32 M instructions for S_J
(offset26).

### 4.1 Decode policy

- **[DEC-1]** The opcode and the *entire* funct field (10 bits in R format, 5 bits in R4) select the
  instruction. A word is an **illegal instruction** (error 1) if its opcode is not assigned, or if it
  is an R/R4 word whose funct value is not listed for that opcode in `isa.toml`. This includes:
  - vector POPC/FF1 (V_ALU funct 0x11/0x12)
  - V_CMP funct ≥ 0x10
  - V_ATOM funct values other than 0x00–0x05 and 0x10–0x15
- **[DEC-2]** Any field an instruction does not use is ignored when decoding: rd, ra, rb, rc, sb or
  any immediate bits that its syntax does not reference. An assembler must write such fields as 0.
  Examples: sb in scalar instructions and shift-immediates, rb/sb in unary FPU/SFU ops, ra in S_LUI,
  ra in S_BZ/S_BNZ, and all register fields of NOP/EXIT/BAR/FENCE.
- **[DEC-3]** For GETSR, the SR ID in ra is also checked: [SR-1] and [SR-2] make some IDs illegal.
- **[DEC-4]** An illegal instruction raises error 1 when the warp reaches it, whatever the EXEC value.
  It has no other effect.

### 4.2 Worked encodings (conformance vectors)

| Assembly | opcode | Fields | Hex |
|---|---|---|---|
| `V_ADD v3, v2, s3` | 0x02 | rd=3 ra=2 rb=3 sb=1 funct=0x000 | `0x08621C00` |
| `V_LW v5, 0(v3)` | 0x12 | rd=5 ra=3 imm=0 | `0x48A30000` |
| `V_FFMA v6, v5, s2, v6` | 0x05 | rd=6 ra=5 rb=2 rc=6 sb=1 funct=0 | `0x14C511A0` |
| `S_ADDI s12, s12, -1` | 0x08 | rd=12 ra=12 imm=0xFFFF | `0x218CFFFF` |
| `S_BNZ s12, -5` | 0x21 | rd=12 ra=0 imm=0xFFFB | `0x8580FFFB` |
| `V_CMP.LT s10, v1, s1` | 0x04 | rd=10 ra=1 rb=1 sb=1 funct=0x002 | `0x11410C02` |

Bit-level walk-through of the first one: `000010 00011 00010 00011 1 0000000000` =
`0000 1000 0110 0010 0001 1100 0000 0000` = `0x08621C00`.

---

## 5. Opcode map

24 of the 64 opcodes are used. The rest decode as illegal instructions, and 0x30–0x3F are reserved
for extensions. [`encoding-tables.md`](encoding-tables.md) has full bit patterns for every instruction.

| Opcode | Group | Format | Register kinds (rd / ra / rb / rc) |
|---|---|---|---|
| 0x00 | SYS (NOP, EXIT, BAR, FENCE) | R | – |
| 0x01 | S_ALU (scalar integer) | R | s / s / s |
| 0x02 | V_ALU (vector integer) | R | v / v / v-or-s |
| 0x03 | V_FPU (vector float) | R | v / v / v-or-s |
| 0x04 | V_CMP (vector compare → mask) | R | **s** / v / v-or-s |
| 0x05 | V_R4 (FFMA, IMAD, SEL) | R4 | v / v / v-or-s / v (SEL: rc is s) |
| 0x06 | V_LANE (READLANE, SHFL) | R | READLANE: s / v / s; SHFL: v / v / v-or-s |
| 0x07 | GETSR (S_GETSR, V_GETSR) | R | s or v / SR id / – |
| 0x08 | S_ADDI | I | s / s |
| 0x09 | S_ANDI | I | s / s |
| 0x0A | S_ORI | I | s / s |
| 0x0B | S_LUI | I | s / – |
| 0x0C | V_ADDI | I | v / v |
| 0x0D | V_ANDI | I | v / v |
| 0x10 | S_LW | I | s / s |
| 0x11 | S_SW | I | s (data) / s |
| 0x12 | V_LW | I | v / v |
| 0x13 | V_SW | I | v (data) / v |
| 0x14 | V_LDS (shared load) | I | v / v |
| 0x15 | V_STS (shared store) | I | v (data) / v |
| 0x16 | V_ATOM | R | v / v (addr) / v-or-s (operand) |
| 0x20 | S_BZ | I | s (tested) / – |
| 0x21 | S_BNZ | I | s (tested) / – |
| 0x22 | S_J | J | – |
| 0x30–0x3F | reserved for extensions | – | – |

### 5.1 Integer funct (shared by S_ALU and V_ALU)

| funct | Mnemonic | Operation `OP(a, b)` | Scalar | Vector |
|---|---|---|---|---|
| 0x00 | ADD | `u32(a + b)` | yes | yes |
| 0x01 | SUB | `u32(a − b)` | yes | yes |
| 0x02 | MUL | low 32 bits of `a × b` (identical for signed and unsigned) | yes | yes |
| 0x03 | AND | `a & b` | yes | yes |
| 0x04 | OR | `a \| b` | yes | yes |
| 0x05 | XOR | `a ^ b` | yes | yes |
| 0x06 | ANDN | `a & ~b` | yes | yes |
| 0x07 | SHL | `u32(a << (b & 31))` | yes | yes |
| 0x08 | SHR | `a >> (b & 31)`, logical | yes | yes |
| 0x09 | SRA | `a >> (b & 31)`, arithmetic | yes | yes |
| 0x0A | SLT | `s32(a) < s32(b) ? 1 : 0` | yes | yes |
| 0x0B | SLTU | `a < b ? 1 : 0` (unsigned) | yes | yes |
| 0x0C | MIN | signed minimum | yes | yes |
| 0x0D | MAX | signed maximum | yes | yes |
| 0x0E | SHLI | `u32(a << shamt)` | yes | yes |
| 0x0F | SHRI | `a >> shamt`, logical | yes | yes |
| 0x10 | SRAI | `a >> shamt`, arithmetic | yes | yes |
| 0x11 | POPC | number of 1 bits in a | yes | **illegal** |
| 0x12 | FF1 | index of lowest 1 bit in a, or `0xFFFFFFFF` if a = 0 | yes | **illegal** |

### 5.2 Float funct (V_FPU)

| funct | Mnemonic | Operation | Unit |
|---|---|---|---|
| 0x00 | FADD | `a + b` | FPU |
| 0x01 | FSUB | `a − b` | FPU |
| 0x02 | FMUL | `a × b` | FPU |
| 0x03 | FMIN | minimum ([FP-7]) | FPU |
| 0x04 | FMAX | maximum ([FP-7]) | FPU |
| 0x05 | FRCP | ≈ `1 / a` ([sfu.md](sfu.md)) | SFU |
| 0x06 | FRSQ | ≈ `1 / sqrt(a)` | SFU |
| 0x07 | FEX2 | ≈ `2^a` | SFU |
| 0x08 | FLG2 | ≈ `log2(a)` | SFU |
| 0x09 | I2F | signed int → float, round to nearest even | FPU |
| 0x0A | F2I | float → signed int, truncate toward zero, saturate; NaN → 0 | FPU |

FRCP, FRSQ, FEX2, FLG2, I2F and F2I are **unary**: they ignore rb and sb.

### 5.3 Compare funct (V_CMP)

| funct | Cond | | funct | Cond |
|---|---|---|---|---|
| 0x0 | EQ (int) | | 0x8 | FEQ |
| 0x1 | NE (int) | | 0x9 | FNE (true if either is NaN) |
| 0x2 | LT signed | | 0xA | FLT |
| 0x3 | LE signed | | 0xB | FLE |
| 0x4 | GT signed | | 0xC | FGT |
| 0x5 | GE signed | | 0xD | FGE |
| 0x6 | LTU | | 0xE | GTU |
| 0x7 | LEU | | 0xF | GEU |

GTU and GEU are unsigned *integer* compares, placed at 0xE and 0xF. GT and GE exist even though they
are LT and LE with the operands swapped: only rb can be a broadcast scalar, so `v1 > s5` cannot be
written by swapping.

### 5.4 Small groups

| Opcode | funct | Mnemonic |
|---|---|---|
| SYS 0x00 | 0 / 1 / 2 / 3 | NOP / EXIT / BAR / FENCE |
| V_R4 0x05 | 0 / 1 / 2 | V_FFMA / V_IMAD / V_SEL |
| V_LANE 0x06 | 0 / 1 | V_READLANE / V_SHFL |
| GETSR 0x07 | 0 / 1 | S_GETSR / V_GETSR |
| V_ATOM 0x16 | 0–5, +0x10 for shared | ADD / MIN / MAX / XCHG / AND / OR (`.S` = shared) |

---

## 6. Instruction semantics

### 6.1 Notation and common rules

```
u32(x)      x mod 2^32                      s32(x)  x interpreted as two's complement
sext16(i)   i[15:0] sign-extended to 32     zext16(i)  i[15:0] zero-extended
sext26(o)   o[25:0] sign-extended to 32
S[r]        scalar register r; S[0] = 0, S[31] = EXEC
V[r][i]     vector register r in lane i; V[0][i] = 0
EXEC        the value of s31 before this instruction ([EX-3])
active(i)   (EXEC >> i) & 1
B(i)        (sb ? S[rb] : V[rb][i])        -- only for operands written vb|sb
writeS(r,x) if r != 0: S[r] = u32(x)
writeV(r,i,x) if r != 0: V[r][i] = u32(x)
next        PC = u32(PC + 4)               -- the default PC update, unless stated otherwise
```

- **[SEM-1] Read before write.** Every source operand of an instruction is read, in every lane,
  before any destination is written. This includes EXEC, B(i), all lanes of `va` for V_SHFL and
  V_READLANE, and S[rc] for V_SEL. Overlapping source and destination registers therefore behave as
  if the instruction read a snapshot of the registers.
- **[SEM-2]** Writes to s0 and v0 are discarded. Writing s31 updates EXEC for later instructions.
- **[SEM-3]** "for each lane i" loops are conceptually parallel. Where an order matters (stores and
  atomics, §10), this document gives it explicitly.

### 6.2 Scalar ALU (S_ALU, S_ADDI, S_ANDI, S_ORI, S_LUI)

Run once per warp; EXEC is ignored.

```
S_<op>   sd, sa, sb       writeS(rd, OP(S[ra], S[rb]))             ; op from 5.1 (funct 0x00-0x0D)
S_SHLI/S_SHRI/S_SRAI sd, sa, shamt
                          writeS(rd, OP(S[ra], rb_field))
S_POPC   sd, sa           writeS(rd, popcount(S[ra]))
S_FF1    sd, sa           writeS(rd, S[ra] == 0 ? 0xFFFFFFFF : index of lowest set bit)
S_ADDI   sd, sa, simm16   writeS(rd, S[ra] + sext16(imm))
S_ANDI   sd, sa, uimm16   writeS(rd, S[ra] & zext16(imm))
S_ORI    sd, sa, uimm16   writeS(rd, S[ra] | zext16(imm))
S_LUI    sd, uimm16       writeS(rd, imm << 16)
```

### 6.3 Vector integer (V_ALU, V_ADDI, V_ANDI)

```
V_<op>   vd, va, vb|sb    for each lane i with active(i): writeV(rd, i, OP(V[ra][i], B(i)))
V_SHLI/V_SHRI/V_SRAI vd, va, shamt
                          for each active lane i: writeV(rd, i, OP(V[ra][i], rb_field))
V_ADDI   vd, va, simm16   for each active lane i: writeV(rd, i, V[ra][i] + sext16(imm))
V_ANDI   vd, va, uimm16   for each active lane i: writeV(rd, i, V[ra][i] & zext16(imm))
```

### 6.4 Vector float (V_FPU)

```
V_F<op>  vd, va, vb|sb    for each active lane i: writeV(rd, i, F<op>(V[ra][i], B(i)))  ; FADD..FMAX
V_<un>   vd, va           for each active lane i: writeV(rd, i, <un>(V[ra][i]))         ; FRCP..F2I
```

`F<op>` and `<un>` are defined bit-exactly in §8 and [`sfu.md`](sfu.md).

### 6.5 Compare and select

```
V_CMP.<c> sd, va, vb|sb   m = 0
                          for each lane i: if active(i) and COND_c(V[ra][i], B(i)): m |= 1 << i
                          writeS(rd, m)
```

The result is a lane mask in a scalar register, and inactive lanes always produce 0, so a compare
result can be written straight into EXEC to narrow it. The integer conditions compare 32-bit values
signed (LT, LE, GT, GE) or unsigned (LTU, LEU, GTU, GEU). The float conditions are in [FP-8].

```
V_FFMA   vd, va, vb|sb, vc   active lanes: writeV(rd, i, FFMA(V[ra][i], B(i), V[rc][i]))  ; one rounding
V_IMAD   vd, va, vb|sb, vc   active lanes: writeV(rd, i, V[ra][i] × B(i) + V[rc][i])      ; low 32 bits
V_SEL    vd, va, vb|sb, sc   active lanes: writeV(rd, i, ((S[rc] >> i) & 1) ? V[ra][i] : B(i))
```

SEL is the branch-free "if": compute both answers, then pick per lane.

### 6.6 Special registers

```
S_GETSR  sd, sr     writeS(rd, SR[ra])                          ; uniform SR only ([SR-2])
V_GETSR  vd, sr     for each active lane i: writeV(rd, i, SR[ra](lane i))
```

### 6.7 Cross-lane (V_LANE)

```
V_READLANE sd, va, sb    writeS(rd, V[ra][S[rb] & 31])     ; scalar: ignores EXEC; rb is always scalar
V_SHFL     vd, va, vb|sb for each active lane i: writeV(rd, i, V[ra][B(i) & 31])
```

The source lane is read even if it is inactive. With sb = 1, every active lane reads the same source
lane, which makes a lane broadcast. A warp-level reduction takes five SHFL steps (offsets 16, 8, 4,
2, 1) and needs no shared memory.

### 6.8 Memory

```
S_LW   sd, imm(sa)    a = u32(S[ra] + sext16(imm)); CHECK(global, a); writeS(rd, MEM[a])
S_SW   st, imm(sa)    a = u32(S[ra] + sext16(imm)); CHECK(global, a); MEM[a] = S[rd]
V_LW   vd, imm(va)    a_i = u32(V[ra][i] + sext16(imm)) for active i; CHECK_LANES(global)
                      for each active lane i: writeV(rd, i, MEM[a_i])
V_SW   vt, imm(va)    a_i as above; CHECK_LANES(global)
                      for each active lane i in ascending order: MEM[a_i] = V[rd][i]
V_LDS  vd, imm(va)    like V_LW on shared memory
V_STS  vt, imm(va)    like V_SW on shared memory
```

`CHECK` and `CHECK_LANES` are defined in §10.1. Scalar memory instructions ignore EXEC.

### 6.9 Atomics

```
V_ATOM.<op>[.S] vd, va, vb|sb
    a_i = V[ra][i] for active i                      ; no immediate offset
    CHECK_LANES(space)                               ; space = shared if funct bit 4 else global
    for each active lane i in ascending lane order:  ; each step is indivisible ([MEM-5])
        old_i = M[a_i]
        M[a_i] = AOP(old_i, B(i))
    for each active lane i: writeV(rd, i, old_i)
```

| op | `AOP(old, x)` |
|---|---|
| ADD | `u32(old + x)` |
| MIN | signed minimum |
| MAX | signed maximum |
| XCHG | `x` |
| AND | `old & x` |
| OR | `old \| x` |

Unlike plain stores, every lane's update is applied, even when lanes collide on one address. Because
the lane order is fixed, the values returned within one instruction are deterministic.
Compare-and-swap is a listed extension (it needs a fourth operand).

### 6.10 Control flow

```
S_BZ   sd, target    if S[rd] == 0: PC = target  else next      ; target per [ENC-5], imm16
S_BNZ  sd, target    if S[rd] != 0: PC = target  else next
S_J    target        PC = u32(PC + 4 + (sext26(offset26) << 2))
```

All branches are uniform: the whole warp goes one way. There is no branch delay slot and no
call/return in version 1. The idiom `S_BZ s31, skip` means "jump over this code if no lane is active".

### 6.11 System

```
NOP      no effect
EXIT     the warp terminates and releases its resources ([EX-4]); see also [BAR-3]
BAR      block barrier ([BAR-1]..[BAR-4])
FENCE    global-memory ordering point for this warp ([MM-4])
```

---

## 7. Pseudo-instructions

The assembler expands these. They cost no opcode space.

| Pseudo | Expands to | Notes |
|---|---|---|
| `S_MOV sd, sa` | `S_ADD sd, sa, s0` | |
| `S_LI sd, imm32` | `S_LUI sd, imm32[31:16]` ; `S_ORI sd, sd, imm32[15:0]` | Always 2 instructions |
| `V_MOV vd, va` | `V_ADD vd, va, v0` | |
| `V_BCAST vd, sa` | `V_ADD vd, v0, sa` (sb = 1) | |
| `V_NOT vd, va` | `V_SUB vd, v0, va` ; `V_ADDI vd, vd, -1` | `~x = -x - 1`; needs no scratch register, and vd = va is allowed |
| `V_FNEG vd, va, st` | `S_LUI st, 0x8000` ; `V_XOR vd, va, st` | `st` is a scratch scalar register and is overwritten |
| `V_FABS vd, va, st` | `S_LUI st, 0x8000` ; `V_ANDN vd, va, st` | `st` is a scratch scalar register and is overwritten |
| `V_FDIV vd, va, vb, vt` | `V_FRCP vt, vb` ; `V_FMUL vd, va, vt` | `vt` is a scratch vector register; `vt` may equal `vd` only if `vd ≠ va` |
| `S_MOVEXEC sd` | `S_ADD sd, s31, s0` | Saves EXEC |

FNEG and FABS are not instructions: flipping or clearing bit 31 with integer XOR or ANDN is
bit-exact, and canonical-NaN rules do not apply to integer operations.

---

## 8. Floating-point rules

Float instructions use IEEE-754 binary32 encodings, simplified so the FPU stays small and every
result is bit-exact. There is one rounding mode (round to nearest, ties to even), no exception flags
and no traps.

- **[FP-1] Flush-to-zero on input.** Every float instruction (FADD, FSUB, FMUL, FMIN, FMAX, FFMA, the
  SFU ops, F2I, and the float compares) first replaces each denormal input (exponent field 0 with a
  non-zero fraction) by a zero of the same sign. Integer instructions such as XOR/ANDN never flush.
- **[FP-2] Canonical NaN.** Every float instruction that returns a NaN returns exactly `0x7FC00000`.
  NaN payloads and signs are never propagated. An arithmetic float op returns the canonical NaN if
  any input is a NaN, or if the operation is invalid ([FP-4], [FP-5], [FP-6]).
- **[FP-3] Rounding.** If the exact mathematical result *r* is finite and non-zero, it is rounded as
  follows:
  1. Round *r* to 24 significant bits, nearest-even, with an unbounded exponent range, giving *r′*.
  2. If |*r′*| ≥ 2¹²⁸, the result is an infinity with the sign of *r*.
  3. If |*r′*| < 2⁻¹²⁶, the result is a zero with the sign of *r* (output flush; tininess is detected
     *after* rounding).
  4. Otherwise *r′* is encoded as a normal number.

  No instruction ever produces a denormal. The reference definition is `round_fixed` in
  `python/mgpu/fp/f32.py`.
- **[FP-4] FADD / FSUB.** `FSUB(a, b)` is `FADD(a, b')`, where *b′* is *b* with bit 31 flipped (applied
  after the NaN check of [FP-2]). For `FADD(a, b)`:
  - (+∞) + (−∞) is invalid.
  - ∞ + finite = that ∞.
  - If the exact sum is zero: if both operands are zeros of the same sign, the result is that zero;
    otherwise it is +0.
  - Otherwise the result follows [FP-3].
- **[FP-5] FMUL.** 0 × ∞ is invalid. ∞ × (non-zero) = ∞. A zero product is a zero. The result sign is
  the XOR of the operand signs; finite non-zero products follow [FP-3].
- **[FP-6] FFMA(a, b, c) = a × b + c with a single rounding.** After [FP-1] and the NaN check:
  - If a × b is 0 × ∞, the result is invalid.
  - If a × b is an infinity and c is the opposite infinity, the result is invalid.
  - If a × b is an infinity, the result is that infinity; otherwise, if c is an infinity, the result
    is c.
  - Otherwise let *p* = *a*·*b* exactly and *r* = *p* + *c* exactly:
    - If *r* = 0: if *p* and *c* are both zeros of the same sign, the result is that zero; otherwise +0.
    - Otherwise the result follows [FP-3].

  The sign of a zero *p* is the XOR of the signs of *a* and *b*.
- **[FP-7] FMIN / FMAX.** After [FP-1]:
  - If both inputs are NaN, the result is the canonical NaN.
  - If exactly one input is NaN, the result is the other input (already flushed).
  - Otherwise the result is the smaller (FMIN) or larger (FMAX) input, with −0 ordered below +0.
    Equal inputs return that value.
- **[FP-8] Compares.** After [FP-1]:
  - FEQ, FLT, FLE, FGT and FGE are false if either input is NaN.
  - FNE is true if either input is NaN, and otherwise is `!FEQ`.
  - −0 and +0 compare equal; ∞ compares as usual.
- **[FP-9] I2F.** Converts `s32(a)` with [FP-3]. Zero gives +0, and the result is always finite.
- **[FP-10] F2I.** After [FP-1]:
  - NaN → 0.
  - +∞ → `0x7FFFFFFF` and −∞ → `0x80000000`.
  - Otherwise truncate toward zero, then saturate to [−2³¹, 2³¹ − 1].
  - ±0 → 0.
- **[FP-11] SFU.** FRCP, FRSQ, FEX2 and FLG2 are defined bit-exactly by [`sfu.md`](sfu.md). They are
  accurate to better than 1 ULP (at least 23.8 bits), and exact for the special cases listed there.

Consequences *(informative)*. There is no FDIV or FSQRT: division is `a × FRCP(b)` and square root
is `x × FRSQ(x)`. One Newton–Raphson step with FFMA gives full precision when needed.

---

## 9. Divergence *(informative)*

When lanes disagree on a condition, the warp runs **both** paths one after the other, with EXEC
switched so each lane only writes on its own path. MGPU-32 makes the compiler (or you) write that
switching explicitly with scalar instructions. A divergent if/else costs the time of *both* sides,
and a divergent loop runs as many iterations as its *slowest* lane.

### 9.1 if / else

```
; if (a < b) { x = a; } else { x = b; }      a in v1, b in v2, x in v3
        V_CMP.LT  s10, v1, v2      ; s10 = lanes where a < b (already ANDed with EXEC)
        S_MOV     s11, s31         ; save EXEC
        S_MOV     s31, s10         ; EXEC = then-lanes
        S_BZ      s31, else        ; nobody takes 'then'? skip it
        V_MOV     v3, v1           ; then: x = a
else:   S_ANDN    s31, s11, s10    ; EXEC = saved & ~cond = else-lanes
        S_BZ      s31, endif       ; nobody takes 'else'? skip it
        V_MOV     v3, v2           ; else: x = b
endif:  S_MOV     s31, s11         ; restore EXEC (reconverge)
```

The `S_BZ` guards are optional for correctness, since vector ops with EXEC = 0 do nothing, but they
skip wasted cycles. For a body this short, `V_SEL v3, v1, v2, s10` does the same in one instruction.

### 9.2 Nested ifs

Each nesting level saves the current EXEC into a fresh scalar register before narrowing it, and
restores it at its endif. That is a software stack held in scalar registers. It spills to memory with
S_SW when registers run out.

### 9.3 Divergent loops

```
; while (i < n) { ...body...; i++; }         i in v1, per-lane n in v2
        S_MOV     s12, s31         ; save EXEC at loop entry
loop:   V_CMP.LT  s10, v1, v2      ; which active lanes still have i < n?
        S_MOV     s31, s10         ; lanes that finished drop out of EXEC
        S_BZ      s31, done        ; everyone finished: leave
        ...body...
        V_ADDI    v1, v1, 1
        S_J       loop
done:   S_MOV     s31, s12         ; restore EXEC
```

### 9.4 Uniform loops

When the trip count is the same for every lane, keep the counter in a scalar register and never touch
EXEC. This is cheaper, and it is *required* around `BAR`, since every warp must reach the same barriers.

```
        S_ADDI    s12, s0, 16      ; 16 iterations
loop:   ...body...
        S_ADDI    s12, s12, -1
        S_BNZ     s12, loop
```

### 9.5 Early exit for some lanes

`EXIT` ends the whole warp. A per-lane `return` is compiled as "clear this lane's EXEC bit and jump to
the shared exit path". Inside nested ifs, keep a separate "still alive" mask and AND it into every
EXEC restore, so a returned lane is not re-enabled by an outer endif.

---

## 10. Memory access rules

### 10.1 Address checks and faults

`CHECK(space, a)` raises a fault and stops the instruction if:
- `a & 3 != 0`: **misaligned access** (error 2); otherwise
- `a > SIZE(space) − 4`: **out-of-range access** (error 3).

`SIZE(global) = MEM_SIZE` and `SIZE(shared) = 16384`.

`CHECK_LANES(space)` applies `CHECK` to each active lane in ascending lane order. The first lane that
fails sets the error code and the faulting lane.

- **[MEM-3] Precise faults.** A faulting instruction has no architectural effect: no register write,
  no memory write and no atomic update in any lane. Inactive lanes are never checked.
- **[MEM-4] Store collisions.** If two or more active lanes of one V_SW or V_STS write the same address,
  the value from the **highest-numbered** lane is the one stored. This follows from the ascending lane
  order in §6.8.
- **[MEM-5] Atomic indivisibility.** Each lane's read-modify-write in V_ATOM is indivisible with
  respect to every other access to that word, from any warp, block or core.
- **[MEM-6] Coalescing** of lanes that hit the same 128-byte segment is a performance property only.

### 10.2 Instruction fetch

**[MEM-7]** Before executing the instruction at PC, the warp fetches the word at PC. If
`PC > MEM_SIZE − 4`, the fetch raises an out-of-range fault (error 3). PC is always 4-aligned, because
ENTRY is checked at launch ([HOST-4]) and every branch offset is a multiple of 4.

---

## 11. Faults

| Code | Name | Raised by |
|---|---|---|
| 0 | OK | – |
| 1 | Illegal instruction | [DEC-1], [DEC-3] |
| 2 | Misaligned access | `CHECK` (§10.1); misaligned ENTRY ([HOST-4]) |
| 3 | Out-of-range access | `CHECK`; instruction fetch ([MEM-7]) |
| 4 | Barrier mismatch | [BAR-2] |
| 5 | Invalid launch | `BLOCK_DIM` = 0 or > `MAX_BLOCK_DIM` ([HOST-4]) |

Codes 6–255 are reserved.

- **[FLT-1]** When a warp raises a fault, the kernel is **aborted**. The faulting instruction has no
  effect ([MEM-3], [DEC-4]). No warp of the kernel starts a new instruction after the abort is
  recognized. Instructions of *other* warps that are already in progress may complete, so global
  memory contents after an aborted kernel are undefined, apart from the faulting instruction having no
  effect.
- **[FLT-2]** The fault is reported in the host registers (§13.1): STATUS, ERR_PC, ERR_BLOCK, ERR_WARP,
  ERR_LANE and ERR_INFO. If more than one warp faults before the abort takes effect, an implementation
  reports exactly one of those faults, chosen in an implementation-defined way.
- **[FLT-3]** Implementations are not required to detect non-terminating kernels (for example, a warp
  spinning on a flag). A host-side timeout is the remedy.

---

## 12. Synchronization and memory model

### 12.1 BAR

Each resident block has a barrier state: a set **W** of waiting warps and a barrier PC **P**.

- **[BAR-1] Arrival.** When warp *w* executes `BAR` at address *pc*, it joins W, P is set to *pc*, and
  *w* stops issuing instructions.
- **[BAR-2] Mismatch.** If W is non-empty and *pc* ≠ P when *w* arrives, warp *w* raises error 4
  (barrier mismatch), and ERR_INFO = P. Programs must reach barriers in block-uniform control flow:
  every non-exited warp executes the same `BAR` instruction.
- **[BAR-3] Release.** Whenever W is non-empty and every non-exited warp of the block is in W, the
  barrier releases: every warp in W continues at P + 4 and W becomes empty. This check also runs when
  a warp executes `EXIT`. Exited warps are excused from all later barriers, so a warp may leave early
  once it will never be needed at another barrier. A block with a single live warp passes `BAR`
  immediately.
- **[BAR-4] Ordering.** Every shared- and global-memory write made by any warp of the block before it
  arrived at a barrier is visible to every warp of the block after the release.

A barrier can never deadlock under these rules. The common bug where a warp skips a BAR that others
wait on, then reaches a different barrier, is reported deterministically as error 4.

### 12.2 Memory model

MGPU-32 promises very little ordering between threads: program order inside one thread, BAR inside
one block, atomics and FENCE across blocks, and everything once the kernel ends. The rules, from
strongest guarantee to weakest:

1. **[MM-1] Within one lane.** Loads and stores to the same address appear in program order, and a
   lane always reads its own latest write. Scalar memory instructions are ordered with the warp's
   vector memory instructions in the same way.
2. **[MM-2] Within one instruction.** All lanes of a V_SW or V_STS write "at once", and colliding
   addresses resolve by [MEM-4]. Atomics never lose an update.
3. **[MM-3] Within one block.** Every shared and global write made before a BAR is visible to every
   thread of the block after that BAR ([BAR-4]). Without a BAR, another warp in the block may see old
   or new values.
4. **[MM-4] Across blocks.** There is no ordering at all, except that atomics on one address are
   applied one at a time ([MEM-5]), and FENCE keeps one warp's global writes in order: every global
   write before the FENCE becomes visible GPU-wide before any global write after it. This supports the
   pattern "write data, FENCE, then set flag". There is no grid-wide barrier, because blocks are not
   guaranteed to be resident at the same time; a block spinning on another block's flag can deadlock.
5. **[MM-5] With the host.** All of a kernel's writes are visible to the host after the kernel
   completes, and the host's writes before launch are visible to the kernel.

*(Informative.)* Ordering is this weak because strong ordering forces every core to coordinate with
every other core on every store. GPUs bet that most kernels communicate only through BAR within a
block and through separate kernel launches across the whole grid.

---

## 13. Kernel launch and host interface

The host launches a kernel by writing a small set of memory-mapped control registers and then writing
1 to GO. The GPU sets up every warp's registers and starts them at the entry point. These registers
sit on the host's control bus, **not** in the GPU's global address space. How the host reads and writes
global memory is implementation-defined (a simulator API, or a host memory port in RTL), but such
accesses must not overlap a running kernel.

### 13.1 Host register map

All registers are 32 bits wide, accessed with aligned 32-bit reads and writes. Unlisted offsets read
as 0 and ignore writes.

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x00 | `ID` | RO | `0x4D475055` (ASCII "MGPU") |
| 0x04 | `VERSION` | RO | ISA version, `major << 16 \| minor`: `0x00010000` |
| 0x08 | `MAX_BLOCK_DIM` | RO | Implementation parameter (§1.2) |
| 0x0C | `MEM_SIZE` | RO | Implementation parameter, in bytes; 0 means 2³² |
| 0x10 | `ENTRY` | RW | Byte address of the kernel's first instruction |
| 0x14 | `GRID_DIM` | RW | Number of blocks |
| 0x18 | `BLOCK_DIM` | RW | Threads per block |
| 0x1C | `GO` | RW | Write 1 to start; reads 1 while running, 0 when idle |
| 0x20 | `STATUS` | RO | Bits [7:0]: error code of the last launch (§11); other bits 0 |
| 0x24 | `ERR_PC` | RO | PC of the faulting instruction |
| 0x28 | `ERR_BLOCK` | RO | `BLOCK_ID` of the faulting warp |
| 0x2C | `ERR_WARP` | RO | `WARP_ID` of the faulting warp |
| 0x30 | `ERR_LANE` | RO | Faulting lane for data-access faults; otherwise 0 |
| 0x34 | `ERR_INFO` | RO | Code 1: the instruction word. Codes 2/3: the faulting address (for fetch faults, the PC). Code 4: the barrier PC P of [BAR-2]. Code 5: 0 |
| 0x38 | `CYCLES` | RO | Implementation-defined length of the last launch in cycles, or 0 |
| 0x40–0x5C | `ARG0`–`ARG7` | RW | Eight 32-bit kernel arguments (pointers, sizes, float bits) |

### 13.2 Launch sequence

- **[HOST-1]** The host writes ENTRY, GRID_DIM, BLOCK_DIM and ARG0–ARG7 while GO reads 0. Writes to
  these registers while GO reads 1 are ignored.
- **[HOST-2]** Writing a value with bit 0 = 1 to GO while GO reads 0 starts a launch. Other writes to
  GO are ignored; in particular, a running kernel cannot be cancelled through GO.
- **[HOST-3]** At launch, STATUS and ERR_* are cleared to 0.
- **[HOST-4]** The launch is validated in this order:
  1. If `BLOCK_DIM = 0` or `BLOCK_DIM > MAX_BLOCK_DIM`, STATUS = 5.
  2. Otherwise, if `ENTRY & 3 ≠ 0`, STATUS = 2, ERR_PC = ERR_INFO = ENTRY.
  3. Otherwise, if `GRID_DIM = 0`, the launch completes immediately with STATUS = 0.

  In each of these cases, no instruction executes and GO reads 0 afterwards.
- **[HOST-5]** Otherwise GO reads 1 until every block has finished (STATUS = 0) or the kernel has
  aborted (STATUS ≠ 0); after that, GO reads 0.

### 13.3 Warp state at start

| State | Initial value |
|---|---|
| PC | `ENTRY` |
| s1–s8 | `ARG0`–`ARG7` |
| s9–s30 | 0 |
| s31 (EXEC) | Bit *i* = 1 if `32 × WARP_ID + i < BLOCK_DIM`: all ones except possibly in the last warp |
| v1–v31 | 0 in every lane |
| Shared memory | Undefined |

Thread and block IDs are *not* preloaded into registers; read them with GETSR. That keeps launch
simple and leaves all general registers free.

### 13.4 Register conventions *(informative)*

| Registers | Use |
|---|---|
| s0, v0 | Hard-wired zero |
| s1–s8 | Kernel arguments at entry; free to reuse afterward |
| s9–s30 | Saved EXEC masks, loop counters, uniform addresses, pseudo-instruction scratch |
| s31 | EXEC |
| v1–v31 | Per-thread values |

There is no stack and no calling convention in version 1, because there are no calls.

---

## 14. Assembly syntax

- One instruction per line. `;` starts a comment, and `label:` defines a branch target.
- Operands are written `rd, ra, rb[, rc]`, and memory operands as `imm(base)`.
- Mnemonic suffixes select the funct: `V_CMP.LT`, `V_ATOM.ADD`, `V_ATOM.ADD.S` (shared).
- Writing an `s` register in the rb slot of a `vb|sb` operand sets the sb bit.
- Mnemonics, register names and SR names are case-insensitive. `exec` is an alias for `s31`.
- The operand of a branch is a label or a signed integer instruction offset ([ENC-5]).
- Special registers are written by name (`GTID`) or number.

The complete grammar (expressions, directives, output formats) is defined in the toolchain
documentation, `docs/toolchain/assembler.md`.

---

## 15. Example kernels *(informative)*

### 15.1 SAXPY: y[i] = a × x[i] + y[i]

Arguments: s1 = n, s2 = a (float bits), s3 = &x, s4 = &y. Launch with `GRID_DIM = ceil(n / BLOCK_DIM)`.

```
saxpy:  V_GETSR   v1, GTID          ; i = global thread index
        V_CMP.LT  s10, v1, s1       ; lanes with i < n
        S_MOV     s31, s10          ; turn off out-of-range lanes
        S_BZ      s31, end          ; whole warp out of range
        V_SHLI    v2, v1, 2         ; byte offset = i * 4
        V_ADD     v3, v2, s3        ; &x[i]   (s3 broadcast)
        V_ADD     v4, v2, s4        ; &y[i]
        V_LW      v5, 0(v3)         ; x[i]    (gather; coalesces to 1 transaction)
        V_LW      v6, 0(v4)         ; y[i]
        V_FFMA    v6, v5, s2, v6    ; a * x[i] + y[i]
        V_SW      v6, 0(v4)         ; y[i] = result
end:    EXIT
```

With n = 40 and `BLOCK_DIM = 32`, the launch is 2 blocks of 1 warp each. Block 1's lanes have GTID
32–63, so the compare leaves EXEC = `0x000000FF` (lanes 0–7, i.e. i = 32–39).

### 15.2 Block sum reduction

Each block sums its slice of `in` and writes one float to `out[BLOCK_ID]`. Arguments: s1 = &in,
s2 = &out, s3 = n. `BLOCK_DIM` must be a power of two.

```
reduce: V_GETSR   v1, GTID
        V_GETSR   v2, TID
        S_MOV     s11, s31          ; save launch EXEC
        V_MOV     v4, v0            ; default 0.0 for out-of-range lanes
        V_CMP.LT  s10, v1, s3       ; in-range lanes
        S_MOV     s31, s10
        V_SHLI    v3, v1, 2
        V_ADD     v3, v3, s1        ; &in[i]
        V_LW      v4, 0(v3)         ; only in-range lanes load
        S_MOV     s31, s11          ; all threads again
        V_SHLI    v5, v2, 2         ; shared address = tid * 4
        V_STS     v4, 0(v5)         ; sh[tid] = value
        BAR                         ; whole array in shared memory

        S_GETSR   s12, BLOCK_DIM
        S_SHRI    s12, s12, 1       ; stride = BLOCK_DIM / 2   (uniform)
loop:   S_BZ      s12, done
        V_CMP.LT  s10, v2, s12      ; tid < stride
        S_MOV     s31, s10
        S_SHLI    s13, s12, 2       ; stride in bytes
        V_ADD     v6, v5, s13       ; &sh[tid + stride]
        V_LDS     v7, 0(v6)
        V_LDS     v8, 0(v5)
        V_FADD    v8, v8, v7
        V_STS     v8, 0(v5)         ; sh[tid] += sh[tid + stride]
        S_MOV     s31, s11
        BAR                         ; every warp reaches this: loop count is uniform
        S_SHRI    s12, s12, 1
        S_J       loop

done:   V_CMP.EQ  s10, v2, v0       ; tid == 0
        S_MOV     s31, s10
        S_BZ      s31, end          ; only warp 0 continues
        V_LDS     v8, 0(v0)         ; sh[0] = block total
        S_GETSR   s14, BLOCK_ID
        S_SHLI    s14, s14, 2
        S_ADD     s14, s14, s2      ; &out[BLOCK_ID]
        V_BCAST   v9, s14
        V_SW      v8, 0(v9)
end:    EXIT
```

Every BAR in this kernel is reached by all warps at the same PC. Warps other than warp 0 exit after
the last BAR, which is allowed by [BAR-3].

---

## 16. Left out on purpose, and where to take it next *(informative)*

Version 1 leaves out everything a compute kernel can live without. Each item below is a
self-contained extension that fits in the reserved opcodes 0x30–0x3F.

| Left out | Why it is fine for now | How you would add it |
|---|---|---|
| Byte/halfword memory access | Most GPU data is 32-bit | `V_LB`, `V_LH`, `V_SB`, `V_SH` in I format |
| Integer divide | Rare in kernels; slow in hardware | Software routine, or a multi-cycle `V_DIV` |
| Compare-and-swap | ADD/XCHG cover most uses | R4 format: address, expected, new |
| Calls and a stack | GPU compilers inline everything | `S_JAL`, `S_JR`, plus a per-lane scratch region |
| 2-D/3-D thread IDs | A 1-D index plus divide works | Special registers `TID_Y`, `BLOCK_ID_Y`, … in the reserved SR IDs |
| FP16 / packed math | Precision rarely matters for a first build | Two FP16 values per register, `V_HFMA2` |
| Texture sampling | Graphics, not compute | A separate texture unit with its own load opcode |
| Matrix (tensor) instructions | A throughput feature, not a functional one | One instruction doing a small tile multiply across the warp |
| Caches and coherence | Not visible to software here | Purely microarchitectural |

The original note's reference microarchitecture (its §12) is not part of the ISA. It is developed
into a full specification under `docs/uarch/`.
