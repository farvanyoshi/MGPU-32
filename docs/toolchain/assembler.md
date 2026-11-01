# MGPU-32 assembler (`mgpu-as`)

The assembler turns MGPU-32 assembly into a flat memory image. It implements the syntax of ISA §14
and extends it with expressions, directives and pseudo-instructions. Instruction operand syntax is
taken from `spec/isa.toml`, so the assembler always matches the ISA tables.

```bash
mgpu-as kernel.s                                   # -> kernel.bin (load address 0x1000)
mgpu-as kernel.s -o k.bin --base 0x2000 --hex k.hex --sym k.sym.json --lst k.lst
```

```python
from mgpu.asm import assemble, assemble_file, AsmError
prog = assemble(source_text, base=0x1000)          # Program: .image .words .symbols .kernels
prog.entry("saxpy")                                # address of a kernel
```

## 1. Lines

```
[label:]... [statement] [; comment]
```

- A line may hold any number of labels, then at most one statement: an instruction, a
  pseudo-instruction, a directive or an assignment.
- `;` starts a comment that runs to the end of the line. It is ignored inside character literals.
- Mnemonics, register names and special-register names are **case-insensitive**. Symbols are
  case-sensitive.
- Symbol names match `[A-Za-z_][A-Za-z0-9_]*`. Register names (`s0`–`s31`, `v0`–`v31`, `exec`)
  cannot be used as symbols.

## 2. Operands

Each instruction takes the operands listed in its syntax in
[`encoding-tables.md`](../isa/encoding-tables.md).

| Syntax token | Accepted operand |
|---|---|
| `sd`, `sa`, `sb`, `sc`, `st` | A scalar register `s0`–`s31`; `exec` is `s31` |
| `vd`, `va`, `vc`, `vt` | A vector register `v0`–`v31` |
| `vb\|sb` | A vector register, or a scalar register (which sets the sb bit) |
| `shamt` | An expression in 0–31 |
| `simm16` | An expression in −32768–32767 |
| `uimm16` | An expression in 0–65535. Negative values are rejected: these immediates are zero-extended |
| `sr` | A special-register name (`GTID`, `BLOCK_ID`, …) or number. Reserved IDs are rejected, and so are per-lane SRs in `S_GETSR` |
| `simm16(sa)`, `simm16(va)` | `offset(register)` or `(register)`. The offset is an expression |
| `target16`, `target26` | A branch target, see §4 |

Every value is range-checked, and register kinds are enforced. A wrong operand is an error, never a
silent truncation.

## 3. Expressions

Expressions use C precedence, from loosest to tightest binding: `|`, `^`, `&`, `<< >>`, `+ -`,
`* / %`, then unary `- + ~`. Parentheses group. Arithmetic uses unbounded integers; `/` and `%`
truncate toward zero, as in C.

| Form | Meaning |
|---|---|
| `42`, `0x2A`, `0b101010`, `1_000` | Integer literals (`_` separates digits) |
| `'A'`, `'\n'` | Character code (escapes: `\n \t \r \0 \\ \' \"`) |
| `name` | A label (an address) or a constant from `=`/`.equ` |
| `.` | Address of the current statement |
| `%hi(e)`, `%lo(e)` | Bits [31:16] and [15:0] of `e` |
| `f32(1.5)` | The binary32 bits of a float literal (nearest-even; `inf`, `nan` allowed) |

Each value also records whether it is an **address**. Labels and `.` are addresses; `a + b` is an
address if either side is; `addr − addr` is a plain number; every other operator gives a plain number.

## 4. Branch operands

`S_BZ`, `S_BNZ` and `S_J` take one target operand:

- If the expression is an **address** (it uses a label or `.`), it is the target. The assembler
  computes the instruction offset `(target − (PC + 4)) / 4` ([ENC-5]). A target that is not 4-byte
  aligned, or out of range (±32 K instructions for branches, ±32 M for `S_J`), is an error.
- Otherwise the value **is** the instruction offset, as in the spec's `S_BNZ s12, -5`. This includes
  constants defined with `=`.

`S_J .` is an infinite loop, and `S_J . + 8` skips the next instruction.

## 5. Directives

| Directive | Effect |
|---|---|
| `.org addr` | Continue at `addr`, zero-filling the gap. It may only move forward |
| `.align n` | Pad with zeros to a multiple of `n` bytes (`n` a power of two) |
| `.word e, …` | 32-bit little-endian words. Values may lie in −2³¹–2³²−1 and may use labels |
| `.float f, …` | binary32 values of float literals, e.g. `1.5`, `-0.0`, `1e-3`, `inf`, `nan` |
| `.space n[, fill]` | `n` bytes of `fill` (default 0) |
| `name = e`, `.equ name, e`, `.set name, e` | Define a constant. Forward references are fine; cycles are errors |
| `.kernel name` | Export the label `name` as a launchable kernel entry, recorded in the symbol map |
| `.include "file"` | Insert another source file, resolved relative to the including file |

Instructions, `.word` and `.float` must start at a 4-byte-aligned address; after `.space` with an
odd size, use `.align 4`. The size of `.org`, `.align` and `.space` must be computable when they are
reached: they may use constants and labels defined *above* them.

## 6. Pseudo-instructions

Pseudo-instructions expand to fixed instruction sequences (ISA §7, `[[pseudo]]` in `spec/isa.toml`).
Their sizes never depend on operand values, so code layout is always predictable.

| Pseudo | Expansion | Words |
|---|---|---|
| `S_MOV sd, sa` | `S_ADD sd, sa, s0` | 1 |
| `S_LI sd, imm32` | `S_LUI sd, %hi(imm32)` ; `S_ORI sd, sd, %lo(imm32)` | 2 |
| `V_MOV vd, va` | `V_ADD vd, va, v0` | 1 |
| `V_BCAST vd, sa` | `V_ADD vd, v0, sa` | 1 |
| `V_NOT vd, va` | `V_SUB vd, v0, va` ; `V_ADDI vd, vd, -1` | 2 |
| `V_FNEG vd, va, st` | `S_LUI st, 0x8000` ; `V_XOR vd, va, st` | 2 |
| `V_FABS vd, va, st` | `S_LUI st, 0x8000` ; `V_ANDN vd, va, st` | 2 |
| `V_FDIV vd, va, vb, vt` | `V_FRCP vt, vb` ; `V_FMUL vd, va, vt` | 2 |
| `S_MOVEXEC sd` | `S_ADD sd, s31, s0` | 1 |

Notes:
- `S_LI` accepts any value from −2³¹ to 2³²−1, including labels and `f32(...)`.
- `st` and `vt` are scratch registers that the expansion overwrites.
- `V_FDIV` requires `vt ≠ va`. It may use `vt = vd` when `vd ≠ va`.

## 7. Diagnostics

The assembler reports **every** error it finds, up to 20, in source order:

```
kernel.s:12: error: S_ADDI: signed immediate 40000 out of range [-32768, 32767]
    S_ADDI s1, s2, 40000
kernel.s:14: error: unknown mnemonic 'V_FOO' (did you mean V_OR?)
    V_FOO v1
```

The Python API raises `AsmError`; its `.diagnostics` holds `Diagnostic(file, line, message, text)`
entries. `mgpu-as` prints them and exits with status 1.

## 8. Output formats

| Option | Format |
|---|---|
| `-o out.bin` | Flat little-endian image starting at the base address. Gaps are zero-filled and the size is a multiple of 4 |
| `--hex out.hex` | Verilog `$readmemh` image for 32-bit memories: a first line `@<base/4>` in hex, then one 8-digit word per line |
| `--sym out.sym.json` | `{"base", "size", "kernels": {name: addr}, "labels": {name: addr}, "constants": {name: value}}` |
| `--lst out.lst` | Listing: address, encoded word(s) and source line for every statement |

The default base address is `0x1000`. Programs are position-dependent: assemble them for the address
where they will be loaded. `Device.load()` and `mgpu-run` do this for you.

## 9. A complete example

```
; saxpy.s - y[i] = a * x[i] + y[i]
; args: s1 = n, s2 = a (float bits), s3 = &x, s4 = &y; grid = ceil(n / block)
        .kernel saxpy
saxpy:  V_GETSR   v1, GTID          ; i = global thread index
        V_CMP.LT  s10, v1, s1       ; lanes with i < n
        S_MOV     exec, s10
        S_BZ      exec, done        ; whole warp out of range
        V_SHLI    v2, v1, 2
        V_ADD     v3, v2, s3        ; &x[i]
        V_ADD     v4, v2, s4        ; &y[i]
        V_LW      v5, (v3)
        V_LW      v6, (v4)
        V_FFMA    v6, v5, s2, v6
        V_SW      v6, (v4)
done:   EXIT
```
