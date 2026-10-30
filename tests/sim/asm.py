"""A tiny test-only assembler: labels, numeric operands, and the simple pseudo-instructions.

The real assembler (mgpu.asm) arrives in phase 3; this helper keeps the simulator tests
readable until then.
"""

import re

from mgpu.isa import BY_MNEMONIC, assemble_one

BRANCH = {"S_BZ": 1, "S_BNZ": 1, "S_J": 0}   # mnemonic -> index of the target operand


def _pseudo(mn, ops):
    if mn == "S_MOV":
        return [f"S_ADD {ops[0]}, {ops[1]}, s0"]
    if mn == "V_MOV":
        return [f"V_ADD {ops[0]}, {ops[1]}, v0"]
    if mn == "V_BCAST":
        return [f"V_ADD {ops[0]}, v0, {ops[1]}"]
    if mn == "S_LI":
        v = int(ops[1], 0) & 0xFFFFFFFF
        return [f"S_LUI {ops[0]}, {v >> 16}", f"S_ORI {ops[0]}, {ops[0]}, {v & 0xFFFF}"]
    if mn == "V_NOT":
        return [f"V_SUB {ops[0]}, v0, {ops[1]}", f"V_ADDI {ops[0]}, {ops[0]}, -1"]
    return None


def assemble(src: str, base: int = 0) -> list[int]:
    lines = []
    for raw in src.splitlines():
        line = raw.split(";")[0].strip()
        while (m := re.match(r"^([A-Za-z_]\w*):\s*(.*)$", line)):
            lines.append(("label", m[1]))
            line = m[2]
        if not line:
            continue
        if line.startswith(".word"):
            lines += [("word", int(x, 0)) for x in line[5:].split(",")]
            continue
        mn, _, rest = line.partition(" ")
        ops = [o.strip() for o in rest.split(",")] if rest.strip() else []
        exp = _pseudo(mn.upper(), ops)
        lines += [("ins", x) for x in (exp or [line])]
    labels, pc = {}, base
    for kind, val in lines:
        if kind == "label":
            labels[val] = pc
        else:
            pc += 4
    words, pc = [], base
    for kind, val in lines:
        if kind == "label":
            continue
        if kind == "word":
            words.append(val & 0xFFFFFFFF)
        else:
            mn, _, rest = val.partition(" ")
            ops = [o.strip() for o in rest.split(",")] if rest.strip() else []
            ti = BRANCH.get(mn.upper())
            if ti is not None and ops[ti] in labels:
                ops[ti] = str((labels[ops[ti]] - (pc + 4)) // 4)
            assert mn.upper() in BY_MNEMONIC, f"unknown mnemonic {mn}"
            words.append(assemble_one(f"{mn} {', '.join(ops)}"))
        pc += 4
    return words
