"""Disassemble MGPU-32 instruction words into canonical assembly text."""

from .encoding import SR_BY_ID, IllegalInstruction, decode


def _reg(kind: str, n: int) -> str:
    return f"{kind}{n}"


def disasm(word: int, pc: int | None = None) -> str:
    """One instruction as text. Branch targets are shown as offsets, plus the absolute
    target address when `pc` is given. Illegal words become `.word 0x...`."""
    try:
        d = decode(word)
    except IllegalInstruction:
        return f".word {word & 0xFFFFFFFF:#010x}"
    ins = d.instr
    field = {"d": d.rd, "t": d.rd, "a": d.ra, "b": d.rb, "c": d.rc}
    ops = []
    for tok in ins.syntax:
        if tok == "vb|sb":
            ops.append(_reg("s" if d.sb else "v", d.rb))
        elif tok == "shamt":
            ops.append(str(d.rb))
        elif tok == "sr":
            sr = SR_BY_ID.get(d.ra)
            ops.append(sr["name"] if sr else str(d.ra))
        elif tok == "simm16":
            ops.append(str(d.simm16))
        elif tok == "uimm16":
            ops.append(f"{d.imm16:#x}")
        elif tok in ("target16", "target26"):
            off = d.simm16 if tok == "target16" else d.soffset26
            t = f"{off}"
            if pc is not None:
                t += f"  ; -> {(pc + 4 + 4 * off) & 0xFFFFFFFF:#x}"
            ops.append(t)
        elif "(" in tok:
            ops.append(f"{d.simm16}({tok[-3]}{d.ra})")
        else:
            ops.append(_reg(tok[0], field[tok[-1]]))
    text = ins.mnemonic + (" " + ", ".join(ops) if ops else "")
    if ";" in text:  # keep the branch comment last
        head, _, comment = text.partition(";")
        text = head.rstrip() + "  ;" + comment
    return text


def listing(words, base: int = 0) -> str:
    return "\n".join(f"{base + 4 * i:08x}: {w & 0xFFFFFFFF:08x}  {disasm(w, base + 4 * i)}"
                     for i, w in enumerate(words))
