"""Disassemble MGPU-32 instruction words into assembly text.

`disasm` renders one word. `disassemble` renders a whole image as source that
mgpu.asm reassembles to the *identical* bytes: words that are illegal, or that set fields
the instruction ignores (non-canonical encodings, rule DEC-2), are emitted as `.word`
with the decoded instruction in a comment.
"""

from .encoding import SR_BY_ID, Decoded, IllegalInstruction, decode, encode

_FIELD = {"d": "rd", "t": "rd", "a": "ra", "b": "rb", "c": "rc"}


def _sr_legal(d: Decoded) -> bool:
    sr = SR_BY_ID.get(d.ra)
    return sr is not None and not (d.instr.mnemonic == "S_GETSR" and sr["kind"] == "lane")


def canonical(d: Decoded) -> int:
    """The word with every field the instruction ignores cleared ([DEC-2])."""
    f = dict(rd=0, ra=0, rb=0, rc=0, sb=0, imm=0, offset=0)
    for tok in d.instr.syntax:
        if tok == "vb|sb":
            f["rb"], f["sb"] = d.rb, d.sb
        elif tok == "shamt":
            f["rb"] = d.rb
        elif tok == "sr":
            f["ra"] = d.ra
        elif tok in ("simm16", "uimm16", "target16"):
            f["imm"] = d.imm16
        elif tok == "target26":
            f["offset"] = d.offset26
        elif "(" in tok:
            f["imm"], f["ra"] = d.imm16, d.ra
        else:
            f[_FIELD[tok[-1]]] = getattr(d, _FIELD[tok[-1]])
    return encode(d.instr.mnemonic, **f)


def _render(d: Decoded, pc: int | None, labels: dict | None) -> tuple[str, str]:
    """(instruction text, comment)."""
    ops, comment = [], ""
    for tok in d.instr.syntax:
        if tok == "vb|sb":
            ops.append(f"{'s' if d.sb else 'v'}{d.rb}")
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
            if pc is not None:
                target = (pc + 4 + 4 * off) & 0xFFFFFFFF
                if labels and target in labels:
                    ops.append(labels[target])
                    continue
                comment = f"-> {target:#x}"
            ops.append(str(off))
        elif "(" in tok:
            ops.append(f"{d.simm16}({tok[-3]}{d.ra})")
        else:
            ops.append(f"{tok[0]}{getattr(d, _FIELD[tok[-1]])}")
    return d.instr.mnemonic + (" " + ", ".join(ops) if ops else ""), comment


def disasm(word: int, pc: int | None = None, labels: dict | None = None) -> str:
    """One word as text. With `pc`, branch targets are annotated (or shown as labels from
    `labels`, a dict address -> name). Words that cannot be reassembled as written become
    `.word`."""
    word &= 0xFFFFFFFF
    try:
        d = decode(word)
    except IllegalInstruction:
        return f".word {word:#010x}"
    text, comment = _render(d, pc, labels)
    if d.instr.group == "GETSR" and not _sr_legal(d):
        return f".word {word:#010x}  ; {text} (illegal special register)"
    if canonical(d) != word:
        return f".word {word:#010x}  ; {text} (non-canonical encoding)"
    return f"{text}  ; {comment}" if comment else text


def listing(words, base: int = 0) -> str:
    """Address, word and text per line (not reassemblable; see `disassemble`)."""
    return "\n".join(f"{base + 4 * i:08x}: {w & 0xFFFFFFFF:08x}  {disasm(w, base + 4 * i)}"
                     for i, w in enumerate(words))


def disassemble(words, base: int = 0, symbols: dict | None = None) -> str:
    """A whole image as reassemblable source. `symbols` maps names to addresses; names at
    instruction addresses become labels, and branch targets use them."""
    labels: dict[int, str] = {}
    for name, addr in sorted((symbols or {}).items(), key=lambda kv: (kv[1], kv[0])):
        labels.setdefault(addr, name)
    extra = {a: n for n, a in (symbols or {}).items() if labels.get(a) != n}
    out = []
    for i, w in enumerate(words):
        pc = base + 4 * i
        for name in sorted(n for a, n in extra.items() if a == pc):
            out.append(f"{name}:")
        if pc in labels:
            out.append(f"{labels[pc]}:")
        text = disasm(w, pc, labels)
        out.append(f"    {text:<40} ; {pc:08x}: {w & 0xFFFFFFFF:08x}" if ";" not in text else
                   f"    {text}  [{pc:08x}: {w & 0xFFFFFFFF:08x}]")
    return "\n".join(out) + "\n"
