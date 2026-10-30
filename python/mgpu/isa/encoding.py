"""Encode and decode MGPU-32 instruction words (docs/isa/MGPU-32-ISA.md, sections 5 and 6).

The decoder implements the normative decode policy (rule DEC-1): the opcode and the full
funct field must name a listed instruction, otherwise the word is illegal. Every other
field that an instruction does not use is ignored.
"""

import re
from dataclasses import dataclass

from .tables import GROUPS, PSEUDOS, SRS  # noqa: F401  (re-exported for tools)


class IllegalInstruction(Exception):
    pass


@dataclass(frozen=True)
class Instr:
    mnemonic: str
    group: str
    opcode: int
    funct: int | None
    format: str
    syntax: tuple
    op: str
    exec: str
    unit: str
    space: str | None


def _build():
    by_mn, by_op = {}, {}
    for g in GROUPS:
        for i in g["instructions"]:
            ins = Instr(
                mnemonic=i["mnemonic"], group=g["name"], opcode=g["opcode"],
                funct=i.get("funct"), format=g["format"],
                syntax=tuple(t.strip() for t in i["syntax"].split(",") if t.strip()),
                op=i["op"], exec=i.get("exec", g["exec"]), unit=i.get("unit", g["unit"]),
                space=i.get("space"))
            by_mn[ins.mnemonic] = ins
            by_op.setdefault(ins.opcode, {})[ins.funct] = ins
    return by_mn, by_op


BY_MNEMONIC, BY_OPCODE = _build()
SR_BY_ID = {s["id"]: s for s in SRS}
SR_BY_NAME = {s["name"]: s for s in SRS}


@dataclass(frozen=True)
class Decoded:
    instr: Instr
    rd: int
    ra: int
    rb: int
    rc: int
    sb: int
    imm16: int
    offset26: int

    @property
    def simm16(self) -> int:
        return self.imm16 - 0x10000 if self.imm16 & 0x8000 else self.imm16

    @property
    def soffset26(self) -> int:
        return self.offset26 - (1 << 26) if self.offset26 & (1 << 25) else self.offset26


def decode(word: int) -> Decoded:
    word &= 0xFFFFFFFF
    opcode = word >> 26
    group = BY_OPCODE.get(opcode)
    if group is None:
        raise IllegalInstruction(f"reserved opcode {opcode:#04x}")
    any_ins = next(iter(group.values()))
    fmt = any_ins.format
    rd, ra, rb = (word >> 21) & 31, (word >> 16) & 31, (word >> 11) & 31
    rc = (word >> 6) & 31
    if fmt == "R":
        sb, funct = (word >> 10) & 1, word & 0x3FF
    elif fmt == "R4":
        sb, funct = (word >> 5) & 1, word & 0x1F
    else:
        sb, funct = 0, None
    ins = group.get(funct)
    if ins is None:
        raise IllegalInstruction(f"opcode {opcode:#04x}: unassigned funct {funct:#x}")
    return Decoded(ins, rd, ra, rb, rc, sb, word & 0xFFFF, word & 0x3FFFFFF)


def encode(mnemonic: str, rd=0, ra=0, rb=0, rc=0, sb=0, imm=0, offset=0) -> int:
    """Pack fields for `mnemonic`. imm/offset may be negative (two's complement)."""
    ins = BY_MNEMONIC[mnemonic.upper()]
    w = ins.opcode << 26
    if ins.format == "R":
        w |= rd << 21 | ra << 16 | rb << 11 | sb << 10 | ins.funct
    elif ins.format == "R4":
        w |= rd << 21 | ra << 16 | rb << 11 | rc << 6 | sb << 5 | ins.funct
    elif ins.format == "I":
        w |= rd << 21 | ra << 16 | (imm & 0xFFFF)
    else:
        w |= offset & 0x3FFFFFF
    return w


_REG = re.compile(r"^(?P<k>[sv])(?P<n>\d+)$")
_MEM = re.compile(r"^(?P<imm>[^()]+)\((?P<base>[^()]+)\)$")


def _reg(tok: str, kinds: str) -> tuple[str, int]:
    tok = tok.strip().lower()
    if tok == "exec":
        tok = "s31"
    m = _REG.match(tok)
    if not m or int(m["n"]) > 31 or m["k"] not in kinds:
        raise ValueError(f"expected {'/'.join(kinds)} register, got {tok!r}")
    return m["k"], int(m["n"])


def assemble_one(text: str) -> int:
    """Encode one instruction with numeric operands only (no labels, no pseudos).

    This is a small helper for conformance vectors and tests; the full assembler
    lives in mgpu.asm.
    """
    text = text.split(";")[0].strip()
    mn, _, rest = text.partition(" ")
    ins = BY_MNEMONIC[mn.upper()]
    ops = [o.strip() for o in rest.split(",")] if rest.strip() else []
    if len(ops) != len(ins.syntax):
        raise ValueError(f"{mn}: expected {len(ins.syntax)} operands, got {len(ops)}")
    f = dict(rd=0, ra=0, rb=0, rc=0, sb=0, imm=0, offset=0)

    def ranged(text, lo, hi, what):
        val = int(text, 0)
        if not lo <= val <= hi:
            raise ValueError(f"{mn}: {what} {val} out of range [{lo}, {hi}]")
        return val

    field = {"d": "rd", "t": "rd", "a": "ra", "b": "rb", "c": "rc"}
    for tok, op in zip(ins.syntax, ops):
        if tok == "vb|sb":
            k, n = _reg(op, "vs")
            f["rb"], f["sb"] = n, int(k == "s")
        elif tok == "shamt":
            f["rb"] = ranged(op, 0, 31, "shift amount")
        elif tok == "sr":
            f["ra"] = (SR_BY_NAME[op.upper()]["id"] if op.upper() in SR_BY_NAME
                       else ranged(op, 0, 31, "special register"))
        elif tok in ("simm16", "target16"):
            f["imm"] = ranged(op, -0x8000, 0x7FFF, "immediate")
        elif tok == "uimm16":
            f["imm"] = ranged(op, 0, 0xFFFF, "immediate")
        elif tok == "target26":
            f["offset"] = ranged(op, -(1 << 25), (1 << 25) - 1, "offset")
        elif "(" in tok:
            m = _MEM.match(op.replace(" ", ""))
            f["imm"] = ranged(m["imm"], -0x8000, 0x7FFF, "offset")
            f["ra"] = _reg(m["base"], tok[-3])[1]
        else:
            f[field[tok[-1]]] = _reg(op, tok[0])[1]
    return encode(ins.mnemonic, **f)
