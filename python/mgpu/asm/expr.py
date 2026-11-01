"""Constant expressions for the MGPU-32 assembler (docs/toolchain/assembler.md, "Expressions").

An expression evaluates to (value, is_address). `is_address` is true when the value is
derived from a label or `.` (the current address). Branch operands use it to tell a target
address from a raw instruction offset.
"""

import re
import struct
from dataclasses import dataclass


class ExprError(Exception):
    pass


class Undefined(ExprError):
    def __init__(self, name):
        super().__init__(f"undefined symbol '{name}'")
        self.name = name


_F32 = re.compile(r"\bf32\s*\(([^()]*)\)", re.IGNORECASE)
_TOKEN = re.compile(r"""\s*(?:
    (?P<num>0[xX][0-9a-fA-F_]+|0[bB][01_]+|\d[\d_]*)
   |(?P<char>'(?:\\.|[^\\'])')
   |(?P<func>%(?:hi|lo))
   |(?P<id>[A-Za-z_][A-Za-z0-9_]*)
   |(?P<op><<|>>|[-+*/%&|^~()])
   |(?P<dot>\.)
   )""", re.X)
_ESCAPES = {"n": 10, "t": 9, "r": 13, "0": 0, "\\": 92, "'": 39, '"': 34}


def f32_bits(text: str) -> int:
    """Bits of a binary32 float literal (round to nearest even), e.g. '1.5', '-inf', 'nan'."""
    try:
        v = float(text.strip())
    except ValueError:
        raise ExprError(f"invalid float literal '{text.strip()}'") from None
    try:
        return struct.unpack("<I", struct.pack("<f", v))[0]
    except OverflowError:
        raise ExprError(f"float literal '{text.strip()}' overflows binary32") from None


def _tokenize(text: str):
    text = _F32.sub(lambda m: str(f32_bits(m[1])), text)
    pos, out = 0, []
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ExprError(f"unexpected character '{text[pos:].strip()[0]}' in expression")
        pos = m.end()
        kind = m.lastgroup
        tok = m[kind]
        if kind == "num":
            out.append(("num", int(tok.replace("_", ""), 0)))
        elif kind == "char":
            body = tok[1:-1]
            if body.startswith("\\"):
                if body[1] not in _ESCAPES:
                    raise ExprError(f"unknown escape '{body}'")
                out.append(("num", _ESCAPES[body[1]]))
            else:
                out.append(("num", ord(body)))
        else:
            out.append((kind, tok))
    if not out:
        raise ExprError("empty expression")
    return out


@dataclass
class Node:
    kind: str                  # num, sym, dot, unary, binary, func
    value: object = None
    args: tuple = ()


class _Parser:
    BINARY = {"|": 1, "^": 2, "&": 3, "<<": 4, ">>": 4, "+": 5, "-": 5, "*": 6, "/": 6, "%": 6}

    def __init__(self, tokens):
        self.toks, self.i = tokens, 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def expect(self, op):
        if self.take() != ("op", op):
            raise ExprError(f"expected '{op}'")

    def parse(self, min_prec=0):
        left = self.unary()
        while True:
            kind, tok = self.peek()
            prec = self.BINARY.get(tok) if kind == "op" else None
            if prec is None or prec < min_prec:
                return left
            self.take()
            right = self.parse(prec + 1)
            left = Node("binary", tok, (left, right))

    def unary(self):
        kind, tok = self.take()
        if kind == "op" and tok in "-+~":
            return Node("unary", tok, (self.unary(),))
        if kind == "op" and tok == "(":
            e = self.parse()
            self.expect(")")
            return e
        if kind == "num":
            return Node("num", tok)
        if kind == "id":
            return Node("sym", tok)
        if kind == "dot":
            return Node("dot")
        if kind == "func":
            self.expect("(")
            e = self.parse()
            self.expect(")")
            return Node("func", tok.lower(), (e,))
        raise ExprError("expected a value" if kind is None else f"unexpected '{tok}'")


def parse(text: str) -> Node:
    p = _Parser(_tokenize(text))
    node = p.parse()
    if p.i != len(p.toks):
        raise ExprError(f"unexpected '{p.peek()[1]}' after expression")
    return node


def symbols_in(node: Node) -> set:
    if node.kind == "sym":
        return {node.value}
    out = set()
    for a in node.args:
        out |= symbols_in(a)
    return out


def _cdiv(a, b):
    if b == 0:
        raise ExprError("division by zero")
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def evaluate(node: Node, lookup, dot: int | None) -> tuple[int, bool]:
    """lookup(name) -> (value, is_address); dot is the current address (None if unavailable)."""
    k = node.kind
    if k == "num":
        return node.value, False
    if k == "sym":
        return lookup(node.value)
    if k == "dot":
        if dot is None:
            raise ExprError("'.' is not available here")
        return dot, True
    if k == "unary":
        v, _ = evaluate(node.args[0], lookup, dot)
        return {"-": -v, "+": v, "~": ~v}[node.value], False
    if k == "func":
        v, _ = evaluate(node.args[0], lookup, dot)
        v &= 0xFFFFFFFF
        return (v >> 16 if node.value == "%hi" else v & 0xFFFF), False
    (a, aa), (b, ba) = (evaluate(x, lookup, dot) for x in node.args)
    op = node.value
    if op == "+":
        return a + b, aa or ba
    if op == "-":
        return a - b, aa != ba          # address - address is a plain distance
    if op == "*":
        return a * b, False
    if op == "/":
        return _cdiv(a, b), False
    if op == "%":
        return a - _cdiv(a, b) * b, False
    if op in ("<<", ">>"):
        if b < 0 or b > 64:
            raise ExprError(f"shift amount {b} out of range")
        return (a << b if op == "<<" else a >> b), False
    return {"&": a & b, "|": a | b, "^": a ^ b}[op], False
