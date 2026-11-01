"""The MGPU-32 assembler (docs/toolchain/assembler.md).

Two passes over a fixed-size instruction stream: pass 1 assigns addresses to every
statement and label (every instruction and pseudo-instruction has a fixed length), pass 2
evaluates operands and encodes. Operand syntax comes from spec/isa.toml (via mgpu.isa),
and pseudo-instructions expand through its templates.
"""

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..isa import BY_MNEMONIC, SR_BY_NAME, encode
from ..isa.tables import PSEUDOS
from .expr import ExprError, Undefined, evaluate, f32_bits, parse

MAX_ERRORS = 20
PSEUDO = {p["mnemonic"]: p for p in PSEUDOS}
DIRECTIVES = {".org", ".align", ".word", ".float", ".space", ".equ", ".set", ".kernel", ".include"}
_REG = re.compile(r"^([sv])(\d+)$")
_LABEL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:(?!=)")
_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_INCLUDE = re.compile(r'^\s*\.include\s+"([^"]+)"\s*$', re.IGNORECASE)


# ---- results and diagnostics --------------------------------------------------------------
@dataclass
class Diagnostic:
    file: str
    line: int
    message: str
    text: str = ""

    def __str__(self) -> str:
        s = f"{self.file}:{self.line}: error: {self.message}"
        return s + (f"\n    {self.text.strip()}" if self.text.strip() else "")


class AsmError(Exception):
    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = diagnostics
        super().__init__("\n".join(map(str, diagnostics)))


@dataclass
class ListingEntry:
    addr: int
    words: list
    file: str
    line: int
    text: str


@dataclass
class Program:
    base: int
    image: bytes
    symbols: dict = field(default_factory=dict)     # name -> value (labels and constants)
    labels: set = field(default_factory=set)        # which symbols are labels
    kernels: dict = field(default_factory=dict)     # .kernel name -> address
    listing: list = field(default_factory=list)

    @property
    def words(self) -> list[int]:
        b = self.image
        return [int.from_bytes(b[i:i + 4], "little") for i in range(0, len(b), 4)]

    @property
    def end(self) -> int:
        return self.base + len(self.image)

    def entry(self, kernel: str | None = None) -> int:
        """Address of `kernel`, or of the only kernel, or the base if none is declared."""
        if kernel is not None:
            if kernel in self.kernels:
                return self.kernels[kernel]
            if kernel in self.labels:
                return self.symbols[kernel]
            raise KeyError(f"no kernel or label named '{kernel}'")
        if len(self.kernels) == 1:
            return next(iter(self.kernels.values()))
        if not self.kernels:
            return self.base
        raise KeyError(f"several kernels ({', '.join(self.kernels)}); name one")


# ---- source handling ------------------------------------------------------------------------
@dataclass
class Stmt:
    file: str
    line: int
    text: str
    labels: list
    op: str | None = None          # upper-case mnemonic or lower-case directive
    args: list = field(default_factory=list)
    assign: str | None = None      # for `name = expr`
    addr: int = 0
    size: int = 0


def strip_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == "\\":
                continue
            if ch == quote and line[i - 1] != "\\":
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == ";":
            return line[:i]
    return line


def split_operands(text: str) -> list[str]:
    """Split on top-level commas (not inside parentheses or quotes)."""
    out, depth, cur, quote = [], 0, "", None
    for ch in text:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip() or out:
        out.append(cur.strip())
    return out


def is_reserved_name(name: str) -> bool:
    """Register names cannot be symbols: `s5` in an expression would be ambiguous."""
    low = name.lower()
    return low == "exec" or bool(_REG.match(low))


class Assembler:
    def __init__(self, base: int):
        if base % 4 or not 0 <= base < 1 << 32:
            raise ValueError("base must be a 4-aligned 32-bit address")
        self.base = base
        self.diags: list[Diagnostic] = []
        self.labels: dict[str, int] = {}
        self.equs: dict[str, tuple] = {}          # name -> (Node, Stmt)
        self.equ_values: dict[str, tuple] = {}
        self.evaluating: set = set()
        self.kernel_decls: list[tuple] = []        # (name, Stmt)

    # ---- diagnostics
    def error(self, st, msg):
        if len(self.diags) < MAX_ERRORS:
            self.diags.append(Diagnostic(st.file, st.line, msg, st.text))
        elif len(self.diags) == MAX_ERRORS:
            self.diags.append(Diagnostic(st.file, st.line, "too many errors; stopping"))

    # ---- reading
    def read(self, source: str, filename: str, stack: tuple = ()) -> list[Stmt]:
        stmts = []
        for n, raw in enumerate(source.splitlines(), 1):
            line = strip_comment(raw)
            st = Stmt(filename, n, raw, [])
            m = _INCLUDE.match(line)
            if m:
                stmts += self.include(m[1], st, stack + (filename,))
                continue
            while (lm := _LABEL.match(line)):
                st.labels.append(lm[1])
                line = line[lm.end():]
            am = _ASSIGN.match(line)
            if am:
                st.assign, st.args = am[1], [am[2].strip()]
            elif line.strip():
                op, _, rest = line.strip().partition(" ")
                op = op.strip()
                st.op = op.lower() if op.startswith(".") else op.upper()
                st.args = split_operands(rest.strip()) if rest.strip() else []
            stmts.append(st)
        return stmts

    def include(self, name: str, st: Stmt, stack: tuple) -> list[Stmt]:
        here = Path(stack[-1]).parent if not stack[-1].startswith("<") else Path.cwd()
        path = (here / name).resolve()
        if str(path) in stack:
            self.error(st, f"recursive .include of '{name}'")
            return []
        try:
            text = path.read_text()
        except OSError as e:
            self.error(st, f"cannot read '{name}': {e.strerror}")
            return []
        return self.read(text, str(path), stack)

    # ---- symbols
    def lookup(self, name: str) -> tuple[int, bool]:
        if name in self.labels:
            return self.labels[name], True
        if name in self.equs:
            if name in self.equ_values:
                return self.equ_values[name]
            if name in self.evaluating:
                raise ExprError(f"circular definition of '{name}'")
            self.evaluating.add(name)
            try:
                node, st = self.equs[name]
                val = evaluate(node, self.lookup, st.addr)
            finally:
                self.evaluating.discard(name)
            self.equ_values[name] = val
            return val
        raise Undefined(name)

    def define(self, st, name, kind, value):
        if is_reserved_name(name):
            self.error(st, f"'{name}' is a register name and cannot be a symbol")
        elif name in self.labels or name in self.equs:
            self.error(st, f"symbol '{name}' is already defined")
        elif kind == "label":
            self.labels[name] = value
        else:
            self.equs[name] = value

    def value(self, text: str, dot: int | None) -> tuple[int, bool]:
        return evaluate(parse(text), self.lookup, dot)

    def int_value(self, text, dot, lo, hi, what) -> int:
        v, _ = self.value(text, dot)
        if not lo <= v <= hi:
            raise ExprError(f"{what} {v} out of range [{lo}, {hi}]")
        return v

    # ---- pass 1: layout
    def size_of(self, st: Stmt, loc: int) -> int:
        op = st.op
        if op in BY_MNEMONIC:
            return 4
        if op in PSEUDO:
            return 4 * len(PSEUDO[op]["expansion"])
        if op in (".word", ".float"):
            if not st.args:
                raise ExprError(f"{op} needs at least one value")
            return 4 * len(st.args)
        if op == ".space":
            if len(st.args) not in (1, 2):
                raise ExprError(".space takes a size and an optional fill byte")
            n = self.int_value(st.args[0], loc, 0, 1 << 32, ".space size")
            return n
        if op == ".org":
            target = self.int_value(st.args[0], loc, 0, (1 << 32) - 1, ".org address") if \
                len(st.args) == 1 else None
            if target is None:
                raise ExprError(".org takes one address")
            if target < loc:
                raise ExprError(f".org {target:#x} would move backwards from {loc:#x}")
            return target - loc
        if op == ".align":
            if len(st.args) != 1:
                raise ExprError(".align takes one byte alignment")
            n = self.int_value(st.args[0], loc, 1, 1 << 16, ".align")
            if n & (n - 1):
                raise ExprError(f".align {n} is not a power of two")
            return -loc % n
        raise AssertionError(op)

    def pass1(self, stmts: list[Stmt]) -> int:
        loc = self.base
        for st in stmts:
            st.addr = loc
            for lab in st.labels:
                self.define(st, lab, "label", loc)
            if st.assign:
                try:
                    self.define(st, st.assign, "equ", (parse(st.args[0]), st))
                except ExprError as e:
                    self.error(st, str(e))
                continue
            op = st.op
            if op is None:
                continue
            try:
                if op in (".equ", ".set"):
                    if len(st.args) != 2:
                        raise ExprError(f"{op} takes a name and a value")
                    self.define(st, st.args[0], "equ", (parse(st.args[1]), st))
                elif op == ".kernel":
                    if len(st.args) != 1:
                        raise ExprError(".kernel takes one label name")
                    self.kernel_decls.append((st.args[0], st))
                elif op == ".include":
                    raise ExprError('.include expects a quoted file name: .include "file.s"')
                elif op.startswith(".") and op not in DIRECTIVES:
                    raise ExprError(f"unknown directive '{op}'")
                elif not op.startswith(".") and op not in BY_MNEMONIC and op not in PSEUDO:
                    close = difflib.get_close_matches(op, list(BY_MNEMONIC) + list(PSEUDO), 1)
                    hint = f" (did you mean {close[0]}?)" if close else ""
                    raise ExprError(f"unknown mnemonic '{op}'{hint}")
                else:
                    if (op in BY_MNEMONIC or op in PSEUDO or op in (".word", ".float")) and loc % 4:
                        raise ExprError(f"{op} at unaligned address {loc:#x}; use .align 4")
                    st.size = self.size_of(st, loc)
                    loc += st.size
                    if loc > 1 << 32:
                        raise ExprError("program extends beyond the 32-bit address space")
            except Undefined as e:
                self.error(st, f"{e} (layout directives may only use symbols defined above)")
            except ExprError as e:
                self.error(st, str(e))
        return loc

    # ---- pass 2: encoding
    def reg(self, text: str, kinds: str, what: str) -> tuple[str, int]:
        t = text.strip().lower()
        if t == "exec":
            t = "s31"
        m = _REG.match(t)
        if not m or int(m[2]) > 31 or m[1] not in kinds:
            want = {"s": "a scalar register", "v": "a vector register",
                    "vs": "a vector or scalar register"}[kinds]
            raise ExprError(f"{what} must be {want}, got '{text.strip()}'")
        return m[1], int(m[2])

    def target(self, text: str, pc: int, bits: int) -> int:
        v, is_addr = self.value(text, pc)
        if is_addr:
            if v & 3:
                raise ExprError(f"branch target {v:#x} is not 4-byte aligned")
            v = (v - (pc + 4)) >> 2
            what = "branch target out of range"
        else:
            what = "branch offset out of range"
        lim = 1 << (bits - 1)
        if not -lim <= v < lim:
            raise ExprError(f"{what}: {v} instructions (limit +-{lim})")
        return v

    def encode_ins(self, mn: str, args: list[str], pc: int) -> int:
        ins = BY_MNEMONIC[mn]
        if len(args) != len(ins.syntax):
            raise ExprError(f"{mn} expects {len(ins.syntax)} operand(s): "
                            f"{mn} {', '.join(ins.syntax)}".rstrip(": "))
        f = dict(rd=0, ra=0, rb=0, rc=0, sb=0, imm=0, offset=0)
        field_of = {"d": "rd", "t": "rd", "a": "ra", "b": "rb", "c": "rc"}
        for tok, arg in zip(ins.syntax, args):
            if tok == "vb|sb":
                k, n = self.reg(arg, "vs", f"{mn}: operand '{tok}'")
                f["rb"], f["sb"] = n, int(k == "s")
            elif tok == "shamt":
                f["rb"] = self.int_value(arg, pc, 0, 31, "shift amount")
            elif tok == "simm16":
                f["imm"] = self.int_value(arg, pc, -0x8000, 0x7FFF, "signed immediate")
            elif tok == "uimm16":
                f["imm"] = self.int_value(arg, pc, 0, 0xFFFF, "unsigned immediate")
            elif tok == "sr":
                f["ra"] = self.special_register(mn, arg, pc)
            elif tok == "target16":
                f["imm"] = self.target(arg, pc, 16)
            elif tok == "target26":
                f["offset"] = self.target(arg, pc, 26)
            elif "(" in tok:
                off, base = self.memory_operand(arg)
                f["imm"] = self.int_value(off, pc, -0x8000, 0x7FFF, "memory offset") if off else 0
                f["ra"] = self.reg(base, tok[-3], f"{mn}: base register")[1]
            else:
                f[field_of[tok[-1]]] = self.reg(arg, tok[0], f"{mn}: operand '{tok}'")[1]
        return encode(mn, **f)

    def special_register(self, mn: str, arg: str, pc: int) -> int:
        from ..isa import SR_BY_ID
        name = arg.strip().upper()
        sr_id = SR_BY_NAME[name]["id"] if name in SR_BY_NAME else \
            self.int_value(arg, pc, 0, 31, "special register")
        sr = SR_BY_ID.get(sr_id)
        if sr is None:
            raise ExprError(f"special register {sr_id} is reserved")
        if mn == "S_GETSR" and sr["kind"] == "lane":
            raise ExprError(f"{sr['name']} is per-lane; read it with V_GETSR")
        return sr_id

    @staticmethod
    def memory_operand(arg: str) -> tuple[str, str]:
        a = arg.strip()
        if not a.endswith(")"):
            raise ExprError(f"expected a memory operand 'offset(register)', got '{arg}'")
        depth = 0
        for i in range(len(a) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(a[i], 0)
            if depth == 0:
                return a[:i].strip(), a[i + 1:-1]
        raise ExprError(f"unbalanced parentheses in '{arg}'")

    def expand_pseudo(self, mn: str, args: list[str], pc: int) -> list[int]:
        p = PSEUDO[mn]
        toks = [t.strip() for t in p["syntax"].split(",")]
        if len(args) != len(toks):
            raise ExprError(f"{mn} expects {len(toks)} operand(s): {mn} {p['syntax']}")
        subst = {}
        for tok, arg in zip(toks, args):
            if tok == "imm32":
                self.int_value(arg, pc, -(1 << 31), (1 << 32) - 1, f"{mn}: 32-bit value")
            else:
                self.reg(arg, tok[0], f"{mn}: operand '{tok}'")
            subst[tok] = arg
        if mn == "V_FDIV" and self.reg(subst["vt"], "v", "")[1] == self.reg(subst["va"], "v", "")[1]:
            raise ExprError("V_FDIV: the scratch register vt must differ from va")
        words = []
        for k, line in enumerate(p["expansion"]):
            text = re.sub(r"\$\{(\w+)\}", lambda m: subst[m[1]], line)
            emn, _, rest = text.partition(" ")
            try:
                words.append(self.encode_ins(emn, split_operands(rest), pc + 4 * k))
            except ExprError as e:
                raise ExprError(f"{e} (in expansion of {mn})") from None
        return words

    def pass2(self, stmts: list[Stmt], end: int) -> tuple[bytearray, list]:
        image = bytearray(end - self.base)
        listing = []
        for st in stmts:
            words = []
            try:
                op, pc = st.op, st.addr
                if op in BY_MNEMONIC:
                    words = [self.encode_ins(op, st.args, pc)]
                elif op in PSEUDO:
                    words = self.expand_pseudo(op, st.args, pc)
                elif op == ".word":
                    words = [self.int_value(a, pc + 4 * i, -(1 << 31), (1 << 32) - 1, ".word value")
                             & 0xFFFFFFFF for i, a in enumerate(st.args)]
                elif op == ".float":
                    words = [f32_bits(a) for a in st.args]
                elif op == ".space" and st.size:
                    fill = self.int_value(st.args[1], pc, 0, 255, "fill byte") if len(st.args) > 1 else 0
                    image[pc - self.base:pc - self.base + st.size] = bytes([fill]) * st.size
                elif st.assign or op in (".equ", ".set"):
                    name = st.assign or st.args[0]
                    if name in self.equs and self.equs[name][1] is st:
                        self.lookup(name)                     # report errors in definitions
            except ExprError as e:
                self.error(st, str(e))
                words = [0] * (st.size // 4)
            for i, w in enumerate(words):
                o = st.addr - self.base + 4 * i
                image[o:o + 4] = w.to_bytes(4, "little")
            listing.append(ListingEntry(st.addr, words, st.file, st.line, st.text))
        return image, listing


def assemble(source: str, base: int = 0x1000, filename: str = "<input>") -> Program:
    """Assemble MGPU-32 source text. Raises AsmError listing every problem found."""
    asm = Assembler(base)
    stmts = asm.read(source, filename)
    end = asm.pass1(stmts)
    end += -end % 4
    kernels = {}
    for name, st in asm.kernel_decls:
        if name not in asm.labels:
            asm.error(st, f".kernel '{name}' does not name a label")
        elif asm.labels[name] % 4:
            asm.error(st, f"kernel '{name}' is not 4-byte aligned")
        else:
            kernels[name] = asm.labels[name]
    image, listing = asm.pass2(stmts, end)
    if asm.diags:
        order = {f: i for i, f in enumerate(dict.fromkeys(st.file for st in stmts))}
        raise AsmError(sorted(asm.diags, key=lambda d: (order.get(d.file, 0), d.line)))
    symbols = dict(asm.labels)
    for name in asm.equs:
        symbols[name] = asm.lookup(name)[0]
    return Program(base, bytes(image), symbols, set(asm.labels), kernels, listing)


def assemble_file(path, base: int = 0x1000) -> Program:
    p = Path(path)
    return assemble(p.read_text(), base, str(p))
