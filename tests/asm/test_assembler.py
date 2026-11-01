"""The assembler: syntax, expressions, directives, pseudo-instructions, errors, outputs."""

import json
import struct

import pytest

from mgpu.asm import AsmError, assemble, hex_text, listing_text, symbol_map
from mgpu.asm.cli import main as as_main
from mgpu.isa import assemble_one, decode
from mgpu.isa.tables import VECTORS


def words(src, base=0x1000):
    return assemble(src, base).words


def errors(src):
    with pytest.raises(AsmError) as e:
        assemble(src, filename="t.s")
    return [d.message for d in e.value.diagnostics]


@pytest.mark.parametrize("vec", VECTORS, ids=lambda v: v["asm"])
def test_conformance_vectors(vec):
    assert words(vec["asm"]) == [vec["hex"]]


def test_case_insensitive_and_aliases():
    assert words("v_add V3, v2, S3") == [0x08621C00]
    assert words("S_MOV s1, EXEC") == words("S_ADD s1, s31, s0")
    assert words("v_getsr v1, gtid") == [assemble_one("V_GETSR v1, 18")]


def test_comments_labels_and_blank_lines():
    src = """
    ; a comment
    a: b:   NOP        ; trailing comment with ; inside
    c:
            EXIT
    """
    p = assemble(src)
    assert p.words == [0, 1]
    assert p.symbols["a"] == p.symbols["b"] == 0x1000 and p.symbols["c"] == 0x1004


@pytest.mark.parametrize("expr, value", [
    ("1+2*3", 7), ("(1+2)*3", 9), ("1 << 4 | 1", 17), ("-7/2", -3), ("-7%2", -1),
    ("~0 & 0xFF", 255), ("0b1010_0101", 0xA5), ("'A'", 65), ("'\\n'", 10),
    ("%hi(0x12345678)", 0x1234), ("%lo(0x12345678)", 0x5678), ("f32(1.0)", 0x3F800000),
    ("f32(-2.5e1)", 0xC1C80000), ("8 >> 1 + 1", 2), ("1 - 2 - 3", -4), ("6 ^ 3", 5),
])
def test_expressions(expr, value):
    assert words(f".word {expr}") == [value & 0xFFFFFFFF]


def test_symbols_and_constants():
    p = assemble("""
    SIZE = 16
            .equ COUNT, SIZE / 4
            .set MASK, (1 << COUNT) - 1
    start:  S_ADDI s1, s0, COUNT
            S_ANDI s2, s1, MASK
    after:  .word after - start, SIZE * 2, start
    """)
    assert p.words[:2] == [assemble_one("S_ADDI s1, s0, 4"), assemble_one("S_ANDI s2, s1, 15")]
    assert p.words[2:] == [8, 32, 0x1000]
    assert p.symbols["MASK"] == 15 and "MASK" not in p.labels


def test_forward_references():
    p = assemble("S_LI s1, data\nS_J done\nNOP\ndone: EXIT\ndata: .word data")
    assert p.words[0] == assemble_one(f"S_LUI s1, 0")
    assert p.words[1] == assemble_one(f"S_ORI s1, s1, {0x1014}")
    assert decode(p.words[2]).soffset26 == 1
    assert p.words[-1] == 0x1014


def test_branch_operand_rule():
    # labels and '.' are targets; plain numbers (and constants) are raw offsets
    p = assemble("""
    K = -5
    top:    S_BNZ s12, -5
            S_BNZ s12, K
            S_BNZ s12, top
            S_J   .
            S_J   . + 8
    """)
    offs = [decode(w).simm16 if i < 3 else decode(w).soffset26 for i, w in enumerate(p.words)]
    assert offs == [-5, -5, -3, -1, 1]


def test_branch_errors():
    msgs = errors("S_BZ s1, 40000\nS_J . + 2\nfar: .space 0\n")
    assert "branch offset out of range" in msgs[0]
    assert "not 4-byte aligned" in msgs[1]
    msgs = errors("S_BZ s1, far\n.space 0x40000\nfar: EXIT")
    assert "branch target out of range" in msgs[0]


def test_directives_layout():
    p = assemble("""
            NOP
            .align 16
    a:      .word 7
            .space 8, 0xFF
            .org 0x1020
    b:      .float 1.5, -0.0, inf, nan
            .space 3
            .align 4
    c:      EXIT
    """)
    assert (p.symbols["a"], p.symbols["b"], p.symbols["c"]) == (0x1010, 0x1020, 0x1034)
    w = p.words
    assert w[4] == 7 and w[5] == w[6] == 0xFFFFFFFF
    assert w[8:12] == [0x3FC00000, 0x80000000, 0x7F800000, 0x7FC00000]
    assert w[-1] == 1 and p.end == 0x1038


def test_kernels_and_entry():
    p = assemble(".kernel k1\n.kernel k2\nk1: NOP\nk2: EXIT")
    assert p.kernels == {"k1": 0x1000, "k2": 0x1004}
    assert p.entry("k2") == 0x1004
    with pytest.raises(KeyError):
        p.entry()
    assert assemble(".kernel only\nonly: EXIT").entry() == 0x1000
    assert assemble("NOP").entry() == 0x1000


@pytest.mark.parametrize("pseudo, expansion", [
    ("S_MOV s1, s2", ["S_ADD s1, s2, s0"]),
    ("S_LI s1, 0x12345678", ["S_LUI s1, 0x1234", "S_ORI s1, s1, 0x5678"]),
    ("S_LI s1, -1", ["S_LUI s1, 0xFFFF", "S_ORI s1, s1, 0xFFFF"]),
    ("S_LI s1, 5", ["S_LUI s1, 0", "S_ORI s1, s1, 5"]),
    ("V_MOV v1, v2", ["V_ADD v1, v2, v0"]),
    ("V_BCAST v1, s2", ["V_ADD v1, v0, s2"]),
    ("V_NOT v1, v1", ["V_SUB v1, v0, v1", "V_ADDI v1, v1, -1"]),
    ("V_FNEG v1, v2, s9", ["S_LUI s9, 0x8000", "V_XOR v1, v2, s9"]),
    ("V_FABS v1, v2, s9", ["S_LUI s9, 0x8000", "V_ANDN v1, v2, s9"]),
    ("V_FDIV v1, v2, v3, v1", ["V_FRCP v1, v3", "V_FMUL v1, v2, v1"]),
    ("S_MOVEXEC s4", ["S_ADD s4, s31, s0"]),
])
def test_pseudo_expansions(pseudo, expansion):
    assert words(pseudo) == [assemble_one(x) for x in expansion]


@pytest.mark.parametrize("src, msg", [
    ("V_FOO v1", "unknown mnemonic 'V_FOO'"),
    ("S_ADD s1, s2", "S_ADD expects 3 operand(s)"),
    ("V_ADD v1, s2, v3", "must be a vector register"),
    ("S_ADD s1, v2, s3", "must be a scalar register"),
    ("V_ADD v1, v2, x3", "must be a vector or scalar register"),
    ("V_ADD v32, v2, v3", "must be a vector register"),
    ("S_ADDI s1, s2, 0x8000", "signed immediate 32768 out of range"),
    ("S_ANDI s1, s2, -1", "unsigned immediate -1 out of range"),
    ("S_SHLI s1, s2, 32", "shift amount 32 out of range"),
    ("V_LW v1, 0x8000(v2)", "memory offset 32768 out of range"),
    ("V_LW v1, 0(s2)", "base register must be a vector register"),
    ("V_LW v1, v2", "expected a memory operand"),
    ("S_GETSR s1, LANE_ID", "per-lane"),
    ("V_GETSR v1, 7", "special register 7 is reserved"),
    ("S_LI s1, 0x100000000", "32-bit value"),
    ("V_BCAST v1, v2", "must be a scalar register"),
    ("V_FDIV v1, v2, v3, v2", "vt must differ from va"),
    (".word undefined_thing", "undefined symbol 'undefined_thing'"),
    ("a = b\nb = a\n.word a", "circular definition"),
    ("x: NOP\nx: NOP", "symbol 'x' is already defined"),
    ("v3 = 1", "register name"),
    (".bogus 1", "unknown directive '.bogus'"),
    (".org 0x800", "would move backwards"),
    (".align 3", "not a power of two"),
    (".org later\nlater: NOP", "layout directives may only use symbols defined above"),
    (".space 2\nNOP", "unaligned address"),
    (".kernel nothing", "does not name a label"),
    (".word 1 +", "expected a value"),
    (".float abc", "invalid float literal"),
    (".float 1e40", "overflows binary32"),
    (".word 5/0", "division by zero"),
])
def test_errors(src, msg):
    assert any(msg in m for m in errors(src)), errors(src)


def test_all_errors_reported_in_line_order():
    msgs = errors("S_ADDI s1, s2, 99999\nV_FOO\nS_BZ s1, nope\nx: NOP\nx: NOP")
    assert len(msgs) == 4
    assert "immediate" in msgs[0] and "V_FOO" in msgs[1] and "nope" in msgs[2] and "'x'" in msgs[3]


def test_include(tmp_path):
    (tmp_path / "lib.s").write_text("helper: NOP\nCONST = 7\n")
    (tmp_path / "main.s").write_text('.include "lib.s"\nmain: S_ADDI s1, s0, CONST\nS_J helper\n')
    from mgpu.asm import assemble_file
    p = assemble_file(tmp_path / "main.s")
    assert p.symbols["helper"] == 0x1000 and p.symbols["main"] == 0x1004
    (tmp_path / "loop.s").write_text('.include "loop.s"\n')
    with pytest.raises(AsmError, match="recursive"):
        assemble_file(tmp_path / "loop.s")
    (tmp_path / "missing.s").write_text('.include "nope.s"\n')
    with pytest.raises(AsmError, match="cannot read"):
        assemble_file(tmp_path / "missing.s")


def test_outputs():
    p = assemble(".kernel k\nk: NOP\nN = 3\nEXIT\ndata: .word 0xCAFE", base=0x2000)
    assert hex_text(p).splitlines() == ["@00000800", "00000000", "00000001", "0000cafe"]
    sm = symbol_map(p)
    assert sm["base"] == 0x2000 and sm["size"] == 12 and sm["kernels"] == {"k": 0x2000}
    assert sm["labels"] == {"k": 0x2000, "data": 0x2008} and sm["constants"] == {"N": 3}
    lst = listing_text(p)
    assert "00002000  00000000  k: NOP" in lst and "00002008  0000cafe" in lst
    assert p.image == struct.pack("<3I", 0, 1, 0xCAFE)


def test_cli(tmp_path, capsys):
    src = tmp_path / "k.s"
    src.write_text(".kernel k\nk: S_LI s1, 0xDEADBEEF\nEXIT\n")
    rc = as_main([str(src), "--hex", str(tmp_path / "k.hex"), "--sym", str(tmp_path / "k.json"),
                  "--lst", str(tmp_path / "k.lst"), "--base", "0x4000"])
    assert rc == 0
    assert (tmp_path / "k.bin").read_bytes() == assemble(src.read_text(), 0x4000).image
    assert json.loads((tmp_path / "k.json").read_text())["kernels"] == {"k": 0x4000}
    assert (tmp_path / "k.hex").read_text().startswith("@00001000\n")
    bad = tmp_path / "bad.s"
    bad.write_text("NOP\nV_FOO\n")
    assert as_main([str(bad)]) == 1
    assert "bad.s:2: error: unknown mnemonic" in capsys.readouterr().err
