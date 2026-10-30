"""Checks on the normative ISA artifacts: spec/isa.toml, generated tables, decode policy."""

import subprocess
import sys

import pytest

from conftest import ROOT
from mgpu.isa import BY_MNEMONIC, IllegalInstruction, assemble_one, decode, encode
from mgpu.isa.tables import VECTORS


def test_generated_files_up_to_date():
    r = subprocess.run([sys.executable, ROOT / "tools/gen_isa.py", "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("vec", VECTORS, ids=lambda v: v["asm"])
def test_worked_encodings(vec):
    assert assemble_one(vec["asm"]) == vec["hex"]


def test_counts():
    assert len({i.opcode for i in BY_MNEMONIC.values()}) == 24
    assert len(BY_MNEMONIC) == 101


@pytest.mark.parametrize("op", [0x0E, 0x0F, *range(0x17, 0x20), *range(0x23, 0x40)])
def test_reserved_opcodes_illegal(op):
    with pytest.raises(IllegalInstruction):
        decode(op << 26)


@pytest.mark.parametrize("word", [
    0x02 << 26 | 0x11,              # V_POPC  (DEC-1, C-04)
    0x02 << 26 | 0x12,              # V_FF1
    0x01 << 26 | 0x13,              # S_ALU funct 0x13
    0x01 << 26 | 0x200,             # S_ALU funct with a high bit set
    0x03 << 26 | 0x0B,              # V_FPU funct 0x0B
    0x04 << 26 | 0x10,              # V_CMP funct 0x10
    0x05 << 26 | 0x03,              # V_R4 funct 3
    0x06 << 26 | 0x02,              # V_LANE funct 2
    0x07 << 26 | 0x02,              # GETSR funct 2
    0x00 << 26 | 0x04,              # SYS funct 4
    0x16 << 26 | 0x06,              # V_ATOM funct 6
    0x16 << 26 | 0x08,              # V_ATOM funct with bit 3 set
    0x16 << 26 | 0x30,              # V_ATOM funct with bit 5 set
])
def test_unassigned_funct_illegal(word):
    with pytest.raises(IllegalInstruction):
        decode(word)


def test_unused_fields_ignored():
    # DEC-2: sb in a scalar op, ra in S_LUI, ra in S_BNZ, fields of EXIT, rb/sb in unary FPU ops.
    assert decode(encode("S_ADD", 1, 2, 3) | 1 << 10).instr.mnemonic == "S_ADD"
    assert decode(encode("S_LUI", 1, imm=5) | 7 << 16).instr.mnemonic == "S_LUI"
    assert decode(0x8580FFFB | 9 << 16).rd == 12
    assert decode(encode("EXIT") | 0x03FFFC00).instr.mnemonic == "EXIT"   # rd, ra, rb, sb all ones
    assert decode(encode("V_FRCP", 1, 2) | 5 << 11 | 1 << 10).instr.mnemonic == "V_FRCP"


def test_sb_positions():
    assert assemble_one("V_ADD v1, v2, s3") >> 10 & 1 == 1          # R: bit 10
    assert assemble_one("V_FFMA v1, v2, s3, v4") >> 5 & 1 == 1      # R4: bit 5
    assert assemble_one("V_FFMA v1, v2, v3, v4") >> 5 & 1 == 0


def test_every_instruction_roundtrips():
    for mn, ins in BY_MNEMONIC.items():
        d = decode(encode(mn, rd=1, ra=2, rb=3, rc=4, sb=0, imm=0x1234, offset=0x123456))
        assert d.instr is ins
