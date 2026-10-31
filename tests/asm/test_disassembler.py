"""Disassembler: exact assemble(disassemble(image)) round trips, symbols, CLI."""

import json
import random

import pytest

from mgpu.asm import assemble, hex_text, symbol_map
from mgpu.asm.dis_cli import main as dis_main, read_hex
from mgpu.isa import BY_MNEMONIC, IllegalInstruction, canonical, decode, disasm, disassemble, encode

BASE = 0x1000


def roundtrip(words, symbols=None):
    text = disassemble(words, BASE, symbols)
    return assemble(text, BASE).words


@pytest.mark.parametrize("seed", range(5))
def test_random_words_roundtrip_exactly(seed):
    rng = random.Random(seed)
    words = [rng.getrandbits(32) for _ in range(4000)]
    assert roundtrip(words) == words


def test_every_instruction_canonical_roundtrip():
    rng = random.Random(1)
    for mn in BY_MNEMONIC:
        for _ in range(200):
            w = encode(mn, rd=rng.randrange(32), ra=rng.randrange(32), rb=rng.randrange(32),
                       rc=rng.randrange(32), sb=rng.getrandbits(1), imm=rng.getrandbits(16),
                       offset=rng.getrandbits(26))
            d = decode(w)
            c = canonical(d)
            text = disasm(c)
            if mn in ("S_GETSR", "V_GETSR") and text.startswith(".word"):
                continue  # reserved / per-lane SR ids are illegal and stay .word
            assert not text.startswith(".word"), (mn, hex(c), text)
            assert assemble(text).words == [c], (mn, text)


def test_noncanonical_words_decode_like_their_canonical_form():
    rng = random.Random(2)
    for _ in range(20000):
        w = rng.getrandbits(32)
        try:
            d = decode(w)
        except IllegalInstruction:
            continue
        c = decode(canonical(d))
        assert c.instr is d.instr
        if canonical(d) != w:
            assert "non-canonical" in disasm(w)


def test_fuzz_programs_roundtrip():
    from mgpu.fuzz import random_case
    for seed in range(40):
        case = random_case(seed, "diff")
        assert roundtrip(case.words) == case.words


def test_symbols_become_labels():
    p = assemble(".kernel main\nmain: S_ADDI s1, s0, 3\nloop: S_ADDI s1, s1, -1\nS_BNZ s1, loop\nEXIT")
    syms = {k: v for k, v in p.symbols.items() if k in p.labels}
    text = disassemble(p.words, BASE, syms)
    assert "loop:" in text and "S_BNZ s1, loop" in text
    assert roundtrip(p.words, syms) == p.words


def test_cli_bin_and_hex(tmp_path, capsys):
    p = assemble(".kernel k\nk: S_LI s1, 0xCAFE\nback: S_BZ s0, back\nEXIT\n.word 0xFFFFFFFF", 0x2000)
    (tmp_path / "k.bin").write_bytes(p.image)
    (tmp_path / "k.hex").write_text(hex_text(p))
    (tmp_path / "k.json").write_text(json.dumps(symbol_map(p)))
    assert dis_main([str(tmp_path / "k.bin"), "--base", "0x2000", "--sym", str(tmp_path / "k.json"),
                     "-o", str(tmp_path / "out.s")]) == 0
    out = (tmp_path / "out.s").read_text()
    assert ".kernel k" in out and "S_BZ s0, back" in out
    assert assemble(out, 0x2000).image == p.image
    assert dis_main([str(tmp_path / "k.hex"), "--listing"]) == 0
    assert "00002000: " in capsys.readouterr().out
    assert read_hex(hex_text(p)) == (0x2000, p.words)
