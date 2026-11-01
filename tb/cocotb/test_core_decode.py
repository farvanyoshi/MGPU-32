"""mgpu_decode (ctrl_t) against the operand syntax of spec/isa.toml."""

import random

import cocotb
from cocotb.triggers import Timer

from mgpu.isa import BY_MNEMONIC, SR_BY_ID, IllegalInstruction, decode

OPS = list(BY_MNEMONIC)
UNIT = {"SALU": 0, "BRU": 0, "SYS": 0, "VALU": 1, "FPU": 2, "SFU": 3, "LSU": 4}
M = 0xFFFFFFFF


def expected(word):
    try:
        d = decode(word)
    except IllegalInstruction:
        return None
    ins = d.instr
    syn = ins.syntax
    illegal = False
    if ins.group == "GETSR":
        sr = SR_BY_ID.get(d.ra)
        illegal = sr is None or (ins.mnemonic == "S_GETSR" and sr["kind"] == "lane")
    imm = None
    for tok in syn:
        if tok in ("simm16",) or "(" in tok:
            imm = d.simm16 & M
        elif tok == "uimm16":
            imm = d.imm16
        elif tok == "shamt":
            imm = d.rb
        elif tok == "target16":
            imm = (d.simm16 * 4) & M
        elif tok == "target26":
            imm = (d.soffset26 * 4) & M
    return dict(illegal=illegal, op=OPS.index(ins.mnemonic), unit=UNIT[ins.unit], imm=imm,
                sb=int("vb|sb" in syn and bool(d.sb)),
                vec_dst=int(bool(syn) and syn[0] == "vd"),
                scl_dst=int(bool(syn) and syn[0] == "sd" and ins.group not in ("S_BZ", "S_BNZ")),
                branch=int(ins.group in ("S_BZ", "S_BNZ", "S_J")),
                b_imm=int(any(t in ("simm16", "uimm16", "shamt") for t in syn)),
                mem_shared=int(ins.space == "shared"))


async def check(dut, word):
    dut.word.value = word
    await Timer(1, unit="ns")
    want = expected(word)
    if want is None or want["illegal"]:
        assert int(dut.illegal.value) == 1, f"{word:#010x} should be illegal"
        return
    assert int(dut.illegal.value) == 0, f"{word:#010x} decoded illegal"
    for k, v in want.items():
        if k == "illegal" or v is None:
            continue
        got = int(getattr(dut, k).value)
        assert got == v, f"{word:#010x} {OPS[want['op']]}: {k} = {got:#x}, expected {v:#x}"


@cocotb.test()
async def every_instruction_random_fields(dut):
    rng = random.Random(4)
    from mgpu.isa import encode
    for mn in OPS:
        for _ in range(300):
            await check(dut, encode(mn, rd=rng.randrange(32), ra=rng.randrange(32),
                                    rb=rng.randrange(32), rc=rng.randrange(32), sb=rng.getrandbits(1),
                                    imm=rng.getrandbits(16), offset=rng.getrandbits(26)))


@cocotb.test()
async def random_words(dut):
    rng = random.Random(5)
    for _ in range(30000):
        await check(dut, rng.getrandbits(32))
