"""mgpu_isa_pkg::decode_op and unit_of against mgpu.isa (rule DEC-1)."""

import random

import cocotb
from cocotb.triggers import Timer

from mgpu.isa import BY_MNEMONIC, IllegalInstruction, decode

OPS = list(BY_MNEMONIC)            # same order as the SV op_e enum
UNIT = {"SALU": 0, "BRU": 0, "SYS": 0, "VALU": 1, "FPU": 2, "SFU": 3, "LSU": 4}


def expect(word):
    try:
        ins = decode(word).instr
    except IllegalInstruction:
        return None
    return OPS.index(ins.mnemonic), UNIT[ins.unit]


async def check(dut, word):
    dut.word.value = word
    await Timer(1, unit="ns")
    want = expect(word)
    if want is None:
        assert int(dut.valid.value) == 0, f"{word:#010x} should be illegal"
    else:
        got = (int(dut.op.value), int(dut.unit.value))
        assert int(dut.valid.value) == 1 and got == want, f"{word:#010x}: {got} != {want}"


@cocotb.test()
async def every_opcode_and_funct(dut):
    """All 64 opcodes x all 1024 values of the R-format funct field (covers R4's 5 bits too),
    with the other fields randomized."""
    rng = random.Random(1)
    for opc in range(64):
        for funct in range(1024):
            await check(dut, opc << 26 | rng.getrandbits(16) << 10 | funct)


@cocotb.test()
async def random_words(dut):
    rng = random.Random(2)
    for _ in range(50000):
        await check(dut, rng.getrandbits(32))
