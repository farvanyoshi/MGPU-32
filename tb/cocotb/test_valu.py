"""mgpu_valu / mgpu_int_lane against the reference model (mgpu.model.execute, mgpu.fp.ops)."""

import os
import random

import cocotb

from mgpu.fp import ops as fp
from mgpu.fp.testvals import SPECIALS
from mgpu.isa import BY_MNEMONIC
from mgpu.model.execute import int_cond, int_op

from tbutil import pack, start, stream, unpack

L = int(os.environ.get("L", "8"))
LAT = int(os.environ.get("LAT", "2"))
OPS = list(BY_MNEMONIC)
M = 0xFFFFFFFF
EDGES = [0, 1, 2, 3, 31, 32, 33, 0x7FFF, 0x8000, 0xFFFF, 0x7FFFFFFF, 0x80000000, 0x80000001,
         0xFFFFFFFE, M, 0x12345678]


def reference(mn):
    """mnemonic -> f(a, b, c, sel) -> (result, cond); None if this lane does not handle it."""
    ins = BY_MNEMONIC[mn]
    if mn.startswith("V_CMP."):
        cond = ins.op
        if cond.startswith("f"):
            return lambda a, b, c, s: (None, fp.fcmp(cond, a, b))
        return lambda a, b, c, s: (None, int_cond(cond, a, b))
    if mn in ("V_IMAD",):
        return lambda a, b, c, s: ((a * b + c) & M, None)
    if mn == "V_SEL":
        return lambda a, b, c, s: (a if s else b, None)
    if mn == "S_LUI":
        return lambda a, b, c, s: ((b & 0xFFFF) << 16, None)
    if ins.group in ("S_ALU", "V_ALU", "S_ADDI", "S_ANDI", "S_ORI", "V_ADDI", "V_ANDI"):
        return lambda a, b, c, s, op=ins.op: (int_op(op, a, b), None)
    return None


TESTED = [mn for mn in OPS if reference(mn)]


def operands(rng, mn):
    float_cmp = mn.startswith("V_CMP.F")
    pool = SPECIALS if float_cmp else EDGES

    def one():
        r = rng.random()
        return rng.choice(pool) if r < 0.5 else rng.getrandbits(32)
    a, b = one(), one()
    if rng.random() < 0.2:
        b = a                      # equal operands for compares, MIN/MAX
    return a, b, rng.getrandbits(32), rng.getrandbits(1)


@cocotb.test()
async def all_ops(dut):
    await start(dut)
    rng = random.Random(7)
    items = []
    for mn in TESTED:
        for _ in range(64):           # 64 passes = 512 lane operations per op
            items.append((mn, [operands(rng, mn) for _ in range(L)]))

    def apply(d, item):
        mn, lanes = item
        d.op.value = OPS.index(mn)
        d.a.value = pack([x[0] for x in lanes])
        d.b.value = pack([x[1] for x in lanes])
        d.c.value = pack([x[2] for x in lanes])
        d.sel.value = pack([x[3] for x in lanes], 1)

    def read(d):
        return unpack(d.result.value, L), unpack(d.cond.value, L, 1)

    outs = await stream(dut, items, apply, read, LAT, bubble_every=7)
    checked = 0
    for (mn, lanes), (res, cond) in zip(items, outs):
        ref = reference(mn)
        for i, (a, b, c, s) in enumerate(lanes):
            want_r, want_c = ref(a, b, c, s)
            if want_r is not None:
                assert res[i] == want_r, f"{mn}({a:#x}, {b:#x}, {c:#x}, {s}): {res[i]:#x} != {want_r:#x}"
            if want_c is not None:
                assert cond[i] == int(want_c), f"{mn}({a:#x}, {b:#x}): cond {cond[i]} != {int(want_c)}"
            checked += 1
    dut._log.info(f"{len(TESTED)} ops, {checked} lane operations checked")
