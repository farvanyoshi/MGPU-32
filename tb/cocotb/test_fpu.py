"""mgpu_fpu / mgpu_fpu_lane bit-exact against the Rust engine (mgpu._core.fp_batch)."""

import os
import random

import cocotb

from mgpu import _core
from mgpu.fp import ops as pyops
from mgpu.fp.testvals import SPECIALS, random_f32, random_pair
from mgpu.isa import BY_MNEMONIC

from tbutil import pack, start, stream, unpack

L = int(os.environ.get("L", "8"))
LAT = int(os.environ.get("LAT", "4"))
N = int(os.environ.get("FPU_VECTORS", "3000"))      # random operand sets per op
OPS = list(BY_MNEMONIC)
FPU_OPS = {"V_FADD": "fadd", "V_FSUB": "fsub", "V_FMUL": "fmul", "V_FFMA": "ffma",
           "V_FMIN": "fmin", "V_FMAX": "fmax", "V_I2F": "i2f", "V_F2I": "f2i"}


def operand_sets(name, rng):
    sets = []
    if name in ("i2f", "f2i"):
        sets += [(x, 0, 0) for x in SPECIALS]
        sets += [(rng.getrandbits(32) if name == "i2f" else random_f32(rng), 0, 0) for _ in range(N)]
    elif name == "ffma":
        sets += [(x, y, z) for x in SPECIALS[::4] for y in SPECIALS[::3] for z in SPECIALS[::2]]
        for _ in range(N):
            a, b = random_pair(rng)
            c = random_f32(rng)
            if rng.random() < 0.3:
                c = pyops.fmul(a, b) ^ 0x80000000 ^ rng.randint(0, 3)
            sets.append((a, b, c))
    else:
        sets += [(x, y, 0) for x in SPECIALS for y in SPECIALS]
        sets += [(*random_pair(rng), 0) for _ in range(N)]
    return sets


@cocotb.test()
async def all_ops_bit_exact(dut):
    await start(dut)
    rng = random.Random(3)
    items, refs = [], []
    for mn, name in FPU_OPS.items():
        sets = operand_sets(name, rng)
        a, b, c = zip(*sets)
        want = _core.fp_batch(name, list(a), list(b), list(c))
        for k in range(0, len(sets), L):
            chunk = sets[k:k + L]
            chunk += [(0, 0, 0)] * (L - len(chunk))
            items.append((mn, chunk))
            refs.append((mn, chunk, want[k:k + L]))

    def apply(d, item):
        mn, lanes = item
        d.op.value = OPS.index(mn)
        d.a.value = pack([x[0] for x in lanes])
        d.b.value = pack([x[1] for x in lanes])
        d.c.value = pack([x[2] for x in lanes])

    outs = await stream(dut, items, apply, lambda d: unpack(d.result.value, L), LAT, bubble_every=5)
    checked = 0
    for (mn, lanes, want), got in zip(refs, outs):
        for i, w in enumerate(want):
            a, b, c = lanes[i]
            assert got[i] == w, f"{mn}({a:#010x}, {b:#010x}, {c:#010x}): rtl {got[i]:#010x} != {w:#010x}"
            checked += 1
    dut._log.info(f"{checked} lane operations bit-exact across {len(FPU_OPS)} ops")
