"""mgpu_sfu / mgpu_sfu_lane bit-exact against the Rust engine (mgpu._core.fp_batch)."""

import os
import random

import cocotb

from mgpu import _core
from mgpu.fp.testvals import SPECIALS, random_f32
from mgpu.isa import BY_MNEMONIC

from tbutil import pack, start, stream, unpack

S = int(os.environ.get("S", "2"))
LAT = int(os.environ.get("LAT", "8"))
N = int(os.environ.get("SFU_VECTORS", "2500"))
OPS = list(BY_MNEMONIC)
SFU_OPS = {"V_FRCP": "frcp", "V_FRSQ": "frsq", "V_FEX2": "fex2", "V_FLG2": "flg2"}

# exponents that select different paths: near one (log2), odd/even (rsqrt), exp2 range limits
EXPONENTS = [-126, -100, -30, -24, -2, -1, 0, 1, 2, 3, 6, 7, 8, 30, 100, 127]


def operands(rng):
    xs = list(SPECIALS)
    xs += [(rng.getrandbits(1) << 31) | ((e + 127) << 23) | rng.getrandbits(23)
           for e in EXPONENTS for _ in range(N // len(EXPONENTS))]
    xs += [((e + 127) << 23) for e in EXPONENTS]                       # exact powers of two
    xs += [random_f32(rng) for _ in range(N // 2)]
    return xs


@cocotb.test()
async def all_ops_bit_exact(dut):
    await start(dut)
    rng = random.Random(5)
    items, refs = [], []
    for mn, name in SFU_OPS.items():
        xs = operands(rng)
        want = _core.fp_batch(name, xs)
        for k in range(0, len(xs), S):
            chunk = xs[k:k + S] + [0] * (S - len(xs[k:k + S]))
            items.append((mn, chunk))
            refs.append((mn, chunk, want[k:k + S]))

    def apply(d, item):
        mn, lanes = item
        d.op.value = OPS.index(mn)
        d.a.value = pack(lanes)

    outs = await stream(dut, items, apply, lambda d: unpack(d.result.value, S), LAT, bubble_every=9)
    checked = 0
    for (mn, lanes, want), got in zip(refs, outs):
        for i, w in enumerate(want):
            assert got[i] == w, f"{mn}({lanes[i]:#010x}): rtl {got[i]:#010x} != {w:#010x}"
            checked += 1
    dut._log.info(f"{checked} SFU operations bit-exact")
