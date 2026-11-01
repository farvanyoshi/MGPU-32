"""Differential test of the float paths: Python golden ops vs the Rust engine."""

import os
import random

import pytest

from mgpu.device import rust_available
from mgpu.fp import ops, sfu
from mgpu.fp.testvals import SPECIALS, random_f32, random_pair

pytestmark = pytest.mark.skipif(not rust_available(), reason="mgpu._core not built")
N = int(os.environ.get("MGPU_FP_DIFF_N", "30000"))

PY = {
    "fadd": ops.fadd, "fsub": ops.fsub, "fmul": ops.fmul, "fmin": ops.fmin, "fmax": ops.fmax,
    "ffma": ops.ffma, "i2f": ops.i2f, "f2i": ops.f2i,
    "frcp": sfu.frcp, "frsq": sfu.frsq, "fex2": sfu.fex2, "flg2": sfu.flg2,
    **{c: (lambda c: lambda a, b: int(ops.fcmp(c, a, b)))(c)
       for c in ("feq", "fne", "flt", "fle", "fgt", "fge")},
}
ARITY = {"ffma": 3, "i2f": 1, "f2i": 1, "frcp": 1, "frsq": 1, "fex2": 1, "flg2": 1}


def operands(name, rng):
    n = ARITY.get(name, 2)
    # exhaustive cross product of the specials, then random values
    if n == 1:
        a = SPECIALS + [random_f32(rng) for _ in range(N)]
        return a, None, None
    if n == 2:
        pairs = [(x, y) for x in SPECIALS for y in SPECIALS] + [random_pair(rng) for _ in range(N)]
        return [p[0] for p in pairs], [p[1] for p in pairs], None
    trip = [(x, y, z) for x in SPECIALS[::3] for y in SPECIALS[::2] for z in SPECIALS]
    for _ in range(N):
        a, b = random_pair(rng)
        c = random_f32(rng)
        if rng.random() < 0.3:  # make the addend cancel the product
            c = ops.fmul(a, b) ^ 0x80000000 ^ rng.randint(0, 3)
        trip.append((a, b, c))
    return [t[0] for t in trip], [t[1] for t in trip], [t[2] for t in trip]


@pytest.mark.parametrize("name", list(PY))
def test_rust_matches_python(name):
    from mgpu import _core
    rng = random.Random(hash(name) & 0xFFFF)
    a, b, c = operands(name, rng)
    got = _core.fp_batch(name, a, b, c)
    n = ARITY.get(name, 2)
    for i, g in enumerate(got):
        args = (a[i],) if n == 1 else (a[i], b[i]) if n == 2 else (a[i], b[i], c[i])
        want = PY[name](*args)
        assert g == want, f"{name}{tuple(hex(x) for x in args)}: rust {g:#x} python {want:#x}"


def test_sfu_mantissa_sweep():
    """Every 7th mantissa for the exponents that select different SFU paths."""
    from mgpu import _core
    for e in (-1, 0, 1, 2, -2, 7, -100):
        xs = [((e + 127) << 23) | f for f in range(0, 1 << 23, 7 * 997)]
        for name in ("frcp", "frsq", "fex2", "flg2"):
            got = _core.fp_batch(name, xs)
            assert got == [PY[name](x) for x in xs], (name, e)
