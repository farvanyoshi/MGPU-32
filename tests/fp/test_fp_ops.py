"""Float ops vs an independent Fraction oracle, numpy, and hand-checked special cases."""

import random
from fractions import Fraction

import pytest

from mgpu.fp import ops
from mgpu.fp.f32 import CANON_NAN, NEG_INF, POS_INF
from mgpu.fp.testvals import SPECIALS, random_f32, random_pair

N = 20000


def is_nan(x):
    return (x >> 23) & 0xFF == 0xFF and x & 0x7FFFFF


def value(x):
    """Exact value of a flushed, finite binary32 as a Fraction (denormals -> 0)."""
    e = (x >> 23) & 0xFF
    if e == 0:
        return Fraction(0)
    v = Fraction((1 << 23) | (x & 0x7FFFFF)) * Fraction(2) ** (e - 150)
    return -v if x >> 31 else v


def oracle_round(r: Fraction, zero_sign: int = 0) -> int:
    """Independent implementation of rule FP-3 using rational arithmetic."""
    if r == 0:
        return zero_sign << 31
    s = int(r < 0)
    m = abs(r)
    e = m.numerator.bit_length() - m.denominator.bit_length()
    if Fraction(2) ** e > m:
        e -= 1
    if Fraction(2) ** (e + 1) <= m:
        e += 1
    scaled = m / Fraction(2) ** (e - 23)        # in [2^23, 2^24)
    q, rem = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * rem
    if twice > scaled.denominator or (twice == scaled.denominator and q & 1):
        q += 1
    if q == 1 << 24:
        q >>= 1
        e += 1
    if e > 127:
        return (s << 31) | 0x7F800000
    if e < -126:
        return s << 31
    return (s << 31) | ((e + 127) << 23) | (q & 0x7FFFFF)


def finite(x):
    return (x >> 23) & 0xFF != 0xFF


def nonzero(x):
    return (x >> 23) & 0xFF != 0


@pytest.mark.parametrize("seed", range(4))
def test_fadd_fsub_fmul_vs_oracle(seed):
    rng = random.Random(seed)
    for _ in range(N):
        a, b = random_pair(rng)
        if not (finite(a) and finite(b) and nonzero(a) and nonzero(b)):
            continue
        va, vb = value(a), value(b)
        assert ops.fadd(a, b) == oracle_round(va + vb), (hex(a), hex(b))
        assert ops.fsub(a, b) == oracle_round(va - vb), (hex(a), hex(b))
        assert ops.fmul(a, b) == oracle_round(va * vb, (a ^ b) >> 31), (hex(a), hex(b))


@pytest.mark.parametrize("seed", range(4))
def test_ffma_vs_oracle(seed):
    rng = random.Random(100 + seed)
    for _ in range(N):
        a, b = random_pair(rng)
        c = random_f32(rng)
        if rng.random() < 0.3 and finite(a) and finite(b):  # make c cancel a*b
            c = ops.fmul(a, b) ^ 0x80000000
        if not all(finite(x) and nonzero(x) for x in (a, b, c)):
            continue
        assert ops.ffma(a, b, c) == oracle_round(value(a) * value(b) + value(c)), \
            (hex(a), hex(b), hex(c))


def test_i2f_vs_oracle():
    rng = random.Random(7)
    vals = [0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0x01000001, 0xFEFFFFFF]
    vals += [rng.getrandbits(32) for _ in range(N)]
    for v in vals:
        s = v - (1 << 32) if v >> 31 else v
        assert ops.i2f(v) == oracle_round(Fraction(s)), hex(v)


def test_f2i_vs_oracle():
    rng = random.Random(8)
    for x in SPECIALS + [random_f32(rng) for _ in range(N)]:
        got = ops.f2i(x)
        if is_nan(x):
            exp = 0
        elif not finite(x):
            exp = 0x80000000 if x >> 31 else 0x7FFFFFFF
        else:
            v = value(x)
            t = int(v)  # truncates toward zero
            exp = max(-(1 << 31), min((1 << 31) - 1, t)) & 0xFFFFFFFF
        assert got == exp, hex(x)


def test_compares_and_minmax_vs_oracle():
    rng = random.Random(9)
    for _ in range(N):
        a, b = random_pair(rng)
        na, nb = is_nan(a), is_nan(b)
        if na or nb:
            assert not any(ops.fcmp(c, a, b) for c in ("feq", "flt", "fle", "fgt", "fge"))
            assert ops.fcmp("fne", a, b)
            continue
        inf = Fraction(10) ** 50
        va = value(a) if finite(a) else (-inf if a >> 31 else inf)
        vb = value(b) if finite(b) else (-inf if b >> 31 else inf)
        assert ops.fcmp("feq", a, b) == (va == vb)
        assert ops.fcmp("flt", a, b) == (va < vb)
        assert ops.fcmp("fge", a, b) == (va >= vb)
        lo, hi = ops.fmin(a, b), ops.fmax(a, b)
        assert {lo, hi} <= {ops.fmin(a, a), ops.fmin(b, b)}  # returns one of the flushed inputs
        if va != vb:
            assert (lo, hi) == ((ops.fmin(a, a), ops.fmin(b, b)) if va < vb else (ops.fmin(b, b), ops.fmin(a, a)))


@pytest.mark.parametrize("fn, a, b, want", [
    ("fadd", 0x7F800000, 0xFF800000, CANON_NAN),          # inf - inf
    ("fadd", 0x80000000, 0x80000000, 0x80000000),         # -0 + -0 = -0
    ("fadd", 0x80000000, 0x00000000, 0x00000000),         # -0 + +0 = +0
    ("fadd", 0x3F800000, 0xBF800000, 0x00000000),         # x - x = +0
    ("fadd", 0x00000001, 0x80000000, 0x00000000),         # denormal flushes to +0
    ("fadd", 0x80000001, 0x80000000, 0x80000000),         # -denormal -> -0, -0 + -0
    ("fadd", 0x7F7FFFFF, 0x7F7FFFFF, POS_INF),            # overflow
    ("fadd", 0x00800001, 0x80800000, 0x00000000),         # tiny exact result flushes to +0
    ("fadd", 0x80800001, 0x00800000, 0x80000000),         # ... keeping the sign of the result
    ("fsub", 0x00000000, 0x00000000, 0x00000000),         # +0 - +0 = +0
    ("fsub", 0x80000000, 0x00000000, 0x80000000),         # -0 - +0 = -0
    ("fsub", 0x7FC00001, 0x00000000, CANON_NAN),
    ("fmul", 0x7F800000, 0x00000000, CANON_NAN),          # inf * 0
    ("fmul", 0x7F800000, 0x00000001, CANON_NAN),          # inf * denormal(=0)
    ("fmul", 0xFF800000, 0x3F800000, NEG_INF),
    ("fmul", 0x80000000, 0x3F800000, 0x80000000),
    ("fmul", 0x00800000, 0x3F000000, 0x00000000),         # 2^-126 * 0.5 flushes
    ("fmul", 0x00FFFFFF, 0x3F000001, 0x00800000),         # rounds up to 2^-126: tininess after rounding
    ("fmul", 0xFFC00001, 0x3F800000, CANON_NAN),          # NaN payload not propagated
    ("fmin", 0x00000000, 0x80000000, 0x80000000),         # -0 < +0
    ("fmax", 0x80000000, 0x00000000, 0x00000000),
    ("fmin", 0x7FC00000, 0x3F800000, 0x3F800000),
    ("fmax", 0x7FC00001, 0xFFC00000, CANON_NAN),
    ("fmin", 0x7FC00000, 0x00000001, 0x00000000),         # the other input, flushed
])
def test_binary_special_cases(fn, a, b, want):
    assert ops.BINARY[fn](a, b) == want


@pytest.mark.parametrize("a, b, c, want", [
    (0x00000000, 0x7F800000, 0x7FC00000, CANON_NAN),
    (0x00000000, 0x7F800000, 0x3F800000, CANON_NAN),      # 0 * inf
    (0x7F800000, 0x3F800000, 0xFF800000, CANON_NAN),      # inf - inf
    (0x7F800000, 0xBF800000, 0xFF800000, NEG_INF),
    (0x3F800000, 0x3F800000, 0x7F800000, POS_INF),
    (0x80000000, 0x3F800000, 0x80000000, 0x80000000),     # -0 + -0
    (0x00000000, 0xBF800000, 0x00000000, 0x00000000),     # -0 + +0 = +0
    (0x3F800000, 0x3F800000, 0xBF800000, 0x00000000),     # exact cancellation = +0
    (0x3F800001, 0x3F800001, 0xBF800002, 0x28800000),     # no intermediate rounding
    (0x00000000, 0x3F800000, 0x40000000, 0x40000000),
])
def test_ffma_special_cases(a, b, c, want):
    assert ops.ffma(a, b, c) == want


def test_i2f_f2i_cases():
    assert ops.i2f(0) == 0
    assert ops.i2f(0x80000000) == 0xCF000000
    assert ops.i2f(0x01000001) == 0x4B800000  # 16777217 -> ties to even 16777216
    assert ops.i2f(0x01000003) == 0x4B800002
    assert ops.f2i(0x4F000000) == 0x7FFFFFFF    # 2^31 saturates
    assert ops.f2i(0xCF000000) == 0x80000000    # -2^31 exact
    assert ops.f2i(0xBFC00000) == 0xFFFFFFFF    # -1.5 -> -1
    assert ops.f2i(0x7F800000) == 0x7FFFFFFF
    assert ops.f2i(0xFF800000) == 0x80000000
    assert ops.f2i(0x7FC00000) == 0


def test_numpy_sanity():
    np = pytest.importorskip("numpy")
    rng = random.Random(11)
    a = np.array([random_f32(rng) for _ in range(N)], dtype=np.uint32)
    b = np.array([random_f32(rng) for _ in range(N)], dtype=np.uint32)
    fa, fb = a.view(np.float32), b.view(np.float32)

    def normal(arr):
        e = (arr.view(np.uint32) >> 23) & 0xFF
        return (e > 1) & (e < 0xFF)

    with np.errstate(all="ignore"):
        for name, res in (("fadd", fa + fb), ("fsub", fa - fb), ("fmul", fa * fb)):
            ok = normal(a) & normal(b) & normal(res)
            for x, y, r in zip(a[ok], b[ok], res[ok].view(np.uint32)):
                assert ops.BINARY[name](int(x), int(y)) == int(r), (name, hex(x), hex(y))
