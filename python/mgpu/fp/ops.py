"""Bit-exact MGPU-32 float32 operations (docs/isa/MGPU-32-ISA.md, section 8).

Every function takes and returns raw 32-bit patterns. Finite values are decoded into an
exact (sign, M, E) triple with value (-1)^sign * M * 2^E. Sums and products are computed
exactly with Python integers and rounded once by `round_fixed` (rule FP-3).
"""

from .f32 import (CANON_NAN, SIGN, exp_field, frac_field, ftz, is_inf, is_nan,
                  is_zero, round_fixed, sign_of, signed_inf, signed_zero)
from .sfu import fex2, flg2, frcp, frsq

INT_MAX = 0x7FFFFFFF
INT_MIN = 0x80000000


def _unpack(x: int) -> tuple[int, int, int]:
    """Finite, normal x -> (sign, M, E) with x = (-1)^sign * M * 2^E, M a 24-bit integer."""
    return sign_of(x), (1 << 23) | frac_field(x), exp_field(x) - 150


def _add_exact(sa: int, ma: int, ea: int, sb: int, mb: int, eb: int) -> tuple[int, int]:
    """Exact signed sum of two (sign, M, E) values, as (integer, exponent)."""
    e = min(ea, eb)
    va = (ma << (ea - e)) * (-1 if sa else 1)
    vb = (mb << (eb - e)) * (-1 if sb else 1)
    return va + vb, e


def fadd(a: int, b: int) -> int:
    """[FP-4]"""
    a, b = ftz(a), ftz(b)
    if is_nan(a) or is_nan(b):
        return CANON_NAN
    if is_inf(a) and is_inf(b):
        return a if sign_of(a) == sign_of(b) else CANON_NAN
    if is_inf(a):
        return a
    if is_inf(b):
        return b
    if is_zero(a) and is_zero(b):
        return a if sign_of(a) == sign_of(b) else 0
    if is_zero(a):
        return b
    if is_zero(b):
        return a
    v, e = _add_exact(*_unpack(a), *_unpack(b))
    if v == 0:
        return 0
    return round_fixed(int(v < 0), abs(v), e)


def fsub(a: int, b: int) -> int:
    """[FP-4]: a - b = a + (b with bit 31 flipped)."""
    if is_nan(a) or is_nan(b):
        return CANON_NAN
    return fadd(a, b ^ SIGN)


def fmul(a: int, b: int) -> int:
    """[FP-5]"""
    a, b = ftz(a), ftz(b)
    if is_nan(a) or is_nan(b):
        return CANON_NAN
    s = sign_of(a) ^ sign_of(b)
    if (is_inf(a) and is_zero(b)) or (is_zero(a) and is_inf(b)):
        return CANON_NAN
    if is_inf(a) or is_inf(b):
        return signed_inf(s)
    if is_zero(a) or is_zero(b):
        return signed_zero(s)
    _, ma, ea = _unpack(a)
    _, mb, eb = _unpack(b)
    return round_fixed(s, ma * mb, ea + eb)


def ffma(a: int, b: int, c: int) -> int:
    """[FP-6]: a * b + c with a single rounding."""
    a, b, c = ftz(a), ftz(b), ftz(c)
    if is_nan(a) or is_nan(b) or is_nan(c):
        return CANON_NAN
    sp = sign_of(a) ^ sign_of(b)
    p_inf = is_inf(a) or is_inf(b)
    p_zero = is_zero(a) or is_zero(b)
    if p_inf and p_zero:
        return CANON_NAN
    if p_inf:
        if is_inf(c) and sign_of(c) != sp:
            return CANON_NAN
        return signed_inf(sp)
    if is_inf(c):
        return c
    if p_zero:
        if is_zero(c):
            return c if sign_of(c) == sp else 0
        return c
    _, ma, ea = _unpack(a)
    _, mb, eb = _unpack(b)
    if is_zero(c):
        return round_fixed(sp, ma * mb, ea + eb)
    sc, mc, ec = _unpack(c)
    e = min(ea + eb, ec)
    v = (ma * mb << (ea + eb - e)) * (-1 if sp else 1) + (mc << (ec - e)) * (-1 if sc else 1)
    if v == 0:
        return 0
    return round_fixed(int(v < 0), abs(v), e)


def _order_key(x: int) -> int:
    """Total order on non-NaN values with -0 < +0 (used by FMIN/FMAX)."""
    mag = x & 0x7FFFFFFF
    return -mag - 1 if sign_of(x) else mag


def fmin(a: int, b: int) -> int:
    """[FP-7]"""
    a, b = ftz(a), ftz(b)
    if is_nan(a) and is_nan(b):
        return CANON_NAN
    if is_nan(a):
        return b
    if is_nan(b):
        return a
    return a if _order_key(a) <= _order_key(b) else b


def fmax(a: int, b: int) -> int:
    """[FP-7]"""
    a, b = ftz(a), ftz(b)
    if is_nan(a) and is_nan(b):
        return CANON_NAN
    if is_nan(a):
        return b
    if is_nan(b):
        return a
    return a if _order_key(a) >= _order_key(b) else b


def i2f(a: int) -> int:
    """[FP-9]"""
    v = a - (1 << 32) if a & SIGN else a
    return round_fixed(int(v < 0), abs(v), 0)


def f2i(a: int) -> int:
    """[FP-10]"""
    a = ftz(a)
    if is_nan(a) or is_zero(a):
        return 0
    if is_inf(a):
        return INT_MIN if sign_of(a) else INT_MAX
    s, m, e = _unpack(a)
    mag = m << e if e >= 0 else m >> -e
    v = -mag if s else mag
    v = max(-(1 << 31), min((1 << 31) - 1, v))
    return v & 0xFFFFFFFF


def _cmp_key(x: int) -> int:
    """Total order on non-NaN, flushed values in which -0 == +0 ([FP-8])."""
    return 0 if is_zero(x) else _order_key(x)


def fcmp(cond: str, a: int, b: int) -> bool:
    """[FP-8] cond is one of feq, fne, flt, fle, fgt, fge."""
    a, b = ftz(a), ftz(b)
    if is_nan(a) or is_nan(b):
        return cond == "fne"
    ka, kb = _cmp_key(a), _cmp_key(b)
    return {"feq": ka == kb, "fne": ka != kb, "flt": ka < kb,
            "fle": ka <= kb, "fgt": ka > kb, "fge": ka >= kb}[cond]


BINARY = {"fadd": fadd, "fsub": fsub, "fmul": fmul, "fmin": fmin, "fmax": fmax}
UNARY = {"frcp": frcp, "frsq": frsq, "fex2": fex2, "flg2": flg2, "i2f": i2f, "f2i": f2i}
