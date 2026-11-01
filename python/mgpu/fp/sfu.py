"""Bit-exact SFU operations: FRCP, FRSQ, FEX2, FLG2 (docs/isa/sfu.md).

This module is the executable form of the normative algorithm. The coefficient tables
in spec/sfu/*.txt are normative; sfu_tables.py is generated from them.
"""

from .f32 import (CANON_NAN, NEG_INF, ONE, POS_INF, frac_field, exp_field, ftz,
                  is_inf, is_nan, is_zero, round_fixed, sign_of, signed_inf,
                  signed_zero)
from .sfu_tables import TABLES

SEG_BITS = 7     # K: 128 segments per table
Y_FRAC = 30      # result fixed point: Y * 2^-30
C1_FRAC = 22     # C1 = c1 * 2^22
C2_FRAC = 16     # C2 = c2 * 2^16

# Width of the table input t (a fraction in [0, 1), t = T * 2^-W) for each table.
T_BITS = {
    "rcp": 23, "rsqrt_even": 23, "rsqrt_odd": 23, "exp2": 30,
    "log2": 23, "log2_near_hi": 23, "log2_near_lo": 23,
}


def poly(table: str, t: int) -> int:
    """Evaluate the piecewise quadratic of `table` at T = t. Returns Y (value Y * 2^-30)."""
    d_bits = T_BITS[table] - SEG_BITS
    i = t >> d_bits
    d = t & ((1 << d_bits) - 1)
    c0, c1, c2 = TABLES[table][i]
    sq = (d * d) >> (2 * d_bits - 16)
    return c0 + ((c1 * d) >> (d_bits - 1)) + ((c2 * sq) >> 16)


def frcp(x: int) -> int:
    x = ftz(x)
    s = sign_of(x)
    if is_nan(x):
        return CANON_NAN
    if is_inf(x):
        return signed_zero(s)
    if is_zero(x):
        return signed_inf(s)
    e = exp_field(x) - 127
    f = frac_field(x)
    if f == 0:
        return round_fixed(s, 1, -e)
    return round_fixed(s, poly("rcp", f), -Y_FRAC - e)


def frsq(x: int) -> int:
    x = ftz(x)
    s = sign_of(x)
    if is_nan(x):
        return CANON_NAN
    if is_zero(x):
        return signed_inf(s)
    if s:
        return CANON_NAN
    if is_inf(x):
        return 0
    e = exp_field(x) - 127
    f = frac_field(x)
    half = e >> 1          # floor(e / 2)
    if e & 1:
        return round_fixed(0, poly("rsqrt_odd", f), -Y_FRAC - half)
    if f == 0:
        return round_fixed(0, 1, -half)
    return round_fixed(0, poly("rsqrt_even", f), -Y_FRAC - half)


def fex2(x: int) -> int:
    x = ftz(x)
    s = sign_of(x)
    if is_nan(x):
        return CANON_NAN
    if is_inf(x):
        return 0 if s else POS_INF
    if is_zero(x):
        return ONE
    e = exp_field(x) - 127
    if e >= 8:                         # |x| >= 256
        return 0 if s else POS_INF
    mant = (1 << 23) | frac_field(x)   # |x| = mant * 2^(e-23)
    sh = e + 7                         # X = floor(|x| * 2^30)
    xf = mant << sh if sh >= 0 else mant >> -sh
    if s:
        xf = -xf
    n = xf >> 30                       # floor
    t = xf & ((1 << 30) - 1)
    if t == 0:
        return round_fixed(0, 1, n)
    return round_fixed(0, poly("exp2", t), n - Y_FRAC)


def flg2(x: int) -> int:
    x = ftz(x)
    s = sign_of(x)
    if is_nan(x):
        return CANON_NAN
    if is_zero(x):
        return NEG_INF
    if s:
        return CANON_NAN
    if is_inf(x):
        return POS_INF
    e = exp_field(x) - 127
    f = frac_field(x)
    if f == 0:
        return round_fixed(1 if e < 0 else 0, abs(e), 0)
    if e == 0:                         # x in (1, 2): log2(x) = (x - 1) * G(x)
        return round_fixed(0, f * poly("log2_near_hi", f), -23 - Y_FRAC)
    if e == -1:                        # x in (0.5, 1)
        return round_fixed(1, ((1 << 23) - f) * poly("log2_near_lo", f), -24 - Y_FRAC)
    v = (e << Y_FRAC) + poly("log2", f)
    return round_fixed(1 if v < 0 else 0, abs(v), -Y_FRAC)
