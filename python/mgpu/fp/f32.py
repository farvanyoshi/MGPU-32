"""binary32 bit-level helpers shared by every MGPU-32 floating-point operation.

All values are handled as raw 32-bit integers. Nothing here uses host floats, so the
results are identical on every platform.
"""

CANON_NAN = 0x7FC00000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
ONE = 0x3F800000
SIGN = 0x80000000

MIN_EXP = -126  # smallest normal exponent
MAX_EXP = 127   # largest finite exponent


def sign_of(x: int) -> int:
    return (x >> 31) & 1


def exp_field(x: int) -> int:
    return (x >> 23) & 0xFF


def frac_field(x: int) -> int:
    return x & 0x7FFFFF


def is_nan(x: int) -> bool:
    return exp_field(x) == 0xFF and frac_field(x) != 0


def is_inf(x: int) -> bool:
    return exp_field(x) == 0xFF and frac_field(x) == 0


def is_zero(x: int) -> bool:
    """True for +-0 *after* flushing, i.e. also for denormals."""
    return exp_field(x) == 0


def ftz(x: int) -> int:
    """Flush a denormal input to a zero of the same sign (rule FP-1)."""
    if exp_field(x) == 0:
        return x & SIGN
    return x


def signed_zero(sign: int) -> int:
    return SIGN if sign else 0


def signed_inf(sign: int) -> int:
    return NEG_INF if sign else POS_INF


def round_fixed(sign: int, m: int, scale: int) -> int:
    """Round the exact value (-1)^sign * m * 2^scale to binary32 (rule FP-3).

    m is a non-negative integer of any size. Rounding is to nearest, ties to even, at
    24 significant bits with an unbounded exponent. The rounded result is then flushed
    to a signed zero if its exponent is below -126 (tininess after rounding), or becomes
    a signed infinity if its exponent is above 127.
    """
    if m == 0:
        return signed_zero(sign)
    p = m.bit_length() - 1  # index of the leading one
    if p > 23:
        sh = p - 23
        q = m >> sh
        rem = m & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
            if q == 1 << 24:
                q >>= 1
                p += 1
    else:
        q = m << (23 - p)
    e = p + scale
    if e > MAX_EXP:
        return signed_inf(sign)
    if e < MIN_EXP:
        return signed_zero(sign)
    return (sign << 31) | ((e + 127) << 23) | (q & 0x7FFFFF)
