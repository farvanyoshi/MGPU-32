//! Bit-exact MGPU-32 binary32 arithmetic (ISA section 8).
//!
//! Mirrors python/mgpu/fp/ops.py. Host floating point is never used: every result is
//! produced by exact integer arithmetic followed by a single `round_fixed`.

pub const CANON_NAN: u32 = 0x7FC0_0000;
pub const POS_INF: u32 = 0x7F80_0000;
pub const NEG_INF: u32 = 0xFF80_0000;
pub const SIGN: u32 = 0x8000_0000;

#[inline]
pub fn sign_of(x: u32) -> u32 {
    x >> 31
}
#[inline]
fn exp_field(x: u32) -> u32 {
    (x >> 23) & 0xFF
}
#[inline]
pub fn is_nan(x: u32) -> bool {
    exp_field(x) == 0xFF && x & 0x7F_FFFF != 0
}
#[inline]
pub fn is_inf(x: u32) -> bool {
    x & 0x7FFF_FFFF == POS_INF
}
/// True for +-0 after flushing, i.e. also for denormals.
#[inline]
pub fn is_zero(x: u32) -> bool {
    exp_field(x) == 0
}
/// [FP-1]: flush a denormal input to a zero of the same sign.
#[inline]
pub fn ftz(x: u32) -> u32 {
    if exp_field(x) == 0 {
        x & SIGN
    } else {
        x
    }
}
#[inline]
pub fn signed_zero(s: u32) -> u32 {
    s << 31
}
#[inline]
pub fn signed_inf(s: u32) -> u32 {
    if s != 0 {
        NEG_INF
    } else {
        POS_INF
    }
}

/// [FP-3]: round (-1)^sign * m * 2^scale to binary32 (nearest-even at 24 bits, unbounded
/// exponent, then overflow to infinity and flush below 2^-126).
pub fn round_fixed(sign: u32, m: u128, scale: i32) -> u32 {
    if m == 0 {
        return signed_zero(sign);
    }
    let mut p = 127 - m.leading_zeros() as i32;
    let q: u128 = if p > 23 {
        let sh = (p - 23) as u32;
        let mut q = m >> sh;
        let rem = m & ((1u128 << sh) - 1);
        let half = 1u128 << (sh - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
            if q == 1 << 24 {
                q >>= 1;
                p += 1;
            }
        }
        q
    } else {
        m << (23 - p) as u32
    };
    let e = p + scale;
    if e > 127 {
        return signed_inf(sign);
    }
    if e < -126 {
        return signed_zero(sign);
    }
    (sign << 31) | (((e + 127) as u32) << 23) | (q as u32 & 0x7F_FFFF)
}

/// Finite normal x -> (sign, M, E) with x = (-1)^sign * M * 2^E.
#[inline]
fn unpack(x: u32) -> (u32, u64, i32) {
    (
        sign_of(x),
        ((x & 0x7F_FFFF) | 0x80_0000) as u64,
        exp_field(x) as i32 - 150,
    )
}

/// Exact signed sum of two non-zero values (sign, M, E), rounded once. The smaller operand
/// is jammed into a sticky bit when it lies more than 63 bits below the larger one, which
/// cannot change the rounded result (see docs/sim/README.md, "Float arithmetic").
fn add_round(sa: u32, ma: u64, ea: i32, sb: u32, mb: u64, eb: i32) -> u32 {
    // Normalize both mantissas so the leading one is bit 63.
    let (la, lb) = (ma.leading_zeros(), mb.leading_zeros());
    let (mut sa, mut ma, mut ea) = (sa, ma << la, ea - la as i32);
    let (mut sb, mut mb, mut eb) = (sb, mb << lb, eb - lb as i32);
    if eb > ea || (eb == ea && mb > ma) {
        std::mem::swap(&mut sa, &mut sb);
        std::mem::swap(&mut ma, &mut mb);
        std::mem::swap(&mut ea, &mut eb);
    }
    let d = (ea - eb) as u32;
    let (big, small, e) = if d <= 63 {
        ((ma as u128) << d, mb as u128, eb)
    } else {
        let sh = d - 63;
        let shifted = if sh >= 64 { 0 } else { mb >> sh };
        let lost = sh >= 64 || mb & ((1u64 << sh) - 1) != 0;
        ((ma as u128) << 63, (shifted | lost as u64) as u128, ea - 63)
    };
    // |big| >= |small| by construction, so the result sign is the sign of `big`.
    if sa == sb {
        round_fixed(sa, big + small, e)
    } else if big == small {
        0
    } else {
        round_fixed(sa, big - small, e)
    }
}

/// [FP-4]
pub fn fadd(a: u32, b: u32) -> u32 {
    let (a, b) = (ftz(a), ftz(b));
    if is_nan(a) || is_nan(b) {
        return CANON_NAN;
    }
    if is_inf(a) && is_inf(b) {
        return if sign_of(a) == sign_of(b) {
            a
        } else {
            CANON_NAN
        };
    }
    if is_inf(a) {
        return a;
    }
    if is_inf(b) {
        return b;
    }
    if is_zero(a) && is_zero(b) {
        return if sign_of(a) == sign_of(b) { a } else { 0 };
    }
    if is_zero(a) {
        return b;
    }
    if is_zero(b) {
        return a;
    }
    let (sa, ma, ea) = unpack(a);
    let (sb, mb, eb) = unpack(b);
    add_round(sa, ma, ea, sb, mb, eb)
}

/// [FP-4]
pub fn fsub(a: u32, b: u32) -> u32 {
    if is_nan(a) || is_nan(b) {
        return CANON_NAN;
    }
    fadd(a, b ^ SIGN)
}

/// [FP-5]
pub fn fmul(a: u32, b: u32) -> u32 {
    let (a, b) = (ftz(a), ftz(b));
    if is_nan(a) || is_nan(b) {
        return CANON_NAN;
    }
    let s = sign_of(a) ^ sign_of(b);
    if (is_inf(a) && is_zero(b)) || (is_zero(a) && is_inf(b)) {
        return CANON_NAN;
    }
    if is_inf(a) || is_inf(b) {
        return signed_inf(s);
    }
    if is_zero(a) || is_zero(b) {
        return signed_zero(s);
    }
    let (_, ma, ea) = unpack(a);
    let (_, mb, eb) = unpack(b);
    round_fixed(s, (ma * mb) as u128, ea + eb)
}

/// [FP-6]: a * b + c with a single rounding.
pub fn ffma(a: u32, b: u32, c: u32) -> u32 {
    let (a, b, c) = (ftz(a), ftz(b), ftz(c));
    if is_nan(a) || is_nan(b) || is_nan(c) {
        return CANON_NAN;
    }
    let sp = sign_of(a) ^ sign_of(b);
    let p_inf = is_inf(a) || is_inf(b);
    let p_zero = is_zero(a) || is_zero(b);
    if p_inf && p_zero {
        return CANON_NAN;
    }
    if p_inf {
        if is_inf(c) && sign_of(c) != sp {
            return CANON_NAN;
        }
        return signed_inf(sp);
    }
    if is_inf(c) {
        return c;
    }
    if p_zero {
        if is_zero(c) {
            return if sign_of(c) == sp { c } else { 0 };
        }
        return c;
    }
    let (_, ma, ea) = unpack(a);
    let (_, mb, eb) = unpack(b);
    if is_zero(c) {
        return round_fixed(sp, (ma * mb) as u128, ea + eb);
    }
    let (sc, mc, ec) = unpack(c);
    add_round(sp, ma * mb, ea + eb, sc, mc, ec)
}

/// Total order on non-NaN values with -0 < +0.
#[inline]
fn order_key(x: u32) -> i64 {
    let mag = (x & 0x7FFF_FFFF) as i64;
    if sign_of(x) != 0 {
        -mag - 1
    } else {
        mag
    }
}

/// [FP-7]
pub fn fmin(a: u32, b: u32) -> u32 {
    let (a, b) = (ftz(a), ftz(b));
    match (is_nan(a), is_nan(b)) {
        (true, true) => CANON_NAN,
        (true, false) => b,
        (false, true) => a,
        _ => {
            if order_key(a) <= order_key(b) {
                a
            } else {
                b
            }
        }
    }
}

/// [FP-7]
pub fn fmax(a: u32, b: u32) -> u32 {
    let (a, b) = (ftz(a), ftz(b));
    match (is_nan(a), is_nan(b)) {
        (true, true) => CANON_NAN,
        (true, false) => b,
        (false, true) => a,
        _ => {
            if order_key(a) >= order_key(b) {
                a
            } else {
                b
            }
        }
    }
}

/// [FP-9]
pub fn i2f(a: u32) -> u32 {
    let v = a as i32 as i64;
    round_fixed((v < 0) as u32, v.unsigned_abs() as u128, 0)
}

/// [FP-10]
pub fn f2i(a: u32) -> u32 {
    let a = ftz(a);
    if is_nan(a) || is_zero(a) {
        return 0;
    }
    if is_inf(a) {
        return if sign_of(a) != 0 {
            0x8000_0000
        } else {
            0x7FFF_FFFF
        };
    }
    let (s, m, e) = unpack(a);
    let mag: i64 = if e >= 0 {
        if e >= 40 {
            i64::MAX
        } else {
            (m << e) as i64
        }
    } else if e <= -64 {
        0
    } else {
        (m >> -e) as i64
    };
    let v = if s != 0 { -mag } else { mag };
    v.clamp(i32::MIN as i64, i32::MAX as i64) as i32 as u32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FCond {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// [FP-8]
pub fn fcmp(cond: FCond, a: u32, b: u32) -> bool {
    let (a, b) = (ftz(a), ftz(b));
    if is_nan(a) || is_nan(b) {
        return cond == FCond::Ne;
    }
    let key = |x: u32| if is_zero(x) { 0 } else { order_key(x) };
    let (ka, kb) = (key(a), key(b));
    match cond {
        FCond::Eq => ka == kb,
        FCond::Ne => ka != kb,
        FCond::Lt => ka < kb,
        FCond::Le => ka <= kb,
        FCond::Gt => ka > kb,
        FCond::Ge => ka >= kb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specials() {
        assert_eq!(fadd(0x7F80_0000, 0xFF80_0000), CANON_NAN);
        assert_eq!(fadd(0x8000_0000, 0x8000_0000), 0x8000_0000);
        assert_eq!(fadd(0x3F80_0000, 0xBF80_0000), 0);
        assert_eq!(fadd(0x7F7F_FFFF, 0x7F7F_FFFF), POS_INF);
        assert_eq!(fadd(0x8080_0001, 0x0080_0000), 0x8000_0000);
        assert_eq!(fmul(0x00FF_FFFF, 0x3F00_0001), 0x0080_0000);
        assert_eq!(ffma(0x3F80_0001, 0x3F80_0001, 0xBF80_0002), 0x2880_0000);
        assert_eq!(fmin(0, 0x8000_0000), 0x8000_0000);
        assert_eq!(i2f(0x0100_0001), 0x4B80_0000);
        assert_eq!(f2i(0x4F00_0000), 0x7FFF_FFFF);
        assert_eq!(f2i(0xBFC0_0000), 0xFFFF_FFFF);
        assert!(fcmp(FCond::Eq, 0, 0x8000_0001));
    }

    #[test]
    fn far_apart_addends_round_correctly() {
        // 1 + 2^-100: the tiny addend only acts as a sticky bit.
        assert_eq!(fadd(0x3F80_0000, 0x0D80_0000), 0x3F80_0000);
        assert_eq!(fsub(0x3F80_0000, 0x0D80_0000), 0x3F80_0000);
        // (1 + 2^-23) * 1.5 lies exactly on a rounding midpoint; a +-2^-100 addend, far
        // below the jam threshold, must still decide the direction.
        assert_eq!(ffma(0x3F80_0001, 0x3FC0_0000, 0), 0x3FC0_0002); // tie -> even
        assert_eq!(ffma(0x3F80_0001, 0x3FC0_0000, 0x0D80_0000), 0x3FC0_0002);
        assert_eq!(ffma(0x3F80_0001, 0x3FC0_0000, 0x8D80_0000), 0x3FC0_0001);
    }
}
