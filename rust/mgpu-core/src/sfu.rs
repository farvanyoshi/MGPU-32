//! Bit-exact SFU operations (docs/isa/sfu.md). Mirrors python/mgpu/fp/sfu.py.

use crate::fp::{
    ftz, is_inf, is_nan, is_zero, round_fixed, sign_of, signed_inf, signed_zero, CANON_NAN,
    NEG_INF, POS_INF,
};
use crate::sfu_tables::{Row, EXP2, LOG2, LOG2_NEAR_HI, LOG2_NEAR_LO, RCP, RSQRT_EVEN, RSQRT_ODD};

const SEG_BITS: u32 = 7;
const Y_FRAC: i32 = 30;
const ONE: u32 = 0x3F80_0000;

/// The shared table datapath: Y ~= F(t) * 2^30 for t = T * 2^-t_bits.
#[inline]
fn poly(table: &[Row; 128], t: u64, t_bits: u32) -> i64 {
    let d_bits = t_bits - SEG_BITS;
    let i = (t >> d_bits) as usize;
    let d = (t & ((1u64 << d_bits) - 1)) as i64;
    let (c0, c1, c2) = table[i];
    let sq = (d * d) >> (2 * d_bits - 16);
    c0 as i64 + ((c1 as i64 * d) >> (d_bits - 1)) + ((c2 as i64 * sq) >> 16)
}

#[inline]
fn fields(x: u32) -> (i32, u64) {
    (((x >> 23) & 0xFF) as i32 - 127, (x & 0x7F_FFFF) as u64)
}

pub fn frcp(x: u32) -> u32 {
    let x = ftz(x);
    let s = sign_of(x);
    if is_nan(x) {
        return CANON_NAN;
    }
    if is_inf(x) {
        return signed_zero(s);
    }
    if is_zero(x) {
        return signed_inf(s);
    }
    let (e, f) = fields(x);
    if f == 0 {
        return round_fixed(s, 1, -e);
    }
    round_fixed(s, poly(&RCP, f, 23) as u128, -Y_FRAC - e)
}

pub fn frsq(x: u32) -> u32 {
    let x = ftz(x);
    let s = sign_of(x);
    if is_nan(x) {
        return CANON_NAN;
    }
    if is_zero(x) {
        return signed_inf(s);
    }
    if s != 0 {
        return CANON_NAN;
    }
    if is_inf(x) {
        return 0;
    }
    let (e, f) = fields(x);
    let half = e >> 1;
    if e & 1 != 0 {
        return round_fixed(0, poly(&RSQRT_ODD, f, 23) as u128, -Y_FRAC - half);
    }
    if f == 0 {
        return round_fixed(0, 1, -half);
    }
    round_fixed(0, poly(&RSQRT_EVEN, f, 23) as u128, -Y_FRAC - half)
}

pub fn fex2(x: u32) -> u32 {
    let x = ftz(x);
    let s = sign_of(x);
    if is_nan(x) {
        return CANON_NAN;
    }
    if is_inf(x) {
        return if s != 0 { 0 } else { POS_INF };
    }
    if is_zero(x) {
        return ONE;
    }
    let (e, f) = fields(x);
    if e >= 8 {
        return if s != 0 { 0 } else { POS_INF };
    }
    let mant = (1u64 << 23) | f;
    let sh = e + 7;
    let mag = if sh >= 0 {
        (mant << sh) as i64
    } else if -sh >= 64 {
        0
    } else {
        (mant >> -sh) as i64
    };
    let xf = if s != 0 { -mag } else { mag };
    let n = (xf >> 30) as i32;
    let t = (xf & ((1i64 << 30) - 1)) as u64;
    if t == 0 {
        return round_fixed(0, 1, n);
    }
    round_fixed(0, poly(&EXP2, t, 30) as u128, n - Y_FRAC)
}

pub fn flg2(x: u32) -> u32 {
    let x = ftz(x);
    let s = sign_of(x);
    if is_nan(x) {
        return CANON_NAN;
    }
    if is_zero(x) {
        return NEG_INF;
    }
    if s != 0 {
        return CANON_NAN;
    }
    if is_inf(x) {
        return POS_INF;
    }
    let (e, f) = fields(x);
    if f == 0 {
        return round_fixed((e < 0) as u32, e.unsigned_abs() as u128, 0);
    }
    if e == 0 {
        let g = poly(&LOG2_NEAR_HI, f, 23) as u128;
        return round_fixed(0, f as u128 * g, -23 - Y_FRAC);
    }
    if e == -1 {
        let g = poly(&LOG2_NEAR_LO, f, 23) as u128;
        return round_fixed(1, ((1u128 << 23) - f as u128) * g, -24 - Y_FRAC);
    }
    let v = ((e as i64) << Y_FRAC) + poly(&LOG2, f, 23);
    round_fixed((v < 0) as u32, v.unsigned_abs() as u128, -Y_FRAC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_cases() {
        assert_eq!(frcp(0x4080_0000), 0x3E80_0000); // 1/4
        assert_eq!(frsq(0x4080_0000), 0x3F00_0000); // 1/sqrt(4)
        assert_eq!(frsq(0x7F00_0000), 0x1FB5_04F3); // 2^127 (odd exponent)
        assert_eq!(fex2(0x3F80_0000), 0x4000_0000); // 2^1
        assert_eq!(fex2(0xC316_0000), 0); // 2^-150
        assert_eq!(fex2(0xAB80_0000), ONE); // -2^-40 -> exactly 1
        assert_eq!(flg2(0x3F80_0000), 0); // log2(1)
        assert_eq!(flg2(0x3F00_0000), 0xBF80_0000); // log2(0.5) = -1
        assert_eq!(frcp(0xC316_0000), 0xBBDA_740E); // 1/-150
        assert_eq!(flg2(0x8000_0001), NEG_INF);
    }
}
