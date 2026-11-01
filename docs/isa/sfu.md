# MGPU-32 SFU algorithm (normative)

This document defines **bit-exactly** the results of the four special-function instructions:
`V_FRCP` (≈ 1/x), `V_FRSQ` (≈ 1/√x), `V_FEX2` (≈ 2ˣ) and `V_FLG2` (≈ log₂ x).

Every conforming implementation must return exactly the bits produced by the algorithm below. The
original design note required only "at least 22 bits of accuracy", which allowed two correct
implementations to disagree; see clarification C-01.

The executable reference is `python/mgpu/fp/sfu.py`. The coefficient tables in `spec/sfu/*.txt` are
**normative data**. They were produced by `tools/gen_sfu_tables.py`, which uses only `Decimal` and
integer arithmetic, so the same tables are generated on every machine.

## 1. Measured accuracy

Measured with `tools/sfu_accuracy.py`, which runs every mantissa of the listed exponents:

| Operation | Max error | Min relative accuracy | Correctly rounded |
|---|---|---|---|
| FRCP | 0.767 ULP | 23.88 bits | 95.3 % |
| FRSQ | 0.610 ULP | 23.91 bits | 98.0–98.4 % |
| FLG2 | 0.577 ULP | 23.80 bits | 98.5–99.98 % |
| FEX2 | 0.532 ULP | 23.92 bits | 98.4–100 % |

Every result is within 1 ULP, so all four operations comfortably meet the original ≥ 22-bit target.
Results are exact for the special cases in §5.

## 2. Notation

All quantities are unbounded integers unless stated otherwise.

- `>>` is an arithmetic shift right, which rounds toward −∞ (floor).
- `round_fixed(s, M, k)` rounds the exact value (−1)ˢ · M · 2ᵏ, for an integer M ≥ 0, to binary32
  following rule [FP-3] of the ISA: nearest-even at 24 bits with an unbounded exponent, then
  overflow → ±∞ and |result| < 2⁻¹²⁶ → ±0. M = 0 gives ±0.
- For a finite, non-zero input x (after flushing), `e` = biased exponent − 127 and `f` = the 23-bit
  fraction field, so x = (−1)ˢ · (1 + f·2⁻²³) · 2ᵉ.

## 3. Shared table datapath

Each table describes a function F(t) on t ∈ [0, 1), split into **128 segments**. The input is an
unsigned integer T of width `W` bits, with t = T · 2⁻ᵂ. Let `D = W − 7`.

```
i  = T >> D                    ; segment index, 0..127
d  = T & (2^D − 1)             ; offset inside the segment
(C0, C1, C2) = TABLE[i]
sq = (d * d) >> (2*D − 16)
Y  = C0 + ((C1 * d) >> (D − 1)) + ((C2 * sq) >> 16)
```

`Y` approximates F(t) · 2³⁰. The coefficient formats are:

| Coefficient | Format | Scale |
|---|---|---|
| C0 | unsigned, 32 bits | × 2⁻³⁰ |
| C1 | signed two's complement, 24 bits | × 2⁻²² |
| C2 | signed two's complement, 18 bits | × 2⁻¹⁶ |

In the table files, each row is `index C0 C1 C2` in hexadecimal, with C1 and C2 written as two's
complement in their field widths.

| Table | F(t) | W | Used by |
|---|---|---|---|
| `rcp` | 1 / (1 + t) | 23 | FRCP |
| `rsqrt_even` | 1 / √(1 + t) | 23 | FRSQ, even exponent |
| `rsqrt_odd` | 1 / √(2(1 + t)) | 23 | FRSQ, odd exponent |
| `exp2` | 2ᵗ | 30 | FEX2 |
| `log2` | log₂(1 + t) | 23 | FLG2, exponent ∉ {−1, 0} |
| `log2_near_hi` | log₂(1 + t) / t | 23 | FLG2, x ∈ (1, 2) |
| `log2_near_lo` | log₂(x) / (x − 1), x = (1 + t)/2 | 23 | FLG2, x ∈ (0.5, 1) |

The two `log2_near_*` tables compute log₂ x as (x − 1) · G(x) near x = 1. This keeps the *relative*
error small where log₂ x → 0, which a direct table of log₂ cannot do.

## 4. Operations

Every operation first flushes a denormal input to a zero of the same sign ([FP-1]).

### FRCP(x)

```
NaN            -> 0x7FC00000
±Inf           -> ±0
±0             -> ±Inf
f == 0         -> round_fixed(s, 1, −e)                     ; exact power of two
otherwise      -> round_fixed(s, Y_rcp(T = f), −30 − e)
```

### FRSQ(x)

```
NaN            -> 0x7FC00000
±0             -> ±Inf          (rsqrt(−0) = −Inf)
x < 0          -> 0x7FC00000     (including −Inf)
+Inf           -> +0
h = e >> 1                       ; floor(e / 2)
e odd          -> round_fixed(0, Y_rsqrt_odd(T = f), −30 − h)
e even, f == 0 -> round_fixed(0, 1, −h)                      ; exact
e even         -> round_fixed(0, Y_rsqrt_even(T = f), −30 − h)
```

### FEX2(x)

```
NaN            -> 0x7FC00000
+Inf           -> +Inf
−Inf           -> +0
±0             -> 1.0 (0x3F800000)
e >= 8         -> x > 0 ? +Inf : +0                          ; |x| >= 256
m  = 2^23 + f                                                ; |x| = m * 2^(e−23)
X  = (e + 7 >= 0) ? (m << (e + 7)) : (m >> −(e + 7))         ; floor(|x| * 2^30)
if x < 0: X = −X
n  = X >> 30                                                 ; floor
T  = X & (2^30 − 1)
T == 0         -> round_fixed(0, 1, n)                       ; exact power of two
otherwise      -> round_fixed(0, Y_exp2(T), n − 30)
```

The magnitude is truncated toward zero *before* negation. As a result, a tiny negative x such as
−2⁻⁴⁰ gives exactly 1.0, not 1 − 2⁻²⁴. Overflow (x ≥ 128) and underflow (x < −126, after rounding) are
handled by `round_fixed`.

### FLG2(x)

```
NaN            -> 0x7FC00000
±0             -> −Inf
x < 0          -> 0x7FC00000     (including −Inf)
+Inf           -> +Inf
f == 0         -> round_fixed(e < 0, |e|, 0)                 ; exact integer; log2(1) = +0
e == 0         -> round_fixed(0, f * Y_log2_near_hi(T = f), −53)
e == −1        -> round_fixed(1, (2^23 − f) * Y_log2_near_lo(T = f), −54)
otherwise:
    V = e * 2^30 + Y_log2(T = f)
    -> round_fixed(V < 0, |V|, −30)
```

## 5. Special-case summary

| Input | FRCP | FRSQ | FEX2 | FLG2 |
|---|---|---|---|---|
| NaN (any) | NaN | NaN | NaN | NaN |
| +∞ | +0 | +0 | +∞ | +∞ |
| −∞ | −0 | NaN | +0 | NaN |
| +0 / +denormal | +∞ | +∞ | 1.0 | −∞ |
| −0 / −denormal | −∞ | −∞ | 1.0 | −∞ |
| negative finite | finite | NaN | finite | NaN |
| ±2ᵏ | ±2⁻ᵏ exact (or ±0 if k = 127) | 2^(−k/2) exact for even k | — | k exact |
| integer x | — | — | 2ˣ exact (0 or +∞ outside range) | — |
| 1.0 | 1.0 | 1.0 | 2.0 | +0 |

"NaN" here always means the canonical NaN `0x7FC00000`.

## 6. How the tables were made *(informative)*

For each segment, F is interpolated by a quadratic in the local offset u at the three Chebyshev nodes
of the segment. These values are computed in 60-digit `Decimal`. C1 and C2 are rounded to their
formats. C0 is then re-centred: the error of the exact integer datapath is measured at 65 sample
points in the segment, and C0 is shifted by the midpoint of the minimum and maximum error. Only the
committed tables are normative. `tools/gen_sfu_tables.py --check` confirms that they reproduce.

## 7. Hardware cost *(informative)*

- ROM: 7 tables × 128 entries × 74 bits ≈ 65 kbit. An implementation may merge the tables into one
  ROM, indexed by {table, segment}.
- Multipliers: one 24 × 23 (C1 · d), one 18 × 16 (C2 · sq), one 23 × 23 squarer for d · d (only the
  top 16 bits of the result are needed), and, for FLG2 near 1, one 24 × 32 multiply by `f` or
  `2^23 − f`.
- The final normalize-and-round step is the same `round_fixed` used by the rest of the FPU.

A pipelined SFU can process one lane per cycle per datapath; the microarchitecture fixes the number
of datapaths and the latency.
