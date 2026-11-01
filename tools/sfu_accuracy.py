#!/usr/bin/env python3
"""Measure the accuracy of the MGPU-32 SFU algorithm (docs/isa/sfu.md).

This is a vectorized (numpy) copy of mgpu.fp.sfu, cross-checked against the scalar
reference on random inputs. It sweeps every mantissa for the exponents that matter and
reports the maximum error in ULPs and in bits of relative accuracy.

Usage: sfu_accuracy.py [--quick]
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from mgpu.fp import sfu  # noqa: E402
from mgpu.fp.sfu_tables import TABLES  # noqa: E402

TAB = {k: np.array(v, dtype=np.int64) for k, v in TABLES.items()}


def vpoly(name, t):
    d_bits = sfu.T_BITS[name] - sfu.SEG_BITS
    i = t >> d_bits
    d = t & ((1 << d_bits) - 1)
    c = TAB[name][i]
    sq = (d * d) >> (2 * d_bits - 16)
    return c[:, 0] + ((c[:, 1] * d) >> (d_bits - 1)) + ((c[:, 2] * sq) >> 16)


def vround(sign, m, scale):
    """Vectorized f32.round_fixed for 0 < m < 2^62 and in-range exponents."""
    m = np.asarray(m, dtype=np.int64)
    p = np.frexp(m.astype(np.float64))[1].astype(np.int64) - 1
    p = np.where((m >> p) == 0, p - 1, p)
    p = np.where((m >> (p + 1)) != 0, p + 1, p)
    sh = np.maximum(p - 23, 0)
    q = np.where(p > 23, m >> sh, m << np.maximum(23 - p, 0))
    rem = m & ((np.int64(1) << sh) - 1)
    half = np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), 0)
    up = (sh > 0) & ((rem > half) | ((rem == half) & ((q & 1) == 1)))
    q = q + up
    carry = q == (1 << 24)
    q = np.where(carry, q >> 1, q)
    e = p + carry + scale
    assert np.all((e >= -126) & (e <= 127)), "sweep left the normal range"
    return (np.asarray(sign, dtype=np.int64) << 31) | ((e + 127) << 23) | (q & 0x7FFFFF)


def bits_to_f64(b):
    return np.asarray(b, dtype=np.uint32).view(np.float32).astype(np.float64)


def vfrcp(e, f):
    y = vround(0, vpoly("rcp", f), -30 - e)
    return np.where(f == 0, vround(0, np.ones_like(f), -e), y)


def vfrsq(e, f):
    half = e >> 1
    name = "rsqrt_odd" if e & 1 else "rsqrt_even"
    y = vround(0, vpoly(name, f), -30 - half)
    if not e & 1:
        y = np.where(f == 0, vround(0, np.ones_like(f), -half), y)
    return y


def vfex2(s, e, f):
    mant = (1 << 23) | f
    sh = e + 7
    xf = mant << sh if sh >= 0 else mant >> -sh
    xf = -xf if s else xf
    n = xf >> 30
    t = xf & ((1 << 30) - 1)
    safe_t = np.where(t == 0, 1, t)
    y = vround(0, vpoly("exp2", safe_t), n - 30)
    return np.where(t == 0, vround(0, np.ones_like(t), n), y)


def vflg2(e, f):
    if e == 0:
        y = vround(0, f * vpoly("log2_near_hi", f), -53)
        sign = 0
    elif e == -1:
        y = vround(1, ((1 << 23) - f) * vpoly("log2_near_lo", f), -54)
        sign = 1
    else:
        v = (e << 30) + vpoly("log2", f)
        y = vround((v < 0).astype(np.int64), np.abs(v), -30)
        sign = int(e < 0)
    if e != 0:
        exact = vround(sign, np.full_like(f, abs(e)), 0)
        y = np.where(f == 0, exact, y)
    else:
        y = np.where(f == 0, 0, y)
    return y


def ulp_err(res_bits, ref):
    res = bits_to_f64(res_bits)
    nz = ref != 0
    ulp = np.ones_like(ref)
    ulp[nz] = np.exp2(np.floor(np.log2(np.abs(ref[nz]))) - 23)
    err = np.abs(res - ref)
    rel = np.where(nz, err / np.where(nz, np.abs(ref), 1), np.where(err == 0, 0, np.inf))
    exact = res == ref.astype(np.float32).astype(np.float64)
    return err / ulp, rel, exact


def report(label, res_bits, ref, acc):
    u, r, ex = ulp_err(res_bits, ref)
    acc.append((label, float(u.max()), float(r.max()), float(ex.mean())))


def crosscheck(n=4000):
    """Vectorized path == scalar reference."""
    rng = random.Random(1)
    for _ in range(n):
        f = rng.randrange(1 << 23)
        for e in (0, 1, -1, 5, -6):
            b = ((e + 127) << 23) | f
            fa = np.array([f], dtype=np.int64)
            assert int(vfrcp(e, fa)[0]) == sfu.frcp(b)
            assert int(vfrsq(e, fa)[0]) == sfu.frsq(b)
            assert int(vflg2(e, fa)[0]) == sfu.flg2(b)
            for s in (0, 1):
                assert int(vfex2(s, e, fa)[0]) == sfu.fex2(b | (s << 31)), (s, e, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="sample 1/64 of the mantissas")
    args = ap.parse_args()
    crosscheck(500 if args.quick else 4000)
    step = 64 if args.quick else 1
    f = np.arange(0, 1 << 23, step, dtype=np.int64)
    m = 1.0 + f / float(1 << 23)
    acc = []
    report("FRCP  x in [1,2)", vfrcp(0, f), 1.0 / m, acc)
    report("FRSQ  x in [1,2)", vfrsq(0, f), 1.0 / np.sqrt(m), acc)
    report("FRSQ  x in [2,4)", vfrsq(1, f), 1.0 / np.sqrt(2 * m), acc)
    for e in (-1, 0, 1, -2, 7, -126, 126):
        report(f"FLG2  exp {e:+d}", vflg2(e, f), np.log2(m) + e, acc)
    for s in (0, 1):
        for e in (-24, -10, -1, 0, 1, 3, 6):
            x = (-1) ** s * m * 2.0 ** e
            ok = np.abs(x) < 126  # stay in the normal range for this sweep
            res = vfex2(s, e, f[ok])
            report(f"FEX2  {'-' if s else '+'}2^{e:+d} range", res, np.exp2(x[ok]), acc)
    worst_bits = math.inf
    print(f"{'case':28} {'max ulp':>9} {'min bits':>9} {'exact %':>8}")
    for label, u, r, ex in acc:
        bits = -math.log2(r) if r > 0 else math.inf
        worst_bits = min(worst_bits, bits)
        print(f"{label:28} {u:9.3f} {bits:9.2f} {100 * ex:7.2f}%")
    print(f"worst relative accuracy: {worst_bits:.2f} bits (requirement: >= 22)")
    sys.exit(0 if worst_bits >= 22 else 1)


if __name__ == "__main__":
    main()
