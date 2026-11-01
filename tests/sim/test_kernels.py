"""The example kernels from ISA section 15, checked against numpy."""

import numpy as np
import pytest

from simutil import DATA, f2b

SAXPY = """
saxpy:  V_GETSR   v1, GTID
        V_CMP.LT  s10, v1, s1
        S_MOV     s31, s10
        S_BZ      s31, end
        V_SHLI    v2, v1, 2
        V_ADD     v3, v2, s3
        V_ADD     v4, v2, s4
        V_LW      v5, 0(v3)
        V_LW      v6, 0(v4)
        V_FFMA    v6, v5, s2, v6
        V_SW      v6, 0(v4)
end:    EXIT
"""

REDUCE = """
reduce: V_GETSR   v1, GTID
        V_GETSR   v2, TID
        S_MOV     s11, s31
        V_MOV     v4, v0
        V_CMP.LT  s10, v1, s3
        S_MOV     s31, s10
        V_SHLI    v3, v1, 2
        V_ADD     v3, v3, s1
        V_LW      v4, 0(v3)
        S_MOV     s31, s11
        V_SHLI    v5, v2, 2
        V_STS     v4, 0(v5)
        BAR
        S_GETSR   s12, BLOCK_DIM
        S_SHRI    s12, s12, 1
loop:   S_BZ      s12, done
        V_CMP.LT  s10, v2, s12
        S_MOV     s31, s10
        S_SHLI    s13, s12, 2
        V_ADD     v6, v5, s13
        V_LDS     v7, 0(v6)
        V_LDS     v8, 0(v5)
        V_FADD    v8, v8, v7
        V_STS     v8, 0(v5)
        S_MOV     s31, s11
        BAR
        S_SHRI    s12, s12, 1
        S_J       loop
done:   V_CMP.EQ  s10, v2, v0
        S_MOV     s31, s10
        S_BZ      s31, end
        V_LDS     v8, 0(v0)
        S_GETSR   s14, BLOCK_ID
        S_SHLI    s14, s14, 2
        S_ADD     s14, s14, s2
        V_BCAST   v9, s14
        V_SW      v8, 0(v9)
end:    EXIT
"""


@pytest.mark.parametrize("n, block", [(40, 32), (1000, 128), (64, 64)])
def test_saxpy(run, n, block):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    a = np.float32(2.5)
    xa, ya = DATA, DATA + 4 * n
    grid = -(-n // block)
    r = run(SAXPY, grid=grid, block=block, args=[n, f2b(float(a)), xa, ya],
            data={xa: x.view(np.uint32).tolist(), ya: y.view(np.uint32).tolist()}, keep_state=False)
    assert r.res.ok
    got = r.dev.read_array(ya, n)
    # FFMA is fused: compute the reference in float64 and round once (exact for these inputs)
    want = (np.float64(a) * x.astype(np.float64) + y).astype(np.float32)
    assert np.array_equal(got, want)


@pytest.mark.parametrize("n, block", [(100, 64), (256, 128), (1, 32)])
def test_block_reduction(run, n, block):
    rng = np.random.default_rng(7)
    data = rng.integers(-1000, 1000, n).astype(np.float32)   # exact float sums
    grid = -(-n // block)
    ina, outa = DATA, DATA + 4 * n + 64
    r = run(REDUCE, grid=grid, block=block, args=[ina, outa, n],
            data={ina: data.view(np.uint32).tolist()}, keep_state=False)
    assert r.res.ok
    got = r.dev.read_array(outa, grid)
    want = np.array([data[b * block:(b + 1) * block].sum() for b in range(grid)], dtype=np.float32)
    assert np.array_equal(got, want)
