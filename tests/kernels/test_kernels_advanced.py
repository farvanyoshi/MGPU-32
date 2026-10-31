"""matmul, conv3x3, scan and the bad_barrier negative test."""

import numpy as np
import pytest

from mgpu import hostregs as H
from mgpu.device import KernelFault
from mgpu.fp import ops

from kernutil import bits, device, f32, kernel, size


def matmul_ref_bits(A, B):
    """Bit-exact reference: acc = FFMA(A[r][k], B[k][c], acc) for k = 0..K-1."""
    M, K = A.shape
    N = B.shape[1]
    a, b = bits(A).tolist(), bits(B).tolist()
    out = np.zeros((M, N), np.uint32)
    for r in range(M):
        for c in range(N):
            acc = 0
            for k in range(K):
                acc = ops.ffma(a[r][k], b[k][c], acc)
            out[r, c] = acc
    return out


@pytest.mark.parametrize("M, N, K", [(16, 16, 16), (20, 24, 30), (1, 33, 17)])
def test_matmul_bit_exact(backend, M, N, K):
    rng = np.random.default_rng(M * N + K)
    A, B = f32(rng.standard_normal((M, K))), f32(rng.standard_normal((K, N)))
    dev = device(backend)
    a, b, c = dev.upload(A), dev.upload(B), dev.zeros((M, N))
    grid = -(-M // 16) * -(-N // 16)
    dev.run(kernel("matmul"), grid, 256, a, b, c, M, N, K)
    assert np.array_equal(bits(c.read()), matmul_ref_bits(A, B))


def test_matmul_large_vs_numpy(backend):
    M, N, K = (40, 40, 40) if backend[1] == "small" else (100, 130, 70)
    rng = np.random.default_rng(9)
    A = f32(rng.integers(-8, 8, (M, K)))           # integer values: exact in float32
    B = f32(rng.integers(-8, 8, (K, N)))
    dev = device(backend)
    a, b, c = dev.upload(A), dev.upload(B), dev.zeros((M, N))
    dev.run(kernel("matmul"), -(-M // 16) * -(-N // 16), 256, a, b, c, M, N, K)
    assert np.array_equal(c.read(), A @ B)


def conv_ref_bits(img, w):
    H_, W_ = img.shape
    ib, wb = bits(img).tolist(), bits(w).ravel().tolist()
    out = np.zeros((H_, W_), np.uint32)
    for y in range(H_):
        for x in range(W_):
            acc = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = min(max(y + dy, 0), H_ - 1), min(max(x + dx, 0), W_ - 1)
                    acc = ops.ffma(ib[yy][xx], wb[(dy + 1) * 3 + dx + 1], acc)
            out[y, x] = acc
    return out


def test_conv3x3(backend):
    Hh, Ww = (6, 40) if backend[1] == "small" else (40, 300)
    rng = np.random.default_rng(10)
    img = f32(rng.standard_normal((Hh, Ww)))
    w = f32([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16          # Gaussian blur
    dev = device(backend)
    i, o, wb = dev.upload(img), dev.zeros((Hh, Ww)), dev.upload(w)
    dev.run(kernel("conv3x3"), Hh, Ww, i, o, wb, Ww, Hh)
    got = o.read()
    assert np.array_equal(bits(got), conv_ref_bits(img, w))
    padded = np.pad(img.astype(np.float64), 1, mode="edge")
    ref = sum(w[dy, dx] * padded[dy:dy + Hh, dx:dx + Ww] for dy in range(3) for dx in range(3))
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("block", [32, 64, 256])
def test_scan_three_launches(backend, block):
    n = min(size(backend, 700, 60000), block * block - 5)   # two levels: GRID_DIM <= BLOCK_DIM
    data = np.random.default_rng(11).integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    grid = -(-n // block)
    dev = device(backend)
    prog = kernel("scan")
    inb, out, sums = dev.upload(data), dev.zeros(n, "uint32"), dev.zeros(grid, "uint32")
    dev.run(prog, grid, block, inb, out, sums, n, kernel="scan_block")
    dev.run(prog, 1, block, sums, sums, 0, grid, kernel="scan_block")
    dev.run(prog, grid, block, out, sums, n, kernel="scan_add")
    want = (np.cumsum(data.astype(np.uint64)) & 0xFFFFFFFF).astype(np.uint32)
    assert np.array_equal(out.read(), want)


@pytest.mark.parametrize("block", [64, 128])
def test_bad_barrier_is_reported(backend, block):
    dev = device(backend)
    prog = kernel("bad_barrier")
    with pytest.raises(KernelFault) as e:
        dev.run(prog, 1, block)
    r = e.value.result
    bars = {prog.symbols["even"], prog.symbols["odd"]}
    assert r.status == H.E_BARRIER and {r.err_pc, r.err_info} == bars


def test_bad_barrier_single_warp_passes(backend):
    assert device(backend).run(kernel("bad_barrier"), 2, 32).ok
