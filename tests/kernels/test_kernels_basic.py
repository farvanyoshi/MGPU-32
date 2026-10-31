"""saxpy, reduce, warp_reduce, collatz, histogram and sigmoid against numpy."""

import numpy as np
import pytest

from mgpu.fp import ops, sfu

from kernutil import bits, device, f32, kernel, size


@pytest.mark.parametrize("block", [32, 128])
def test_saxpy(backend, block):
    n = size(backend, 200, 5000)
    rng = np.random.default_rng(1)
    x, y = f32(rng.standard_normal(n)), f32(rng.standard_normal(n))
    a = 1.75
    dev = device(backend)
    xb, yb = dev.upload(x), dev.upload(y)
    dev.run(kernel("saxpy"), -(-n // block), block, n, a, xb, yb)
    # bit-exact reference: one fused rounding per element
    want = [ops.ffma(int(xi), int(bits(a)), int(yi)) for xi, yi in zip(bits(x), bits(y))]
    assert bits(yb.read()).tolist() == want
    assert np.allclose(yb.read(), a * x.astype(np.float64) + y, rtol=1e-6)


@pytest.mark.parametrize("block", [32, 64, 256])
def test_reduce(backend, block):
    n = size(backend, 300, 20000)
    data = f32(np.random.default_rng(2).integers(-1000, 1000, n))   # exact float sums
    grid = -(-n // block)
    dev = device(backend)
    inb, outb = dev.upload(data), dev.zeros(grid)
    dev.run(kernel("reduce"), grid, block, inb, outb, n)
    want = [data[b * block:(b + 1) * block].sum() for b in range(grid)]
    assert np.array_equal(outb.read(), f32(want))


@pytest.mark.parametrize("block", [32, 40, 96])
def test_warp_reduce(backend, block):
    n = size(backend, 333, 50000)
    data = np.random.default_rng(3).integers(-2**31, 2**31, n, dtype=np.int64).astype(np.int32)
    dev = device(backend)
    inb, out = dev.upload(data), dev.zeros(1, "int32")
    grid = -(-n // block) + 1                     # one spare block: every lane out of range
    dev.run(kernel("warp_reduce"), grid, block, inb, out, n)
    assert out.read()[0] == np.int32(data.astype(np.int64).sum() & 0xFFFFFFFF)


def collatz_steps(x: int) -> int:
    n = 0
    while x > 1:
        x = x // 2 if x % 2 == 0 else 3 * x + 1
        n += 1
    return n


def test_collatz(backend):
    n = size(backend, 64, 4000)
    data = np.random.default_rng(4).integers(1, 5000, n).astype(np.uint32)
    data[:3] = [1, 2, 27]                          # 27 takes 111 steps
    dev = device(backend, trace=backend[1] == "small")
    inb, outb = dev.upload(data), dev.zeros(n, "uint32")
    dev.run(kernel("collatz"), -(-n // 32), 32, inb, outb, n)
    assert outb.read().tolist() == [collatz_steps(int(v)) for v in data]


@pytest.mark.parametrize("grid, block", [(1, 32), (3, 64), (2, 100)])
def test_histogram(backend, grid, block):
    n = size(backend, 500, 30000)
    data = np.random.default_rng(5).integers(0, 1 << 20, n).astype(np.uint32)
    dev = device(backend)
    inb, hist = dev.upload(data), dev.zeros(64, "uint32")
    dev.run(kernel("histogram"), grid, block, inb, hist, n)
    assert np.array_equal(hist.read(), np.bincount(data & 63, minlength=64))


def test_sigmoid(backend):
    n = size(backend, 100, 4000)
    x = f32(np.concatenate([[0, 1, -1, 20, -20, 100, -100, 1e-30],
                            np.random.default_rng(6).uniform(-12, 12, n - 8)]))
    dev = device(backend)
    xb, yb = dev.upload(x), dev.zeros(n)
    dev.run(kernel("sigmoid"), -(-n // 64), 64, n, xb, yb)
    got = yb.read()
    # bit-exact reference: the same operation sequence with the golden float ops
    one, log2e = int(bits(1.0)), int(bits(1.4426950408889634))
    want = []
    for xi in bits(x):
        e = sfu.fex2(ops.fmul(int(xi) ^ 0x80000000, log2e))
        want.append(ops.fmul(one, sfu.frcp(ops.fadd(e, one))))
    assert bits(got).tolist() == want
    ref = 1 / (1 + np.exp(-x.astype(np.float64)))
    assert np.allclose(got, ref, rtol=3e-6, atol=1e-30)
