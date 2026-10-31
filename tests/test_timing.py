"""Cycle-level timing model (docs/uarch/core.md section 11)."""

import numpy as np
import pytest

from mgpu.asm import assemble
from mgpu.device import Device, rust_available

pytestmark = pytest.mark.skipif(not rust_available(), reason="mgpu._core not built")


def timed(src, grid=1, block=32, args=(), **timing):
    dev = Device("rust", timing=timing, mem_size=4 << 20)
    res = dev.run(assemble(src), grid, block, *args)
    assert res.ok
    return dev, dev.timing_report


def test_scalar_chain_exact():
    # cold I-cache miss: buffer valid at 113; 4 cycles per SALU instruction; EXIT done +3
    for n in (0, 3, 10):
        _, rep = timed("S_ADDI s1, s1, 1\n" * n + "EXIT")
        assert rep["cycles"] == 116 + 4 * n


def test_dependent_vector_chain_exact():
    # VALU: 4 passes + 2 latency -> issue every 8 cycles for a single warp
    for n in (1, 5):
        _, rep = timed("V_ADDI v1, v1, 1\n" * n + "EXIT")
        assert rep["cycles"] == 116 + 8 * n


def test_branch_costs_refetch():
    # a branch fetches its target at completion (buffer valid at c+2 instead of before c+1):
    # one extra cycle compared with a sequential instruction
    _, seq = timed("S_ADDI s1, s1, 1\nS_ADDI s1, s1, 1\nEXIT")
    _, br = timed("S_J next\nnext: S_ADDI s1, s1, 1\nEXIT")
    assert br["cycles"] == seq["cycles"] + 1


def test_clock_reads_cycles():
    dev, _ = timed("""S_GETSR s1, CLOCK
                      S_ADDI s5, s0, 1
                      S_ADDI s5, s0, 1
                      S_ADDI s5, s0, 1
                      S_GETSR s2, CLOCK
                      S_SUB s3, s2, s1
                      S_SW s3, (s9)
                      EXIT""".replace("(s9)", "(s1)").replace("S_SW s3, (s1)", "S_LI s4, 0x8000\nS_SW s3, (s4)"))
    assert dev.read_words(0x8000, 1)[0] == 16


def test_valu_throughput_saturates_with_warps():
    src = "V_ADDI v1, v1, 1\n" * 64 + "EXIT"
    _, one = timed(src, block=32)
    _, eight = timed(src, block=256)
    # 8 warps x 64 instructions, VALU occupied 4 cycles each: bounded by 2048 busy cycles
    assert eight["busy_by_unit"]["valu"] == 8 * 64 * 4
    assert eight["busy_by_unit"]["valu"] / eight["cycles"] > 0.85
    # one warp issues every 8 cycles (4 passes + latency 2), so the VALU idles half the time;
    # eight warps keep it busy: 8x the work in under 3x the time
    assert eight["cycles"] < 3 * one["cycles"]


LOADS = """
        V_GETSR v1, TID
        V_SHLI  v2, v1, 2
        V_ADD   v2, v2, s1
        S_ADDI  s10, s0, 8
loop:   V_LW    v3, (v2)
        V_ADD   v4, v4, v3
        S_ADDI  s10, s10, -1
        S_BNZ   s10, loop
        EXIT
"""


def test_more_warps_hide_memory_latency():
    cyc = {}
    for warps in (1, 2, 4, 8):
        _, rep = timed(LOADS, grid=8 // warps, block=32 * warps, args=[0x10000])
        cyc[warps] = rep["cycles"]
    assert cyc[1] > cyc[2] > cyc[4] > cyc[8]
    assert cyc[1] > 3 * cyc[8]


def test_memory_latency_parameter():
    _, fast = timed(LOADS, args=[0x10000], mem_latency=20)
    _, slow = timed(LOADS, args=[0x10000], mem_latency=200)
    assert slow["cycles"] > fast["cycles"] + 8 * 150


@pytest.mark.parametrize("stride, segs", [(1, 1), (2, 2), (32, 32)])
def test_coalescing(stride, segs):
    _, rep = timed(f"""V_GETSR v1, LANE_ID
                       V_SHLI v2, v1, {2 + stride.bit_length() - 1}
                       V_ADD v2, v2, s1
                       V_LW v3, (v2)
                       EXIT""", args=[0x10000])
    assert rep["mem_reads"] == 1 + segs                    # + the I-cache fill


@pytest.mark.parametrize("stride, degree", [(1, 1), (2, 2), (32, 32)])
def test_shared_bank_conflicts(stride, degree):
    _, rep = timed(f"""V_GETSR v1, LANE_ID
                       V_SHLI v2, v1, {2 + stride.bit_length() - 1}
                       V_STS v1, (v2)
                       EXIT""")
    assert rep["shared_conflict_cycles"] == degree - 1
    assert rep["busy_by_unit"]["lsu"] == 4 + degree


def test_more_cores_finish_sooner():
    src = "V_ADDI v1, v1, 1\n" * 32 + "EXIT"
    _, one = timed(src, grid=8, block=64, cores=1)
    _, four = timed(src, grid=8, block=64, cores=4)
    assert four["cycles"] < one["cycles"] / 2.5


def test_lanes_parameter():
    src = "V_ADDI v1, v1, 1\n" * 64 + "EXIT"
    _, l8 = timed(src, block=256, lanes=8)
    _, l32 = timed(src, block=256, lanes=32)
    assert l32["cycles"] < l8["cycles"] / 2


def test_block_dim_limited_by_warp_slots():
    dev = Device("rust", timing={"warps": 4}, mem_size=1 << 20)
    res = dev.run(assemble("EXIT"), 1, 160, check=False)
    assert res.status == 5 and dev.timing_report["cycles"] == 1


def test_model_backend_has_no_timing():
    with pytest.raises(ValueError):
        Device("model", timing={})


def test_timed_kernels_match_untimed():
    """Every kernel produces identical memory with and without the timing model."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "kernels"))
    from kernutil import kernel
    rng = np.random.default_rng(0)
    n = 1500
    cases = {
        "saxpy": (12, 128, lambda d: [n, 1.5, d.upload(rng.standard_normal(n).astype(np.float32)),
                                      d.upload(np.ones(n, np.float32))]),
        "reduce": (12, 128, lambda d: [d.upload(np.arange(n, dtype=np.float32)), d.zeros(12), n]),
        "warp_reduce": (6, 256, lambda d: [d.upload(np.arange(n, dtype=np.int32)), d.zeros(1, "int32"), n]),
        "collatz": (12, 128, lambda d: [d.upload(rng.integers(1, 999, n).astype(np.uint32)), d.zeros(n, "uint32"), n]),
        "histogram": (3, 96, lambda d: [d.upload(rng.integers(0, 64, n).astype(np.uint32)), d.zeros(64, "uint32"), n]),
        "sigmoid": (12, 128, lambda d: [n, d.upload(rng.standard_normal(n).astype(np.float32)), d.zeros(n)]),
        "matmul": (6, 256, lambda d: [d.upload(np.ones((40, 24), np.float32)), d.upload(np.ones((24, 40), np.float32)), d.zeros((40, 40)), 40, 40, 24]),
        "conv3x3": (16, 64, lambda d: [d.upload(rng.standard_normal((16, 64)).astype(np.float32)), d.zeros((16, 64)), d.upload(np.ones(9, np.float32)), 64, 16]),
    }
    for name, (grid, block, mk) in cases.items():
        images = []
        for timing in (None, {"cores": 2}):
            dev = Device("rust", timing=timing, mem_size=4 << 20)
            state = rng.bit_generator.state
            args = mk(dev)
            rng.bit_generator.state = state
            dev.run(kernel(name), grid, block, *args)
            images.append(dev.read(0, 4 << 20))
        assert images[0] == images[1], name
