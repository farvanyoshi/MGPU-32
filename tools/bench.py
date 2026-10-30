#!/usr/bin/env python3
"""Compare simulator throughput: pure-Python model vs Rust engine.

Runs an FFMA/integer loop kernel (ALU-bound) and SAXPY (memory-bound) and reports
warp-instructions per second for each backend.

Usage: bench.py [--blocks N] [--iters N]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from mgpu.device import Device, rust_available  # noqa: E402
from mgpu.isa import assemble_one  # noqa: E402

CODE, DATA = 0x1000, 0x10000


def program(lines):
    return [assemble_one(x) for x in lines]


LOOP = program(["V_GETSR v1, GTID", "V_I2F v2, v1", "S_ADD s10, s3, s0",
                "V_FFMA v3, v2, s2, v3",          # loop body: 4 instructions
                "V_ADDI v4, v4, 1",
                "S_ADDI s10, s10, -1",
                "S_BNZ s10, -4",
                "V_SHLI v5, v1, 2", "V_ADD v5, v5, s1", "V_SW v3, 0(v5)", "EXIT"])

SAXPY = program(["V_GETSR v1, GTID", "V_CMP.LT s10, v1, s1", "S_ADD s31, s10, s0", "S_BZ s31, 7",
                 "V_SHLI v2, v1, 2", "V_ADD v3, v2, s3", "V_ADD v4, v2, s4", "V_LW v5, 0(v3)",
                 "V_LW v6, 0(v4)", "V_FFMA v6, v5, s2, v6", "V_SW v6, 0(v4)", "EXIT"])


def run(backend, words, grid, block, args):
    dev = Device(backend, mem_size=8 << 20)
    dev.write_words(CODE, words)
    t = time.perf_counter()
    res = dev.launch(CODE, grid, block, args)
    dt = time.perf_counter() - t
    assert res.ok, res
    return res.steps, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    cases = [("ffma loop", LOOP, args.blocks, 256, [DATA, 0x3F800000, args.iters]),
             ("saxpy", SAXPY, args.blocks * 8, 256, [args.blocks * 8 * 256, 0x40000000, DATA, DATA + (2 << 20)])]
    backends = ["model"] + (["rust"] if rust_available() else [])
    print(f"{'kernel':12} {'backend':8} {'warp-instr':>11} {'seconds':>9} {'instr/s':>12} {'lane-ops/s':>12}")
    for name, words, grid, block, kargs in cases:
        rates = {}
        for b in backends:
            steps, dt = run(b, words, grid, block, kargs)
            rates[b] = steps / dt
            print(f"{name:12} {b:8} {steps:11d} {dt:9.3f} {steps / dt:12,.0f} {32 * steps / dt:12,.0f}")
        if len(rates) == 2:
            print(f"{'':12} speedup  {rates['rust'] / rates['model']:.0f}x")


if __name__ == "__main__":
    main()
