#!/usr/bin/env python3
"""Design-space sweep with the timing model: kernel suite x warp slots x lanes x memory latency.

Prints markdown tables (used in docs/uarch/performance.md).

Usage: sweep.py [--quick]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mgpu.asm import assemble_file  # noqa: E402
from mgpu.device import Device  # noqa: E402

N = 8192
M = 64  # matmul / conv size


def kernels():
    rng = np.random.default_rng(0)
    xs = rng.standard_normal(N).astype(np.float32)
    ints = rng.integers(1, 3000, N).astype(np.uint32)
    img = rng.standard_normal((M, M)).astype(np.float32)
    A = rng.standard_normal((M, M)).astype(np.float32)
    # name -> (block(W) -> block size or None, grid(block), args(device))
    blk = lambda W: min(1024, 32 * W)  # noqa: E731  (one resident block fills every slot)
    return {
        "saxpy": (blk, lambda b: -(-N // b), lambda d: [N, 2.0, d.upload(xs), d.upload(xs)]),
        "reduce": (blk, lambda b: -(-N // b), lambda d: [d.upload(xs), d.zeros(N // 32), N]),
        "warp_reduce": (blk, lambda b: -(-N // b), lambda d: [d.upload(ints), d.zeros(1, "int32"), N]),
        "collatz": (blk, lambda b: -(-N // b), lambda d: [d.upload(ints), d.zeros(N, "uint32"), N]),
        "histogram": (blk, lambda b: 4, lambda d: [d.upload(ints), d.zeros(64, "uint32"), N]),
        "sigmoid": (blk, lambda b: -(-N // b), lambda d: [N, d.upload(xs), d.zeros(N)]),
        "matmul": (lambda W: 256 if W >= 8 else None, lambda b: (M // 16) ** 2,
                   lambda d: [d.upload(A), d.upload(A), d.zeros((M, M)), M, M, M]),
        "conv3x3": (lambda W: M if 32 * W >= M else None, lambda b: M,
                    lambda d: [d.upload(img), d.zeros((M, M)), d.upload(np.ones(9, np.float32)), M, M]),
    }


def run(name, spec, **timing):
    blockf, gridf, argf = spec
    block = blockf(timing.get("warps", 8))
    if block is None:
        return None
    dev = Device("rust", timing=timing, mem_size=8 << 20)
    prog = assemble_file(ROOT / "kernels" / f"{name}.s")
    dev.run(prog, gridf(block), block, *argf(dev))
    return dev.timing_report


def table(title, header, rows):
    out = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="fewer kernels")
    args = ap.parse_args()
    ks = kernels()
    if args.quick:
        ks = {k: ks[k] for k in ("saxpy", "matmul", "collatz")}

    # 1. warp slots (latency hiding): default lanes and memory
    rows = []
    for name, spec in ks.items():
        row = [f"`{name}`"]
        for W in (2, 4, 8, 16):
            r = run(name, spec, warps=W)
            row.append("—" if r is None else f"{r['cycles']:,} ({100 * r['issue_utilization']:.0f}%)")
        rows.append(row)
    print(table("Cycles vs warp slots (issue-slot utilization in parentheses); L = 8, T_mem = 100",
                ["kernel", "W = 2", "W = 4", "W = 8", "W = 16"], rows))

    # 2. physical lanes: datapath width vs cycles at W = 8
    rows = []
    for name, spec in ks.items():
        base = run(name, spec, lanes=8)["cycles"]
        row = [f"`{name}`", f"{base:,}"]
        for L in (16, 32):
            c = run(name, spec, lanes=L)["cycles"]
            row.append(f"{c:,} ({base / c:.2f}x)")
        rows.append(row)
    print(table("Cycles vs physical lanes (speedup over L = 8); W = 8, T_mem = 100",
                ["kernel", "L = 8", "L = 16", "L = 32"], rows))

    # 3. memory latency sensitivity at W = 8
    rows = []
    for name, spec in ks.items():
        cs = [run(name, spec, mem_latency=t)["cycles"] for t in (50, 100, 200, 400)]
        rows.append([f"`{name}`"] + [f"{c:,}" for c in cs] + [f"{cs[-1] / cs[0]:.2f}x"])
    print(table("Cycles vs memory latency; W = 8, L = 8",
                ["kernel", "50", "100", "200", "400", "400 vs 50"], rows))

    # 4. where the cycles go at the default design point
    rows = []
    for name, spec in ks.items():
        r = run(name, spec)
        act = max(1, r["core_active_cycles"])
        busy = r["busy_by_unit"]
        st = r["stalls"]
        rows.append([f"`{name}`", f"{r['cycles']:,}", f"{r['ipc']:.2f}",
                     f"{100 * r['simd_utilization']:.0f}%",
                     " / ".join(f"{100 * busy[u] / act:.0f}" for u in ("valu", "fpu", "sfu", "lsu")),
                     ", ".join(f"{k} {100 * v / act:.0f}%" for k, v in sorted(st.items(), key=lambda kv: -kv[1]) if 100 * v / act >= 5)])
    print(table("Default design point (W = 8, L = 8, T_mem = 100): where the cycles go",
                ["kernel", "cycles", "IPC", "SIMD util.", "busy % valu/fpu/sfu/lsu", "idle cycles (>= 5%)"], rows))


if __name__ == "__main__":
    main()
