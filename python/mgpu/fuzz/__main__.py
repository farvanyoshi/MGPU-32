"""Fuzzing CLI.

    python -m mgpu.fuzz --seeds 2000                 # model vs Rust, all modes
    python -m mgpu.fuzz --mode eq --seeds 500        # schedule-independence checks
    python -m mgpu.fuzz --mode diff --start 1234 --seeds 1 --show   # replay one case
"""

import argparse
import multiprocessing as mp
import sys
import time

from .gen import random_case, raw_case
from .runner import diff_case, equiv_case

MODES = ("diff", "raw", "eq-diff", "eq")


def check(job):
    mode, seed = job
    if mode == "diff":
        return diff_case(random_case(seed, "diff"))
    if mode == "raw":
        return diff_case(raw_case(seed))
    if mode == "eq-diff":                     # well-defined programs, model vs Rust
        return diff_case(random_case(seed, "eq"))
    return equiv_case(random_case(seed, "eq"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mgpu.fuzz", description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=200, help="seeds per mode")
    ap.add_argument("--start", type=int, default=0, help="first seed")
    ap.add_argument("--mode", choices=MODES + ("all",), default="all")
    ap.add_argument("--jobs", type=int, default=mp.cpu_count())
    ap.add_argument("--show", action="store_true", help="print the program of each case")
    args = ap.parse_args(argv)
    modes = MODES if args.mode == "all" else (args.mode,)
    jobs = [(m, s) for m in modes for s in range(args.start, args.start + args.seeds)]
    if args.show:
        from ..isa import listing
        for m, s in jobs:
            c = raw_case(s) if m == "raw" else random_case(s, "eq" if m.startswith("eq") else "diff")
            print(f"--- {m} {c.describe()}\n{listing(c.words, c.entry)}")
    t0 = time.time()
    failures = 0
    with mp.Pool(args.jobs) as pool:
        for (mode, seed), report in zip(jobs, pool.imap(check, jobs, chunksize=4)):
            if report:
                failures += 1
                print(f"[{mode} seed {seed}] {report}\n", flush=True)
    print(f"{len(jobs)} cases ({', '.join(modes)}), {failures} failures, {time.time() - t0:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
