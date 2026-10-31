"""mgpu_core in lockstep with the Rust engine on the fuzzer's programs (mgpu.fuzz):

- diff: structured programs with every instruction, faults, illegal words and races,
        on single-warp blocks (nothing can depend on the warp schedule, so atomic values
        are compared as well);
- raw:  mostly-random instruction words, single-warp blocks;
- eq:   well-defined multi-warp programs (shared memory, barriers, atomics), compared
        under rule E4 plus final memory and STATUS.

MGPU_RTL_FUZZ_SEEDS sets the number of programs per mode, FUZZ_START the first seed and
FUZZ_MODES (comma-separated) the modes to run.
"""

import os
import random

import cocotb

from mgpu import _core, hostregs as hr
from mgpu.fuzz import gen as fuzz_gen

from lockstep import Launch, Lockstep, model_cycles

W = int(os.environ.get("W", "8"))
SEEDS = int(os.environ.get("MGPU_RTL_FUZZ_SEEDS", "16"))
START = int(os.environ.get("FUZZ_START", "0"))
MODES = os.environ.get("FUZZ_MODES", "diff,raw,eq").split(",")


def terminates(case, block, max_steps) -> bool:
    """Does the plain engine finish the launch (no watchdog timeout)?"""
    g = _core.Gpu(mem_size=case.mem_size, max_block_dim=32 * W, max_steps=max_steps)
    g.mem_write(case.entry, b"".join(w.to_bytes(4, "little") for w in case.words))
    for addr, words in case.memory.items():
        g.mem_write(addr, b"".join(w.to_bytes(4, "little") for w in words))
    for off, v in {hr.ENTRY: case.entry, hr.GRID_DIM: case.grid, hr.BLOCK_DIM: block,
                   **{hr.ARG(i): a for i, a in enumerate(case.args)}}.items():
        g.reg_write(off, v)
    try:
        g.reg_write(hr.GO, 1)
    except Exception:            # noqa: BLE001 - SimTimeout
        return False
    return True


async def run_mode(dut, mode):
    if mode not in MODES:
        return
    ran = skipped = faults = 0
    for seed in range(START, START + SEEDS):
        if mode == "raw":
            case = fuzz_gen.raw_case(seed)
        else:
            case = fuzz_gen.random_case(seed, mode)
        block = case.block if mode == "eq" else random.Random(seed).choice([1, 17, 32])
        if not terminates(case, block, case.params.get("max_steps", 50_000)):
            skipped += 1
            continue
        ln = Launch(case.words, entry=case.entry, grid=case.grid, block=block, args=case.args,
                    memory=case.memory, mem_size=case.mem_size)
        try:
            res = await Lockstep(dut, ln, W).run(max_cycles=3_000_000)
        except AssertionError as e:
            raise AssertionError(f"{mode} seed {seed} (grid {case.grid} x {block}): {e}") from e
        want = model_cycles(ln, W)
        assert res.cycles == want, f"{mode} seed {seed}: {res.cycles} cycles, timing model {want}"
        ran += 1
        faults += res.fault is not None
    dut._log.info(f"{mode}: {ran} programs in lockstep and cycle-exact ({faults} ended in a fault), "
                  f"{skipped} skipped (reference timeout)")


@cocotb.test()
async def fuzz_diff(dut):
    await run_mode(dut, "diff")


@cocotb.test()
async def fuzz_raw(dut):
    await run_mode(dut, "raw")


@cocotb.test()
async def fuzz_eq(dut):
    await run_mode(dut, "eq")
