"""mgpu_top driven through its host and memory ports, in lockstep with the Rust engine and
cycle-exact against the timing model with the same number of cores (top_lockstep.py)."""

import os
import random

import cocotb

from mgpu import hostregs as hr
from mgpu.asm import assemble
from mgpu.fuzz import gen as fuzz_gen

from lockstep import Launch
from top_lockstep import TopLockstep

C = int(os.environ.get("C", "1"))
MEM = int(os.environ.get("MEM_SIZE", str(1 << 20)))
SEEDS = int(os.environ.get("MGPU_RTL_FUZZ_SEEDS", "8"))
CODE, DATA = 0x1000, 0x8000


async def run(dut, src_or_words, **kw):
    words = src_or_words if isinstance(src_or_words, list) else assemble(src_or_words, CODE).words
    kw.setdefault("entry", CODE)
    ln = Launch(words, mem_size=MEM, **kw)
    res = await TopLockstep(dut, ln, C).run()
    dut._log.info(f"C={C}: {res.retired} instructions in lockstep, STATUS {res.status}, {res.cycles} cycles")
    return res


@cocotb.test()
async def host_interface(dut):
    """Launch validation through the CSR port, then a small launch and the register map."""
    res = await run(dut, "EXIT", block=0)                       # BLOCK_DIM = 0
    assert res.status == hr.E_LAUNCH and res.cycles == 1
    res = await run(dut, "EXIT", block=32 * 8 + 1)              # BLOCK_DIM > MAX_BLOCK_DIM
    assert res.status == hr.E_LAUNCH
    res = await run(dut, "EXIT", entry=CODE + 2, block=32)      # misaligned ENTRY
    assert res.status == hr.E_MISALIGNED and res.regs[hr.ERR_PC] == CODE + 2
    res = await run(dut, "EXIT", grid=0, block=32)              # nothing to run
    assert res.status == hr.OK and res.cycles == 1
    res = await run(dut, "S_GETSR s1, GRID_DIM\nEXIT", grid=3, block=64)
    assert res.status == hr.OK


@cocotb.test()
async def directed_suite(dut):
    import directed
    n = 0
    for p in directed.harvest(256):
        dut._log.info(f"directed: {p['name']}")
        await run(dut, p["src"], grid=p["grid"], block=p["block"], args=p["args"], memory=p["data"])
        n += 1
    dut._log.info(f"C={C}: {n} directed programs in lockstep and cycle-exact")


@cocotb.test()
async def kernel_suite(dut):
    import directed
    progs = directed.harvest_kernels(256)
    for p in progs:
        dut._log.info(f"kernels: {p['name']}")
        await run(dut, [], entry=p["entry"], grid=p["grid"], block=p["block"], args=p["args"],
                  memory=p["image"])
    dut._log.info(f"C={C}: {len(progs)} kernel launches in lockstep and cycle-exact")


@cocotb.test()
async def fuzz(dut):
    """eq-mode programs (multi-block, multi-warp) and diff/raw programs on one block."""
    start = int(os.environ.get("FUZZ_START", "0"))
    n = 0
    for seed in range(start, start + SEEDS):
        for mode in ("eq", "diff", "raw"):
            case = fuzz_gen.raw_case(seed) if mode == "raw" else fuzz_gen.random_case(seed, mode)
            grid, block = (case.grid, case.block) if mode == "eq" else (1, random.Random(seed).choice([1, 17, 32, 64]))
            if mode != "eq" and block > 32:
                grid = 1
            from test_core_fuzz import terminates
            case.mem_size = MEM
            if not terminates(case, block, case.params.get("max_steps", 50_000)):
                continue
            await run(dut, case.words, entry=case.entry, grid=grid, block=block, args=case.args,
                      memory=case.memory)
            n += 1
    dut._log.info(f"C={C}: {n} fuzz programs in lockstep and cycle-exact")


@cocotb.test()
async def multicore_focus(dut):
    """Simultaneous faults, atomics from every core on one word, and a fault while other
    cores have memory traffic outstanding."""
    # every block reaches an illegal word at almost the same time; exactly one fault is reported
    words = assemble("V_GETSR v1, TID\nS_GETSR s2, BLOCK_ID\nS_ANDI s2, s2, 1\nS_BZ s2, bad\nNOP\n"
                     "bad: EXIT", CODE).words
    words[-1] = 0xFC00_0000
    res = await run(dut, words, grid=2 * C, block=64)
    assert res.status == hr.E_ILLEGAL
    # every thread of every block adds 1 and its GTID into two words
    res = await run(dut, f"""
        S_LI s1, {DATA}
        V_GETSR v1, GTID
        V_ADDI v2, v0, 1
        V_ATOM.ADD v3, v0, v2          ; word 0 (address 0 + s0): +1 per thread
        V_ADD v4, v0, s1
        V_ATOM.ADD v5, v4, v1          ; DATA: + GTID
        V_ATOM.MAX v6, v4, v1          ; DATA: max(GTID) as well (commutative mix)
        EXIT""", grid=3 * C, block=96)
    assert res.status == hr.OK
    # block 0 faults late, while the others stream loads, stores and atomics
    res = await run(dut, f"""
        S_LI s1, {DATA}
        V_GETSR v1, GTID
        V_SHLI v2, v1, 2
        V_ADD v2, v2, s1
        S_GETSR s3, BLOCK_ID
        S_ADDI s5, s0, 6
loop:   V_LW v3, 0(v2)
        V_ADDI v3, v3, 1
        V_SW v3, 0(v2)
        V_ATOM.ADD v4, v0, v3
        S_ADDI s5, s5, -1
        S_BNZ s5, loop
        S_BNZ s3, done
        V_LW v6, 2(v2)                 ; misaligned: fault
done:   EXIT""", grid=2 * C, block=128)
    assert res.status == hr.E_MISALIGNED
