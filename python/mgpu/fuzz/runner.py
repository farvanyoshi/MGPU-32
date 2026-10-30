"""Run fuzz cases on the backends and compare the results.

* ``diff_case``   - model vs Rust under one schedule: identical global trace, final memory,
  host registers and timeout behaviour.
* ``equiv_case``  - one backend, several schedules: equivalence rules E1, E2 and E4 of
  docs/isa/equivalence.md (final memory, STATUS, per-warp streams).
"""

from dataclasses import dataclass

from .. import hostregs as H
from ..device import Device
from ..errors import SimTimeout
from ..isa import listing
from ..trace import per_warp
from .gen import FuzzCase

RESULT_REGS = [H.STATUS, H.ERR_PC, H.ERR_BLOCK, H.ERR_WARP, H.ERR_LANE, H.ERR_INFO, H.CYCLES, H.GO]


@dataclass
class Outcome:
    records: list
    memory: bytes
    regs: list
    timeout: bool


def run_case(case: FuzzCase, backend: str, trace: bool = True, **override) -> Outcome:
    params = {**case.params, **override}
    dev = Device(backend, mem_size=case.mem_size, trace=trace, **params)
    for addr, words in case.memory.items():
        dev.write_words(addr, words)
    dev.write_words(case.entry, case.words)
    timeout = False
    try:
        dev.launch(case.entry, case.grid, case.block, case.args)
    except SimTimeout:
        timeout = True
    return Outcome(dev.trace if trace else [], dev.read(0, case.mem_size),
                   [dev.reg_read(r) for r in RESULT_REGS], timeout)


def _first_mem_diff(a: bytes, b: bytes) -> str:
    for i in range(0, len(a), 4):
        if a[i:i + 4] != b[i:i + 4]:
            return (f"first memory difference at {i:#x}: "
                    f"{int.from_bytes(a[i:i + 4], 'little'):#010x} vs "
                    f"{int.from_bytes(b[i:i + 4], 'little'):#010x}")
    return "memory identical"


def _program(case: FuzzCase, around_pc: int | None = None, ctx: int = 6) -> str:
    lines = listing(case.words, case.entry).splitlines()
    if around_pc is None:
        return "\n".join(lines)
    i = (around_pc - case.entry) // 4
    lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
    return "\n".join((">> " if j == i else "   ") + lines[j] for j in range(lo, hi))


def diff_case(case: FuzzCase) -> str | None:
    """Compare model and Rust on one case. Returns a report on mismatch, else None."""
    m = run_case(case, "model")
    r = run_case(case, "rust")
    mk = [x.key() for x in m.records]
    rk = [x.key() for x in r.records]
    if mk != rk:
        i = next((j for j, (a, b) in enumerate(zip(mk, rk)) if a != b), min(len(mk), len(rk)))
        rec = (m.records + r.records)[i] if i < len(m.records) else r.records[i]
        return (f"TRACE MISMATCH {case.describe()}\n"
                f"  record {i} of {len(mk)} (model) / {len(rk)} (rust)\n"
                f"  model: {mk[i] if i < len(mk) else '<end>'}\n"
                f"  rust:  {rk[i] if i < len(rk) else '<end>'}\n" + _program(case, rec.pc))
    if m.timeout != r.timeout:
        return f"TIMEOUT MISMATCH {case.describe()}: model={m.timeout} rust={r.timeout}"
    if m.regs != r.regs:
        return f"REGISTER MISMATCH {case.describe()}: model={m.regs} rust={r.regs}"
    if m.memory != r.memory:
        return f"MEMORY MISMATCH {case.describe()}: {_first_mem_diff(m.memory, r.memory)}"
    return None


SCHEDULES = [dict(resident_blocks=1, seed=None), dict(resident_blocks=3, seed=None),
             dict(resident_blocks=1, seed=1), dict(resident_blocks=2, seed=77),
             dict(resident_blocks=3, seed=12345), dict(resident_blocks=3, seed=2**40 + 9)]


def equiv_case(case: FuzzCase, backend: str = "rust", poison: bool = True) -> str | None:
    """Run a well-defined case under several schedules (and shared-memory poison values);
    E1, E2 and E4 must hold. Returns a report on violation, else None."""
    base = run_case(case, backend, **SCHEDULES[0])
    if base.timeout:
        return f"TIMEOUT {case.describe()}"
    if base.regs[0] != 0:
        return f"UNEXPECTED FAULT {case.describe()}: regs={base.regs}\n" + _program(case, base.regs[1])
    base_warps = per_warp(base.records)
    for k, sched in enumerate(SCHEDULES[1:], 1):
        extra = {"shared_poison": 0x5A5A0000 + k} if poison else {}
        o = run_case(case, backend, **sched, **extra)
        where = f"{case.describe()} schedule={sched}"
        if o.regs[0] != base.regs[0]:
            return f"E2 STATUS differs under {where}: {o.regs[0]} vs {base.regs[0]}"
        if o.memory != base.memory:
            return f"E1 memory differs under {where}: {_first_mem_diff(base.memory, o.memory)}"
        warps = per_warp(o.records)
        if warps != base_warps:
            for key in sorted(set(warps) | set(base_warps)):
                a, b = base_warps.get(key, []), warps.get(key, [])
                if a != b:
                    i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    return (f"E4 stream of warp {key} differs under {where} at record {i}:\n"
                            f"  base: {a[i] if i < len(a) else '<end>'}\n"
                            f"  this: {b[i] if i < len(b) else '<end>'}")
    return None
