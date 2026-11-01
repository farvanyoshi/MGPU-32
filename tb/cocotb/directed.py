"""Harvest the directed programs of tests/sim for the RTL.

Every test there that takes the `run` fixture is called with a recording `run`, which runs
the program on the Rust engine as usual (so the test's own assertions still pass) and keeps
the program and launch configuration. Parametrized tests are expanded.
"""

import inspect
import itertools
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[2] / "tests" / "sim"
POISON = 0xDEADBEEF


def _expand(fn):
    """Keyword sets for every parametrization of a test function."""
    sets = [{}]
    for mark in getattr(fn, "pytestmark", []):
        if mark.name != "parametrize":
            continue
        names, values = mark.args[0], mark.args[1]
        names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
        new = []
        for v in values:
            v = getattr(v, "values", v)                 # pytest.param
            v = v if len(names) > 1 else (v,)
            new.append(dict(zip(names, v)))
        sets = [{**a, **b} for a, b in itertools.product(sets, new)]
    return sets


def harvest(max_block: int) -> list[dict]:
    """[{name, src, grid, block, args, data, mem}] for programs the core can run."""
    sys.path.insert(0, str(SIM))
    import simutil
    progs = []

    for mod_name in ("test_alu", "test_control", "test_memory", "test_kernels"):
        mod = __import__(mod_name)
        for name, fn in sorted(vars(mod).items()):
            if not name.startswith("test_") or not callable(fn):
                continue
            params = list(inspect.signature(fn).parameters)
            if not params or params[0] != "run":
                continue
            for i, kw in enumerate(_expand(fn)):
                calls = []

                def run(src, _calls=calls, **launch):
                    r = simutil.Run("rust", src, **launch)
                    _calls.append((src, launch, r.res))
                    return r
                try:
                    fn(run, **kw)
                except Exception:                    # noqa: BLE001 - keep what was recorded
                    pass
                for j, (src, launch, res) in enumerate(calls):
                    extra = {k for k in launch if k not in ("grid", "block", "args", "data", "mem")}
                    if launch.get("shared_poison", POISON) != POISON:
                        continue
                    if launch.get("block", 32) > max_block or extra - {"resident_blocks", "seed",
                                                                        "shared_poison", "max_steps",
                                                                        "keep_state", "trace"}:
                        continue
                    if res.status not in range(0, 5):  # timeouts and rejected launches
                        continue
                    progs.append(dict(name=f"{mod_name}.{name}[{i}.{j}]", src=src,
                                      grid=launch.get("grid", 1), block=launch.get("block", 32),
                                      args=list(launch.get("args", ())),
                                      data=launch.get("data") or {}, mem=launch.get("mem", simutil.MEM)))
    return progs


KERNELS = Path(__file__).resolve().parents[2] / "tests" / "kernels"


def _sparse(data: bytes, page: int = 4096) -> dict:
    """{address: words} for the non-zero pages of a memory image."""
    out = {}
    for a in range(0, len(data), page):
        chunk = data[a:a + page]
        if any(chunk):
            out[a] = [int.from_bytes(chunk[i:i + 4], "little") for i in range(0, len(chunk), 4)]
    return out


def harvest_kernels(max_block: int) -> list[dict]:
    """Every launch of the kernel-suite tests (tests/kernels) at the small problem sizes,
    with a snapshot of memory taken just before the launch."""
    sys.path.insert(0, str(KERNELS))
    from mgpu.device import Device
    progs = []
    orig = Device.launch

    def recording_launch(self, entry, grid, block, args=()):
        res = orig(self, entry, grid, block, args)
        if block <= max_block and res.status in range(0, 5):
            progs.append(dict(name=f"{current[0]}[{len(progs)}]", entry=entry, grid=grid,
                              block=block, args=list(args), image=snapshot[0], mem=self.mem_size))
        return res

    def snap_launch(self, entry, grid, block, args=()):
        snapshot[0] = _sparse(self.read(0, self.mem_size))
        return recording_launch(self, entry, grid, block, args)

    current, snapshot = [""], [None]
    Device.launch = snap_launch
    try:
        for mod_name in ("test_kernels_basic", "test_kernels_advanced"):
            mod = __import__(mod_name)
            for name, fn in sorted(vars(mod).items()):
                if not name.startswith("test_") or not callable(fn):
                    continue
                params = list(inspect.signature(fn).parameters)
                if not params or params[0] != "backend":
                    continue
                for kw in _expand(fn):
                    current[0] = f"{mod_name}.{name}{sorted(kw.items()) or ''}"
                    try:
                        fn(("rust", "small"), **kw)
                    except Exception:                # noqa: BLE001 - keep what was recorded
                        pass
    finally:
        Device.launch = orig
    return progs
