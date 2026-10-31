"""mgpu_dispatcher with stand-in cores (Python): launch validation, multi-core dispatch,
completion, and fault handling, checked cycle by cycle against a reference of top.md 3/6."""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

C = int(os.environ.get("C", "4"))
W = 8


class FakeCore:
    def __init__(self):
        self.active = False
        self.left = 0           # cycles until the block's last warp exits
        self.done_pulse = False
        self.fault_at = None    # cycles until a faulting instruction issues
        self.drain = 0          # cycles until other instructions retire
        self.quiet_in = 0       # cycles until memory traffic settles
        self.fault_pending = False
        self.block = None


def bits(vals):
    return sum(int(bool(v)) << i for i, v in enumerate(vals))


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    for s in ("launch", "idle", "block_done", "fault_issue", "busy_ops", "fault_valid"):
        getattr(dut, s).value = 0
    dut.mem_quiet.value = (1 << C) - 1
    dut.idle.value = (1 << C) - 1
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1


async def launch(dut, rng, entry, grid, block, fault_prob):
    """Run one launch; returns (status list, cycles) as reported at the end."""
    cores = [FakeCore() for _ in range(C)]
    fresh = [True] * C
    next_block = done = 0
    fault_seen, fault_core, t_fault, status, quiet_wait = False, None, None, None, False
    dut.entry.value, dut.grid_dim.value, dut.block_dim.value = entry, grid, block
    await FallingEdge(dut.clk)
    dut.launch.value = 1
    cyc = 0
    while True:
        # ---- stand-in core outputs for this cycle
        idle = [not k.active for k in cores]
        bdone = [k.done_pulse for k in cores]
        issue = [k.fault_at == 0 and not fault_seen for k in cores]
        # a fault at issue holds the higher cores in this cycle: only the lowest one issues
        first = next((i for i in range(C) if issue[i]), None)
        issue = [i == first for i in range(C)]
        busy = [k.active and k.drain > 0 for k in cores]
        quiet = [k.quiet_in == 0 for k in cores]
        fvalid = [k.fault_pending and not any(busy) and k.drain == 0 for k in cores]
        dut.idle.value, dut.block_done.value = bits(idle), bits(bdone)
        dut.fault_issue.value, dut.busy_ops.value = bits(issue), bits(busy)
        dut.mem_quiet.value, dut.fault_valid.value = bits(quiet), bits(fvalid)
        for i, k in enumerate(cores):
            if fvalid[i]:
                getattr(dut, "fault_code").value = sum(((3 + i) & 0xFF) << (8 * j) for j in range(C))
        codes = [(3 + i) & 0xFF for i in range(C)]
        dut.fault_code.value = sum(c << (8 * i) for i, c in enumerate(codes))
        dut.fault_pc.value = sum((0x1000 + 4 * i) << (32 * i) for i in range(C))
        dut.fault_warp.value = sum((i + 1) << (5 * i) for i in range(C))
        dut.fault_lane.value = sum((2 * i) << (5 * i) for i in range(C))
        dut.fault_info.value = sum((0xABC0 + i) << (32 * i) for i in range(C))
        await Timer(1, unit="ns")
        # ---- reference expectations
        exp_start = [False] * C
        exp_ids = [0] * C
        exp_end = None
        if cyc == 0:
            if block == 0 or block > 32 * W:
                exp_end = ([5, 0, 0, 0, 0, 0], 1)
            elif entry & 3:
                exp_end = ([2, entry, 0, 0, 0, entry], 1)
            elif grid == 0:
                exp_end = ([0] * 6, 1)
        else:
            if not fault_seen:
                k = 0
                for i in range(C):
                    if (idle[i] or bdone[i]) and next_block + k < grid:
                        exp_start[i], exp_ids[i] = True, next_block + k
                        k += 1
                if done + sum(bdone) == grid:
                    exp_end = ([0] * 6, cyc - 1)
            if quiet_wait and all(quiet):
                exp_end = (status, t_fault + 1)
        exp_halt = [fault_seen or any(issue[:i]) for i in range(C)]
        # ---- compare
        assert [int(dut.start.value) >> i & 1 for i in range(C)] == [int(x) for x in exp_start], f"cycle {cyc}: start"
        for i in range(C):
            if exp_start[i]:
                assert (int(dut.block_id.value) >> (32 * i)) & 0xFFFFFFFF == exp_ids[i], f"cycle {cyc}: block id"
                assert (int(dut.new_launch.value) >> i & 1) == fresh[i], f"cycle {cyc}: new_launch"
        if cyc > 0:
            assert [int(dut.halt.value) >> i & 1 for i in range(C)] == [int(x) for x in exp_halt], f"cycle {cyc}: halt"
            assert int(dut.fault_go.value) == int(not any(busy)), f"cycle {cyc}: fault_go"
        assert int(dut.end_valid.value) == int(exp_end is not None), f"cycle {cyc}: end_valid"
        if exp_end:
            st = [(int(dut.end_status.value) >> (32 * i)) & 0xFFFFFFFF for i in range(6)]
            assert (st, int(dut.end_cycles.value)) == exp_end, f"cycle {cyc}: end {st} {int(dut.end_cycles.value)} != {exp_end}"
            assert int(dut.kill.value) == int(quiet_wait)
            assert int(dut.num_warps.value) == ((block + 31) // 32) & 63
            await RisingEdge(dut.clk)
            dut.launch.value = 0
            await FallingEdge(dut.clk)
            assert int(dut.busy.value) == 0
            return exp_end
        assert int(dut.live_cycles.value) == cyc
        # ---- advance the reference and the stand-in cores
        if any(issue):
            fault_seen, fault_core, t_fault = True, first, cyc
            k = cores[first]
            k.fault_pending, k.fault_at = True, None
        for i, k in enumerate(cores):
            if fvalid[i]:
                status = [codes[i], 0x1000 + 4 * i, k.block, i + 1, 2 * i, 0xABC0 + i]
                quiet_wait = True
                k.fault_pending = False
        done += sum(bdone)
        for i, k in enumerate(cores):
            k.done_pulse = False
            if k.fault_at:
                k.fault_at -= 1
            if k.drain:
                k.drain -= 1
            if k.quiet_in:
                k.quiet_in -= 1
            if k.active and not fault_seen:
                k.left -= 1
                if k.left == 0:
                    k.active, k.done_pulse = False, True
            if exp_start[i]:
                fresh[i] = False
                k.active, k.left, k.block = True, rng.randint(2, 30), exp_ids[i]
                k.fault_at = rng.randint(1, 20) if rng.random() < fault_prob else None
                k.drain, k.quiet_in = rng.randint(0, 6), rng.randint(0, 12)
        next_block += sum(exp_start)
        if fault_seen:
            for k in cores:
                k.active = k.active  # held; drains continue
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        dut.launch.value = 0
        cyc += 1


@cocotb.test()
async def launches(dut):
    await reset(dut)
    rng = random.Random(11 + C)
    seen = {"validation": 0, "done": 0, "fault": 0}
    for n in range(300):
        r = rng.random()
        if r < 0.1:
            entry, grid, block = 0x1000, rng.randint(1, 5), rng.choice([0, 32 * W + 1, 5000])
        elif r < 0.15:
            entry, grid, block = 0x1002, 3, 64
        elif r < 0.2:
            entry, grid, block = 0x1000, 0, 64
        else:
            entry, grid, block = 0x1000, rng.randint(1, 3 * C + 2), rng.randint(1, 32 * W)
        st, cyc = await launch(dut, rng, entry, grid, block, fault_prob=0.1)
        key = "validation" if cyc == 1 and (block == 0 or block > 32 * W or entry & 3 or grid == 0) else (
            "fault" if st[0] else "done")
        seen[key] += 1
    dut._log.info(f"C={C}: {seen}")
    assert all(v > 10 for v in seen.values()), seen
