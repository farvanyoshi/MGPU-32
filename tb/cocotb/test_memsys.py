"""mgpu_mem_arbiter + mgpu_atomic_unit (mgpu_memsys_probe) behind stand-in cores and in
front of a beat-level memory model. Every core-side response must arrive in the cycle, and
with the data, of a segment-level reference of the timing contract (core.md 11.4): requests
taken in (request cycle, core, LSU before I-fetch) order, a = max(r, free), READ done at
a + T + 8, WRITE ack at a + T, ATOMIC done at a + T + 8 with memory free at done + 8."""

import os
import random
from collections import defaultdict

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

from mgpu.model.execute import atom_op

from lockstep import MREQ

C = int(os.environ.get("C", "4"))
T = int(os.environ.get("T_MEM", "20"))
SEGS = 48                       # memory: 48 segments (6 KiB), lots of collisions
ATOM_OPS = ["add", "min", "max", "xchg", "and", "or"]


def word(mem, a):
    return mem[a // 4]


class Reference:
    """Segment-level contract: when each response arrives and what it carries."""

    def __init__(self, init):
        self.mem = list(init)
        self.free = 0
        self.expect = defaultdict(list)           # cycle -> [(core, channel, fields)]

    def request(self, r, core, port, q):
        a = max(r, self.free)
        base = q["addr"] // 4
        if port == "if":
            self.free = a + 8
            self.expect[a + T + 8].append((core, "if", (q["id"], tuple(self.mem[base:base + 32]))))
        elif q["kind"] == 0:
            self.free = a + 8
            self.expect[a + T + 8].append((core, "rd", (q["warp"], q["seg"], tuple(self.mem[base:base + 32]))))
        elif q["kind"] == 1:
            self.free = a + 8
            for k in range(32):
                if q["wmask"] >> k & 1:
                    self.mem[base + k] = (q["wdata"] >> (32 * k)) & 0xFFFFFFFF
            self.expect[a + T].append((core, "ack", (q["warp"],)))
        else:
            done = a + T + 8
            self.free = done + 8
            old = {}
            for lane in range(32):
                if q["lanes"] >> lane & 1:
                    off = (q["off"] >> (5 * lane)) & 31
                    x = (q["opnd"] >> (32 * lane)) & 0xFFFFFFFF
                    old[lane] = self.mem[base + off]
                    self.mem[base + off] = atom_op(ATOM_OPS[q["aop"]], old[lane], x)
            self.expect[done].append((core, "at", (q["warp"], q["seg"], tuple(sorted(old.items())))))


class BeatMemory:
    """The memory behind the 128-bit port."""

    def __init__(self, init):
        self.mem = list(init)
        self.free = 0
        self.rbeats = {}                           # cycle -> (tag, data, last)
        self.acks = {}                             # cycle -> tag
        self.wr = None                             # (addr, mask, beats so far, started)

    def step_inputs(self, now):
        return now >= self.free, self.rbeats.pop(now, None), self.acks.pop(now, None)

    def observe(self, now, dut, ready):
        consumed = self.wr is not None
        if self.wr is not None:
            assert int(dut.mwd_valid.value), f"cycle {now}: write beat missing"
            self.wr[2].append(int(dut.mwd_data.value))
            assert int(dut.mwd_last.value) == (len(self.wr[2]) == 8)
            if len(self.wr[2]) == 8:
                addr, mask, beats = self.wr[:3]
                for k in range(32):
                    if mask >> k & 1:
                        self.mem[addr // 4 + k] = (beats[k // 4] >> (32 * (k % 4))) & 0xFFFFFFFF
                self.wr = None
        if int(dut.mreq_valid.value) and ready:
            op, addr, tag = int(dut.mreq_op.value), int(dut.mreq_addr.value), int(dut.mreq_tag.value)
            assert addr % 128 == 0 and 0 <= addr < SEGS * 128
            self.free = now + 8
            if op == 0:
                data = self.mem[addr // 4:addr // 4 + 32]
                for k in range(8):
                    beat = sum(data[4 * k + i] << (32 * i) for i in range(4))
                    self.rbeats[now + T + 1 + k] = (tag, beat, k == 7)
            else:
                assert int(dut.mwd_valid.value), f"cycle {now}: beat 0 must come with the WRITE"
                self.wr = (addr, int(dut.mreq_mask.value), [int(dut.mwd_data.value)])
                self.acks[now + T] = tag
        elif not consumed:
            assert not int(dut.mwd_valid.value), f"cycle {now}: stray write beat"


def random_lsu(rng):
    kind = rng.choices([0, 1, 2], weights=[4, 4, 2])[0]
    q = dict(valid=1, kind=kind, warp=rng.randrange(8), seg=rng.randrange(32),
             addr=rng.randrange(SEGS) * 128, wmask=0, wdata=0, lanes=0, off=0, opnd=0, aop=0)
    if kind == 1:
        q["wmask"] = rng.getrandbits(32)
        q["wdata"] = rng.getrandbits(1024)
    elif kind == 2:
        q["lanes"] = rng.getrandbits(32) if rng.random() < 0.7 else 1 << rng.randrange(32)
        hot = rng.randrange(32)
        q["off"] = sum((hot if rng.random() < 0.3 else rng.randrange(32)) << (5 * i) for i in range(32))
        q["opnd"] = rng.getrandbits(1024)
        q["aop"] = rng.randrange(6)
    return q


def pack_mreq(q):
    v = 0
    for name, width in MREQ.fields:
        v = (v << width) | (q.get(name, 0) & ((1 << width) - 1))
    return v


@cocotb.test()
async def random_traffic(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    rng = random.Random(21 + C + T)
    init = [rng.getrandbits(32) for _ in range(SEGS * 32)]
    ref, memory = Reference(init), BeatMemory(init)
    dut.rst_n.value = 0
    dut.if_req_valid.value = 0
    dut.lsu_req.value = 0
    dut.mrd_valid.value = 0
    dut.mack_valid.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    burst = [0] * C
    # offered load: 60% of what memory can serve (an atomic holds it for T + 16 cycles)
    service = 0.8 * 8 + 0.2 * (T + 16)
    total = 0.6 / service
    p_single, p_burst = total * 0.5 / C, total * 0.5 / (6.5 * C)
    if_out = [[] for _ in range(C)]             # completion cycles of outstanding I-fetch reads
    counts = defaultdict(int)
    n_cycles = int(os.environ.get("CYCLES", "60000"))
    for now in range(n_cycles):
        await FallingEdge(dut.clk)
        ready, rbeat, ack = memory.step_inputs(now)
        dut.mreq_ready.value = int(ready)
        dut.mrd_valid.value = int(rbeat is not None)
        if rbeat:
            dut.mrd_tag.value, dut.mrd_data.value, dut.mrd_last.value = rbeat
        dut.mack_valid.value = int(ack is not None)
        if ack is not None:
            dut.mack_tag.value = ack
        # stand-in cores: occasional bursts of LSU requests, sparse I-fetch reads
        lsu_v, if_v = 0, 0
        lsu_words, if_addr, if_id = [0] * C, [0] * C, [0] * C
        reqs = []
        # offered load below one request per 8 cycles overall (memory bandwidth), with bursts;
        # the last cycles are quiet so that everything drains before the final comparison
        quiet = now >= n_cycles - 3000
        for c in range(C):
            if quiet:
                continue
            if burst[c] == 0 and rng.random() < p_burst:
                burst[c] = rng.randint(1, 12)
            if burst[c] or rng.random() < p_single:
                burst[c] = max(0, burst[c] - 1)
                q = random_lsu(rng)
                lsu_words[c] = pack_mreq(q)
                reqs.append((c, "lsu", q))
            if_out[c] = [d for d in if_out[c] if d > now]
            if len(if_out[c]) < 8 and rng.random() < 0.2 * total / C:   # at most one fill per warp slot
                if_v |= 1 << c
                if_addr[c], if_id[c] = rng.randrange(SEGS) * 128, rng.randrange(8)
                reqs.append((c, "if", dict(addr=if_addr[c], id=if_id[c])))
        dut.lsu_req.value = sum(w << (MREQ.width * c) for c, w in enumerate(lsu_words))
        dut.if_req_valid.value = if_v
        dut.if_req_addr.value = sum(a << (32 * c) for c, a in enumerate(if_addr))
        dut.if_req_id.value = sum(i << (5 * c) for c, i in enumerate(if_id))
        for c, port, q in sorted(reqs, key=lambda x: (x[0], x[1] != "lsu")):
            ref.request(now, c, port, q)
            if port == "if":
                if_out[c].append(max(d for d, es in ref.expect.items() if any(e[0] == c and e[1] == "if" for e in es)))
        await Timer(1, unit="ns")
        # ---- core-side responses this cycle
        got = []
        for c in range(C):
            if int(dut.if_resp_valid.value) >> c & 1:
                d = int(dut.if_resp_data.value) >> (1024 * c)
                got.append((c, "if", ((int(dut.if_resp_id.value) >> (5 * c)) & 31,
                                      tuple((d >> (32 * k)) & 0xFFFFFFFF for k in range(32)))))
            if int(dut.rd_valid.value) >> c & 1:
                d = int(dut.rd_data.value) >> (1024 * c)
                got.append((c, "rd", ((int(dut.rd_warp.value) >> (5 * c)) & 31, (int(dut.rd_seg.value) >> (5 * c)) & 31,
                                      tuple((d >> (32 * k)) & 0xFFFFFFFF for k in range(32)))))
            if int(dut.ack_valid.value) >> c & 1:
                got.append((c, "ack", ((int(dut.ack_warp.value) >> (5 * c)) & 31,)))
            if int(dut.at_valid.value) >> c & 1:
                d = int(dut.at_old.value) >> (1024 * c)
                exp = [e for e in ref.expect.get(now, []) if e[0] == c and e[1] == "at"]
                lanes = [l for l, _ in exp[0][2][2]] if exp else []
                got.append((c, "at", ((int(dut.at_warp.value) >> (5 * c)) & 31, (int(dut.at_seg.value) >> (5 * c)) & 31,
                                      tuple((l, (d >> (32 * l)) & 0xFFFFFFFF) for l in lanes))))
        want = ref.expect.pop(now, [])
        assert sorted(got) == sorted(want), f"cycle {now}:\n got  {got}\n want {want}"
        for g in got:
            counts[g[1]] += 1
        memory.observe(now, dut, ready)
        if int(dut.overflow.value):
            raise AssertionError(f"cycle {now}: queue overflow, lcnt={dut.u_arb.lcnt.value} icnt={dut.u_arb.icnt.value} "
                                 f"busy={dut.u_atom.a_busy.value} wb={dut.u_atom.a_wb.value} free={memory.free}")
        await RisingEdge(dut.clk)
    assert memory.mem == ref.mem, "final memory differs"
    assert all(counts[k] > 50 for k in ("if", "rd", "ack", "at")), dict(counts)
    dut._log.info(f"C={C} T_mem={T}: responses matched {dict(counts)}; final memory matches")
