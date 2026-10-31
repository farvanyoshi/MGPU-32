"""mgpu_fetch in lockstep with the timing model's fetch rules (core.md 11.3, timing.rs):
round-robin grants, hit / miss / fill-pending timing, fetch faults, overlapping fills and
conflict evictions. A memory stub follows the 11.4 accept rule."""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

W = int(os.environ.get("W", "8"))
LINES = 16
T_MEM, T_SEG = 20, 3
MEM_LAST = 0x1FFC                       # MEM_SIZE = 8 KiB: 64 lines over 16 cache entries


def word_at(a):
    return (a * 2654435761 + 0x9E37) & 0xFFFFFFFF


class Memory:
    def __init__(self):
        self.free = 0

    def read(self, r):
        a = max(r, self.free)
        self.free = a + T_SEG
        return a + T_MEM + 8


class RefFetch:
    """timing.rs fetch: grants and buffer-valid cycles."""

    def __init__(self):
        self.rr = 0
        self.icache = [(None, 0)] * LINES
        self.mem = Memory()
        self.hits = self.misses = 0

    def lookup(self, now, reqs, pcs):
        for k in range(W):
            i = (self.rr + k) % W
            if reqs[i]:
                self.rr = (i + 1) % W
                pc = pcs[i]
                if pc > MEM_LAST:
                    return i, now + 2
                line = pc >> 7
                idx = line % LINES
                tag, valid_from = self.icache[idx]
                if tag == line:
                    self.hits += 1
                    return i, max(valid_from, now) + 2
                self.misses += 1
                done = self.mem.read(now + 1)
                self.icache[idx] = (line, done)
                return i, done + 2
        return None, None


def random_pc(rng):
    r = rng.random()
    if r < 0.05:
        return rng.randrange(MEM_LAST + 4, MEM_LAST + 0x200, 4)    # fetch fault
    if r < 0.5:
        return rng.randrange(0, 0x400, 4)                          # a hot loop region
    return rng.randrange(0, MEM_LAST + 4, 4)


@cocotb.test()
async def lockstep(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    rng = random.Random(77 + W)
    dut.rst_n.value = 0
    dut.inv.value = 0
    dut.mem_last.value = MEM_LAST
    dut.req.value = 0
    dut.consume.value = 0
    dut.mresp_valid.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    ref = RefFetch()
    pc = [random_pc(rng) for _ in range(W)]
    fetch_req = [rng.randrange(0, 2 * W) for _ in range(W)]   # cycle from which to request
    ibuf_at = [None] * W                                       # expected valid cycle
    responses = {}                                             # done cycle -> (id, line)
    stub = Memory()
    delivered = hits = misses = 0
    n_cycles = int(os.environ.get("CYCLES", "6000"))
    for now in range(n_cycles):
        await FallingEdge(dut.clk)
        # ---- registered outputs of this cycle
        valid = int(dut.ibuf_valid.value)
        for w in range(W):
            if ibuf_at[w] is not None and ibuf_at[w] > now:
                assert not (valid >> w & 1), f"warp {w}: buffer valid at {now}, expected {ibuf_at[w]}"
            if ibuf_at[w] == now:
                assert valid >> w & 1, f"warp {w}: buffer not valid at {now}"
                ff = int(dut.ibuf_ff.value) >> w & 1
                word = (int(dut.ibuf_word.value) >> (32 * w)) & 0xFFFFFFFF
                assert ff == (pc[w] > MEM_LAST), f"warp {w}: fetch fault flag {ff} for pc {pc[w]:#x}"
                if not ff:
                    assert word == word_at(pc[w]), f"warp {w}: word {word:#x} for pc {pc[w]:#x}"
                delivered += 1
        if int(dut.mreq_valid.value):
            addr, mid = int(dut.mreq_addr.value), int(dut.mreq_id.value)
            assert addr % 128 == 0
            responses[stub.read(now)] = (mid, addr)
        if now in responses:
            mid, addr = responses.pop(now)
            dut.mresp_valid.value = 1
            dut.mresp_id.value = mid
            dut.mresp_data.value = sum(word_at(addr + 4 * k) << (32 * k) for k in range(32))
        else:
            dut.mresp_valid.value = 0
        # ---- consume buffers (issue), schedule the next fetch
        consume = 0
        for w in range(W):
            if ibuf_at[w] is not None and ibuf_at[w] <= now and rng.random() < 0.4:
                consume |= 1 << w
                ibuf_at[w] = None
                branch = rng.random() < 0.3
                fetch_req[w] = now + (rng.randrange(2, 12) if branch else 1)
                pc[w] = random_pc(rng) if branch or rng.random() < 0.1 else pc[w] + 4
        dut.consume.value = consume
        reqs = [ibuf_at[w] is None and fetch_req[w] <= now for w in range(W)]
        dut.req.value = sum(1 << w for w in range(W) if reqs[w])
        dut.req_pc.value = sum(pc[w] << (32 * w) for w in range(W))
        await Timer(1, unit="ns")
        g, at = ref.lookup(now, reqs, pc)
        grant = int(dut.grant.value)
        assert grant == (0 if g is None else 1 << g), f"cycle {now}: grant {grant:#x}, expected warp {g}"
        if g is not None:
            ibuf_at[g] = at
        hits += int(dut.ev_hit.value)
        misses += int(dut.ev_miss.value)
        await RisingEdge(dut.clk)
    assert delivered > n_cycles // 10, delivered
    assert (hits, misses) == (ref.hits, ref.misses), ((hits, misses), (ref.hits, ref.misses))
    dut._log.info(f"W={W}: {delivered} fetches matched ({ref.hits} hits, {ref.misses} misses)")
