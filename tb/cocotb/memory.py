"""Testbench memory model behind a core's memory ports (docs/uarch/core.md 11.4).

Segment requests are accepted at a = max(r, free), with free = a + T_seg afterwards.
A READ completes at a + T_mem + 8 and a WRITE is acknowledged at a + T_mem. An ATOMIC
(the atomic unit: read, modify, write back) completes at done = a + T_mem + 8 and blocks
the memory until its write-back is accepted at done, so free = done + T_seg. Within a
cycle, LSU requests are accepted before the I-fetch request.
"""

from mgpu.model.execute import atom_op

ATOM_OPS = ["add", "min", "max", "xchg", "and", "or"]


class Memory:
    def __init__(self, size: int, t_mem: int = 100, t_seg: int = 8):
        self.size = size
        self.data = bytearray(size)
        self.t_mem, self.t_seg = t_mem, t_seg
        self.free = 0
        self.ifetch_resp = {}                  # cycle -> (id, line address)
        self.read_resp = {}                    # cycle -> (warp, seg, segment data)
        self.atomic_resp = {}                  # cycle -> (warp, seg, old value per lane)
        self.acks = {}                         # cycle -> warp
        self.reads = self.writes = self.atomics = 0

    # ---- host access
    def write_words(self, addr: int, words) -> None:
        for i, w in enumerate(words):
            self.data[addr + 4 * i:addr + 4 * i + 4] = (w & 0xFFFFFFFF).to_bytes(4, "little")

    def line(self, addr: int) -> int:
        """The 128-byte segment at `addr` as one integer (word 0 in the low bits)."""
        return int.from_bytes(self.data[addr:addr + 128].ljust(128, b"\0"), "little")

    # ---- timing
    def accept(self, r: int, slots: int = 1) -> int:
        a = max(r, self.free)
        self.free = a + slots * self.t_seg
        return a

    def word(self, addr: int) -> int:
        return int.from_bytes(self.data[addr:addr + 4], "little")

    def lsu_request(self, now: int, r: dict) -> None:
        """A segment request from the LSU port (fields of mreq_t). Its effect on memory
        happens at acceptance; requests are accepted in arrival order."""
        base = r["addr"]
        if r["kind"] == 0:                                  # READ
            self.reads += 1
            done = self.accept(now) + self.t_mem + 8
            assert done not in self.read_resp
            self.read_resp[done] = (r["warp"], r["seg"], self.line(base))
        elif r["kind"] == 1:                                # WRITE
            self.writes += 1
            ack = self.accept(now) + self.t_mem
            for k in range(32):
                if r["wmask"] >> k & 1:
                    self.write_words(base + 4 * k, [(r["wdata"] >> (32 * k)) & 0xFFFFFFFF])
            assert ack not in self.acks
            self.acks[ack] = r["warp"]
        else:                                               # ATOMIC, lanes in ascending order
            self.atomics += 1
            done = max(now, self.free) + self.t_mem + 8
            self.free = done + self.t_seg
            op = ATOM_OPS[r["aop"]]
            old = 0
            for lane in range(32):
                if r["lanes"] >> lane & 1:
                    a = base + 4 * ((r["off"] >> (5 * lane)) & 31)
                    x = (r["opnd"] >> (32 * lane)) & 0xFFFFFFFF
                    o = self.word(a)
                    old |= o << (32 * lane)
                    self.write_words(a, [atom_op(op, o, x)])
            assert done not in self.atomic_resp
            self.atomic_resp[done] = (r["warp"], r["seg"], old)

    def ifetch_request(self, now: int, rid: int, addr: int) -> None:
        self.reads += 1
        done = self.accept(now) + self.t_mem + 8
        assert done not in self.ifetch_resp
        self.ifetch_resp[done] = (rid, addr)

    def ifetch_response(self, now: int):
        """(id, line data) delivered in cycle `now`, or None."""
        if now not in self.ifetch_resp:
            return None
        rid, addr = self.ifetch_resp.pop(now)
        return rid, self.line(addr)


class BeatMemory(Memory):
    """The memory behind mgpu_top's 128-bit memory port (interfaces.md 2), with T_seg = 8:
    a request is accepted when memory is free (free = a + 8); a WRITE's beats arrive in
    cycles a .. a + 7 and are applied with the last one, acknowledged at a + T_mem; a READ's
    data (as of cycle a) returns in beats at a + T_mem + 1 .. a + T_mem + 8."""

    def __init__(self, size: int, t_mem: int = 100):
        super().__init__(size, t_mem, 8)
        self.rbeats = {}                           # cycle -> (tag, data, last)
        self.back = {}                             # cycle -> tag (write acknowledgements)
        self.wr = None                             # (addr, mask, beats so far)

    def inputs(self, now: int):
        """(ready, read beat or None, acknowledgement tag or None) for cycle `now`."""
        return now >= self.free, self.rbeats.pop(now, None), self.back.pop(now, None)

    def observe(self, now: int, dut, ready: bool) -> None:
        """Take this cycle's request and write beat from the port (after inputs settled)."""
        consumed = self.wr is not None
        if self.wr is not None:
            assert int(dut.mwd_valid.value), f"cycle {now}: write beat missing"
            self.wr[2].append(int(dut.mwd_data.value))
            assert int(dut.mwd_last.value) == (len(self.wr[2]) == 8)
            if len(self.wr[2]) == 8:
                addr, mask, beats = self.wr
                for k in range(32):
                    if mask >> k & 1:
                        self.write_words(addr + 4 * k, [(beats[k // 4] >> (32 * (k % 4))) & 0xFFFFFFFF])
                self.wr = None
        if int(dut.mreq_valid.value) and ready:
            op, addr, tag = int(dut.mreq_op.value), int(dut.mreq_addr.value), int(dut.mreq_tag.value)
            assert addr % 128 == 0 and addr + 128 <= self.size, f"cycle {now}: bad address {addr:#x}"
            self.free = now + 8
            if op == 0:
                self.reads += 1
                data = self.line(addr)
                for k in range(8):
                    self.rbeats[now + self.t_mem + 1 + k] = (tag, (data >> (128 * k)) & ((1 << 128) - 1), k == 7)
            else:
                self.writes += 1
                assert int(dut.mwd_valid.value), f"cycle {now}: beat 0 must come with the WRITE"
                self.wr = (addr, int(dut.mreq_mask.value), [int(dut.mwd_data.value)])
                self.back[now + self.t_mem] = tag
        elif not consumed:
            assert not int(dut.mwd_valid.value), f"cycle {now}: stray write beat"

    def drive(self, dut, now: int) -> bool:
        """Drive this cycle's memory-side inputs; returns `ready`."""
        ready, rbeat, ack = self.inputs(now)
        dut.mreq_ready.value = int(ready)
        dut.mrd_valid.value = int(rbeat is not None)
        if rbeat:
            dut.mrd_tag.value, dut.mrd_data.value, dut.mrd_last.value = rbeat
        dut.mack_valid.value = int(ack is not None)
        if ack is not None:
            dut.mack_tag.value = ack
        return ready
