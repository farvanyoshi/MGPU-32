"""Testbench memory model behind a core's memory ports (docs/uarch/core.md 11.4).

Segment requests are accepted at a = max(r, free), with free = a + T_seg afterwards
(ATOMIC: two slots). A READ completes at a + T_mem + 8, a WRITE is acknowledged at
a + T_mem, an ATOMIC completes at a + 2 * T_mem + 8. Within a cycle, LSU requests are
accepted before the I-fetch request.
"""

from collections import defaultdict


class Memory:
    def __init__(self, size: int, t_mem: int = 100, t_seg: int = 8):
        self.size = size
        self.data = bytearray(size)
        self.t_mem, self.t_seg = t_mem, t_seg
        self.free = 0
        self.ifetch_resp = {}                  # cycle -> (id, line address)
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
