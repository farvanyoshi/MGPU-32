"""Architectural state of warps and blocks in the reference model (ISA section 3)."""

from ..trace import CommitRecord

WARP = 32
SHARED_WORDS = 16384 // 4

RUN, BAR_WAIT, EXITED = 0, 1, 2


class Fault(Exception):
    """An architectural fault (ISA section 11). Raised before any state is modified."""

    def __init__(self, code: int, lane: int, info: int):
        super().__init__(code, lane, info)
        self.code, self.lane, self.info = code, lane, info & 0xFFFFFFFF


class Warp:
    def __init__(self, block: "Block", wid: int, entry: int, args: list[int], block_dim: int):
        self.block = block
        self.wid = wid
        self.pc = entry
        self.s = [0] * 32
        self.s[1:9] = [a & 0xFFFFFFFF for a in args]
        live = max(0, min(WARP, block_dim - WARP * wid))
        self.s[31] = (1 << live) - 1                 # [HOST] initial EXEC
        self.v = [[0] * WARP for _ in range(32)]     # v[reg][lane]
        self.state = RUN

    @property
    def key(self) -> tuple[int, int]:
        return (self.block.bid, self.wid)


class Block:
    def __init__(self, bid: int, num_warps: int, entry: int, args: list[int], block_dim: int,
                 poison: int):
        self.bid = bid
        self.warps = [Warp(self, w, entry, args, block_dim) for w in range(num_warps)]
        self.shared = [poison & 0xFFFFFFFF] * SHARED_WORDS
        self.bar_pc = None          # P of [BAR-1]; None while no warp waits

    def live(self) -> list[Warp]:
        return [w for w in self.warps if w.state != EXITED]

    def finished(self) -> bool:
        return all(w.state == EXITED for w in self.warps)

    def try_release(self, rec: CommitRecord) -> None:
        """[BAR-3]: release when every non-exited warp waits."""
        live = self.live()
        if self.bar_pc is not None and live and all(w.state == BAR_WAIT for w in live):
            for w in live:
                w.state = RUN
                w.pc = (self.bar_pc + 4) & 0xFFFFFFFF
            self.bar_pc = None
            rec.released = True
