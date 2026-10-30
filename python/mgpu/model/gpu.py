"""The reference model's GPU: host registers, global memory, launch and scheduling.

Implements ISA sections 11-13 and the reference schedule of docs/sim/README.md.
"""

from .. import hostregs as H
from ..sched import Xorshift64Star, pick
from ..trace import EV_FAULT, CommitRecord
from .execute import execute
from .state import EXITED, RUN, WARP, Block, Fault


class SimTimeout(Exception):
    """The kernel exceeded max_steps ([FLT-3]); not an architectural fault."""


class Gpu:
    """Pure-Python executable model of MGPU-32.

    Parameters:
      mem_size         global memory size in bytes (multiple of 4)
      max_block_dim    MAX_BLOCK_DIM (multiple of 32, 32..1024)
      resident_blocks  how many blocks the reference schedule keeps resident at once
      seed             None for round-robin interleaving, else an xorshift64* seed
      shared_poison    initial value of every shared-memory word
      max_steps        watchdog; exceeding it raises SimTimeout
      trace            keep a CommitRecord for every instruction in self.trace
      keep_state       keep finished blocks so warp_state() works after the kernel ends
    """

    def __init__(self, mem_size=64 << 20, max_block_dim=1024, resident_blocks=1, seed=None,
                 shared_poison=0xDEADBEEF, max_steps=50_000_000, trace=False, keep_state=False):
        assert mem_size % 4 == 0 and 4 <= mem_size <= 1 << 32
        assert max_block_dim % 32 == 0 and 32 <= max_block_dim <= 1024
        assert resident_blocks >= 1
        self.mem_size = mem_size
        self.max_block_dim = max_block_dim
        self.resident_blocks = resident_blocks
        self.seed = seed
        self.shared_poison = shared_poison
        self.max_steps = max_steps
        self.keep_trace = trace
        self.keep_state = keep_state
        self.mem = bytearray(mem_size)
        self.regs = {H.ENTRY: 0, H.GRID_DIM: 0, H.BLOCK_DIM: 0, **{H.ARG(n): 0 for n in range(8)}}
        self.status = [0] * 6           # STATUS, ERR_PC, ERR_BLOCK, ERR_WARP, ERR_LANE, ERR_INFO
        self.running = False
        self.steps = 0
        self.trace: list[CommitRecord] = []

    # ---- global memory (host side and kernel side) ---------------------------------------
    def load_word(self, addr: int) -> int:
        return int.from_bytes(self.mem[addr:addr + 4], "little")

    def store_word(self, addr: int, value: int) -> None:
        self.mem[addr:addr + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")

    def mem_write(self, addr: int, data: bytes) -> None:
        assert 0 <= addr and addr + len(data) <= self.mem_size, "host access out of range"
        self.mem[addr:addr + len(data)] = data

    def mem_read(self, addr: int, n: int) -> bytes:
        assert 0 <= addr and addr + n <= self.mem_size, "host access out of range"
        return bytes(self.mem[addr:addr + n])

    # ---- host registers (section 13.1) ----------------------------------------------------
    def reg_read(self, off: int) -> int:
        fixed = {H.ID: H.ID_VALUE, H.VERSION: H.VERSION_VALUE, H.MAX_BLOCK_DIM: self.max_block_dim,
                 H.MEM_SIZE: self.mem_size & 0xFFFFFFFF, H.GO: int(self.running),
                 H.CYCLES: self.steps & 0xFFFFFFFF}
        if off in fixed:
            return fixed[off]
        if off in self.regs:
            return self.regs[off]
        if H.STATUS <= off <= H.ERR_INFO and off % 4 == 0:
            return self.status[(off - H.STATUS) // 4]
        return 0

    def reg_write(self, off: int, value: int) -> None:
        value &= 0xFFFFFFFF
        if self.running:                                    # [HOST-1], [HOST-2]
            return
        if off in self.regs:
            self.regs[off] = value
        elif off == H.GO and value & 1:
            if self.begin():
                self.run()

    # ---- launch ----------------------------------------------------------------------------
    def _set_status(self, code, pc=0, block=0, warp=0, lane=0, info=0):
        self.status = [code, pc & 0xFFFFFFFF, block, warp, lane, info & 0xFFFFFFFF]

    def begin(self) -> bool:
        """Validate and start a launch ([HOST-3], [HOST-4]). Returns True if warps will run."""
        self.status = [0] * 6
        self.steps = 0
        self.trace = []
        entry, grid, bdim = self.regs[H.ENTRY], self.regs[H.GRID_DIM], self.regs[H.BLOCK_DIM]
        if bdim == 0 or bdim > self.max_block_dim:
            self._set_status(H.E_LAUNCH)
            return False
        if entry & 3:
            self._set_status(H.E_MISALIGNED, pc=entry, info=entry)
            return False
        if grid == 0:
            return False
        self.entry, self.grid, self.block_dim = entry, grid, bdim
        self.num_warps = -(-bdim // WARP)
        self.args = [self.regs[H.ARG(n)] for n in range(8)]
        self.next_block = 0
        self.resident: list[Block] = []
        self.blocks: dict[int, Block] = {}      # live blocks (plus finished ones if keep_state)
        self.done: set[int] = set()
        self.last = None
        self.prng = Xorshift64Star(self.seed) if self.seed is not None else None
        self.running = True
        return True

    def _new_block(self, bid: int) -> Block:
        b = Block(bid, self.num_warps, self.entry, self.args, self.block_dim, self.shared_poison)
        self.blocks[bid] = b
        return b

    def _fill(self) -> None:
        while len(self.resident) < self.resident_blocks and self.next_block < self.grid:
            self.resident.append(self._new_block(self.next_block))
            self.next_block += 1

    def sr_value(self, name: str, warp, lane: int, step: int) -> int:
        tid = WARP * warp.wid + lane
        return {
            "BLOCK_ID": warp.block.bid, "BLOCK_DIM": self.block_dim, "GRID_DIM": self.grid,
            "WARP_ID": warp.wid, "NUM_WARPS": self.num_warps, "CLOCK": step & 0xFFFFFFFF,
            "TID": tid, "LANE_ID": lane,
            "GTID": (warp.block.bid * self.block_dim + tid) & 0xFFFFFFFF,
        }[name]

    def _retire(self, warp) -> CommitRecord:
        """Execute one instruction of `warp`; on a fault, abort the kernel ([FLT-1])."""
        try:
            rec = execute(self, warp, self.steps)
        except Fault as f:
            word = self.load_word(warp.pc) if warp.pc <= self.mem_size - 4 else 0
            rec = CommitRecord(self.steps, warp.block.bid, warp.wid, warp.pc, word,
                               exec_after=warp.s[31], event=EV_FAULT,
                               fault=(f.code, f.lane, f.info))
            self._set_status(f.code, warp.pc, warp.block.bid, warp.wid, f.lane, f.info)
            self.running = False
        self.steps += 1
        if self.keep_trace:
            self.trace.append(rec)
        blk = warp.block
        if self.running and blk.finished():
            self.done.add(blk.bid)
            if blk in self.resident:
                self.resident.remove(blk)
            if not self.keep_state:
                del self.blocks[blk.bid]
            if len(self.done) == self.grid:
                self.running = False
        return rec

    def run(self) -> None:
        """Run the launch to completion under the reference schedule."""
        self._fill()
        while self.running:
            if self.steps >= self.max_steps:
                raise SimTimeout(f"kernel exceeded {self.max_steps} steps")
            eligible = [w.key for b in self.resident for w in b.warps if w.state == RUN]
            assert eligible, "no eligible warp: BAR-3 guarantees one exists"
            self.last = pick(eligible, self.last, self.prng)
            self._retire(self.blocks[self.last[0]].warps[self.last[1]])
            self._fill()

    # ---- driven mode (an external harness chooses the warp) -------------------------------
    def runnable(self) -> list[tuple[int, int]]:
        """(block, warp) pairs that may be stepped: every warp of every unfinished block that
        is RUN. Blocks are made resident on first use."""
        if not self.running:
            return []
        out = []
        for bid in range(self.grid):
            if bid in self.done:
                continue
            b = self.blocks.get(bid)
            if b is None:
                out += [(bid, w) for w in range(self.num_warps)]
            else:
                out += [w.key for w in b.warps if w.state == RUN]
        return out

    def step(self, block: int, warp: int) -> CommitRecord:
        """Retire one instruction of (block, warp). Used by lockstep co-simulation."""
        assert self.running and 0 <= block < self.grid and block not in self.done, "not runnable"
        b = self.blocks.get(block) or self._new_block(block)
        w = b.warps[warp]
        assert w.state == RUN, f"warp {(block, warp)} is not runnable"
        return self._retire(w)

    def warp_state(self, block: int, warp: int):
        """(pc, s[32], v[32][32], state) of a warp. After the kernel ends this needs
        keep_state=True."""
        w = self.blocks[block].warps[warp]
        return w.pc, list(w.s), [list(r) for r in w.v], w.state


__all__ = ["Gpu", "SimTimeout", "EXITED", "RUN"]
