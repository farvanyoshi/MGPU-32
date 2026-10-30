"""Commit records: the trace format shared by every MGPU-32 simulator backend.

One record is produced per retired (or faulting) instruction. The model and the Rust
engine must produce identical records, and the RTL co-simulation compares against them
(docs/sim/README.md, "Trace format").
"""

from dataclasses import dataclass, field

# Memory spaces in mem-write tuples
GLOBAL = 0
SHARED = 1

ATOM_OPCODE = 0x16

# Events
EV_NONE = 0
EV_EXIT = 1
EV_BAR = 2     # the warp arrived at a barrier (and waits unless this arrival released it)
EV_FAULT = 3


@dataclass
class CommitRecord:
    step: int                 # global step number (= CLOCK value seen by this instruction)
    block: int
    warp: int
    pc: int
    word: int                 # instruction word (0 for a fetch fault)
    swrite: tuple | None = None          # (reg, value); never reg 0
    vwrite: tuple | None = None          # (reg, (32 lane values after the write)); never reg 0
    exec_after: int = 0                  # s31 after the instruction
    mwrites: list = field(default_factory=list)   # [(space, addr, value)] in program order
    event: int = EV_NONE
    released: bool = False               # this instruction completed a barrier (BAR or EXIT)
    fault: tuple | None = None           # (code, lane, info)

    def key(self) -> tuple:
        """Canonical, comparable form (used for differential testing)."""
        return (self.step, self.block, self.warp, self.pc, self.word, self.swrite, self.vwrite,
                self.exec_after, tuple(self.mwrites), self.event, self.released, self.fault)

    def warp_key(self) -> tuple:
        """Form used for per-warp stream comparison (equivalence rule E4): no step number and
        no barrier-release flag (both depend on the schedule). For V_ATOM only the updated
        addresses count: the values depend on other warps' atomics (C-37)."""
        if self.word >> 26 == ATOM_OPCODE and self.fault is None:
            return (self.pc, self.word, self.vwrite and self.vwrite[0], self.exec_after,
                    tuple((sp, a) for sp, a, _ in self.mwrites), self.event)
        return (self.pc, self.word, self.swrite, self.vwrite, self.exec_after,
                tuple(self.mwrites), self.event, self.fault)


def from_tuple(t: tuple) -> CommitRecord:
    """Build a record from the canonical tuple (as returned by the Rust backend)."""
    (step, block, warp, pc, word, sw, vw, ex, mw, ev, rel, fault) = t
    return CommitRecord(step, block, warp, pc, word, sw, vw, ex, list(mw), ev, rel, fault)


def per_warp(records) -> dict:
    """Group records by (block, warp) in program order, for equivalence rule E4."""
    out: dict = {}
    for r in records:
        out.setdefault((r.block, r.warp), []).append(r.warp_key())
    return out
