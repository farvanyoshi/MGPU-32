"""Execution statistics computed from a commit trace (instruction mix, EXEC utilization)."""

from collections import Counter
from dataclasses import dataclass, field

from .isa import IllegalInstruction, decode
from .trace import EV_FAULT


@dataclass
class TraceStats:
    instructions: int = 0
    by_group: Counter = field(default_factory=Counter)
    by_mnemonic: Counter = field(default_factory=Counter)
    vector_instructions: int = 0
    active_lanes: int = 0          # summed over vector instructions
    memory_words: int = 0          # words written to memory
    barriers: int = 0

    @property
    def simd_utilization(self) -> float:
        """Average fraction of the 32 lanes active in vector instructions."""
        return self.active_lanes / (32 * self.vector_instructions) if self.vector_instructions else 0.0

    def report(self, top: int = 12) -> str:
        lines = [f"instructions        {self.instructions}",
                 f"vector instructions {self.vector_instructions} "
                 f"(SIMD utilization {100 * self.simd_utilization:.1f}% of 32 lanes)",
                 f"memory words written {self.memory_words}, barriers {self.barriers}",
                 "by group:"]
        for g, n in self.by_group.most_common():
            lines.append(f"  {g:10} {n:9}  {100 * n / max(1, self.instructions):5.1f}%")
        lines.append(f"top {top} instructions:")
        for m, n in self.by_mnemonic.most_common(top):
            lines.append(f"  {m:14} {n:9}")
        return "\n".join(lines)


def initial_exec(block_dim: int, warp: int) -> int:
    live = max(0, min(32, block_dim - 32 * warp))
    return (1 << live) - 1


def trace_stats(records, block_dim: int) -> TraceStats:
    """Summarize a launch. EXEC before each instruction is the warp's previous exec_after
    (or the launch EXEC for its first instruction)."""
    st = TraceStats()
    exec_before: dict = {}
    for r in records:
        key = (r.block, r.warp)
        ex = exec_before.get(key, initial_exec(block_dim, r.warp))
        exec_before[key] = r.exec_after
        if r.event == EV_FAULT:
            continue
        try:
            ins = decode(r.word).instr
        except IllegalInstruction:
            continue
        st.instructions += 1
        st.by_group[ins.group] += 1
        st.by_mnemonic[ins.mnemonic] += 1
        if ins.exec == "mask":
            st.vector_instructions += 1
            st.active_lanes += bin(ex).count("1")
        st.memory_words += len(r.mwrites)
        st.barriers += ins.mnemonic == "BAR"
    return st


def timing_text(rep: dict) -> str:
    """Human-readable timing-model report (Device.timing_report)."""
    lines = [f"cycles              {rep['cycles']:,}",
             f"instructions        {rep['issued']:,}  (IPC {rep['ipc']:.3f}, "
             f"issue slots used {100 * rep['issue_utilization']:.1f}%)",
             f"SIMD utilization    {100 * rep['simd_utilization']:.1f}% of 32 lanes",
             "unit busy:          " + ", ".join(
                 f"{u} {100 * b / max(1, rep['core_active_cycles']):.0f}%"
                 for u, b in rep["busy_by_unit"].items()),
             "idle cycles:        " + (", ".join(
                 f"{k} {v:,}" for k, v in rep["stalls"].items() if v) or "none"),
             f"memory segments     {rep['mem_reads']:,} read, {rep['mem_writes']:,} write, "
             f"{rep['mem_atomics']:,} atomic",
             f"I-cache             {rep['icache_hits']:,} hits, {rep['icache_misses']:,} misses",
             f"bank-conflict cycles {rep['shared_conflict_cycles']:,}"]
    return "\n".join(lines)
