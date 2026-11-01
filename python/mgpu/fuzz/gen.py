"""Random MGPU-32 program generators for differential and equivalence testing.

Three generators:

* ``mode="diff"`` - anything goes: every instruction, random EXEC manipulation, faults,
  illegal words, races, CLOCK reads. Used to compare backends under one fixed schedule,
  where even racy programs must produce identical traces.
* ``mode="eq"`` - well-defined programs (docs/isa/equivalence.md, W1-W6) by construction:
  private output slots, double-buffered shared memory separated by BAR, commutative atomics
  whose results are discarded, no CLOCK, no faults. Their final memory and per-warp streams
  must not depend on the schedule.
* ``raw_case`` - mostly-random 32-bit words, to exercise the decoder and fault paths.

Termination is guaranteed for diff/eq programs: branches only go forward, except the
back edge of loops whose counters live in reserved registers that random code never writes.
"""

import random
from dataclasses import dataclass, field

from ..fp.testvals import random_f32
from ..isa import BY_MNEMONIC, encode

MEM_SIZE = 0x40000
CODE = 0x1000
DATA = 0x8000          # input words (read-only in eq mode)
DATA_WORDS = 1024
OUT = 0x10000          # eq mode: private per-thread output slots
OUT_SLOTS = 8
SOUT = 0x18000         # eq mode: private per-warp scalar output slots
SOUT_SLOTS = 8
ACC = 0x1C000          # eq mode: global atomic accumulators, one word per op kind
SH_SLOTS = 8           # eq mode: shared slots per thread and buffer

# Reserved registers: s1-s4 hold region bases (args), s20-s29 are structure/temps,
# v28-v31 are address temporaries. Random code never writes them.
S_DST = [*range(5, 20), 30]
V_DST = list(range(0, 28))
ATOM_OPS = ["ADD", "MIN", "MAX", "AND", "OR"]

INT_IMMS = [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 0x7FFF, 0x8000, 0xFFFF, 0x1234]


@dataclass
class FuzzCase:
    seed: int
    mode: str
    words: list
    memory: dict = field(default_factory=dict)       # address -> list of words
    grid: int = 1
    block: int = 32
    args: list = field(default_factory=list)
    params: dict = field(default_factory=dict)       # extra Gpu parameters
    mem_size: int = MEM_SIZE
    entry: int = CODE

    def describe(self) -> str:
        return (f"seed={self.seed} mode={self.mode} grid={self.grid} block={self.block} "
                f"params={self.params} words={len(self.words)}")


class Builder:
    """Emits instruction words with forward/backward label fixups."""

    def __init__(self, base: int):
        self.base = base
        self.words: list[int] = []
        self.labels: dict[int, int] = {}
        self.fixups: list[tuple[int, int, str]] = []
        self.next_label = 0

    def emit(self, mn: str, **f) -> None:
        self.words.append(encode(mn, **f))

    def raw(self, word: int) -> None:
        self.words.append(word & 0xFFFFFFFF)

    def new_label(self) -> int:
        self.next_label += 1
        return self.next_label

    def place(self, label: int) -> None:
        self.labels[label] = len(self.words)

    def branch(self, mn: str, label: int) -> None:
        """mn is "S_J" or "S_BZ:<reg>" / "S_BNZ:<reg>"."""
        self.fixups.append((len(self.words), label, mn))
        self.words.append(0)

    def li(self, rd: int, value: int) -> None:
        value &= 0xFFFFFFFF
        self.emit("S_LUI", rd=rd, imm=value >> 16)
        self.emit("S_ORI", rd=rd, ra=rd, imm=value & 0xFFFF)

    def finish(self) -> list[int]:
        for idx, label, mn in self.fixups:
            off = self.labels[label] - (idx + 1)
            if mn == "S_J":
                self.words[idx] = encode("S_J", offset=off)
            else:
                mnem, reg = mn.split(":")
                self.words[idx] = encode(mnem, rd=int(reg), imm=off)
        return self.words


class Gen:
    def __init__(self, seed: int, mode: str):
        assert mode in ("diff", "eq")
        self.rng = random.Random(seed)
        self.seed = seed
        self.mode = mode
        self.b = Builder(CODE)
        self.phase = 0          # eq mode: shared-memory buffer phase
        self.loop_depth = 0

    # ---- operand helpers ---------------------------------------------------------------------
    def sdst(self) -> int:
        r = self.rng.random()
        if self.mode == "diff" and r < 0.03:
            return 31
        if r < 0.06:
            return 0
        return self.rng.choice(S_DST)

    def sreg(self) -> int:
        return self.rng.randrange(32)

    def vdst(self) -> int:
        return self.rng.choice(V_DST)

    def vreg(self) -> int:
        return self.rng.randrange(32)

    def imm(self, signed: bool) -> int:
        v = self.rng.choice(INT_IMMS) if self.rng.random() < 0.6 else self.rng.getrandbits(16)
        return (v - 0x10000 if v & 0x8000 else v) if signed else v

    # ---- random straight-line instructions ----------------------------------------------------
    def alu(self) -> None:
        """One random ALU / compare / lane / special-register instruction."""
        rng, b = self.rng, self.b
        groups = ["S_ALU", "V_ALU", "V_FPU", "V_CMP", "V_R4", "V_LANE", "GETSR", "IMM"]
        g = rng.choices(groups, weights=[3, 4, 4, 2, 2, 1, 1, 3])[0]
        cands = [i for i in BY_MNEMONIC.values() if i.group == g] if g != "IMM" else \
            [BY_MNEMONIC[m] for m in ("S_ADDI", "S_ANDI", "S_ORI", "S_LUI", "V_ADDI", "V_ANDI")]
        ins = rng.choice(cands)
        f = dict(rd=0, ra=0, rb=0, rc=0, sb=0, imm=0)
        for tok in ins.syntax:
            if tok == "sd":
                f["rd"] = self.sdst()
            elif tok == "vd":
                f["rd"] = self.vdst()
            elif tok in ("sa", "sb", "sc"):
                f[{"sa": "ra", "sb": "rb", "sc": "rc"}[tok]] = self.sreg()
            elif tok in ("va", "vc"):
                f[{"va": "ra", "vc": "rc"}[tok]] = self.vreg()
            elif tok == "vb|sb":
                f["sb"] = int(rng.random() < 0.4)
                f["rb"] = self.sreg() if f["sb"] else self.vreg()
            elif tok == "shamt":
                f["rb"] = rng.randrange(32)
            elif tok == "simm16":
                f["imm"] = self.imm(True)
            elif tok == "uimm16":
                f["imm"] = self.imm(False)
            elif tok == "sr":
                ok = [0, 1, 2, 3, 4] + ([5] if self.mode == "diff" else [])
                if ins.mnemonic == "V_GETSR":
                    ok += [16, 17, 18]
                f["ra"] = rng.choice(ok)
                if self.mode == "diff" and rng.random() < 0.01:
                    f["ra"] = rng.randrange(32)      # possibly illegal
        b.emit(ins.mnemonic, **f)

    def vaddr(self, base_reg: int, window: int, tmp: int = 28) -> None:
        """v[tmp] = (v[random] & (window - 4)) + s[base_reg]: an aligned address in the window."""
        b = self.b
        b.emit("V_ANDI", rd=tmp, ra=self.vreg(), imm=(window - 4) & 0xFFFC)
        b.emit("V_ADD", rd=tmp, ra=tmp, rb=base_reg, sb=1)

    def mem_diff(self) -> None:
        """Random memory access in diff mode (may fault, may race)."""
        rng, b = self.rng, self.b
        kind = rng.choice(["V_LW", "V_SW", "V_LDS", "V_STS", "S_LW", "S_SW", "ATOM", "ATOM.S"])
        bad = rng.random() < 0.012
        off = rng.choice([0, 4, 8, -4, 64]) + (rng.choice([1, 2, 3, 0x7FF0]) if bad else 0)
        if kind in ("V_LW", "V_SW"):
            self.vaddr(1, DATA_WORDS * 4)
            if bad and rng.random() < 0.5:
                b.emit("V_ADD", rd=28, ra=self.vreg(), rb=self.vreg())
            b.emit(kind, rd=self.vdst() if kind == "V_LW" else self.vreg(), ra=28, imm=off)
        elif kind in ("V_LDS", "V_STS"):
            b.emit("V_ANDI", rd=28, ra=self.vreg(), imm=0x3FF8 if not bad else 0xFFFF)
            b.emit(kind, rd=self.vdst() if kind == "V_LDS" else self.vreg(), ra=28,
                   imm=rng.choice([0, 4]) if not bad else off)
        elif kind in ("S_LW", "S_SW"):
            b.emit("S_ANDI", rd=26, ra=self.sreg(), imm=DATA_WORDS * 4 - 4)
            b.emit("S_ADD", rd=26, ra=26, rb=1)
            b.emit(kind, rd=self.sdst() if kind == "S_LW" else self.sreg(), ra=26, imm=off)
        else:
            op = rng.choice(ATOM_OPS + ["XCHG"])
            if kind == "ATOM":
                self.vaddr(1, 64 if rng.random() < 0.5 else DATA_WORDS * 4)   # small window: collisions
                mn = f"V_ATOM.{op}"
            else:
                b.emit("V_ANDI", rd=28, ra=self.vreg(), imm=0x3C if rng.random() < 0.5 else 0x3FFC)
                mn = f"V_ATOM.{op}.S"
            if bad:
                b.emit("V_ADDI", rd=28, ra=28, imm=rng.choice([1, 2, 0x4000]))
            sb = int(rng.random() < 0.3)
            b.emit(mn, rd=self.vdst(), ra=28, rb=self.sreg() if sb else self.vreg(), sb=sb)

    def mem_eq(self) -> None:
        """Race-free memory access in eq mode."""
        rng, b = self.rng, self.b
        kind = rng.choices(["load_in", "store_out", "load_out", "sstore", "shared", "atom"],
                           weights=[3, 3, 1, 1, 3, 2])[0]
        if kind == "load_in":                        # read-only inputs
            self.vaddr(1, DATA_WORDS * 4)
            b.emit("V_LW", rd=self.vdst(), ra=28, imm=0)
        elif kind in ("store_out", "load_out"):      # this thread's private slots
            slot = rng.randrange(OUT_SLOTS)
            b.emit("V_ADD", rd=29, ra=31, rb=2, sb=1)     # v31 = GTID * OUT_SLOTS * 4 (prologue)
            if kind == "store_out":
                b.emit("V_SW", rd=self.vreg(), ra=29, imm=4 * slot)
            else:
                b.emit("V_LW", rd=self.vdst(), ra=29, imm=4 * slot)
        elif kind == "sstore":                       # this warp's private scalar slots
            b.emit("S_SW", rd=self.sreg(), ra=27, imm=4 * rng.randrange(SOUT_SLOTS))
        elif kind == "shared":
            buf_now, buf_prev = self.phase % 2, (self.phase + 1) % 2
            slot = rng.randrange(SH_SLOTS)
            r = rng.random()
            if r < 0.45:     # write own slot in the current buffer
                b.emit("V_STS", rd=self.vreg(), ra=30, imm=4 * (buf_now * SH_SLOTS + slot))
            elif r < 0.7:    # read own slot (either buffer)
                b.emit("V_LDS", rd=self.vdst(), ra=30, imm=4 * (rng.randrange(2) * SH_SLOTS + slot))
            else:            # read a neighbour's slot from the previous buffer
                b.emit("V_ADDI", rd=28, ra=30, imm=2 * SH_SLOTS * 4 * rng.randrange(1, 8))
                b.emit("V_AND", rd=28, ra=28, rb=25, sb=1)    # s25 = shared window mask
                b.emit("V_LDS", rd=self.vdst(), ra=28, imm=4 * (buf_prev * SH_SLOTS + slot))
        else:                # commutative atomic, result discarded
            op_i = rng.randrange(len(ATOM_OPS))
            sb = int(rng.random() < 0.3)
            if rng.random() < 0.5:
                b.emit("V_ADD", rd=28, ra=0, rb=4, sb=1)      # broadcast &ACC
                b.emit("V_ADDI", rd=28, ra=28, imm=4 * op_i)
                mn = f"V_ATOM.{ATOM_OPS[op_i]}"
            else:
                b.emit("V_ADDI", rd=28, ra=0, imm=0x3F00 + 4 * op_i)   # shared accumulators
                mn = f"V_ATOM.{ATOM_OPS[op_i]}.S"
            b.emit(mn, rd=0, ra=28, rb=self.sreg() if sb else self.vreg(), sb=sb)

    # ---- structure ----------------------------------------------------------------------------
    def block_of(self, n: int, depth: int, allow_bar: bool) -> None:
        for _ in range(n):
            self.item(depth, allow_bar)

    def item(self, depth: int, allow_bar: bool) -> None:
        rng, b = self.rng, self.b
        r = rng.random()
        if r < 0.55 or depth >= 3:
            self.alu()
        elif r < 0.75:
            self.mem_diff() if self.mode == "diff" else self.mem_eq()
        elif r < 0.82:                              # forward skip on a scalar condition
            end = b.new_label()
            b.branch(f"{rng.choice(['S_BZ', 'S_BNZ'])}:{self.sreg()}", end)
            self.block_of(rng.randint(1, 5), depth + 1, False)
            if rng.random() < 0.15:
                b.emit("EXIT")
            b.place(end)
        elif r < 0.88:                              # divergent region under a narrowed EXEC
            save = 21 + depth
            b.emit("S_ADD", rd=save, ra=31, rb=0)
            cmp = rng.choice([m for m in BY_MNEMONIC if m.startswith("V_CMP.")])
            sb = int(rng.random() < 0.4)
            b.emit(cmp, rd=26, ra=self.vreg(), rb=self.sreg() if sb else self.vreg(), sb=sb)
            b.emit("S_AND", rd=31, ra=31, rb=26)
            skip = b.new_label()
            if rng.random() < 0.5:
                b.branch("S_BZ:31", skip)
            self.block_of(rng.randint(1, 5), depth + 1, False)
            b.place(skip)
            b.emit("S_ADD", rd=31, ra=save, rb=0)
        elif r < 0.93 and self.loop_depth < 2:      # bounded loop, uniform trip count
            ctr = 20 if self.loop_depth == 0 else 24
            self.loop_depth += 1
            b.emit("S_ADDI", rd=ctr, ra=0, imm=rng.randint(1, 3))
            top = b.new_label()
            b.place(top)
            # In eq mode a BAR inside a loop body would flip the shared-buffer parity on
            # every iteration, so loop bodies are barrier-free there.
            self.block_of(rng.randint(1, 6), depth + 1, allow_bar and self.mode == "diff")
            b.emit("S_ADDI", rd=ctr, ra=ctr, imm=-1)
            b.branch(f"S_BNZ:{ctr}", top)
            self.loop_depth -= 1
        elif r < 0.96 and allow_bar:
            b.emit("BAR")
            self.phase += 1
        elif r < 0.98:
            end = b.new_label()
            b.branch("S_J", end)
            self.block_of(rng.randint(0, 2), depth + 1, False)
            b.place(end)
        elif self.mode == "diff":
            if rng.random() < 0.1:                  # illegal or random word
                b.raw(rng.choice([0x0E << 26, 0x3F << 26 | rng.getrandbits(26), 0x02 << 26 | 0x11,
                                  rng.getrandbits(32)]))
            else:
                b.emit(rng.choice(["NOP", "FENCE"]))
        else:
            b.emit(rng.choice(["NOP", "FENCE"]))

    def prologue(self, block_dim: int) -> None:
        """Give registers interesting values; set up eq-mode addressing registers."""
        rng, b = self.rng, self.b
        b.emit("V_GETSR", rd=1, ra=17)                       # LANE_ID
        b.emit("V_GETSR", rd=2, ra=18)                       # GTID
        for vr in (3, 4, 5, 6):
            self.vaddr(1, DATA_WORDS * 4)
            b.emit("V_LW", rd=vr, ra=28, imm=0)
        for sr in rng.sample(S_DST, 4):
            b.li(sr, random_f32(rng) if rng.random() < 0.5 else rng.getrandbits(32))
        if self.mode == "eq":
            b.emit("V_GETSR", rd=31, ra=18)                  # v31 = GTID * OUT_SLOTS * 4
            b.emit("V_SHLI", rd=31, ra=31, rb=(OUT_SLOTS * 4).bit_length() - 1)
            b.emit("V_GETSR", rd=30, ra=16)                  # v30 = TID * 2 * SH_SLOTS * 4
            b.emit("V_SHLI", rd=30, ra=30, rb=(2 * SH_SLOTS * 4).bit_length() - 1)
            b.li(25, block_dim * 2 * SH_SLOTS * 4 - 1)       # shared window mask (power of 2)
            b.emit("S_GETSR", rd=27, ra=0)                   # s27 = &SOUT[warp_global] slots
            b.emit("S_GETSR", rd=26, ra=4)
            b.emit("S_MUL", rd=27, ra=27, rb=26)
            b.emit("S_GETSR", rd=26, ra=3)
            b.emit("S_ADD", rd=27, ra=27, rb=26)
            b.emit("S_SHLI", rd=27, ra=27, rb=(SOUT_SLOTS * 4).bit_length() - 1)
            b.emit("S_ADD", rd=27, ra=27, rb=3)
            for buf in range(2):                              # initialize own shared slots (W2)
                for slot in range(SH_SLOTS):
                    b.emit("V_STS", rd=rng.randrange(1, 7), ra=30, imm=4 * (buf * SH_SLOTS + slot))
            b.emit("V_ADDI", rd=28, ra=0, imm=0x3F00)        # shared accumulators (lane 0 inits)
            b.emit("S_ADD", rd=26, ra=31, rb=0)
            b.emit("S_GETSR", rd=29, ra=3)                    # only warp 0, lane 0 initializes
            skip = b.new_label()
            b.branch("S_BNZ:29", skip)
            b.emit("S_ADDI", rd=31, ra=0, imm=1)
            for i in range(len(ATOM_OPS)):
                b.emit("V_STS", rd=0, ra=28, imm=4 * i)
            b.place(skip)
            b.emit("S_ADD", rd=31, ra=26, rb=0)
            b.emit("BAR")
            self.phase = 1


def random_case(seed: int, mode: str = "diff", size: int | None = None) -> FuzzCase:
    """A random structured program plus memory image and launch configuration."""
    g = Gen(seed, mode)
    rng = g.rng
    if mode == "eq":
        block = rng.choice([32, 64, 128])
        grid = rng.randint(1, 3)
    else:
        block = rng.choice([1, 17, 32, 33, 64, 96])
        grid = rng.randint(1, 3)
    g.prologue(block)
    g.block_of(size or rng.randint(20, 120), 0, True)
    g.b.emit("EXIT")
    words = g.b.finish()
    data = [random_f32(rng) if rng.random() < 0.6 else rng.getrandbits(32) for _ in range(DATA_WORDS)]
    memory = {DATA: data}
    if mode == "eq":
        acc_init = [0, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0]   # ADD, MIN, MAX, AND, OR
        memory[ACC] = acc_init
    params = {"resident_blocks": rng.randint(1, 3),
              "seed": None if rng.random() < 0.4 else rng.getrandbits(32),
              "shared_poison": rng.getrandbits(32), "max_steps": 50_000}
    return FuzzCase(seed, mode, words, memory, grid, block, [DATA, OUT, SOUT, ACC], params)


def raw_case(seed: int, n: int = 64) -> FuzzCase:
    """Mostly-random instruction words: decoder and fault-path fuzzing."""
    rng = random.Random(seed)
    words = []
    names = list(BY_MNEMONIC)
    for _ in range(n):
        r = rng.random()
        if r < 0.6:   # valid encoding with random fields
            words.append(encode(rng.choice(names), rd=rng.randrange(32), ra=rng.randrange(32),
                                rb=rng.randrange(32), rc=rng.randrange(32), sb=rng.getrandbits(1),
                                imm=rng.getrandbits(16), offset=rng.randrange(-8, 8)))
        else:
            words.append(rng.getrandbits(32))
    words.append(encode("EXIT"))
    data = [rng.getrandbits(32) for _ in range(DATA_WORDS)]
    params = {"resident_blocks": rng.randint(1, 2), "seed": rng.getrandbits(32),
              "max_steps": 5_000}
    return FuzzCase(seed, "raw", words, {DATA: data}, rng.randint(1, 2), rng.choice([1, 32, 40]),
                    [DATA, OUT, SOUT, ACC], params, mem_size=0x20000)
