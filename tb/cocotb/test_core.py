"""mgpu_core in lockstep with the Rust engine (lockstep.py): directed programs and random
ALU-only programs. Every retired instruction is compared under rule E4; selected
programs also check the launch's cycle count against the timing model (core.md 11)."""

import os
import random

import cocotb

from mgpu.asm import assemble
from mgpu.fuzz import gen as fuzz_gen

from lockstep import Launch, Lockstep, model_cycles

W = int(os.environ.get("W", "8"))
L = int(os.environ.get("L", "8"))
S = int(os.environ.get("S", "2"))
CODE = 0x1000
T_MEM, T_SEG = 100, 8


def timed_cycles(ln: Launch) -> int:
    return model_cycles(ln, W, L, S, T_MEM, T_SEG)


async def lockstep(dut, src_or_words, check_cycles=False, **kw):
    words = src_or_words if isinstance(src_or_words, list) else assemble(src_or_words, CODE).words
    ln = Launch(words, entry=CODE, **kw)
    res = await Lockstep(dut, ln, W, T_MEM, T_SEG).run()
    want = timed_cycles(ln)
    msg = f"{res.retired} instructions retired in lockstep, {res.cycles} cycles (timing model {want})"
    dut._log.info(msg)
    if check_cycles:
        assert res.cycles == want, msg
    return res


@cocotb.test()
async def scalar_chain(dut):
    for n in (0, 1, 5):
        res = await lockstep(dut, "S_ADDI s1, s1, 1\n" * n + "EXIT", check_cycles=True)
        assert res.cycles == 116 + 4 * n


@cocotb.test()
async def vector_chain(dut):
    await lockstep(dut, "V_ADDI v1, v1, 1\n" * 6 + "EXIT", check_cycles=True)
    await lockstep(dut, "V_FADD v1, v1, v1\n" * 4 + "V_FRCP v2, v1\nEXIT", check_cycles=True)


@cocotb.test()
async def scalar_ops(dut):
    await lockstep(dut, """
        S_LUI  s9, 0x8000
        S_ORI  s9, s9, 0x1234
        S_ADDI s10, s0, -7
        S_ADD  s11, s9, s10
        S_SUB  s12, s10, s9
        S_MUL  s13, s9, s10
        S_AND  s14, s9, s10
        S_OR   s15, s9, s10
        S_XOR  s16, s9, s10
        S_ANDN s17, s9, s10
        S_SHL  s18, s9, s10
        S_SHR  s19, s9, s10
        S_SRA  s20, s9, s10
        S_SLT  s21, s10, s9
        S_SLTU s22, s10, s9
        S_MIN  s23, s9, s10
        S_MAX  s24, s9, s10
        S_SHLI s25, s9, 3
        S_SRAI s26, s9, 31
        S_POPC s27, s9
        S_FF1  s28, s9
        S_ANDI s29, s10, 0xF0F0
        S_GETSR s1, BLOCK_ID
        S_GETSR s2, BLOCK_DIM
        S_GETSR s3, GRID_DIM
        S_GETSR s4, WARP_ID
        S_GETSR s5, NUM_WARPS
        S_GETSR s6, CLOCK
        S_GETSR s7, CLOCK
        NOP
        FENCE
        EXIT""", block=96, grid=2, check_cycles=True)


@cocotb.test()
async def vector_ops(dut):
    await lockstep(dut, """
        V_GETSR v1, LANE_ID
        V_GETSR v2, TID
        V_GETSR v3, GTID
        V_GETSR v4, CLOCK
        S_LI   s9, 0x3FC00000          ; 1.5
        V_ADD  v5, v1, v2
        V_SUB  v6, v1, s9
        V_MUL  v7, v2, v3
        V_SHLI v8, v1, 27
        V_SRA  v9, v8, v1
        V_MIN  v10, v8, v9
        V_ADDI v11, v1, -16
        V_ANDI v12, v3, 0x55
        V_IMAD v13, v1, v2, v3
        V_CMP.LT s10, v1, v11
        V_CMP.GEU s11, v11, s9
        V_SEL  v14, v1, v2, s10
        V_SHFL v15, v3, v12
        V_SHFL v16, v3, s11
        V_READLANE s12, v3, s10
        V_I2F  v17, v11
        V_FMUL v18, v17, s9
        V_FADD v19, v18, v17
        V_FFMA v20, v18, v19, v17
        V_FMIN v21, v19, v20
        V_CMP.FLT s13, v18, v17
        V_F2I  v22, v20
        V_FRCP v23, v18
        V_FRSQ v24, v19
        V_FEX2 v25, v17
        V_FLG2 v26, v19
        EXIT""", block=64, check_cycles=True)


@cocotb.test()
async def divergence_and_loops(dut):
    await lockstep(dut, """
               V_GETSR v1, LANE_ID
               S_ADDI s1, s0, 10
               S_ADDI s2, s0, 0
       loop:   S_ADD  s2, s2, s1
               V_ADD  v2, v2, v1
               S_ADDI s1, s1, -1
               S_BNZ  s1, loop
               S_BZ   s1, skip
               S_ADDI s2, s2, 1000
       skip:   S_J    end
               S_ADDI s2, s2, 2000
       end:    V_ADDI v3, v0, 16
               V_CMP.LT  s10, v1, v3
               S_MOV     s11, s31
               S_MOV     s31, s10
               S_BZ      s31, else
               V_ADDI    v4, v1, 1000
       else:   S_ANDN    s31, s11, s10
               S_BZ      s31, endif
               V_ADDI    v4, v1, 2000
       endif:  S_MOV     s31, s11
               S_GETSR   s3, WARP_ID
               S_BNZ     s3, out
               EXIT
       out:    V_FADD    v5, v4, v4
               EXIT""", block=200, grid=3, check_cycles=True)


@cocotb.test()
async def illegal_instruction(dut):
    words = assemble("V_GETSR v1, TID\nV_FRCP v2, v1\nS_ADDI s1, s1, 1\nS_GETSR s2, WARP_ID\n"
                     "S_BNZ s2, bad\nEXIT\nbad: EXIT", CODE).words
    words[-1] = 0xFC00_0000            # warp 1+ reach an illegal word
    res = await lockstep(dut, words, block=128, check_cycles=True)
    assert res.fault and res.fault[0] == 1


@cocotb.test()
async def fetch_fault(dut):
    res = await lockstep(dut, "V_ADDI v1, v1, 3\nS_J far\nEXIT\nfar:", block=64, mem_size=0x1010,
                         check_cycles=True)
    assert res.fault and res.fault[0] == 3


class AluGen(fuzz_gen.Gen):
    """The fuzzer's structured generator without memory instructions or barriers."""

    def mem_diff(self):
        self.alu()

    def prologue(self, block_dim):
        b = self.b
        b.emit("V_GETSR", rd=1, ra=17)
        b.emit("V_GETSR", rd=2, ra=18)
        for vr in (3, 4, 5, 6):
            b.emit("V_I2F", rd=vr, ra=self.rng.choice([1, 2]))
        for sr in self.rng.sample(fuzz_gen.S_DST, 4):
            b.li(sr, fuzz_gen.random_f32(self.rng) if self.rng.random() < 0.5 else self.rng.getrandbits(32))


@cocotb.test()
async def random_alu_programs(dut):
    n = int(os.environ.get("SEEDS", "12"))
    for seed in range(n):
        g = AluGen(seed, "diff")
        block = g.rng.choice([1, 17, 32, 33, 64, 96, 32 * W])
        grid = g.rng.randint(1, 2)
        g.prologue(block)
        g.block_of(g.rng.randint(20, 80), 0, False)
        g.b.emit("EXIT")
        words = g.b.finish()
        dut._log.info(f"seed {seed}: {len(words)} words, grid {grid} x {block}")
        await lockstep(dut, words, grid=grid, block=block)


DATA = 0x8000
LANE_ADDR = "V_GETSR v1, LANE_ID\nV_SHLI v1, v1, {sh}\nV_ADD {dst}, v1, s1\n"


@cocotb.test()
async def global_memory(dut):
    words = list(range(100, 100 + 512))
    await lockstep(dut, f"""
        S_LI s1, {DATA}
        {LANE_ADDR.format(sh=2, dst="v2")}
        V_LW v3, 0(v2)                  ; one segment
        V_LW v4, 64(v2)                 ; two segments
        {LANE_ADDR.format(sh=7, dst="v5")}
        V_LW v6, 4(v5)                  ; 32 segments
        V_ADDI v7, v3, 1
        V_SW v7, 0x800(v2)
        V_SW v6, 0x900(v5)              ; 32 store segments
        V_SW v1, 0x804(v0)              ; every lane to one word: the highest lane wins
        S_LW s2, 8(s1)
        S_SW s2, 0x1000(s1)
        S_LI s31, 0x00F0F00F
        V_LW v8, 0x800(v2)
        V_SW v8, 0x2000(v2)
        S_LI s31, 0
        V_LW v9, 0(v2)                  ; EXEC = 0: no request
        V_SW v9, 0(v2)
        S_LW s3, 0x1000(s1)             ; scalar accesses ignore EXEC
        S_LI s31, -1
        V_LW v10, 0x900(v5)
        EXIT""", block=32, grid=2, args=[DATA], memory={DATA: words}, check_cycles=True)


@cocotb.test()
async def shared_memory(dut):
    await lockstep(dut, f"""
        {LANE_ADDR.format(sh=2, dst="v2")}
        V_LDS v3, 0(v2)                 ; poison
        V_STS v1, 0(v2)                 ; conflict-free
        {LANE_ADDR.format(sh=3, dst="v4")}
        V_STS v1, 0x200(v4)             ; two-way conflict
        V_LDS v5, 0x200(v4)
        {LANE_ADDR.format(sh=7, dst="v6")}
        V_STS v1, 4(v6)                 ; 32-way conflict
        V_LDS v7, 4(v6)
        V_LDS v8, 8(v0)                 ; broadcast
        V_STS v1, 0x40(v0)              ; collisions: the highest lane wins
        V_LDS v9, 0x40(v0)
        S_LI s31, 0x0000FF00
        V_LDS v10, 0x200(v4)
        S_LI s31, 0
        V_LDS v11, 0(v2)
        V_STS v11, 0(v2)
        S_LI s31, -1
        EXIT""", block=32, args=[0], check_cycles=True)


@cocotb.test()
async def atomics(dut):
    await lockstep(dut, f"""
        S_LI s1, {DATA}
        {LANE_ADDR.format(sh=2, dst="v2")}
        V_ADDI v3, v1, 1
        V_ATOM.ADD v4, v0, v3           ; every lane on word 0
        V_ADD v5, v0, s1
        V_ATOM.ADD v6, v5, v3           ; one word, ascending lanes
        V_ATOM.MAX v7, v2, v1
        V_ATOM.XCHG v8, v2, s1
        {LANE_ADDR.format(sh=7, dst="v9")}
        V_ATOM.OR v10, v9, v3           ; 32 segments
        V_ATOM.ADD.S v11, v0, v3        ; shared, one word
        V_ADDI v17, v1, -64
        V_ATOM.MIN.S v12, v2, v17       ; signed: negative operands win
        V_ATOM.MIN v18, v2, v17
        V_ATOM.AND.S v13, v2, s1
        S_LI s31, 0
        V_ATOM.ADD v14, v2, v3          ; EXEC = 0
        V_ATOM.ADD.S v15, v2, v3
        S_LI s31, 0x80000001
        V_ATOM.XCHG v16, v2, v3
        S_LI s31, -1
        EXIT""", block=32, args=[DATA], memory={DATA: list(range(64))}, check_cycles=True)


@cocotb.test()
async def memory_faults(dut):
    base = f"S_LI s1, {DATA}\n{LANE_ADDR.format(sh=2, dst='v2')}V_LW v3, 0(v2)\n"
    for bad, code in (("V_LW v4, 2(v2)", 2), ("S_LI s9, 0xFFFFF\nS_LW s4, 1(s9)", 3),
                      ("V_ADDI v5, v1, -5\nV_SHLI v5, v5, 2\nV_STS v1, 0x3FEC(v5)", 3),
                      ("V_ATOM.ADD v4, v2, v1\nV_GETSR v6, LANE_ID\nV_ATOM.ADD v4, v6, v1", 2)):
        res = await lockstep(dut, base + bad + "\nEXIT", block=64, args=[DATA], check_cycles=True)
        assert res.fault and res.fault[0] == code, res.fault


async def suite(dut, progs, label):
    diffs = []
    for p in progs:
        if "src" in p:
            ln = dict(words=assemble(p["src"], CODE).words, grid=p["grid"], block=p["block"],
                      args=p["args"], memory=p["data"], mem_size=p["mem"])
        else:
            ln = dict(words=[], entry=p["entry"], grid=p["grid"], block=p["block"], args=p["args"],
                      memory=p["image"], mem_size=p["mem"])
        dut._log.info(f"{label}: {p['name']}")
        launch = Launch(**{"entry": CODE, **ln})
        res = await Lockstep(dut, launch, W, T_MEM, T_SEG).run(max_cycles=2_000_000)
        want = timed_cycles(launch)
        if res.cycles != want:
            diffs.append((p["name"], res.cycles, want))
    dut._log.info(f"{label}: {len(progs)} programs in lockstep; cycle count differs from the "
                  f"timing model on {len(diffs)}: {diffs}")


@cocotb.test()
async def directed_suite(dut):
    """Every directed program of tests/sim that fits the core."""
    import directed
    await suite(dut, directed.harvest(32 * W), "directed")


@cocotb.test()
async def kernel_suite(dut):
    """Every launch of the kernel suite (tests/kernels) at the small problem sizes."""
    import directed
    await suite(dut, directed.harvest_kernels(32 * W), "kernels")


@cocotb.test()
async def barriers_and_fences(dut):
    """BAR after outstanding stores, FENCE, and a barrier completed by an EXIT."""
    await lockstep(dut, f"""
        S_LI s1, {DATA}
        {LANE_ADDR.format(sh=2, dst="v2")}
        S_GETSR s2, WARP_ID
        S_SHLI s3, s2, 7
        V_ADD v2, v2, s3
        V_SW v1, 0(v2)
        BAR
        V_LW v3, 128(v2)                ; the next warp's stores are visible
        V_SW v3, 0x1000(v2)
        FENCE
        S_ADDI s4, s2, -3
        S_BZ s4, leave                  ; warp 3 exits; the others wait for it at the BAR
        V_STS v1, 0(v2)
        BAR
        V_LDS v4, 4(v2)
        EXIT
leave:  EXIT""", block=256, args=[DATA], check_cycles=True)


@cocotb.test()
async def kill_and_restart(dut):
    """Abort a block in flight, then run another launch on the same core without a reset."""
    loop = f"""S_LI s1, {DATA}
        V_GETSR v1, TID                 ; one word per thread: no races between warps
        V_SHLI v1, v1, 2
        V_ADD v2, v1, s1
        S_ADDI s5, s0, 200
top:    V_LW v3, 0(v2)
        V_ADDI v3, v3, 1
        V_SW v3, 0(v2)
        S_ADDI s5, s5, -1
        S_BNZ s5, top
        EXIT"""
    words = assemble(loop, CODE).words
    ln = Launch(words, entry=CODE, block=128, args=[DATA])
    await Lockstep(dut, ln, W, T_MEM, T_SEG).run(kill_at=900)
    res = await Lockstep(dut, Launch(words, entry=CODE, block=64, args=[DATA]), W, T_MEM, T_SEG).run(reset=False)
    assert res.fault is None
