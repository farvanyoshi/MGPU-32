"""mgpu_vexec (through mgpu_vexec_probe) against the reference model: whole instructions,
all 32 lanes, random EXEC and registers. Checks every retire record and its cycle."""

import os
import random
from collections import deque
from types import SimpleNamespace

import cocotb
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from mgpu.fp.testvals import SPECIALS
from mgpu.isa import BY_MNEMONIC, assemble_one, decode
from mgpu.model.execute import execute, sext16
from mgpu.model.state import Warp

from tbutil import pack, start, unpack

KIND = int(os.environ.get("KIND", "0"))
LANES = int(os.environ.get("LANES", "8"))
LAT = int(os.environ.get("LAT", "2"))
P = 32 // LANES
UNIT = ("VALU", "FPU", "SFU")[KIND]
M = 0xFFFFFFFF
EDGES = [0, 1, 2, 31, 32, 0x7FFFFFFF, 0x80000000, M]
MNEMONICS = [m for m, i in BY_MNEMONIC.items() if i.unit == UNIT and i.group != "GETSR"]
XLANE = ("V_SHFL", "V_READLANE")


def asm_text(rng, mn):
    ins = BY_MNEMONIC[mn]
    names = []
    for f in ins.syntax:
        r = rng.randrange(32)
        if f in ("vd", "va", "vc"):
            names.append(f"v{r}")
        elif f in ("sd", "sc", "sb"):
            names.append(f"s{r}")
        elif f == "vb|sb":
            names.append(f"{rng.choice('vs')}{r}")
        elif f == "shamt":
            names.append(str(r))
        elif f == "simm16":
            names.append(str(rng.randrange(-32768, 32768)))
        elif f == "uimm16":
            names.append(str(rng.randrange(65536)))
        else:
            raise ValueError(f)
    return f"{mn} " + ", ".join(names)


def value(rng):
    r = rng.random()
    if r < 0.3:
        return rng.choice(SPECIALS if KIND else EDGES)
    if KIND and r < 0.7:                       # ordinary floats around 1.0
        return (rng.randrange(100, 155) << 23) | rng.getrandbits(23) | (rng.getrandbits(1) << 31)
    return rng.getrandbits(32)


def exec_mask(rng):
    return rng.choice([M, M, 0, 1 << rng.randrange(32), rng.getrandbits(32), rng.getrandbits(32)])


def make_case(rng, pc):
    """Return (probe inputs, expected record fields) for one random instruction."""
    mn = rng.choice(MNEMONICS)
    word = assemble_one(asm_text(rng, mn))
    d = decode(word)
    s = [0] + [value(rng) for _ in range(31)]
    s[31] = exec_mask(rng)
    v = [[0] * 32] + [[value(rng) for _ in range(32)] for _ in range(31)]
    syn = d.instr.syntax
    if "simm16" in syn:
        b = sext16(d.imm16) & M
    elif "uimm16" in syn:
        b = d.imm16
    elif "shamt" in syn:
        b = d.rb
    else:
        b = s[d.rb]
    scalar_b = d.sb or "vb|sb" not in syn
    ins = dict(pc=pc, word=word, exec=s[31], sa=s[d.ra], sb=b, sc=s[d.rc], sd=s[d.rd],
               va=pack(v[d.ra]), vb=pack([b] * 32 if scalar_b else v[d.rb]),
               vc=pack(v[d.rc]), vd=pack(v[d.rd]))

    w = Warp.__new__(Warp)
    w.block, w.wid, w.pc, w.s, w.v = SimpleNamespace(bid=0), 0, 0, s, v
    gpu = SimpleNamespace(mem_size=1 << 20, load_word=lambda a: word)
    rec = execute(gpu, w, 0)
    exp = dict(pc=pc, swrite=rec.swrite, vwrite=rec.vwrite, exec_after=rec.exec_after,
               old_vd=list(unpack(ins["vd"], 32)))
    return mn, ins, exp


def read_record(dut):
    sw = (int(dut.sw_reg.value), int(dut.sw_val.value)) if int(dut.sw_valid.value) else None
    vw = (int(dut.vw_reg.value), tuple(unpack(dut.vw_val.value, 32))) if int(dut.vw_valid.value) else None
    return dict(pc=int(dut.out_pc.value), swrite=sw, vwrite=vw, exec_after=int(dut.exec_after.value))


@cocotb.test()
async def random_instructions(dut):
    await start(dut)
    rng = random.Random(1000 + KIND * 10 + LANES)
    n_cases = int(os.environ.get("CASES", "300"))
    pending: deque = deque()
    writes = []                                   # register writes since the last record
    free_at, issued, done, it = 0, 0, 0, 0
    while done < n_cases:
        dut.in_valid.value = 0
        if issued < n_cases and it >= free_at and rng.random() < 0.7:
            mn, ins, exp = make_case(rng, issued * 4)
            for k, x in ins.items():
                getattr(dut, k).value = x
            dut.in_valid.value = 1
            xl = mn in XLANE
            # completion cycle = entry + occupancy + 1 (latency 2) or entry + P + LAT - 1;
            # sampled one iteration earlier (tbutil cycle model)
            due = it + (2 * P if xl else P + LAT - 2)
            pending.append((due, mn, exp))
            free_at = it + (2 * P if xl else P)
            issued += 1
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.vwr_valid.value):
            writes.append((int(dut.vwr_rd.value), int(dut.vwr_mask.value), unpack(dut.vwr_data.value, 32)))
        if int(dut.out_valid.value):
            assert pending, f"unexpected output in iteration {it}"
            due, mn, exp = pending.popleft()
            got = read_record(dut)
            assert due == it, f"{mn}: completed in iteration {it}, expected {due}"
            for k in ("pc", "swrite", "vwrite", "exec_after"):
                assert got[k] == exp[k], f"{mn} (pc {exp['pc']:#x}): {k} {got[k]} != {exp[k]}"
            # the register writes leave the destination with the record's lanes
            if exp["vwrite"]:
                reg = list(exp["old_vd"])
                for rd, mask, data in writes:
                    assert rd == exp["vwrite"][0], f"{mn}: write to v{rd}"
                    for i in range(32):
                        if mask >> i & 1:
                            reg[i] = data[i]
                assert tuple(reg) == exp["vwrite"][1], f"{mn} (pc {exp['pc']:#x}): register writes {reg}"
            else:
                assert not writes, f"{mn}: unexpected register writes"
            writes = []
            done += 1
        else:
            assert not pending or pending[0][0] != it, f"{pending[0][1]}: missing completion in iteration {it}"
        await FallingEdge(dut.clk)
        it += 1
    dut._log.info(f"{done} instructions on {UNIT} (L={LANES}, P={P}, LAT={LAT}) matched")
