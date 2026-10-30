"""Instruction semantics of the reference model (ISA sections 6 and 10).

`execute` runs one instruction of one warp. It reads every source first ([SEM-1]),
validates everything that can fault, and only then writes state, so a raised Fault
leaves the machine untouched ([MEM-3], [DEC-4]).
"""

from ..fp import ops as fp
from ..hostregs import E_BARRIER, E_ILLEGAL, E_MISALIGNED, E_RANGE
from ..isa import SR_BY_ID, IllegalInstruction, decode
from ..trace import EV_BAR, EV_EXIT, GLOBAL, SHARED, CommitRecord
from .state import BAR_WAIT, EXITED, SHARED_WORDS, WARP, Fault

M32 = 0xFFFFFFFF
SHARED_SIZE = SHARED_WORDS * 4


def s32(x: int) -> int:
    return x - (1 << 32) if x & 0x80000000 else x


def int_op(op: str, a: int, b: int) -> int:
    """Integer funct table (ISA 5.1). a, b are u32; result is u32."""
    if op == "add":
        return (a + b) & M32
    if op == "sub":
        return (a - b) & M32
    if op == "mul":
        return (a * b) & M32
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    if op == "andn":
        return a & ~b & M32
    if op == "shl":
        return (a << (b & 31)) & M32
    if op == "shr":
        return a >> (b & 31)
    if op == "sra":
        return (s32(a) >> (b & 31)) & M32
    if op == "slt":
        return int(s32(a) < s32(b))
    if op == "sltu":
        return int(a < b)
    if op == "min":
        return a if s32(a) <= s32(b) else b
    if op == "max":
        return a if s32(a) >= s32(b) else b
    if op == "popc":
        return bin(a).count("1")
    if op == "ff1":
        return (a & -a).bit_length() - 1 if a else M32
    raise AssertionError(op)


def int_cond(cond: str, a: int, b: int) -> bool:
    """Integer V_CMP conditions (ISA 5.3)."""
    if cond == "eq":
        return a == b
    if cond == "ne":
        return a != b
    if cond == "lt":
        return s32(a) < s32(b)
    if cond == "le":
        return s32(a) <= s32(b)
    if cond == "gt":
        return s32(a) > s32(b)
    if cond == "ge":
        return s32(a) >= s32(b)
    if cond == "ltu":
        return a < b
    if cond == "leu":
        return a <= b
    if cond == "gtu":
        return a > b
    if cond == "geu":
        return a >= b
    raise AssertionError(cond)


def atom_op(op: str, old: int, x: int) -> int:
    """V_ATOM operations (ISA 6.9)."""
    if op == "xchg":
        return x
    return int_op(op, old, x)


def check(addr: int, size: int) -> int:
    """CHECK of ISA 10.1: 0 if valid, else the error code."""
    if addr & 3:
        return E_MISALIGNED
    if addr > size - 4:
        return E_RANGE
    return 0


def sext16(i: int) -> int:
    return i - 0x10000 if i & 0x8000 else i


def execute(gpu, warp, step: int) -> CommitRecord:
    blk = warp.block
    pc = warp.pc
    if pc > gpu.mem_size - 4:                               # [MEM-7]
        raise Fault(E_RANGE, 0, pc)
    word = gpu.load_word(pc)
    try:
        d = decode(word)
    except IllegalInstruction:
        raise Fault(E_ILLEGAL, 0, word) from None
    ins = d.instr
    rec = CommitRecord(step, blk.bid, warp.wid, pc, word)
    s, v = warp.s, warp.v
    exec_mask = s[31]                                       # [EX-3]: EXEC before this instruction
    active = [i for i in range(WARP) if exec_mask >> i & 1]
    g, op = ins.group, ins.op
    next_pc = (pc + 4) & M32

    def B():
        """Snapshot of the vb|sb operand for all lanes ([ENC-1], [SEM-1])."""
        return [s[d.rb]] * WARP if d.sb else list(v[d.rb])

    def write_s(r, x):
        if r:
            s[r] = x & M32
            rec.swrite = (r, s[r])

    def write_v(r, values: dict):
        """values: lane -> result, for the active lanes only."""
        if r:
            for i, x in values.items():
                v[r][i] = x & M32
            rec.vwrite = (r, tuple(v[r]))

    def lane_addresses(space_size, base_reg, offset):
        addrs = {i: (v[base_reg][i] + offset) & M32 for i in active}
        for i in active:                                    # ascending lane order (10.1)
            code = check(addrs[i], space_size)
            if code:
                raise Fault(code, i, addrs[i])
        return addrs

    if g == "SYS":
        if op == "exit":
            warp.state = EXITED
            rec.event = EV_EXIT
            rec.exec_after = s[31]
            blk.try_release(rec)
            return rec
        if op == "bar":
            if blk.bar_pc is not None and blk.bar_pc != pc:  # [BAR-2]
                raise Fault(E_BARRIER, 0, blk.bar_pc)
            warp.state = BAR_WAIT
            blk.bar_pc = pc
            rec.event = EV_BAR
            rec.exec_after = s[31]
            blk.try_release(rec)                            # sets pc = P + 4 if released
            return rec
        # nop, fence: no architectural effect in a sequential model

    elif g == "S_ALU":
        b = d.rb if "shamt" in ins.syntax else s[d.rb]
        write_s(d.rd, int_op(op, s[d.ra], b))

    elif g == "V_ALU":
        if "shamt" in ins.syntax:
            write_v(d.rd, {i: int_op(op, v[d.ra][i], d.rb) for i in active})
        else:
            b = B()
            write_v(d.rd, {i: int_op(op, v[d.ra][i], b[i]) for i in active})

    elif g == "V_FPU":
        if op in fp.UNARY:
            f = fp.UNARY[op]
            write_v(d.rd, {i: f(v[d.ra][i]) for i in active})
        else:
            f, b = fp.BINARY[op], B()
            write_v(d.rd, {i: f(v[d.ra][i], b[i]) for i in active})

    elif g == "V_CMP":
        b = B()
        if op.startswith("f"):
            hits = [i for i in active if fp.fcmp(op, v[d.ra][i], b[i])]
        else:
            hits = [i for i in active if int_cond(op, v[d.ra][i], b[i])]
        write_s(d.rd, sum(1 << i for i in hits))

    elif g == "V_R4":
        b = B()
        if op == "ffma":
            write_v(d.rd, {i: fp.ffma(v[d.ra][i], b[i], v[d.rc][i]) for i in active})
        elif op == "imad":
            write_v(d.rd, {i: v[d.ra][i] * b[i] + v[d.rc][i] for i in active})
        else:  # sel
            mask = s[d.rc]
            write_v(d.rd, {i: v[d.ra][i] if mask >> i & 1 else b[i] for i in active})

    elif g == "V_LANE":
        src = list(v[d.ra])
        if op == "readlane":
            write_s(d.rd, src[s[d.rb] & 31])
        else:  # shfl
            b = B()
            write_v(d.rd, {i: src[b[i] & 31] for i in active})

    elif g == "GETSR":
        sr = SR_BY_ID.get(d.ra)
        if sr is None or (ins.mnemonic == "S_GETSR" and sr["kind"] == "lane"):  # [SR-1], [SR-2]
            raise Fault(E_ILLEGAL, 0, word)
        if ins.mnemonic == "S_GETSR":
            write_s(d.rd, gpu.sr_value(sr["name"], warp, 0, step))
        else:
            write_v(d.rd, {i: gpu.sr_value(sr["name"], warp, i, step) for i in active})

    elif g in ("S_ADDI", "S_ANDI", "S_ORI", "S_LUI"):
        imm = d.imm16
        if g == "S_ADDI":
            write_s(d.rd, s[d.ra] + sext16(imm))
        elif g == "S_ANDI":
            write_s(d.rd, s[d.ra] & imm)
        elif g == "S_ORI":
            write_s(d.rd, s[d.ra] | imm)
        else:
            write_s(d.rd, imm << 16)

    elif g == "V_ADDI":
        write_v(d.rd, {i: v[d.ra][i] + sext16(d.imm16) for i in active})

    elif g == "V_ANDI":
        write_v(d.rd, {i: v[d.ra][i] & d.imm16 for i in active})

    elif g in ("S_LW", "S_SW"):
        addr = (s[d.ra] + sext16(d.imm16)) & M32
        code = check(addr, gpu.mem_size)
        if code:
            raise Fault(code, 0, addr)
        if g == "S_LW":
            write_s(d.rd, gpu.load_word(addr))
        else:
            gpu.store_word(addr, s[d.rd])
            rec.mwrites.append((GLOBAL, addr, s[d.rd]))

    elif g in ("V_LW", "V_SW", "V_LDS", "V_STS"):
        shared = ins.space == "shared"
        addrs = lane_addresses(SHARED_SIZE if shared else gpu.mem_size, d.ra, sext16(d.imm16))
        if op == "load":
            if shared:
                write_v(d.rd, {i: blk.shared[a >> 2] for i, a in addrs.items()})
            else:
                write_v(d.rd, {i: gpu.load_word(a) for i, a in addrs.items()})
        else:
            data = list(v[d.rd])
            for i in active:                                # ascending: highest lane wins [MEM-4]
                if shared:
                    blk.shared[addrs[i] >> 2] = data[i]
                else:
                    gpu.store_word(addrs[i], data[i])
                rec.mwrites.append((SHARED if shared else GLOBAL, addrs[i], data[i]))

    elif g == "V_ATOM":
        shared = ins.space == "shared"
        b = B()
        addrs = lane_addresses(SHARED_SIZE if shared else gpu.mem_size, d.ra, 0)
        old = {}
        for i in active:                                    # ascending lane order (6.9)
            a = addrs[i]
            old[i] = blk.shared[a >> 2] if shared else gpu.load_word(a)
            new = atom_op(op, old[i], b[i])
            if shared:
                blk.shared[a >> 2] = new
            else:
                gpu.store_word(a, new)
            rec.mwrites.append((SHARED if shared else GLOBAL, a, new))
        write_v(d.rd, old)

    elif g in ("S_BZ", "S_BNZ"):
        taken = (s[d.rd] == 0) == (g == "S_BZ")
        if taken:
            next_pc = (pc + 4 + (sext16(d.imm16) << 2)) & M32

    elif g == "S_J":
        next_pc = (pc + 4 + (d.soffset26 << 2)) & M32

    else:
        raise AssertionError(g)

    warp.pc = next_pc
    rec.exec_after = s[31]
    return rec
