"""Scalar/vector integer ALU, compares, select, cross-lane and special registers (ISA 6.2-6.7)."""

import pytest

from mgpu.isa import encode

M = 0xFFFFFFFF

INT_CASES = [
    ("ADD", 0xFFFFFFFF, 1, 0), ("ADD", 0x7FFFFFFF, 1, 0x80000000),
    ("SUB", 0, 1, 0xFFFFFFFF), ("SUB", 5, 7, 0xFFFFFFFE),
    ("MUL", 0x10000, 0x10000, 0), ("MUL", 0xFFFFFFFF, 0xFFFFFFFF, 1), ("MUL", 0xFFFFFFFE, 3, 0xFFFFFFFA),
    ("AND", 0xF0F0F0F0, 0xFF00FF00, 0xF000F000), ("OR", 0xF0F0F0F0, 0x0F, 0xF0F0F0FF),
    ("XOR", 0xFFFF0000, 0xFF00FF00, 0x00FFFF00), ("ANDN", 0xFF, 0x0F, 0xF0),
    ("SHL", 1, 31, 0x80000000), ("SHL", 1, 33, 2), ("SHL", 3, 0xFFFFFFE0, 3),
    ("SHR", 0x80000000, 31, 1), ("SHR", 0x80000000, 32, 0x80000000),
    ("SRA", 0x80000000, 31, 0xFFFFFFFF), ("SRA", 0x80000000, 63, 0xFFFFFFFF), ("SRA", 0x40000000, 30, 1),
    ("SLT", 0xFFFFFFFF, 0, 1), ("SLT", 0, 0xFFFFFFFF, 0), ("SLT", 3, 3, 0),
    ("SLTU", 0xFFFFFFFF, 0, 0), ("SLTU", 0, 0xFFFFFFFF, 1),
    ("MIN", 0xFFFFFFFF, 1, 0xFFFFFFFF), ("MIN", 0x80000000, 0x7FFFFFFF, 0x80000000),
    ("MAX", 0x80000000, 0x7FFFFFFF, 0x7FFFFFFF), ("MAX", 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF),
]


@pytest.mark.parametrize("op, a, b, want", INT_CASES)
def test_scalar_int(run, op, a, b, want):
    r = run(f"S_LI s1, {a}\nS_LI s2, {b}\nS_{op} s3, s1, s2\nEXIT")
    assert r.res.ok and r.s(3) == want


@pytest.mark.parametrize("op, a, b, want", INT_CASES)
def test_vector_int_broadcast_and_vector(run, op, a, b, want):
    r = run(f"""S_LI s1, {a}
                S_LI s2, {b}
                V_BCAST v1, s1
                V_BCAST v2, s2
                V_{op} v3, v1, s2      ; sb = 1
                V_{op} v4, v1, v2      ; sb = 0
                EXIT""")
    assert r.v(3) == [want] * 32 and r.v(4) == [want] * 32


@pytest.mark.parametrize("op, a, sh, want", [
    ("SHLI", 0x00000003, 31, 0x80000000), ("SHRI", 0x80000000, 4, 0x08000000),
    ("SRAI", 0x80000000, 4, 0xF8000000), ("SHLI", 5, 0, 5)])
def test_shift_immediate(run, op, a, sh, want):
    r = run(f"S_LI s1, {a}\nS_{op} s3, s1, {sh}\nV_BCAST v1, s1\nV_{op} v3, v1, {sh}\nEXIT")
    assert r.s(3) == want and r.v(3) == [want] * 32


@pytest.mark.parametrize("a, popc, ff1", [(0, 0, M), (1, 1, 0), (0x80000000, 1, 31),
                                          (0xFFFFFFFF, 32, 0), (0x00F0F000, 8, 12)])
def test_popc_ff1(run, a, popc, ff1):
    r = run(f"S_LI s1, {a}\nS_POPC s2, s1\nS_FF1 s3, s1\nEXIT")
    assert (r.s(2), r.s(3)) == (popc, ff1)


def test_immediates(run):
    r = run("""S_ADDI s1, s0, -1        ; sext -> 0xFFFFFFFF
               S_ANDI s2, s1, 0x8001    ; zext
               S_ORI  s3, s0, 0xFFFF    ; zext
               S_LUI  s4, 0xABCD
               S_ADDI s5, s4, 0x7FFF
               V_GETSR v1, LANE_ID
               V_ADDI v2, v1, -2
               V_ANDI v3, v2, 0xFFFF
               EXIT""")
    assert [r.s(i) for i in range(1, 6)] == [M, 0x8001, 0xFFFF, 0xABCD0000, 0xABCD7FFF]
    assert r.v(2) == [(i - 2) & M for i in range(32)]
    assert r.v(3) == [(i - 2) & 0xFFFF for i in range(32)]


def test_zero_registers_discard_writes(run):
    r = run("""S_ADDI s0, s0, 5
               V_GETSR v1, LANE_ID
               V_ADDI v0, v1, 7
               S_ADD s1, s0, s0
               V_ADD v2, v0, v0
               V_CMP.EQ s0, v1, v1
               EXIT""")
    assert r.s(0) == 0 and r.s(1) == 0 and r.v(0) == [0] * 32 and r.v(2) == [0] * 32
    assert all(rec.swrite is None and rec.vwrite is None for rec in r.dev.trace[:1])


def test_exec_masks_vector_writes_only(run):
    r = run("""V_GETSR v1, LANE_ID
               S_LI s31, 0x0000FFFF
               V_ADDI v1, v1, 100
               S_ADDI s5, s0, 9         ; scalar ops ignore EXEC
               S_LI s31, 0
               V_ADDI v1, v1, 1000      ; EXEC = 0: nothing happens
               S_ADDI s6, s0, 10
               EXIT""")
    assert r.v(1) == [i + 100 if i < 16 else i for i in range(32)]
    assert (r.s(5), r.s(6)) == (9, 10)


def test_initial_exec_partial_warp(run):
    r = run("S_MOV s1, s31\nEXIT", block=40)
    assert r.s(1, warp=0) == M and r.s(1, warp=1) == 0xFF


def test_initial_state(run):
    r = run("EXIT", args=[11, 22, 33, 44, 55, 66, 77, 88])
    assert [r.s(i) for i in range(1, 9)] == [11, 22, 33, 44, 55, 66, 77, 88]
    assert all(r.s(i) == 0 for i in range(9, 31))
    assert all(r.v(i) == [0] * 32 for i in range(32))


INT_COND = {
    "EQ": lambda a, b: a == b, "NE": lambda a, b: a != b,
    "LT": lambda a, b: s(a) < s(b), "LE": lambda a, b: s(a) <= s(b),
    "GT": lambda a, b: s(a) > s(b), "GE": lambda a, b: s(a) >= s(b),
    "LTU": lambda a, b: a < b, "LEU": lambda a, b: a <= b,
    "GTU": lambda a, b: a > b, "GEU": lambda a, b: a >= b,
}


def s(x):
    return x - (1 << 32) if x >> 31 else x


@pytest.mark.parametrize("cond", INT_COND)
def test_int_compares(run, cond):
    # v1 = lane - 16 (signed spread across zero), compare against broadcast 0 and vector v2.
    r = run(f"""V_GETSR v1, LANE_ID
                V_ADDI v1, v1, -16
                V_SUB v2, v0, v1
                V_CMP.{cond} s1, v1, s0
                V_CMP.{cond} s2, v1, v2
                S_LI s31, 0x00FF00FF
                V_CMP.{cond} s3, v1, v2  ; inactive lanes produce 0
                EXIT""")
    lanes = [(i - 16) & M for i in range(32)]
    neg = [(-x) & M for x in lanes]
    m1 = sum(1 << i for i in range(32) if INT_COND[cond](lanes[i], 0))
    m2 = sum(1 << i for i in range(32) if INT_COND[cond](lanes[i], neg[i]))
    assert (r.s(1), r.s(2), r.s(3)) == (m1, m2, m2 & 0x00FF00FF)


def test_float_compares(run, ):
    from simutil import f2b
    vals = [f2b(1.0), f2b(-1.0), 0, 0x80000000, 0x7FC00000, 0x7F800000, 0xFF800000, 0x00000001]
    other = [f2b(1.0), f2b(1.0), 0x80000000, 0, f2b(1.0), 0x7F800000, 0, 0x00000000]
    src = "\n".join(
        [f"S_LI s1, {a}\nS_LI s2, {b}\nS_LI s31, {1 << i}\nV_BCAST v1, s1\nV_BCAST v2, s2"
         for i, (a, b) in enumerate(zip(vals, other))]
        + ["S_LI s31, 0xFF"] + [f"V_CMP.{c} s{10 + k}, v1, v2" for k, c in
                                enumerate(["FEQ", "FNE", "FLT", "FLE", "FGT", "FGE"])] + ["EXIT"])
    r = run(src)
    # lanes: 0: 1==1, 1: -1<1, 2: 0==-0, 3: -0==0, 4: NaN, 5: inf==inf, 6: -inf<0, 7: denorm==0
    feq, fne, flt, fle, fgt, fge = (r.s(10 + k) for k in range(6))
    bits = lambda *ls: sum(1 << i for i in ls)  # noqa: E731
    assert feq == bits(0, 2, 3, 5, 7)
    assert fne == bits(1, 4, 6)
    assert flt == bits(1, 6)
    assert fle == bits(0, 1, 2, 3, 5, 6, 7)
    assert fgt == 0
    assert fge == bits(0, 2, 3, 5, 7)


def test_cmp_into_exec_uses_old_exec(run):
    r = run("""V_GETSR v1, LANE_ID
               S_LI s31, 0x0000FFFF
               V_CMP.GEU s31, v1, s0      ; computed under EXEC=0xFFFF, written to EXEC
               V_ADDI v2, v1, 1           ; runs under the new EXEC
               EXIT""")
    assert r.s(31) == 0x0000FFFF
    assert r.v(2) == [i + 1 if i < 16 else 0 for i in range(32)]


def test_r4_ops(run):
    from simutil import f2b
    r = run(f"""V_GETSR v1, LANE_ID
                S_LI s2, 3
                V_ADDI v3, v1, 1000
                V_IMAD v4, v1, s2, v3        ; lane*3 + lane + 1000
                S_LI s5, 0xAAAAAAAA
                V_SUB v6, v0, v1
                V_SEL v7, v1, v6, s5         ; odd lanes: lane, even lanes: -lane
                V_SEL v8, v1, s2, s5         ; even lanes: broadcast 3
                S_LI s9, {f2b(2.0)}
                V_I2F v10, v1
                V_FFMA v11, v10, s9, v10     ; 3 * lane
                EXIT""")
    assert r.v(4) == [i * 4 + 1000 for i in range(32)]
    assert r.v(7) == [i if i & 1 else (-i) & M for i in range(32)]
    assert r.v(8) == [i if i & 1 else 3 for i in range(32)]
    from simutil import b2f
    assert [b2f(x) for x in r.v(11)] == [3.0 * i for i in range(32)]


def test_imad_wraps(run):
    r = run("S_LI s1, 0x10000\nV_BCAST v1, s1\nV_BCAST v2, s31\nV_IMAD v3, v1, s1, v2\nEXIT")
    assert r.v(3) == [M] * 32


def test_readlane_and_shfl(run):
    r = run("""V_GETSR v1, LANE_ID
               V_ADDI v1, v1, 100
               S_LI s31, 0x0000000F          ; only lanes 0-3 active
               S_ADDI s2, s0, 37             ; 37 & 31 = 5 (an inactive lane)
               V_READLANE s3, v1, s2
               V_GETSR v2, LANE_ID
               V_ADDI v2, v2, 7
               V_SHFL v4, v1, v2             ; read lanes 7..10 (inactive sources)
               V_SHFL v5, v1, s2             ; broadcast lane 5
               S_LI s31, 0xFFFFFFFF
               V_GETSR v6, LANE_ID
               V_ADDI v6, v6, 1
               V_SHFL v1, v1, v6             ; vd == va: reads before writes (rotate)
               EXIT""")
    assert r.s(3) == 105
    assert r.v(4)[:4] == [107, 108, 109, 110] and r.v(4)[4:] == [0] * 28
    assert r.v(5)[:4] == [105] * 4
    assert r.v(1) == [100 + (i + 1) % 32 for i in range(32)]


def test_getsr_values(run):
    r = run("""S_GETSR s1, BLOCK_ID
               S_GETSR s2, BLOCK_DIM
               S_GETSR s3, GRID_DIM
               S_GETSR s4, WARP_ID
               S_GETSR s5, NUM_WARPS
               V_GETSR v1, TID
               V_GETSR v2, LANE_ID
               V_GETSR v3, GTID
               V_GETSR v4, BLOCK_ID
               EXIT""", grid=3, block=70)
    for b in range(3):
        for w in range(3):
            assert [r.s(i, b, w) for i in range(1, 6)] == [b, 70, 3, w, 3]
            assert r.v(1, b, w) == [32 * w + i for i in range(32)] or w == 2
            assert r.v(2, b, w)[:6] == list(range(6))
            assert r.v(3, b, w)[:6] == [b * 70 + 32 * w + i for i in range(6)]
            assert r.v(4, b, w)[0] == b
    # last warp of each block: lanes >= 6 are inactive and keep 0
    assert r.v(1, 0, 2) == [64 + i for i in range(6)] + [0] * 26


def test_clock_is_monotonic(run):
    r = run("S_GETSR s1, CLOCK\nS_ADDI s5, s0, 1\nS_GETSR s2, CLOCK\nEXIT")
    assert r.s(2) > r.s(1)


@pytest.mark.parametrize("mn, sr", [("S_GETSR", 16), ("S_GETSR", 18), ("S_GETSR", 6),
                                    ("V_GETSR", 15), ("V_GETSR", 19), ("V_GETSR", 31)])
def test_getsr_illegal_ids(run, mn, sr):
    # The assembler refuses these, so encode them directly; illegal even with EXEC = 0 ([DEC-4])
    word = encode(mn, rd=1, ra=sr)
    r = run(f"S_LI s31, 0\n.word {word}\nEXIT")
    assert r.res.status == 1 and r.res.err_pc == 0x1008 and r.res.err_info == r.words[2]
