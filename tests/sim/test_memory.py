"""Loads, stores, shared memory, atomics and memory faults (ISA 6.8, 6.9, 10)."""

import pytest

from simutil import DATA, MEM

M = 0xFFFFFFFF


def lane_addr(base_reg="s1", stride=4, dst="v2"):
    return f"V_GETSR v1, LANE_ID\nV_SHLI v1, v1, {stride.bit_length() - 1}\nV_ADD {dst}, v1, {base_reg}"


def test_scalar_load_store(run):
    r = run(f"""S_LI s1, {DATA}
               S_LW s2, 4(s1)
               S_ADDI s2, s2, 1
               S_SW s2, -4(s1)
               S_LI s31, 0                 ; scalar memory ignores EXEC
               S_LW s3, 4(s1)
               EXIT""", data={DATA - 4: [0], DATA + 4: [41]})
    assert r.s(3) == 41 and r.mem(DATA - 4, 1) == [42]


def test_vector_gather_scatter_under_exec(run):
    r = run(f"""S_LI s1, {DATA}
               {lane_addr()}
               V_LW v3, 0(v2)
               V_ADDI v3, v3, 1
               S_LI s31, 0x0000FFFF
               V_SW v3, 128(v2)
               EXIT""", data={DATA: list(range(100, 132))})
    assert r.v(3) == list(range(101, 133))
    assert r.mem(DATA + 128, 32) == list(range(101, 117)) + [0] * 16


def test_inactive_lanes_never_fault(run):
    r = run(f"""V_GETSR v1, LANE_ID
               V_SHLI v2, v1, 1               ; lane*2: odd multiples of 2 are misaligned
               S_LI s1, {DATA}
               V_ADD v2, v2, s1
               S_LI s31, 0x55555555           ; even lanes only -> lane*2 % 4 == 0
               V_LW v3, 0(v2)
               EXIT""")
    assert r.res.ok


def test_shared_memory_and_poison(run):
    r = run(f"""V_GETSR v1, LANE_ID
               V_SHLI v2, v1, 2
               V_LDS v3, 0(v2)                ; uninitialized: poison
               V_STS v1, 16380(v0)            ; collision on the last word: lane 31 wins
               V_LDS v4, 16380(v0)
               V_STS v1, 0(v2)
               V_LDS v5, 4(v2)
               EXIT""", shared_poison=0x12345678)
    assert r.v(3) == [0x12345678] * 32
    assert r.v(4) == [31] * 32
    assert r.v(5) == list(range(1, 32)) + [0x12345678]


def test_store_collision_highest_lane_wins(run):
    r = run(f"""S_LI s1, {DATA}
               V_GETSR v1, LANE_ID
               V_ADDI v1, v1, 7
               S_LI s31, 0x00F0F0F0
               V_SW v1, 0(s1)               ; base via scalar? no: va must be a vector
               EXIT""".replace("V_SW v1, 0(s1)", "V_BCAST v2, s1\nV_SW v1, 0(v2)"))
    assert r.mem(DATA, 1) == [23 + 7]


@pytest.mark.parametrize("op, init, want_mem, want_old", [
    ("ADD", 5, 5 + sum(range(32)), [5 + sum(range(i)) for i in range(32)]),
    ("MAX", 10, 31, [10] * 11 + list(range(10, 31))),
    ("MIN", 10, 0xFFFFFFF0, [10] + [0xFFFFFFF0] * 31),
    ("XCHG", 99, 31, [99] + list(range(31))),
    ("OR", 0x100, 0x11F, None),
    ("AND", 0xFFFFFFFF, 0, None),
])
def test_atomics_same_address(run, op, init, want_mem, want_old):
    # every lane hits one word; lanes apply in ascending order (ISA 6.9)
    operand = "V_GETSR v3, LANE_ID" if op != "MIN" else "V_GETSR v3, LANE_ID\nV_ADDI v3, v3, -16\nV_MIN v3, v3, v0\nV_ADDI v4, v0, -16\nS_LI s31, 0xFFFFFFFE\nV_MOV v3, v4\nS_LI s31, 0xFFFFFFFF"
    r = run(f"""S_LI s1, {DATA}
               V_BCAST v2, s1
               {operand}
               V_ATOM.{op} v5, v2, v3
               EXIT""", data={DATA: [init]})
    assert r.mem(DATA, 1) == [want_mem]
    if want_old:
        assert r.v(5) == want_old


def test_atomic_shared_and_scalar_operand(run):
    r = run("""S_ADDI s2, s0, 3
               V_ADD v2, v0, v0          ; address 0 for every lane
               V_STS v0, 0(v0)
               V_ATOM.ADD.S v5, v2, s2   ; broadcast operand
               V_LDS v6, 0(v0)
               V_GETSR v7, LANE_ID
               V_SHLI v7, v7, 2
               V_ATOM.XCHG.S v7, v7, v7  ; vd == va: address read before write
               EXIT""")
    assert r.v(5) == [3 * i for i in range(32)]
    assert r.v(6) == [96] * 32
    assert r.v(7)[0] == 96 and r.v(7)[1] == 0xDEADBEEF


@pytest.mark.parametrize("src, code, lane, info", [
    ("S_LI s1, 0x10002\nS_LW s2, 0(s1)", 2, 0, 0x10002),
    (f"S_LI s1, {MEM}\nS_LW s2, 0(s1)", 3, 0, MEM),
    (f"S_LI s1, {MEM - 4}\nS_SW s2, 0(s1)", 0, 0, 0),
    (f"S_LI s1, {MEM - 4}\nS_SW s2, 4(s1)", 3, 0, MEM),
    ("S_ADDI s1, s0, -4\nS_LW s2, 0(s1)", 3, 0, 0xFFFFFFFC),
    # lane 3 out of range and lane 5 misaligned: lowest faulting lane wins
    (f"V_GETSR v1, LANE_ID\nV_SHLI v2, v1, 2\nS_LI s3, {MEM - 12}\nV_ADD v2, v2, s3\n"
     "S_LI s31, 0x28\nV_LW v4, 0(v2)", 3, 3, MEM),
    (f"V_GETSR v1, LANE_ID\nV_SHLI v2, v1, 2\nS_LI s3, {MEM - 12}\nV_ADD v2, v2, s3\n"
     "V_ADDI v2, v2, 1\nS_LI s31, 0x28\nV_SW v4, 0(v2)", 2, 3, MEM + 1),
    ("V_GETSR v1, LANE_ID\nV_SHLI v2, v1, 9\nV_STS v1, 0(v2)", 3, 32, 0),  # placeholder
    ("V_ADDI v1, v0, 16384\nV_LDS v2, 0(v1)", 3, 0, 16384),
    ("V_ADDI v1, v0, 2\nV_ATOM.ADD.S v2, v1, v1", 2, 0, 2),
])
def test_memory_faults(run, src, code, lane, info):
    if lane == 32:  # lane*512 >= 16384 first at lane 32 -> no fault; use lane*1024
        src = src.replace("V_SHLI v2, v1, 9", "V_SHLI v2, v1, 10")
        lane, info = 16, 16384
    r = run(src + "\nEXIT")
    assert r.res.status == code
    if code:
        assert (r.res.err_lane, r.res.err_info) == (lane, info)
        assert r.res.err_pc == 0x1000 + 4 * (len(r.words) - 2)  # the instruction before EXIT


def test_faulting_store_has_no_effect(run):
    r = run(f"""V_GETSR v1, LANE_ID
               V_SHLI v2, v1, 2
               S_LI s3, {MEM - 64}
               V_ADD v2, v2, s3           ; lanes >= 16 are out of range
               V_ADDI v5, v0, 77
               V_SW v5, 0(v2)
               EXIT""")
    assert r.res.status == 3 and r.res.err_lane == 16
    assert r.mem(MEM - 64, 16) == [0] * 16
    assert r.dev.trace[-1].mwrites == [] and r.dev.trace[-1].fault == (3, 16, MEM)
