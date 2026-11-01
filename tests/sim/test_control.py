"""Branches, EXIT, BAR, illegal instructions, launch validation and host registers."""

import pytest

from mgpu import hostregs as H
from mgpu.device import Device
from mgpu.isa import encode

from simutil import CODE, DATA, MEM


def test_branches_and_loops(run):
    r = run("""S_ADDI s1, s0, 10
               S_ADDI s2, s0, 0
       loop:   S_ADD  s2, s2, s1
               S_ADDI s1, s1, -1
               S_BNZ  s1, loop
               S_BZ   s1, skip
               S_ADDI s2, s2, 1000      ; skipped
       skip:   S_J    end
               S_ADDI s2, s2, 2000      ; skipped
       end:    EXIT""")
    assert r.s(2) == 55


def test_branch_on_exec_with_exec_zero(run):
    r = run("""S_MOV s10, s31
               S_LI s31, 0
               S_BZ s31, out           ; scalar branch still runs with EXEC = 0
               S_ADDI s1, s0, 1
       out:    S_MOV s31, s10
               EXIT""")
    assert r.s(1) == 0 and r.s(31) == 0xFFFFFFFF


def test_divergent_if_else_from_spec(run):
    r = run("""V_GETSR v1, LANE_ID
               V_ADDI v2, v0, 16
               V_CMP.LT  s10, v1, v2
               S_MOV     s11, s31
               S_MOV     s31, s10
               S_BZ      s31, else
               V_ADDI    v3, v1, 1000
       else:   S_ANDN    s31, s11, s10
               S_BZ      s31, endif
               V_ADDI    v3, v1, 2000
       endif:  S_MOV     s31, s11
               EXIT""")
    assert r.v(3) == [i + (1000 if i < 16 else 2000) for i in range(32)]


def test_branch_target_wraps(run):
    # S_J backwards from CODE by more than CODE bytes wraps around to the top of memory
    off = -(CODE + 4) // 4 - 1          # target = 0xFFFFFFFC -> out of range fetch
    r = run(f"S_J {off}\nEXIT")
    assert r.res.status == 3 and r.res.err_pc == 0xFFFFFFFC and r.res.err_info == 0xFFFFFFFC


def test_fetch_past_end_of_memory(backend):
    d = Device(backend, mem_size=0x2000)
    d.write_words(0x1FFC, [encode("NOP")])
    res = d.launch(0x1FFC, 1, 32)
    assert (res.status, res.err_pc, res.err_info) == (3, 0x2000, 0x2000)


@pytest.mark.parametrize("word", [0x0E << 26, 0x3F << 26, 0x02 << 26 | 0x11, 0x00000004,
                                  0x16 << 26 | 0x06])
def test_illegal_instruction(run, word):
    r = run(f"S_ADDI s1, s0, 1\n.word {word}\nEXIT")
    assert (r.res.status, r.res.err_pc, r.res.err_info) == (1, CODE + 4, word)
    assert r.s(1) == 1


def test_multi_warp_multi_block_exit(run):
    r = run(f"""V_GETSR v1, GTID
               V_SHLI v2, v1, 2
               S_LI s1, {DATA}
               V_ADD v2, v2, s1
               V_SW v1, 0(v2)
               EXIT""", grid=5, block=96, keep_state=False)
    assert r.res.ok and r.mem(DATA, 480) == list(range(480))


def test_barrier_orders_shared_memory(run):
    # each warp writes its slice, BAR, then reads the next warp's slice
    r = run("""V_GETSR v1, TID
               V_SHLI v2, v1, 2
               V_STS v1, 0(v2)
               BAR
               V_ADDI v3, v2, 128
               V_ANDI v3, v3, 511          ; 4 warps * 32 lanes * 4 bytes = 512
               V_LDS v4, 0(v3)
               EXIT""", block=128)
    for w in range(4):
        assert r.v(4, warp=w) == [(32 * ((w + 1) % 4)) + i for i in range(32)]
    assert sum(rec.released for rec in r.dev.trace) == 1


def test_barrier_exit_excuses_warp(run):
    # warp 1 exits early; warps 0, 2 still pass the barrier
    r = run("""S_GETSR s1, WARP_ID
               S_ADDI s2, s1, -1
               S_BZ s2, leave
               BAR
               S_ADDI s3, s0, 7
       leave:  EXIT""", block=96)
    assert r.res.ok and r.s(3, warp=0) == 7 and r.s(3, warp=2) == 7


def test_barrier_mismatch(run):
    r = run("""S_GETSR s1, WARP_ID
               S_BZ s1, first
               BAR                        ; warp 1 waits here
               EXIT
       first:  BAR                        ; warp 0 arrives at a different BAR
               EXIT""", block=64)
    assert r.res.status == 4
    pcs = {CODE + 8, CODE + 16}
    assert r.res.err_pc in pcs and r.res.err_info in pcs and r.res.err_pc != r.res.err_info


def test_single_warp_bar_passes(run):
    r = run("BAR\nBAR\nS_ADDI s1, s0, 1\nEXIT")
    assert r.res.ok and r.s(1) == 1


def test_host_registers(backend):
    d = Device(backend, mem_size=MEM, max_block_dim=256)
    assert d.reg_read(H.ID) == 0x4D475055
    assert d.reg_read(H.VERSION) == 0x00010000
    assert d.reg_read(H.MAX_BLOCK_DIM) == 256
    assert d.reg_read(H.MEM_SIZE) == MEM
    assert d.reg_read(0x3C) == 0 and d.reg_read(0x100) == 0
    d.reg_write(H.ID, 5)
    assert d.reg_read(H.ID) == 0x4D475055
    d.reg_write(H.ARG(7), 0x1234)
    assert d.reg_read(H.ARG(7)) == 0x1234


@pytest.mark.parametrize("entry, grid, block, status, info", [
    (CODE, 1, 0, 5, 0),
    (CODE, 1, 257, 5, 0),
    (CODE + 2, 1, 32, 2, CODE + 2),
    (CODE, 0, 32, 0, 0),
    (CODE + 2, 0, 0, 5, 0),          # validation order: BLOCK_DIM first
])
def test_launch_validation(backend, entry, grid, block, status, info):
    d = Device(backend, mem_size=MEM, max_block_dim=256)
    d.write_words(CODE, [encode("EXIT")])
    res = d.launch(entry, grid, block)
    assert (res.status, res.err_info) == (status, info) and d.reg_read(H.GO) == 0
    assert res.steps == 0


def test_status_cleared_on_next_launch(backend):
    d = Device(backend, mem_size=MEM)
    d.write_words(CODE, [0xFFFFFFFF, encode("EXIT")])
    assert d.launch(CODE, 1, 32).status == 1
    assert d.launch(CODE + 4, 1, 32).ok and d.reg_read(H.ERR_PC) == 0


def test_mem_size_full_4gib_reads_zero():
    from mgpu.model import Gpu
    g = Gpu.__new__(Gpu)          # avoid allocating 4 GiB: check the register encoding only
    g.mem_size, g.max_block_dim, g.running, g.steps = 1 << 32, 1024, False, 0
    g.regs, g.status = {}, [0] * 6
    assert g.reg_read(H.MEM_SIZE) == 0
