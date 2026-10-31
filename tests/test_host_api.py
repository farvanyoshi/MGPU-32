"""Host API (Device.load/upload/run, Buffer) and the mgpu-run CLI."""

import numpy as np
import pytest

from mgpu.asm import assemble
from mgpu.device import HEAP_START, Device, KernelFault, kernel_arg, rust_available
from mgpu.runner import main as run_main

BACKENDS = ["model"] + (["rust"] if rust_available() else [])

SAXPY = """
        .kernel saxpy
saxpy:  V_GETSR   v1, GTID
        V_CMP.LT  s10, v1, s1
        S_MOV     exec, s10
        S_BZ      exec, done
        V_SHLI    v2, v1, 2
        V_ADD     v3, v2, s3
        V_ADD     v4, v2, s4
        V_LW      v5, (v3)
        V_LW      v6, (v4)
        V_FFMA    v6, v5, s2, v6
        V_SW      v6, (v4)
done:   EXIT
"""


@pytest.mark.parametrize("backend", BACKENDS)
def test_run_with_buffers(backend):
    dev = Device(backend, mem_size=1 << 20)
    prog = assemble(SAXPY)
    x = dev.upload(np.arange(70, dtype=np.float32))
    y = dev.upload(np.full(70, 0.5, np.float32))
    res = dev.run(prog, 3, 32, 70, 2.0, x, y)
    assert res.ok
    assert np.array_equal(y.read(), np.arange(70, dtype=np.float32) * 2 + 0.5)
    assert x.addr >= HEAP_START and y.addr >= x.addr + x.nbytes and y.addr % 256 == 0


def test_kernel_fault_raises_with_instruction():
    dev = Device("model", mem_size=1 << 20)
    prog = assemble(SAXPY)
    y = dev.zeros(32)
    with pytest.raises(KernelFault) as e:
        dev.run(prog, 1, 32, 32, 1.0, 6, y)           # misaligned x pointer
    assert e.value.result.status == 2 and "V_LW v5, 0(v3)" in str(e.value)
    assert not dev.run(prog, 1, 32, 32, 1.0, 6, y, check=False).ok


def test_kernel_args():
    dev = Device("model", mem_size=1 << 20)
    buf = dev.zeros((2, 3), "int32")
    assert kernel_arg(buf) == buf.addr and buf.shape == (2, 3) and buf.read().shape == (2, 3)
    assert kernel_arg(1.0) == 0x3F800000 and kernel_arg(np.float32(-2)) == 0xC0000000
    assert kernel_arg(-1) == 0xFFFFFFFF and kernel_arg(np.int32(7)) == 7 and kernel_arg(True) == 1
    with pytest.raises(ValueError):
        kernel_arg(1 << 32)
    with pytest.raises(TypeError):
        kernel_arg("x")


def test_heap_and_program_placement():
    dev = Device("model", mem_size=0x20000)
    big = assemble(".org 0x10100\nEXIT")          # extends into the default heap area
    dev.load(big)
    assert dev.alloc(16) >= big.end
    with pytest.raises(MemoryError):
        dev.alloc(0x20000)
    dev2 = Device("model", mem_size=0x20000)
    dev2.zeros(4)
    with pytest.raises(ValueError, match="overlaps allocated"):
        dev2.load(assemble(".org 0x10000\nEXIT"))


def test_named_kernels():
    dev = Device("model", mem_size=1 << 20)
    prog = assemble(""".kernel one
                       .kernel two
                one:   S_ADDI s5, s0, 1
                       S_SW s5, (s1)
                       EXIT
                two:   S_ADDI s5, s0, 2
                       S_SW s5, (s1)
                       EXIT""")
    out = dev.zeros(1, "uint32")
    dev.run(prog, 1, 32, out, kernel="two")
    assert out.read()[0] == 2
    with pytest.raises(KeyError):
        dev.run(prog, 1, 32, out)


def test_runner_cli(tmp_path, capsys):
    src = tmp_path / "saxpy.s"
    src.write_text(SAXPY)
    np.save(tmp_path / "x.npy", np.arange(100, dtype=np.float32))
    np.save(tmp_path / "y.npy", np.ones(100, np.float32))
    rc = run_main([str(src), "--grid", "4", "--block", "32", "--arg", "100", "--arg", "f32:3.0",
                   "--arg", f"in:{tmp_path / 'x.npy'}", "--arg", f"inout:{tmp_path / 'y.npy'}",
                   "--stats", "--trace", "3", "--dump", "saxpy:2", "--backend", "model"])
    out = capsys.readouterr().out
    assert rc == 0 and "ok (48 instructions)" in out and "SIMD utilization" in out
    assert "V_GETSR v1, GTID" in out and "... 45 more" in out and "0x1000: [" in out
    assert np.array_equal(np.load(tmp_path / "y.npy"), np.arange(100, dtype=np.float32) * 3 + 1)


def test_runner_cli_out_and_fault(tmp_path, capsys):
    src = tmp_path / "k.s"
    src.write_text(".kernel k\nk: V_GETSR v1, GTID\nV_SHLI v2, v1, 2\nV_ADD v2, v2, s1\n"
                   "V_SW v1, (v2)\nEXIT\n")
    assert run_main([str(src), "--grid", "2", "--block", "32",
                     "--arg", f"out:{tmp_path / 'o.npy'}:64:uint32"]) == 0
    assert np.array_equal(np.load(tmp_path / "o.npy"), np.arange(64, dtype=np.uint32))
    assert run_main([str(src), "--arg", "2"]) == 2
    assert "misaligned access" in capsys.readouterr().out
    assert run_main([str(src), "--arg", "bogus:1"]) == 1
    assert run_main([str(src), "--dump", "nope"]) == 1


def test_runner_binary_image(tmp_path, capsys):
    prog = assemble(".kernel k\nk: S_ADDI s5, s0, 9\nS_SW s5, (s1)\nEXIT", base=0x2000)
    (tmp_path / "k.bin").write_bytes(prog.image)
    assert run_main([str(tmp_path / "k.bin"), "--base", "0x2000", "--arg", "0x3000",
                     "--dump", "0x3000:1", "--backend", "model"]) == 0
    assert "0x3000: [9]" in capsys.readouterr().out
