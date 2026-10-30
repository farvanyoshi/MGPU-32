"""SFU algorithm: normative tables reproduce, special cases, accuracy."""

import struct
import subprocess
import sys

import pytest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from mgpu.fp.sfu import fex2, flg2, frcp, frsq

NAN = 0x7FC00000


def b(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def test_tables_reproduce():
    r = subprocess.run([sys.executable, ROOT / "tools/gen_sfu_tables.py", "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("x, rcp, rsq, ex2, lg2", [
    (0x7FA00001, NAN, NAN, NAN, NAN),
    (0x7F800000, 0, 0, 0x7F800000, 0x7F800000),
    (0xFF800000, 0x80000000, NAN, 0, NAN),
    (0x00000000, 0x7F800000, 0x7F800000, b(1.0), 0xFF800000),
    (0x80000000, 0xFF800000, 0xFF800000, b(1.0), 0xFF800000),
    (0x00000001, 0x7F800000, 0x7F800000, b(1.0), 0xFF800000),   # denormal flushes
    (0x80000001, 0xFF800000, 0xFF800000, b(1.0), 0xFF800000),
    (b(1.0), b(1.0), b(1.0), b(2.0), 0),
    (b(4.0), b(0.25), b(0.5), b(16.0), b(2.0)),
    (b(-2.0), b(-0.5), NAN, b(0.25), NAN),
    (b(2.0 ** 127), 0, 0x1FB504F3, 0x7F800000, b(127.0)),
    (b(0.5), b(2.0), 0x3FB504F3, 0x3FB504F3, b(-1.0)),
    (b(-150.0), 0xBBDA740E, NAN, 0, NAN),
    (b(-2.0 ** -40), b(-2.0 ** 40), NAN, b(1.0), NAN),
])
def test_special_cases(x, rcp, rsq, ex2, lg2):
    assert (frcp(x), frsq(x), fex2(x), flg2(x)) == (rcp, rsq, ex2, lg2)


def test_accuracy_quick():
    pytest.importorskip("numpy")
    r = subprocess.run([sys.executable, ROOT / "tools/sfu_accuracy.py", "--quick"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
