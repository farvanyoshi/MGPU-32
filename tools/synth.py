#!/usr/bin/env python3
"""Synthesize the MGPU-32 RTL with Yosys and report FPGA resources.

Yosys's built-in SystemVerilog reader does not cover the language subset the RTL uses
(packed structs in functions, typed function returns, `inside`), so the sources are first
converted to Verilog-2005 with sv2v (https://github.com/zachjs/sv2v). Point $SV2V at the
sv2v binary or put it on PATH. Yosys is taken from PATH (`yosys`) or from the yowasp-yosys
pip package (`pip install yowasp-yosys`).

Usage: synth.py [--top mgpu_gpu|mgpu_core] [--cores 4] [--target ecp5|xc7|both]

The synthesis top `mgpu_gpu` leaves mgpu_top's verification outputs (retire port) unconnected,
so the logic that only serves them (retire records, the register file's old-vd copy) is
removed. Prints a markdown table of LUTs, flip-flops, block RAMs, distributed-RAM cells and
DSP blocks.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl"
OUT = ROOT / "build" / "synth"


def sources() -> list[Path]:
    files = [RTL / "pkg" / n for n in ("mgpu_isa_pkg.sv", "mgpu_fp_pkg.sv", "mgpu_core_pkg.sv")]
    for d in ("common", "units", "core", "top"):
        files += sorted((RTL / d).glob("*.sv"))
    return files


def find_tools():
    sv2v = os.environ.get("SV2V") or shutil.which("sv2v")
    yosys = shutil.which("yosys") or shutil.which("yowasp-yosys")
    missing = []
    if not sv2v:
        missing.append("sv2v (set $SV2V or put it on PATH; https://github.com/zachjs/sv2v/releases)")
    if not yosys:
        missing.append("yosys (install it, or `pip install yowasp-yosys`)")
    if missing:
        sys.exit("synth.py needs: " + "; ".join(missing))
    return sv2v, yosys


# cell name -> report column, per target
COLUMNS = {
    "ecp5": [("LUT4", "LUTs"), ("TRELLIS_FF", "FFs"), ("DP16KD", "BRAM (DP16KD)"),
             ("TRELLIS_DPR16X4", "LUTRAM (DPR16X4)"), ("MULT18X18D", "DSP (MULT18)")],
    "xc7": [("LUT", "LUTs"), ("FD", "FFs"), ("RAMB", "BRAM (RAMB18/36)"),
            ("RAM", "LUTRAM (RAM32/64)"), ("DSP48E1", "DSP48")],
}


def count(stat: str, target: str) -> dict:
    cells = {name: int(n) for n, name in re.findall(r"^\s+(\d+)\s+(\S+)\s*$", stat, re.M)}
    if not cells:  # older Yosys layout: "name   count"
        cells = {name: int(n) for name, n in re.findall(r"^\s+(\S+)\s+(\d+)\s*$", stat, re.M)}
    out = {}
    for prefix, col in COLUMNS[target]:
        if target == "xc7" and prefix == "RAM":
            out[col] = sum(n for c, n in cells.items() if c.startswith("RAM") and not c.startswith("RAMB"))
        else:
            out[col] = sum(n for c, n in cells.items() if c.startswith(prefix))
    return out


def run(top: str, cores: int, target: str, sv2v: str, yosys: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    verilog = OUT / "mgpu.v"
    subprocess.run([sv2v, "-w", str(verilog)] + [str(s) for s in sources()], check=True)
    params = f"-chparam C {cores}" if top == "mgpu_gpu" else ""
    synth = {"ecp5": f"synth_ecp5 -top {top}", "xc7": f"synth_xilinx -family xc7 -top {top}"}[target]
    script = f"read_verilog -sv {verilog}; hierarchy -top {top} {params}; {synth}; tee -o {OUT}/stat_{top}_{target}.txt stat"
    log = OUT / f"yosys_{top}_{cores}_{target}.log"
    with open(log, "w") as f:
        subprocess.run([yosys, "-q", "-p", script], check=True, stdout=f, stderr=subprocess.STDOUT)
    return count((OUT / f"stat_{top}_{target}.txt").read_text(), target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top", default="mgpu_gpu", choices=["mgpu_gpu", "mgpu_core"])
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target", default="both", choices=["ecp5", "xc7", "both"])
    a = ap.parse_args()
    sv2v, yosys = find_tools()
    for target in (["ecp5", "xc7"] if a.target == "both" else [a.target]):
        res = run(a.top, a.cores, target, sv2v, yosys)
        what = f"{a.top}" + (f", C = {a.cores}" if a.top == "mgpu_gpu" else "")
        print(f"\n### {what}, {target} (Yosys {'synth_ecp5' if target == 'ecp5' else 'synth_xilinx'})\n")
        print("| " + " | ".join(res) + " |")
        print("|" + "---|" * len(res))
        print("| " + " | ".join(f"{v:,}" for v in res.values()) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
