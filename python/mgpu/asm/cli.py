"""mgpu-as: assemble MGPU-32 source files.

    mgpu-as kernel.s                      # writes kernel.bin
    mgpu-as kernel.s -o k.bin --hex k.hex --sym k.sym.json --lst k.lst --base 0x2000
"""

import argparse
import sys
from pathlib import Path

from .assembler import AsmError, assemble_file
from .output import hex_text, listing_text, symbol_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mgpu-as", description="MGPU-32 assembler")
    ap.add_argument("source", help="assembly source file")
    ap.add_argument("-o", "--output", help="flat binary output (default: <source>.bin)")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0x1000,
                    help="load address of the first byte (default 0x1000)")
    ap.add_argument("--hex", help="also write a $readmemh hex image")
    ap.add_argument("--sym", help="also write a JSON symbol map")
    ap.add_argument("--lst", help="also write a listing")
    args = ap.parse_args(argv)
    try:
        prog = assemble_file(args.source, args.base)
    except AsmError as e:
        print(e, file=sys.stderr)
        n = len(e.diagnostics)
        print(f"{n} error{'s' * (n != 1)}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"mgpu-as: {e}", file=sys.stderr)
        return 1
    out = Path(args.output) if args.output else Path(args.source).with_suffix(".bin")
    out.write_bytes(prog.image)
    if args.hex:
        Path(args.hex).write_text(hex_text(prog))
    if args.sym:
        Path(args.sym).write_text(symbol_json(prog))
    if args.lst:
        Path(args.lst).write_text(listing_text(prog))
    kernels = ", ".join(f"{k}@{v:#x}" for k, v in prog.kernels.items()) or "none"
    print(f"{out}: {len(prog.image)} bytes at {prog.base:#x}; kernels: {kernels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
