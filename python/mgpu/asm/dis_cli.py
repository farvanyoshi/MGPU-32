"""mgpu-dis: disassemble an MGPU-32 image back into assembly source.

    mgpu-dis kernel.bin                      # base 0x1000
    mgpu-dis kernel.hex --sym kernel.sym.json
    mgpu-dis kernel.bin --base 0x2000 --listing

The default output reassembles (mgpu-as, same base) to the identical image.
"""

import argparse
import json
import sys
from pathlib import Path

from ..isa.disasm import disassemble, listing


def read_hex(text: str) -> tuple[int, list[int]]:
    """Parse a $readmemh image of 32-bit words. Returns (byte base, words)."""
    mem: dict[int, int] = {}
    addr = 0
    for line in text.splitlines():
        for tok in line.split("//")[0].split():
            if tok.startswith("@"):
                addr = int(tok[1:], 16)
            else:
                mem[addr] = int(tok, 16) & 0xFFFFFFFF
                addr += 1
    if not mem:
        return 0, []
    lo, hi = min(mem), max(mem)
    return 4 * lo, [mem.get(a, 0) for a in range(lo, hi + 1)]


def read_image(path: Path, base: int) -> tuple[int, list[int]]:
    if path.suffix.lower() in (".hex", ".mem"):
        return read_hex(path.read_text())
    data = path.read_bytes()
    data += b"\0" * (-len(data) % 4)
    return base, [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mgpu-dis", description="MGPU-32 disassembler")
    ap.add_argument("image", help=".bin (flat binary) or .hex ($readmemh) image")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0x1000,
                    help="load address of a .bin image (default 0x1000)")
    ap.add_argument("--sym", help="symbol map from mgpu-as --sym (labels branch targets)")
    ap.add_argument("--listing", action="store_true",
                    help="address/word/instruction listing instead of reassemblable source")
    ap.add_argument("-o", "--output", help="write to a file instead of stdout")
    args = ap.parse_args(argv)
    try:
        base, words = read_image(Path(args.image), args.base)
        symbols = {}
        if args.sym:
            sm = json.loads(Path(args.sym).read_text())
            symbols = {**sm.get("labels", {}), **sm.get("kernels", {})}
    except (OSError, ValueError) as e:
        print(f"mgpu-dis: {e}", file=sys.stderr)
        return 1
    if args.listing:
        text = listing(words, base) + "\n"
    else:
        kernels = "".join(f"        .kernel {k}\n" for k in sorted(sm.get("kernels", {}))) if args.sym else ""
        text = f"; disassembled from {args.image}, base {base:#x}\n{kernels}" + disassemble(words, base, symbols)
    if args.output:
        Path(args.output).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
