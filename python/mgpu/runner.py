"""mgpu-run: assemble (or load) a kernel, bind arguments, run it, and save results.

    mgpu-run kernels/saxpy.s --grid 8 --block 128 \\
        --arg 1000 --arg f32:2.0 --arg in:x.npy --arg inout:y.npy --stats

Argument forms (one --arg per kernel argument, in order, at most 8):
    123, 0x10, -1                  integer
    f32:2.5                        float, passed as its binary32 bits
    in:FILE.npy                    upload the array; pass its address
    inout:FILE.npy                 upload; after the run, save the array back to FILE.npy
    out:FILE.npy:COUNT[:DTYPE]     zero-filled buffer (default float32); saved after the run
    zeros:COUNT[:DTYPE]            zero-filled scratch buffer
    sym:NAME                       address of a symbol in the program
"""

import argparse
import sys
from pathlib import Path

from .asm import AsmError, assemble_file
from .asm.dis_cli import read_image
from .device import Device, kernel_arg
from .errors import SimTimeout
from .isa import disasm


class ArgError(Exception):
    pass


def parse_arg(spec: str, dev: Device, program, outputs: list):
    import numpy as np
    kind, _, rest = spec.partition(":")
    if not rest:
        try:
            return int(spec, 0)
        except ValueError:
            raise ArgError(f"bad argument '{spec}' (see --help for the forms)") from None
    if kind == "f32":
        return kernel_arg(float(rest))
    if kind in ("in", "inout"):
        buf = dev.upload(np.load(rest))
        if kind == "inout":
            outputs.append((rest, buf))
        return buf
    if kind in ("out", "zeros"):
        parts = rest.split(":")
        if kind == "out":
            path, parts = parts[0], parts[1:]
        if not parts:
            raise ArgError(f"'{spec}': missing element count")
        buf = dev.zeros(int(parts[0], 0), parts[1] if len(parts) > 1 else "float32")
        if kind == "out":
            outputs.append((path, buf))
        return buf
    if kind == "sym":
        if program is None or rest not in program.symbols:
            raise ArgError(f"unknown symbol '{rest}'")
        return program.symbols[rest]
    raise ArgError(f"unknown argument kind '{kind}:'")


def parse_dump(spec: str, program) -> tuple[int, int, str]:
    """ADDR:COUNT[:DTYPE]; ADDR may be a number or a program symbol."""
    import numpy as np
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ArgError(f"--dump '{spec}': expected ADDR:COUNT[:DTYPE]")
    try:
        addr = program.symbols[parts[0]] if program and parts[0] in program.symbols else int(parts[0], 0)
        count = int(parts[1], 0)
        dtype = np.dtype(parts[2] if len(parts) == 3 else "uint32").str
    except (ValueError, TypeError):
        raise ArgError(f"--dump '{spec}': bad address, count or dtype") from None
    return addr, count, dtype


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mgpu-run", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[5:]))
    ap.add_argument("program", help=".s source, or a .bin/.hex image")
    ap.add_argument("--kernel", help="kernel name (default: the only .kernel)")
    ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--arg", action="append", default=[], help="kernel argument (repeatable)")
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0x1000,
                    help="load address (assembly base, or where a .bin is loaded)")
    ap.add_argument("--entry", type=lambda s: int(s, 0), help="entry address for .bin/.hex images")
    ap.add_argument("--backend", choices=["rust", "model"], default="rust")
    ap.add_argument("--mem-size", type=lambda s: int(s, 0), default=64 << 20)
    ap.add_argument("--max-steps", type=int, default=50_000_000)
    ap.add_argument("--trace", type=int, nargs="?", const=200, metavar="N",
                    help="print the first N committed instructions (default 200)")
    ap.add_argument("--stats", action="store_true", help="print instruction mix and SIMD utilization")
    t = ap.add_argument_group("timing model (docs/uarch)")
    t.add_argument("--timing", action="store_true", help="run under the cycle-level timing model")
    t.add_argument("--cores", type=int, default=1)
    t.add_argument("--warps", type=int, default=8, help="warp slots per core")
    t.add_argument("--lanes", type=int, default=8, help="physical vector lanes")
    t.add_argument("--mem-latency", type=int, default=100)
    t.add_argument("--seg-cycles", type=int, default=8, help="cycles per 128-byte segment")
    ap.add_argument("--dump", action="append", default=[], metavar="ADDR:COUNT[:DTYPE]",
                    help="print COUNT elements of memory after the run (ADDR may be a symbol)")
    args = ap.parse_args(argv)

    trace = args.trace is not None or args.stats
    timing = None
    if args.timing:
        if args.backend != "rust":
            print("mgpu-run: --timing needs the rust backend", file=sys.stderr)
            return 1
        timing = dict(cores=args.cores, warps=args.warps, lanes=args.lanes,
                      mem_latency=args.mem_latency, seg_cycles=args.seg_cycles)
    dev = Device(args.backend, timing=timing, mem_size=args.mem_size, max_steps=args.max_steps,
                 trace=trace)
    program = None
    try:
        path = Path(args.program)
        if path.suffix == ".s":
            program = assemble_file(path, args.base)
            dev.load(program)
            entry = program.entry(args.kernel)
        else:
            base, words = read_image(path, args.base)
            dev.write_words(base, words)
            dev.heap_ptr = max(dev.heap_ptr, base + 4 * len(words))
            entry = args.entry if args.entry is not None else base
        outputs: list = []
        kargs = [parse_arg(a, dev, program, outputs) for a in args.arg]
        dumps = [parse_dump(d, program) for d in args.dump]
        if len(kargs) > 8:
            raise ArgError("at most 8 kernel arguments")
    except AsmError as e:
        print(e, file=sys.stderr)
        return 1
    except (ArgError, KeyError, OSError, ValueError) as e:
        print(f"mgpu-run: {e}", file=sys.stderr)
        return 1

    name = args.kernel or next((k for k, v in (program.kernels if program else {}).items() if v == entry), None)
    label = f"{name} @ {entry:#x}" if name else f"{entry:#x}"
    try:
        res = dev.launch(entry, args.grid, args.block, [kernel_arg(a) for a in kargs])
    except (SimTimeout, ValueError) as e:
        print(f"mgpu-run: {e}", file=sys.stderr)
        return 3
    if args.trace is not None:
        for r in dev.trace[:args.trace]:
            text = disasm(r.word, r.pc)
            print(f"{r.step:7d}  b{r.block:<3} w{r.warp:<2} {r.pc:08x}  {text:<36} exec={r.exec_after:08x}")
        if len(dev.trace) > args.trace:
            print(f"... {len(dev.trace) - args.trace} more")
    print(f"{label}  grid={args.grid} block={args.block} [{args.backend}]: {res}")
    if not res.ok:
        detail = dev.describe_fault(res)
        if detail:
            print(f"  {detail}")
    if args.stats:
        from .stats import trace_stats
        print(trace_stats(dev.trace, args.block).report())
    if dev.timing_report is not None:
        from .stats import timing_text
        print(timing_text(dev.timing_report))
    import numpy as np
    for path, buf in outputs:
        np.save(path, buf.read())
        print(f"saved {path} ({buf.count} x {np.dtype(buf.dtype).name})")
    for addr, count, dtype in dumps:
        arr = dev.read_array(addr, count, dtype)
        print(f"{addr:#x}: {np.array2string(arr, threshold=64)}")
    return 0 if res.ok else 2


if __name__ == "__main__":
    sys.exit(main())
