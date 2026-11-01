"""Output formats of the assembler: flat binary, $readmemh hex, symbol map, listing."""

import json


def hex_text(program) -> str:
    """`$readmemh` image: an @<word address> line, then one 32-bit word per line."""
    lines = [f"@{program.base // 4:08x}"] + [f"{w:08x}" for w in program.words]
    return "\n".join(lines) + "\n"


def symbol_map(program) -> dict:
    return {
        "base": program.base,
        "size": len(program.image),
        "kernels": dict(sorted(program.kernels.items(), key=lambda kv: kv[1])),
        "labels": {k: program.symbols[k] for k in sorted(program.labels, key=program.symbols.get)},
        "constants": {k: v for k, v in sorted(program.symbols.items()) if k not in program.labels},
    }


def symbol_json(program) -> str:
    return json.dumps(symbol_map(program), indent=2) + "\n"


def listing_text(program) -> str:
    """Address, encoded word(s) and source line for every statement."""
    out = []
    for e in program.listing:
        src = e.text.rstrip()
        if not e.words:
            out.append(f"{'':8}  {'':8}  {src}" if src.strip() else "")
            continue
        for i, w in enumerate(e.words):
            out.append(f"{e.addr + 4 * i:08x}  {w:08x}  {src if i == 0 else ''}".rstrip())
    return "\n".join(out) + "\n"
