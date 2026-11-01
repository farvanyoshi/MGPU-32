"""The MGPU-32 assembler. See docs/toolchain/assembler.md."""

from .assembler import AsmError, Diagnostic, Program, assemble, assemble_file  # noqa: F401
from .output import hex_text, listing_text, symbol_map  # noqa: F401
