"""MGPU-32 instruction-set tables, encoder and decoder."""

from .encoding import (Decoded, IllegalInstruction, assemble_one, decode,  # noqa: F401
                       encode, BY_MNEMONIC, SR_BY_ID, SR_BY_NAME)
from .disasm import disasm, listing  # noqa: F401,E402
