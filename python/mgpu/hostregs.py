"""Host register map and error codes (docs/isa/MGPU-32-ISA.md, sections 11 and 13)."""

ID = 0x00
VERSION = 0x04
MAX_BLOCK_DIM = 0x08
MEM_SIZE = 0x0C
ENTRY = 0x10
GRID_DIM = 0x14
BLOCK_DIM = 0x18
GO = 0x1C
STATUS = 0x20
ERR_PC = 0x24
ERR_BLOCK = 0x28
ERR_WARP = 0x2C
ERR_LANE = 0x30
ERR_INFO = 0x34
CYCLES = 0x38
ARG0 = 0x40          # ARG0..ARG7 at 0x40..0x5C


def ARG(n: int) -> int:
    return ARG0 + 4 * n


ID_VALUE = 0x4D475055        # "MGPU"
VERSION_VALUE = 0x00010000   # 1.0

NAMES = {ID: "ID", VERSION: "VERSION", MAX_BLOCK_DIM: "MAX_BLOCK_DIM", MEM_SIZE: "MEM_SIZE",
         ENTRY: "ENTRY", GRID_DIM: "GRID_DIM", BLOCK_DIM: "BLOCK_DIM", GO: "GO",
         STATUS: "STATUS", ERR_PC: "ERR_PC", ERR_BLOCK: "ERR_BLOCK", ERR_WARP: "ERR_WARP",
         ERR_LANE: "ERR_LANE", ERR_INFO: "ERR_INFO", CYCLES: "CYCLES",
         **{ARG(n): f"ARG{n}" for n in range(8)}}

# Error codes
OK = 0
E_ILLEGAL = 1
E_MISALIGNED = 2
E_RANGE = 3
E_BARRIER = 4
E_LAUNCH = 5

ERROR_NAMES = {OK: "ok", E_ILLEGAL: "illegal instruction", E_MISALIGNED: "misaligned access",
               E_RANGE: "out-of-range access", E_BARRIER: "barrier mismatch",
               E_LAUNCH: "invalid launch"}
