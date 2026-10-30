"""Random program generation and differential/equivalence testing (python -m mgpu.fuzz)."""

from .gen import FuzzCase, random_case, raw_case  # noqa: F401
from .runner import diff_case, equiv_case, run_case  # noqa: F401
