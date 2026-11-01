"""Randomized differential and equivalence testing (see python -m mgpu.fuzz for soak runs).

MGPU_FUZZ_SEEDS scales the number of seeds per mode (default 40).
"""

import os

import pytest

from mgpu.device import rust_available
from mgpu.fuzz import diff_case, equiv_case, random_case, raw_case

pytestmark = pytest.mark.skipif(not rust_available(), reason="mgpu._core not built")
SEEDS = int(os.environ.get("MGPU_FUZZ_SEEDS", "40"))
# Offset the seeds so pytest and the CLI default range (0..) cover different cases.
BASE = 100_000


@pytest.mark.parametrize("seed", range(BASE, BASE + SEEDS))
def test_diff_structured(seed):
    assert diff_case(random_case(seed, "diff")) is None


@pytest.mark.parametrize("seed", range(BASE, BASE + SEEDS))
def test_diff_raw_words(seed):
    assert diff_case(raw_case(seed)) is None


@pytest.mark.parametrize("seed", range(BASE, BASE + SEEDS // 2))
def test_diff_well_defined(seed):
    assert diff_case(random_case(seed, "eq")) is None


@pytest.mark.parametrize("seed", range(BASE, BASE + SEEDS))
def test_schedule_independence(seed):
    assert equiv_case(random_case(seed, "eq")) is None


def test_model_schedule_independence():
    """The model also satisfies E1/E2/E4 (a few cases; it is slow)."""
    for seed in range(BASE, BASE + 3):
        assert equiv_case(random_case(seed, "eq", size=25), backend="model") is None


def test_generators_are_deterministic():
    a, b = random_case(5, "diff"), random_case(5, "diff")
    assert a.words == b.words and a.memory == b.memory and a.params == b.params


def test_racy_program_is_detected_by_equivalence_runner():
    """Sanity check of the checker: a cross-warp race must be reported."""
    from mgpu.fuzz.gen import DATA, FuzzCase
    from mgpu.isa import encode
    # every warp writes its WARP_ID to the same word: the last writer depends on the schedule
    words = [encode("V_GETSR", rd=1, ra=3),                  # v1 = WARP_ID
             encode("V_ADD", rd=2, ra=0, rb=1, sb=1),        # v2 = s1 = DATA
             encode("V_SW", rd=1, ra=2, imm=0),
             encode("EXIT")]
    case = FuzzCase(0, "eq", words, {DATA: [0]}, grid=1, block=128, args=[DATA], params={})
    assert "E1 memory differs" in equiv_case(case)
