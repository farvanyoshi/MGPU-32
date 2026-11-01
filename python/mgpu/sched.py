"""The reference schedule shared by all MGPU-32 simulator backends (docs/sim/README.md).

Both the Python model and the Rust engine pick warps with exactly this logic, so they
produce identical global traces for any program.
"""

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


class Xorshift64Star:
    """xorshift64* PRNG used for seeded warp interleaving."""

    def __init__(self, seed: int):
        self.state = (seed ^ GOLDEN) & MASK64 or GOLDEN

    def next(self) -> int:
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & MASK64
        x ^= x >> 27
        self.state = x
        return (x * 0x2545F4914F6CDD1D) & MASK64

    def below(self, n: int) -> int:
        """Uniform-ish index in [0, n)."""
        return (self.next() >> 32) % n


def pick(eligible: list, last, prng: Xorshift64Star | None):
    """Choose the next warp key from `eligible`, a list of (block_id, warp_id) sorted ascending.

    Round-robin (prng is None): the first key strictly greater than `last`, wrapping to the
    smallest key. Seeded: eligible[prng.below(len(eligible))].
    """
    if prng is not None:
        return eligible[prng.below(len(eligible))]
    if last is not None:
        for k in eligible:
            if k > last:
                return k
    return eligible[0]
