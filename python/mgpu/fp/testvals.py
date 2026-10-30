"""Interesting binary32 operand generators shared by the float tests and fuzzers."""

import random

SPECIALS = [
    0x00000000, 0x80000000,                 # +-0
    0x00000001, 0x807FFFFF, 0x00400000,     # denormals
    0x00800000, 0x80800000, 0x00800001,     # smallest normals
    0x7F7FFFFF, 0xFF7FFFFF, 0x7F7FFFFE,     # largest finite
    0x7F800000, 0xFF800000,                 # +-inf
    0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FBFFFFF,   # NaNs (quiet, negative, signalling)
    0x3F800000, 0xBF800000, 0x3F800001, 0x3F7FFFFF,   # around 1
    0x40000000, 0x3F000000, 0x4B000000, 0x4B7FFFFF,   # 2, 0.5, 2^23, 2^24 - 1
    0x4F000000, 0xCF000000, 0x4EFFFFFF, 0xCF000001,   # around +-2^31 (F2I saturation)
    0x34000000, 0x33800000, 0x0C800000, 0x1F800000,   # small powers of two
]


def random_f32(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.15:
        return rng.choice(SPECIALS)
    if r < 0.30:  # near the flush boundary
        return (rng.getrandbits(1) << 31) | (rng.randint(0, 3) << 23) | rng.getrandbits(23)
    if r < 0.40:  # near overflow
        return (rng.getrandbits(1) << 31) | (rng.randint(0xFC, 0xFE) << 23) | rng.getrandbits(23)
    if r < 0.75:  # moderate exponents
        return (rng.getrandbits(1) << 31) | (rng.randint(100, 154) << 23) | rng.getrandbits(23)
    return rng.getrandbits(32)


def random_pair(rng: random.Random) -> tuple[int, int]:
    a = random_f32(rng)
    r = rng.random()
    if r < 0.25:  # cancellation: b close to -a
        return a, (a ^ 0x80000000) + rng.randint(-3, 3) & 0xFFFFFFFF
    if r < 0.40:  # same binade
        return a, (a & 0xFF800000) | rng.getrandbits(23)
    return a, random_f32(rng)
