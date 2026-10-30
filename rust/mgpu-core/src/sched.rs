//! The reference schedule shared by all backends (docs/sim/README.md). Mirrors
//! python/mgpu/sched.py exactly.

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// xorshift64* PRNG used for seeded warp interleaving.
#[derive(Clone, Debug)]
pub struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    pub fn new(seed: u64) -> Self {
        let s = seed ^ GOLDEN;
        Xorshift64Star {
            state: if s == 0 { GOLDEN } else { s },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 32) % n as u64) as usize
    }
}

/// Pick the next warp key from `eligible` (sorted ascending). See python/mgpu/sched.py.
pub fn pick(
    eligible: &[(u32, u32)],
    last: Option<(u32, u32)>,
    prng: Option<&mut Xorshift64Star>,
) -> (u32, u32) {
    if let Some(p) = prng {
        return eligible[p.below(eligible.len())];
    }
    if let Some(l) = last {
        if let Some(k) = eligible.iter().find(|&&k| k > l) {
            return *k;
        }
    }
    eligible[0]
}
