# Kernel suite

Example MGPU-32 kernels. Each file starts with a comment giving its arguments (s1, s2, …) and launch
shape. Every kernel is checked against numpy, on both simulator backends, by `tests/kernels/`.

| Kernel | Computes | Shows |
|---|---|---|
| [`saxpy.s`](saxpy.s) | `y = a*x + y` | Per-lane addressing; masking the out-of-range tail with EXEC (ISA §15.1) |
| [`reduce.s`](reduce.s) | Per-block float sums | Shared memory, BAR and uniform loops (ISA §15.2) |
| [`warp_reduce.s`](warp_reduce.s) | Integer sum of an array | 5-step `V_SHFL` butterfly: no shared memory or barriers; one global atomic per warp |
| [`collatz.s`](collatz.s) | Collatz step counts | A divergent loop: lanes drop out of EXEC; `V_SEL` instead of branches |
| [`histogram.s`](histogram.s) | 64-bin histogram | Shared-memory atomics (`V_ATOM.ADD.S`) merged with global atomics; grid-stride loop |
| [`sigmoid.s`](sigmoid.s) | `1 / (1 + e^-x)` | SFU (`V_FEX2`, `V_FRCP`); the `V_FDIV`, `V_FNEG` and `S_LI f32(...)` pseudos |
| [`matmul.s`](matmul.s) | `C = A·B` | 16×16 shared-memory tiling, two BARs per tile, FFMA inner loop, division-free 2-D block indexing |
| [`conv3x3.s`](conv3x3.s) | 3×3 image filter | 2-D work as one block per row; uniform weights in scalar registers (`S_LW`) |
| [`scan.s`](scan.s) | Inclusive prefix sum | Hillis–Steele scan in shared memory; a multi-kernel algorithm over three launches |
| [`bad_barrier.s`](bad_barrier.s) | — | The classic bug: warps waiting at different BARs, reported as STATUS 4 |

## Running one

```bash
python -c "import numpy as n; n.save('in.npy', n.random.randint(1, 5000, 4096).astype(n.uint32))"
mgpu-run kernels/collatz.s --grid 32 --block 128 \
         --arg in:in.npy --arg out:steps.npy:4096:uint32 --arg 4096 --stats
```

## What the statistics show

Measured with `--stats` on the Rust engine, 4096 elements unless noted. SIMD utilization is the
average fraction of a warp's 32 lanes active in vector instructions.

| Kernel | Grid × block | Warp instructions | SIMD utilization |
|---|---|---|---|
| `saxpy` | 32 × 128 | 1,536 | 100% |
| `reduce` | 32 × 128 | 14,400 | 42% |
| `warp_reduce` | 32 × 128 | 5,120 | 92% |
| `collatz` | 32 × 128 | 218,712 | 46% |
| `histogram` | 8 × 128 | 2,496 | 100% |
| `sigmoid` | 32 × 128 | 2,816 | 100% |
| `matmul` (64×64×64) | 16 × 256 | 78,144 | 100% |
| `conv3x3` (64×64) | 64 × 64 | 8,192 | 100% |
| `scan` (`scan_block`) | 16 × 256 | 18,304 | 91% |

- **`reduce`** halves its active lanes every round. Its shared-memory tree wastes more than half the
  machine. `warp_reduce` does the same job with shuffles, at 92% utilization and a third of the
  instructions.
- **`collatz`** is the cost of divergence (ISA §9). Each warp loops until its *slowest* lane
  finishes, so more than half of the lane-slots do nothing.
