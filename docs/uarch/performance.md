# Performance model and design choices

The timing model (`rust/mgpu-core/src/timing.rs`) implements the timing contract of
[`core.md`](core.md) §11. It runs the functional engine in driven mode and decides, cycle by cycle,
when each warp issues. As a result:
- **Its results are always the same as a plain run.** `tests/test_timing.py` checks this on the whole
  kernel suite.
- **Its cycle counts are exact for the contract.** The micro-benchmark tests check hand-derived
  numbers: 116 + 4N cycles for N scalar instructions, 116 + 8N for a dependent vector chain, and one
  extra cycle per branch.

```bash
mgpu-run kernels/matmul.s --grid 16 --block 256 --arg in:A.npy --arg in:B.npy \
         --arg out:C.npy:4096 --arg 64 --arg 64 --arg 64 --timing --warps 8 --lanes 8
python tools/sweep.py          # regenerates every table below (about 4 seconds)
```

From Python, use `Device(timing={"warps": 8, "lanes": 8, "mem_latency": 100})`; the report is in
`dev.timing_report`. From C, use `mgpu_launch_timed`.

## 1. Reading the report

| Field | Meaning |
|---|---|
| cycles | Cycles from launch to the last block finishing (also in the `CYCLES` register) |
| IPC | Instructions issued per cycle, over the whole machine (at most 1 per core) |
| issue-slot utilization | Fraction of a resident core's cycles in which it issued |
| SIMD utilization | Average fraction of the 32 lanes enabled in vector instructions (divergence) |
| busy % | Cycles each unit was occupied, relative to resident-core cycles |
| idle cycles | Why a core issued nothing in a cycle. `unit`: the next instruction's unit was busy. `mem`: warps waiting for loads. `ifetch`: instruction buffers empty. `bar`/`fence`: waiting at barriers or fences. `inflight`: every warp's instruction is still in the pipeline |

## 2. Measurements

8192-element problems and 64×64 matrices and images, from `tools/sweep.py`.

### Cycles vs warp slots (issue-slot utilization in parentheses); L = 8, T_mem = 100

| kernel | W = 2 | W = 4 | W = 8 | W = 16 |
|---|---|---|---|---|
| `saxpy` | 52,589 (8%) | 28,525 (14%) | 17,005 (22%) | 11,469 (31%) |
| `reduce` | 115,926 (25%) | 77,652 (40%) | 70,036 (48%) | 68,712 (52%) |
| `warp_reduce` | 83,053 (12%) | 52,141 (20%) | 39,885 (26%) | 34,605 (30%) |
| `collatz` | 1,604,637 (26%) | 1,271,368 (33%) | 1,232,595 (34%) | 1,225,888 (34%) |
| `histogram` | 30,258 (10%) | 20,330 (17%) | 16,509 (23%) | 15,574 (30%) |
| `sigmoid` | 52,589 (14%) | 29,869 (24%) | 20,269 (33%) | 15,469 (41%) |
| `matmul` | — | — | 150,949 (53%) | 150,949 (53%) |
| `conv3x3` | 162,965 (5%) | 162,965 (5%) | 162,965 (5%) | 162,965 (5%) |

### Cycles vs physical lanes (speedup over L = 8); W = 8, T_mem = 100

| kernel | L = 8 | L = 16 | L = 32 |
|---|---|---|---|
| `saxpy` | 17,005 | 15,213 (1.12x) | 14,573 (1.17x) |
| `reduce` | 70,036 | 52,850 (1.33x) | 44,983 (1.56x) |
| `warp_reduce` | 39,885 | 27,373 (1.46x) | 23,405 (1.70x) |
| `collatz` | 1,232,595 | 662,690 (1.86x) | 447,296 (2.76x) |
| `histogram` | 16,509 | 15,183 (1.09x) | 14,694 (1.12x) |
| `sigmoid` | 20,269 | 18,765 (1.08x) | 18,381 (1.10x) |
| `matmul` | 150,949 | 108,528 (1.39x) | 104,241 (1.45x) |
| `conv3x3` | 162,965 | 156,759 (1.04x) | 153,624 (1.06x) |

### Cycles vs memory latency; W = 8, L = 8

| kernel | 50 | 100 | 200 | 400 | 400 vs 50 |
|---|---|---|---|---|---|
| `saxpy` | 12,155 | 17,005 | 26,705 | 46,105 | 3.79x |
| `reduce` | 66,736 | 70,036 | 76,636 | 89,836 | 1.35x |
| `warp_reduce` | 35,035 | 39,885 | 49,585 | 68,985 | 1.97x |
| `collatz` | 1,229,345 | 1,232,595 | 1,239,095 | 1,252,095 | 1.02x |
| `histogram` | 14,459 | 16,509 | 20,615 | 28,981 | 2.00x |
| `sigmoid` | 17,019 | 20,269 | 26,769 | 39,769 | 2.34x |
| `matmul` | 143,151 | 150,949 | 166,225 | 195,625 | 1.37x |
| `conv3x3` | 102,065 | 162,965 | 284,765 | 528,365 | 5.18x |

### Default design point (W = 8, L = 8, T_mem = 100): where the cycles go

| kernel | cycles | IPC | SIMD util. | busy % valu/fpu/sfu/lsu | idle cycles (>= 5%) |
|---|---|---|---|---|---|
| `saxpy` | 17,005 | 0.18 | 100% | 37 / 7 / 0 / 28 | mem 52%, unit 18%, ifetch 6% |
| `reduce` | 70,036 | 0.46 | 39% | 37 / 12 / 0 / 50 | unit 32%, bar 6%, ifetch 6%, mem 6% |
| `warp_reduce` | 39,885 | 0.26 | 92% | 69 / 0 / 0 / 6 | unit 34%, mem 29%, ifetch 6%, inflight 5% |
| `collatz` | 1,232,595 | 0.34 | 45% | 95 / 0 / 0 / 0 | unit 49%, ifetch 15% |
| `histogram` | 16,509 | 0.23 | 100% | 44 / 0 / 0 / 64 | unit 32%, mem 26%, ifetch 10%, inflight 6% |
| `sigmoid` | 20,269 | 0.28 | 100% | 42 / 18 / 48 / 15 | unit 29%, mem 18%, inflight 13%, ifetch 7% |
| `matmul` | 150,949 | 0.52 | 100% | 72 / 22 / 0 / 63 | unit 30%, mem 12% |
| `conv3x3` | 162,965 | 0.05 | 100% | 6 / 3 / 0 / 8 | mem 89% |


## 3. What the numbers say, and the resulting choices

**Warp slots: W = 8.**
- Going from 2 to 8 warps speeds the suite up by 1.3–3.1×. More warps are the only way this
  machine hides memory latency.
- From 8 to 16 warps:
  - compute-bound kernels gain under 15% (reduce, collatz, warp_reduce, histogram);
  - matmul gains nothing, since its 256-thread block already fills 8 slots;
  - only the purely memory-bound saxpy and sigmoid gain 1.3–1.5×.
- 16 slots would double the vector register file to 3 × 64 KiB of block RAM. **8 is the knee.**

**Physical lanes: L = 8.**
- Quadrupling the datapath to L = 32 costs about 4× the ALU, FPU and register-port area, but buys
  only 1.06–1.7× on most kernels.
- The exception is collatz (2.8×). It is entirely VALU-bound because divergence keeps it issuing
  vector instructions with half the lanes idle. More lanes speed that up, but do not fix the waste.
- Memory-bound kernels barely move. **L = 8 is the efficient point** for an FPGA-sized core; L = 16 is
  the natural upgrade if area allows.

**Memory.**
- Latency-sensitive kernels (saxpy 3.8×, conv3x3 5.2× from 50 to 400 cycles) have little arithmetic
  per byte. Kernels with reuse (matmul through shared memory, reduce) are barely affected.
- The 128-byte segment coalescing keeps saxpy at one request per warp access. Without it, saxpy would
  be 32× more memory-bound.

**I-cache: 2 KiB is plenty.**
- Every kernel hits at least 99.95% of the time; the only misses are the 1–3 cold misses for its
  lines.
- Even a single 128-byte line costs saxpy nothing, but costs matmul 12% (169,328 vs 150,949 cycles),
  because its loops span three lines.
- Without any cache, every fetch would wait a full memory round trip. That is why the cache is in
  the design, and 2 KiB leaves room for larger kernels.

**One resident block per core is a real limitation (conv3x3).**
- conv3x3 launches 64-thread blocks, so only 2 of the 8 warp slots are ever used, and the core sits
  idle 89% of the time waiting for memory. More slots or lanes do nothing for it.
- The cure is to let a core hold several blocks at once, each with its own shared-memory partition.
- This is the most valuable extension after v1. It is kept out of v1 to keep the dispatcher and
  barrier unit simple, and kernels can avoid the problem by using blocks of ≥ 256 threads.

**Where the remaining time goes at W = 8, L = 8.**
- `unit` stalls (18–49%) mean a warp was ready but the VALU was still busy with the previous
  4-pass instruction. That is the expected price of a narrow datapath: the VALU is the most-used unit,
  69–95% busy in the compute-bound kernels.
- `mem` stalls dominate the streaming kernels.
- The SFU (2 datapaths, 16 cycles per instruction) is 48% busy in sigmoid. That is acceptable for a
  unit few kernels use.

## 4. Model limitations *(informative)*

- **Request order.** Memory requests are granted in the order the model generates them, not strictly
  by request cycle across cores. This only matters when several cores contend for memory in the same
  cycles.
- **Execution order.** Instructions execute functionally at issue. For racy programs, the interleaving
  may therefore differ from what the hardware would produce (equivalence.md W1). Well-defined programs
  are unaffected.
- **No DRAM detail.** Banks, refresh and row-buffer effects are not modelled: memory is a fixed
  latency plus a bandwidth token.
