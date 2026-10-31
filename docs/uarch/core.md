# Core

A core runs one block at a time, with up to `W` warps (default 8). This document covers its
structure (§1–10) and its exact timing (§11). The timing model in `rust/mgpu-core/src/timing.rs`
implements §11, and the RTL is compared against it.

`P = 32/L` is the number of passes a vector instruction takes (4 for the default `L = 8`).

## 1. Pipeline overview

```
 fetch (F) -> decode (D) -> instruction buffer (per warp)
                                   |
 schedule (S) -> operand read (R) -> execute (X: unit-specific, P passes) -> writeback (W)
```

- **Instruction buffer.** Each warp has a one-entry *instruction buffer*: its next instruction,
  fetched and decoded ahead of time.
- **Scheduling.** The scheduler issues at most one instruction per cycle, from a warp whose buffer is
  full, that has no instruction in flight, and whose execution unit will be free when the instruction
  gets there.
- **No hazard logic.** Because a warp never has two instructions in flight, there is no forwarding,
  scoreboarding or hazard detection.

## 2. Warp table (`mgpu_warp_table`)

### 2.1 Per-slot state

| Field | Width | Meaning |
|---|---|---|
| `live` | 1 | The slot holds a warp of the current block |
| `pc` | 32 | PC of the next instruction to issue |
| `ibuf_valid`, `ibuf_word`, `ibuf_ctrl` | 1, 32, — | The instruction buffer |
| `in_flight` | 1 | An instruction of this warp is between S and W |
| `wait_bar` | 1 | Arrived at BAR, waiting for release |
| `wait_fence` | 1 | Waiting for outstanding stores (FENCE, or BAR before arrival) |
| `stores_out` | 6 | Outstanding global store segments (acknowledged by memory) |
| `exited` | 1 | Executed EXIT |

### 2.2 Block start

On `start`, the core sets up slots `0 … NUM_WARPS−1` in one cycle:
- `live = 1`, `pc = ENTRY`
- every flag cleared
- the warp's scalar registers s1–s8 from ARG0–7, and s31 = the launch EXEC ([HOST] §13.3)

It then spends `NUM_WARPS` cycles clearing s9–s30 and v1–v31: one warp per cycle, every register file
bank written in parallel. Fetch starts as soon as each warp is initialized.

The block is done when every live slot has `exited = 1` and the core has no outstanding memory
requests. Then `block_done` pulses and the core returns to idle.

## 3. Scheduler (`mgpu_scheduler`)

A warp is **ready** if all of these hold:
- `live`
- `ibuf_valid`
- not `in_flight`, `wait_bar`, `wait_fence` or `exited`

Its instruction is **issuable** if the execution unit it needs will accept it at the start of
execute (§11.2).

The scheduler grants the first issuable warp in round-robin order, starting from the slot after the
one granted last. It grants at most one warp per cycle. Fetch requests are arbitrated the same way.

## 4. Fetch and I-cache (`mgpu_fetch`, `mgpu_icache`)

The I-cache is 2 KiB, direct-mapped, with 16 lines of 128 bytes (one memory segment):
- **Tag:** address bits [31:11]; **index:** bits [10:7]; **word:** bits [6:2].
- **Hit:** the word is available one cycle after the request (F). Decode takes one more cycle (D),
  then the instruction buffer is valid.
- **Miss:** the line is read through the I-fetch memory port. The entry immediately takes the new
  tag and becomes *pending*. Requests that hit a pending line wait for its fill. Each fill is tagged
  with the slot of the warp that missed (that warp cannot fetch again before its word arrives, so the
  tag is unique). Fills may overlap; the memory serializes them (§11.4).
- **Fill response:** the word goes to every warp waiting for that fill. The line is installed only if
  the entry is still pending for that same fill; a later miss to the same index may have replaced it.
- **Invalidation:** every line is invalidated at launch. Code must not change during a kernel
  ([MEM-2]), so no other coherence is needed.

**When the next instruction is fetched:**
- For a *sequential* instruction (anything except S_BZ, S_BNZ, S_J and EXIT), fetch of PC + 4
  starts in the cycle after issue. It is a prefetch that overlaps execution.
- For a branch, the fetch starts in the writeback cycle, once the target is known. The buffer is
  then valid one cycle after the warp could otherwise issue, so a branch costs one extra cycle per
  warp. There is no misprediction penalty, because there is no prediction.

A fetch from a PC beyond `MEM_SIZE − 4` does not touch the cache. It turns the buffer entry into a
fetch-fault marker, which faults when the warp is next scheduled ([MEM-7]).

## 5. Decode (`mgpu_decode`)

`mgpu_decode` is combinational and generated from `spec/isa.toml` (`mgpu_isa_pkg`). It produces
`ctrl_t`:

| Field | Meaning |
|---|---|
| unit | SALU, VALU, FPU, SFU or LSU |
| op | Operation within the unit |
| register kinds | Scalar or vector, for rd, ra, rb and rc |
| sb use | Whether rb is broadcast |
| immediate kind | Which immediate field is used and how it is extended |
| memory space | Global or shared |
| flags | Branch / exit / bar / fence |
| `illegal` | [DEC-1], or [DEC-3] for GETSR |

An illegal instruction is carried as a fault marker and faults when issued (code 1, info = word).

## 6. Register files (`mgpu_srf`, `mgpu_vrf`)

**Scalar register file:** W × 32 × 32 bits (1 KiB at W = 8).
- Three read ports, one write port, built from flops or LUTRAM.
- s0 reads as 0. s31 is EXEC, read with the other operands in R ([EX-3]).

**Vector register file:**
- Each copy holds W × 32 registers × P passes entries, each `L` × 32 bits wide: 1024 × 256 bits =
  32 KiB at the defaults.
- There are **three identical copies**: every write goes to all three, and each copy serves one
  source operand. That gives the 3 reads FFMA/IMAD/SEL need in one cycle, using ordinary dual-port
  block RAM (96 KiB in total).
- v0 reads as 0 and writes to it are dropped.

**Operand read in the reference RTL.** An issued instruction reads its *whole* 32-lane
source registers in the operand-read stage. The execution unit is then fed one `L`-lane slice per
pass from that latched copy, and the merged 32-lane result is written once, at completion.
- **EXEC.** Inactive lanes keep the old value, which is read with the operands ([EX-3]).
- **Why this is identical.** A warp never has two instructions in flight, so this is
  indistinguishable from reading and writing slice by slice, and the timing of §11 is unchanged.
- **Ports.** Units complete independently, so the register files have one write port per writing
  unit. A warp still has at most one instruction in flight, so two ports never write the same warp.
  - Scalar register file: the SALU, VALU (V_CMP, V_READLANE) and LSU (S_LW).
  - Vector register file: the VALU, FPU, SFU and two LSU paths.
- **FPGA mapping.** In `rtl/core/mgpu_srf.sv` and `mgpu_vrf.sv` these are behavioural multi-port
  arrays. Mapping them onto block RAM (the three operand copies described above, banked by pass, plus
  a live-value table for the multiple writers) is a synthesis-time refinement, evaluated with Yosys in
  Phase 5c.

## 7. Execution units

| Unit | Executes | Lanes | Occupancy (cycles busy) | Latency (cycles to result) |
|---|---|---|---|---|
| SALU | S_ALU, immediates, S_GETSR, branches, NOP, FENCE, EXIT, BAR, scalar compares | 1 | 1 | 1 |
| VALU | V_ALU, V_ADDI/ANDI, V_CMP (int and float), V_R4 IMAD/SEL, V_GETSR | `L` | `P` | 2 |
| VALU (cross-lane) | V_READLANE, V_SHFL | `L` | `2P` | 2 |
| FPU | FADD, FSUB, FMUL, FMIN, FMAX, FFMA, I2F, F2I | `L` | `P` | 4 |
| SFU | FRCP, FRSQ, FEX2, FLG2 | `S` (2) | `32/S` (16) | 8 |
| LSU | Loads, stores, atomics | see §8 | see §8 | see §8 |

- **Pipelining.** Units are pipelined: a new pass enters every cycle, so occupancy is the number of
  passes. Latency counts from the last pass entering to its result being written back.
- **Passes.** The VALU, FPU and SFU write each pass back as it completes. The instruction completes
  with its last pass.
- **V_CMP** builds its 32-bit mask over the P passes and writes the scalar destination once at the end.
- **V_SHFL and V_READLANE** first read all P slices of `va` into a 32-lane staging register (P cycles),
  then select (P cycles). That is why their occupancy is doubled.
- **Branches** resolve in the SALU, and the new PC is written at W.

## 8. Load/store unit (`mgpu_lsu`)

### 8.1 Address generation and precise faults

1. During the first `P` cycles, the LSU computes the 32 lane addresses (`va + sext(imm)`, or the
   scalar address for S_LW/S_SW).
2. It checks every active lane: misaligned first, then out of range; the lowest faulting lane wins.
3. If any lane faults, the instruction ends here with a fault record, before any request has been
   issued ([MEM-3]).

### 8.2 Global loads and stores (coalescing)

The active lanes' addresses are grouped by 128-byte segment (`addr[31:7]`), producing `n_seg`
distinct segments (1 to 32), in ascending order of each segment's lowest lane:

- **Load.** One READ is issued per segment, one per cycle, on the LSU port. Each lane takes its word
  from its segment's response. The vector write happens when the last response arrives. The warp
  stays `in_flight` until then.
- **Store.** One WRITE is issued per segment, with a word mask of the lanes it covers. Colliding lanes
  are merged into one word in ascending lane order, so the highest lane wins ([MEM-4]). The
  instruction completes when the last request has been *sent*. Each send increments `stores_out`, and
  each acknowledgement decrements it.
- **Scalar S_LW / S_SW** behave as single-lane accesses.

The LSU handles one instruction at a time. Its occupancy is `P` (addresses) plus `n_seg` (request
issue).

### 8.3 Shared memory (`mgpu_shared_mem`)

Shared memory is 16 KiB in 32 banks of 4 bytes; word address *a* lives in bank *a* mod 32. In each
cycle, every bank serves one word, and all lanes that read the same word get it in that cycle
(broadcast).

The **conflict degree** `d` is the largest number of *distinct* words any one bank must serve for the
instruction. A shared load or store occupies the LSU for `P + d` cycles, and its latency is 1.
`sh[tid]` has `d = 1`; `sh[2*tid]` has `d = 2`.

A store writes lanes in ascending order within each bank cycle, so the highest lane wins.

### 8.4 Atomics

- **Shared atomics** (`.S`) execute in the LSU on the shared-memory array, one active lane per
  cycle, in ascending lane order. They occupy the LSU for `P + n_active` cycles, with latency 1.
- **Global atomics** are sent to the memory-side atomic unit (top.md §5), one request per segment,
  carrying the lanes of that segment in ascending order. Each request returns the old values. The warp
  waits for all responses. Lanes in different segments are applied segment by segment, which is
  still ascending lane order within each word.

## 9. Barriers and fences (`mgpu_barrier`)

**BAR** completes in the SALU. Then:
1. If the warp has `stores_out ≠ 0`, it waits (`wait_fence`) until its stores are acknowledged.
   This gives [BAR-4] ordering for global memory; shared stores complete synchronously.
2. It **arrives**:
   - If another warp already waits at a different PC, the core reports a barrier fault ([BAR-2]),
     with `info` = the waiting PC.
   - Otherwise the warp sets `wait_bar`, and the barrier records the PC.
3. **Release** ([BAR-3]): when every live, non-exited warp has `wait_bar`, all of them clear it in
   the same cycle and continue at PC + 4. The same check runs whenever a warp exits.

**FENCE** completes in the SALU, then waits until `stores_out = 0` ([MM-4]).

## 10. CLOCK and performance counters

**CLOCK** is a per-core 32-bit cycle counter, reset to 0 when the block starts. `S_GETSR` and
`V_GETSR` read the value at their issue cycle ([SR-3]).

These counters are readable through a debug port in the RTL and reported by the timing model:

| Counter | Counts |
|---|---|
| `cycles` | Cycles from the launch until the end |
| `issued` | Instructions issued |
| `issue_<unit>` | Instructions issued to each unit |
| `busy_<unit>` | Cycles each unit is occupied |
| `lane_slots` | Sum of active lanes over vector instructions |
| `stall_empty` | Cycles with no live warp able to issue, split by reason: `mem` (a warp waits for a load), `bar`, `fence`, `ifetch` (buffer empty), `unit` (ready warp, busy unit), `inflight` (all ready warps already have an instruction in flight) |
| `mem_reads`, `mem_writes`, `mem_atomics` | Segment requests |
| `icache_hits`, `icache_misses` | I-cache lookups |
| `shared_conflict_cycles` | Extra cycles caused by bank conflicts (`d − 1` summed) |

## 11. Timing contract

These rules define the cycle count of a launch. The timing model implements them exactly. The RTL
targets them, and Phase 5 documents and bounds any difference.

### 11.1 Launch and dispatch

- Launch validation takes 1 cycle (cycle 0).
- A block is dispatched to the lowest-numbered idle core in the next cycle: cycle 1 for the first
  blocks, or the cycle after a core becomes idle.
- Warp *w* of a block started at cycle `s` is initialized at `s + 1 + w`. Its first fetch is
  requested in that cycle.

### 11.2 Issue and completion

When warp *w* issues instruction *I* to unit *U* at cycle *t*:
- Operand read is at *t* + 1, and the first pass enters *U* at *t* + 2.
- *U* must be free at *t* + 2 (`free_at(U) ≤ t + 2`), and is then busy until
  `free_at(U) = t + 2 + occ(I)`.
- The instruction completes (writes back) at `c = t + 2 + occ(I) + lat(I) − 1`.
- The warp is ready again at `c + 1`, if its instruction buffer is valid by then.

Shared-memory and LSU occupancy include the conflict and segment terms of §8.

### 11.3 Fetch

- One fetch request is served per cycle per core, in round-robin order among the warps that need one.
- **Hit:** a request at cycle *f* makes the buffer valid at *f* + 2.
- **Miss:** the I-cache sends a READ at *f* + 1. The buffer is valid at `done + 2`, where `done` is
  the cycle the memory model completes the read (§11.4).
- Sequential instructions request PC + 4 at *t* + 1. Branches request their next PC at *c*. A warp
  that has executed EXIT fetches nothing.

### 11.4 Memory

The memory system is shared by all cores and both ports of each core:
- It accepts at most one segment request every `T_seg` cycles. A request made at cycle *r* is
  accepted at `a = max(r, mem_free)`, and then `mem_free = a + T_seg`.
- A READ completes at `a + T_mem + 8` (the eight data beats).
- A WRITE is acknowledged at `a + T_mem`.
- An ATOMIC holds the memory for two slots (`mem_free = a + 2·T_seg`) and completes at
  `a + 2·T_mem + 8`.
- The LSU sends its segment requests at consecutive cycles: request *k* at `t + 2 + P + k`.
- A **global load** completes at `max(done_k) + 1`.
- A **global store** completes at `t + 2 + P + n_seg − 1`; each acknowledgement lowers `stores_out`
  when it arrives.

### 11.5 Barriers, fences, exit

- A BAR completes at *c* like any SALU instruction. It arrives at `max(c, last store ack)`.
- Release happens at the arrival time of the last arriving warp, or at the completion of the EXIT
  that leaves only waiting warps. Every released warp is ready at `release + 1`.
- FENCE behaves like BAR's store wait, without the barrier.
- EXIT frees the slot at *c*. The block is done when the last warp has exited and every store is
  acknowledged; the core is idle in the next cycle.

### 11.6 End of launch

The launch ends at the first cycle in which every block is done, or when a fault is reported. Then
`CYCLES` equals that cycle number.

## 12. Notes on the choices *(informative)*

- **Narrow lanes (`L = 8`).** A 32-lane FPU is 32 FFMA datapaths. Eight datapaths running four passes
  cost a quarter of the area, and with 8 warps the VALU and FPU stay busy anyway. This is checked in
  [`performance.md`](performance.md).
- **Three VRF copies instead of an operand collector.** This trades block RAM for simpler control and
  one-cycle operand read. An operand collector would save two copies at the cost of an extra pipeline
  stage and arbitration; it is a later optimization.
- **No data cache.** Coalescing plus warp interleaving is how GPUs tolerate DRAM latency. The
  instruction cache *is* needed: without it, every instruction would cost a full memory round trip.
