# Module interfaces

Conventions:
- Every channel is a **valid/ready** handshake. A transfer happens in a cycle where both `valid` and
  `ready` are 1.
- Once `valid` is raised, the payload must hold steady until the transfer happens.
- `ready` may depend on `valid`, but `valid` must not depend on `ready`.
- Widths use the parameters of [README.md](README.md) §2: `C` cores, `W` warp slots, `L` lanes,
  `P = 32/L` passes, `WID = log2(W)`.

## 1. Host CSR port (`mgpu_top`)

| Signal | Dir | Width | Meaning |
|---|---|---|---|
| `csr_valid` | in | 1 | Request |
| `csr_ready` | out | 1 | Always 1: one request per cycle |
| `csr_write` | in | 1 | 1 = write, 0 = read |
| `csr_addr` | in | 8 | Byte offset (ISA §13.1); bits [1:0] are ignored |
| `csr_wdata` | in | 32 | Write data |
| `csr_rvalid` | out | 1 | Read data valid, one cycle after the read request |
| `csr_rdata` | out | 32 | Read data |
| `irq_done` | out | 1 | One-cycle pulse when a launch ends (`busy` falls) |

## 2. Memory port (`mgpu_top` ↔ memory)

Four channels, like AXI's address, write-data, read-data and write-response channels. Read data and
write acknowledgements have separate channels, because a READ's data and a WRITE's acknowledgement can
complete in the same cycle and neither may be delayed (core.md §11.4). The GPU is always ready for
read data and acknowledgements.

**Request channel**

| Signal | Dir | Width | Meaning |
|---|---|---|---|
| `mreq_valid` / `mreq_ready` | out / in | 1 / 1 | Handshake. Memory accepts a request when it is free (one every `T_seg` = 8 cycles) |
| `mreq_op` | out | 2 | 0 READ, 1 WRITE |
| `mreq_addr` | out | 32 | Segment address, 128-byte aligned |
| `mreq_tag` | out | 16 | Returned with the response |
| `mreq_mask` | out | 32 | WRITE: word enables (bit *i* = word *i* of the segment) |

**Write-data channel**: the 8 beats of a WRITE, beat *k* in cycle `a + k`, where `a` is the cycle
the request is accepted (beat 0 comes with the request).

| Signal | Dir | Width | Meaning |
|---|---|---|---|
| `mwd_valid` | out | 1 | Beat valid |
| `mwd_data` | out | 128 | Words 4*k* .. 4*k*+3 in beat *k* |
| `mwd_last` | out | 1 | Beat 7 |

**Read-data channel**: the 8 beats of a READ, in cycles `a + T_mem + 1` … `a + T_mem + 8`.

| Signal | Dir | Width | Meaning |
|---|---|---|---|
| `mrd_valid` | in | 1 | Beat valid |
| `mrd_tag` | in | 16 | The request's tag |
| `mrd_data` | in | 128 | Words 4*k* .. 4*k*+3 in beat *k* |
| `mrd_last` | in | 1 | Beat 7 |

**Write-acknowledgement channel**: in cycle `a + T_mem`, once the WRITE is applied.

| Signal | Dir | Width | Meaning |
|---|---|---|---|
| `mack_valid` | in | 1 | Acknowledgement |
| `mack_tag` | in | 16 | The request's tag |

Atomics never appear on this port: the atomic unit turns each one into a READ followed by a WRITE
(top.md §5).

## 3. Dispatcher ↔ core

| Signal | Dir (core) | Width | Meaning |
|---|---|---|---|
| `start_valid` / `start_ready` | in / out | 1 / 1 | Start a block. `start_ready` = the core is idle |
| `start_block_id` | in | 32 | `BLOCK_ID` |
| `start_block_dim` | in | 11 | `BLOCK_DIM` (≤ 1024) |
| `start_grid_dim` | in | 32 | `GRID_DIM` |
| `start_entry` | in | 32 | `ENTRY` |
| `start_args` | in | 8×32 | `ARG0`–`ARG7` |
| `block_done` | out | 1 | One-cycle pulse: all warps of the block exited |
| `fault_valid` | out | 1 | One-cycle pulse with the fault record below |
| `fault_code`, `fault_pc`, `fault_block`, `fault_warp`, `fault_lane`, `fault_info` | out | 8, 32, 32, 5, 5, 32 | ISA §11 |
| `abort` | in | 1 | Stop issuing; drain; return to idle |
| `idle` | out | 1 | No warps live and no memory traffic outstanding |

## 4. Core ↔ memory arbiter (per core, two ports)

The **I-fetch port** uses the READ subset of §2, with request and response channels only. The **LSU
port** uses the full request, write-data and response channels of §2, plus atomic requests:

| Signal | Dir (core) | Width | Meaning |
|---|---|---|---|
| `lreq_atomic` | out | 1 | This request is an atomic, not a READ/WRITE |
| `lreq_aop` | out | 3 | ADD, MIN, MAX, XCHG, AND, OR (funct[2:0]) |
| `lreq_lanes` | out | 32 | Participating lanes |
| `lreq_offs` | out | 32×5 | Word offset within the segment, per lane |
| `lreq_operands` | out | 32×32 | Operand per lane |

The arbiter adds the `{core, port}` bits to the tag on the way out and strips them on the way back.

### 4.1 As built in Phase 5b (`mgpu_core`)

The core's ports are the segment-wide form of §3 and §4. The Phase 5c top adds the beat conversion,
tags and arbitration:

| Port | Signals | Meaning |
|---|---|---|
| Dispatch | `start`, `new_launch`, `entry`, `block_id`, `block_dim`, `grid_dim`, `num_warps`, `args`, `mem_last` | `start` is taken when the core is idle or in its `block_done` cycle; `new_launch` invalidates the I-cache |
| Status | `idle`, `block_done`, `kill` | `block_done` is high in the first cycle after the block completes (all warps exited, stores acknowledged); `kill` aborts the block at once |
| I-fetch | `if_req_{valid,addr,id}`, `if_resp_{valid,id,data}` | READ of one segment; `id` = the slot of the warp that missed; data is the whole segment (32 words) in the response cycle |
| LSU request | `lsu_req` (`mreq_t`) | One segment per cycle: kind, `{warp, segment index}` tag, address, WRITE mask and data, ATOMIC lanes, offsets, operands and operation |
| LSU responses | `rd_*`, `at_*`, `ack_*` | READ segment data, ATOMIC old values per lane, WRITE acknowledgements; each at most one per cycle |
| Retire | `ret_valid`, `ret` (`retire_t` × NRET) | One record per completed instruction (`mgpu_record` fields plus the issue clock); a fault is a record with event 3 |

A response is delivered in its completion cycle as defined in core.md §11.4 (a READ at
`a + T_mem + 8`).

## 5. Core-internal pipeline interfaces

Every stage register carries:
- the warp slot `wid` (`WID` bits)
- the PC
- the instruction word and the decoded control bundle (`mgpu_isa_pkg::ctrl_t`)
- EXEC, sampled at operand read

| Boundary | Payload beyond the common fields |
|---|---|
| scheduler → fetch | (none) |
| fetch → decode | Instruction word, or a fetch fault (`code 3`, `info = pc`) |
| decode → operand read | `ctrl_t`, or an illegal-instruction fault (`code 1`, `info = word`) |
| operand read → unit | Scalar operands A, B, C (32 bits each); for vector units, per pass *p*, the lane slices `va`, `vb`, `vc` (`L`×32 each, sb-broadcast applied) |
| unit → writeback | Scalar result + `swe`, or vector result slice + pass index + lane-write mask (EXEC slice) + `vwe`; branch outcome (`taken`, `target`) |
| writeback → warp table | `wid`, next PC, `exit`, `bar_arrive`, `wait_mem`, `fault` |

Vector register file ports, per copy:
- one read port of `L`×32 bits, addressed by `{wid, reg, pass}`
- one write port of `L`×32 bits with an `L`-bit lane mask

There are three copies, written in parallel, so an instruction can read three source registers in the
same cycle. The scalar register file has three read ports and one write port, indexed `{wid, reg}`.
