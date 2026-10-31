# Top level

Signal names and widths are listed in [`interfaces.md`](interfaces.md). All logic is synchronous to
one clock `clk`, with an active-low synchronous reset `rst_n`.

## 1. `mgpu_top`

`mgpu_top` instantiates:
- one `mgpu_host_regs`
- one `mgpu_dispatcher`
- `C` cores (`mgpu_core`)
- one `mgpu_mem_arbiter`
- one `mgpu_atomic_unit`

It exposes two ports:
- the **host CSR port**, a simple register bus. An AXI4-Lite adapter, `mgpu_axil_csr`, is provided
  for FPGA integration.
- the **memory port**, which carries 128-byte segment transactions. In simulation the testbench
  provides the memory. On an FPGA a DRAM controller adapter sits behind it.

The host reads and writes global memory **through the memory port's owner**, not through the GPU. The
testbench or SoC preloads memory before launch and reads it after GO returns to 0 ([MM-5]).

## 2. Host registers (`mgpu_host_regs`)

Implements the register map of ISA §13.1 exactly.

| Register | Behaviour |
|---|---|
| `ID`, `VERSION` | Constants |
| `MAX_BLOCK_DIM` | The parameter `32·W` |
| `MEM_SIZE` | The parameter; 0 means 2³² |
| `ENTRY`, `GRID_DIM`, `BLOCK_DIM`, `ARG0`–`ARG7` | Writable only while idle ([HOST-1]); writes while running are dropped |
| `GO` | A write with bit 0 = 1 while idle raises a one-cycle `launch` pulse to the dispatcher ([HOST-2]). Reads return `busy` |
| `STATUS`, `ERR_*` | Cleared on `launch` ([HOST-3]); loaded by the dispatcher's `done` / `fault` report |
| `CYCLES` | A counter that runs while `busy`; frozen when the launch ends |

Reads of unmapped offsets return 0, and writes to them are ignored. The CSR port takes one transaction
per cycle with a one-cycle read latency.

## 3. Block dispatcher (`mgpu_dispatcher`)

The dispatcher works through these states:

1. **VALIDATE** (1 cycle after `launch`): applies [HOST-4] in order.
   - If `BLOCK_DIM = 0` or `BLOCK_DIM > 32·W`, it reports fault code 5.
   - Otherwise, if `ENTRY & 3 ≠ 0`, it reports code 2 with `ERR_PC = ERR_INFO = ENTRY`.
   - Otherwise, if `GRID_DIM = 0`, it reports done.

   In each of these cases no core starts, and `busy` drops after this cycle.
2. **DISPATCH**: holds `next_block` (32 bits) and `blocks_done`. Each cycle, if some core is
   `idle` and `next_block < GRID_DIM`, it sends that core a `start` with:
   - the block ID
   - `BLOCK_DIM`
   - `GRID_DIM`
   - `NUM_WARPS = ceil(BLOCK_DIM/32)`
   - `ENTRY`
   - `ARG0`–`ARG7`

   It then increments `next_block`. Cores are chosen in ascending index order, at most one start per
   cycle. The core initializes its warps itself (core.md §2.2).
3. **RUN**: each core raises `block_done` for one cycle when all its warps have exited. When
   `blocks_done = GRID_DIM`, the launch is done: `busy` drops and `STATUS` = 0.
4. **ABORT**: when any core reports a fault, the dispatcher latches it (§6), stops dispatching,
   raises `abort` to every core, and waits until every core is idle. Then `busy` drops and `STATUS`
   holds the code.

Blocks may run on any core, in any order, concurrently ([EX-1]). All warps of a block start on the
same core, in the same cycle.

## 4. Memory arbiter (`mgpu_mem_arbiter`)

Each core has two request ports: **I-fetch** (line reads) and **LSU** (segment reads, segment writes
and atomics). The arbiter grants one request per cycle to the memory side, round robin over the
2·C ports. Responses carry a tag `{core, port, id}` and are routed back.

| Request type | Payload |
|---|---|
| `READ` | 128-byte-aligned segment address |
| `WRITE` | Segment address + 32-bit word mask + 32 data words (sent as 8 beats of 128 bits) |
| `ATOMIC` | Segment address, op, global/shared flag (always global here), and the participating lanes' word offsets and operands, in ascending lane order |

**Ordering.** Requests from one LSU port reach memory in issue order, and memory applies them in
arrival order. A WRITE is acknowledged when it has been applied. This gives [MM-1] within a warp, and
is what FENCE and BAR wait for.

## 5. Atomic unit (`mgpu_atomic_unit`)

The atomic unit sits between the arbiter and memory. A `READ` or `WRITE` passes straight through. An
`ATOMIC` is executed here:
1. It reads the segment.
2. It applies each participating lane's operation in ascending lane order to the word at that lane's
   offset, updating a local copy so that colliding lanes see each other's results.
3. It writes the segment back.
4. It returns the 32 old values.

The unit processes one atomic segment at a time, and blocks other traffic to memory while it does.
That makes every atomic indivisible with respect to every other access, from any core ([MEM-5]).
Shared-memory atomics never reach this unit: the LSU executes them in the core (core.md §8.4).

## 6. Faults and abort

A core reports a fault with `{code, pc, block, warp, lane, info}`. The dispatcher keeps the **first**
report. If several arrive in the same cycle, the lowest core index wins ([FLT-2]).

On `abort`:
- every core stops issuing new instructions;
- instructions already past operand read may complete ([FLT-1]);
- outstanding memory requests drain;
- each core then returns to idle.

The faulting instruction itself never writes registers or memory. The LSU guarantees this by checking
all lanes before issuing any request (core.md §8.1). Other units fault only in decode, before execution.
