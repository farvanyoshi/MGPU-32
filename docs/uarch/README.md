# MGPU-32 reference microarchitecture

This directory specifies the hardware that the Phase 5 RTL implements. The ISA
([`docs/isa`](../isa/MGPU-32-ISA.md)) says *what* every instruction does. This specification says
*how* one particular implementation does it, cycle by cycle and signal by signal.

Nothing here changes the ISA. Any behaviour visible to software must match the reference simulators,
as defined in [`equivalence.md`](../isa/equivalence.md).

| Document | Contents |
|---|---|
| [`top.md`](top.md) | Top level: host interface, block dispatcher, memory arbiter, atomic unit, fault handling |
| [`core.md`](core.md) | One core: warp table, scheduler, fetch, decode, register files, execution units, LSU, barriers |
| [`interfaces.md`](interfaces.md) | Signal-level tables for every module boundary |
| [`performance.md`](performance.md) | The cycle-level timing model and the measurements behind the parameter choices |
| [`verification.md`](verification.md) | How the RTL is verified against the reference simulators |

## 1. Design goals

1. **Correct by construction.** Each warp has at most one instruction in flight. This is the key
   simplification of the original design note: there is no forwarding, no register hazard detection
   and no branch prediction. Latency is hidden by interleaving warps.
2. **FPGA-friendly.** Register files and shared memory map onto dual-port block RAM, and the vector
   datapath is narrower than the warp (8 physical lanes run a 32-lane instruction in 4 passes).
3. **Parameterized.** Warp slots, physical lanes, number of cores and memory timing are parameters
   of both the RTL and the timing model.
4. **Observable.** Performance counters expose the same statistics as the timing model, so the two can
   be compared.

## 2. Parameters

| Parameter | Symbol | Default | Range | Consequence |
|---|---|---|---|---|
| Cores | `C` | 1 (RTL), 1–4 (model) | 1–16 | Blocks are dispatched to idle cores |
| Warp slots per core | `W` | 8 | 2–32, a power of two | `MAX_BLOCK_DIM = 32·W` (256 by default) |
| Resident blocks per core | — | 1 | fixed | One 16 KiB shared memory per core |
| Physical vector lanes | `L` | 8 | 8, 16, 32 | A vector instruction takes `P = 32/L` passes |
| SFU datapaths | `S` | 2 | 1–8 | The SFU takes 32/`S` cycles per instruction |
| I-cache | — | 2 KiB, direct-mapped, 128 B lines | — | 16 lines |
| Memory latency | `T_mem` | 100 cycles | ≥ 1 | From request accepted to first response beat |
| Memory bandwidth | `T_seg` | 8 cycles per 128 B segment | ≥ 1 | 128-bit data bus, 8 beats per segment |
| Global memory size | `MEM_SIZE` | 64 MiB (simulation) | multiple of 4 KiB | Readable through the host register `MEM_SIZE` |

[`performance.md`](performance.md) explains why these are the defaults.

## 3. Block diagram

```
                 host CSR bus (valid/ready or AXI4-Lite)
                               |
                     +---------v----------+
                     |   mgpu_host_regs   |  ISA section 13: launch registers, GO, STATUS, ERR_*
                     +---------+----------+
                               | launch / done / fault
                     +---------v----------+
                     |  mgpu_dispatcher   |  validates the launch, hands blocks to idle cores
                     +--+--------------+--+
                        |              |
              +---------v---+     +----v--------+
              |  mgpu_core  | ... |  mgpu_core  |   C cores (one block resident per core)
              +--+-------+--+     +--+-------+--+
          I-fetch|  LSU  |           |       |
              +--v-------v-----------v-------v--+
              |          mgpu_mem_arbiter       |  round robin, 128 B segment requests
              +----------------+----------------+
                               |
                     +---------v----------+
                     |   mgpu_atomic_unit |  executes V_ATOM read-modify-writes at memory
                     +---------+----------+
                               |  memory port: 128-bit data, 8-beat bursts
                          global memory (testbench model or DRAM controller)
```

Inside a core:

```
  +-------------+   +-------+   +--------+   +-----------------+   +---------+   +-----------+
  | warp table  |-->| sched |-->| fetch  |-->| decode + operand|-->| execute |-->| writeback |
  | W slots     |   | (RR)  |   | I-cache|   | read (SRF, VRF) |   |  units  |   | (EXEC)    |
  +------^------+   +-------+   +--------+   +-----------------+   +----+----+   +-----+-----+
         |                                                            |              |
         +---------------------- warp ready again <-------------------+--------------+
  execute units: SALU+branch | VALU (L lanes) | FPU (L lanes) | SFU (S lanes) | LSU (+ shared memory)
```

## 4. RTL module hierarchy

| Module | File | Spec |
|---|---|---|
| `mgpu_top` | `rtl/mgpu_top.sv` | [top.md](top.md) §1 |
| `mgpu_host_regs` | `rtl/top/mgpu_host_regs.sv` | top.md §2 |
| `mgpu_dispatcher` | `rtl/top/mgpu_dispatcher.sv` | top.md §3 |
| `mgpu_mem_arbiter` | `rtl/top/mgpu_mem_arbiter.sv` | top.md §4 |
| `mgpu_atomic_unit` | `rtl/top/mgpu_atomic_unit.sv` | top.md §5 |
| `mgpu_core` | `rtl/core/mgpu_core.sv` | [core.md](core.md) |
| `mgpu_warp_table`, `mgpu_scheduler` | `rtl/core/` | core.md §2–3 |
| `mgpu_fetch`, `mgpu_icache` | `rtl/core/` | core.md §4 |
| `mgpu_decode` | `rtl/core/` | core.md §5 |
| `mgpu_srf`, `mgpu_vrf` | `rtl/core/` | core.md §6 |
| `mgpu_salu`, `mgpu_valu`, `mgpu_fpu`, `mgpu_sfu` | `rtl/units/` | core.md §7 |
| `mgpu_lsu`, `mgpu_shared_mem` | `rtl/core/` | core.md §8 |
| `mgpu_barrier` | `rtl/core/` | core.md §9 |
| `mgpu_isa_pkg` | `rtl/pkg/mgpu_isa_pkg.sv` (generated from `spec/isa.toml`) | — |
| `mgpu_sfu_rom` | `rtl/units/mgpu_sfu_rom.sv` (generated from `spec/sfu/`) | ISA `sfu.md` |

## 5. How the ISA rules map to hardware

| ISA rule | Where it is enforced |
|---|---|
| [EX-2], [EX-5]: one PC per warp, sequential semantics | Warp table: one instruction in flight per warp (core.md §2) |
| [EX-3]: EXEC sampled before the instruction | Operand read latches s31 with the other operands (core.md §6) |
| [DEC-1..4]: decode policy | `mgpu_decode`, generated table (core.md §5) |
| [SEM-1]: read before write | Every source is read in the operand stage, before any writeback of the same instruction |
| [MEM-3], [MEM-4]: precise faults, highest lane wins | LSU checks every active lane before issuing any request, then writes lanes in ascending order (core.md §8) |
| [MEM-5]: atomic indivisibility | Atomics execute in the memory-side atomic unit, one lane at a time (top.md §5) |
| [BAR-1..4] | `mgpu_barrier` plus store-drain before arrival (core.md §9) |
| [MM-4]: FENCE | The warp waits until its outstanding stores are acknowledged (core.md §9) |
| [FLT-1], [FLT-2]: abort | The first fault reported to the dispatcher wins and stops every core (top.md §6) |
| [HOST-1..5] | `mgpu_host_regs` + `mgpu_dispatcher` (top.md §2–3) |
| [SR-3]: CLOCK | A per-core cycle counter, reset at launch (core.md §10) |
