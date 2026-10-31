/*
 * MGPU-32 simulator engine - C interface.
 *
 * Link against rust/target/release/libmgpu_core.a (build with
 * `cargo build --release --manifest-path rust/Cargo.toml -p mgpu-core`) plus
 * -lpthread -ldl -lm. Semantics are those of the Rust engine and the Python reference
 * model (docs/sim/README.md); host register offsets are listed in the ISA, section 13.1.
 */
#ifndef MGPU_CORE_H
#define MGPU_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes */
#define MGPU_OK               0
#define MGPU_ERR_TIMEOUT      1   /* kernel exceeded max_steps */
#define MGPU_ERR_NOT_RUNNABLE 2   /* driven-mode step of a warp that cannot run */
#define MGPU_ERR_RANGE        3   /* host memory access outside global memory */
#define MGPU_ERR_NULL         4   /* null pointer argument */

/* Host registers (ISA section 13.1) */
#define MGPU_REG_ID            0x00
#define MGPU_REG_VERSION       0x04
#define MGPU_REG_MAX_BLOCK_DIM 0x08
#define MGPU_REG_MEM_SIZE      0x0C
#define MGPU_REG_ENTRY         0x10
#define MGPU_REG_GRID_DIM      0x14
#define MGPU_REG_BLOCK_DIM     0x18
#define MGPU_REG_GO            0x1C
#define MGPU_REG_STATUS        0x20
#define MGPU_REG_ERR_PC        0x24
#define MGPU_REG_ERR_BLOCK     0x28
#define MGPU_REG_ERR_WARP      0x2C
#define MGPU_REG_ERR_LANE      0x30
#define MGPU_REG_ERR_INFO      0x34
#define MGPU_REG_CYCLES        0x38
#define MGPU_REG_ARG(n)        (0x40 + 4 * (n))

/* Commit-record events */
#define MGPU_EV_NONE  0
#define MGPU_EV_EXIT  1
#define MGPU_EV_BAR   2
#define MGPU_EV_FAULT 3

typedef struct mgpu_gpu mgpu_gpu;   /* opaque */

typedef struct {
    uint64_t mem_size;          /* bytes, multiple of 4, <= 2^32 (default 64 MiB) */
    uint32_t max_block_dim;     /* multiple of 32, 32..1024 (default 1024) */
    uint32_t resident_blocks;   /* reference schedule: blocks resident at once (default 1) */
    uint64_t seed;              /* xorshift64* seed, used when use_seed != 0 */
    int32_t  use_seed;          /* 0: round-robin warp interleaving */
    uint32_t shared_poison;     /* initial shared-memory word (default 0xDEADBEEF) */
    uint64_t max_steps;         /* watchdog (default 50,000,000) */
    int32_t  trace;             /* keep commit records internally (Python API only) */
    int32_t  keep_state;        /* keep finished blocks' registers */
} mgpu_config;

typedef struct {
    uint32_t space;             /* 0 global, 1 shared */
    uint32_t addr;
    uint32_t value;
} mgpu_mwrite;

/* One retired instruction (see "Trace format" in docs/sim/README.md). */
typedef struct {
    uint64_t step;
    uint32_t block, warp, pc, word;
    int32_t  has_swrite;
    uint32_t swrite_reg, swrite_val;
    int32_t  has_vwrite;
    uint32_t vwrite_reg;
    uint32_t vwrite_val[32];    /* all 32 lanes after the write */
    uint32_t exec_after;
    uint32_t n_mwrites;
    mgpu_mwrite mwrites[32];    /* in program order */
    uint32_t event;             /* MGPU_EV_* */
    int32_t  released;          /* this instruction completed a barrier */
    int32_t  has_fault;
    uint32_t fault_code, fault_lane, fault_info;
} mgpu_record;

void      mgpu_config_default(mgpu_config *cfg);
mgpu_gpu *mgpu_create(const mgpu_config *cfg);          /* NULL if cfg is invalid */
void      mgpu_destroy(mgpu_gpu *gpu);

int32_t   mgpu_mem_write(mgpu_gpu *gpu, uint64_t addr, const uint8_t *data, uint64_t n);
int32_t   mgpu_mem_read(const mgpu_gpu *gpu, uint64_t addr, uint8_t *out, uint64_t n);
uint32_t  mgpu_reg_read(const mgpu_gpu *gpu, uint32_t off);
/* Writing 1 to MGPU_REG_GO runs the kernel to completion before returning. */
int32_t   mgpu_reg_write(mgpu_gpu *gpu, uint32_t off, uint32_t value);

/* Driven mode: begin a launch, then step warps chosen by the caller. */
int32_t   mgpu_begin(mgpu_gpu *gpu);                     /* 1 if warps will run */
int32_t   mgpu_run(mgpu_gpu *gpu);                       /* finish under the reference schedule */
int32_t   mgpu_is_running(const mgpu_gpu *gpu);
uint64_t  mgpu_steps(const mgpu_gpu *gpu);
int32_t   mgpu_step(mgpu_gpu *gpu, uint32_t block, uint32_t warp, mgpu_record *out);
uint32_t  mgpu_runnable(const mgpu_gpu *gpu, uint32_t *pairs, uint32_t max_pairs);

/* Bit-exact float operations of the reference engine (for RTL unit testbenches). */
enum {
    MGPU_FP_FADD, MGPU_FP_FSUB, MGPU_FP_FMUL, MGPU_FP_FMIN, MGPU_FP_FMAX, MGPU_FP_FFMA,
    MGPU_FP_I2F, MGPU_FP_F2I, MGPU_FP_FRCP, MGPU_FP_FRSQ, MGPU_FP_FEX2, MGPU_FP_FLG2,
    MGPU_FP_FEQ, MGPU_FP_FNE, MGPU_FP_FLT, MGPU_FP_FLE, MGPU_FP_FGT, MGPU_FP_FGE
};
/* Unused operands are ignored; compares return 0/1; an unknown op returns 0xFFFFFFFF. */
uint32_t  mgpu_fp_op(uint32_t op, uint32_t a, uint32_t b, uint32_t c);

/* Cycle-level timing model (docs/uarch/core.md section 11). */
typedef struct {
    uint32_t cores, warps, lanes, sfu_lanes;   /* defaults: 1, 8, 8, 2 */
    uint64_t mem_latency, seg_cycles;          /* defaults: 100, 8 */
    uint32_t icache_lines;                     /* default: 16 (128-byte lines) */
} mgpu_timing_config;

typedef struct {
    uint64_t cycles, issued;
    uint64_t issue_by_unit[5];                 /* salu, valu, fpu, sfu, lsu */
    uint64_t busy_by_unit[5];
    uint64_t vector_issued, lane_slots;
    uint64_t stalls[6];                        /* unit, ifetch, mem, bar, fence, inflight */
    uint64_t core_active_cycles;
    uint64_t mem_reads, mem_writes, mem_atomics;
    uint64_t icache_hits, icache_misses, shared_conflict_cycles;
} mgpu_timing_report;

void      mgpu_timing_config_default(mgpu_timing_config *cfg);
/* Launch the kernel in the host registers under the timing model (like writing GO). */
int32_t   mgpu_launch_timed(mgpu_gpu *gpu, const mgpu_timing_config *cfg, mgpu_timing_report *out);

#ifdef __cplusplus
}
#endif

#endif /* MGPU_CORE_H */
