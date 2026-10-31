//! C ABI (include/mgpu_core.h). Lets C/C++ code, such as a Verilator testbench, drive the
//! engine directly, including the driven mode used for lockstep co-simulation.

use std::ptr;

use crate::gpu::{Config, Gpu, SimError};
use crate::timing::{launch_timed, TimingConfig};
use crate::trace::Record;

pub const MGPU_OK: i32 = 0;
pub const MGPU_ERR_TIMEOUT: i32 = 1;
pub const MGPU_ERR_NOT_RUNNABLE: i32 = 2;
pub const MGPU_ERR_RANGE: i32 = 3;
pub const MGPU_ERR_NULL: i32 = 4;

#[repr(C)]
pub struct MgpuConfig {
    pub mem_size: u64,
    pub max_block_dim: u32,
    pub resident_blocks: u32,
    pub seed: u64,
    pub use_seed: i32,
    pub shared_poison: u32,
    pub max_steps: u64,
    pub trace: i32,
    pub keep_state: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MgpuMemWrite {
    pub space: u32,
    pub addr: u32,
    pub value: u32,
}

#[repr(C)]
pub struct MgpuRecord {
    pub step: u64,
    pub block: u32,
    pub warp: u32,
    pub pc: u32,
    pub word: u32,
    pub has_swrite: i32,
    pub swrite_reg: u32,
    pub swrite_val: u32,
    pub has_vwrite: i32,
    pub vwrite_reg: u32,
    pub vwrite_val: [u32; 32],
    pub exec_after: u32,
    pub n_mwrites: u32,
    pub mwrites: [MgpuMemWrite; 32],
    pub event: u32,
    pub released: i32,
    pub has_fault: i32,
    pub fault_code: u32,
    pub fault_lane: u32,
    pub fault_info: u32,
}

fn err_code(e: &SimError) -> i32 {
    match e {
        SimError::Timeout(_) => MGPU_ERR_TIMEOUT,
        SimError::NotRunnable(_) | SimError::InvalidConfig(_) => MGPU_ERR_NOT_RUNNABLE,
        SimError::HostRange(..) => MGPU_ERR_RANGE,
    }
}

fn fill_record(r: &Record, out: &mut MgpuRecord) {
    out.step = r.step;
    out.block = r.block;
    out.warp = r.warp;
    out.pc = r.pc;
    out.word = r.word;
    out.has_swrite = r.swrite.is_some() as i32;
    let (sr, sv) = r.swrite.unwrap_or((0, 0));
    out.swrite_reg = sr as u32;
    out.swrite_val = sv;
    out.has_vwrite = r.vwrite.is_some() as i32;
    let (vr, vv) = r.vwrite.unwrap_or((0, [0; 32]));
    out.vwrite_reg = vr as u32;
    out.vwrite_val = vv;
    out.exec_after = r.exec_after;
    out.n_mwrites = r.mwrites.len() as u32;
    out.mwrites = [MgpuMemWrite {
        space: 0,
        addr: 0,
        value: 0,
    }; 32];
    for (dst, &(space, addr, value)) in out.mwrites.iter_mut().zip(&r.mwrites) {
        *dst = MgpuMemWrite {
            space: space as u32,
            addr,
            value,
        };
    }
    out.event = r.event as u32;
    out.released = r.released as i32;
    out.has_fault = r.fault.is_some() as i32;
    let (c, l, i) = r.fault.unwrap_or((0, 0, 0));
    out.fault_code = c;
    out.fault_lane = l;
    out.fault_info = i;
}

/// Fill `cfg` with the default configuration.
///
/// # Safety
/// `cfg` must be null or point to writable memory for one `MgpuConfig`.
#[no_mangle]
pub unsafe extern "C" fn mgpu_config_default(cfg: *mut MgpuConfig) {
    let Some(cfg) = cfg.as_mut() else { return };
    let d = Config::default();
    *cfg = MgpuConfig {
        mem_size: d.mem_size,
        max_block_dim: d.max_block_dim,
        resident_blocks: d.resident_blocks as u32,
        seed: 0,
        use_seed: 0,
        shared_poison: d.shared_poison,
        max_steps: d.max_steps,
        trace: 0,
        keep_state: 0,
    };
}

/// Create a GPU. Returns null if `cfg` is null or invalid. Free with `mgpu_destroy`.
///
/// # Safety
/// `cfg` must be null or point to a valid `MgpuConfig`.
#[no_mangle]
pub unsafe extern "C" fn mgpu_create(cfg: *const MgpuConfig) -> *mut Gpu {
    let Some(c) = cfg.as_ref() else {
        return ptr::null_mut();
    };
    let config = Config {
        mem_size: c.mem_size,
        max_block_dim: c.max_block_dim,
        resident_blocks: c.resident_blocks as usize,
        seed: (c.use_seed != 0).then_some(c.seed),
        shared_poison: c.shared_poison,
        max_steps: c.max_steps,
        trace: c.trace != 0,
        keep_state: c.keep_state != 0,
    };
    match Gpu::new(config) {
        Ok(g) => Box::into_raw(Box::new(g)),
        Err(_) => ptr::null_mut(),
    }
}

/// # Safety
/// `gpu` must be null or a pointer returned by `mgpu_create` that was not yet destroyed.
#[no_mangle]
pub unsafe extern "C" fn mgpu_destroy(gpu: *mut Gpu) {
    if !gpu.is_null() {
        drop(Box::from_raw(gpu));
    }
}

/// Copy `n` bytes from `data` into global memory at `addr`.
///
/// # Safety
/// `gpu` must be valid; `data` must point to `n` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn mgpu_mem_write(gpu: *mut Gpu, addr: u64, data: *const u8, n: u64) -> i32 {
    let (Some(g), false) = (gpu.as_mut(), data.is_null() && n > 0) else {
        return MGPU_ERR_NULL;
    };
    let src = if n == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(data, n as usize)
    };
    g.mem_write(addr, src)
        .map_or_else(|e| err_code(&e), |_| MGPU_OK)
}

/// Copy `n` bytes of global memory at `addr` into `out`.
///
/// # Safety
/// `gpu` must be valid; `out` must point to `n` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn mgpu_mem_read(gpu: *const Gpu, addr: u64, out: *mut u8, n: u64) -> i32 {
    let (Some(g), false) = (gpu.as_ref(), out.is_null() && n > 0) else {
        return MGPU_ERR_NULL;
    };
    match g.mem_read(addr, n) {
        Ok(src) => {
            ptr::copy_nonoverlapping(src.as_ptr(), out, n as usize);
            MGPU_OK
        }
        Err(e) => err_code(&e),
    }
}

/// Read a host register (ISA 13.1). Returns 0 for a null `gpu`.
///
/// # Safety
/// `gpu` must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn mgpu_reg_read(gpu: *const Gpu, off: u32) -> u32 {
    gpu.as_ref().map_or(0, |g| g.reg_read(off))
}

/// Write a host register. Writing 1 to GO runs the kernel to completion before returning;
/// the result is MGPU_ERR_TIMEOUT if it exceeded max_steps.
///
/// # Safety
/// `gpu` must be valid.
#[no_mangle]
pub unsafe extern "C" fn mgpu_reg_write(gpu: *mut Gpu, off: u32, value: u32) -> i32 {
    let Some(g) = gpu.as_mut() else {
        return MGPU_ERR_NULL;
    };
    g.reg_write(off, value)
        .map_or_else(|e| err_code(&e), |_| MGPU_OK)
}

/// Validate and start the launch described by the host registers without running it
/// (driven mode). Returns 1 if warps will run, 0 otherwise (see STATUS).
///
/// # Safety
/// `gpu` must be valid.
#[no_mangle]
pub unsafe extern "C" fn mgpu_begin(gpu: *mut Gpu) -> i32 {
    gpu.as_mut().map_or(0, |g| g.begin() as i32)
}

/// Run a begun launch to completion under the reference schedule.
///
/// # Safety
/// `gpu` must be valid.
#[no_mangle]
pub unsafe extern "C" fn mgpu_run(gpu: *mut Gpu) -> i32 {
    let Some(g) = gpu.as_mut() else {
        return MGPU_ERR_NULL;
    };
    g.run().map_or_else(|e| err_code(&e), |_| MGPU_OK)
}

/// 1 while a kernel is running (GO reads 1).
///
/// # Safety
/// `gpu` must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn mgpu_is_running(gpu: *const Gpu) -> i32 {
    gpu.as_ref().map_or(0, |g| g.running() as i32)
}

/// Instructions retired in the current or last launch.
///
/// # Safety
/// `gpu` must be null or valid.
#[no_mangle]
pub unsafe extern "C" fn mgpu_steps(gpu: *const Gpu) -> u64 {
    gpu.as_ref().map_or(0, |g| g.steps)
}

/// Driven mode: retire one instruction of (block, warp) and describe it in `out`
/// (which may be null).
///
/// # Safety
/// `gpu` must be valid; `out` must be null or point to a writable `MgpuRecord`.
#[no_mangle]
pub unsafe extern "C" fn mgpu_step(
    gpu: *mut Gpu,
    block: u32,
    warp: u32,
    out: *mut MgpuRecord,
) -> i32 {
    let Some(g) = gpu.as_mut() else {
        return MGPU_ERR_NULL;
    };
    match g.step(block, warp) {
        Ok(r) => {
            if let Some(o) = out.as_mut() {
                fill_record(&r, o);
            }
            MGPU_OK
        }
        Err(e) => err_code(&e),
    }
}

/// Driven mode: write up to `max_pairs` runnable (block, warp) pairs into `pairs` as
/// [block0, warp0, block1, warp1, ...]. Returns the total number of runnable pairs.
///
/// # Safety
/// `gpu` must be valid; `pairs` must be null or point to `2 * max_pairs` writable u32s.
#[no_mangle]
pub unsafe extern "C" fn mgpu_runnable(gpu: *const Gpu, pairs: *mut u32, max_pairs: u32) -> u32 {
    let Some(g) = gpu.as_ref() else { return 0 };
    let all = g.runnable();
    if !pairs.is_null() {
        for (i, &(b, w)) in all.iter().take(max_pairs as usize).enumerate() {
            *pairs.add(2 * i) = b;
            *pairs.add(2 * i + 1) = w;
        }
    }
    all.len() as u32
}

#[repr(C)]
pub struct MgpuTimingConfig {
    pub cores: u32,
    pub warps: u32,
    pub lanes: u32,
    pub sfu_lanes: u32,
    pub mem_latency: u64,
    pub seg_cycles: u64,
    pub icache_lines: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct MgpuTimingReport {
    pub cycles: u64,
    pub issued: u64,
    pub issue_by_unit: [u64; 5],
    pub busy_by_unit: [u64; 5],
    pub vector_issued: u64,
    pub lane_slots: u64,
    pub stalls: [u64; 6],
    pub core_active_cycles: u64,
    pub mem_reads: u64,
    pub mem_writes: u64,
    pub mem_atomics: u64,
    pub icache_hits: u64,
    pub icache_misses: u64,
    pub shared_conflict_cycles: u64,
}

/// Fill `cfg` with the default timing parameters (docs/uarch/README.md).
///
/// # Safety
/// `cfg` must be null or point to a writable `MgpuTimingConfig`.
#[no_mangle]
pub unsafe extern "C" fn mgpu_timing_config_default(cfg: *mut MgpuTimingConfig) {
    let Some(cfg) = cfg.as_mut() else { return };
    let d = TimingConfig::default();
    *cfg = MgpuTimingConfig {
        cores: d.cores as u32,
        warps: d.warps,
        lanes: d.lanes,
        sfu_lanes: d.sfu_lanes,
        mem_latency: d.mem_latency,
        seg_cycles: d.seg_cycles,
        icache_lines: d.icache_lines as u32,
    };
}

/// Launch the kernel in the host registers under the timing model; fill `out`.
///
/// # Safety
/// `gpu` must be valid; `cfg` must point to a valid config; `out` may be null.
#[no_mangle]
pub unsafe extern "C" fn mgpu_launch_timed(
    gpu: *mut Gpu,
    cfg: *const MgpuTimingConfig,
    out: *mut MgpuTimingReport,
) -> i32 {
    let (Some(g), Some(c)) = (gpu.as_mut(), cfg.as_ref()) else {
        return MGPU_ERR_NULL;
    };
    let tc = TimingConfig {
        cores: c.cores as usize,
        warps: c.warps,
        lanes: c.lanes,
        sfu_lanes: c.sfu_lanes,
        mem_latency: c.mem_latency,
        seg_cycles: c.seg_cycles,
        icache_lines: c.icache_lines as usize,
        ..TimingConfig::default()
    };
    match launch_timed(g, &tc) {
        Ok(r) => {
            if let Some(o) = out.as_mut() {
                *o = MgpuTimingReport {
                    cycles: r.cycles,
                    issued: r.issued,
                    issue_by_unit: r.issue_by_unit,
                    busy_by_unit: r.busy_by_unit,
                    vector_issued: r.vector_issued,
                    lane_slots: r.lane_slots,
                    stalls: r.stalls,
                    core_active_cycles: r.core_active_cycles,
                    mem_reads: r.mem_reads,
                    mem_writes: r.mem_writes,
                    mem_atomics: r.mem_atomics,
                    icache_hits: r.icache_hits,
                    icache_misses: r.icache_misses,
                    shared_conflict_cycles: r.shared_conflict_cycles,
                };
            }
            MGPU_OK
        }
        Err(e) => err_code(&e),
    }
}
