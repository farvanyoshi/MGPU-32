//! Cycle-level timing model of the reference microarchitecture (docs/uarch/core.md §11).
//!
//! The model decides *when* each warp issues; the functional engine (driven mode,
//! `Gpu::step`) decides *what* the instruction does, at its issue cycle. A warp has at
//! most one instruction in flight, so per-warp program order is preserved and kernel
//! results equal those of a plain run (for well-defined programs, equivalence.md).

use crate::decode::decode;
use crate::gpu::{AccessKind, Gpu, SimError, WARP};
use crate::isa_gen::Op;
use crate::trace::{EV_BAR, EV_EXIT, EV_FAULT};

/// Execution units (core.md §7).
pub const SALU: usize = 0;
pub const VALU: usize = 1;
pub const FPU: usize = 2;
pub const SFU: usize = 3;
pub const LSU: usize = 4;
pub const UNIT_NAMES: [&str; 5] = ["salu", "valu", "fpu", "sfu", "lsu"];

/// Stall reasons for cycles in which a core with live warps issues nothing.
pub const STALL_NAMES: [&str; 6] = ["unit", "ifetch", "mem", "bar", "fence", "inflight"];

#[derive(Clone, Debug)]
pub struct TimingConfig {
    pub cores: usize,
    /// Warp slots per core (W); MAX_BLOCK_DIM = 32 * W.
    pub warps: u32,
    /// Physical vector lanes (L); vector instructions take 32 / L passes.
    pub lanes: u32,
    /// SFU datapaths (S).
    pub sfu_lanes: u32,
    /// Memory latency T_mem (cycles).
    pub mem_latency: u64,
    /// Cycles per 128-byte segment, T_seg (bandwidth).
    pub seg_cycles: u64,
    /// I-cache lines (128 bytes each, direct-mapped).
    pub icache_lines: usize,
    /// Watchdog on the cycle count.
    pub max_cycles: u64,
}

impl Default for TimingConfig {
    fn default() -> Self {
        TimingConfig {
            cores: 1,
            warps: 8,
            lanes: 8,
            sfu_lanes: 2,
            mem_latency: 100,
            seg_cycles: 8,
            icache_lines: 16,
            max_cycles: 1 << 40,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TimingReport {
    pub cycles: u64,
    pub issued: u64,
    pub issue_by_unit: [u64; 5],
    pub busy_by_unit: [u64; 5],
    pub vector_issued: u64,
    pub lane_slots: u64,
    /// Per-core cycles with live warps but no issue, by reason (STALL_NAMES).
    pub stalls: [u64; 6],
    /// Per-core cycles with a block resident (the denominator for issue utilization).
    pub core_active_cycles: u64,
    pub mem_reads: u64,
    pub mem_writes: u64,
    pub mem_atomics: u64,
    pub icache_hits: u64,
    pub icache_misses: u64,
    pub shared_conflict_cycles: u64,
    pub blocks: u64,
}

impl TimingReport {
    /// Instructions per cycle, over the whole machine.
    pub fn ipc(&self) -> f64 {
        self.issued as f64 / self.cycles.max(1) as f64
    }
    /// Fraction of a resident core's cycles in which it issued an instruction.
    pub fn issue_utilization(&self) -> f64 {
        self.issued as f64 / self.core_active_cycles.max(1) as f64
    }
    /// Average fraction of the 32 lanes active in vector instructions.
    pub fn simd_utilization(&self) -> f64 {
        self.lane_slots as f64 / (32 * self.vector_issued.max(1)) as f64
    }
}

/// Shared memory system (core.md §11.4).
struct Memory {
    free: u64,
    latency: u64,
    seg: u64,
}

impl Memory {
    fn accept(&mut self, r: u64, slots: u64) -> u64 {
        let a = r.max(self.free);
        self.free = a + slots * self.seg;
        a
    }
    fn read(&mut self, r: u64) -> u64 {
        self.accept(r, 1) + self.latency + 8
    }
    fn write(&mut self, r: u64) -> u64 {
        self.accept(r, 1) + self.latency
    }
    fn atomic(&mut self, r: u64) -> u64 {
        self.accept(r, 2) + 2 * self.latency + 8
    }
}

#[derive(Clone, Debug)]
struct WarpT {
    wid: u32,
    exited: bool,
    /// The warp may issue at or after this cycle (no instruction in flight).
    ready_at: u64,
    /// The instruction buffer is valid at or after this cycle (None: not yet fetched).
    ibuf_at: Option<u64>,
    /// Fetch to request at or after this cycle (None: nothing to fetch).
    fetch_req: Option<u64>,
    wait_bar: bool,
    /// Arrival cycle at the barrier (valid while wait_bar).
    bar_arrival: u64,
    last_store_ack: u64,
    exec: u32,
    /// Waiting reason while not ready: 2 mem, 3 bar, 4 fence, 5 in flight (STALL_NAMES index).
    waiting: usize,
}

struct CoreT {
    block: Option<u32>,
    start: u64,
    warps: Vec<WarpT>,
    unit_free: [u64; 5],
    rr_issue: usize,
    rr_fetch: usize,
    /// (tag, valid_from) per line; tag u64::MAX = invalid.
    icache: Vec<(u64, u64)>,
    /// Cycle at which the resident block completes (all warps exited, stores acknowledged).
    done_at: Option<u64>,
    idle_from: u64,
}

fn unit_of(op: Op) -> usize {
    use Op::*;
    match op {
        VFadd | VFsub | VFmul | VFmin | VFmax | VI2f | VF2i | VFfma => FPU,
        VFrcp | VFrsq | VFex2 | VFlg2 => SFU,
        SLw | SSw | VLw | VSw | VLds | VSts | VAtomAdd | VAtomMin | VAtomMax | VAtomXchg
        | VAtomAnd | VAtomOr | VAtomAddS | VAtomMinS | VAtomMaxS | VAtomXchgS | VAtomAndS
        | VAtomOrS => LSU,
        VReadlane | VShfl | VGetsr | VAddi | VAndi | VImad | VSel => VALU,
        _ => {
            let m = crate::isa_gen::MNEMONICS[op as usize];
            if m.starts_with("V_") {
                VALU
            } else {
                SALU
            }
        }
    }
}

fn is_branch(op: Op) -> bool {
    matches!(op, Op::SBz | Op::SBnz | Op::SJ)
}

fn is_vector(op: Op) -> bool {
    use Op::*;
    let m = crate::isa_gen::MNEMONICS[op as usize];
    m.starts_with("V_") && !matches!(op, VReadlane)
}

/// Bank-conflict degree of a shared access (core.md §8.3).
fn conflict_degree(lanes: u32, addrs: &[u32; WARP]) -> u64 {
    let mut per_bank: [Vec<u32>; 32] = Default::default();
    for (i, &a) in addrs.iter().enumerate() {
        if lanes >> i & 1 != 0 {
            let word = a >> 2;
            let bank = &mut per_bank[(word % 32) as usize];
            if !bank.contains(&word) {
                bank.push(word);
            }
        }
    }
    per_bank
        .iter()
        .map(|b| b.len() as u64)
        .max()
        .unwrap_or(0)
        .max(1)
}

/// Distinct 128-byte segments, in order of their lowest lane (core.md §8.2).
fn segments(lanes: u32, addrs: &[u32; WARP]) -> usize {
    let mut seen: Vec<u32> = Vec::new();
    for (i, &a) in addrs.iter().enumerate() {
        if lanes >> i & 1 != 0 && !seen.contains(&(a >> 7)) {
            seen.push(a >> 7);
        }
    }
    seen.len()
}

fn launch_exec(block_dim: u32, wid: u32) -> u32 {
    let live = block_dim.saturating_sub(32 * wid).min(32);
    if live == 32 {
        u32::MAX
    } else {
        (1u32 << live) - 1
    }
}

/// Launch the kernel described by the host registers under the timing model. This is the
/// timed equivalent of writing GO: the same launch validation, the same results.
pub fn launch_timed(gpu: &mut Gpu, tc: &TimingConfig) -> Result<TimingReport, SimError> {
    if tc.lanes == 0 || 32 % tc.lanes != 0 || tc.sfu_lanes == 0 || 32 % tc.sfu_lanes != 0 {
        return Err(SimError::InvalidConfig(
            "lanes and sfu_lanes must divide 32".into(),
        ));
    }
    if tc.cores == 0 || tc.warps == 0 || tc.warps > 32 || tc.icache_lines == 0 {
        return Err(SimError::InvalidConfig(
            "bad core/warp/icache parameters".into(),
        ));
    }
    gpu.cfg.max_block_dim = (32 * tc.warps).min(1024);
    let mut rep = TimingReport::default();
    if !gpu.begin() {
        rep.cycles = 1; // validation only (core.md §11.1)
        gpu.reported_cycles = Some(rep.cycles);
        return Ok(rep);
    }
    let (grid, block_dim, num_warps) = gpu.launch_shape();
    let entry = gpu.reg_read(crate::gpu::REG_ENTRY);
    let passes = (32 / tc.lanes) as u64;
    let sfu_occ = (32 / tc.sfu_lanes) as u64;
    let mut mem = Memory {
        free: 0,
        latency: tc.mem_latency,
        seg: tc.seg_cycles,
    };
    let mut cores: Vec<CoreT> = (0..tc.cores)
        .map(|_| CoreT {
            block: None,
            start: 0,
            warps: Vec::new(),
            unit_free: [0; 5],
            rr_issue: 0,
            rr_fetch: 0,
            icache: vec![(u64::MAX, 0); tc.icache_lines],
            done_at: None,
            idle_from: 1,
        })
        .collect();
    let mem_size = gpu.cfg.mem_size;
    let mut next_block: u64 = 0;
    let mut blocks_done: u64 = 0;
    let mut end_cycle: u64 = 0;
    let mut now: u64 = 1;

    while blocks_done < grid as u64 {
        if now > tc.max_cycles {
            return Err(SimError::Timeout(tc.max_cycles));
        }
        let mut next_event = u64::MAX;
        for core in cores.iter_mut() {
            // ---- block completion and dispatch (§11.1)
            if let Some(d) = core.done_at {
                if now > d {
                    core.block = None;
                    core.done_at = None;
                    core.idle_from = d + 1;
                    blocks_done += 1;
                    end_cycle = end_cycle.max(d);
                }
            }
            if core.block.is_none() && next_block < grid as u64 && now >= core.idle_from {
                let bid = next_block as u32;
                next_block += 1;
                core.block = Some(bid);
                core.start = now;
                core.unit_free = [0; 5];
                core.warps = (0..num_warps)
                    .map(|w| WarpT {
                        wid: w,
                        exited: false,
                        ready_at: now + 1 + w as u64,
                        ibuf_at: None,
                        fetch_req: Some(now + 1 + w as u64),
                        wait_bar: false,
                        bar_arrival: 0,
                        last_store_ack: 0,
                        exec: launch_exec(block_dim, w),
                        waiting: 5,
                    })
                    .collect();
            }
            let Some(bid) = core.block else {
                if next_block < grid as u64 {
                    next_event = next_event.min(core.idle_from.max(now + 1));
                }
                continue;
            };
            if let Some(d) = core.done_at {
                next_event = next_event.min(d + 1);
                continue;
            }
            rep.core_active_cycles += 1;

            // ---- fetch: one request per cycle, round robin (§11.3)
            let nw = core.warps.len();
            for k in 0..nw {
                let i = (core.rr_fetch + k) % nw;
                let w = &core.warps[i];
                if w.exited || w.ibuf_at.is_some() {
                    continue;
                }
                let Some(f) = w.fetch_req else { continue };
                if f > now {
                    next_event = next_event.min(f);
                    continue;
                }
                // blocks are created on their first step; until then a warp is at ENTRY
                let pc = gpu.warp(bid, w.wid).map_or(entry, |x| x.pc);
                let ready = if pc as u64 + 4 > mem_size {
                    now + 2
                } else {
                    let line = (pc >> 7) as u64;
                    let idx = (line as usize) % tc.icache_lines;
                    let (tag, valid_from) = core.icache[idx];
                    if tag == line {
                        rep.icache_hits += 1;
                        valid_from.max(now) + 2
                    } else {
                        rep.icache_misses += 1;
                        rep.mem_reads += 1;
                        let done = mem.read(now + 1);
                        core.icache[idx] = (line, done);
                        done + 2
                    }
                };
                let w = &mut core.warps[i];
                w.ibuf_at = Some(ready);
                w.fetch_req = None;
                core.rr_fetch = (i + 1) % nw;
                break;
            }

            // ---- issue: one instruction per cycle, round robin (§3, §11.2)
            let mut issued = false;
            let mut unit_blocked = false;
            for k in 0..nw {
                let i = (core.rr_issue + k) % nw;
                let w = &core.warps[i];
                if w.exited || w.wait_bar || w.ready_at > now {
                    continue;
                }
                let Some(ib) = w.ibuf_at else { continue };
                if ib > now {
                    continue;
                }
                let wid = w.wid;
                let pc = gpu.warp(bid, wid).map_or(entry, |x| x.pc);
                let word = if pc as u64 + 4 <= mem_size {
                    let b = gpu.mem_read(pc as u64, 4)?;
                    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
                } else {
                    0
                };
                let op = decode(word).map(|d| d.op);
                let unit = op.map(unit_of).unwrap_or(SALU);
                if core.unit_free[unit] > now + 2 {
                    unit_blocked = true;
                    next_event = next_event.min(core.unit_free[unit] - 2);
                    continue;
                }
                // Execute functionally at issue; CLOCK reads cycles since the block started.
                let t = now;
                gpu.clock_override = Some(t - core.start);
                let rec = gpu.step(bid, wid)?;
                let acc = gpu.last_access;
                issued = true;
                core.rr_issue = (i + 1) % nw;
                rep.issued += 1;
                rep.issue_by_unit[unit] += 1;
                let w = &mut core.warps[i];
                if rec.event == EV_FAULT {
                    rep.cycles = t + 1;
                    gpu.reported_cycles = Some(rep.cycles);
                    gpu.clock_override = None;
                    return Ok(rep);
                }
                let op = op.expect("non-faulting instruction decodes");
                if is_vector(op) {
                    rep.vector_issued += 1;
                    rep.lane_slots += w.exec.count_ones() as u64;
                }
                w.exec = rec.exec_after;
                // occupancy and latency (§7, §8, §11.4)
                let (occ, lat, mut complete) = match unit {
                    SALU => (1, 1, None),
                    VALU if matches!(op, Op::VShfl | Op::VReadlane) => (2 * passes, 2, None),
                    VALU => (passes, 2, None),
                    FPU => (passes, 4, None),
                    SFU => (sfu_occ, 8, None),
                    _ => {
                        let n_act = acc.lanes.count_ones() as u64;
                        if acc.shared {
                            let d = if acc.kind == AccessKind::Atomic {
                                n_act
                            } else {
                                conflict_degree(acc.lanes, &acc.addrs)
                            };
                            if acc.kind != AccessKind::Atomic {
                                rep.shared_conflict_cycles += d.saturating_sub(1);
                            }
                            (passes + d, 1, None)
                        } else {
                            let n = segments(acc.lanes, &acc.addrs) as u64;
                            let first = t + 2 + passes;
                            let mut last_done = first;
                            for k in 0..n {
                                let r = first + k;
                                match acc.kind {
                                    AccessKind::Load => {
                                        rep.mem_reads += 1;
                                        last_done = last_done.max(mem.read(r));
                                    }
                                    AccessKind::Store => {
                                        rep.mem_writes += 1;
                                        w.last_store_ack = w.last_store_ack.max(mem.write(r));
                                    }
                                    _ => {
                                        rep.mem_atomics += 1;
                                        last_done = last_done.max(mem.atomic(r));
                                    }
                                }
                            }
                            let c = match acc.kind {
                                AccessKind::Store => {
                                    (t + 2 + passes + n).saturating_sub(1).max(t + 2)
                                }
                                _ => last_done + 1,
                            };
                            (passes + n, 1, Some(c))
                        }
                    }
                };
                core.unit_free[unit] = t + 2 + occ;
                rep.busy_by_unit[unit] += occ;
                let c = complete.take().unwrap_or(t + 2 + occ + lat - 1);
                w.ready_at = c + 1;
                w.waiting = if unit == LSU && complete_is_mem(&acc) {
                    2
                } else {
                    5
                };
                w.ibuf_at = None;
                w.fetch_req = if rec.event == EV_EXIT {
                    None
                } else if is_branch(op) {
                    Some(c)
                } else {
                    Some(t + 1)
                };
                match (op, rec.event) {
                    (_, EV_EXIT) => {
                        w.exited = true;
                        w.ready_at = u64::MAX;
                        if rec.released {
                            release(&mut core.warps, c);
                        }
                    }
                    (_, EV_BAR) => {
                        w.wait_bar = true;
                        w.bar_arrival = c.max(w.last_store_ack);
                        w.waiting = 3;
                        if rec.released {
                            release(&mut core.warps, 0);
                        }
                    }
                    (Op::Fence, _) => {
                        w.ready_at = c.max(w.last_store_ack) + 1;
                        w.waiting = 4;
                    }
                    _ => {}
                }
                // block completion (§11.5)
                if core.warps.iter().all(|w| w.exited) {
                    let acks = core
                        .warps
                        .iter()
                        .map(|w| w.last_store_ack)
                        .max()
                        .unwrap_or(0);
                    let last_exit = c;
                    core.done_at = Some(last_exit.max(acks));
                }
                break;
            }
            if !issued {
                let live: Vec<&WarpT> = core.warps.iter().filter(|w| !w.exited).collect();
                let reason = if unit_blocked {
                    0
                } else if live
                    .iter()
                    .any(|w| w.ibuf_at.is_none_or(|b| b > now) && w.ready_at <= now && !w.wait_bar)
                {
                    1
                } else if live.iter().any(|w| w.ready_at > now && w.waiting == 2) {
                    2
                } else if live.iter().any(|w| w.wait_bar) {
                    3
                } else if live.iter().any(|w| w.ready_at > now && w.waiting == 4) {
                    4
                } else {
                    5
                };
                rep.stalls[reason] += 1;
            }
            for w in core.warps.iter().filter(|w| !w.exited) {
                if w.ready_at > now && w.ready_at != u64::MAX {
                    next_event = next_event.min(w.ready_at);
                }
                if let Some(b) = w.ibuf_at {
                    if b > now {
                        next_event = next_event.min(b);
                    }
                }
                if let Some(f) = w.fetch_req {
                    next_event = next_event.min(f.max(now + 1));
                }
            }
            if issued {
                next_event = now + 1;
            }
        }
        // advance; idle cycles are skipped but still counted in core_active_cycles/stalls
        let next = if next_event == u64::MAX {
            now + 1
        } else {
            next_event.max(now + 1)
        };
        if next > now + 1 {
            let skipped = next - now - 1;
            for core in cores.iter() {
                if core.block.is_some() && core.done_at.is_none() {
                    rep.core_active_cycles += skipped;
                    let live: Vec<&WarpT> = core.warps.iter().filter(|w| !w.exited).collect();
                    let reason = if live.iter().any(|w| w.ready_at > now && w.waiting == 2) {
                        2
                    } else if live.iter().any(|w| w.wait_bar) {
                        3
                    } else if live.iter().any(|w| w.ready_at > now && w.waiting == 4) {
                        4
                    } else if live.iter().any(|w| w.ibuf_at.is_none_or(|b| b > now)) {
                        1
                    } else {
                        5
                    };
                    rep.stalls[reason] += skipped;
                }
            }
        }
        now = next;
    }
    rep.cycles = end_cycle.max(1);
    rep.blocks = grid as u64;
    gpu.reported_cycles = Some(rep.cycles);
    gpu.clock_override = None;
    Ok(rep)
}

fn complete_is_mem(acc: &crate::gpu::Access) -> bool {
    !acc.shared && acc.kind != AccessKind::Store && acc.kind != AccessKind::None
}

/// [BAR-3] release: every waiting warp continues at max(arrivals, extra) + 1.
fn release(warps: &mut [WarpT], extra: u64) {
    let r = warps
        .iter()
        .filter(|w| w.wait_bar)
        .map(|w| w.bar_arrival)
        .max()
        .unwrap_or(0)
        .max(extra);
    for w in warps.iter_mut().filter(|w| w.wait_bar) {
        w.wait_bar = false;
        w.ready_at = r + 1;
        w.waiting = 5;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{Config, REG_ARG0, REG_BLOCK_DIM, REG_ENTRY, REG_GRID_DIM};

    const CODE: u32 = 0x1000;
    const S_ADDI_S1: u32 = 0x2021_0001; // S_ADDI s1, s1, 1
    const EXIT: u32 = 0x0000_0001;
    // V_GETSR v1, GTID; V_SHLI v2, v1, 2; V_ADD v2, v2, s1; V_SW v1, 0(v2); EXIT
    const STORE_GTID: [u32; 5] = [0x1c32_0001, 0x0841_100e, 0x0842_0c00, 0x4c22_0000, EXIT];

    fn setup(prog: &[u32], grid: u32, block: u32, arg0: u32) -> Gpu {
        let mut g = Gpu::new(Config {
            mem_size: 1 << 20,
            ..Config::default()
        })
        .unwrap();
        let bytes: Vec<u8> = prog.iter().flat_map(|w| w.to_le_bytes()).collect();
        g.mem_write(CODE as u64, &bytes).unwrap();
        for (r, v) in [
            (REG_ENTRY, CODE),
            (REG_GRID_DIM, grid),
            (REG_BLOCK_DIM, block),
            (REG_ARG0, arg0),
        ] {
            g.reg_write(r, v).unwrap();
        }
        g
    }

    #[test]
    fn scalar_chain_exact_cycles() {
        // cold miss: read accepted at 3, done at 111, buffer valid at 113;
        // each SALU instruction then takes 4 cycles; EXIT completes 3 cycles after issue.
        for n in [0u64, 1, 4, 10] {
            let mut prog = vec![S_ADDI_S1; n as usize];
            prog.push(EXIT);
            let mut g = setup(&prog, 1, 32, 0);
            let rep = launch_timed(&mut g, &TimingConfig::default()).unwrap();
            assert_eq!(rep.cycles, 116 + 4 * n, "n = {n} rep = {rep:?}");
            assert_eq!(rep.issued, n + 1);
            assert_eq!((rep.icache_misses, rep.mem_reads), (1, 1));
        }
    }

    #[test]
    fn timed_results_match_untimed() {
        for (cores, warps) in [(1, 8), (2, 2), (3, 4)] {
            let mut plain = setup(&STORE_GTID, 5, 64, 0x8000);
            plain.reg_write(crate::gpu::REG_GO, 1).unwrap();
            let mut timed = setup(&STORE_GTID, 5, 64, 0x8000);
            let tc = TimingConfig {
                cores,
                warps,
                ..TimingConfig::default()
            };
            let rep = launch_timed(&mut timed, &tc).unwrap();
            assert_eq!(
                plain.mem_read(0x8000, 4 * 320).unwrap(),
                timed.mem_read(0x8000, 4 * 320).unwrap()
            );
            assert_eq!(rep.issued, 5 * 2 * 5);
            assert_eq!(timed.reg_read(crate::gpu::REG_CYCLES) as u64, rep.cycles);
            assert!(!timed.running());
        }
    }

    #[test]
    fn launch_validation_takes_one_cycle() {
        let mut g = setup(&STORE_GTID, 1, 300, 0x8000); // > 32 * 8
        let rep = launch_timed(&mut g, &TimingConfig::default()).unwrap();
        assert_eq!((rep.cycles, g.reg_read(crate::gpu::REG_STATUS)), (1, 5));
    }

    #[test]
    fn coalescing_counts_segments() {
        let mut g = setup(&STORE_GTID, 1, 32, 0x8000);
        let rep = launch_timed(&mut g, &TimingConfig::default()).unwrap();
        assert_eq!(rep.mem_writes, 1); // 32 consecutive words = one 128-byte segment
    }
}
