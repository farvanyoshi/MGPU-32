//! The MGPU-32 engine: host registers, global memory, launch, scheduling and instruction
//! semantics (ISA sections 6, 10-13). Mirrors python/mgpu/model exactly, including the
//! reference schedule, so both produce identical traces.

use std::collections::{BTreeMap, HashSet};

use crate::decode::{decode, Decoded};
use crate::fp::{self, FCond};
use crate::isa_gen::{Op, SR_LANE, SR_UNIFORM};
use crate::sched::{pick, Xorshift64Star};
use crate::sfu;
use crate::trace::{Record, EV_BAR, EV_EXIT, EV_FAULT, GLOBAL, SHARED};

pub const WARP: usize = 32;
const SHARED_WORDS: usize = 4096;
const SHARED_SIZE: u64 = 16384;

// Host register offsets (ISA 13.1)
pub const REG_ID: u32 = 0x00;
pub const REG_VERSION: u32 = 0x04;
pub const REG_MAX_BLOCK_DIM: u32 = 0x08;
pub const REG_MEM_SIZE: u32 = 0x0C;
pub const REG_ENTRY: u32 = 0x10;
pub const REG_GRID_DIM: u32 = 0x14;
pub const REG_BLOCK_DIM: u32 = 0x18;
pub const REG_GO: u32 = 0x1C;
pub const REG_STATUS: u32 = 0x20;
pub const REG_ERR_INFO: u32 = 0x34;
pub const REG_CYCLES: u32 = 0x38;
pub const REG_ARG0: u32 = 0x40;

pub const E_ILLEGAL: u32 = 1;
pub const E_MISALIGNED: u32 = 2;
pub const E_RANGE: u32 = 3;
pub const E_BARRIER: u32 = 4;
pub const E_LAUNCH: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarpState {
    Run = 0,
    BarWait = 1,
    Exited = 2,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub mem_size: u64,
    pub max_block_dim: u32,
    pub resident_blocks: usize,
    pub seed: Option<u64>,
    pub shared_poison: u32,
    pub max_steps: u64,
    pub trace: bool,
    pub keep_state: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            mem_size: 64 << 20,
            max_block_dim: 1024,
            resident_blocks: 1,
            seed: None,
            shared_poison: 0xDEAD_BEEF,
            max_steps: 50_000_000,
            trace: false,
            keep_state: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SimError {
    /// The kernel exceeded `max_steps` ([FLT-3]); not an architectural fault.
    Timeout(u64),
    /// A driven-mode step named a warp that cannot run, or no kernel is running.
    NotRunnable(String),
    /// A host memory access outside global memory.
    HostRange(u64, u64),
    InvalidConfig(String),
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimError::Timeout(n) => write!(f, "kernel exceeded {n} steps"),
            SimError::NotRunnable(s) => write!(f, "{s}"),
            SimError::HostRange(a, n) => write!(f, "host access [{a:#x}, +{n}) out of range"),
            SimError::InvalidConfig(s) => write!(f, "invalid configuration: {s}"),
        }
    }
}

impl std::error::Error for SimError {}

/// What memory an instruction touched, for the timing model: the address of every
/// active lane (or the single scalar address in lane 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AccessKind {
    #[default]
    None,
    Load,
    Store,
    Atomic,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Access {
    pub kind: AccessKind,
    pub shared: bool,
    /// Lanes that accessed memory (bit i = lane i).
    pub lanes: u32,
    pub addrs: [u32; WARP],
}

impl Access {
    fn vector(kind: AccessKind, shared: bool, lanes: u32, addrs: [u32; WARP]) -> Self {
        Access {
            kind,
            shared,
            lanes,
            addrs,
        }
    }
}

/// An architectural fault (ISA section 11), raised before any state is modified.
#[derive(Clone, Copy, Debug)]
struct Fault {
    code: u32,
    lane: u32,
    info: u32,
}

#[inline]
fn fault(code: u32, lane: u32, info: u32) -> Fault {
    Fault { code, lane, info }
}

pub struct Warp {
    pub pc: u32,
    pub s: [u32; 32],
    /// v[reg][lane]
    pub v: Box<[[u32; WARP]; 32]>,
    pub state: WarpState,
}

struct Block {
    bid: u32,
    warps: Vec<Warp>,
    shared: Vec<u32>,
    /// P of [BAR-1]; None while no warp waits.
    bar_pc: Option<u32>,
}

impl Block {
    fn finished(&self) -> bool {
        self.warps.iter().all(|w| w.state == WarpState::Exited)
    }
}

/// [BAR-3]: release when every non-exited warp waits.
fn try_release(warps: &mut [Warp], bar_pc: &mut Option<u32>, rec: &mut Record) {
    let Some(p) = *bar_pc else { return };
    let mut any_live = false;
    for w in warps.iter() {
        match w.state {
            WarpState::Run => return,
            WarpState::BarWait => any_live = true,
            WarpState::Exited => {}
        }
    }
    if !any_live {
        return;
    }
    for w in warps.iter_mut().filter(|w| w.state == WarpState::BarWait) {
        w.state = WarpState::Run;
        w.pc = p.wrapping_add(4);
    }
    *bar_pc = None;
    rec.released = true;
}

/// Launch parameters visible to executing instructions.
#[derive(Clone, Copy, Debug, Default)]
struct Ctx {
    mem_size: u64,
    entry: u32,
    grid: u32,
    block_dim: u32,
    num_warps: u32,
    args: [u32; 8],
}

#[inline]
fn load(mem: &[u8], a: u32) -> u32 {
    let a = a as usize;
    u32::from_le_bytes([mem[a], mem[a + 1], mem[a + 2], mem[a + 3]])
}

#[inline]
fn store(mem: &mut [u8], a: u32, x: u32) {
    let a = a as usize;
    mem[a..a + 4].copy_from_slice(&x.to_le_bytes());
}

/// CHECK of ISA 10.1: 0 if valid, else the error code.
#[inline]
fn check(addr: u32, size: u64) -> u32 {
    if addr & 3 != 0 {
        E_MISALIGNED
    } else if addr as u64 + 4 > size {
        E_RANGE
    } else {
        0
    }
}

#[inline]
fn active(exec: u32, i: usize) -> bool {
    exec >> i & 1 != 0
}

// ---- integer funct table (ISA 5.1) --------------------------------------------------------
fn int_op(op: Op) -> fn(u32, u32) -> u32 {
    use Op::*;
    match op {
        SAdd | VAdd | SAddi | VAddi | VAtomAdd | VAtomAddS => u32::wrapping_add,
        SSub | VSub => u32::wrapping_sub,
        SMul | VMul => u32::wrapping_mul,
        SAnd | VAnd | SAndi | VAndi | VAtomAnd | VAtomAndS => |a, b| a & b,
        SOr | VOr | SOri | VAtomOr | VAtomOrS => |a, b| a | b,
        SXor | VXor => |a, b| a ^ b,
        SAndn | VAndn => |a, b| a & !b,
        SShl | VShl | SShli | VShli => |a, b| a << (b & 31),
        SShr | VShr | SShri | VShri => |a, b| a >> (b & 31),
        SSra | VSra | SSrai | VSrai => |a, b| ((a as i32) >> (b & 31)) as u32,
        SSlt | VSlt => |a, b| ((a as i32) < (b as i32)) as u32,
        SSltu | VSltu => |a, b| (a < b) as u32,
        SMin | VMin | VAtomMin | VAtomMinS => |a, b| if (a as i32) <= (b as i32) { a } else { b },
        SMax | VMax | VAtomMax | VAtomMaxS => |a, b| if (a as i32) >= (b as i32) { a } else { b },
        VAtomXchg | VAtomXchgS => |_, b| b,
        _ => unreachable!("not an integer op: {op:?}"),
    }
}

/// V_CMP conditions (ISA 5.3).
fn cmp(op: Op, a: u32, b: u32) -> bool {
    use Op::*;
    let (sa, sb) = (a as i32, b as i32);
    match op {
        VCmpEq => a == b,
        VCmpNe => a != b,
        VCmpLt => sa < sb,
        VCmpLe => sa <= sb,
        VCmpGt => sa > sb,
        VCmpGe => sa >= sb,
        VCmpLtu => a < b,
        VCmpLeu => a <= b,
        VCmpGtu => a > b,
        VCmpGeu => a >= b,
        VCmpFeq => fp::fcmp(FCond::Eq, a, b),
        VCmpFne => fp::fcmp(FCond::Ne, a, b),
        VCmpFlt => fp::fcmp(FCond::Lt, a, b),
        VCmpFle => fp::fcmp(FCond::Le, a, b),
        VCmpFgt => fp::fcmp(FCond::Gt, a, b),
        VCmpFge => fp::fcmp(FCond::Ge, a, b),
        _ => unreachable!(),
    }
}

/// Snapshot of the vb|sb operand for all lanes ([ENC-1], [SEM-1]).
#[inline]
fn b_operand(w: &Warp, d: &Decoded) -> [u32; WARP] {
    if d.sb {
        [w.s[d.rb]; WARP]
    } else {
        w.v[d.rb]
    }
}

#[inline]
fn write_s(w: &mut Warp, rec: &mut Record, r: usize, x: u32) {
    if r != 0 {
        w.s[r] = x;
        rec.swrite = Some((r as u8, x));
    }
}

/// Compute f for every active lane (reading the pre-instruction state), then write rd.
#[inline]
fn vec_op(w: &mut Warp, rec: &mut Record, rd: usize, exec: u32, f: impl Fn(&Warp, usize) -> u32) {
    let mut out = w.v[rd];
    for (i, o) in out.iter_mut().enumerate() {
        if active(exec, i) {
            *o = f(w, i);
        }
    }
    if rd != 0 {
        w.v[rd] = out;
        rec.vwrite = Some((rd as u8, out));
    }
}

/// Per-lane addresses for a vector memory access, checked in ascending lane order.
fn lane_addresses(
    w: &Warp,
    exec: u32,
    base: usize,
    off: i32,
    size: u64,
) -> Result<[u32; WARP], Fault> {
    let mut a = [0u32; WARP];
    for (i, (ai, &b)) in a.iter_mut().zip(w.v[base].iter()).enumerate() {
        if active(exec, i) {
            *ai = b.wrapping_add(off as u32);
            let code = check(*ai, size);
            if code != 0 {
                return Err(fault(code, i as u32, *ai));
            }
        }
    }
    Ok(a)
}

fn sr_value(id: u32, ctx: &Ctx, bid: u32, wid: u32, lane: u32, clock: u64) -> u32 {
    use crate::isa_gen::*;
    let tid = WARP as u32 * wid + lane;
    match id {
        SR_BLOCK_ID => bid,
        SR_BLOCK_DIM => ctx.block_dim,
        SR_GRID_DIM => ctx.grid,
        SR_WARP_ID => wid,
        SR_NUM_WARPS => ctx.num_warps,
        SR_CLOCK => clock as u32,
        SR_TID => tid,
        SR_LANE_ID => lane,
        SR_GTID => bid.wrapping_mul(ctx.block_dim).wrapping_add(tid),
        _ => unreachable!(),
    }
}

fn is_shared_atom(op: Op) -> bool {
    use Op::*;
    matches!(
        op,
        VAtomAddS | VAtomMinS | VAtomMaxS | VAtomXchgS | VAtomAndS | VAtomOrS
    )
}

/// Execute one instruction of warp `wid` of `blk`. On Err nothing has been modified.
fn execute(
    blk: &mut Block,
    wid: usize,
    mem: &mut [u8],
    ctx: &Ctx,
    step: u64,
    clock: u64,
    acc: &mut Access,
) -> Result<Record, Fault> {
    use Op::*;
    let Block {
        bid,
        warps,
        shared,
        bar_pc,
    } = blk;
    let bid = *bid;
    let w = &mut warps[wid];
    let pc = w.pc;
    if pc as u64 + 4 > ctx.mem_size {
        return Err(fault(E_RANGE, 0, pc)); // [MEM-7]
    }
    let word = load(mem, pc);
    let d = decode(word).ok_or(fault(E_ILLEGAL, 0, word))?;
    let mut rec = Record::new(step, bid, wid as u32, pc, word);
    let exec = w.s[31]; // [EX-3]
    let mut next_pc = pc.wrapping_add(4);

    match d.op {
        Nop | Fence => {}
        Exit => {
            w.state = WarpState::Exited;
            rec.event = EV_EXIT;
            rec.exec_after = w.s[31];
            try_release(warps, bar_pc, &mut rec);
            return Ok(rec);
        }
        Bar => {
            if let Some(p) = *bar_pc {
                if p != pc {
                    return Err(fault(E_BARRIER, 0, p)); // [BAR-2]
                }
            }
            w.state = WarpState::BarWait;
            *bar_pc = Some(pc);
            rec.event = EV_BAR;
            rec.exec_after = w.s[31];
            try_release(warps, bar_pc, &mut rec);
            return Ok(rec);
        }

        SAdd | SSub | SMul | SAnd | SOr | SXor | SAndn | SShl | SShr | SSra | SSlt | SSltu
        | SMin | SMax => {
            let r = int_op(d.op)(w.s[d.ra], w.s[d.rb]);
            write_s(w, &mut rec, d.rd, r);
        }
        SShli | SShri | SSrai => {
            let r = int_op(d.op)(w.s[d.ra], d.rb as u32);
            write_s(w, &mut rec, d.rd, r);
        }
        SPopc => {
            let r = w.s[d.ra].count_ones();
            write_s(w, &mut rec, d.rd, r);
        }
        SFf1 => {
            let a = w.s[d.ra];
            write_s(
                w,
                &mut rec,
                d.rd,
                if a == 0 { u32::MAX } else { a.trailing_zeros() },
            );
        }

        VAdd | VSub | VMul | VAnd | VOr | VXor | VAndn | VShl | VShr | VSra | VSlt | VSltu
        | VMin | VMax => {
            let (f, b, ra) = (int_op(d.op), b_operand(w, &d), d.ra);
            vec_op(w, &mut rec, d.rd, exec, |w, i| f(w.v[ra][i], b[i]));
        }
        VShli | VShri | VSrai => {
            let (f, ra, sh) = (int_op(d.op), d.ra, d.rb as u32);
            vec_op(w, &mut rec, d.rd, exec, |w, i| f(w.v[ra][i], sh));
        }

        VFadd | VFsub | VFmul | VFmin | VFmax => {
            let f: fn(u32, u32) -> u32 = match d.op {
                VFadd => fp::fadd,
                VFsub => fp::fsub,
                VFmul => fp::fmul,
                VFmin => fp::fmin,
                _ => fp::fmax,
            };
            let (b, ra) = (b_operand(w, &d), d.ra);
            vec_op(w, &mut rec, d.rd, exec, |w, i| f(w.v[ra][i], b[i]));
        }
        VFrcp | VFrsq | VFex2 | VFlg2 | VI2f | VF2i => {
            let f: fn(u32) -> u32 = match d.op {
                VFrcp => sfu::frcp,
                VFrsq => sfu::frsq,
                VFex2 => sfu::fex2,
                VFlg2 => sfu::flg2,
                VI2f => fp::i2f,
                _ => fp::f2i,
            };
            let ra = d.ra;
            vec_op(w, &mut rec, d.rd, exec, |w, i| f(w.v[ra][i]));
        }

        VCmpEq | VCmpNe | VCmpLt | VCmpLe | VCmpGt | VCmpGe | VCmpLtu | VCmpLeu | VCmpFeq
        | VCmpFne | VCmpFlt | VCmpFle | VCmpFgt | VCmpFge | VCmpGtu | VCmpGeu => {
            let b = b_operand(w, &d);
            let mut m = 0u32;
            for (i, &bi) in b.iter().enumerate() {
                if active(exec, i) && cmp(d.op, w.v[d.ra][i], bi) {
                    m |= 1 << i;
                }
            }
            write_s(w, &mut rec, d.rd, m);
        }

        VFfma => {
            let (b, ra, rc) = (b_operand(w, &d), d.ra, d.rc);
            vec_op(w, &mut rec, d.rd, exec, |w, i| {
                fp::ffma(w.v[ra][i], b[i], w.v[rc][i])
            });
        }
        VImad => {
            let (b, ra, rc) = (b_operand(w, &d), d.ra, d.rc);
            vec_op(w, &mut rec, d.rd, exec, |w, i| {
                w.v[ra][i].wrapping_mul(b[i]).wrapping_add(w.v[rc][i])
            });
        }
        VSel => {
            let (b, ra, mask) = (b_operand(w, &d), d.ra, w.s[d.rc]);
            vec_op(w, &mut rec, d.rd, exec, |w, i| {
                if active(mask, i) {
                    w.v[ra][i]
                } else {
                    b[i]
                }
            });
        }

        VReadlane => {
            let x = w.v[d.ra][(w.s[d.rb] & 31) as usize];
            write_s(w, &mut rec, d.rd, x);
        }
        VShfl => {
            let (b, src) = (b_operand(w, &d), w.v[d.ra]);
            vec_op(w, &mut rec, d.rd, exec, |_, i| src[(b[i] & 31) as usize]);
        }

        SGetsr | VGetsr => {
            let id = d.ra as u32;
            let legal = if d.op == SGetsr {
                SR_UNIFORM
            } else {
                SR_UNIFORM | SR_LANE
            };
            if legal >> id & 1 == 0 {
                return Err(fault(E_ILLEGAL, 0, word)); // [SR-1], [SR-2]
            }
            if d.op == SGetsr {
                write_s(
                    w,
                    &mut rec,
                    d.rd,
                    sr_value(id, ctx, bid, wid as u32, 0, clock),
                );
            } else {
                vec_op(w, &mut rec, d.rd, exec, |_, i| {
                    sr_value(id, ctx, bid, wid as u32, i as u32, clock)
                });
            }
        }

        SAddi => write_s(w, &mut rec, d.rd, w.s[d.ra].wrapping_add(d.simm16 as u32)),
        SAndi => write_s(w, &mut rec, d.rd, w.s[d.ra] & d.imm16),
        SOri => write_s(w, &mut rec, d.rd, w.s[d.ra] | d.imm16),
        SLui => write_s(w, &mut rec, d.rd, d.imm16 << 16),
        VAddi => {
            let (ra, imm) = (d.ra, d.simm16 as u32);
            vec_op(w, &mut rec, d.rd, exec, |w, i| w.v[ra][i].wrapping_add(imm));
        }
        VAndi => {
            let (ra, imm) = (d.ra, d.imm16);
            vec_op(w, &mut rec, d.rd, exec, |w, i| w.v[ra][i] & imm);
        }

        SLw | SSw => {
            let addr = w.s[d.ra].wrapping_add(d.simm16 as u32);
            let code = check(addr, ctx.mem_size);
            if code != 0 {
                return Err(fault(code, 0, addr));
            }
            let mut addrs = [0u32; WARP];
            addrs[0] = addr;
            let kind = if d.op == SLw {
                AccessKind::Load
            } else {
                AccessKind::Store
            };
            *acc = Access::vector(kind, false, 1, addrs);
            if d.op == SLw {
                write_s(w, &mut rec, d.rd, load(mem, addr));
            } else {
                store(mem, addr, w.s[d.rd]);
                rec.mwrites.push((GLOBAL, addr, w.s[d.rd]));
            }
        }
        VLw | VLds => {
            let sh = d.op == VLds;
            let a = lane_addresses(
                w,
                exec,
                d.ra,
                d.simm16,
                if sh { SHARED_SIZE } else { ctx.mem_size },
            )?;
            *acc = Access::vector(AccessKind::Load, sh, exec, a);
            let m: &[u8] = mem;
            vec_op(w, &mut rec, d.rd, exec, |_, i| {
                if sh {
                    shared[(a[i] >> 2) as usize]
                } else {
                    load(m, a[i])
                }
            });
        }
        VSw | VSts => {
            let sh = d.op == VSts;
            let a = lane_addresses(
                w,
                exec,
                d.ra,
                d.simm16,
                if sh { SHARED_SIZE } else { ctx.mem_size },
            )?;
            *acc = Access::vector(AccessKind::Store, sh, exec, a);
            let data = w.v[d.rd];
            for i in 0..WARP {
                if active(exec, i) {
                    // ascending lane order: the highest lane wins a collision [MEM-4]
                    if sh {
                        shared[(a[i] >> 2) as usize] = data[i];
                    } else {
                        store(mem, a[i], data[i]);
                    }
                    rec.mwrites
                        .push((if sh { SHARED } else { GLOBAL }, a[i], data[i]));
                }
            }
        }
        VAtomAdd | VAtomMin | VAtomMax | VAtomXchg | VAtomAnd | VAtomOr | VAtomAddS | VAtomMinS
        | VAtomMaxS | VAtomXchgS | VAtomAndS | VAtomOrS => {
            let sh = is_shared_atom(d.op);
            let f = int_op(d.op);
            let b = b_operand(w, &d);
            let a = lane_addresses(
                w,
                exec,
                d.ra,
                0,
                if sh { SHARED_SIZE } else { ctx.mem_size },
            )?;
            *acc = Access::vector(AccessKind::Atomic, sh, exec, a);
            let mut old = [0u32; WARP];
            for i in 0..WARP {
                if active(exec, i) {
                    // ascending lane order (ISA 6.9)
                    old[i] = if sh {
                        shared[(a[i] >> 2) as usize]
                    } else {
                        load(mem, a[i])
                    };
                    let new = f(old[i], b[i]);
                    if sh {
                        shared[(a[i] >> 2) as usize] = new;
                    } else {
                        store(mem, a[i], new);
                    }
                    rec.mwrites
                        .push((if sh { SHARED } else { GLOBAL }, a[i], new));
                }
            }
            vec_op(w, &mut rec, d.rd, exec, |_, i| old[i]);
        }

        SBz | SBnz => {
            if (w.s[d.rd] == 0) == (d.op == SBz) {
                next_pc = pc.wrapping_add(4).wrapping_add((d.simm16 << 2) as u32);
            }
        }
        SJ => next_pc = pc.wrapping_add(4).wrapping_add((d.soffset26 << 2) as u32),
    }

    w.pc = next_pc;
    rec.exec_after = w.s[31];
    Ok(rec)
}

/// The fast MGPU-32 simulator. See python/mgpu/model/gpu.py for the reference semantics.
pub struct Gpu {
    pub cfg: Config,
    mem: Vec<u8>,
    entry: u32,
    grid_dim: u32,
    block_dim: u32,
    args: [u32; 8],
    /// STATUS, ERR_PC, ERR_BLOCK, ERR_WARP, ERR_LANE, ERR_INFO
    status: [u32; 6],
    running: bool,
    pub steps: u64,
    pub trace: Vec<Record>,
    ctx: Ctx,
    blocks: BTreeMap<u32, Block>,
    resident: Vec<u32>,
    done: HashSet<u32>,
    next_block: u64,
    last: Option<(u32, u32)>,
    prng: Option<Xorshift64Star>,
    eligible: Vec<(u32, u32)>,
    /// Memory touched by the most recent step (for the timing model).
    pub last_access: Access,
    /// When set, CLOCK reads this value instead of the step number (timing mode).
    pub clock_override: Option<u64>,
}

impl Gpu {
    pub fn new(cfg: Config) -> Result<Gpu, SimError> {
        if !cfg.mem_size.is_multiple_of(4) || cfg.mem_size < 4 || cfg.mem_size > 1 << 32 {
            return Err(SimError::InvalidConfig(format!(
                "mem_size {}",
                cfg.mem_size
            )));
        }
        if !cfg.max_block_dim.is_multiple_of(32) || !(32..=1024).contains(&cfg.max_block_dim) {
            return Err(SimError::InvalidConfig(format!(
                "max_block_dim {}",
                cfg.max_block_dim
            )));
        }
        if cfg.resident_blocks == 0 {
            return Err(SimError::InvalidConfig(
                "resident_blocks must be >= 1".into(),
            ));
        }
        Ok(Gpu {
            mem: vec![0; cfg.mem_size as usize],
            cfg,
            entry: 0,
            grid_dim: 0,
            block_dim: 0,
            args: [0; 8],
            status: [0; 6],
            running: false,
            steps: 0,
            trace: Vec::new(),
            ctx: Ctx::default(),
            blocks: BTreeMap::new(),
            resident: Vec::new(),
            done: HashSet::new(),
            next_block: 0,
            last: None,
            prng: None,
            eligible: Vec::new(),
            last_access: Access::default(),
            clock_override: None,
        })
    }

    // ---- host memory ---------------------------------------------------------------------
    fn host_range(&self, addr: u64, n: u64) -> Result<std::ops::Range<usize>, SimError> {
        if addr + n > self.cfg.mem_size {
            return Err(SimError::HostRange(addr, n));
        }
        Ok(addr as usize..(addr + n) as usize)
    }

    pub fn mem_write(&mut self, addr: u64, data: &[u8]) -> Result<(), SimError> {
        let r = self.host_range(addr, data.len() as u64)?;
        self.mem[r].copy_from_slice(data);
        Ok(())
    }

    pub fn mem_read(&self, addr: u64, n: u64) -> Result<&[u8], SimError> {
        let r = self.host_range(addr, n)?;
        Ok(&self.mem[r])
    }

    // ---- host registers (ISA 13.1) ----------------------------------------------------------
    pub fn running(&self) -> bool {
        self.running
    }

    pub fn reg_read(&self, off: u32) -> u32 {
        match off {
            REG_ID => 0x4D47_5055,
            REG_VERSION => 0x0001_0000,
            REG_MAX_BLOCK_DIM => self.cfg.max_block_dim,
            REG_MEM_SIZE => self.cfg.mem_size as u32,
            REG_ENTRY => self.entry,
            REG_GRID_DIM => self.grid_dim,
            REG_BLOCK_DIM => self.block_dim,
            REG_GO => self.running as u32,
            REG_STATUS..=REG_ERR_INFO if off.is_multiple_of(4) => {
                self.status[((off - REG_STATUS) / 4) as usize]
            }
            REG_CYCLES => self.steps as u32,
            0x40..=0x5C if off.is_multiple_of(4) => self.args[((off - REG_ARG0) / 4) as usize],
            _ => 0,
        }
    }

    /// Writing GO runs the whole kernel before returning.
    pub fn reg_write(&mut self, off: u32, value: u32) -> Result<(), SimError> {
        if self.running {
            return Ok(()); // [HOST-1], [HOST-2]
        }
        match off {
            REG_ENTRY => self.entry = value,
            REG_GRID_DIM => self.grid_dim = value,
            REG_BLOCK_DIM => self.block_dim = value,
            0x40..=0x5C if off.is_multiple_of(4) => {
                self.args[((off - REG_ARG0) / 4) as usize] = value
            }
            REG_GO if value & 1 != 0 => self.launch()?,
            _ => {}
        }
        Ok(())
    }

    // ---- launch -----------------------------------------------------------------------------
    fn set_status(&mut self, code: u32, pc: u32, block: u32, warp: u32, lane: u32, info: u32) {
        self.status = [code, pc, block, warp, lane, info];
    }

    /// What writing 1 to GO does: validate the launch, then run it to completion.
    pub fn launch(&mut self) -> Result<(), SimError> {
        if self.begin() {
            self.run()?;
        }
        Ok(())
    }

    /// Validate and start a launch ([HOST-3], [HOST-4]). Returns true if warps will run.
    pub fn begin(&mut self) -> bool {
        self.status = [0; 6];
        self.steps = 0;
        self.trace.clear();
        let (entry, grid, bdim) = (self.entry, self.grid_dim, self.block_dim);
        if bdim == 0 || bdim > self.cfg.max_block_dim {
            self.set_status(E_LAUNCH, 0, 0, 0, 0, 0);
            return false;
        }
        if entry & 3 != 0 {
            self.set_status(E_MISALIGNED, entry, 0, 0, 0, entry);
            return false;
        }
        if grid == 0 {
            return false;
        }
        self.ctx = Ctx {
            mem_size: self.cfg.mem_size,
            entry,
            grid,
            block_dim: bdim,
            num_warps: bdim.div_ceil(WARP as u32),
            args: self.args,
        };
        self.blocks.clear();
        self.resident.clear();
        self.done.clear();
        self.next_block = 0;
        self.last = None;
        self.prng = self.cfg.seed.map(Xorshift64Star::new);
        self.running = true;
        true
    }

    fn new_block(&mut self, bid: u32) {
        let ctx = &self.ctx;
        let warps = (0..ctx.num_warps)
            .map(|wid| {
                let mut s = [0u32; 32];
                s[1..9].copy_from_slice(&ctx.args);
                let live = ctx
                    .block_dim
                    .saturating_sub(WARP as u32 * wid)
                    .min(WARP as u32);
                s[31] = if live == 32 {
                    u32::MAX
                } else {
                    (1u32 << live) - 1
                };
                Warp {
                    pc: ctx.entry,
                    s,
                    v: Box::new([[0; WARP]; 32]),
                    state: WarpState::Run,
                }
            })
            .collect();
        let b = Block {
            bid,
            warps,
            shared: vec![self.cfg.shared_poison; SHARED_WORDS],
            bar_pc: None,
        };
        self.blocks.insert(bid, b);
    }

    fn fill(&mut self) {
        while self.resident.len() < self.cfg.resident_blocks
            && self.next_block < self.ctx.grid as u64
        {
            let bid = self.next_block as u32;
            self.new_block(bid);
            self.resident.push(bid);
            self.next_block += 1;
        }
    }

    /// Execute one instruction of (bid, wid); on a fault, abort the kernel ([FLT-1]).
    fn retire(&mut self, bid: u32, wid: usize) -> Record {
        let step = self.steps;
        let ctx = self.ctx;
        let clock = self.clock_override.unwrap_or(step);
        let blk = self.blocks.get_mut(&bid).expect("block is resident");
        self.last_access = Access::default();
        let rec = match execute(
            blk,
            wid,
            &mut self.mem,
            &ctx,
            step,
            clock,
            &mut self.last_access,
        ) {
            Ok(r) => r,
            Err(f) => {
                let w = &blk.warps[wid];
                let word = if w.pc as u64 + 4 <= ctx.mem_size {
                    load(&self.mem, w.pc)
                } else {
                    0
                };
                let mut r = Record::new(step, bid, wid as u32, w.pc, word);
                r.exec_after = w.s[31];
                r.event = EV_FAULT;
                r.fault = Some((f.code, f.lane, f.info));
                let pc = w.pc;
                self.set_status(f.code, pc, bid, wid as u32, f.lane, f.info);
                self.running = false;
                r
            }
        };
        self.steps += 1;
        if self.running && self.blocks[&bid].finished() {
            self.done.insert(bid);
            self.resident.retain(|&b| b != bid);
            if !self.cfg.keep_state {
                self.blocks.remove(&bid);
            }
            if self.done.len() as u64 == ctx.grid as u64 {
                self.running = false;
            }
        }
        rec
    }

    /// Run the launch to completion under the reference schedule.
    pub fn run(&mut self) -> Result<(), SimError> {
        self.fill();
        let mut elig = std::mem::take(&mut self.eligible);
        while self.running {
            if self.steps >= self.cfg.max_steps {
                self.eligible = elig;
                return Err(SimError::Timeout(self.cfg.max_steps));
            }
            elig.clear();
            for &bid in &self.resident {
                let b = &self.blocks[&bid];
                for (wid, w) in b.warps.iter().enumerate() {
                    if w.state == WarpState::Run {
                        elig.push((bid, wid as u32));
                    }
                }
            }
            assert!(
                !elig.is_empty(),
                "no eligible warp: BAR-3 guarantees one exists"
            );
            let k = pick(&elig, self.last, self.prng.as_mut());
            self.last = Some(k);
            let rec = self.retire(k.0, k.1 as usize);
            if self.cfg.trace {
                self.trace.push(rec);
            }
            self.fill();
        }
        self.eligible = elig;
        Ok(())
    }

    // ---- driven mode --------------------------------------------------------------------------
    /// (block, warp) pairs that may be stepped; blocks become resident on first use.
    pub fn runnable(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        if !self.running {
            return out;
        }
        for bid in 0..self.ctx.grid {
            if self.done.contains(&bid) {
                continue;
            }
            match self.blocks.get(&bid) {
                None => out.extend((0..self.ctx.num_warps).map(|w| (bid, w))),
                Some(b) => out.extend(
                    b.warps
                        .iter()
                        .enumerate()
                        .filter(|(_, w)| w.state == WarpState::Run)
                        .map(|(i, _)| (bid, i as u32)),
                ),
            }
        }
        out
    }

    /// Retire one instruction of (block, warp). Used by lockstep co-simulation.
    pub fn step(&mut self, block: u32, warp: u32) -> Result<Record, SimError> {
        if !self.running
            || block >= self.ctx.grid
            || self.done.contains(&block)
            || warp >= self.ctx.num_warps
        {
            return Err(SimError::NotRunnable(format!(
                "warp ({block}, {warp}) is not runnable"
            )));
        }
        if !self.blocks.contains_key(&block) {
            self.new_block(block);
        }
        if self.blocks[&block].warps[warp as usize].state != WarpState::Run {
            return Err(SimError::NotRunnable(format!(
                "warp ({block}, {warp}) is not runnable"
            )));
        }
        let rec = self.retire(block, warp as usize);
        if self.cfg.trace {
            self.trace.push(rec.clone());
        }
        Ok(rec)
    }

    /// (GRID_DIM, BLOCK_DIM, NUM_WARPS) of the current or last launch.
    pub fn launch_shape(&self) -> (u32, u32, u32) {
        (self.ctx.grid, self.ctx.block_dim, self.ctx.num_warps)
    }

    /// Architectural state of a warp (needs keep_state once its block has finished).
    pub fn warp(&self, block: u32, warp: u32) -> Option<&Warp> {
        self.blocks
            .get(&block)
            .and_then(|b| b.warps.get(warp as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CODE: u32 = 0x1000;
    // V_GETSR v1, GTID; V_SHLI v2, v1, 2; V_ADD v2, v2, s1; V_SW v1, 0(v2); EXIT
    const STORE_GTID: [u32; 5] = [
        0x1c32_0001,
        0x0841_100e,
        0x0842_0c00,
        0x4c22_0000,
        0x0000_0001,
    ];
    // S_GETSR s1, WARP_ID; S_BZ s1, +2; BAR; EXIT; BAR; EXIT
    const BAR_MISMATCH: [u32; 6] = [
        0x1c23_0000,
        0x8020_0002,
        0x0000_0002,
        0x0000_0001,
        0x0000_0002,
        0x0000_0001,
    ];

    fn gpu(cfg: Config, prog: &[u32]) -> Gpu {
        let mut g = Gpu::new(Config {
            mem_size: 1 << 20,
            ..cfg
        })
        .unwrap();
        let bytes: Vec<u8> = prog.iter().flat_map(|w| w.to_le_bytes()).collect();
        g.mem_write(CODE as u64, &bytes).unwrap();
        g.reg_write(REG_ENTRY, CODE).unwrap();
        g
    }

    fn launch(g: &mut Gpu, grid: u32, block: u32, arg0: u32) {
        g.reg_write(REG_GRID_DIM, grid).unwrap();
        g.reg_write(REG_BLOCK_DIM, block).unwrap();
        g.reg_write(REG_ARG0, arg0).unwrap();
        g.reg_write(REG_GO, 1).unwrap();
    }

    fn word(g: &Gpu, a: u64) -> u32 {
        u32::from_le_bytes(g.mem_read(a, 4).unwrap().try_into().unwrap())
    }

    #[test]
    fn multi_block_kernel_under_every_schedule() {
        for (resident, seed) in [(1, None), (3, None), (2, Some(7))] {
            let cfg = Config {
                resident_blocks: resident,
                seed,
                ..Config::default()
            };
            let mut g = gpu(cfg, &STORE_GTID);
            launch(&mut g, 3, 70, 0x8000);
            assert_eq!(g.reg_read(REG_STATUS), 0);
            assert_eq!(g.reg_read(REG_GO), 0);
            assert_eq!(g.steps, 3 * 3 * 5);
            for i in 0..210 {
                assert_eq!(word(&g, 0x8000 + 4 * i as u64), i);
            }
            assert_eq!(word(&g, 0x8000 + 4 * 210), 0); // lanes beyond BLOCK_DIM never store
        }
    }

    #[test]
    fn launch_validation() {
        let mut g = gpu(
            Config {
                max_block_dim: 256,
                ..Config::default()
            },
            &STORE_GTID,
        );
        launch(&mut g, 1, 257, 0x8000);
        assert_eq!(g.reg_read(REG_STATUS), E_LAUNCH);
        g.reg_write(REG_ENTRY, CODE + 2).unwrap();
        launch(&mut g, 1, 32, 0x8000);
        assert_eq!(
            (g.reg_read(REG_STATUS), g.reg_read(REG_ERR_INFO)),
            (E_MISALIGNED, CODE + 2)
        );
        assert_eq!(g.steps, 0);
    }

    #[test]
    fn barrier_mismatch_is_reported() {
        let mut g = gpu(Config::default(), &BAR_MISMATCH);
        launch(&mut g, 1, 64, 0);
        assert_eq!(g.reg_read(REG_STATUS), E_BARRIER);
        let (pc, info) = (g.reg_read(0x24), g.reg_read(REG_ERR_INFO));
        assert!(
            pc != info
                && [CODE + 8, CODE + 16].contains(&pc)
                && [CODE + 8, CODE + 16].contains(&info)
        );
    }

    #[test]
    fn misaligned_store_faults_precisely() {
        let mut g = gpu(
            Config {
                trace: true,
                ..Config::default()
            },
            &STORE_GTID,
        );
        launch(&mut g, 1, 32, 0x8002);
        assert_eq!(g.reg_read(REG_STATUS), E_MISALIGNED);
        let last = g.trace.last().unwrap();
        assert_eq!(last.fault, Some((E_MISALIGNED, 0, 0x8002)));
        assert!(last.mwrites.is_empty());
        assert_eq!(word(&g, 0x8004), 0);
    }

    #[test]
    fn driven_mode_matches_free_run() {
        let mut free = gpu(
            Config {
                trace: true,
                ..Config::default()
            },
            &STORE_GTID,
        );
        launch(&mut free, 2, 64, 0x8000);
        let mut driven = gpu(Config::default(), &STORE_GTID);
        driven.reg_write(REG_GRID_DIM, 2).unwrap();
        driven.reg_write(REG_BLOCK_DIM, 64).unwrap();
        driven.reg_write(REG_ARG0, 0x8000).unwrap();
        assert!(driven.begin());
        // replay the free run's schedule: identical records
        for rec in &free.trace {
            assert_eq!(&driven.step(rec.block, rec.warp).unwrap(), rec);
        }
        assert!(!driven.running());
        assert!(matches!(driven.step(0, 0), Err(SimError::NotRunnable(_))));
    }
}
