//! Commit records (docs/sim/README.md, "Trace format"). Mirrors python/mgpu/trace.py.

pub const GLOBAL: u8 = 0;
pub const SHARED: u8 = 1;

pub const EV_NONE: u8 = 0;
pub const EV_EXIT: u8 = 1;
pub const EV_BAR: u8 = 2;
pub const EV_FAULT: u8 = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub step: u64,
    pub block: u32,
    pub warp: u32,
    pub pc: u32,
    pub word: u32,
    pub swrite: Option<(u8, u32)>,
    pub vwrite: Option<(u8, [u32; 32])>,
    pub exec_after: u32,
    /// (space, address, value) in program order.
    pub mwrites: Vec<(u8, u32, u32)>,
    pub event: u8,
    pub released: bool,
    /// (code, lane, info)
    pub fault: Option<(u32, u32, u32)>,
}

impl Record {
    pub fn new(step: u64, block: u32, warp: u32, pc: u32, word: u32) -> Self {
        Record {
            step,
            block,
            warp,
            pc,
            word,
            swrite: None,
            vwrite: None,
            exec_after: 0,
            mwrites: Vec::new(),
            event: EV_NONE,
            released: false,
            fault: None,
        }
    }
}
