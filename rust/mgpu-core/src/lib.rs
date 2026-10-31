//! Fast MGPU-32 simulator engine.
//!
//! The semantics mirror the pure-Python reference model (python/mgpu/model) instruction for
//! instruction, and both follow the same reference schedule, so they produce identical
//! commit traces. See docs/sim/README.md.

pub mod capi;
pub mod decode;
pub mod fp;
pub mod gpu;
#[rustfmt::skip]
pub mod isa_gen;
pub mod sched;
pub mod sfu;
#[rustfmt::skip]
pub mod sfu_tables;
pub mod trace;

pub use gpu::{Access, AccessKind, Config, Gpu, SimError, Warp, WarpState};
pub use trace::Record;
