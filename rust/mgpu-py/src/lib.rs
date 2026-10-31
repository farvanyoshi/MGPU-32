//! Python bindings: `mgpu._core`. The `Gpu` class has the same API as
//! `mgpu.model.Gpu`, so `mgpu.device.Device` can use either backend.

use mgpu_core::timing::{STALL_NAMES, UNIT_NAMES};
use mgpu_core::{fp, launch_timed, sfu, Config, Record, SimError, TimingConfig};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

fn to_pyerr(py: Python<'_>, e: SimError) -> PyErr {
    match e {
        SimError::Timeout(_) => match py
            .import("mgpu.errors")
            .and_then(|m| m.getattr("SimTimeout"))
            .and_then(|cls| cls.call1((e.to_string(),)))
        {
            Ok(exc) => PyErr::from_value(exc),
            Err(err) => err,
        },
        SimError::HostRange(..) => PyIndexError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

fn record<'py>(py: Python<'py>, r: &Record) -> PyResult<Bound<'py, PyTuple>> {
    let vwrite = match &r.vwrite {
        Some((reg, vals)) => Some((*reg, PyTuple::new(py, vals.iter())?)),
        None => None,
    };
    (
        r.step,
        r.block,
        r.warp,
        r.pc,
        r.word,
        r.swrite,
        vwrite,
        r.exec_after,
        r.mwrites.clone(),
        r.event,
        r.released,
        r.fault,
    )
        .into_pyobject(py)
}

/// (pc, s[32], v[32][32], state) of one warp.
type WarpState = (u32, Vec<u32>, Vec<Vec<u32>>, u8);

#[pyclass(name = "Gpu", module = "mgpu._core", unsendable)]
struct PyGpu {
    inner: mgpu_core::Gpu,
}

#[pymethods]
impl PyGpu {
    #[new]
    #[pyo3(signature = (mem_size=64 << 20, max_block_dim=1024, resident_blocks=1, seed=None,
                        shared_poison=0xDEADBEEF, max_steps=50_000_000, trace=false,
                        keep_state=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        mem_size: u64,
        max_block_dim: u32,
        resident_blocks: usize,
        seed: Option<u64>,
        shared_poison: u32,
        max_steps: u64,
        trace: bool,
        keep_state: bool,
    ) -> PyResult<Self> {
        let cfg = Config {
            mem_size,
            max_block_dim,
            resident_blocks,
            seed,
            shared_poison,
            max_steps,
            trace,
            keep_state,
        };
        mgpu_core::Gpu::new(cfg)
            .map(|inner| PyGpu { inner })
            .map_err(|e| to_pyerr(py, e))
    }

    #[getter]
    fn mem_size(&self) -> u64 {
        self.inner.cfg.mem_size
    }

    #[getter]
    fn steps(&self) -> u64 {
        self.inner.steps
    }

    #[getter]
    fn running(&self) -> bool {
        self.inner.running()
    }

    fn mem_write(&mut self, py: Python<'_>, addr: u64, data: &[u8]) -> PyResult<()> {
        self.inner
            .mem_write(addr, data)
            .map_err(|e| to_pyerr(py, e))
    }

    fn mem_read<'py>(&self, py: Python<'py>, addr: u64, n: u64) -> PyResult<Bound<'py, PyBytes>> {
        let data = self.inner.mem_read(addr, n).map_err(|e| to_pyerr(py, e))?;
        Ok(PyBytes::new(py, data))
    }

    fn reg_read(&self, off: u32) -> u32 {
        self.inner.reg_read(off)
    }

    fn reg_write(&mut self, py: Python<'_>, off: u32, value: u64) -> PyResult<()> {
        self.inner
            .reg_write(off, value as u32)
            .map_err(|e| to_pyerr(py, e))
    }

    fn begin(&mut self) -> bool {
        self.inner.begin()
    }

    fn run(&mut self, py: Python<'_>) -> PyResult<()> {
        self.inner.run().map_err(|e| to_pyerr(py, e))
    }

    fn runnable(&self) -> Vec<(u32, u32)> {
        self.inner.runnable()
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        block: u32,
        warp: u32,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let r = self.inner.step(block, warp).map_err(|e| to_pyerr(py, e))?;
        record(py, &r)
    }

    /// Launch the kernel described by the host registers under the cycle-level timing model
    /// (docs/uarch/core.md section 11). Returns the timing report as a dict.
    #[pyo3(signature = (cores=1, warps=8, lanes=8, sfu_lanes=2, mem_latency=100, seg_cycles=8,
                        icache_lines=16, max_cycles=1u64 << 40))]
    #[allow(clippy::too_many_arguments)]
    fn launch_timed<'py>(
        &mut self,
        py: Python<'py>,
        cores: usize,
        warps: u32,
        lanes: u32,
        sfu_lanes: u32,
        mem_latency: u64,
        seg_cycles: u64,
        icache_lines: usize,
        max_cycles: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let tc = TimingConfig {
            cores,
            warps,
            lanes,
            sfu_lanes,
            mem_latency,
            seg_cycles,
            icache_lines,
            max_cycles,
        };
        let r = launch_timed(&mut self.inner, &tc).map_err(|e| to_pyerr(py, e))?;
        let d = PyDict::new(py);
        d.set_item("cycles", r.cycles)?;
        d.set_item("issued", r.issued)?;
        d.set_item("ipc", r.ipc())?;
        d.set_item("issue_utilization", r.issue_utilization())?;
        d.set_item("simd_utilization", r.simd_utilization())?;
        d.set_item("vector_issued", r.vector_issued)?;
        d.set_item("core_active_cycles", r.core_active_cycles)?;
        let by_unit = |v: &[u64; 5]| -> PyResult<Bound<'py, PyDict>> {
            let u = PyDict::new(py);
            for (n, x) in UNIT_NAMES.iter().zip(v) {
                u.set_item(*n, *x)?;
            }
            Ok(u)
        };
        d.set_item("issue_by_unit", by_unit(&r.issue_by_unit)?)?;
        d.set_item("busy_by_unit", by_unit(&r.busy_by_unit)?)?;
        let st = PyDict::new(py);
        for (n, x) in STALL_NAMES.iter().zip(&r.stalls) {
            st.set_item(*n, *x)?;
        }
        d.set_item("stalls", st)?;
        for (k, v) in [
            ("mem_reads", r.mem_reads),
            ("mem_writes", r.mem_writes),
            ("mem_atomics", r.mem_atomics),
            ("icache_hits", r.icache_hits),
            ("icache_misses", r.icache_misses),
            ("shared_conflict_cycles", r.shared_conflict_cycles),
        ] {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// Commit records of the last launch as canonical tuples (see mgpu.trace).
    #[getter]
    fn trace<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items = self
            .inner
            .trace
            .iter()
            .map(|r| record(py, r))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    /// (pc, s[32], v[32][32], state); needs keep_state=True after the kernel ends.
    fn warp_state(&self, block: u32, warp: u32) -> PyResult<WarpState> {
        let w = self
            .inner
            .warp(block, warp)
            .ok_or_else(|| PyValueError::new_err(format!("no state for warp ({block}, {warp})")))?;
        Ok((
            w.pc,
            w.s.to_vec(),
            w.v.iter().map(|r| r.to_vec()).collect(),
            w.state as u8,
        ))
    }
}

fn fp_fn(name: &str) -> PyResult<fn(u32, u32, u32) -> u32> {
    Ok(match name {
        "fadd" => |a, b, _| fp::fadd(a, b),
        "fsub" => |a, b, _| fp::fsub(a, b),
        "fmul" => |a, b, _| fp::fmul(a, b),
        "fmin" => |a, b, _| fp::fmin(a, b),
        "fmax" => |a, b, _| fp::fmax(a, b),
        "ffma" => fp::ffma,
        "i2f" => |a, _, _| fp::i2f(a),
        "f2i" => |a, _, _| fp::f2i(a),
        "frcp" => |a, _, _| sfu::frcp(a),
        "frsq" => |a, _, _| sfu::frsq(a),
        "fex2" => |a, _, _| sfu::fex2(a),
        "flg2" => |a, _, _| sfu::flg2(a),
        "feq" => |a, b, _| fp::fcmp(fp::FCond::Eq, a, b) as u32,
        "fne" => |a, b, _| fp::fcmp(fp::FCond::Ne, a, b) as u32,
        "flt" => |a, b, _| fp::fcmp(fp::FCond::Lt, a, b) as u32,
        "fle" => |a, b, _| fp::fcmp(fp::FCond::Le, a, b) as u32,
        "fgt" => |a, b, _| fp::fcmp(fp::FCond::Gt, a, b) as u32,
        "fge" => |a, b, _| fp::fcmp(fp::FCond::Ge, a, b) as u32,
        _ => return Err(PyValueError::new_err(format!("unknown float op {name:?}"))),
    })
}

/// Apply a float operation to one operand triple (unused operands are ignored).
#[pyfunction]
#[pyo3(signature = (name, a, b=0, c=0))]
fn fp_op(name: &str, a: u32, b: u32, c: u32) -> PyResult<u32> {
    Ok(fp_fn(name)?(a, b, c))
}

/// Apply a float operation element-wise to equal-length operand lists.
#[pyfunction]
#[pyo3(signature = (name, a, b=None, c=None))]
fn fp_batch(
    name: &str,
    a: Vec<u32>,
    b: Option<Vec<u32>>,
    c: Option<Vec<u32>>,
) -> PyResult<Vec<u32>> {
    let f = fp_fn(name)?;
    let n = a.len();
    let b = b.unwrap_or_else(|| vec![0; n]);
    let c = c.unwrap_or_else(|| vec![0; n]);
    if b.len() != n || c.len() != n {
        return Err(PyValueError::new_err("operand lists differ in length"));
    }
    Ok((0..n).map(|i| f(a[i], b[i], c[i])).collect())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGpu>()?;
    m.add_function(wrap_pyfunction!(fp_op, m)?)?;
    m.add_function(wrap_pyfunction!(fp_batch, m)?)?;
    Ok(())
}
