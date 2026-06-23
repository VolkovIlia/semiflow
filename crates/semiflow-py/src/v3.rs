//! v3.0 `PyO3` surface (ADR-0076, Wave E, Approach A).
//!
//! Wraps `semiflow` v3 types (`Evolver`, `Growth<F>`, `apply_into`) for
//! Python callers.  **Additive** to the existing v2 pyclasses; the v2
//! compatibility shim layer was hard-removed at v4.0 (ADR-0084).
//!
//! ## New v3 pyclasses
//!
//! - [`PyGrowthV3`] (`GrowthV3`) — Python-facing mirror of `Growth<f64>`.
//!   Has `.multiplier` and `.omega` float attributes.  Tuple-iteration (e.g.
//!   `m, w = ev.growth()`) is NOT supported on a pyclass; callers use named
//!   fields.
//!
//! - [`PyEvolverHeat1DUnitV3`] (`EvolverHeat1DUnitV3`) — wraps
//!   `Evolver<DiffusionChernoff<f64>, f64>` with a `current: GridFn1D<f64>`
//!   in-place state.  GIL is released via `py.detach` during `evolve_into`
//!   (same three-phase design as `Heat2D::evolve_into` — ADR-0031).
//!
//! ## Design choices (recorded per ADR-0076 §"`PyO3` Approach A")
//!
//! - `GrowthV3` is a `#[pyclass]` (not a `PyTuple`): richer inspection from
//!   Python (`repr`, named fields, no positional-index magic).  The tradeoff
//!   is that Python callers must write `g.multiplier` instead of unpacking
//!   the tuple.  Wave D (FFI) uses a `#[repr(C)]` struct by value; this is
//!   the closest `PyO3` analogue.
//!
//! - `evolve_into` takes an in-place `PyReadwriteArray1<f64>` buffer (zero-
//!   alloc, mirrors Wave D `smf_evolver_evolve_into_v3`).  The caller
//!   pre-allocates the output buffer, matching the v3.0 preference for
//!   zero-alloc hot paths.  An allocating `evolve` convenience is NOT added
//!   here (it belongs on the existing `Heat1D` pyclass; this is a clean v3
//!   surface).
//!
//! - GIL release via `ptr-cast` pattern (same as `state_2d.rs`): the raw
//!   pointer address is cast to `usize` to cross the `py.detach` closure
//!   boundary, then reconstructed as `*mut f64` inside the closure.  This is
//!   sound because (a) the buffer is owned by the Python array and lives for
//!   `'py`, (b) no other thread can touch it while the GIL is held by our
//!   caller and then released to US only, (c) no Python code runs during the
//!   `py.detach` window.

#![allow(unsafe_code)]

use numpy::{PyArray1, PyReadwriteArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    ChernoffFunction, DiffusionChernoff, Evolver, Grid1D, GridFn1D, Growth, ScratchPool,
};

use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;

// ---------------------------------------------------------------------------
// GrowthV3 — Python mirror of Growth<f64>
// ---------------------------------------------------------------------------

/// Growth bound `‖S(τ)‖ ≤ M · exp(ω · τ)` (v3.0 pyclass, ADR-0074 / ADR-0076).
///
/// Attributes
/// ----------
/// multiplier : float
///     `M ≥ 1.0`.  For unit-diffusion: `1.0`.
/// omega : float
///     `ω` (finite).  For unit-diffusion: `0.0`.
///
/// Examples
/// --------
/// >>> ev = EvolverHeat1DUnitV3(-10.0, 10.0, 64, u0, 100)
/// >>> g = ev.growth()
/// >>> assert g.multiplier == 1.0
/// >>> assert g.omega == 0.0
#[pyclass(name = "GrowthV3")]
pub struct PyGrowthV3 {
    /// M in `‖S(τ)‖ ≤ M · exp(ω · τ)`.
    #[pyo3(get)]
    pub multiplier: f64,
    /// ω in `‖S(τ)‖ ≤ M · exp(ω · τ)`.
    #[pyo3(get)]
    pub omega: f64,
}

#[pymethods]
impl PyGrowthV3 {
    fn __repr__(&self) -> String {
        format!(
            "GrowthV3(multiplier={}, omega={})",
            self.multiplier, self.omega
        )
    }
}

impl From<Growth<f64>> for PyGrowthV3 {
    fn from(g: Growth<f64>) -> Self {
        Self {
            multiplier: g.multiplier,
            omega: g.omega,
        }
    }
}

// ---------------------------------------------------------------------------
// EvolverHeat1DUnitV3 — inner Rust state
// ---------------------------------------------------------------------------

/// Inner data for `EvolverHeat1DUnitV3` (Rust-private, heap-owned by pyclass).
struct EvolverInner {
    evolver: Evolver<DiffusionChernoff<f64>>,
    current: GridFn1D<f64>,
}

// DiffusionChernoff<f64> + GridFn1D<f64> + ScratchPool<f64> are Send + Sync.
// Verified by the workspace's send_assertions module and by the PyO3 Ungil
// requirement on py.detach closures.

// ---------------------------------------------------------------------------
// EvolverHeat1DUnitV3 pyclass
// ---------------------------------------------------------------------------

/// v3.0 Evolver for the unit-diffusion heat equation (ADR-0076, Wave E).
///
/// Solves `∂_t u = ∂²u` on `[domain_lo, domain_hi]` with `n_grid` nodes and
/// `n_chernoff` Chernoff iterations per `evolve_into` call.
///
/// This is the **v3-native** pyclass wrapping `Evolver<DiffusionChernoff<f64>,
/// f64>` directly (zero-alloc `apply_into` hot path).  For the v2 allocating
/// API, use `Heat1D` which is preserved for 12-month compat per ADR-0035 §9.
///
/// Parameters
/// ----------
/// `domain_lo` : float
///     Left boundary; must be finite.
/// `domain_hi` : float
///     Right boundary; must be finite and `> domain_lo`.
/// `n_grid` : int
///     Number of grid nodes; must be `≥ 4`.
/// u0 : array-like
///     Initial state; 1-D float64 array of length `n_grid`.
/// `n_chernoff` : int
///     Chernoff iteration count; must be `≥ 1`.
///
/// Raises
/// ------
/// `SemiflowError`
///     `kind='GridMismatch'` — invalid geometry or `len(u0) != n_grid`.
///     `kind='NanInf'` — `u0` contains NaN or Inf.
///     `kind='OutOfDomain'` — `n_chernoff == 0`.
#[pyclass(name = "EvolverHeat1DUnitV3")]
pub struct PyEvolverHeat1DUnitV3 {
    inner: EvolverInner,
}

#[pymethods]
impl PyEvolverHeat1DUnitV3 {
    /// Create a new v3 Evolver for unit-diffusion heat.
    #[new]
    fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        u0: &Bound<'_, pyo3::types::PyAny>,
        n_chernoff: usize,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let u0_vec = extract_f64_slice(u0)?;
            let inner = build_evolver_inner(domain_lo, domain_hi, n_grid, &u0_vec, n_chernoff)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Evolve the current state in-place by time `t`.
    ///
    /// Writes the evolved values into `buf` (zero-alloc).  The internal
    /// current state is updated to the result.  The GIL is released during
    /// the pure-Rust compute loop (ADR-0031 three-phase pattern).
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time to advance.  Must be non-negative and finite.
    /// buf : numpy.ndarray[float64]
    ///     Output buffer; length must equal `size()`.
    ///     Modified in-place with the evolved values.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     `kind='GridMismatch'` — `len(buf) != n_grid`.
    ///     `kind='OutOfDomain'` — `t < 0` or non-finite.
    fn evolve_into<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        mut buf: PyReadwriteArray1<'py, f64>,
    ) -> PyResult<()> {
        catch_panic_py!({
            validate_t_param(t)?;
            let expected = self.inner.current.values.len();
            let slice = buf
                .as_slice_mut()
                .map_err(|e| new_pyerr("GridMismatch", &format!("buf not contiguous: {e}")))?;
            if slice.len() != expected {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("buf length {} != n_grid {}", slice.len(), expected),
                ));
            }
            // --- Phase 1: extract Send data (under GIL) ---
            let chernoff_clone = self.inner.evolver.func.clone();
            let n_chernoff = self.inner.evolver.n;
            let src_values = self.inner.current.values.clone();
            let grid = self.inner.current.grid;
            // SAFETY: buf lives for 'py; GIL released only inside py.detach.
            let raw_addr: usize = slice.as_mut_ptr() as usize;
            let raw_len = slice.len();

            // --- Phase 2+3: compute (GIL released) then update state ---
            let new_values = evolve_detached(
                py,
                t,
                chernoff_clone,
                n_chernoff,
                src_values,
                grid,
                raw_addr,
                raw_len,
            )?;
            self.inner.current.values = new_values;
            Ok(())
        })
    }

    /// Return the growth bound of the underlying Chernoff function.
    ///
    /// Returns
    /// -------
    /// `GrowthV3`
    ///     Named-field growth bound `(multiplier, omega)`.
    fn growth(&self) -> PyGrowthV3 {
        let g: Growth<f64> = self.inner.evolver.func.growth();
        PyGrowthV3::from(g)
    }

    /// Return the number of grid nodes.
    fn size(&self) -> usize {
        self.inner.current.values.len()
    }

    /// Return the Chernoff iteration count.
    fn n_chernoff(&self) -> usize {
        self.inner.evolver.n
    }

    /// Return the current grid values as a 1-D `numpy.ndarray[float64]`.
    ///
    /// Returns a **copy**; mutations to the returned array do not affect
    /// the internal state.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            let arr = self.inner.current.values.as_slice().to_pyarray(py);
            Ok(arr)
        })
    }

    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }
}

// ---------------------------------------------------------------------------
// Evolve helper (Phase 2 pure-Rust compute + output copy, GIL released)
// ---------------------------------------------------------------------------

/// Pure-Rust compute phase for `evolve_into` (GIL released via `py.detach`).
///
/// Constructs an ephemeral `Evolver`, runs it into `dst`, copies results into
/// the pre-allocated output buffer at `raw_addr`, and returns the evolved
/// value vector for Phase 3 state update.
///
/// # Safety
/// `raw_addr` / `raw_len` must describe a valid mutable `f64` slice that
/// lives for the duration of the `py.detach` closure (`'py` lifetime of the
/// caller's `PyReadwriteArray1`).
#[allow(clippy::too_many_arguments)]
fn evolve_detached(
    py: Python<'_>,
    t: f64,
    chernoff: DiffusionChernoff<f64>,
    n_chernoff: usize,
    src_values: Vec<f64>,
    grid: Grid1D,
    raw_addr: usize,
    raw_len: usize,
) -> PyResult<Vec<f64>> {
    let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(|| {
        let evolver = Evolver::new(chernoff, n_chernoff)?;
        let src = GridFn1D::new(grid, src_values)?;
        let mut dst = src.clone();
        let mut scratch = ScratchPool::<f64>::new();
        evolver.evolve_into(t, &src, &mut dst, &mut scratch)?;
        // SAFETY: raw_addr came from caller's buf slice (lives for 'py).
        let out = unsafe { std::slice::from_raw_parts_mut(raw_addr as *mut f64, raw_len) };
        out.copy_from_slice(&dst.values);
        Ok(dst.values)
    });
    result.map_err(|e| from_core(&e))
}

// ---------------------------------------------------------------------------
// Builder helper
// ---------------------------------------------------------------------------

/// Build an `EvolverInner` for unit-diffusion heat on `[lo, hi]`.
///
/// Validates `u0` for finiteness.  Mirrors the pattern in
/// `crates/semiflow-ffi/src/v3.rs` `build_evolver_heat_unit`.
fn build_evolver_inner(
    lo: f64,
    hi: f64,
    n_grid: usize,
    u0: &[f64],
    n_chernoff: usize,
) -> Result<EvolverInner, semiflow::SemiflowError> {
    validate_u0(u0)?;
    let grid = Grid1D::new(lo, hi, n_grid)?;
    let chernoff = DiffusionChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
    let evolver = Evolver::new(chernoff, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(EvolverInner { evolver, current })
}

// ---------------------------------------------------------------------------
// Coefficient stubs (a = 1.0, a' = 0, a'' = 0)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}

extern "Rust" fn zero_d(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Convert a Python array-like to a `Vec<f64>`.
///
/// Mirrors `state_1d::extract_f64_slice` (private to that module; duplicated
/// here per the rule-of-three in `handle.rs`).
fn extract_f64_slice(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.extract::<Vec<f64>>() {
        return Ok(arr);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "u0 must be a numpy.ndarray[float64] or a sequence of floats",
    ))
}

/// Validate `t` for the `evolve_into` call.
fn validate_t_param(t: f64) -> PyResult<()> {
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

/// Validate all elements of `u0` are finite.
fn validate_u0(u0: &[f64]) -> Result<(), semiflow::SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(semiflow::SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register v3.0 pyclasses into the parent `semiflow` module.
///
/// Called from `lib.rs` `#[pymodule]` alongside the existing v2 registration
/// calls.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyGrowthV3>()?;
    m.add_class::<PyEvolverHeat1DUnitV3>()?;
    Ok(())
}
