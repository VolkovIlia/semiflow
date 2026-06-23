//! `Shift1D` — universal 1-D Chernoff kernel `L = a(x)∂² + b(x)∂ + c(x)`.
//!
//! Wraps `ShiftChernoff1D` with the same ergonomic Python API as `Heat1D`:
//!   `Shift1D(xmin, xmax, n, u0, *, boundary='reflect')`
//!   `evolve(t, n_steps=100)`
//!   `values()` / `__len__()`
//!   `with_arrays(xmin, xmax, n, a, b, c, c_norm_bound, u0, ...)` — variable coeff
//!
//! ## GIL-release pattern (ADR-0031)
//!
//! Three-phase: validate under GIL, compute inside `py.detach`, update under GIL.
//! The closure path uses `ShiftChernoff1D::with_closure` (ADR-0034 ext) which
//! stores `Arc<dyn Fn + Send + Sync>` — safe to cross the GIL boundary.
//!
//! ## Note on formula (6) preconditions
//!
//! The caller must supply `a(x) > 0` everywhere (strict ellipticity).  The
//! kernel validates this at each grid node and returns `SemiflowError::DomainViolation`
//! on failure.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use numpy::{PyArray1, ToPyArray};
use pyo3::{prelude::*, types::PyAnyMethods};
use semiflow::{ChernoffFunction, ChernoffSemigroup, GridFn1D, ScratchPool, ShiftChernoff1D};

use crate::{
    boundary::parse_boundary,
    coeff::closure_from_array,
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

/// Inner state for `Shift1D`.
pub(crate) struct Shift1DInner {
    /// The Chernoff semigroup wrapping `ShiftChernoff1D`.
    pub semigroup: ChernoffSemigroup<ShiftChernoff1D<f64>, GridFn1D<f64>>,
    /// Current grid function state.
    pub current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Shift1D Python class
// ---------------------------------------------------------------------------

/// 1-D CDR state using the universal Chernoff kernel `L = a(x)∂² + b(x)∂ + c(x)`.
///
/// Solves `∂_t u = a(x)∂²u + b(x)∂_x u + c(x)u`.
/// Default: `a = 0.5`, `b = 0`, `c = 0`.  Implements formula (6), Remizov 2025.
/// Global consistency order 1.
///
/// Parameters
/// ----------
/// xmin : float
///     Left boundary.
/// xmax : float
///     Right boundary (> xmin).
/// n : int
///     Number of grid nodes (≥ 4).
/// u0 : numpy.ndarray[float64]
///     Initial condition, length exactly `n`.
/// boundary : str, optional
///     One of ``"reflect"`` (default), ``"periodic"``, ``"zero"``, ``"linear"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` for invalid grid params.
///     ``kind='NanInf'`` if `u0` contains NaN or Inf.
///     ``kind='OutOfDomain'`` if `a(x) <= 0` at any node during `evolve`.
#[pyclass(name = "Shift1D")]
pub struct Shift1D {
    inner: Shift1DInner,
}

#[pymethods]
impl Shift1D {
    /// Create a new `Shift1D`.
    ///
    /// `a`, `b`, `c` are constant scalar coefficients; defaults are `0.5`, `0.0`,
    /// `0.0`.  For spatially-varying coefficients use `with_arrays`.
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, a = 0.5, b = 0.0, c = 0.0, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        a: f64,
        b: f64,
        c: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let slice = extract_f64_slice(u0)?;
            let inner = build_shift_scalar(xmin, xmax, n, 100, &slice, a, b, c, policy)
                .map_err(|e| from_core(&e))?;
            Ok(Shift1D { inner })
        })
    }

    /// Advance state by time `t` using `n_steps` Chernoff iterations.
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_evolve_params(t, n_steps)?;
            let func = self.inner.semigroup.func.clone();
            let grid = self.inner.current.grid;
            let input: Vec<f64> = self.inner.current.values.clone();
            let func_for_sg = func.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| compute_evolve_shift(func, grid, input, t, n_steps));
            let result = result.map_err(|e| from_core(&e))?;
            self.inner.current.values = result;
            let sg = ChernoffSemigroup::new(func_for_sg, n_steps).map_err(|e| from_core(&e))?;
            self.inner.semigroup = sg;
            Ok(())
        })
    }

    /// Return the current grid values as `numpy.ndarray[float64]`.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Number of grid nodes.
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }

    /// Create a `Shift1D` from pre-sampled coefficient arrays.
    ///
    /// Parameters
    /// ----------
    /// xmin, xmax : float
    ///     Grid domain boundaries.
    /// n : int
    ///     Number of grid nodes.
    /// a : numpy.ndarray[float64]
    ///     Pre-sampled `a(x_i)` values, length `n`.  Must satisfy `a(x) > 0`.
    /// b : numpy.ndarray[float64]
    ///     Pre-sampled `b(x_i)` values, length `n`.  Must be finite.
    /// c : numpy.ndarray[float64]
    ///     Pre-sampled `c(x_i)` values, length `n`.  Must be finite.
    /// `c_norm_bound` : float
    ///     Upper bound on `‖c‖_∞`.  Must be non-negative.
    /// u0 : numpy.ndarray[float64]
    ///     Initial condition, length `n`.
    /// boundary : str, optional
    ///     Boundary policy; default ``"reflect"``.
    #[staticmethod]
    #[pyo3(signature = (xmin, xmax, n, a, b, c, c_norm_bound, u0, *, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn with_arrays(
        xmin: f64,
        xmax: f64,
        n: usize,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        c: &Bound<'_, PyAny>,
        c_norm_bound: f64,
        u0: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            build_shift_from_arrays(xmin, xmax, n, a, b, c, c_norm_bound, u0, boundary)
        })
    }

    /// Evolve with a piecewise-constant-in-time ``a`` schedule (D3 — ADR-0113).
    ///
    /// The ``a_schedule`` array contains ``n_segments`` constant values of the
    /// diffusion coefficient ``a(t)`` — one per segment on a uniform partition of
    /// ``[0, t_final]``.  ``b`` and ``c`` remain the spatially constant scalars
    /// supplied here.
    ///
    /// **GIL policy (ADR-0034)**: ``a_schedule`` is pre-sampled and copied to a
    /// plain ``Vec<f64>`` before ``py.detach``; NO Python callback enters the loop.
    ///
    /// **Scope note**: genuine joint space-time ``a(x, t)`` is out of scope —
    /// the core closures take only ``x``.  This method covers the
    /// time-varying-but-spatially-constant (piecewise) case.
    ///
    /// Parameters
    /// ----------
    /// `t_final` : float
    ///     Total evolution time.  Must be finite and >= 0.
    /// `n_steps_per_segment` : int
    ///     Number of Chernoff steps within each time segment.  Must be >= 1.
    /// `a_schedule` : numpy.ndarray[float64]
    ///     Constant ``a`` value for each segment; length = number of segments.
    ///     All values must be > 0.
    /// b : float, optional
    ///     Spatially constant drift coefficient (default 0.0).
    /// c : float, optional
    ///     Spatially constant reaction coefficient (default 0.0).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``OutOfDomain`` on invalid parameters.
    #[pyo3(signature = (t_final, n_steps_per_segment, a_schedule, *, b = 0.0, c = 0.0))]
    #[allow(clippy::too_many_arguments)]
    fn evolve_with_time_schedule(
        &mut self,
        py: Python<'_>,
        t_final: f64,
        n_steps_per_segment: usize,
        a_schedule: &Bound<'_, PyAny>,
        b: f64,
        c: f64,
    ) -> PyResult<()> {
        catch_panic_py!({
            validate_evolve_params(t_final, n_steps_per_segment)?;
            let schedule = extract_f64_slice(a_schedule)?;
            if schedule.is_empty() {
                return Err(new_pyerr("OutOfDomain", "a_schedule must be non-empty"));
            }
            let grid = self.inner.current.grid;
            let input: Vec<f64> = self.inner.current.values.clone();
            let result: Result<Vec<f64>, _> = py.detach(|| {
                compute_shift_time_schedule(
                    grid,
                    input,
                    t_final,
                    n_steps_per_segment,
                    schedule,
                    b,
                    c,
                )
            });
            self.inner.current.values = result.map_err(|e| from_core(&e))?;
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Scalar-coefficient builder
// ---------------------------------------------------------------------------

/// Build `Shift1DInner` with constant scalar `a`, `b`, `c`.
///
/// Uses `ShiftChernoff1D::with_closure` with captured scalars so the
/// GIL-release pattern in `evolve` is fully effective (ADR-0031 / ADR-0034).
///
/// # Errors
/// Propagates `SemiflowError` from grid or semigroup construction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_shift_scalar(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    u0: &[f64],
    a_val: f64,
    b_val: f64,
    c_val: f64,
    boundary: semiflow::BoundaryPolicy,
) -> Result<Shift1DInner, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = semiflow::Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let a_fn = move |_: f64| a_val;
    let b_fn = move |_: f64| b_val;
    let c_fn = move |_: f64| c_val;
    let norm = a_val.abs() + b_val.abs() + c_val.abs();
    let chernoff = ShiftChernoff1D::with_closure(a_fn, b_fn, c_fn, norm, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Shift1DInner { semigroup, current })
}

// ---------------------------------------------------------------------------
// Array-based builder (extracted to keep method body ≤50 lines)
// ---------------------------------------------------------------------------

/// Build `Shift1D` from pre-sampled `a`, `b`, `c` arrays.
#[allow(clippy::too_many_arguments)]
fn build_shift_from_arrays(
    xmin: f64,
    xmax: f64,
    n: usize,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    c: &Bound<'_, PyAny>,
    c_norm_bound: f64,
    u0: &Bound<'_, PyAny>,
    boundary: &str,
) -> PyResult<Shift1D> {
    let policy = parse_boundary(boundary)?;
    let slice = extract_f64_slice(u0)?;
    validate_u0_finite(&slice).map_err(|e| from_core(&e))?;
    let a_fn = closure_from_array(a, xmin, xmax, n)?;
    let b_fn = closure_from_array(b, xmin, xmax, n)?;
    let c_fn = closure_from_array(c, xmin, xmax, n)?;
    let grid = semiflow::Grid1D::new(xmin, xmax, n)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    let chernoff = ShiftChernoff1D::with_closure(a_fn, b_fn, c_fn, c_norm_bound, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, 100).map_err(|e| from_core(&e))?;
    let current = GridFn1D::new(grid, slice).map_err(|e| from_core(&e))?;
    Ok(Shift1D {
        inner: Shift1DInner { semigroup, current },
    })
}

// ---------------------------------------------------------------------------
// Phase 2 compute helper (called inside py.detach)
// ---------------------------------------------------------------------------

/// Evolve `ShiftChernoff1D` for `n_steps` steps of `t/n_steps`.
///
/// # Errors
/// Propagates `SemiflowError` from `ChernoffSemigroup`.
fn compute_evolve_shift(
    func: ShiftChernoff1D<f64>,
    grid: semiflow::Grid1D<f64>,
    input: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, input)?;
    Ok(sg.evolve(t, &f)?.values)
}

/// Piecewise-constant-in-time ``a`` schedule for ``Shift1D`` (D3 — ADR-0113).
///
/// Walks ``n_segments`` uniform time intervals of length ``dt = t_final / n``.
/// For each segment, constructs a `ShiftChernoff1D` with the segment's constant
/// ``a_k`` and runs ``n_steps_per_segment`` `apply_into` steps.
/// No Python objects are captured; safe to call inside `py.detach`.
fn compute_shift_time_schedule(
    grid: semiflow::Grid1D<f64>,
    input: Vec<f64>,
    t_final: f64,
    n_steps_per_segment: usize,
    a_schedule: Vec<f64>,
    b_val: f64,
    c_val: f64,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let n_segments = a_schedule.len();
    #[allow(clippy::cast_precision_loss)]
    let dt = t_final / n_segments as f64;
    let b_fn = move |_: f64| b_val;
    let c_fn = move |_: f64| c_val;
    let norm = b_val.abs() + c_val.abs();
    let mut state = GridFn1D::new(grid, input)?;
    let mut scratch = ScratchPool::new();
    for &a_k in &a_schedule {
        let a_fn = move |_: f64| a_k;
        let chernoff = ShiftChernoff1D::with_closure(a_fn, b_fn, c_fn, norm, grid);
        #[allow(clippy::cast_precision_loss)]
        let tau = dt / n_steps_per_segment as f64;
        for _ in 0..n_steps_per_segment {
            let mut next = state.clone();
            chernoff.apply_into(tau, &state, &mut next, &mut scratch)?;
            state = next;
        }
    }
    Ok(state.values)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_evolve_params(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn validate_u0_finite(u0: &[f64]) -> Result<(), semiflow::SemiflowError> {
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

/// Convert any Python array-like to a `Vec<f64>`.
fn extract_f64_slice(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "u0 must be a numpy.ndarray[float64] or a sequence of floats",
        )
    })
}
