//! `DriftReaction1D` — RK2 characteristic-flow Chernoff for `b(x)∂_x + c(x)`.
//!
//! Wraps `DriftReactionChernoff` with the same ergonomic Python API as `Heat1D`:
//!   `DriftReaction1D(xmin, xmax, n, u0, *, boundary='reflect')`
//!   `evolve(t, n_steps=100)`
//!   `values()` / `__len__()`
//!   `with_arrays(xmin, xmax, n, b, c, c_norm_bound, u0, ...)` — variable coefficients
//!
//! ## GIL-release pattern (ADR-0031)
//!
//! Same three-phase structure as `Heat1D`. The closure path uses
//! `DriftReactionChernoff::with_closure` which stores `Arc<dyn Fn>` —
//! both `Send + Sync`, so `py.detach` is safe.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use numpy::{PyArray1, ToPyArray};
use pyo3::{prelude::*, types::PyAnyMethods};
use semiflow::{ChernoffFunction, ChernoffSemigroup, DriftReactionChernoff, GridFn1D, ScratchPool};

use crate::{
    boundary::parse_boundary,
    coeff::closure_from_array,
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

/// Inner state for `DriftReaction1D`.
pub(crate) struct DriftReactionInner {
    /// The Chernoff semigroup wrapping `DriftReactionChernoff`.
    pub semigroup: ChernoffSemigroup<DriftReactionChernoff<f64>, GridFn1D<f64>>,
    /// Current grid function state.
    pub current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// DriftReaction1D Python class
// ---------------------------------------------------------------------------

/// 1-D drift+reaction state using the RK2 characteristic-flow Chernoff kernel.
///
/// Solves `∂_t u = b(x)∂_x u + c(x)u`.  Default coefficients: `b = 0.5`, `c = 0`.
/// Uses `DriftReactionChernoff` — global order 2 in time.
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
#[pyclass(name = "DriftReaction1D")]
pub struct DriftReaction1D {
    inner: DriftReactionInner,
}

#[pymethods]
impl DriftReaction1D {
    /// Create a new `DriftReaction1D`.
    ///
    /// `b` and `c` are constant scalar coefficients; they default to `0.5` and
    /// `0.0` respectively.  For spatially-varying coefficients use `with_arrays`.
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, b = 0.5, c = 0.0, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        b: f64,
        c: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let slice = extract_f64_slice(u0)?;
            let inner = build_drift_scalar(xmin, xmax, n, 100, &slice, b, c, policy)
                .map_err(|e| from_core(&e))?;
            Ok(DriftReaction1D { inner })
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
                py.detach(|| compute_evolve_drift(func, grid, input, t, n_steps));
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

    /// Create a `DriftReaction1D` from pre-sampled coefficient arrays.
    ///
    /// Parameters
    /// ----------
    /// xmin, xmax : float
    ///     Grid domain boundaries.
    /// n : int
    ///     Number of grid nodes.
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
    #[pyo3(signature = (xmin, xmax, n, b, c, c_norm_bound, u0, *, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn with_arrays(
        xmin: f64,
        xmax: f64,
        n: usize,
        b: &Bound<'_, PyAny>,
        c: &Bound<'_, PyAny>,
        c_norm_bound: f64,
        u0: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            build_drift_from_arrays(xmin, xmax, n, b, c, c_norm_bound, u0, boundary)
        })
    }

    /// Evolve with a piecewise-constant-in-time ``b`` schedule (D3 — ADR-0113).
    ///
    /// The ``b_schedule`` array contains ``n_segments`` constant values of the
    /// drift coefficient ``b(t)`` — one per segment on a uniform partition of
    /// ``[0, t_final]``.  ``c`` remains the spatially constant scalar supplied here.
    ///
    /// **GIL policy (ADR-0034)**: ``b_schedule`` is pre-sampled and copied to a
    /// plain ``Vec<f64>`` before ``py.detach``; NO Python callback enters the loop.
    ///
    /// **Scope note**: genuine joint space-time ``b(x, t)`` is out of scope —
    /// the core closures take only ``x``.  This method covers the
    /// time-varying-but-spatially-constant (piecewise) case.
    ///
    /// Parameters
    /// ----------
    /// `t_final` : float
    ///     Total evolution time.  Must be finite and >= 0.
    /// `n_steps_per_segment` : int
    ///     Number of Chernoff steps within each time segment.  Must be >= 1.
    /// `b_schedule` : numpy.ndarray[float64]
    ///     Constant ``b`` value for each segment; length = number of segments.
    /// c : float, optional
    ///     Spatially constant reaction coefficient (default 0.0).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``OutOfDomain`` on invalid parameters.
    #[pyo3(signature = (t_final, n_steps_per_segment, b_schedule, *, c = 0.0))]
    fn evolve_with_time_schedule(
        &mut self,
        py: Python<'_>,
        t_final: f64,
        n_steps_per_segment: usize,
        b_schedule: &Bound<'_, PyAny>,
        c: f64,
    ) -> PyResult<()> {
        catch_panic_py!({
            validate_evolve_params(t_final, n_steps_per_segment)?;
            let schedule = extract_f64_slice(b_schedule)?;
            if schedule.is_empty() {
                return Err(new_pyerr("OutOfDomain", "b_schedule must be non-empty"));
            }
            let grid = self.inner.current.grid;
            let input: Vec<f64> = self.inner.current.values.clone();
            let result: Result<Vec<f64>, _> = py.detach(|| {
                compute_drift_time_schedule(grid, input, t_final, n_steps_per_segment, schedule, c)
            });
            self.inner.current.values = result.map_err(|e| from_core(&e))?;
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Scalar-coefficient builder
// ---------------------------------------------------------------------------

/// Build `DriftReactionInner` with constant scalar `b` and `c`.
///
/// Uses heap-allocated `Arc<dyn Fn>` closures that capture the scalar values,
/// so the GIL-release pattern in `evolve` is fully effective.
///
/// # Errors
/// Propagates `SemiflowError` from grid or semigroup construction.
pub(crate) fn build_drift_scalar(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    u0: &[f64],
    b_val: f64,
    c_val: f64,
    boundary: semiflow::BoundaryPolicy,
) -> Result<DriftReactionInner, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = semiflow::Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let b_fn = move |_: f64| b_val;
    let c_fn = move |_: f64| c_val;
    let norm = b_val.abs() + c_val.abs();
    let chernoff = DriftReactionChernoff::with_closure(b_fn, c_fn, norm, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(DriftReactionInner { semigroup, current })
}

// ---------------------------------------------------------------------------
// Array-based builder (extracted to keep method body ≤50 lines)
// ---------------------------------------------------------------------------

/// Build `DriftReaction1D` from pre-sampled `b` and `c` arrays.
#[allow(clippy::too_many_arguments)]
fn build_drift_from_arrays(
    xmin: f64,
    xmax: f64,
    n: usize,
    b: &Bound<'_, PyAny>,
    c: &Bound<'_, PyAny>,
    c_norm_bound: f64,
    u0: &Bound<'_, PyAny>,
    boundary: &str,
) -> PyResult<DriftReaction1D> {
    let policy = parse_boundary(boundary)?;
    let slice = extract_f64_slice(u0)?;
    validate_u0_finite(&slice).map_err(|e| from_core(&e))?;
    let b_fn = closure_from_array(b, xmin, xmax, n)?;
    let c_fn = closure_from_array(c, xmin, xmax, n)?;
    let grid = semiflow::Grid1D::new(xmin, xmax, n)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    let chernoff = DriftReactionChernoff::with_closure(b_fn, c_fn, c_norm_bound, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, 100).map_err(|e| from_core(&e))?;
    let current = GridFn1D::new(grid, slice).map_err(|e| from_core(&e))?;
    Ok(DriftReaction1D {
        inner: DriftReactionInner { semigroup, current },
    })
}

// ---------------------------------------------------------------------------
// Phase 2 compute helper (called inside py.detach)
// ---------------------------------------------------------------------------

/// Evolve `DriftReactionChernoff` for `n_steps` steps of `t/n_steps`.
///
/// # Errors
/// Propagates `SemiflowError` from `ChernoffSemigroup`.
fn compute_evolve_drift(
    func: DriftReactionChernoff<f64>,
    grid: semiflow::Grid1D<f64>,
    input: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, input)?;
    Ok(sg.evolve(t, &f)?.values)
}

/// Piecewise-constant-in-time ``b`` schedule for ``DriftReaction1D`` (D3 — ADR-0113).
///
/// Walks ``n_segments`` uniform time intervals of length ``dt = t_final / n``.
/// For each segment, constructs a `DriftReactionChernoff` with the segment's constant
/// ``b_k`` and runs ``n_steps_per_segment`` `apply_into` steps.
/// No Python objects are captured; safe to call inside `py.detach`.
fn compute_drift_time_schedule(
    grid: semiflow::Grid1D<f64>,
    input: Vec<f64>,
    t_final: f64,
    n_steps_per_segment: usize,
    b_schedule: Vec<f64>,
    c_val: f64,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let n_segments = b_schedule.len();
    #[allow(clippy::cast_precision_loss)]
    let dt = t_final / n_segments as f64;
    let c_fn = move |_: f64| c_val;
    let norm = c_val.abs();
    let mut state = GridFn1D::new(grid, input)?;
    let mut scratch = ScratchPool::new();
    for &b_k in &b_schedule {
        let b_fn = move |_: f64| b_k;
        let chernoff = DriftReactionChernoff::with_closure(b_fn, c_fn, norm, grid);
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
