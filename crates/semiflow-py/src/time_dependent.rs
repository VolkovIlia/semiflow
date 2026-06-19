//! Wave P4 — nonautonomous + subordinated (`time_dependent.rs`).
//!
//! Exposes two Chernoff families as Python classes:
//!
//! | pyclass | Core type | M# | Module |
//! |---------|-----------|-----|--------|
//! | `Howland1D` | `HowlandLift<DiffusionChernoff<f64>>` | M11 | here |
//! | `Subordinated1D` | `SubordinatedChernoff<DiffusionChernoff<f64>, SubordinatorEnum, f64>` | M12 | `subordinated_py` |
//!
//! ## Design notes
//!
//! ### `Howland1D`
//!
//! `HowlandLift::apply_into` enforces `tau == delta_s = t_horizon / (n_t − 1)`.
//! `ChernoffSemigroup::evolve(t, state)` calls `apply_into` with `tau = t / n_steps`.
//! So `n_steps` MUST equal `n_t − 1` to get `tau = delta_s` exactly.  The binding
//! therefore fixes `n_steps = n_t − 1` internally; the Python method `evolve()` is
//! parameter-free (the time horizon is set at construction).
//!
//! ## GIL policy
//!
//! All `evolve` methods: validate + copy under GIL → `py.detach` compute →
//! write result under GIL (ADR-0031 three-phase pattern).

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::unused_self
)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow_core::{
    diffusion::DiffusionChernoff,
    howland::{HowlandLift, HowlandState},
    ChernoffSemigroup, Grid1D, GridFn1D,
};

use crate::{
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// Re-export Subordinated1D from its own module for registration.
pub(crate) use crate::subordinated_py::PySubordinated1D;

// ---------------------------------------------------------------------------
// Shared helpers (pub(crate) so subordinated_py.rs can access them)
// ---------------------------------------------------------------------------

pub(crate) extern "Rust" fn unit_a_td(_: f64) -> f64 {
    1.0
}

pub(crate) extern "Rust" fn zero_td(_: f64) -> f64 {
    0.0
}

pub(crate) fn validate_params_td(n_steps: usize, t: f64) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

pub(crate) fn validate_u0_td(u0: &[f64]) -> Result<(), semiflow_core::SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(semiflow_core::SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}

pub(crate) fn extract_f64_vec_td(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "u0 must be a numpy.ndarray[float64] or a sequence of floats",
        )
    })
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type HowlandKernel = HowlandLift<DiffUnit, f64>;

// ---------------------------------------------------------------------------
// Howland1D inner state
// ---------------------------------------------------------------------------

struct Howland1DInner {
    lift: HowlandKernel,
    grid: Grid1D<f64>,
    /// Current `HowlandState`: `n_t` time slices of `GridFn1D`<f64>.
    state: HowlandState<GridFn1D<f64>, f64>,
    n_t: usize,
    t_horizon: f64,
}

// ---------------------------------------------------------------------------
// Howland1D pyclass (M11)
// ---------------------------------------------------------------------------

/// 1-D nonautonomous heat via Howland lift (M11).
///
/// Wraps ``HowlandLift<DiffusionChernoff<f64>>`` — the autonomous unit-diffusion
/// generator lifted to ``L²([0, t_horizon], L²([xmin, xmax]))`` (Howland 1974
/// *Trans. AMS* **207** Theorem 1, math.md §23, ADR-0070).
///
/// For an **autonomous** base generator (the only kind bindable here), the
/// Howland-lifted evolution at ``t = t_horizon`` is exactly equal to the
/// regular heat semigroup ``S(t_horizon)``.  This identity is used as the
/// oracle in the smoke test:
/// ``Howland1D(…, n_t=101, t_horizon=T).evolve()  ==  Heat1D(…).evolve(T, 100)``
/// to within the Chernoff discretisation error.
///
/// Parameters
/// ----------
/// xmin : float
///     Left boundary of the spatial domain.
/// xmax : float
///     Right boundary (must be > xmin).
/// n : int
///     Number of spatial grid nodes (must be >= 4).
/// u0 : array-like
///     Initial condition; float64 array of length n.  Replicated across all `n_t`
///     time slices to form the Howland initial state.
/// `n_t` : int, optional
///     Number of temporal grid points in ``[0, t_horizon]`` (default 11,
///     giving ``delta_s = t_horizon / 10``).  Must be >= 2.
/// `t_horizon` : float, optional
///     Time horizon ``T`` (default 0.1).  Must be finite and > 0.
/// boundary : str, optional
///     Spatial boundary policy (keyword-only); default ``"reflect"``.
///
/// Notes
/// -----
/// The matched-step constraint ``tau = delta_s = T / (n_t − 1)`` is enforced
/// internally.  Calling ``evolve()`` is parameter-free because ``t_horizon``
/// and ``n_t`` are fixed at construction.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if xmin >= xmax or n < 4 or len(u0) != n.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if `n_t` < 2, `t_horizon` <= 0, or boundary is unrecognised.
#[pyclass(name = "Howland1D")]
pub struct PyHowland1D {
    inner: Howland1DInner,
}

#[pymethods]
impl PyHowland1D {
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, n_t = 11, t_horizon = 0.1, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        n_t: usize,
        t_horizon: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = crate::boundary::parse_boundary(boundary)?;
            let u0_vec = extract_f64_vec_td(u0)?;
            let inner = build_howland(xmin, xmax, n, &u0_vec, n_t, t_horizon, policy)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (always 1 — left-endpoint shift is order-1).
    fn order(&self) -> u32 {
        1
    }

    /// Return the time-grid spacing ``delta_s = t_horizon / (n_t − 1)``.
    fn delta_s(&self) -> f64 {
        self.inner.lift.delta_s()
    }

    /// Return the number of temporal grid points ``n_t``.
    fn n_t(&self) -> usize {
        self.inner.n_t
    }

    /// Return the time horizon ``T`` set at construction.
    fn t_horizon(&self) -> f64 {
        self.inner.t_horizon
    }

    /// Advance the Howland state by one full ``t_horizon`` evolution.
    ///
    /// Uses ``n_steps = n_t − 1`` Chernoff iterations with ``tau = delta_s``
    /// (matched-step requirement of ``HowlandLift``, math §23.4).
    ///
    /// GIL released during inner Rust compute (ADR-0031).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if the matched-step constraint is violated
    ///     (should not happen for correctly constructed state).
    fn evolve(&mut self, py: Python<'_>) -> PyResult<()> {
        catch_panic_py!({
            let lift = self.inner.lift.clone();
            let state = self.inner.state.clone();
            let result: Result<HowlandState<GridFn1D<f64>, f64>, _> =
                py.detach(|| evolve_howland(lift, state));
            self.inner.state = result.map_err(|e| from_core(&e))?;
            Ok(())
        })
    }

    /// Return the last time slice ``u(t_horizon, ·)`` as ``numpy.ndarray[float64]``.
    ///
    /// Returns a copy; dtype is float64; length is ``n`` (spatial grid size).
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            let last = &self.inner.state.samples[self.inner.n_t - 1];
            Ok(last.values.as_slice().to_pyarray(py))
        })
    }

    /// Number of spatial grid nodes ``n``.
    fn __len__(&self) -> usize {
        self.inner.grid.n
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fn build_howland(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    n_t: usize,
    t_horizon: f64,
    boundary: semiflow_core::BoundaryPolicy,
) -> Result<Howland1DInner, semiflow_core::SemiflowError> {
    validate_u0_td(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let diff = DiffusionChernoff::new(unit_a_td, zero_td, zero_td, 1.0, grid);
    let lift = HowlandLift::new(diff, t_horizon, n_t)?;
    // Replicate u0 across all n_t time slices.
    let slice = GridFn1D::new(grid, u0.to_vec())?;
    let samples: Vec<GridFn1D<f64>> = (0..n_t).map(|_| slice.clone()).collect();
    let state = HowlandState::new(samples)?;
    Ok(Howland1DInner {
        lift,
        grid,
        state,
        n_t,
        t_horizon,
    })
}

// ---------------------------------------------------------------------------
// GIL-free compute helper
// ---------------------------------------------------------------------------

fn evolve_howland(
    lift: HowlandKernel,
    state: HowlandState<GridFn1D<f64>, f64>,
) -> Result<HowlandState<GridFn1D<f64>, f64>, semiflow_core::SemiflowError> {
    let n_t = lift.n_t();
    // n_steps = n_t - 1 so that tau = t_horizon / (n_t - 1) = delta_s exactly.
    let n_steps = n_t - 1;
    let sg = ChernoffSemigroup::new(lift.clone(), n_steps)?;
    sg.evolve(lift.delta_s() * n_steps as f64, &state)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register Wave P4 pyclasses into the `semiflow` module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyHowland1D>()?;
    m.add_class::<PySubordinated1D>()?;
    Ok(())
}
