//! `DriftReaction4th1D` — order-4 palindromic Strang drift-reaction binding.
//!
//! Wraps `DriftReactionZeta4Chernoff` (ADR-0127, math §35).
//! Palindromic `R_sym(τ/2)` ∘ K5(τ) ∘ `R_sym(τ/2)`; order 4.
//!
//! Default coefficients: b(x) = 0.5, b'(x) = 0.0, c(x) = 0.0.
//! Variable-coefficient support (via closures) is deferred per the
//! directive: `DriftReactionZeta4Chernoff::new` accepts `fn(f64)->f64`
//! pointers only; closure-capture API is a separate architect task.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    ChernoffSemigroup, Diffusion4thChernoff, DriftReactionZeta4Chernoff, Grid1D, GridFn1D,
};

use crate::{
    boundary::parse_boundary,
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct Zeta4DrInner {
    semigroup: ChernoffSemigroup<DriftReactionZeta4Chernoff, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Static function pointers for default coefficients
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_dr4(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_dr4(_: f64) -> f64 {
    0.0
}
extern "Rust" fn half_b_dr4(_: f64) -> f64 {
    0.5
}

// ---------------------------------------------------------------------------
// DriftReaction4th1D pyclass
// ---------------------------------------------------------------------------

/// 1-D drift + reaction with order-4 palindromic Strang-K5 Chernoff kernel.
///
/// Solves ``∂_t u = b(x)∂_x u + c(x)u`` using ``DriftReactionZeta4Chernoff``
/// (palindromic `R_sym(τ/2)` ∘ K5(τ) ∘ `R_sym(τ/2)`; order 4, ADR-0127).
///
/// Default coefficients: ``b(x) = 0.5``, ``b'(x) = 0.0``, ``c(x) = 0.0``.
/// Variable-coefficient support is deferred (closure-capture ABI is a
/// separate architect task; ``DriftReactionZeta4Chernoff::new`` takes fn ptrs).
///
/// Parameters
/// ----------
/// xmin : float
///     Left boundary.
/// xmax : float
///     Right boundary (must be > xmin).
/// n : int
///     Number of grid nodes (must be >= 4).
/// u0 : array-like of float
///     Initial condition; length n, all finite.
/// boundary : str, optional
///     Boundary policy (keyword-only).  One of ``"reflect"`` (default),
///     ``"periodic"``, ``"zero"``, ``"linear"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if xmin >= xmax or n < 4 or len(u0) != n.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if boundary is not recognised.
#[pyclass(name = "DriftReaction4th1D")]
pub struct DriftReaction4th1D {
    inner: Zeta4DrInner,
}

#[pymethods]
impl DriftReaction4th1D {
    /// Create a new ``DriftReaction4th1D`` state.
    ///
    /// Uses default coefficients: ``b(x) = 0.5``, ``b'(x) = 0.0``, ``c(x) = 0.0``.
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, boundary = "reflect"))]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let u0_vec = extract_f64_vec(u0)?;
            let inner =
                build_zeta4_dr(xmin, xmax, n, &u0_vec, policy).map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (always 4).
    // unused_self: PyO3 requires `&self` on instance methods; the value is constant.
    #[allow(clippy::unused_self)]
    fn order(&self) -> u32 {
        4
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// The GIL is released during the inner Rust compute loop (ADR-0031).
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_params(t, n_steps)?;
            let func = self.inner.semigroup.func.clone();
            let grid = self.inner.current.grid;
            let values: Vec<f64> = self.inner.current.values.clone();
            let func_clone = func.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| compute_evolve_zeta4_dr(func, grid, values, t, n_steps));
            self.inner.current.values = result.map_err(|e| from_core(&e))?;
            let sg = ChernoffSemigroup::new(func_clone, n_steps).map_err(|e| from_core(&e))?;
            self.inner.semigroup = sg;
            Ok(())
        })
    }

    /// Return the current grid values as a 1-D ``numpy.ndarray[float64]`` (copy).
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Return the number of grid nodes.
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }
}

// ---------------------------------------------------------------------------
// Builder / helpers
// ---------------------------------------------------------------------------

fn build_zeta4_dr(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    boundary: semiflow::BoundaryPolicy,
) -> Result<Zeta4DrInner, semiflow::SemiflowError> {
    validate_u0(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let d4 = Diffusion4thChernoff::new(unit_a_dr4, zero_dr4, zero_dr4, 1.0, grid);
    // c_norm_bound = 0.5 (b=0.5, b'=0, c=0 → growth dominated by b=0.5)
    let kernel = DriftReactionZeta4Chernoff::new(d4, half_b_dr4, zero_dr4, zero_dr4, 0.5, grid);
    let semigroup = ChernoffSemigroup::new(kernel, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Zeta4DrInner { semigroup, current })
}

fn compute_evolve_zeta4_dr(
    func: DriftReactionZeta4Chernoff,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}

fn validate_params(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

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

fn extract_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "u0 must be a numpy.ndarray[float64] or a sequence of floats",
        )
    })
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register `DriftReaction4th1D` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DriftReaction4th1D>()
}
