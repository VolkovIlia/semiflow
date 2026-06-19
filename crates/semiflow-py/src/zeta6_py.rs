//! `PyO3` wrapper for `Diffusion6thZeta6Chernoff` (v4.1 Phase D, Item 4).
//!
//! Exposes `Heat1DZeta6` — the order-6-temporal ζ⁶ heat kernel Python class.
//! Builds the chain: `Diffusion4thChernoff → Diffusion4thZeta4Chernoff →
//! Diffusion6thZeta6Chernoff` with unit diffusion coefficient `a = 1`.
//!
//! ADR-0028: f64-only; `a_kth_bound = Some(1.0)` for unit `a`.
//! v7.0: `QuinticHermite` removed (ADR-0109 clock); ζ⁶ uses `CubicHermite` K5 default.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::unused_self)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow_core::{
    ChernoffSemigroup, Diffusion4thChernoff, Diffusion4thZeta4Chernoff, Diffusion6thZeta6Chernoff,
    Grid1D, GridFn1D,
};

use crate::boundary::parse_boundary;
use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct Zeta6Inner {
    semigroup: ChernoffSemigroup<Diffusion6thZeta6Chernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Heat1DZeta6 pyclass
// ---------------------------------------------------------------------------

/// 1-D heat equation with order-6-temporal ζ⁶ Chernoff kernel (v4.1).
///
/// Solves ``∂_t u = ∂²u`` (unit diffusion ``a = 1``) using the
/// ``Diffusion6thZeta6Chernoff`` kernel (order-6 temporal, path β-ladder
/// rung K=3; ADR-0086 / ADR-0089 AMENDMENT 1).
///
/// The inner ζ⁴ uses default `CubicHermite` spatial sampling.
/// (`QuinticHermite` removed at v7.0; use `with_chebyshev_sampling` for higher accuracy.)
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
#[pyclass(name = "Heat1DZeta6")]
pub struct PyHeat1DZeta6 {
    inner: Zeta6Inner,
}

#[pymethods]
impl PyHeat1DZeta6 {
    /// Create a new ``Heat1DZeta6`` state with unit diffusion coefficient.
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
                build_zeta6_unit(xmin, xmax, n, &u0_vec, policy).map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (always 6 for ζ⁶ kernel).
    ///
    /// Returns
    /// -------
    /// int
    ///     Approximation order = 6.
    fn order(&self) -> u32 {
        6
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// The GIL is released during the inner Rust compute loop (ADR-0031).
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time to advance.  Must be non-negative and finite.
    /// `n_steps` : int, optional
    ///     Number of Chernoff steps (default 100).  Must be >= 1.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if t < 0, t is non-finite, or `n_steps` == 0.
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_params(t, n_steps)?;
            let func = self.inner.semigroup.func.clone();
            let grid = self.inner.current.grid;
            let values: Vec<f64> = self.inner.current.values.clone();
            let func_clone = func.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| compute_evolve(func, grid, values, t, n_steps));
            self.inner.current.values = result.map_err(|e| from_core(&e))?;
            let sg = ChernoffSemigroup::new(func_clone, n_steps).map_err(|e| from_core(&e))?;
            self.inner.semigroup = sg;
            Ok(())
        })
    }

    /// Return the current grid values as a 1-D ``numpy.ndarray[float64]``.
    ///
    /// Returns a copy; mutations do not affect the internal state.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Return the number of grid nodes.
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_d(_: f64) -> f64 {
    0.0
}

/// Build `Zeta6Inner` for unit diffusion.
fn build_zeta6_unit(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    boundary: semiflow_core::BoundaryPolicy,
) -> Result<Zeta6Inner, semiflow_core::SemiflowError> {
    validate_u0(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    // Chain: Diffusion4thChernoff → Diffusion4thZeta4Chernoff → Diffusion6thZeta6Chernoff
    let d4 = Diffusion4thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
    let zeta4 = Diffusion4thZeta4Chernoff::new(d4, Some(1.0_f64))?;
    let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, Some(1.0_f64))?;
    let semigroup = ChernoffSemigroup::new(zeta6, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Zeta6Inner { semigroup, current })
}

/// Evolve `Diffusion6thZeta6Chernoff` for `n_steps` steps. GIL-free.
fn compute_evolve(
    func: Diffusion6thZeta6Chernoff<f64>,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_params(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn validate_u0(u0: &[f64]) -> Result<(), semiflow_core::SemiflowError> {
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

/// Register ζ⁶ pyclasses into the `semiflow` module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyHeat1DZeta6>()?;
    Ok(())
}
