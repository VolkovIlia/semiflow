//! `Killing2nd1D` — order-2 soft-killing Chernoff for Feynman-Kac `e^{t(L−κ)}`.
//!
//! Wraps `Killing2ndChernoff<DiffusionChernoff<f64>, ConstKappaRate, f64>`
//! (ADR-0126, math.md §21.8). Constant killing rate `κ ≥ 0`.
//!
//! GIL policy: validate + copy under GIL → `py.detach` compute → write under GIL.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value, clippy::unused_self)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    killing_soft::{Killing2ndChernoff, KillingRate},
    ChernoffSemigroup,
};

use crate::{
    boundary::parse_boundary,
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Constant killing rate (no closures → Clone + Copy)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ConstKappaRate(f64);

impl KillingRate<f64> for ConstKappaRate {
    fn kappa(&self, _x: f64) -> f64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Concrete type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type Killing2ndUnit = Killing2ndChernoff<DiffUnit, ConstKappaRate, f64>;

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct Killing2ndInner {
    semigroup: ChernoffSemigroup<Killing2ndUnit, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Killing2nd1D pyclass
// ---------------------------------------------------------------------------

/// Order-2 soft-killing 1-D Chernoff for Feynman-Kac `e^{t(L−κ)}` (ADR-0126, §21.8).
///
/// Solves ``∂_t u = ∂²u − κ·u`` for constant ``κ ≥ 0``.
/// Uses palindromic Strang: ``e^{−τκ/2} · C(τ) · e^{−τκ/2}`` — global order 2.
///
/// Parameters
/// ----------
/// xmin : float
///     Left boundary.
/// xmax : float
///     Right boundary (must be > xmin).
/// n : int
///     Number of grid nodes (must be >= 4).
/// u0 : array-like
///     Initial condition; float64 array of length n, all finite.
/// kappa : float, optional
///     Constant killing rate (must be >= 0; default 0.0).
/// boundary : str, optional
///     Boundary policy (keyword-only). One of ``"reflect"`` (default),
///     ``"periodic"``, ``"zero"``, ``"linear"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if xmin >= xmax, n < 4, or len(u0) != n.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if kappa < 0.
#[pyclass(name = "Killing2nd1D")]
pub struct Killing2nd1D {
    inner: Killing2ndInner,
}

#[pymethods]
impl Killing2nd1D {
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, kappa = 0.0_f64, *, boundary = "reflect"))]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        kappa: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            if !kappa.is_finite() || kappa < 0.0 {
                return Err(new_pyerr("OutOfDomain", "kappa must be finite and >= 0"));
            }
            let policy = parse_boundary(boundary)?;
            let u0_vec: Vec<f64> = u0.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("u0 must be a float64 array")
            })?;
            let inner = build_killing2nd(xmin, xmax, n, kappa, &u0_vec, policy)
                .map_err(|e| from_core(&e))?;
            Ok(Killing2nd1D { inner })
        })
    }

    /// Return the approximation order (always 2 — palindromic Strang).
    fn order(&self) -> u32 {
        2
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// GIL released during inner Rust compute (ADR-0031).
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_params(t, n_steps)?;
            let func = self.inner.semigroup.func.clone();
            let grid = self.inner.current.grid;
            let values: Vec<f64> = self.inner.current.values.clone();
            let func_clone = func.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| compute_killing2nd(func, grid, values, t, n_steps));
            self.inner.current.values = result.map_err(|e| from_core(&e))?;
            let sg = ChernoffSemigroup::new(func_clone, n_steps).map_err(|e| from_core(&e))?;
            self.inner.semigroup = sg;
            Ok(())
        })
    }

    /// Return current grid values as ``numpy.ndarray[float64]`` (copy).
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Number of grid nodes.
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Killing2nd1D(n={}, order=2)",
            self.inner.current.values.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
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

extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero(_: f64) -> f64 {
    0.0
}

fn build_killing2nd(
    xmin: f64,
    xmax: f64,
    n: usize,
    kappa: f64,
    u0: &[f64],
    boundary: semiflow::BoundaryPolicy,
) -> Result<Killing2ndInner, semiflow::SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(semiflow::SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let diff = DiffusionChernoff::new(unit_a, zero, zero, 1.0, grid);
    let rate = ConstKappaRate(kappa);
    let kernel = Killing2ndUnit::new(diff, rate, grid)?;
    let semigroup = ChernoffSemigroup::new(kernel, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Killing2ndInner { semigroup, current })
}

fn compute_killing2nd(
    func: Killing2ndUnit,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let src = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &src)?.values)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Killing2nd1D>()
}
