//! Wave P1 — 1-D diffusion completeness: `Heat1DZeta8`, `TruncatedExp1D`,
//! `TruncatedExp4th1D`, and `Strang1D`.
//!
//! All four pyclasses follow the ADR-0111 contract (per-type pyclass, three-phase
//! GIL release, numpy float64, `SemiflowError` mapping, abi3).
//!
//! ## Kernels
//!
//! | pyclass | Core type | Notes |
//! |---------|-----------|-------|
//! | `Heat1DZeta8` | `Diffusion8thZeta8Chernoff` | M1; order-8-temporal, Chebyshev default |
//! | `TruncatedExp1D` | `TruncatedExpDiffusionChernoff` | M2; K=4 truncated series, unit a=1 |
//! | `TruncatedExp4th1D` | `TruncatedExp4thDiffusionChernoff` | M3; same stencil, higher resolution |
//! | `Strang1D` | `StrangSplit<DiffusionChernoff, DriftReactionChernoff>` | M4; D(τ/2)∘R(τ)∘D(τ/2) |
//!
//! ## GIL policy
//!
//! All `evolve` methods: validate + copy under GIL → `py.detach` compute → update under GIL.
//! `Send+Sync` proofs are added to `send_assertions.rs`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::unused_self)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow_core::{
    ChernoffSemigroup, Diffusion4thChernoff, Diffusion4thZeta4Chernoff, Diffusion6thZeta6Chernoff,
    Diffusion8thZeta8Chernoff, Grid1D, GridFn1D, TruncatedExpDiffusionChernoff,
};

// Re-export from diffusion_extra2 for registration.
pub(crate) use crate::diffusion_extra2::{PyStrang1D, PyTruncatedExp4th1D};

use crate::boundary::parse_boundary;
use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;

// ---------------------------------------------------------------------------
// Shared fn-pointer constants (unit coefficients)
// ---------------------------------------------------------------------------

pub(crate) extern "Rust" fn unit_a_de(_: f64) -> f64 {
    1.0
}
pub(crate) extern "Rust" fn zero_d_de(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Inner state types
// ---------------------------------------------------------------------------

struct Zeta8Inner {
    semigroup: ChernoffSemigroup<Diffusion8thZeta8Chernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

struct TruncExpInner {
    semigroup: ChernoffSemigroup<TruncatedExpDiffusionChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Heat1DZeta8
// ---------------------------------------------------------------------------

/// 1-D heat equation with order-8-temporal ζ⁸ Chernoff kernel (v6.0.0).
///
/// Solves ``∂_t u = ∂²u`` (unit diffusion) using ``Diffusion8thZeta8Chernoff``
/// (order-8 temporal via nested Richardson; ADR-0088 Wave II, Chebyshev ON by default).
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if grid is invalid or len(u0) != n.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if boundary or `n_steps` invalid.
#[pyclass(name = "Heat1DZeta8")]
pub struct PyHeat1DZeta8 {
    inner: Zeta8Inner,
}

#[pymethods]
impl PyHeat1DZeta8 {
    /// Construct ``Heat1DZeta8`` (unit diffusion, Chebyshev ON).
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
            let u0_vec = extract_f64_vec_de(u0)?;
            let inner = build_zeta8(xmin, xmax, n, &u0_vec, policy).map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (always 8 for ζ⁸ kernel).
    fn order(&self) -> u32 {
        8
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
                py.detach(|| evolve_zeta8(func, grid, values, t, n_steps));
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
}

// ---------------------------------------------------------------------------
// TruncatedExp1D
// ---------------------------------------------------------------------------

/// 1-D diffusion with K=4 truncated-exp Chernoff kernel.
///
/// Solves ``∂_t u = ∂²u`` (unit diffusion ``a = 1``) using
/// ``TruncatedExpDiffusionChernoff`` (divergence-form stencil, CFL-conditional).
///
/// The CFL condition ``2·τ·‖a‖_∞ < dx²`` is checked on every evolve call.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`CflViolated`' if the CFL condition is violated.
///     kind='`GridMismatch`' / '`NanInf`' / '`OutOfDomain`' for invalid inputs.
#[pyclass(name = "TruncatedExp1D")]
pub struct PyTruncatedExp1D {
    inner: TruncExpInner,
}

#[pymethods]
impl PyTruncatedExp1D {
    /// Construct ``TruncatedExp1D`` (unit diffusion).
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
            let u0_vec = extract_f64_vec_de(u0)?;
            let inner =
                build_trunc_exp(xmin, xmax, n, &u0_vec, policy).map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the kernel approximation order (always 2).
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
            let func = self.inner.semigroup.func;
            let grid = self.inner.current.grid;
            let values: Vec<f64> = self.inner.current.values.clone();
            let func_clone = func;
            let result: Result<Vec<f64>, _> =
                py.detach(|| evolve_trunc_exp(func, grid, values, t, n_steps));
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
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

fn build_zeta8(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    boundary: semiflow_core::BoundaryPolicy,
) -> Result<Zeta8Inner, semiflow_core::SemiflowError> {
    validate_u0(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let k5 = Diffusion4thChernoff::new(unit_a_de, zero_d_de, zero_d_de, 1.0, grid);
    let zeta4 = Diffusion4thZeta4Chernoff::new(k5, Some(1.0_f64))?;
    let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, Some(1.0_f64))?;
    let zeta8 = Diffusion8thZeta8Chernoff::new(zeta6, Some(1.0_f64))?;
    let semigroup = ChernoffSemigroup::new(zeta8, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Zeta8Inner { semigroup, current })
}

fn build_trunc_exp(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    boundary: semiflow_core::BoundaryPolicy,
) -> Result<TruncExpInner, semiflow_core::SemiflowError> {
    validate_u0(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let trunc = TruncatedExpDiffusionChernoff::new(unit_a_de, zero_d_de, zero_d_de, 1.0, grid);
    let semigroup = ChernoffSemigroup::new(trunc, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(TruncExpInner { semigroup, current })
}

// ---------------------------------------------------------------------------
// GIL-free compute helpers (called inside py.detach)
// ---------------------------------------------------------------------------

fn evolve_zeta8(
    func: Diffusion8thZeta8Chernoff<f64>,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}

fn evolve_trunc_exp(
    func: TruncatedExpDiffusionChernoff<f64>,
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

pub(crate) fn validate_params(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

pub(crate) fn validate_u0(u0: &[f64]) -> Result<(), semiflow_core::SemiflowError> {
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

pub(crate) fn extract_f64_vec_de(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "u0 must be a numpy.ndarray[float64] or a sequence of floats",
        )
    })
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register Wave P1 pyclasses into the `semiflow` module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyHeat1DZeta8>()?;
    m.add_class::<PyTruncatedExp1D>()?;
    m.add_class::<PyTruncatedExp4th1D>()?;
    m.add_class::<PyStrang1D>()?;
    Ok(())
}
