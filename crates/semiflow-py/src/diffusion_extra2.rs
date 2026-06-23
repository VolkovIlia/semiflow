//! Wave P1 — `TruncatedExp4th1D` (M3) and `Strang1D` (M4).
//!
//! Split from `diffusion_extra.rs` for suckless file-size compliance.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::unused_self)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    ChernoffSemigroup, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, StrangSplit,
    TruncatedExp4thDiffusionChernoff,
};

use crate::boundary::parse_boundary;
use crate::diffusion_extra::{
    extract_f64_vec_de, unit_a_de, validate_params, validate_u0, zero_d_de,
};
use crate::error::from_core;
use crate::panic::catch_panic_py;

// Type aliases
type StrangSplitConcrete = StrangSplit<DiffusionChernoff<f64>, DriftReactionChernoff<f64>>;

struct TruncExp4thInner {
    semigroup: ChernoffSemigroup<TruncatedExp4thDiffusionChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

struct Strang1DInner {
    semigroup: ChernoffSemigroup<StrangSplitConcrete, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// TruncatedExp4th1D (M3)
// ---------------------------------------------------------------------------

/// 1-D diffusion with 4th-order truncated-exp Chernoff kernel.
///
/// Solves ``∂_t u = ∂²u`` (unit diffusion) using
/// ``TruncatedExp4thDiffusionChernoff`` (higher-resolution K=4 stencil, CFL-conditional).
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`CflViolated`' if the CFL condition is violated.
///     kind='`GridMismatch`' / '`NanInf`' / '`OutOfDomain`' for invalid inputs.
#[pyclass(name = "TruncatedExp4th1D")]
pub struct PyTruncatedExp4th1D {
    inner: TruncExp4thInner,
}

#[pymethods]
impl PyTruncatedExp4th1D {
    /// Construct ``TruncatedExp4th1D`` (unit diffusion).
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
                build_trunc_exp4th(xmin, xmax, n, &u0_vec, policy).map_err(|e| from_core(&e))?;
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
                py.detach(|| evolve_trunc_exp4th(func, grid, values, t, n_steps));
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
// Strang1D (M4)
// ---------------------------------------------------------------------------

/// 1-D advection-diffusion via Strang operator splitting.
///
/// Solves ``∂_t u = a·∂²u + b·∂_x u`` using ``StrangSplit<DiffusionChernoff,
/// DriftReactionChernoff>``.  Default: unit diffusion ``a = 1``, drift ``b = 0.5``.
///
/// Uses the palindromic Strang sandwich ``D(τ/2) ∘ R(τ) ∘ D(τ/2)`` (ADR-0006,
/// math.md §9.4, global order 2).
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' / '`NanInf`' / '`OutOfDomain`' for invalid inputs.
#[pyclass(name = "Strang1D")]
pub struct PyStrang1D {
    inner: Strang1DInner,
}

#[pymethods]
impl PyStrang1D {
    /// Construct ``Strang1D``.
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, b = 0.5, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        b: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let u0_vec = extract_f64_vec_de(u0)?;
            let inner =
                build_strang1d(xmin, xmax, n, &u0_vec, b, policy).map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the global approximation order (always 2 for Strang splitting).
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
                py.detach(|| evolve_strang(func, grid, values, t, n_steps));
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

fn build_trunc_exp4th(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    boundary: semiflow::BoundaryPolicy,
) -> Result<TruncExp4thInner, semiflow::SemiflowError> {
    validate_u0(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let trunc4 = TruncatedExp4thDiffusionChernoff::new(unit_a_de, zero_d_de, zero_d_de, 1.0, grid);
    let semigroup = ChernoffSemigroup::new(trunc4, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(TruncExp4thInner { semigroup, current })
}

fn build_strang1d(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    b_const: f64,
    boundary: semiflow::BoundaryPolicy,
) -> Result<Strang1DInner, semiflow::SemiflowError> {
    validate_u0(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let diff = DiffusionChernoff::new(unit_a_de, zero_d_de, zero_d_de, 1.0, grid);
    let drift =
        DriftReactionChernoff::with_closure(move |_| b_const, |_| 0.0_f64, b_const.abs(), grid);
    let split = StrangSplit::new(diff, drift);
    let semigroup = ChernoffSemigroup::new(split, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Strang1DInner { semigroup, current })
}

// ---------------------------------------------------------------------------
// GIL-free compute helpers
// ---------------------------------------------------------------------------

fn evolve_trunc_exp4th(
    func: TruncatedExp4thDiffusionChernoff<f64>,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}

fn evolve_strang(
    func: StrangSplitConcrete,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}
