//! Wave P3 — boundary-condition kernels (`bc_kernels.rs`).
//!
//! Exposes four boundary-condition Chernoff families as Python classes:
//!
//! | pyclass | Core type | M# |
//! |---------|-----------|-----|
//! | `Resolvent1D` | `LaplaceChernoffResolvent<DiffusionChernoff<f64>, f64>` | M7 |
//! | `Killing1D` | `KillingChernoff<DiffusionChernoff<f64>, BoxRegion<f64, 1>>` | M8 |
//! | `Reflected1D` | `ReflectedHeatChernoff<DiffusionChernoff<f64>, HalfSpaceRegion<f64, 1>>` | M9 |
//! | `Robin1D` | `RobinHeatChernoff<DiffusionChernoff<f64>, HalfSpaceRobin<f64, 1>>` | M10 |
//!
//! Region types bind as constructor kwargs (string/tuple), not as separate pyclasses
//! (ADR-0111 §3 — suckless minimal surface).  `Resolvent1D` exposes the residual
//! gate value as a `residual()` method (ADR-0083).
//!
//! ## GIL policy
//!
//! All `evolve`/`eval` methods: validate + copy under GIL → `py.detach` compute
//! → write result under GIL (ADR-0031 three-phase pattern).
//!
//! ## Send+Sync
//!
//! All four concrete kernel types are `f64` fn-pointer / scalar structs — auto
//! `Send+Sync`.  Assertions added to `send_assertions.rs`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::unused_self
)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    killing::{BoxRegion, KillingChernoff},
    resolvent::{LaplaceChernoffResolvent, LaplaceQuadrature},
    ChernoffSemigroup,
};

use crate::{
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// Re-export bc_kernels2 pyclasses for registration.
pub(crate) use crate::bc_kernels2::PyDirichletHeat2nd1D;
pub(crate) use crate::bc_kernels2::PyReflected1D;
pub(crate) use crate::bc_kernels2::PyRobin1D;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

pub(crate) extern "Rust" fn unit_a_bc(_: f64) -> f64 {
    1.0
}
pub(crate) extern "Rust" fn zero_bc(_: f64) -> f64 {
    0.0
}

pub(crate) fn validate_params(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

pub(crate) fn validate_u0(u0: &[f64]) -> Result<(), semiflow::SemiflowError> {
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

pub(crate) fn extract_f64_vec_bc(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "u0 must be a numpy.ndarray[float64] or a sequence of floats",
        )
    })
}

// ---------------------------------------------------------------------------
// Type aliases for concrete kernel combinations used in this module
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type ResolventKernel = LaplaceChernoffResolvent<DiffUnit, f64>;
type KillingKernel = KillingChernoff<DiffUnit, BoxRegion<f64, 1>, f64>;
// ---------------------------------------------------------------------------
// Resolvent1D inner state
// ---------------------------------------------------------------------------

struct Resolvent1DInner {
    resolvent: ResolventKernel,
    grid: Grid1D<f64>,
}

// ---------------------------------------------------------------------------
// Killing1D inner state
// ---------------------------------------------------------------------------

struct Killing1DInner {
    semigroup: ChernoffSemigroup<KillingKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Resolvent1D
// ---------------------------------------------------------------------------

/// 1-D Laplace-Chernoff resolvent ``(λI − ∂²)⁻¹ g`` (M7).
///
/// Computes ``R̃(λ) g = ∫₀^∞ exp(−λt) S(t)g dt`` via Gauss-Laguerre-32
/// quadrature (Remizov 2025, Vladikavkaz Math. J. 27(4) Theorem 3).
///
/// Parameters
/// ----------
/// xmin : float
///     Left boundary.
/// xmax : float
///     Right boundary (must be > xmin).
/// n : int
///     Number of grid nodes (must be >= 4).
/// `n_chernoff` : int, optional
///     Chernoff truncation level for the inner heat semigroup (default 32).
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if xmin >= xmax or n < 4.
///     kind='`OutOfDomain`' if `n_chernoff` == 0.
#[pyclass(name = "Resolvent1D")]
pub struct PyResolvent1D {
    inner: Resolvent1DInner,
}

#[pymethods]
impl PyResolvent1D {
    /// Construct ``Resolvent1D`` (unit diffusion, Gauss-Laguerre-32).
    #[new]
    #[pyo3(signature = (xmin, xmax, n, *, n_chernoff = 32))]
    fn new(xmin: f64, xmax: f64, n: usize, n_chernoff: usize) -> PyResult<Self> {
        catch_panic_py!({
            let inner = build_resolvent(xmin, xmax, n, n_chernoff).map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Evaluate ``R̃(lambda) g`` and return the result as float64 array.
    ///
    /// GIL released during inner Rust compute (ADR-0031).
    ///
    /// Parameters
    /// ----------
    /// lambda : float
    ///     Resolvent parameter; must be > 0 and finite.
    /// g : array-like
    ///     Right-hand side; float64 array of length n.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if lambda <= 0 or non-finite.
    ///     kind='`GridMismatch`' if len(g) != n.
    ///     kind='`NanInf`' if g contains NaN or Inf.
    fn eval<'py>(
        &self,
        py: Python<'py>,
        lambda: f64,
        g: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            let g_vec = extract_f64_vec_bc(g)?;
            validate_u0(&g_vec).map_err(|e| from_core(&e))?;
            let grid = self.inner.grid;
            let resolvent = self.inner.resolvent.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| eval_resolvent(resolvent, grid, g_vec, lambda));
            let values = result.map_err(|e| from_core(&e))?;
            Ok(values.as_slice().to_pyarray(py))
        })
    }

    /// Compute the residual ``‖(λI − ∂²) R̃(λ) g − g‖_∞`` for validation.
    ///
    /// Uses 3-point FD Laplacian on interior nodes.
    /// GIL released during compute (ADR-0031).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if lambda <= 0 or n < 3.
    fn residual(&self, py: Python<'_>, lambda: f64, g: &Bound<'_, PyAny>) -> PyResult<f64> {
        catch_panic_py!({
            let g_vec = extract_f64_vec_bc(g)?;
            validate_u0(&g_vec).map_err(|e| from_core(&e))?;
            let grid = self.inner.grid;
            let resolvent = self.inner.resolvent.clone();
            let result: Result<f64, _> =
                py.detach(|| compute_residual(resolvent, grid, g_vec, lambda));
            result.map_err(|e| from_core(&e))
        })
    }

    /// Number of grid nodes.
    fn __len__(&self) -> usize {
        self.inner.grid.n
    }
}

// ---------------------------------------------------------------------------
// Killing1D
// ---------------------------------------------------------------------------

/// 1-D heat equation with absorbing Dirichlet BC via Feynman-Kac killing (M8).
///
/// Solves ``∂_t u = ∂²u`` with ``u = 0`` outside the box ``[lo, hi)``.
/// Backed by ``KillingChernoff<DiffusionChernoff, BoxRegion>`` (Butko 2018,
/// order 1 globally; math.md §21).
///
/// Parameters
/// ----------
/// xmin : float
///     Left domain boundary.
/// xmax : float
///     Right domain boundary (must be > xmin).
/// n : int
///     Number of grid nodes (must be >= 4).
/// u0 : array-like
///     Initial condition; float64 array of length n.
/// lo : float, optional
///     Lower bound of the killing-region box (default = xmin + 1/4 of range).
/// hi : float, optional
///     Upper bound of the killing-region box, exclusive (default = xmax - 1/4).
/// boundary : str, optional
///     Boundary policy (keyword-only); default ``"reflect"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if xmin >= xmax or n < 4 or len(u0) != n.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if lo >= hi or boundary is unrecognised.
#[pyclass(name = "Killing1D")]
pub struct PyKilling1D {
    inner: Killing1DInner,
}

#[pymethods]
impl PyKilling1D {
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, lo = f64::NAN, hi = f64::NAN, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        lo: f64,
        hi: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = crate::boundary::parse_boundary(boundary)?;
            let u0_vec = extract_f64_vec_bc(u0)?;
            // Default: box = middle half of domain.
            let range = xmax - xmin;
            let lo_eff = if lo.is_finite() {
                lo
            } else {
                xmin + range * 0.25
            };
            let hi_eff = if hi.is_finite() {
                hi
            } else {
                xmax - range * 0.25
            };
            let inner = build_killing(xmin, xmax, n, &u0_vec, lo_eff, hi_eff, policy)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (always 1 for Killing/Feynman-Kac).
    fn order(&self) -> u32 {
        1
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
                py.detach(|| evolve_killing(func, grid, values, t, n_steps));
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

fn build_resolvent(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
) -> Result<Resolvent1DInner, semiflow::SemiflowError> {
    let grid = Grid1D::new(xmin, xmax, n)?;
    let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
    let resolvent =
        LaplaceChernoffResolvent::new(diff, n_chernoff, LaplaceQuadrature::GaussLaguerre32)?;
    Ok(Resolvent1DInner { resolvent, grid })
}

fn build_killing(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    lo: f64,
    hi: f64,
    boundary: semiflow::BoundaryPolicy,
) -> Result<Killing1DInner, semiflow::SemiflowError> {
    validate_u0(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
    let region = BoxRegion::<f64, 1>::new([lo], [hi])?;
    let killing = KillingChernoff::new(diff, region)?;
    let semigroup = ChernoffSemigroup::new(killing, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Killing1DInner { semigroup, current })
}

// ---------------------------------------------------------------------------
// GIL-free compute helpers (called inside py.detach)
// ---------------------------------------------------------------------------

fn eval_resolvent(
    resolvent: ResolventKernel,
    grid: Grid1D<f64>,
    g_vec: Vec<f64>,
    lambda: f64,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let g = GridFn1D::new(grid, g_vec)?;
    Ok(resolvent.eval(lambda, &g)?.values)
}

fn compute_residual(
    resolvent: ResolventKernel,
    grid: Grid1D<f64>,
    g_vec: Vec<f64>,
    lambda: f64,
) -> Result<f64, semiflow::SemiflowError> {
    use semiflow::resolvent::LaplaceChernoffResolventResidual;
    let g = GridFn1D::new(grid, g_vec)?;
    let residual_gate = LaplaceChernoffResolventResidual::new(resolvent, 1e-2);
    residual_gate.verify_residual(lambda, &g)
}

fn evolve_killing(
    func: KillingKernel,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register Wave P3 pyclasses into the `semiflow` module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyResolvent1D>()?;
    m.add_class::<PyKilling1D>()?;
    m.add_class::<PyReflected1D>()?;
    m.add_class::<PyRobin1D>()?;
    m.add_class::<PyDirichletHeat2nd1D>()?;
    Ok(())
}
