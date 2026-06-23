//! Wave P3 — `Reflected1D` (M9), `Robin1D` (M10), `DirichletHeat2nd1D` (M11).
//!
//! Split from `bc_kernels.rs` for suckless file-size compliance.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::too_many_arguments, clippy::unused_self)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    killing_order2::DirichletHeat2ndChernoff,
    reflection::{HalfSpaceRegion, ReflectedHeatChernoff},
    robin::{HalfSpaceRobin, RobinHeatChernoff},
    ChernoffSemigroup, InterpKind,
};

use crate::{
    bc_kernels::{extract_f64_vec_bc, unit_a_bc, validate_params, validate_u0, zero_bc},
    error::from_core,
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type ReflectedKernel = ReflectedHeatChernoff<DiffUnit, HalfSpaceRegion<f64, 1>, f64>;
type RobinKernel = RobinHeatChernoff<DiffUnit, HalfSpaceRobin<f64, 1>, f64>;
type DirichletHeat2ndKernel = DirichletHeat2ndChernoff<DiffUnit, HalfSpaceRegion<f64, 1>, f64>;

// ---------------------------------------------------------------------------
// Reflected1D inner state
// ---------------------------------------------------------------------------

struct Reflected1DInner {
    semigroup: ChernoffSemigroup<ReflectedKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Reflected1D pyclass (M9)
// ---------------------------------------------------------------------------

/// 1-D heat equation with Neumann BC via the image method (M9).
///
/// Solves ``∂_t u = ∂²u`` on the half-line ``[0, xmax]`` with
/// Neumann zero-flux condition ``∂_x u = 0`` at ``x = origin``.
/// Backed by ``ReflectedHeatChernoff<DiffusionChernoff, HalfSpaceRegion>``
/// (Walsh 1986 image method, order 2; math.md §25).
///
/// Parameters
/// ----------
/// xmin : float
///     Left domain boundary.
/// xmax : float
///     Right domain boundary.
/// n : int
///     Number of grid nodes (must be >= 4).
/// u0 : array-like
///     Initial condition; float64 array of length n.
/// origin : float, optional
///     Point on the reflecting boundary (default = xmin).
/// boundary : str, optional
///     Boundary policy for the inner diffusion kernel; default ``"reflect"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if xmin >= xmax or n < 4 or len(u0) != n.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if boundary is unrecognised.
#[pyclass(name = "Reflected1D")]
pub struct PyReflected1D {
    inner: Reflected1DInner,
}

#[pymethods]
impl PyReflected1D {
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, origin = f64::NAN, boundary = "reflect"))]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        origin: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = crate::boundary::parse_boundary(boundary)?;
            let u0_vec = extract_f64_vec_bc(u0)?;
            let origin_eff = if origin.is_finite() { origin } else { xmin };
            let inner = build_reflected(xmin, xmax, n, &u0_vec, origin_eff, policy)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (inherits inner order = 2).
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
                py.detach(|| evolve_reflected(func, grid, values, t, n_steps));
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

fn build_reflected(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    origin: f64,
    boundary: semiflow::BoundaryPolicy,
) -> Result<Reflected1DInner, semiflow::SemiflowError> {
    validate_u0(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
    let region = HalfSpaceRegion::<f64, 1>::new([origin], [1.0])?;
    let reflected = ReflectedHeatChernoff::new(diff, region)?;
    let semigroup = ChernoffSemigroup::new(reflected, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Reflected1DInner { semigroup, current })
}

fn evolve_reflected(
    func: ReflectedKernel,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}

// ===========================================================================
// Robin1D
// ===========================================================================

// ---------------------------------------------------------------------------
// Robin1D inner state
// ---------------------------------------------------------------------------

struct Robin1DInner {
    semigroup: ChernoffSemigroup<RobinKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Robin1D pyclass (M10)
// ---------------------------------------------------------------------------

/// 1-D heat equation with Robin BC via the skew image method (M10).
///
/// Solves ``∂_t u = ∂²u`` with Robin BC ``α·u(0) − β·∂_x u(0) = 0``
/// at the left boundary ``x = origin``.
/// Backed by ``RobinHeatChernoff<DiffusionChernoff, HalfSpaceRobin>``
/// (Carslaw-Jaeger 1959 §14.2, Walsh 1986, order 1; math.md §3.5.tris).
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
/// alpha : float, optional
///     Robin coefficient on u (default 1.0); must be >= 0.
/// beta : float, optional
///     Robin coefficient on ∂_n u (default 1.0); must be > 0.
/// origin : float, optional
///     Point on the Robin boundary (default = xmin).
/// boundary : str, optional
///     Boundary policy for the inner kernel; default ``"reflect"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if xmin >= xmax or n < 4 or len(u0) != n.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if alpha < 0, beta <= 0, or boundary unrecognised.
#[pyclass(name = "Robin1D")]
pub struct PyRobin1D {
    inner: Robin1DInner,
}

#[pymethods]
impl PyRobin1D {
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, alpha = 1.0, beta = 1.0, origin = f64::NAN, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        alpha: f64,
        beta: f64,
        origin: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = crate::boundary::parse_boundary(boundary)?;
            let u0_vec = extract_f64_vec_bc(u0)?;
            let origin_eff = if origin.is_finite() { origin } else { xmin };
            let inner = build_robin(xmin, xmax, n, &u0_vec, alpha, beta, origin_eff, policy)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (always 1 for Robin/skew-image).
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
                py.detach(|| evolve_robin(func, grid, values, t, n_steps));
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
// Builder
// ---------------------------------------------------------------------------

fn build_robin(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    alpha: f64,
    beta: f64,
    origin: f64,
    boundary: semiflow::BoundaryPolicy,
) -> Result<Robin1DInner, semiflow::SemiflowError> {
    validate_u0(u0)?;
    // RobinHeatChernoff::apply_into calls reflect_in_place which uses
    // sample_generic.  SepticHermite is unsupported in the generic path
    // (Grid1D<F>::interp_generic, Grid1D::new f64 default = SepticHermite).
    // Downgrade to CubicHermite so interp_generic can service the ghost calls.
    let grid = Grid1D::new(xmin, xmax, n)?
        .with_interp(InterpKind::CubicHermite)
        .with_boundary(boundary);
    let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
    // Normal points inward (positive x direction).
    let region = HalfSpaceRobin::<f64, 1>::new([origin], [1.0], alpha, beta)?;
    let robin = RobinHeatChernoff::new(diff, region)?;
    let semigroup = ChernoffSemigroup::new(robin, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Robin1DInner { semigroup, current })
}

// ---------------------------------------------------------------------------
// GIL-free compute helper
// ---------------------------------------------------------------------------

fn evolve_robin(
    func: RobinKernel,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}

// ===========================================================================
// DirichletHeat2nd1D — order-2 Dirichlet absorbing BC via odd-image method
// ===========================================================================

// ---------------------------------------------------------------------------
// DirichletHeat2nd1D inner state
// ---------------------------------------------------------------------------

struct DirichletHeat2nd1DInner {
    semigroup: ChernoffSemigroup<DirichletHeat2ndKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// DirichletHeat2nd1D pyclass (M11)
// ---------------------------------------------------------------------------

/// 1-D heat equation with Dirichlet BC via the odd image method (M11).
///
/// Solves ``∂_t u = ∂²u`` on the half-line ``[0, xmax]`` with
/// absorbing Dirichlet condition ``u = 0`` at ``x = origin``.
/// Backed by ``DirichletHeat2ndChernoff<DiffusionChernoff, HalfSpaceRegion>``
/// (odd image method, order 2; math.md §21.9, ADR-0176).
///
/// This is the order-2 companion of ``Killing1D`` (order 1) and the
/// Dirichlet mirror of ``Reflected1D`` (Neumann, order 2, §25).
///
/// Parameters
/// ----------
/// xmin : float
///     Left domain boundary.
/// xmax : float
///     Right domain boundary.
/// n : int
///     Number of grid nodes (must be >= 4).
/// u0 : array-like
///     Initial condition; float64 array of length n.
/// origin : float, optional
///     Point on the absorbing boundary (default = xmin).
/// boundary : str, optional
///     Boundary policy for the inner diffusion kernel; default ``"reflect"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if xmin >= xmax or n < 4 or len(u0) != n.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if boundary is unrecognised.
///
/// Notes
/// -----
/// The odd ghost subtracts mass, so the solution does NOT preserve
/// non-negativity.  This is physically correct: an absorbing wall removes mass.
#[pyclass(name = "DirichletHeat2nd1D")]
pub struct PyDirichletHeat2nd1D {
    inner: DirichletHeat2nd1DInner,
}

#[pymethods]
impl PyDirichletHeat2nd1D {
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, origin = f64::NAN, boundary = "reflect"))]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        origin: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = crate::boundary::parse_boundary(boundary)?;
            let u0_vec = extract_f64_vec_bc(u0)?;
            let origin_eff = if origin.is_finite() { origin } else { xmin };
            let inner = build_dirichlet_heat2nd(xmin, xmax, n, &u0_vec, origin_eff, policy)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (inherits inner order = 2).
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
                py.detach(|| evolve_dirichlet_heat2nd(func, grid, values, t, n_steps));
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

fn build_dirichlet_heat2nd(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    origin: f64,
    boundary: semiflow::BoundaryPolicy,
) -> Result<DirichletHeat2nd1DInner, semiflow::SemiflowError> {
    validate_u0(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
    let region = HalfSpaceRegion::<f64, 1>::new([origin], [1.0])?;
    let kernel = DirichletHeat2ndChernoff::new(diff, region)?;
    let semigroup = ChernoffSemigroup::new(kernel, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(DirichletHeat2nd1DInner { semigroup, current })
}

fn evolve_dirichlet_heat2nd(
    func: DirichletHeat2ndKernel,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}
