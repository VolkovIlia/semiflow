//! `MatrixDiffusion2D` and `MatrixDiffusion3D` — coupled 2-component
//! 2D/3D diffusion via palindromic Strang splitting (ADR-0124).
//!
//! Mirrors `MatrixDiffusion1D` (`structured_matrix.rs`) for higher dimensions.
//!
//! ## Buffer layout (2D)
//!
//! Flat `2*nx*ny` buffer: `buf[(j*nx+i)*2+c]` where j=y-index, i=x-index, c∈{0,1}.
//!
//! ## Buffer layout (3D)
//!
//! Flat `2*nx*ny*nz` buffer: `buf[(k*nx*ny+j*nx+i)*2+c]`.

// Binding layer.
#![allow(clippy::needless_pass_by_value, clippy::unused_self, clippy::too_many_arguments)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    matrix_2d3d::{MatrixDiffusionChernoff2D, MatrixGridFn2D},
    matrix_system::MatrixDiffusionChernoff,
    ChernoffSemigroup, Grid1D, Grid2D,
};

use crate::{
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
    structured_matrix::build_matrix_kernel,
};

// ===========================================================================
// MatrixDiffusion2D — coupled 2-component 2D diffusion
// ===========================================================================

/// Coupled 2-component 2D diffusion via palindromic Strang (ADR-0124, §33.2).
///
/// Solves ``∂_t u = a·∂²u + c·u`` (M=2) on ``[xmin,xmax]×[ymin,ymax]``.
/// Palindromic 3-leg Strang: ``Lx(τ/2) Ly(τ) Lx(τ/2)``; order 2.
///
/// Buffer layout: flat float64 of length ``2*nx*ny``.
/// Index ``(j*nx+i)*2+c`` for x-index ``i``, y-index ``j``, component ``c``.
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     X-axis domain.
/// nx : int
///     X-axis nodes (>= 5).
/// ymin, ymax : float
///     Y-axis domain.
/// ny : int
///     Y-axis nodes (>= 5).
/// `a_diag` : float
///     Diagonal diffusion ``a_00=a_11`` (default 1.0, must be > 0).
/// `c_coupling` : float
///     Off-diagonal reaction ``c_01=c_10`` (default 0.0).
/// u0 : array-like
///     Initial condition; float64, length ``2*nx*ny``.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='GridMismatch' if grid params invalid or u0 length wrong.
///     kind='NanInf' if u0 contains NaN or Inf.
///     kind='OutOfDomain' if nx<5 or ny<5 or a_diag<=0.
#[pyclass(name = "MatrixDiffusion2D")]
pub struct MatrixDiffusion2D {
    a_diag: f64,
    c_coupling: f64,
    grid2d: Grid2D<f64>,
    current: Vec<f64>,
    nx: usize,
    ny: usize,
}

#[pymethods]
impl MatrixDiffusion2D {
    #[new]
    #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, u0, *, a_diag = 1.0_f64, c_coupling = 0.0_f64))]
    fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        u0: &Bound<'_, PyAny>,
        a_diag: f64,
        c_coupling: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            validate_2d_args(nx, ny, a_diag)?;
            let vals: Vec<f64> = u0.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("u0 must be a float64 array")
            })?;
            validate_u0_2d(&vals, nx, ny)?;
            let grid2d = build_grid2d(xmin, xmax, nx, ymin, ymax, ny)
                .map_err(|e| from_core(&e))?;
            // Eagerly validate kernel construction.
            build_matrix_kernel(a_diag, c_coupling, grid2d.x).map_err(|e| from_core(&e))?;
            Ok(MatrixDiffusion2D { a_diag, c_coupling, grid2d, current: vals, nx, ny })
        })
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations (GIL released).
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_t_nsteps(t, n_steps)?;
            let a = self.a_diag;
            let c = self.c_coupling;
            let g = self.grid2d;
            let vals = self.current.clone();
            let result: Result<Vec<f64>, semiflow::SemiflowError> =
                py.detach(|| evolve_2d(a, c, g, vals, t, n_steps));
            self.current = result.map_err(|e| from_core(&e))?;
            Ok(())
        })
    }

    /// Return current state as flat ``numpy.ndarray[float64]`` of length ``2*nx*ny``.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.current.as_slice().to_pyarray(py)) })
    }

    /// Return approximation order (always 2).
    fn order(&self) -> u32 {
        2
    }

    fn __repr__(&self) -> String {
        format!(
            "MatrixDiffusion2D(nx={}, ny={}, M=2, a_diag={:.3}, c_coupling={:.3}, order=2)",
            self.nx, self.ny, self.a_diag, self.c_coupling
        )
    }
}

// ===========================================================================
// MatrixDiffusion3D — coupled 2-component 3D diffusion
// ===========================================================================

use semiflow::{
    matrix_2d3d::{MatrixDiffusionChernoff3D, MatrixGridFn3D},
    Grid3D,
};

/// Coupled 2-component 3D diffusion via palindromic 5-leg Strang (ADR-0124, §33.3).
///
/// Buffer layout: flat float64 of length ``2*nx*ny*nz``.
/// Index ``(k*nx*ny+j*nx+i)*2+c`` for z-index ``k``, y-index ``j``,
/// x-index ``i``, component ``c``.
///
/// Parameters
/// ----------
/// xmin, xmax, nx : float, float, int
///     X-axis.
/// ymin, ymax, ny : float, float, int
///     Y-axis.
/// zmin, zmax, nz : float, float, int
///     Z-axis.
/// `a_diag` : float
///     Diagonal diffusion (default 1.0, must be > 0).
/// `c_coupling` : float
///     Off-diagonal reaction (default 0.0).
/// u0 : array-like
///     Initial condition; float64, length ``2*nx*ny*nz``.
#[pyclass(name = "MatrixDiffusion3D")]
pub struct MatrixDiffusion3D {
    a_diag: f64,
    c_coupling: f64,
    grid3d: Grid3D<f64>,
    current: Vec<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
}

#[pymethods]
impl MatrixDiffusion3D {
    #[new]
    #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, u0, *, a_diag = 1.0_f64, c_coupling = 0.0_f64))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        zmin: f64,
        zmax: f64,
        nz: usize,
        u0: &Bound<'_, PyAny>,
        a_diag: f64,
        c_coupling: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            validate_3d_args(nx, ny, nz, a_diag)?;
            let vals: Vec<f64> = u0.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("u0 must be a float64 array")
            })?;
            validate_u0_3d(&vals, nx, ny, nz)?;
            let grid3d = build_grid3d(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz)
                .map_err(|e| from_core(&e))?;
            build_matrix_kernel(a_diag, c_coupling, grid3d.x).map_err(|e| from_core(&e))?;
            Ok(MatrixDiffusion3D { a_diag, c_coupling, grid3d, current: vals, nx, ny, nz })
        })
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations (GIL released).
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_t_nsteps(t, n_steps)?;
            let a = self.a_diag;
            let c = self.c_coupling;
            let g = self.grid3d;
            let vals = self.current.clone();
            let result: Result<Vec<f64>, semiflow::SemiflowError> =
                py.detach(|| evolve_3d(a, c, g, vals, t, n_steps));
            self.current = result.map_err(|e| from_core(&e))?;
            Ok(())
        })
    }

    /// Return current state as flat ``numpy.ndarray[float64]`` of length ``2*nx*ny*nz``.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.current.as_slice().to_pyarray(py)) })
    }

    /// Return approximation order (always 2).
    fn order(&self) -> u32 {
        2
    }

    fn __repr__(&self) -> String {
        format!(
            "MatrixDiffusion3D(nx={}, ny={}, nz={}, M=2, a_diag={:.3}, order=2)",
            self.nx, self.ny, self.nz, self.a_diag
        )
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn validate_t_nsteps(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn validate_2d_args(nx: usize, ny: usize, a_diag: f64) -> PyResult<()> {
    if nx < 5 {
        return Err(new_pyerr("OutOfDomain", "nx must be >= 5"));
    }
    if ny < 5 {
        return Err(new_pyerr("OutOfDomain", "ny must be >= 5"));
    }
    if !a_diag.is_finite() || a_diag <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "a_diag must be finite and > 0"));
    }
    Ok(())
}

fn validate_3d_args(nx: usize, ny: usize, nz: usize, a_diag: f64) -> PyResult<()> {
    validate_2d_args(nx, ny, a_diag)?;
    if nz < 5 {
        return Err(new_pyerr("OutOfDomain", "nz must be >= 5"));
    }
    Ok(())
}

fn validate_u0_2d(vals: &[f64], nx: usize, ny: usize) -> PyResult<()> {
    if vals.len() != 2 * nx * ny {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("u0 length {}, expected 2*nx*ny={}", vals.len(), 2 * nx * ny),
        ));
    }
    for &v in vals {
        if !v.is_finite() {
            return Err(new_pyerr("NanInf", "u0 contains NaN or Inf"));
        }
    }
    Ok(())
}

fn validate_u0_3d(vals: &[f64], nx: usize, ny: usize, nz: usize) -> PyResult<()> {
    if vals.len() != 2 * nx * ny * nz {
        return Err(new_pyerr(
            "GridMismatch",
            &format!(
                "u0 length {}, expected 2*nx*ny*nz={}",
                vals.len(),
                2 * nx * ny * nz
            ),
        ));
    }
    for &v in vals {
        if !v.is_finite() {
            return Err(new_pyerr("NanInf", "u0 contains NaN or Inf"));
        }
    }
    Ok(())
}

fn build_grid2d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
) -> Result<Grid2D<f64>, semiflow::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    Ok(Grid2D::new(gx, gy))
}

fn build_grid3d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    zmin: f64,
    zmax: f64,
    nz: usize,
) -> Result<Grid3D<f64>, semiflow::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    let gz = Grid1D::new(zmin, zmax, nz)?;
    Grid3D::new(gx, gy, gz)
}

fn build_axis_kernel(
    a_diag: f64,
    c_coupling: f64,
    axis: Grid1D<f64>,
) -> Result<MatrixDiffusionChernoff<f64, 2>, semiflow::SemiflowError> {
    build_matrix_kernel(a_diag, c_coupling, axis)
}

fn evolve_2d(
    a_diag: f64,
    c_coupling: f64,
    g: Grid2D<f64>,
    vals: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let kx = build_axis_kernel(a_diag, c_coupling, g.x)?;
    let ky = build_axis_kernel(a_diag, c_coupling, g.y)?;
    let kernel = MatrixDiffusionChernoff2D::new(kx, ky);
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let mut src = MatrixGridFn2D::<f64, 2>::new(g);
    src.values.copy_from_slice(&vals);
    let out = sg.evolve(t, &src)?;
    Ok(out.values)
}

fn evolve_3d(
    a_diag: f64,
    c_coupling: f64,
    g: Grid3D<f64>,
    vals: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let kx = build_axis_kernel(a_diag, c_coupling, g.x)?;
    let ky = build_axis_kernel(a_diag, c_coupling, g.y)?;
    let kz = build_axis_kernel(a_diag, c_coupling, g.z)?;
    let kernel = MatrixDiffusionChernoff3D::new(kx, ky, kz);
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let mut src = MatrixGridFn3D::<f64, 2>::new(g);
    src.values.copy_from_slice(&vals);
    let out = sg.evolve(t, &src)?;
    Ok(out.values)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MatrixDiffusion2D>()?;
    m.add_class::<MatrixDiffusion3D>()
}
