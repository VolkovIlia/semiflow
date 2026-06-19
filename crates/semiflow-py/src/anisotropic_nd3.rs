//! Wave P7 — `Heat2DVarA` (M21) and `Heat3DVarA` (M21).
//!
//! Variable-coefficient 2D and 3D heat via palindromic Strang splitting.
//! Split from `anisotropic_nd.rs` for suckless file-size compliance.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unused_self
)]

use std::sync::Arc;

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;

use semiflow_core::{
    ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D, Grid3D, GridFn2D, GridFn3D, ScratchPool,
    Strang2D, Strang3D,
};

use crate::{
    anisotropic_nd_helpers::{extract_pos_coeff_vec, interp_1d, validate_tau_nsteps},
    boundary::parse_boundary,
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// ===========================================================================
// Heat2DVarA — variable-coefficient 2D heat (M21)
// ===========================================================================

/// Variable-coefficient 2D heat Chernoff (M21, ADR-0111).
///
/// Solves ``∂_t u = a_x(x)·∂_xx u + a_y(y)·∂_yy u`` on
/// ``[xmin, xmax] × [ymin, ymax]`` using palindromic Strang splitting
/// (``Strang2D``).
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     X-axis domain (finite, xmin < xmax).
/// nx : int
///     Number of X-axis nodes (>= 4).
/// ymin, ymax : float
///     Y-axis domain (finite, ymin < ymax).
/// ny : int
///     Number of Y-axis nodes (>= 4).
/// `a_x` : array-like[float64]
///     ``a(x_i)`` values on the X-axis grid; length ``nx``.  Must be > 0 and finite.
/// `a_y` : array-like[float64]
///     ``a(y_j)`` values on the Y-axis grid; length ``ny``.  Must be > 0 and finite.
/// boundary : str, optional
///     Boundary policy; default ``"reflect"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' / '`NanInf`' / '`OutOfDomain`'.
#[pyclass(name = "Heat2DVarA")]
pub struct PyHeat2DVarA {
    strang: Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    grid: Grid2D<f64>,
    nx: usize,
    ny: usize,
}

#[pymethods]
impl PyHeat2DVarA {
    #[new]
    #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, a_x, a_y, *, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        a_x: &Bound<'_, PyAny>,
        a_y: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let ax_vals = extract_pos_coeff_vec(a_x, nx, "a_x")?;
            let ay_vals = extract_pos_coeff_vec(a_y, ny, "a_y")?;
            let (strang, grid) =
                build_strang2d(xmin, xmax, nx, ymin, ymax, ny, ax_vals, ay_vals, policy)
                    .map_err(|e| from_core(&e))?;
            Ok(Self {
                strang,
                grid,
                nx,
                ny,
            })
        })
    }

    /// Evolve ``u0`` (flat row-major, length ``nx * ny``) by ``n_steps`` steps
    /// of size ``tau``.
    ///
    /// Returns flat row-major ``numpy.ndarray[float64]``.  GIL released during
    /// inner compute (ADR-0031).
    #[pyo3(signature = (u0, tau, n_steps))]
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        u0: &Bound<'_, PyAny>,
        tau: f64,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            validate_tau_nsteps(tau, n_steps)?;
            let input: Vec<f64> = u0.extract::<Vec<f64>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("u0 must be numpy.ndarray[float64]")
            })?;
            let expected = self.nx * self.ny;
            if input.len() != expected {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("u0 length {} != nx*ny={}", input.len(), expected),
                ));
            }
            for &v in &input {
                if !v.is_finite() {
                    return Err(new_pyerr("NanInf", "u0 contains NaN or Inf"));
                }
            }
            let strang = self.strang.clone();
            let grid = self.grid;
            let result: Result<Vec<f64>, _> =
                py.detach(|| evolve_strang_2d(strang, grid, input, tau, n_steps));
            Ok(result.map_err(|e| from_core(&e))?.as_slice().to_pyarray(py))
        })
    }

    /// Return approximation order (2 — palindromic Strang).
    fn order(&self) -> u32 {
        2
    }

    /// Number of X-axis nodes.
    #[getter]
    fn nx(&self) -> usize {
        self.nx
    }

    /// Number of Y-axis nodes.
    #[getter]
    fn ny(&self) -> usize {
        self.ny
    }

    fn __len__(&self) -> usize {
        self.nx * self.ny
    }

    fn __repr__(&self) -> String {
        format!("Heat2DVarA(nx={}, ny={}, order=2)", self.nx, self.ny)
    }
}

fn evolve_strang_2d(
    strang: Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    grid: Grid2D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let mut state = GridFn2D::new(grid, input)?;
    let mut dst = GridFn2D::new(grid, vec![0.0; state.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        strang.apply_into(tau, &state, &mut dst, &mut scratch)?;
        core::mem::swap(&mut state, &mut dst);
    }
    Ok(state.values)
}

// ===========================================================================
// Heat3DVarA — variable-coefficient 3D heat (M21)
// ===========================================================================

/// Variable-coefficient 3D heat Chernoff (M21, ADR-0111).
///
/// Solves ``∂_t u = a_x(x)·∂_xx u + a_y(y)·∂_yy u + a_z(z)·∂_zz u`` on
/// ``[xmin, xmax] × [ymin, ymax] × [zmin, zmax]`` using palindromic Strang
/// splitting (``Strang3D``).
///
/// Parameters
/// ----------
/// xmin, xmax, nx : x-axis.
/// ymin, ymax, ny : y-axis.
/// zmin, zmax, nz : z-axis.
/// `a_x`, `a_y`, `a_z` : array-like[float64]
///     Per-axis diffusion coefficients; lengths ``nx``, ``ny``, ``nz`` resp.
///     Must be > 0 and finite.
/// boundary : str, optional
///     Boundary policy; applied to all three axes.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' / '`NanInf`' / '`OutOfDomain`'.
#[pyclass(name = "Heat3DVarA")]
pub struct PyHeat3DVarA {
    strang: Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    grid: Grid3D<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
}

#[pymethods]
impl PyHeat3DVarA {
    #[new]
    #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz,
                        a_x, a_y, a_z, *, boundary = "reflect"))]
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
        a_x: &Bound<'_, PyAny>,
        a_y: &Bound<'_, PyAny>,
        a_z: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let ax_vals = extract_pos_coeff_vec(a_x, nx, "a_x")?;
            let ay_vals = extract_pos_coeff_vec(a_y, ny, "a_y")?;
            let az_vals = extract_pos_coeff_vec(a_z, nz, "a_z")?;
            let (strang, grid) = build_strang3d(
                xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, ax_vals, ay_vals, az_vals, policy,
            )
            .map_err(|e| from_core(&e))?;
            Ok(Self {
                strang,
                grid,
                nx,
                ny,
                nz,
            })
        })
    }

    /// Evolve ``u0`` (flat x-fastest, length ``nx * ny * nz``) by ``n_steps``
    /// steps of size ``tau``.
    ///
    /// Returns flat x-fastest ``numpy.ndarray[float64]``.  GIL released during
    /// compute (ADR-0031).
    #[pyo3(signature = (u0, tau, n_steps))]
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        u0: &Bound<'_, PyAny>,
        tau: f64,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            validate_tau_nsteps(tau, n_steps)?;
            let input: Vec<f64> = u0.extract::<Vec<f64>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("u0 must be numpy.ndarray[float64]")
            })?;
            let expected = self.nx * self.ny * self.nz;
            if input.len() != expected {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("u0 length {} != nx*ny*nz={}", input.len(), expected),
                ));
            }
            for &v in &input {
                if !v.is_finite() {
                    return Err(new_pyerr("NanInf", "u0 contains NaN or Inf"));
                }
            }
            let strang = self.strang.clone();
            let grid = self.grid;
            let result: Result<Vec<f64>, _> =
                py.detach(|| evolve_strang_3d(strang, grid, input, tau, n_steps));
            Ok(result.map_err(|e| from_core(&e))?.as_slice().to_pyarray(py))
        })
    }

    /// Return approximation order (2 — palindromic Strang).
    fn order(&self) -> u32 {
        2
    }

    fn __len__(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    fn __repr__(&self) -> String {
        format!(
            "Heat3DVarA(nx={}, ny={}, nz={}, order=2)",
            self.nx, self.ny, self.nz,
        )
    }
}

fn evolve_strang_3d(
    strang: Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    grid: Grid3D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let mut state = GridFn3D::new(grid, input)?;
    let mut dst = GridFn3D::new(grid, vec![0.0; state.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        strang.apply_into(tau, &state, &mut dst, &mut scratch)?;
        core::mem::swap(&mut state, &mut dst);
    }
    Ok(state.values)
}

/// Build a `Strang2D` from pre-sampled 1D diffusion coefficient arrays.
#[allow(clippy::too_many_arguments)]
fn build_strang2d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    ax_vals: Vec<f64>,
    ay_vals: Vec<f64>,
    policy: semiflow_core::BoundaryPolicy,
) -> Result<
    (
        Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
        Grid2D<f64>,
    ),
    semiflow_core::SemiflowError,
> {
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(policy);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(policy);
    let grid = Grid2D::new(gx, gy);
    let dx = build_axis_diff(ax_vals, xmin, xmax, nx, gx);
    let dy = build_axis_diff(ay_vals, ymin, ymax, ny, gy);
    Ok((Strang2D::new(dx, dy), grid))
}

/// Build a `Strang3D` from pre-sampled 1D diffusion coefficient arrays.
/// Build a `DiffusionChernoff` with constant-zero drift/reaction from a tabulated `a_vals`.
fn build_axis_diff(
    a_vals: Vec<f64>,
    amin: f64,
    amax: f64,
    n: usize,
    grid: Grid1D<f64>,
) -> DiffusionChernoff<f64> {
    let norm = a_vals.iter().copied().fold(0.0_f64, f64::max);
    let arc = Arc::new(a_vals);
    DiffusionChernoff::with_closure(
        move |t: f64| interp_1d(&arc, amin, amax, n, t),
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        norm,
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_strang3d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    zmin: f64,
    zmax: f64,
    nz: usize,
    ax_vals: Vec<f64>,
    ay_vals: Vec<f64>,
    az_vals: Vec<f64>,
    policy: semiflow_core::BoundaryPolicy,
) -> Result<
    (
        Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
        Grid3D<f64>,
    ),
    semiflow_core::SemiflowError,
> {
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(policy);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(policy);
    let gz = Grid1D::new(zmin, zmax, nz)?.with_boundary(policy);
    let grid = Grid3D::new(gx, gy, gz)?;
    let dx = build_axis_diff(ax_vals, xmin, xmax, nx, gx);
    let dy = build_axis_diff(ay_vals, ymin, ymax, ny, gy);
    let dz = build_axis_diff(az_vals, zmin, zmax, nz, gz);
    Ok((Strang3D::new(dx, dy, dz), grid))
}
