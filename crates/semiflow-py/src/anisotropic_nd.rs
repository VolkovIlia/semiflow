//! Wave P7 — multi-D anisotropic + 2D/3D variable-coefficient constructors.
//!
//! ADR-0111 parity items M19–M21.  ADR-0112 normalization/order-1 fix applied.
//!
//! | pyclass                  | Core type                                          | M#  | Module |
//! |--------------------------|---------------------------------------------------|-----|--------|
//! | `AnisotropicShiftND2`    | `AnisotropicShiftChernoffND<f64, 2>`              | M19 | here |
//! | `AnisotropicShiftND3`    | `AnisotropicShiftChernoffND<f64, 3>`              | M19 | here |
//! | `NonSeparable2DAniso`    | `NonSeparable2DAnisotropicChernoff<Dc, Dc, f64>`  | M20 | `anisotropic_nd2` |
//! | `Heat2DVarA`             | `Strang2D<DiffusionChernoff<f64>, Dc<f64>>`       | M21 | `anisotropic_nd3` |
//! | `Heat3DVarA`             | `Strang3D<Dc, Dc, Dc>` with per-axis var-a        | M21 | `anisotropic_nd3` |
//!
//! ## GIL policy (ADR-0031)
//!
//! Three-phase pattern for every `evolve`.
//! Binding-layer workaround (M19): pre-sampled-array constructors only —
//! coefficient closures from Python would require GIL re-acquisition in py.detach.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unused_self
)]

use std::sync::Arc;

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    grid_nd::{GridFnND, GridND},
    shift_nd::AnisotropicShiftChernoffND,
    ChernoffFunction, Grid1D, ScratchPool,
};

// Re-export sibling pyclasses for registration.
pub(crate) use crate::anisotropic_nd2::PyNonSeparable2DAniso;
pub(crate) use crate::anisotropic_nd3::{PyHeat2DVarA, PyHeat3DVarA};
use crate::{
    anisotropic_nd_helpers::{
        build_aniso_nd3_kernel, extract_finite_f64_vec, nd_flat_idx_2, validate_t_nsteps,
    },
    error::from_core,
    panic::catch_panic_py,
};

// ===========================================================================
// AnisotropicShiftND2 — D=2 specialisation (M19)
// ===========================================================================

/// Anisotropic shift Chernoff kernel on a 2-D tensor-product grid (M19,
/// ADR-0081 + ADR-0112, math §32).
///
/// Solves ``∂_t u = A(x)·∇²u + b(x)·∇u + c(x)·u`` where ``A(x)`` is a
/// 2×2 symmetric positive-definite diffusion tensor, ``b(x)`` is a 2-D drift
/// vector and ``c(x)`` is a scalar reaction coefficient.
///
/// **Order 1** (global O(1/n), honest ADR-0112).  Exact for constant A.
/// F(0) = I guaranteed by the ``π^{-D/2}`` normalization (ADR-0112 §Decision 1).
///
/// This is the D=2 specialisation (``AnisotropicShiftND3`` covers D=3).
///
/// Parameters
/// ----------
/// nx, ny : int
///     Number of grid nodes on axis 0 (x) and axis 1 (y).  Each must be >= 4.
/// xmin, xmax : float
///     Axis-0 (x) domain boundaries.
/// ymin, ymax : float
///     Axis-1 (y) domain boundaries.
/// `a_values` : array-like[float64]
///     Pre-sampled diffusion-tensor values.  Length ``nx * ny * 4``.
///     The tensor must be SPD at every grid point.
/// `b_values` : array-like[float64] or None
///     Pre-sampled drift vector.  Length ``nx * ny * 2``.  ``None`` → zero.
/// `c_values` : array-like[float64] or None
///     Pre-sampled reaction coefficient.  Length ``nx * ny``.  ``None`` → c = 0.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' / '`NanInf`' / '`OutOfDomain`'.
#[pyclass(name = "AnisotropicShiftND2")]
pub struct PyAnisotropicShiftND2 {
    kernel: std::sync::Arc<AnisotropicShiftChernoffND<f64, 2>>,
    nx: usize,
    ny: usize,
    current: Vec<f64>,
    grid: GridND<f64, 2>,
}

#[pymethods]
impl PyAnisotropicShiftND2 {
    #[new]
    #[pyo3(signature = (nx, ny, xmin, xmax, ymin, ymax, a_values, *, b_values = None, c_values = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nx: usize,
        ny: usize,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        a_values: &Bound<'_, PyAny>,
        b_values: Option<&Bound<'_, PyAny>>,
        c_values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let n_pts = nx * ny;
            let a_raw = extract_finite_f64_vec(a_values, 4 * n_pts, "a_values")?;
            let b_raw = match b_values {
                Some(b) => extract_finite_f64_vec(b, 2 * n_pts, "b_values")?,
                None => vec![0.0_f64; 2 * n_pts],
            };
            let c_raw = match c_values {
                Some(c) => extract_finite_f64_vec(c, n_pts, "c_values")?,
                None => vec![0.0_f64; n_pts],
            };
            let kernel =
                build_aniso_nd2_kernel(nx, ny, xmin, xmax, ymin, ymax, a_raw, b_raw, c_raw)
                    .map_err(|e| from_core(&e))?;
            let grid = kernel.grid().clone();
            let current = vec![0.0_f64; nx * ny];
            Ok(Self {
                kernel: std::sync::Arc::new(kernel),
                nx,
                ny,
                current,
                grid,
            })
        })
    }

    /// Set the initial condition from a flat float64 array of length ``nx * ny``.
    fn set_state(&mut self, u0: &Bound<'_, PyAny>) -> PyResult<()> {
        catch_panic_py!({
            let n = self.nx * self.ny;
            let vals = extract_finite_f64_vec(u0, n, "u0")?;
            self.current = vals;
            Ok(())
        })
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// GIL released during inner Rust compute (ADR-0031).
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_t_nsteps(t, n_steps)?;
            let tau = t / n_steps as f64;
            let kernel = std::sync::Arc::clone(&self.kernel);
            let grid = self.grid.clone();
            let input = self.current.clone();
            let result: Result<Vec<f64>, semiflow::SemiflowError> =
                py.detach(|| evolve_nd_2(kernel, grid, input, tau, n_steps));
            self.current = result.map_err(|e| from_core(&e))?;
            Ok(())
        })
    }

    /// Return current state as flat ``numpy.ndarray[float64]`` (copy).
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.current.as_slice().to_pyarray(py)) })
    }

    /// Return approximation order (1 — honest ADR-0112).
    fn order(&self) -> u32 {
        1
    }

    /// Number of grid points (``nx * ny``).
    fn __len__(&self) -> usize {
        self.nx * self.ny
    }

    fn __repr__(&self) -> String {
        format!(
            "AnisotropicShiftND2(nx={}, ny={}, order=1)",
            self.nx, self.ny
        )
    }
}

fn evolve_nd_2(
    kernel: std::sync::Arc<AnisotropicShiftChernoffND<f64, 2>>,
    grid: GridND<f64, 2>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let mut src = GridFnND::<f64, 2>::new(grid.clone(), input)?;
    let mut dst = GridFnND::<f64, 2>::new(grid, vec![0.0; src.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

/// Build an `AnisotropicShiftChernoffND<f64, 2>` from pre-sampled coefficient arrays.
fn build_aniso_nd2_kernel(
    nx: usize,
    ny: usize,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    a_raw: Vec<f64>,
    b_raw: Vec<f64>,
    c_raw: Vec<f64>,
) -> Result<AnisotropicShiftChernoffND<f64, 2>, semiflow::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    let grid_nd = GridND::<f64, 2>::new([gx, gy])?;
    let a_arc = Arc::new(a_raw);
    let b_arc = Arc::new(b_raw);
    let c_arc = Arc::new(c_raw);
    let ns = [nx, ny];
    let axes = [(xmin, xmax, nx), (ymin, ymax, ny)];
    let a_arc2 = Arc::clone(&a_arc);
    let b_arc2 = Arc::clone(&b_arc);
    let c_arc2 = Arc::clone(&c_arc);
    let axes2 = axes;
    let axes3 = axes;
    AnisotropicShiftChernoffND::<f64, 2>::new(
        move |x, mat| {
            let flat = nd_flat_idx_2(x, &ns, &axes);
            let base = flat * 4;
            mat.set(0, 0, a_arc2[base]);
            mat.set(0, 1, a_arc2[base + 1]);
            mat.set(1, 0, a_arc2[base + 2]);
            mat.set(1, 1, a_arc2[base + 3]);
        },
        move |x, bv| {
            let flat = nd_flat_idx_2(x, &ns, &axes2);
            let base = flat * 2;
            bv[0] = b_arc2[base];
            bv[1] = b_arc2[base + 1];
        },
        move |x| {
            let flat = nd_flat_idx_2(x, &ns, &axes3);
            c_arc2[flat]
        },
        grid_nd,
    )
}

// ===========================================================================
// AnisotropicShiftND3 — D=3 specialisation (M19)
// ===========================================================================

/// Anisotropic shift Chernoff kernel on a 3-D tensor-product grid (M19, D=3).
///
/// Identical contract to :class:`AnisotropicShiftND2` but for 3 spatial
/// dimensions.  State layout: flat array of length ``nx * ny * nz``,
/// axis-0 (x) fastest.
///
/// Parameters
/// ----------
/// nx, ny, nz : int
///     Number of grid nodes per axis.  Each must be >= 4.
/// xmin/xmax, ymin/ymax, zmin/zmax : float
///     Domain boundaries.
/// `a_values` : array-like[float64]
///     Length ``nx * ny * nz * 9`` (3×3 per point, row-major).  SPD required.
/// `b_values` : array-like[float64] or None
///     Drift; length ``nx * ny * nz * 3``.  ``None`` → zero.
/// `c_values` : array-like[float64] or None
///     Reaction; length ``nx * ny * nz``.  ``None`` → zero.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' / '`NanInf`' / '`OutOfDomain`'.
#[pyclass(name = "AnisotropicShiftND3")]
pub struct PyAnisotropicShiftND3 {
    kernel: std::sync::Arc<AnisotropicShiftChernoffND<f64, 3>>,
    nx: usize,
    ny: usize,
    nz: usize,
    current: Vec<f64>,
    grid: GridND<f64, 3>,
}

#[pymethods]
impl PyAnisotropicShiftND3 {
    #[new]
    #[pyo3(signature = (nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, a_values, *,
                        b_values = None, c_values = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        zmin: f64,
        zmax: f64,
        a_values: &Bound<'_, PyAny>,
        b_values: Option<&Bound<'_, PyAny>>,
        c_values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let n_pts = nx * ny * nz;
            let a_raw = extract_finite_f64_vec(a_values, 9 * n_pts, "a_values")?;
            let b_raw = match b_values {
                Some(b) => extract_finite_f64_vec(b, 3 * n_pts, "b_values")?,
                None => vec![0.0_f64; 3 * n_pts],
            };
            let c_raw = match c_values {
                Some(c) => extract_finite_f64_vec(c, n_pts, "c_values")?,
                None => vec![0.0_f64; n_pts],
            };
            let kernel = build_aniso_nd3_kernel(
                nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, a_raw, b_raw, c_raw,
            )
            .map_err(|e| from_core(&e))?;
            let grid = kernel.grid().clone();
            let current = vec![0.0_f64; n_pts];
            Ok(Self {
                kernel: std::sync::Arc::new(kernel),
                nx,
                ny,
                nz,
                current,
                grid,
            })
        })
    }

    /// Set the initial condition from a flat float64 array of length ``nx * ny * nz``.
    fn set_state(&mut self, u0: &Bound<'_, PyAny>) -> PyResult<()> {
        catch_panic_py!({
            let n = self.nx * self.ny * self.nz;
            let vals = extract_finite_f64_vec(u0, n, "u0")?;
            self.current = vals;
            Ok(())
        })
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// GIL released during compute (ADR-0031).
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_t_nsteps(t, n_steps)?;
            let tau = t / n_steps as f64;
            let kernel = std::sync::Arc::clone(&self.kernel);
            let grid = self.grid.clone();
            let input = self.current.clone();
            let result: Result<Vec<f64>, semiflow::SemiflowError> =
                py.detach(|| evolve_nd_3(kernel, grid, input, tau, n_steps));
            self.current = result.map_err(|e| from_core(&e))?;
            Ok(())
        })
    }

    /// Return current state as flat ``numpy.ndarray[float64]`` (copy).
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.current.as_slice().to_pyarray(py)) })
    }

    /// Return approximation order (1 — honest ADR-0112).
    fn order(&self) -> u32 {
        1
    }

    /// Number of grid points (``nx * ny * nz``).
    fn __len__(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    fn __repr__(&self) -> String {
        format!(
            "AnisotropicShiftND3(nx={}, ny={}, nz={}, order=1)",
            self.nx, self.ny, self.nz,
        )
    }
}

fn evolve_nd_3(
    kernel: std::sync::Arc<AnisotropicShiftChernoffND<f64, 3>>,
    grid: GridND<f64, 3>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let mut src = GridFnND::<f64, 3>::new(grid.clone(), input)?;
    let mut dst = GridFnND::<f64, 3>::new(grid, vec![0.0; src.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ===========================================================================
// Registration
// ===========================================================================

/// Register Wave P7 pyclasses into the `semiflow` module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyAnisotropicShiftND2>()?;
    m.add_class::<PyAnisotropicShiftND3>()?;
    m.add_class::<PyNonSeparable2DAniso>()?;
    m.add_class::<PyHeat2DVarA>()?;
    m.add_class::<PyHeat3DVarA>()?;
    Ok(())
}
