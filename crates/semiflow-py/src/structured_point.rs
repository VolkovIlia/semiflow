//! Wave P6 — `PointEval` + `sample_gridfn2d` (M18).
//!
//! Pointwise evaluation via `DiffusionChernoff<f64>` and bilinear interpolation
//! on `GridFn2D<f64>`.
//! Split from `structured.rs` for suckless file-size compliance.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3 wrapper patterns.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

use semiflow_core::{
    diffusion::DiffusionChernoff,
    point_eval::{sample_gridfn2d as core_sample_gridfn2d, PointEval as CorePointEval},
    Grid1D, Grid2D, GridFn1D, GridFn2D,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_py::extract_f64_vec,
    panic::catch_panic_py,
};

// ===========================================================================
// PointEval — pointwise evaluation via DiffusionChernoff (M18)
// ===========================================================================

/// Pointwise evaluation via the ``DiffusionChernoff<f64>`` Backend A (M18,
/// ADR-0080, math §31.2).
///
/// For a 1D diffusion kernel, ``eval_at(tau, u0, x, n_steps)`` returns the
/// scalar ``(F(τ))^{n_steps} u0`` evaluated at the single query point ``x``.
///
/// Byte-identity contract (Proposition 31.1): the returned scalar is
/// bit-identical to sampling the result of ``n_steps`` full ``apply_into``
/// calls at ``x`` (bilinear interpolation on grid nodes).
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     Grid boundaries (finite, xmin < xmax).
/// n : int
///     Number of grid nodes (>= 4).
/// `a_fn` : callable or None
///     Diffusion coefficient closure ``a(x) -> float``.
///     ``None`` → unit diffusion a ≡ 1.0.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if grid params invalid.
#[pyclass(name = "PointEval")]
pub struct PyPointEval {
    xmin: f64,
    xmax: f64,
    n: usize,
}

#[pymethods]
impl PyPointEval {
    #[new]
    #[pyo3(signature = (xmin, xmax, n))]
    fn new(xmin: f64, xmax: f64, n: usize) -> PyResult<Self> {
        catch_panic_py!({
            // Validate grid early.
            Grid1D::new(xmin, xmax, n).map_err(|e| from_core(&e))?;
            Ok(Self { xmin, xmax, n })
        })
    }

    /// Evaluate ``(F(τ))^{n_steps} u0`` at point ``x``.
    ///
    /// Parameters
    /// ----------
    /// tau : float
    ///     Chernoff step size (>= 0, finite).
    /// u0 : array-like
    ///     Initial condition; float64 array of length ``n``.
    /// x : float
    ///     Query point.
    /// `n_steps` : int, optional
    ///     Number of Chernoff iterations (default 1; must be >= 1).
    ///
    /// Returns
    /// -------
    /// float
    ///     Scalar approximation at ``x``.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`NanInf`' if u0 contains NaN or Inf.
    ///     kind='`GridMismatch`' if u0 length != n.
    ///     kind='`OutOfDomain`' if `n_steps` == 0 or tau < 0 or non-finite.
    #[pyo3(signature = (tau, u0, x, n_steps = 1_u32))]
    fn eval_at(
        &self,
        py: Python<'_>,
        tau: f64,
        u0: &Bound<'_, PyAny>,
        x: f64,
        n_steps: u32,
    ) -> PyResult<f64> {
        catch_panic_py!({
            if n_steps == 0 {
                return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
            }
            if !tau.is_finite() || tau < 0.0 {
                return Err(new_pyerr("OutOfDomain", "tau must be finite and >= 0"));
            }
            let vals = extract_f64_vec(u0).map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("u0 must be a float64 array")
            })?;
            if vals.len() != self.n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("u0 length {}, expected {}", vals.len(), self.n),
                ));
            }
            for &v in &vals {
                if !v.is_finite() {
                    return Err(new_pyerr("NanInf", "u0 contains NaN or Inf"));
                }
            }
            let xmin = self.xmin;
            let xmax = self.xmax;
            let n = self.n;
            let result: Result<f64, semiflow_core::SemiflowError> =
                py.detach(|| eval_at_rust(xmin, xmax, n, vals, tau, x, n_steps));
            result.map_err(|e| from_core(&e))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PointEval(xmin={}, xmax={}, n={})",
            self.xmin, self.xmax, self.n,
        )
    }
}

fn eval_at_rust(
    xmin: f64,
    xmax: f64,
    n: usize,
    vals: Vec<f64>,
    tau: f64,
    x: f64,
    n_steps: u32,
) -> Result<f64, semiflow_core::SemiflowError> {
    let grid = Grid1D::new(xmin, xmax, n)?;
    let kernel = DiffusionChernoff::new(
        |_: f64| 1.0_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        1.0_f64,
        grid,
    );
    let src = GridFn1D { values: vals, grid };
    kernel.eval_at(tau, &src, &[x], n_steps)
}

// ===========================================================================
// sample_gridfn2d — free function for bilinear interpolation (M18)
// ===========================================================================

/// Bilinear interpolation of a 2D grid function at chart position ``(cx, cy)``.
///
/// This free function exposes ``semiflow_core::point_eval::sample_gridfn2d``
/// (math §31.3, Proposition 31.1) to Python.  It is the canonical primitive for
/// evaluating a ``GridFn2D<f64>`` at an arbitrary chart position without running
/// further Chernoff steps.
///
/// Parameters
/// ----------
/// values : array-like
///     Flat float64 array of length ``nx * ny`` (row-major, x-axis fast).
///     Raises ``SemiflowError(GridMismatch)`` if length != nx * ny.
/// x0min, x0max : float
///     x-axis (axis-0) boundaries.
/// nx : int
///     Number of nodes on axis 0 (>= 2).
/// x1min, x1max : float
///     y-axis (axis-1) boundaries.
/// ny : int
///     Number of nodes on axis 1 (>= 2).
/// cx : float
///     Query coordinate on axis 0.
/// cy : float
///     Query coordinate on axis 1.
///
/// Returns
/// -------
/// float
///     Bilinearly interpolated value (clamped to domain, no extrapolation).
///
/// Raises
/// ------
/// SemiflowError
///     kind='`GridMismatch`' if values length != nx * ny.
#[pyfunction]
#[pyo3(signature = (values, x0min, x0max, nx, x1min, x1max, ny, cx, cy))]
#[allow(clippy::too_many_arguments)]
pub fn sample_gridfn2d(
    values: &Bound<'_, PyAny>,
    x0min: f64,
    x0max: f64,
    nx: usize,
    x1min: f64,
    x1max: f64,
    ny: usize,
    cx: f64,
    cy: f64,
) -> PyResult<f64> {
    catch_panic_py!({
        let vals = extract_f64_vec(values).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("values must be a float64 array")
        })?;
        if vals.len() != nx * ny {
            return Err(new_pyerr(
                "GridMismatch",
                &format!("values length {}, expected nx*ny={}", vals.len(), nx * ny),
            ));
        }
        let gx = Grid1D::new(x0min, x0max, nx).map_err(|e| from_core(&e))?;
        let gy = Grid1D::new(x1min, x1max, ny).map_err(|e| from_core(&e))?;
        let grid = Grid2D::new(gx, gy);
        let state = GridFn2D { values: vals, grid };
        Ok(core_sample_gridfn2d(&state, cx, cy))
    })
}
