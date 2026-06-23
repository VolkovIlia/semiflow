//! `NonSeparable2D` — non-separable anisotropic 2D diffusion with mixed
//! derivative coupling `β(x,y)·∂_x∂_y` (v2.3 Phase 4, ADR-0058).
//!
//! Wraps `NonSeparableMixedChernoff` with the same three-phase GIL-release
//! pattern as `Heat2D` (ADR-0031).
//!
//! ## GIL policy
//!
//! 1. **Validate + extract** (under GIL): parse params, extract `u0`/`beta_values`.
//! 2. **Compute** (GIL released via `py.detach`): pure-Rust Chernoff steps.
//! 3. **Return** (under GIL): wrap `Vec<f64>` into `numpy.ndarray`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::assigning_clones,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    nonseparable_mixed_closure, ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D, GridFn2D,
    NonSeparableMixedChernoff, ScratchPool,
};

use crate::{
    boundary::parse_boundary,
    coeff2d::{closure_2d_from_array, magnitude_max},
    error::{from_core, new_pyerr},
    handle::unit_diffusion_1d,
    panic::catch_panic_py,
};

/// Concrete 2D operator type used internally.
type Nsm = NonSeparableMixedChernoff<DiffusionChernoff<f64>, DiffusionChernoff<f64>>;

// ---------------------------------------------------------------------------
// NonSeparable2D Python class
// ---------------------------------------------------------------------------

/// Non-separable anisotropic 2D diffusion state.
///
/// Solves `∂_t u = ∂_{xx}u + ∂_{yy}u + β(x,y)·∂_x∂_y u` on
/// `[xmin, xmax] × [ymin, ymax]` via the palindromic 5-leg Chernoff operator
/// (math.md §10.7-ter, ADR-0058).
///
/// Use the default constructor for constant scalar coupling `c`, or
/// :meth:`with_beta_array` for a spatially-varying `β(x,y)`.
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     X-axis domain; must be finite and `xmin < xmax`.
/// nx : int
///     Number of X-axis grid nodes (>= 4).
/// ymin, ymax : float
///     Y-axis domain; must be finite and `ymin < ymax`.
/// ny : int
///     Number of Y-axis grid nodes (>= 4).
/// u0 : numpy.ndarray[float64]
///     Flat row-major initial condition, length `nx * ny`.
/// c : float, optional
///     Constant scalar coupling (default 0.0 — reduces to `Strang2D`).
/// boundary : str, optional
///     Boundary policy; one of ``"reflect"`` (default), ``"periodic"``,
///     ``"zero"``, ``"linear"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` for grid-size or IC-length mismatches.
///     ``kind='NanInf'`` if `u0` contains NaN/Inf.
///     ``kind='CflViolated'`` if the coupling+step-size combination violates CFL.
#[pyclass(name = "NonSeparable2D")]
pub struct NonSeparable2D {
    /// The compiled Chernoff operator (cloned on each `evolve`).
    kernel: Nsm,
    /// 2D grid geometry.
    grid: Grid2D<f64>,
    /// Current flat row-major state, length `nx * ny`.
    current: Vec<f64>,
    /// Number of X-axis nodes.
    nx: usize,
    /// Number of Y-axis nodes.
    ny: usize,
}

#[pymethods]
impl NonSeparable2D {
    /// Construct `NonSeparable2D` with constant scalar coupling `c`.
    #[new]
    #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, u0, *, c = 0.0, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        u0: &Bound<'_, PyAny>,
        c: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let u0_vec = extract_u0_flat(u0, nx, ny)?;
            let (gx, gy, grid) = build_grid_2d(xmin, xmax, nx, ymin, ymax, ny, policy)?;
            let c_norm = c.abs();
            let c_val = c;
            let arc_c: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'static> =
                Arc::new(move |_x, _y| c_val);
            let kernel = nonseparable_mixed_closure::with_closure_c(
                unit_diffusion_1d(gx),
                unit_diffusion_1d(gy),
                arc_c,
                c_norm,
                grid,
            )
            .map_err(|e| from_core(&e))?;
            Ok(NonSeparable2D {
                kernel,
                grid,
                current: u0_vec,
                nx,
                ny,
            })
        })
    }

    /// Construct `NonSeparable2D` from a pre-sampled 2-D `beta_values` array.
    ///
    /// Parameters
    /// ----------
    /// `beta_values` : numpy.ndarray[float64]
    ///     Shape `(nx, ny)` row-major array of `β(x_i, y_j)` values.
    /// `beta_norm_bound` : float or None
    ///     Upper bound on `‖β‖_∞`.  Auto-computed as `1.1 * max(|β|)` if None.
    #[staticmethod]
    #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, beta_values, u0, *,
                        beta_norm_bound = None, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn with_beta_array(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        beta_values: &Bound<'_, PyAny>,
        u0: &Bound<'_, PyAny>,
        beta_norm_bound: Option<f64>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let u0_vec = extract_u0_flat(u0, nx, ny)?;
            let beta_cls = closure_2d_from_array(beta_values, xmin, xmax, nx, ymin, ymax, ny)?;
            let beta_raw: Vec<f64> = beta_values.extract::<Vec<f64>>().map_err(|_| {
                new_pyerr("GridMismatch", "beta_values must be numpy.ndarray[float64]")
            })?;
            let norm_bound = beta_norm_bound.unwrap_or_else(|| {
                let m = magnitude_max(&beta_raw);
                if m == 0.0 {
                    0.0
                } else {
                    m * 1.1
                }
            });
            let (gx, gy, grid) = build_grid_2d(xmin, xmax, nx, ymin, ymax, ny, policy)?;
            let kernel = nonseparable_mixed_closure::with_closure_beta(
                unit_diffusion_1d(gx),
                unit_diffusion_1d(gy),
                beta_cls,
                norm_bound,
                grid,
            )
            .map_err(|e| from_core(&e))?;
            Ok(NonSeparable2D {
                kernel,
                grid,
                current: u0_vec,
                nx,
                ny,
            })
        })
    }

    /// Evolve the current state by time `t` using `n_steps` Chernoff steps.
    ///
    /// Returns a flat row-major `numpy.ndarray[float64]` of length `nx * ny`.
    ///
    /// The GIL is released during the inner Rust compute loop (ADR-0031).
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            if n_steps == 0 {
                return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
            }
            if !t.is_finite() || t <= 0.0 {
                return Err(new_pyerr("OutOfDomain", "t must be finite and > 0"));
            }
            let tau = t / n_steps as f64;
            let kernel = self.kernel.clone();
            let grid = self.grid;
            let input = self.current.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| evolve_nonsep(kernel, grid, input, tau, n_steps));
            let result = result.map_err(|e| from_core(&e))?;
            self.current = result.clone();
            Ok(result.as_slice().to_pyarray(py))
        })
    }

    /// Number of state values (`nx * ny`).
    fn __len__(&self) -> usize {
        self.nx * self.ny
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust compute helper (called inside py.detach)
// ---------------------------------------------------------------------------

fn evolve_nonsep(
    kernel: Nsm,
    grid: Grid2D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let mut state = GridFn2D::new(grid, input)?;
    let mut dst = GridFn2D::new(grid, vec![0.0; state.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &state, &mut dst, &mut scratch)?;
        core::mem::swap(&mut state, &mut dst);
    }
    Ok(state.values)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Extract and validate a flat `u0` of length `nx*ny`.
fn extract_u0_flat(u0: &Bound<'_, PyAny>, nx: usize, ny: usize) -> PyResult<Vec<f64>> {
    let v: Vec<f64> = u0
        .extract::<Vec<f64>>()
        .map_err(|_| new_pyerr("GridMismatch", "u0 must be numpy.ndarray[float64]"))?;
    let expected = nx * ny;
    if v.len() != expected {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("u0 length {} != nx*ny={}", v.len(), expected),
        ));
    }
    for &val in &v {
        if !val.is_finite() {
            return Err(new_pyerr("NanInf", "u0 contains NaN or Inf"));
        }
    }
    Ok(v)
}

/// Build `(gx, gy, grid)` from domain parameters and a boundary policy.
fn build_grid_2d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    policy: semiflow::BoundaryPolicy,
) -> PyResult<(Grid1D<f64>, Grid1D<f64>, Grid2D<f64>)> {
    use semiflow::Grid1D;
    let gx = Grid1D::new(xmin, xmax, nx)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    let gy = Grid1D::new(ymin, ymax, ny)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    let grid = Grid2D::new(gx, gy);
    Ok((gx, gy, grid))
}
