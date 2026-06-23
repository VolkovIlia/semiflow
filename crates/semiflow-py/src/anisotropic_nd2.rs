//! Wave P7 — `NonSeparable2DAniso` (M20).
//!
//! Non-separable 2D diffusion with position-dependent anisotropy β(x,y).
//! Split from `anisotropic_nd.rs` for suckless file-size compliance.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::assigning_clones,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value
)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;

use semiflow::{
    nonseparable_mixed_closure, ChernoffFunction, DiffusionChernoff, Grid2D, GridFn2D, ScratchPool,
};

use crate::{
    anisotropic_nd_helpers::{build_grid_2d, extract_u0_flat_2d, validate_t_pos},
    boundary::parse_boundary,
    coeff2d::{closure_2d_from_array, magnitude_max},
    error::{from_core, new_pyerr},
    handle::unit_diffusion_1d,
    panic::catch_panic_py,
};

/// Concrete `NonSeparableMixed` type for the aniso variant.
type NsmAniso =
    semiflow::NonSeparableMixedChernoff<DiffusionChernoff<f64>, DiffusionChernoff<f64>>;

// ===========================================================================
// NonSeparable2DAniso — position-dependent beta coupling (M20)
// ===========================================================================

/// Non-separable 2D diffusion with position-dependent anisotropy β(x,y) (M20).
///
/// Solves ``∂_t u = ∂_xx u + ∂_yy u + β(x,y)·∂_xy u`` on
/// ``[xmin, xmax] × [ymin, ymax]``.  This is the anisotropic variant of
/// :class:`NonSeparable2D`; the coupling field ``β(x,y)`` is accepted as a
/// pre-sampled flat array to avoid Python-callable GIL re-acquisition inside
/// ``py.detach`` (ADR-0111, ADR-0031, ADR-0034).
///
/// Use :class:`NonSeparable2D` for constant or Python-closure coupling;
/// use this class for spatially-varying ``β(x,y)`` on the full 2-D grid.
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     X-axis domain.
/// nx : int
///     Number of X-axis nodes (>= 4).
/// ymin, ymax : float
///     Y-axis domain.
/// ny : int
///     Number of Y-axis nodes (>= 4).
/// `beta_values` : array-like[float64]
///     Pre-sampled ``β`` on the grid; flat row-major, length ``nx * ny``.
/// u0 : array-like[float64]
///     Initial condition; flat row-major, length ``nx * ny``.
/// `beta_norm_bound` : float or None
///     ``‖β‖_∞`` upper bound.  Auto-computed as ``1.1 × max|β|`` if None.
/// boundary : str, optional
///     One of ``"reflect"`` (default), ``"periodic"``, ``"zero"``,
///     ``"linear"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' / '`NanInf`' / '`CflViolated`'.
#[pyclass(name = "NonSeparable2DAniso")]
pub struct PyNonSeparable2DAniso {
    kernel: NsmAniso,
    grid: Grid2D<f64>,
    current: Vec<f64>,
    nx: usize,
    ny: usize,
}

#[pymethods]
impl PyNonSeparable2DAniso {
    #[new]
    #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, beta_values, u0, *,
                        beta_norm_bound = None, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
            let u0_vec = extract_u0_flat_2d(u0, nx, ny)?;
            let beta_raw = extract_finite_beta(beta_values, nx, ny)?;
            let norm_bound = beta_norm_bound.unwrap_or_else(|| {
                let m = magnitude_max(&beta_raw);
                if m == 0.0 {
                    0.0
                } else {
                    m * 1.1
                }
            });
            let beta_cls = closure_2d_from_array(beta_values, xmin, xmax, nx, ymin, ymax, ny)?;
            let (gx, gy, grid) = build_grid_2d(xmin, xmax, nx, ymin, ymax, ny, policy)?;
            let kernel = nonseparable_mixed_closure::with_closure_beta(
                unit_diffusion_1d(gx),
                unit_diffusion_1d(gy),
                beta_cls,
                norm_bound,
                grid,
            )
            .map_err(|e| from_core(&e))?;
            Ok(Self {
                kernel,
                grid,
                current: u0_vec,
                nx,
                ny,
            })
        })
    }

    /// Evolve state by time ``t`` using ``n_steps`` Chernoff steps.
    ///
    /// Returns flat row-major ``numpy.ndarray[float64]`` of length ``nx * ny``.
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            validate_t_pos(t, n_steps)?;
            let tau = t / n_steps as f64;
            let kernel = self.kernel.clone();
            let grid = self.grid;
            let input = self.current.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| evolve_nonsep_aniso(kernel, grid, input, tau, n_steps));
            let out = result.map_err(|e| from_core(&e))?;
            self.current = out.clone();
            Ok(out.as_slice().to_pyarray(py))
        })
    }

    /// Number of state values (``nx * ny``).
    fn __len__(&self) -> usize {
        self.nx * self.ny
    }

    fn __repr__(&self) -> String {
        format!("NonSeparable2DAniso(nx={}, ny={})", self.nx, self.ny)
    }
}

/// Extract and validate `beta_values` as a finite `f64` Vec of length `nx * ny`.
fn extract_finite_beta(
    beta_values: &Bound<'_, pyo3::types::PyAny>,
    nx: usize,
    ny: usize,
) -> pyo3::PyResult<Vec<f64>> {
    let beta_raw: Vec<f64> = beta_values
        .extract::<Vec<f64>>()
        .map_err(|_| new_pyerr("GridMismatch", "beta_values must be numpy.ndarray[float64]"))?;
    if beta_raw.len() != nx * ny {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("beta_values length {} != nx*ny={}", beta_raw.len(), nx * ny),
        ));
    }
    for &v in &beta_raw {
        if !v.is_finite() {
            return Err(new_pyerr("NanInf", "beta_values contains NaN or Inf"));
        }
    }
    Ok(beta_raw)
}

fn evolve_nonsep_aniso(
    kernel: NsmAniso,
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
