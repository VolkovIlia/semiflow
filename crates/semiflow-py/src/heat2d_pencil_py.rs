//! Full-grid `a_x(x,y)` / `a_y(x,y)` for `Heat2DVarA` (#21, ADR-0196).
//!
//! `Heat2DVarA` takes `a_x(x)` and `a_y(y)` — each diagonal coefficient may vary
//! only along **its own** axis, because `Strang2D` applies one shared kernel to
//! every pencil. The Heston generator, after the standard log-price +
//! decorrelation transform `z = x − (ρ/ξ)v`, is cross-term-free:
//!
//! ```text
//!   ∂_τ u = ½v(1−ρ²)·u_zz + ½ξ²v·u_vv + b_z(v)·u_z + κ(θ−v)·u_v − r·u
//! ```
//!
//! but **both** diagonal coefficients depend on `v` — each varies along the
//! *other* axis. That single restriction is what stood between the library and a
//! one-kernel order-2 Heston solve.
//!
//! `Strang2DPencil` carries one 1-D kernel per pencil; this module builds those
//! kernel lists from the two full-grid arrays.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use semiflow::{
    strang2d_pencil::Strang2DPencil, ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D,
    GridFn2D, ScratchPool,
};

use crate::{anisotropic_nd_helpers::interp_1d, error::new_pyerr};
use pyo3::prelude::*;

/// The per-pencil composition used by `Heat2DVarA.with_grid_arrays`.
pub(crate) type PencilStrang2D =
    Strang2DPencil<DiffusionChernoff<f64>, DiffusionChernoff<f64>, f64>;

/// Build a 1-D diffusion kernel from one pencil's coefficient profile.
///
/// `a' ≡ 0` / `a'' ≡ 0` deliberately, matching the existing `Heat2DVarA`
/// contract and its documented non-divergence operator `a·u_xx` exactly — see
/// the open question recorded at `anisotropic_nd3.rs::build_axis_diff`. This
/// entry point changes *which coefficients are expressible*, not which operator
/// is being discretised; mixing the two would confound them.
fn pencil_kernel(profile: Vec<f64>, amin: f64, amax: f64, n: usize, g: Grid1D<f64>)
    -> DiffusionChernoff<f64>
{
    let norm = profile.iter().copied().fold(0.0_f64, f64::max);
    let arc = Arc::new(profile);
    DiffusionChernoff::with_closure(
        move |t: f64| interp_1d(&arc, amin, amax, n, t),
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        norm,
        g,
    )
}

/// Validate a full-grid coefficient array: length `nx*ny`, finite, strictly > 0.
fn validate_grid_coeff(v: &[f64], nx: usize, ny: usize, name: &str) -> PyResult<()> {
    if v.len() != nx * ny {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("{name} has length {}, expected nx*ny={}", v.len(), nx * ny),
        ));
    }
    for &x in v {
        if !x.is_finite() {
            return Err(new_pyerr("NanInf", &format!("{name} contains NaN or Inf")));
        }
        if x <= 0.0 {
            return Err(new_pyerr("OutOfDomain", &format!("{name} must be > 0 everywhere")));
        }
    }
    Ok(())
}

/// Build the per-pencil composition from two full-grid arrays.
///
/// Layout is `values[j*nx + i]` = value at `(x_i, y_j)` — the same x-fastest
/// convention as `NonSeparable2D.with_beta_array`. That makes each X-pencil
/// profile a **contiguous** row slice and each Y-pencil profile a strided
/// gather.
///
/// # Errors
/// `GridMismatch` / `NanInf` / `OutOfDomain` on a malformed coefficient array.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pencil_strang2d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    ax: &[f64],
    ay: &[f64],
    policy: semiflow::BoundaryPolicy,
) -> PyResult<(PencilStrang2D, Grid2D<f64>)> {
    validate_grid_coeff(ax, nx, ny, "a_x")?;
    validate_grid_coeff(ay, nx, ny, "a_y")?;
    let gx = Grid1D::new(xmin, xmax, nx)
        .map_err(|e| crate::error::from_core(&e))?
        .with_boundary(policy);
    let gy = Grid1D::new(ymin, ymax, ny)
        .map_err(|e| crate::error::from_core(&e))?
        .with_boundary(policy);
    let grid = Grid2D::new(gx, gy);

    // X-pencil j: a_x(·, y_j) — contiguous row slice, length nx.
    let x_kernels: Vec<DiffusionChernoff<f64>> = (0..ny)
        .map(|j| pencil_kernel(ax[j * nx..(j + 1) * nx].to_vec(), xmin, xmax, nx, gx))
        .collect();
    // Y-pencil i: a_y(x_i, ·) — strided gather, length ny.
    let y_kernels: Vec<DiffusionChernoff<f64>> = (0..nx)
        .map(|i| {
            let col: Vec<f64> = (0..ny).map(|j| ay[j * nx + i]).collect();
            pencil_kernel(col, ymin, ymax, ny, gy)
        })
        .collect();

    let strang = Strang2DPencil::new(x_kernels, y_kernels, grid)
        .map_err(|e| crate::error::from_core(&e))?;
    Ok((strang, grid))
}
pub(crate) fn evolve_pencil_2d(
    strang: &crate::heat2d_pencil_py::PencilStrang2D,
    grid: Grid2D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let mut state = GridFn2D::new(grid, input)?;
    let mut dst = GridFn2D::new(grid, vec![0.0; state.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        strang.apply_into(tau, &state, &mut dst, &mut scratch)?;
        core::mem::swap(&mut state, &mut dst);
    }
    Ok(state.values)
}

