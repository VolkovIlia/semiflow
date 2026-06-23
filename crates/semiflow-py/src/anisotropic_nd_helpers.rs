//! Shared helpers for Wave P7 (`anisotropic_nd`*.rs modules).
//!
//! Extracted from `anisotropic_nd.rs` for suckless file-size compliance.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use pyo3::prelude::*;
use semiflow::{
    grid_nd::GridND, shift_nd::AnisotropicShiftChernoffND, BoundaryPolicy, Grid1D, Grid2D,
};

use crate::error::{from_core, new_pyerr};

// ---------------------------------------------------------------------------
// Array extraction helpers
// ---------------------------------------------------------------------------

/// Extract a flat `Vec<f64>` of exactly `expected_len` finite values.
pub(crate) fn extract_finite_f64_vec(
    py_arr: &Bound<'_, PyAny>,
    expected_len: usize,
    name: &'static str,
) -> PyResult<Vec<f64>> {
    let v: Vec<f64> = py_arr.extract::<Vec<f64>>().map_err(|_| {
        new_pyerr(
            "GridMismatch",
            &format!("{name} must be numpy.ndarray[float64]"),
        )
    })?;
    if v.len() != expected_len {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("{name} length {} != expected {}", v.len(), expected_len),
        ));
    }
    for &val in &v {
        if !val.is_finite() {
            return Err(new_pyerr("NanInf", &format!("{name} contains NaN or Inf")));
        }
    }
    Ok(v)
}

/// Extract a positive-definite 1D coefficient array.
pub(crate) fn extract_pos_coeff_vec(
    py_arr: &Bound<'_, PyAny>,
    expected_len: usize,
    name: &'static str,
) -> PyResult<Vec<f64>> {
    let v = extract_finite_f64_vec(py_arr, expected_len, name)?;
    for &val in &v {
        if val <= 0.0 {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("{name} must be > 0 everywhere"),
            ));
        }
    }
    Ok(v)
}

/// Extract and validate a flat u0 of length nx*ny.
pub(crate) fn extract_u0_flat_2d(
    u0: &Bound<'_, PyAny>,
    nx: usize,
    ny: usize,
) -> PyResult<Vec<f64>> {
    extract_finite_f64_vec(u0, nx * ny, "u0")
}

/// Build `(gx, gy, grid)` with a boundary policy.
pub(crate) fn build_grid_2d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    policy: BoundaryPolicy,
) -> PyResult<(Grid1D<f64>, Grid1D<f64>, Grid2D<f64>)> {
    let gx = Grid1D::new(xmin, xmax, nx)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    let gy = Grid1D::new(ymin, ymax, ny)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    let grid = Grid2D::new(gx, gy);
    Ok((gx, gy, grid))
}

// ---------------------------------------------------------------------------
// Interpolation and indexing helpers
// ---------------------------------------------------------------------------

/// Linear interpolation of a pre-sampled 1D array at position x.
///
/// Clamps x to [xmin, xmax]; handles edge nodes safely.
pub(crate) fn interp_1d(vals: &[f64], xmin: f64, xmax: f64, n: usize, x: f64) -> f64 {
    if n == 1 {
        return vals[0];
    }
    let fi = ((x - xmin) / (xmax - xmin)) * (n as f64 - 1.0);
    let fi = fi.clamp(0.0, (n - 1) as f64);
    let i0 = (fi as usize).min(n - 2);
    let t = fi - i0 as f64;
    vals[i0] * (1.0 - t) + vals[i0 + 1] * t
}

/// Compute flat index for a 2D grid point from physical coords `x`.
///
/// Storage: axis-0 fastest: flat = k0 + k1 * n0.
pub(crate) fn nd_flat_idx_2(x: &[f64; 2], ns: &[usize; 2], axes: &[(f64, f64, usize)]) -> usize {
    let k0 = phys_to_idx(x[0], axes[0].0, axes[0].1, ns[0]);
    let k1 = phys_to_idx(x[1], axes[1].0, axes[1].1, ns[1]);
    k0 + k1 * ns[0]
}

/// Compute flat index for a 3D grid point from physical coords `x`.
///
/// Storage: axis-0 fastest: flat = k0 + k1*n0 + k2*n0*n1.
pub(crate) fn nd_flat_idx_3(x: &[f64; 3], ns: &[usize; 3], axes: &[(f64, f64, usize)]) -> usize {
    let k0 = phys_to_idx(x[0], axes[0].0, axes[0].1, ns[0]);
    let k1 = phys_to_idx(x[1], axes[1].0, axes[1].1, ns[1]);
    let k2 = phys_to_idx(x[2], axes[2].0, axes[2].1, ns[2]);
    k0 + k1 * ns[0] + k2 * ns[0] * ns[1]
}

/// Map a physical coordinate to the nearest grid index.
#[inline]
pub(crate) fn phys_to_idx(x: f64, xmin: f64, xmax: f64, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let fi = (x - xmin) / (xmax - xmin) * (n as f64 - 1.0);
    (fi.round() as isize).clamp(0, n as isize - 1) as usize
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate t >= 0 and `n_steps` >= 1.
pub(crate) fn validate_t_nsteps(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

/// Validate t > 0 and `n_steps` >= 1 (for Strang-based kernels).
pub(crate) fn validate_t_pos(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and > 0"));
    }
    Ok(())
}

/// Validate tau > 0 and `n_steps` >= 1 (for Heat2D/3DVarA).
pub(crate) fn validate_tau_nsteps(tau: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "tau must be finite and > 0"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ND3 kernel construction helpers (moved from anisotropic_nd.rs)
// ---------------------------------------------------------------------------

type Arcs3 = (
    Arc<Vec<f64>>,
    Arc<Vec<f64>>,
    Arc<Vec<f64>>,
    [usize; 3],
    [(f64, f64, usize); 3],
    [(f64, f64, usize); 3],
    [(f64, f64, usize); 3],
);

/// Build an `AnisotropicShiftChernoffND<f64, 3>` from pre-sampled coefficient arrays.
#[allow(clippy::type_complexity)]
pub(crate) fn build_aniso_nd3_kernel(
    nx: usize,
    ny: usize,
    nz: usize,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: f64,
    zmax: f64,
    a_raw: Vec<f64>,
    b_raw: Vec<f64>,
    c_raw: Vec<f64>,
) -> Result<AnisotropicShiftChernoffND<f64, 3>, semiflow::SemiflowError> {
    let grid_nd = build_grid_nd3(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz)?;
    let (a_arc2, b_arc2, c_arc2, ns, axes, axes2, axes3) = pack_nd3_arcs(
        nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, a_raw, b_raw, c_raw,
    );
    AnisotropicShiftChernoffND::<f64, 3>::new(
        move |x, mat| {
            let flat = nd_flat_idx_3(x, &ns, &axes);
            let base = flat * 9;
            for r in 0..3 {
                for c_idx in 0..3 {
                    mat.set(r, c_idx, a_arc2[base + r * 3 + c_idx]);
                }
            }
        },
        move |x, bv| {
            let flat = nd_flat_idx_3(x, &ns, &axes2);
            let base = flat * 3;
            bv[0] = b_arc2[base];
            bv[1] = b_arc2[base + 1];
            bv[2] = b_arc2[base + 2];
        },
        move |x| {
            let flat = nd_flat_idx_3(x, &ns, &axes3);
            c_arc2[flat]
        },
        grid_nd,
    )
}

fn build_grid_nd3(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    zmin: f64,
    zmax: f64,
    nz: usize,
) -> Result<GridND<f64, 3>, semiflow::SemiflowError> {
    GridND::<f64, 3>::new([
        Grid1D::new(xmin, xmax, nx)?,
        Grid1D::new(ymin, ymax, ny)?,
        Grid1D::new(zmin, zmax, nz)?,
    ])
}

#[allow(clippy::type_complexity)]
fn pack_nd3_arcs(
    nx: usize,
    ny: usize,
    nz: usize,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: f64,
    zmax: f64,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
) -> Arcs3 {
    let axes = [(xmin, xmax, nx), (ymin, ymax, ny), (zmin, zmax, nz)];
    let a_arc = Arc::new(a);
    let b_arc = Arc::new(b);
    let c_arc = Arc::new(c);
    (
        Arc::clone(&a_arc),
        Arc::clone(&b_arc),
        Arc::clone(&c_arc),
        [nx, ny, nz],
        axes,
        axes,
        axes,
    )
}
