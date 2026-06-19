//! Convert a pre-sampled 2-D `numpy.ndarray[float64]` into a zero-GIL-cost
//! closure for use with `nonseparable_mixed_closure::with_closure_beta`.
//!
//! Bilinear interpolation is used inside the domain; boundary values are
//! clamped for out-of-domain queries.  All Python work happens before the
//! `py.detach` window — the produced closure is pure-Rust.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use pyo3::prelude::*;

use crate::error::new_pyerr;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Convert a `numpy.ndarray[float64]` shape `(nx, ny)` row-major into a
/// `Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>` bilinear-interpolant closure.
///
/// The closure performs bilinear interpolation for `(x, y)` inside
/// `[xmin, xmax] × [ymin, ymax]` and clamps to the nearest boundary value
/// outside.  All computation is pure-Rust (zero GIL cost inside `py.detach`).
///
/// # Errors
/// - `SemiflowError(kind='GridMismatch')` if `array.len() != nx * ny`.
/// - `SemiflowError(kind='NanInf')` if any value is non-finite.
pub(crate) fn closure_2d_from_array(
    py_arr: &Bound<'_, PyAny>,
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
) -> PyResult<Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'static>> {
    let values: Vec<f64> = py_arr.extract::<Vec<f64>>().map_err(|_| {
        new_pyerr(
            "GridMismatch",
            "2D coefficient array must be numpy.ndarray[float64]",
        )
    })?;
    validate_2d_array(&values, nx, ny)?;
    let dx = if nx > 1 {
        (xmax - xmin) / (nx as f64 - 1.0)
    } else {
        1.0
    };
    let dy = if ny > 1 {
        (ymax - ymin) / (ny as f64 - 1.0)
    } else {
        1.0
    };
    let shared = Arc::new(values);
    Ok(Arc::new(move |x: f64, y: f64| {
        bilinear_interp(&shared, xmin, dx, nx, ymin, dy, ny, x, y)
    }))
}

/// Return `max(|v|)` for `v` over the array, used for auto-computing norm bounds.
///
/// Returns `0.0` if the slice is empty.
pub(crate) fn magnitude_max(arr: &[f64]) -> f64 {
    arr.iter().copied().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Validate that `values` has length `nx*ny` and all elements are finite.
fn validate_2d_array(values: &[f64], nx: usize, ny: usize) -> PyResult<()> {
    let expected = nx * ny;
    if values.len() != expected {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("2D array length {} != nx*ny={}", values.len(), expected),
        ));
    }
    for &v in values {
        if !v.is_finite() {
            return Err(new_pyerr(
                "NanInf",
                "2D coefficient array contains NaN or Inf",
            ));
        }
    }
    Ok(())
}

/// Bilinear interpolation at `(x, y)` in a row-major `(nx, ny)` grid.
///
/// Clamps `(x, y)` to the domain boundary for out-of-domain queries.
/// Row-major layout: `values[j * nx + i]` = value at `(x_i, y_j)`.
fn bilinear_interp(
    values: &[f64],
    xmin: f64,
    dx: f64,
    nx: usize,
    ymin: f64,
    dy: f64,
    ny: usize,
    x: f64,
    y: f64,
) -> f64 {
    if nx == 0 || ny == 0 {
        return 0.0;
    }
    if nx == 1 && ny == 1 {
        return values[0];
    }
    // Fractional indices, clamped.
    let fi = ((x - xmin) / dx).clamp(0.0, (nx - 1) as f64);
    let fj = ((y - ymin) / dy).clamp(0.0, (ny - 1) as f64);
    let i0 = (fi as usize).min(nx - 1);
    let j0 = (fj as usize).min(ny - 1);
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(ny - 1);
    let tx = fi - i0 as f64;
    let ty = fj - j0 as f64;
    // Bilinear combination of the four corners.
    let v00 = values[j0 * nx + i0];
    let v10 = values[j0 * nx + i1];
    let v01 = values[j1 * nx + i0];
    let v11 = values[j1 * nx + i1];
    let lo = v00 * (1.0 - tx) + v10 * tx;
    let hi = v01 * (1.0 - tx) + v11 * tx;
    lo * (1.0 - ty) + hi * ty
}
