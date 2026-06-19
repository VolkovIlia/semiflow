//! Convert a pre-sampled `numpy.ndarray[float64]` into a zero-GIL-cost
//! [`CoeffClosure`] for use with `DiffusionChernoff::with_closure`.
//!
//! All Python work (GIL acquisition, array extraction, validation) happens
//! before the `py.detach` window. The produced closure does **no** Python
//! work; it clamps out-of-domain queries and interpolates inside the domain.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::match_same_arms,
    clippy::needless_range_loop
)]

use std::sync::Arc;

use pyo3::prelude::*;

use crate::{error::new_pyerr, handle::CoeffClosure};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Convert a `numpy.ndarray[float64]` of length `n` into a [`CoeffClosure`].
///
/// The closure performs cubic-Hermite (Catmull-Rom) interpolation for `x`
/// inside `[xmin, xmax]` and clamps to the nearest boundary value outside.
/// All computation is pure-Rust (zero GIL cost inside `py.detach`).
///
/// # Errors
/// - `SemiflowError(kind='GridMismatch')` if `len(array) != n`.
/// - `SemiflowError(kind='NanInf')` if any value is non-finite.
pub(crate) fn closure_from_array(
    py_arr: &Bound<'_, PyAny>,
    xmin: f64,
    xmax: f64,
    n: usize,
) -> PyResult<CoeffClosure> {
    let values: Vec<f64> = py_arr.extract::<Vec<f64>>().map_err(|_| {
        new_pyerr(
            "GridMismatch",
            "coefficient array must be numpy.ndarray[float64]",
        )
    })?;
    validate_coeff_array(&values, n)?;
    let dx = if n > 1 {
        (xmax - xmin) / (n as f64 - 1.0)
    } else {
        1.0
    };
    let shared = Arc::new(values);
    Ok(Box::new(move |x: f64| {
        interp_catmull_rom(&shared, xmin, dx, x)
    }))
}

/// 4th-order central-difference first derivative of pre-sampled values.
///
/// Uses a 5-point interior stencil and one-sided stencils at/near boundaries.
///
/// # Panics
/// Does not panic; `values` may be empty (returns empty `Vec`).
pub(crate) fn derivative_4th(values: &[f64], dx: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0_f64; n];
    let inv = 1.0 / dx;
    for i in 0..n {
        out[i] = fd1_stencil(values, n, i, inv);
    }
    out
}

/// 4th-order central-difference second derivative of pre-sampled values.
///
/// Uses a 5-point interior stencil and one-sided stencils at/near boundaries.
pub(crate) fn second_derivative_4th(values: &[f64], dx: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0_f64; n];
    let inv2 = 1.0 / (dx * dx);
    for i in 0..n {
        out[i] = fd2_stencil(values, n, i, inv2);
    }
    out
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Validate that `values` has length `n` and all elements are finite.
fn validate_coeff_array(values: &[f64], n: usize) -> PyResult<()> {
    if values.len() != n {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("coefficient array length {} != n={}", values.len(), n),
        ));
    }
    for &v in values {
        if !v.is_finite() {
            return Err(new_pyerr("NanInf", "coefficient array contains NaN or Inf"));
        }
    }
    Ok(())
}

/// Cubic-Hermite (Catmull-Rom) interpolation at `x` (public to crate).
///
/// Clamps to boundary value for `x` outside `[xmin, xmin + (n-1)*dx]`.
pub(crate) fn interp_catmull_rom_pub(values: &[f64], xmin: f64, dx: f64, x: f64) -> f64 {
    interp_catmull_rom(values, xmin, dx, x)
}

/// Cubic-Hermite (Catmull-Rom) interpolation at `x`.
///
/// Clamps to boundary value for `x` outside `[xmin, xmin + (n-1)*dx]`.
fn interp_catmull_rom(values: &[f64], xmin: f64, dx: f64, x: f64) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return values[0];
    }
    // Fractional index.
    let fi = (x - xmin) / dx;
    // Clamp out-of-domain.
    if fi <= 0.0 {
        return values[0];
    }
    if fi >= (n - 1) as f64 {
        return *values.last().unwrap();
    }
    let i = fi as usize;
    let i = i.min(n - 2);
    let t = fi - i as f64; // in [0, 1)
                           // Catmull-Rom: 4 control points p0..p3
    let p1 = values[i];
    let p2 = values[i + 1];
    let p0 = if i == 0 { 2.0 * p1 - p2 } else { values[i - 1] };
    let p3 = if i + 2 < n {
        values[i + 2]
    } else {
        2.0 * p2 - p1
    };
    // Standard Catmull-Rom formula
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// First-derivative finite-difference stencil (4th-order) at index `i`.
///
/// Interior: `(-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12·dx)`.
/// Near boundaries: degrade to 2nd/1st order one-sided stencils.
fn fd1_stencil(v: &[f64], n: usize, i: usize, inv_dx: f64) -> f64 {
    let f = |j: usize| v[j];
    match i {
        0 => inv_dx * (-3.0 * f(0) + 4.0 * f(1) - f(2)) / 2.0,
        1 => inv_dx * (-f(0) + f(2)) / 2.0,
        _ if i == n - 1 => inv_dx * (f(n - 3) - 4.0 * f(n - 2) + 3.0 * f(n - 1)) / 2.0,
        _ if i == n - 2 => inv_dx * (-f(n - 3) + f(n - 1)) / 2.0,
        _ => inv_dx * (-f(i + 2) + 8.0 * f(i + 1) - 8.0 * f(i - 1) + f(i - 2)) / 12.0,
    }
}

/// Second-derivative finite-difference stencil (4th-order) at index `i`.
///
/// Interior: `(-f[i+2] + 16f[i+1] - 30f[i] + 16f[i-1] - f[i-2]) / (12·dx²)`.
/// Near boundaries: degrade to 2nd-order one-sided stencils.
fn fd2_stencil(v: &[f64], n: usize, i: usize, inv_dx2: f64) -> f64 {
    let f = |j: usize| v[j];
    match i {
        0 => inv_dx2 * (f(0) - 2.0 * f(1) + f(2)),
        1 => inv_dx2 * (f(0) - 2.0 * f(1) + f(2)),
        _ if i == n - 1 => inv_dx2 * (f(n - 3) - 2.0 * f(n - 2) + f(n - 1)),
        _ if i == n - 2 => inv_dx2 * (f(n - 3) - 2.0 * f(n - 2) + f(n - 1)),
        _ => {
            inv_dx2 * (-f(i + 2) + 16.0 * f(i + 1) - 30.0 * f(i) + 16.0 * f(i - 1) - f(i - 2))
                / 12.0
        }
    }
}
