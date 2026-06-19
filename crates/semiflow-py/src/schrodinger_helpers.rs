//! Internal helpers for `Schrodinger1D`: builders, compute helpers, extractors.
//!
//! Split from `schrodinger.rs` for suckless file-size compliance.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use numpy::Complex64;
use pyo3::prelude::*;
use semiflow_core::{
    ChernoffFunction, Diffusion4thChernoff, GridFn1D, SchrodingerChernoff, SchrodingerState,
};

use crate::{
    coeff::closure_from_array,
    error::{from_core, new_pyerr},
    schrodinger::Schrodinger1DInner,
};

// ---------------------------------------------------------------------------
// Complex array assembly
// ---------------------------------------------------------------------------

/// Assemble a `numpy.ndarray[complex128]` from separate re/im `Vec<f64>` slices.
///
/// `Complex64` in the numpy crate is `num_complex::Complex<f64>`, which maps
/// to numpy's `complex128` dtype (`numpy.cdouble`).
pub(crate) fn assemble_complex128<'py>(
    py: Python<'py>,
    re: &[f64],
    im: &[f64],
) -> PyResult<Bound<'py, numpy::PyArray1<Complex64>>> {
    use numpy::ToPyArray;
    let buf: Vec<Complex64> = re
        .iter()
        .zip(im.iter())
        .map(|(&r, &i)| Complex64::new(r, i))
        .collect();
    Ok(buf.as_slice().to_pyarray(py))
}

// ---------------------------------------------------------------------------
// Phase 2: pure-Rust compute helper (called inside py.detach)
// ---------------------------------------------------------------------------

/// Evolve `SchrodingerChernoff` by `t` using `n_steps` signed-`τ` steps.
///
/// D2 (ADR-0113): uses a manual `apply_into` loop with `τ = t / n_steps`,
/// which is negative when `t < 0`.  `ChernoffSemigroup::evolve` is NOT used
/// here because it rejects negative `t` via its own guard.  The kernel
/// `apply_strang_step` is analytic in `τ` and handles signed `τ` correctly
/// (round-trip residual 1.19e-13 verified by architect).
///
/// Returns `(psi_re_out, psi_im_out)`.  No Python types cross this boundary.
///
/// # Errors
/// Propagates `SemiflowError` from grid/state construction or `apply_into`.
pub(crate) fn compute_schrodinger(
    chernoff: SchrodingerChernoff<f64>,
    grid: semiflow_core::Grid1D<f64>,
    re_in: Vec<f64>,
    im_in: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<(Vec<f64>, Vec<f64>), semiflow_core::SemiflowError> {
    use semiflow_core::ScratchPool;

    #[allow(clippy::cast_precision_loss)]
    let tau = t / n_steps as f64;
    let psi_re = GridFn1D::new(grid, re_in)?;
    let psi_im = GridFn1D::new(grid, im_in)?;
    let mut state = SchrodingerState::new(psi_re, psi_im)?;
    let mut scratch = ScratchPool::new();

    for _ in 0..n_steps {
        let mut next = state.clone();
        chernoff.apply_into(tau, &state, &mut next, &mut scratch)?;
        state = next;
    }

    Ok((state.psi_re.values, state.psi_im.values))
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

/// Build `Schrodinger1DInner` with zero potential `V = 0`.
///
/// # Errors
/// Propagates `SemiflowError` from grid, state, or kernel construction.
pub(crate) fn build_schrodinger_zero_v(
    xmin: f64,
    xmax: f64,
    n: usize,
    policy: semiflow_core::BoundaryPolicy,
    re: &[f64],
    im: &[f64],
) -> Result<Schrodinger1DInner, semiflow_core::SemiflowError> {
    validate_psi_finite(re, im)?;
    let grid = semiflow_core::Grid1D::new(xmin, xmax, n)?.with_boundary(policy);
    let kinetic = Diffusion4thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
    let chernoff = SchrodingerChernoff::new(kinetic, |_| 0.0)?;
    let psi_re = GridFn1D::new(grid, re.to_vec())?;
    let psi_im = GridFn1D::new(grid, im.to_vec())?;
    let state = SchrodingerState::new(psi_re, psi_im)?;
    Ok(Schrodinger1DInner { chernoff, state })
}

/// Build `Schrodinger1DInner` with a pre-sampled potential array `V(x)`.
///
/// # Errors
/// Propagates from `closure_from_array`, grid, state, or kernel construction.
pub(crate) fn build_schrodinger_with_v(
    xmin: f64,
    xmax: f64,
    n: usize,
    policy: semiflow_core::BoundaryPolicy,
    v_arr: &Bound<'_, PyAny>,
    re: &[f64],
    im: &[f64],
) -> PyResult<Schrodinger1DInner> {
    validate_psi_finite(re, im).map_err(|e| from_core(&e))?;
    let v_closure = closure_from_array(v_arr, xmin, xmax, n)?;
    let grid = semiflow_core::Grid1D::new(xmin, xmax, n)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    let kinetic = Diffusion4thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
    let chernoff = SchrodingerChernoff::new(kinetic, v_closure).map_err(|e| from_core(&e))?;
    let psi_re = GridFn1D::new(grid, re.to_vec()).map_err(|e| from_core(&e))?;
    let psi_im = GridFn1D::new(grid, im.to_vec()).map_err(|e| from_core(&e))?;
    let state = SchrodingerState::new(psi_re, psi_im).map_err(|e| from_core(&e))?;
    Ok(Schrodinger1DInner { chernoff, state })
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

/// Split a Python complex128 array into `(Vec<f64>, Vec<f64>)` re/im parts.
///
/// Reads `.real` and `.imag` attributes (works for any numpy complex array).
pub(crate) fn split_complex_array(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let re_attr = obj
        .getattr("real")
        .map_err(|_| new_pyerr("GridMismatch", "psi0 must be a numpy.ndarray[complex128]"))?;
    let im_attr = obj
        .getattr("imag")
        .map_err(|_| new_pyerr("GridMismatch", "psi0 must be a numpy.ndarray[complex128]"))?;
    let re: Vec<f64> = re_attr
        .extract::<Vec<f64>>()
        .map_err(|_| new_pyerr("GridMismatch", "psi0.real must be extractable as float64"))?;
    let im: Vec<f64> = im_attr
        .extract::<Vec<f64>>()
        .map_err(|_| new_pyerr("GridMismatch", "psi0.imag must be extractable as float64"))?;
    if re.len() != im.len() {
        return Err(new_pyerr(
            "GridMismatch",
            "psi0.real and psi0.imag length mismatch",
        ));
    }
    Ok((re, im))
}

/// Extract a Python array-like as `Vec<f64>` with a named error message.
pub(crate) fn extract_f64_vec(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        new_pyerr(
            "GridMismatch",
            &format!("{name} must be numpy.ndarray[float64]"),
        )
    })
}

// ---------------------------------------------------------------------------
// Validation and small helpers
// ---------------------------------------------------------------------------

/// Validate that both `re` and `im` slices contain only finite values.
pub(crate) fn validate_psi_finite(
    re: &[f64],
    im: &[f64],
) -> Result<(), semiflow_core::SemiflowError> {
    for &v in re.iter().chain(im.iter()) {
        if !v.is_finite() {
            return Err(semiflow_core::SemiflowError::DomainViolation {
                what: "psi0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}

/// Validate time-evolution parameters.
///
/// D2 (ADR-0113): `t < 0` is now VALID (backward unitary evolution).
/// Only non-finite `t` and `n_steps == 0` are rejected.
pub(crate) fn validate_evolve_params(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() {
        return Err(new_pyerr(
            "OutOfDomain",
            "t must be finite (negative t is allowed for backward evolution)",
        ));
    }
    Ok(())
}

/// Compute grid spacing `dx = (xmax - xmin) / (n - 1)`.
pub(crate) fn compute_dx(xmin: f64, xmax: f64, n: usize) -> f64 {
    if n > 1 {
        (xmax - xmin) / (n as f64 - 1.0)
    } else {
        1.0
    }
}

/// Unit diffusion coefficient `a(x) = 1.0` (free-particle kinetic operator).
extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}

/// Zero coefficient: `a'(x) = a''(x) = 0` for constant `a`.
extern "Rust" fn zero_d(_: f64) -> f64 {
    0.0
}
