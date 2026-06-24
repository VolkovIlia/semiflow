//! Chunked GIL-cooperative evolve helpers for `Heat1D` (ADR-0141).
//!
//! Extracted from `state_1d.rs` to keep that file within the 500-line suckless
//! budget.  Exports:
//!   - [`compute_chunk`] — pure-Rust ping-pong kernel (no Python types).
//!   - [`build_heat1d_from_arrays`] — core logic for `Heat1D::with_a_array`.
//!   - [`build_presampled_closure`] — build a `CoeffClosure` from a Rust `Vec`.
//!   - [`make_coeff_closure`] — wrap a Python callable in a `CoeffClosure`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::similar_names
)]

use pyo3::{prelude::*, types::PyAnyMethods};

use crate::{
    boundary::parse_boundary,
    coeff::{closure_from_array, derivative_4th, second_derivative_4th},
    dtype_dispatch::Dtype,
    error::from_core,
    handle::{build_heat_closure, CoeffClosure},
    state::{extract_f64_slice, Heat1D},
};

// ---------------------------------------------------------------------------
// Closure factory: wrap a PyObject callable into a CoeffClosure (ADR-0034)
// ---------------------------------------------------------------------------

/// Wrap a Python callable `obj(x: float) -> float` in a `CoeffClosure`.
///
/// Each invocation calls `Python::attach` to re-acquire the GIL before
/// entering Python.  This is required because the closures are stored inside
/// `DiffusionChernoff::with_closure` and are called from within the
/// `py.detach` (GIL-released) window of `Heat1D::evolve` (ADR-0031).
///
/// On Python-side exception the closure returns `f64::NAN`; the Chernoff
/// kernel's `validate_a_x` then converts NAN to a `DomainViolation` error.
pub(crate) fn make_coeff_closure(obj: Py<PyAny>) -> CoeffClosure {
    Box::new(move |x: f64| {
        Python::attach(|py| {
            obj.call1(py, (x,))
                .and_then(|r| r.extract::<f64>(py))
                .unwrap_or(f64::NAN)
        })
    })
}

// ---------------------------------------------------------------------------
// with_a_array implementation helper (extracted to keep method ≤50 lines)
// ---------------------------------------------------------------------------

/// Core logic for `Heat1D::with_a_array`.
///
/// Extracted from the `#[pymethods]` impl to keep the pymethod body ≤50 lines.
///
/// # Errors
/// Propagates `PyResult` errors from array extraction and `build_heat_closure`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_heat1d_from_arrays(
    xmin: f64,
    xmax: f64,
    n: usize,
    a: &Bound<'_, PyAny>,
    u0: &Bound<'_, PyAny>,
    a_prime: Option<&Bound<'_, PyAny>>,
    a_double_prime: Option<&Bound<'_, PyAny>>,
    a_norm_bound: Option<f64>,
    boundary: &str,
) -> PyResult<Heat1D> {
    let policy = parse_boundary(boundary)?;
    let slice = extract_f64_slice(u0)?;
    let a_vals: Vec<f64> = a
        .extract::<Vec<f64>>()
        .map_err(|_| crate::error::new_pyerr("GridMismatch", "a must be numpy.ndarray[float64]"))?;
    let dx = if n > 1 {
        (xmax - xmin) / (n as f64 - 1.0)
    } else {
        1.0
    };
    let ap_vals = match a_prime {
        Some(arr) => arr.extract::<Vec<f64>>().map_err(|_| {
            crate::error::new_pyerr("GridMismatch", "a_prime must be numpy.ndarray[float64]")
        })?,
        None => derivative_4th(&a_vals, dx),
    };
    let app_vals = match a_double_prime {
        Some(arr) => arr.extract::<Vec<f64>>().map_err(|_| {
            crate::error::new_pyerr(
                "GridMismatch",
                "a_double_prime must be numpy.ndarray[float64]",
            )
        })?,
        None => second_derivative_4th(&a_vals, dx),
    };
    let norm = a_norm_bound
        .unwrap_or_else(|| 1.1 * a_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let a_fn = closure_from_array(a, xmin, xmax, n)?;
    let ap_fn = build_presampled_closure(&ap_vals, xmin, xmax, n)?;
    let app_fn = build_presampled_closure(&app_vals, xmin, xmax, n)?;
    let inner = build_heat_closure(
        xmin, xmax, n, 100, norm, &slice, a_fn, ap_fn, app_fn, policy,
    )
    .map_err(|e| from_core(&e))?;
    Ok(Heat1D::from_inner(inner, Dtype::F64))
}

/// Build a `CoeffClosure` from an already-extracted Rust `Vec<f64>`.
///
/// Like `closure_from_array` but takes an owned `Vec` instead of a Python object.
pub(crate) fn build_presampled_closure(
    vals: &[f64],
    xmin: f64,
    xmax: f64,
    n: usize,
) -> PyResult<CoeffClosure> {
    use std::sync::Arc;
    if vals.len() != n {
        return Err(crate::error::new_pyerr(
            "GridMismatch",
            &format!("derivative array length {} != n={}", vals.len(), n),
        ));
    }
    let dx = if n > 1 {
        (xmax - xmin) / (n as f64 - 1.0)
    } else {
        1.0
    };
    let shared = Arc::new(vals.to_vec());
    Ok(Box::new(move |x: f64| {
        crate::coeff::interp_catmull_rom_pub(&shared, xmin, dx, x)
    }))
}

// ---------------------------------------------------------------------------
// Pure-Rust compute kernel (no Python types, called inside py.detach)
// ---------------------------------------------------------------------------

/// Run `k` Chernoff ping-pong steps with fixed `tau` on an owned buffer.
///
/// Equivalent to k applications of `S(tau)` — does NOT create a semigroup
/// with `n = k` (which would recompute `tau` from scratch).  Instead it
/// directly applies the kernel `k` times using the pre-computed `tau`.
///
/// This is the building block for `evolve_chunked`: calling it with
/// `tau = t / total_steps` a total of `total_steps` times (across all chunks)
/// produces a result **bit-identical** to `compute_evolve(..., t, total_steps)`.
///
/// No Python types cross this boundary.  All parameters are `Send + Sync`.
///
/// # Errors
/// Propagates [`semiflow::SemiflowError`] from `apply_into`.
pub(crate) fn compute_chunk(
    chernoff_func: semiflow::DiffusionChernoff<f64>,
    grid: semiflow::Grid1D<f64>,
    input: Vec<f64>,
    tau: f64,
    k: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    use semiflow::{ChernoffFunction, GridFn1D, ScratchPool, State};
    let mut buf_a: GridFn1D<f64> = GridFn1D::new(grid, input)?;
    let mut buf_b: GridFn1D<f64> = buf_a.clone();
    buf_b.zero_into();
    let mut scratch: ScratchPool<f64> = ScratchPool::new();
    let mut src_is_a = true;
    for _ in 0..k {
        if src_is_a {
            chernoff_func.apply_into(tau, &buf_a, &mut buf_b, &mut scratch)?;
        } else {
            chernoff_func.apply_into(tau, &buf_b, &mut buf_a, &mut scratch)?;
        }
        src_is_a = !src_is_a;
    }
    let result = if src_is_a { buf_a } else { buf_b };
    Ok(result.values)
}
