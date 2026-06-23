//! Read-only Laplacian introspection helpers for `PyLaplacian` (Issue #5, ADR-0115).
//!
//! All functions are `pub(crate)` and called from the single `#[pymethods]`
//! block in `graph_extra.rs`.  Keeping them here preserves the 500-line cap
//! on `graph_extra.rs` without needing a second `#[pymethods]` block (which
//! would require the `multiple-pymethods` pyo3 feature).
//!
//! # Invariants
//!
//! - All functions return **copies** of the frozen CSR data.
//! - No mutable handle to `Arc<Laplacian<f64>>` is exposed.
//! - The frozen-topology invariant (ADR-0048) is preserved.
//!
//! # Memory cost of `laplacian_to_dense`
//!
//! Allocates O(n²) floats.  Inputs where `n * n` overflows `usize` are
//! rejected with `SemiflowError(OutOfDomain)`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_possible_wrap, clippy::unnecessary_wraps)]

use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use std::sync::Arc;

use semiflow::Laplacian;

use crate::error::new_pyerr;

// ---------------------------------------------------------------------------
// to_dense
// ---------------------------------------------------------------------------

/// Reconstruct the dense `n × n` row-major matrix from a `Laplacian<f64>`.
///
/// Returns a newly allocated `ndarray::Array2<f64>` wrapped as `PyArray2`.
/// Raises `OutOfDomain` if `n * n` would overflow `usize`.
pub(crate) fn laplacian_to_dense<'py>(
    py: Python<'py>,
    inner: &Arc<Laplacian<f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = inner.n_nodes();
    let size = n
        .checked_mul(n)
        .ok_or_else(|| new_pyerr("OutOfDomain", "n * n overflows usize"))?;
    let mut buf = vec![0.0_f64; size];
    let row_ptr = inner.row_ptr();
    let col_idx = inner.col_idx();
    let vals = inner.vals();
    for row in 0..n {
        for k in row_ptr[row]..row_ptr[row + 1] {
            let col = col_idx[k] as usize;
            buf[row * n + col] = vals[k];
        }
    }
    let arr =
        numpy::ndarray::Array2::from_shape_vec((n, n), buf).expect("shape matches allocation");
    Ok(arr.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// CSR accessors (copies)
// ---------------------------------------------------------------------------

/// Copy of the `row_ptr` slice as an `int64` numpy array.
pub(crate) fn laplacian_row_ptr<'py>(
    py: Python<'py>,
    inner: &Arc<Laplacian<f64>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let v: Vec<i64> = inner.row_ptr().iter().map(|&x| x as i64).collect();
    Ok(v.as_slice().to_pyarray(py))
}

/// Copy of the `col_idx` slice as an `int64` numpy array.
pub(crate) fn laplacian_col_idx<'py>(
    py: Python<'py>,
    inner: &Arc<Laplacian<f64>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let v: Vec<i64> = inner.col_idx().iter().map(|&x| i64::from(x)).collect();
    Ok(v.as_slice().to_pyarray(py))
}

/// Copy of the `vals` slice as a `float64` numpy array.
pub(crate) fn laplacian_vals<'py>(
    py: Python<'py>,
    inner: &Arc<Laplacian<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(inner.vals().to_pyarray(py))
}
