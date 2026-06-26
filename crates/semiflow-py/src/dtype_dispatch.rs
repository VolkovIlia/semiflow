//! Shared dtype-dispatch helpers for Issue #3 (f32 opt-in path).
//!
//! ## Design
//!
//! `PyO3` classes cannot be generic across the wheel boundary; the per-class
//! `DtypeKernel` enum mirrors the `KernelVariant` pattern in `adjoint.rs`.
//! Two variants — `F64` and `F32` — each wrapping the concrete core type
//! parametrised over the matching scalar type.
//!
//! ## Boundary casting
//!
//! Input `numpy.ndarray` values are extracted as `Vec<f64>` (the dtype that
//! Python users always provide) and cast to `f32` at the Rust boundary when
//! `dtype="f32"`.  Output is converted back to the chosen dtype before being
//! handed to numpy.
//!
//! ## Validation
//!
//! `parse_dtype` accepts `"f64"` and `"f32"` only; anything else raises
//! `SemiflowError(kind="OutOfDomain")`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_possible_truncation)]

use pyo3::prelude::*;

use crate::error::new_pyerr;

// ---------------------------------------------------------------------------
// Dtype selector
// ---------------------------------------------------------------------------

/// Parsed dtype choice — one of two variants, default F64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dtype {
    F64,
    F32,
}

/// Parse the `dtype` kwarg string into a [`Dtype`] value.
///
/// Accepts `"f64"` (default) and `"f32"`.  Rejects `"f16"`, `"fp16"`,
/// and any other value with `SemiflowError(kind="OutOfDomain")`.
pub(crate) fn parse_dtype(s: &str) -> PyResult<Dtype> {
    match s {
        "f64" => Ok(Dtype::F64),
        "f32" => Ok(Dtype::F32),
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!(
                "dtype '{other}' is not supported; \
                 accepted values: \"f64\" (default), \"f32\". \
                 fp16 is explicitly REJECTED (dep-budget violation)."
            ),
        )),
    }
}

// ---------------------------------------------------------------------------
// Casting helpers (boundary conversion)
// ---------------------------------------------------------------------------

/// Cast a `Vec<f64>` to `Vec<f32>`.
///
/// Used to convert the boundary input before entering the f32 kernel.
/// Non-finite values are preserved as `f32::INFINITY`, `f32::NEG_INFINITY`,
/// or `f32::NAN` — the core validator will reject them with `NanInf`.
#[inline]
pub(crate) fn cast_f64_to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// Cast a `Vec<f32>` back to `Vec<f64>` for the output path.
#[inline]
pub(crate) fn cast_f32_to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// Rejection error for `evolve_batched` when `dtype="f32"` is set at construction.
///
/// Centralises the message across all graph kernel classes that expose `evolve_batched`.
pub(crate) fn reject_f32_for_batched() -> PyErr {
    new_pyerr(
        "OutOfDomain",
        "evolve_batched requires dtype=\"f64\"; use evolve() per channel for f32",
    )
}
