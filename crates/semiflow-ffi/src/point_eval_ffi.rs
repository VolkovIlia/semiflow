//! FFI surface for `PointEval` via `DiffusionChernoff<f64>` (Round 11, M18).
//!
//! Mirrors Python `PointEval` class (`structured_point.rs`, ADR-0080, math §31).
//!
//! Exposes Backend A (1-D unit diffusion): evaluates `(F(τ))^{n_steps} u0` at
//! a single query point `x` using `DiffusionChernoff<f64>` with `a ≡ 1`.
//!
//! ## Entry points
//!
//! - `smf_point_eval_new(xmin, xmax, n, out)` → `SemiflowStatus`
//! - `smf_point_eval_eval_at(ev, tau, u0, u0_len, x, n_steps, out_val)` → `SemiflowStatus`
//! - `smf_point_eval_size(ev)` → `n` (0 if null)
//! - `smf_point_eval_drop(ev)` — null-safe
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
)]

use std::os::raw::c_double;

use semiflow::{
    diffusion::DiffusionChernoff,
    grid_fn::GridFn1D,
    point_eval::PointEval as CorePointEval,
    Grid1D,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle for 1-D unit-diffusion `PointEval`.
///
/// Allocate with `smf_point_eval_new`; free with `smf_point_eval_drop`.
#[repr(C)]
pub struct SmfPointEval {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner wrapper (stores grid geometry only — kernel is stateless)
// ---------------------------------------------------------------------------

struct PointEvalInner {
    xmin: f64,
    xmax: f64,
    n: usize,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Construct a `PointEval` with 1-D unit diffusion (`a ≡ 1`).
///
/// Parameters:
/// - `xmin`, `xmax`: grid boundaries (finite, `xmin < xmax`).
/// - `n`: number of grid nodes (`>= 4`).
/// - `out`: receives handle on success.
///
/// # Safety
/// `out` must be a valid writable `*mut *mut SmfPointEval`.
#[no_mangle]
pub unsafe extern "C" fn smf_point_eval_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    out: *mut *mut SmfPointEval,
) -> SemiflowStatus {
    if out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // Eagerly validate grid params (same checks as Python binding).
        if let Err(e) = Grid1D::new(xmin, xmax, n) {
            return SemiflowStatus::from(&e);
        }
        let raw = Box::into_raw(Box::new(PointEvalInner { xmin, xmax, n }))
            .cast::<SmfPointEval>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// eval_at
// ---------------------------------------------------------------------------

/// Evaluate `(F(τ))^{n_steps} u0` at point `x`.
///
/// `u0` is a flat `f64` array of length `n`.
/// Returns the scalar approximation at `x` through `*out_val`.
/// `n_steps >= 1`, `tau >= 0` and finite.
///
/// # Safety
/// `ev` live from `smf_point_eval_new`;
/// `u0` readable `u0_len` f64s; `out_val` writable f64 pointer.
#[no_mangle]
pub unsafe extern "C" fn smf_point_eval_eval_at(
    ev: *const SmfPointEval,
    tau: c_double,
    u0: *const c_double,
    u0_len: usize,
    x: c_double,
    n_steps: u32,
    out_val: *mut c_double,
) -> SemiflowStatus {
    if ev.is_null() || u0.is_null() || out_val.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if n_steps == 0 {
            return SemiflowStatus::OutOfDomain;
        }
        if !tau.is_finite() || tau < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        let inner = unsafe { &*ev.cast::<PointEvalInner>() };
        if u0_len != inner.n {
            return SemiflowStatus::GridMismatch;
        }
        let vals = unsafe { std::slice::from_raw_parts(u0, inner.n) };
        for &v in vals {
            if !v.is_finite() {
                return SemiflowStatus::NanInf;
            }
        }
        match eval_at_rust(inner.xmin, inner.xmax, inner.n, vals, tau, x, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(scalar) => {
                unsafe { *out_val = scalar };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return grid node count `n`; 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_point_eval_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_point_eval_size(ev: *const SmfPointEval) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<PointEvalInner>() };
    inner.n
}

/// Free a `SmfPointEval`. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_point_eval_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_point_eval_drop(ev: *mut SmfPointEval) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<PointEvalInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// Pure-Rust compute (GIL-free equivalent)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn eval_at_rust(
    xmin: f64,
    xmax: f64,
    n: usize,
    vals: &[f64],
    tau: f64,
    x: f64,
    n_steps: u32,
) -> Result<f64, semiflow::SemiflowError> {
    let grid = Grid1D::new(xmin, xmax, n)?;
    let kernel = DiffusionChernoff::new(
        |_: f64| 1.0_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        1.0_f64,
        grid,
    );
    let src = GridFn1D { values: vals.to_vec(), grid };
    kernel.eval_at(tau, &src, &[x], n_steps)
}
