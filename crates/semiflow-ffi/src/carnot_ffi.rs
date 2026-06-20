//! FFI surface for `ComplexTripleJump` (Round 11, F4).
//!
//! Mirrors Python `ComplexTripleJumpV8` class (`carnot_complex_py.rs`, ADR-0138).
//!
//! ## Scope (ABI-safety invariant, ADR-0138)
//!
//! No `Complex<f64>`, `CplxGridFn5`, or `SemiflowComplex` type crosses the
//! boundary. Only `apply_real` (real input → real output) is exposed.
//! The flat `f64` buffer is the only wire format.
//!
//! D=5 ONLY (filiform-N5 Carnot group is fixed by the kernel).
//!
//! ## Entry points
//!
//! - `smf_carnot_ctj_new(domain_lo, domain_hi, n_per_axis, out)` → `SemiflowStatus`
//! - `smf_carnot_ctj_apply_real(ev, tau, u0, u0_len, dst, dst_len)` → `SemiflowStatus`
//! - `smf_carnot_ctj_size(ev)` → `n_per_axis^5` (0 if null)
//! - `smf_carnot_ctj_verify_gamma_star()` → 1 if passes, 0 otherwise
//! - `smf_carnot_ctj_drop(ev)` — null-safe
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

use semiflow_core::{
    carnot_complex::ComplexTripleJump,
    grid_nd::{GridFnND, GridND},
    Grid1D,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const D: usize = 5;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a `ComplexTripleJump` evolver (filiform-N5, D=5 fixed).
///
/// Allocate with `smf_carnot_ctj_new`; free with `smf_carnot_ctj_drop`.
/// Only `apply_real` is exposed — no complex buffer crossing (ADR-0138).
#[repr(C)]
pub struct SmfCarnotCtj {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner wrapper
// ---------------------------------------------------------------------------

struct CarnotCtjInner {
    domain_lo: f64,
    domain_hi: f64,
    n_per_axis: usize,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Construct a `ComplexTripleJump` evolver.
///
/// Parameters:
/// - `domain_lo`: lower bound of each axis (finite, same for all 5 axes).
/// - `domain_hi`: upper bound (`> domain_lo`).
/// - `n_per_axis`: grid nodes per axis (`>= 4`).
/// - `out`: receives the handle on success.
///
/// # Safety
/// `out` must be a valid writable `*mut *mut SmfCarnotCtj`.
#[no_mangle]
pub unsafe extern "C" fn smf_carnot_ctj_new(
    domain_lo: c_double,
    domain_hi: c_double,
    n_per_axis: usize,
    out: *mut *mut SmfCarnotCtj,
) -> SemiflowStatus {
    if out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // Validate grid params eagerly (same checks as Python binding).
        if !domain_lo.is_finite() || !domain_hi.is_finite() {
            return SemiflowStatus::NanInf;
        }
        if domain_lo >= domain_hi {
            return SemiflowStatus::GridMismatch;
        }
        if n_per_axis < 4 {
            return SemiflowStatus::GridMismatch;
        }
        // Validate kernel construction (γ⋆ check).
        if let Err(e) = ComplexTripleJump::new() {
            return SemiflowStatus::from(&e);
        }
        let raw = Box::into_raw(Box::new(CarnotCtjInner {
            domain_lo,
            domain_hi,
            n_per_axis,
        }))
        .cast::<SmfCarnotCtj>();
        unsafe { *out = raw };
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// apply_real
// ---------------------------------------------------------------------------

/// Apply one order-4 complex triple-jump step; write real projection to `dst`.
///
/// `u0` is a flat `f64` array of length `n_per_axis^5` (real input).
/// `dst` receives the real projection `Re(Ψ(τ)f)`; must be length `n_per_axis^5`.
/// `tau >= 0`, finite.
///
/// # Safety
/// `ev` live from `smf_carnot_ctj_new`;
/// `u0` readable `u0_len` f64s; `dst` writable `dst_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_carnot_ctj_apply_real(
    ev: *const SmfCarnotCtj,
    tau: c_double,
    u0: *const c_double,
    u0_len: usize,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || u0.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if !tau.is_finite() || tau < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        let inner = unsafe { &*ev.cast::<CarnotCtjInner>() };
        let total = inner.n_per_axis.pow(D as u32);
        if u0_len != total || dst_len < total {
            return SemiflowStatus::GridMismatch;
        }
        let src_slice = unsafe { std::slice::from_raw_parts(u0, total) };
        match run_apply_real(inner, tau, src_slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(out_vec) => {
                let dst_s = unsafe { std::slice::from_raw_parts_mut(dst, total) };
                dst_s.copy_from_slice(&out_vec);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return `n_per_axis^5`; 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_carnot_ctj_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_carnot_ctj_size(ev: *const SmfCarnotCtj) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<CarnotCtjInner>() };
    inner.n_per_axis.pow(D as u32)
}

/// Verify γ⋆ satisfies the cubic `2γ³ + (1−2γ)³ = 0` with Re > 0.
///
/// Returns 1 on pass, 0 on failure.
#[no_mangle]
pub extern "C" fn smf_carnot_ctj_verify_gamma_star() -> i32 {
    i32::from(ComplexTripleJump::verify_gamma_star())
}

/// Free a `SmfCarnotCtj`. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_carnot_ctj_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn smf_carnot_ctj_drop(ev: *mut SmfCarnotCtj) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<CarnotCtjInner>())) };
    }));
}

// ---------------------------------------------------------------------------
// Pure-Rust compute (extracted per suckless ≤50-line limit)
// ---------------------------------------------------------------------------

fn run_apply_real(
    inner: &CarnotCtjInner,
    tau: f64,
    src: &[f64],
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let grid = build_grid(inner.domain_lo, inner.domain_hi, inner.n_per_axis)?;
    let src_fn = GridFnND::new(grid, src.to_vec())?;
    let kernel = ComplexTripleJump::new()?;
    let out = kernel.apply_real(tau, &src_fn)?;
    Ok(out.values)
}

fn build_grid(
    lo: f64,
    hi: f64,
    n: usize,
) -> Result<GridND<f64, D>, semiflow_core::SemiflowError> {
    let ax = Grid1D::new(lo, hi, n)?;
    GridND::new([ax; D])
}
