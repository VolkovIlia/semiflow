//! FFI surface for the 6-D Smolyak sparse-grid Chernoff engine (ADR-0138).
//!
//! ## Engine
//!
//! `SmolyakD6` — `SmolyakGridND<f64, 6>`, unit diffusion (A=I, b=0, c=0).
//! Default level ℓ = D+3 = 9 → 533 Smolyak nodes.
//! This is the NARROW unit-only scope: variable coefficients are NOT bound
//! (TIER-3 per ADR-0138).
//!
//! ## ABI-safety invariant (ADR-0138)
//!
//! `GridFnND<f64,6>` does not cross the boundary.
//! Caller passes and receives a flat `f64[n^6]` buffer.
//!
//! ## Buffer layout
//!
//! Flat `f64[n_per_axis^6]`, axis-0 fastest (same as all ND engines).
//! All 6 axes are isotropic: same domain `[domain_lo, domain_hi]` and
//! same `n_per_axis` nodes.
//!
//! ## Entry points
//!
//! - `smf_smolyak_d6_new(domain_lo,domain_hi,n_per_axis,out)` → `SemiflowStatus`
//! - `smf_smolyak_d6_apply(ev,tau,u0,u0_len,n_steps,out,out_len)` → stateless
//! - `smf_smolyak_d6_size(ev)` → `n_per_axis^6`
//! - `smf_smolyak_d6_n_nodes(ev)` → sparse-grid node count
//! - `smf_smolyak_d6_free(ev)` — null-safe
//!
//! **Note**: `.apply` is stateless (mirrors Python `SmolyakD6V8.apply`).
//! The caller owns the flat buffer and manages evolving state.
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]

use std::os::raw::c_double;

use semiflow::{
    grid_nd::{GridFnND, GridND},
    smolyak::SmolyakGridND,
    ChernoffFunction, Grid1D, ScratchPool, SquareMatrix,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const D: usize = 6;
const DEFAULT_LEVEL: usize = D + 3; // ℓ=9

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle for the 6-D Smolyak sparse-grid evolver.
///
/// Obtain from `smf_smolyak_d6_new`; free with `smf_smolyak_d6_free`.
/// The kernel is stateless; the caller holds the flat `f64` buffer.
#[repr(C)]
pub struct SmfSmolyakD6 {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct InnerSmolyakD6 {
    domain_lo: f64,
    domain_hi: f64,
    n_per_axis: usize,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Construct a 6-D Smolyak evolver (unit diffusion, level ℓ=9).
///
/// ## Preconditions
/// `domain_lo < domain_hi` (both finite); `n_per_axis >= 4`.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `out_ev` must be a writable `*mut *mut SmfSmolyakD6`.
#[no_mangle]
pub unsafe extern "C" fn smf_smolyak_d6_new(
    domain_lo: c_double,
    domain_hi: c_double,
    n_per_axis: usize,
    out_ev: *mut *mut SmfSmolyakD6,
) -> SemiflowStatus {
    if out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if let Err(st) = validate_domain(domain_lo, domain_hi, n_per_axis) {
            return st;
        }
        // Eagerly validate that the kernel builds without error.
        if let Err(e) = build_kernel(domain_lo, domain_hi, n_per_axis) {
            return SemiflowStatus::from(&e);
        }
        let inner = InnerSmolyakD6 {
            domain_lo,
            domain_hi,
            n_per_axis,
        };
        let raw = Box::into_raw(Box::new(inner)).cast::<SmfSmolyakD6>();
        unsafe { *out_ev = raw };
        SemiflowStatus::Ok
    })
}

/// Apply `n_steps` Smolyak Chernoff steps to `u0`, writing the result to `out`.
///
/// This function is **stateless**: the caller manages the evolving buffer.
/// To time-step, pass the previous `out` as the next `u0`.
///
/// ## Preconditions
/// `tau >= 0` finite; `n_steps >= 1`.
/// `u0_len == out_len == n_per_axis^6`.
///
/// # Safety
/// `u0` readable for `u0_len` f64s; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_smolyak_d6_apply(
    ev: *const SmfSmolyakD6,
    tau: c_double,
    u0: *const c_double,
    u0_len: usize,
    n_steps: usize,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerSmolyakD6>() };
        let expected = inner.n_per_axis.pow(D as u32);
        if u0_len != expected || out_len != expected {
            return SemiflowStatus::GridMismatch;
        }
        if let Err(st) = validate_tau_steps(tau, n_steps) {
            return st;
        }
        let u0_sl = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match run_smolyak(
            inner.domain_lo,
            inner.domain_hi,
            inner.n_per_axis,
            tau,
            u0_sl,
            n_steps,
        ) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                let sl = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
                sl.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return total grid size `n_per_axis^6`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or live from `smf_smolyak_d6_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_smolyak_d6_size(ev: *const SmfSmolyakD6) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerSmolyakD6>() };
    inner.n_per_axis.pow(D as u32)
}

/// Return the number of Smolyak sparse-grid nodes; 0 on error.
///
/// # Safety
/// `ev` must be null or live from `smf_smolyak_d6_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_smolyak_d6_n_nodes(ev: *const SmfSmolyakD6) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerSmolyakD6>() };
    match build_kernel(inner.domain_lo, inner.domain_hi, inner.n_per_axis) {
        Ok(k) => k.n_nodes(),
        Err(_) => 0,
    }
}

/// Free a `SmfSmolyakD6` handle. Null-safe.
///
/// # Safety
/// `ev` must be null or live from `smf_smolyak_d6_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_smolyak_d6_free(ev: *mut SmfSmolyakD6) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerSmolyakD6>())) };
    }));
}

// ---------------------------------------------------------------------------
// Pure-Rust compute (per-crate dup, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

fn run_smolyak(
    lo: f64,
    hi: f64,
    n: usize,
    tau: f64,
    u0: &[f64],
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let kernel = build_kernel(lo, hi, n)?;
    let grid = build_grid(lo, hi, n)?;
    let mut src = GridFnND::new(grid.clone(), u0.to_vec())?;
    let mut dst = GridFnND::new(grid, vec![0.0_f64; u0.len()])?;
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut pool)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

fn build_grid(lo: f64, hi: f64, n: usize) -> Result<GridND<f64, D>, semiflow::SemiflowError> {
    let ax = Grid1D::new(lo, hi, n)?;
    GridND::new([ax; D])
}

fn build_kernel(
    lo: f64,
    hi: f64,
    n: usize,
) -> Result<SmolyakGridND<f64, D>, semiflow::SemiflowError> {
    let grid = build_grid(lo, hi, n)?;
    SmolyakGridND::with_level(
        |_x: &[f64; D], a: &mut SquareMatrix<f64, D>| {
            for i in 0..D {
                a.set(i, i, 1.0);
            }
        },
        |_x: &[f64; D], b: &mut [f64; D]| {
            for v in b.iter_mut() {
                *v = 0.0;
            }
        },
        |_x: &[f64; D]| 0.0_f64,
        grid,
        DEFAULT_LEVEL,
    )
}

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

fn validate_domain(lo: f64, hi: f64, n: usize) -> Result<(), SemiflowStatus> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(SemiflowStatus::NanInf);
    }
    if lo >= hi {
        return Err(SemiflowStatus::GridMismatch);
    }
    if n < 4 {
        return Err(SemiflowStatus::GridMismatch);
    }
    Ok(())
}

fn validate_tau_steps(tau: f64, n_steps: usize) -> Result<(), SemiflowStatus> {
    if n_steps == 0 || !tau.is_finite() || tau < 0.0 {
        return Err(SemiflowStatus::OutOfDomain);
    }
    Ok(())
}
