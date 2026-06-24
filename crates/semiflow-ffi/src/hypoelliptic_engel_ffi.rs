//! FFI surface for the Engel-group hypoelliptic Chernoff engine (Round 8, split).
//!
//! Split from `hypoelliptic_ffi.rs` for suckless file-size compliance (≤500 lines).
//!
//! | C handle        | Core type                         | Python class                 |
//! |-----------------|-----------------------------------|------------------------------|
//! | `SmfHypoEngel`  | `HypoellipticChernoff<f64, 4, 2>` | `HypoellipticChernoffEngel`  |
//!
//! ## Buffer layout (axis-0-fastest)
//! State type `GridFnND<f64, 4>` — flat f64 array of length `n**4`.
//! All 4 axes share the same `[xmin, xmax, n]` parameters.
//! **axis-0-fastest**: `idx(i0,i1,i2,i3) = i3*n³ + i2*n² + i1*n + i0`.
//!
//! ## Entry points
//!
//! - `smf_hypo_engel_new(xmin,xmax,n,u0,u0_len,out)` — all 4 axes share `[xmin,xmax,n]`
//! - `smf_hypo_engel_evolve(ev,t,n_steps,dst,dst_len)` — τ = t / `n_steps`
//! - `smf_hypo_engel_values(ev,out,out_len)`
//! - `smf_hypo_engel_size(ev)` → `n**4`
//! - `smf_hypo_engel_free(ev)` — null-safe
//!
//! ## Panic safety
//! Every `extern "C"` body is wrapped in `catch_panic!`.

#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)]

use std::os::raw::c_double;

use semiflow::{
    grid_nd::{GridFnND, GridND},
    hormander::HypoellipticChernoff,
    ChernoffSemigroup, Grid1D,
};

use crate::{handle::validate_u0_finite, hypoelliptic_ffi::check_len, status::SemiflowStatus};

// ─── Opaque handle ────────────────────────────────────────────────────────────

/// Opaque handle for Engel-group hypoelliptic Chernoff evolver.
///
/// Obtain from `smf_hypo_engel_new`; free with `smf_hypo_engel_free`.
#[repr(C)]
pub struct SmfHypoEngel {
    _private: [u8; 0],
}

// ─── Inner state ──────────────────────────────────────────────────────────────

struct EngelState {
    grid: GridND<f64, 4>,
    current: Vec<f64>,
    size: usize, // n^4
}

// ─── Public entry points ──────────────────────────────────────────────────────

/// Allocate an Engel-group hypoelliptic Chernoff evolver.
///
/// Wraps `HypoellipticChernoff<f64, 4, 2>` on ℝ⁴ (step-3 Carnot, D=4, M=2).
/// All 4 axes share the same `[xmin, xmax, n]` parameters.
///
/// ## Buffer layout (axis-0-fastest)
/// Flat f64 array of length `n**4`:
///   `idx(i0,i1,i2,i3) = i3*n³ + i2*n² + i1*n + i0`
///
/// ## Preconditions
/// `xmin < xmax` (finite); `n >= 4`. `u0` non-null, `u0_len == n**4`, all finite.
/// `out` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` readable f64 values.
/// `out` must be a valid writable `*mut *mut SmfHypoEngel`.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_engel_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfHypoEngel,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_engel(xmin, xmax, n, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfHypoEngel>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Advance Engel state by time `t` using `n_steps` iterations (τ = `t/n_steps`).
///
/// Writes `n**4` values into `dst` (axis-0-fastest). Updates internal state.
///
/// # Safety
/// `ev` must be a live pointer from `smf_hypo_engel_new`.
/// `dst` must be valid for `dst_len` writable f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_engel_evolve(
    ev: *mut SmfHypoEngel,
    t: c_double,
    n_steps: usize,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &mut *ev.cast::<EngelState>() };
        if dst_len != s.size {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 || n_steps == 0 {
            return SemiflowStatus::OutOfDomain;
        }
        let tau = t / n_steps as f64;
        match evolve_engel(s.grid.clone(), s.current.clone(), tau, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                s.current.clone_from(&result);
                let buf = unsafe { std::slice::from_raw_parts_mut(dst, dst_len) };
                buf.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Copy current Engel state into `out` (axis-0-fastest, length `n**4`).
///
/// # Safety
/// `ev` must be a live pointer from `smf_hypo_engel_new`.
/// `out` must be valid for `out_len` writable f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_engel_values(
    ev: *const SmfHypoEngel,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<EngelState>() };
        if out_len != s.size {
            return SemiflowStatus::GridMismatch;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        buf.copy_from_slice(&s.current);
        SemiflowStatus::Ok
    })
}

/// Return `n**4`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_hypo_engel_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_engel_size(ev: *const SmfHypoEngel) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<EngelState>() }.size
}

/// Free an Engel handle. Null-safe; do not use after this call.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_hypo_engel_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_hypo_engel_free(ev: *mut SmfHypoEngel) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<EngelState>())) };
    }));
}

// ─── Builder ──────────────────────────────────────────────────────────────────

fn build_engel(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
) -> Result<EngelState, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let ax = Grid1D::new(xmin, xmax, n)?;
    let grid = GridND::<f64, 4>::new([ax, ax, ax, ax])?;
    let size = n * n * n * n;
    check_len("u0 length must equal n**4", u0.len(), size)?;
    // Verify Engel bracket condition at construction.
    let _ = HypoellipticChernoff::<f64, 4, 2>::new_engel()?;
    Ok(EngelState {
        grid,
        current: u0.to_vec(),
        size,
    })
}

// ─── Compute helper (kernel reconstructed each call — kernel is !Clone) ───────

fn evolve_engel(
    grid: GridND<f64, 4>,
    values: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let kernel = HypoellipticChernoff::<f64, 4, 2>::new_engel()?;
    let f = GridFnND { values, grid };
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    Ok(sg.evolve(tau * n_steps as f64, &f)?.values)
}
