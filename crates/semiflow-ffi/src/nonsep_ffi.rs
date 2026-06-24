//! FFI surface for non-separable 2D diffusion engines.
//!
//! ## Engines
//!
//! - `NonSep2D` — constant-beta coupling `c·∂_xy u` (`NonSeparableMixedChernoff`).
//! - `NonSep2DAniso` — pre-sampled position-dependent `β(x,y)·∂_xy u`.
//!
//! ## Buffer layout (NORMATIVE — x-fastest)
//!
//! Flat `f64` buffers of length `nx * ny` use **x-fastest** storage:
//!   `idx(i, j) = j * nx + i`   (`i` = x-index, `j` = y-index).
//! This matches `GridFn2D` internal layout.
//!
//! ## Entry points — `NonSep2D`
//!
//! - `smf_nonsep2d_new(xmin,xmax,nx,ymin,ymax,ny,c,u0,u0_len,out)`
//! - `smf_nonsep2d_evolve(ev,tau,n_steps,out,out_len)`
//! - `smf_nonsep2d_size(ev)` → `nx * ny`
//! - `smf_nonsep2d_values(ev,out,out_len)`
//! - `smf_nonsep2d_free(ev)` — null-safe
//!
//! ## Entry points — `NonSep2DAniso`
//!
//! - `smf_nonsep2d_aniso_new(xmin,xmax,nx,ymin,ymax,ny,beta,beta_len,beta_norm_bound,u0,u0_len,out)`
//! - `smf_nonsep2d_aniso_evolve(ev,tau,n_steps,out,out_len)`
//! - `smf_nonsep2d_aniso_size(ev)`
//! - `smf_nonsep2d_aniso_values(ev,out,out_len)`
//! - `smf_nonsep2d_aniso_free(ev)` — null-safe
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.

#![allow(unsafe_code)]
#![allow(
    clippy::assigning_clones,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use std::{os::raw::c_double, sync::Arc};

use semiflow::{
    nonseparable_mixed_closure, BoundaryPolicy, ChernoffFunction, DiffusionChernoff, Grid1D,
    Grid2D, GridFn2D, NonSeparableMixedChernoff, ScratchPool,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Shared concrete type
// ---------------------------------------------------------------------------

type Nsm = NonSeparableMixedChernoff<DiffusionChernoff<f64>, DiffusionChernoff<f64>>;

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque handle for `NonSep2D` (constant-beta coupling).
#[repr(C)]
pub struct SmfNonSep2D {
    _private: [u8; 0],
}

/// Opaque handle for `NonSep2DAniso` (pre-sampled β array).
#[repr(C)]
pub struct SmfNonSep2DAniso {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct InnerNonSep2D {
    kernel: Nsm,
    grid: Grid2D<f64>,
    current: Vec<f64>,
    size: usize,
}

// ---------------------------------------------------------------------------
// NonSep2D — constant scalar coupling
// ---------------------------------------------------------------------------

/// Construct a non-separable 2D evolver with constant coupling `c`.
///
/// Solves `∂_t u = ∂_xx u + ∂_yy u + c·∂_xy u`.
/// Buffer layout: x-fastest, `idx(i,j) = j*nx + i`.
///
/// ## Preconditions
/// `xmin < xmax`, `ymin < ymax` (finite); `nx,ny >= 4`.
/// `u0` non-null, length `nx*ny`, all finite.
/// `c` finite.
///
/// # Safety
/// `u0` readable for `u0_len` f64s; `out_ev` writable as `*mut *mut SmfNonSep2D`.
#[no_mangle]
pub unsafe extern "C" fn smf_nonsep2d_new(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    ymin: c_double,
    ymax: c_double,
    ny: usize,
    c: c_double,
    u0: *const c_double,
    u0_len: usize,
    out_ev: *mut *mut SmfNonSep2D,
) -> SemiflowStatus {
    if u0.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if !c.is_finite() {
            return SemiflowStatus::NanInf;
        }
        if u0_len != nx * ny {
            return SemiflowStatus::GridMismatch;
        }
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        for &v in u0_slice {
            if !v.is_finite() {
                return SemiflowStatus::NanInf;
            }
        }
        match build_nonsep2d_const(xmin, xmax, nx, ymin, ymax, ny, c, u0_slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfNonSep2D>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve state by `n_steps` of size `tau`. Output written to `out` (x-fastest).
///
/// # Safety
/// `ev` non-null from `smf_nonsep2d_new`; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_nonsep2d_evolve(
    ev: *mut SmfNonSep2D,
    tau: c_double,
    n_steps: usize,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerNonSep2D>() };
        if out_len != inner.size {
            return SemiflowStatus::GridMismatch;
        }
        if let Err(st) = validate_tau_steps(tau, n_steps) {
            return st;
        }
        match evolve_nonsep(
            &inner.kernel,
            inner.grid,
            inner.current.clone(),
            tau,
            n_steps,
        ) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                inner.current = result.clone();
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
                out_slice.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return `nx * ny`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_nonsep2d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_nonsep2d_size(ev: *const SmfNonSep2D) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<InnerNonSep2D>() }.size
}

/// Copy current state into `out` (x-fastest, length `out_len`).
///
/// # Safety
/// `ev` non-null; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_nonsep2d_values(
    ev: *const SmfNonSep2D,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerNonSep2D>() };
        if out_len != inner.size {
            return SemiflowStatus::GridMismatch;
        }
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        out_slice.copy_from_slice(&inner.current);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfNonSep2D` handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_nonsep2d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_nonsep2d_free(ev: *mut SmfNonSep2D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerNonSep2D>())) };
    }));
}

// ---------------------------------------------------------------------------
// NonSep2DAniso — pre-sampled β(x,y)
// ---------------------------------------------------------------------------

/// Construct a non-separable 2D evolver with position-dependent `β(x,y)`.
///
/// Solves `∂_t u = ∂_xx u + ∂_yy u + β(x,y)·∂_xy u`.
/// `beta` is a flat x-fastest array of length `nx*ny`.
/// `beta_norm_bound`: upper bound on `‖β‖_∞`.
///   Pass `< 0.0` to auto-compute as `1.1 * max|β|`.
///
/// # Safety
/// `beta` readable for `beta_len` f64s; `u0` readable for `u0_len` f64s.
/// `out_ev` writable as `*mut *mut SmfNonSep2DAniso`.
#[no_mangle]
pub unsafe extern "C" fn smf_nonsep2d_aniso_new(
    xmin: c_double,
    xmax: c_double,
    nx: usize,
    ymin: c_double,
    ymax: c_double,
    ny: usize,
    beta: *const c_double,
    beta_len: usize,
    beta_norm_bound: c_double,
    u0: *const c_double,
    u0_len: usize,
    out_ev: *mut *mut SmfNonSep2DAniso,
) -> SemiflowStatus {
    if beta.is_null() || u0.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if beta_len != nx * ny || u0_len != nx * ny {
            return SemiflowStatus::GridMismatch;
        }
        let beta_slice = unsafe { std::slice::from_raw_parts(beta, beta_len) };
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        if let Err(st) = validate_finite_slice(beta_slice) {
            return st;
        }
        if let Err(st) = validate_finite_slice(u0_slice) {
            return st;
        }
        let norm = compute_beta_norm_bound(beta_slice, beta_norm_bound);
        match build_nonsep2d_aniso(xmin, xmax, nx, ymin, ymax, ny, beta_slice, norm, u0_slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfNonSep2DAniso>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve aniso state by `n_steps` of size `tau`.
///
/// # Safety
/// `ev` non-null; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_nonsep2d_aniso_evolve(
    ev: *mut SmfNonSep2DAniso,
    tau: c_double,
    n_steps: usize,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerNonSep2D>() };
        if out_len != inner.size {
            return SemiflowStatus::GridMismatch;
        }
        if let Err(st) = validate_tau_steps(tau, n_steps) {
            return st;
        }
        match evolve_nonsep(
            &inner.kernel,
            inner.grid,
            inner.current.clone(),
            tau,
            n_steps,
        ) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => {
                inner.current = result.clone();
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
                out_slice.copy_from_slice(&result);
                SemiflowStatus::Ok
            }
        }
    })
}

/// Return `nx * ny`; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_nonsep2d_aniso_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_nonsep2d_aniso_size(ev: *const SmfNonSep2DAniso) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<InnerNonSep2D>() }.size
}

/// Copy current state into `out`.
///
/// # Safety
/// `ev` non-null; `out` writable for `out_len` f64s.
#[no_mangle]
pub unsafe extern "C" fn smf_nonsep2d_aniso_values(
    ev: *const SmfNonSep2DAniso,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerNonSep2D>() };
        if out_len != inner.size {
            return SemiflowStatus::GridMismatch;
        }
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        out_slice.copy_from_slice(&inner.current);
        SemiflowStatus::Ok
    })
}

/// Free a `SmfNonSep2DAniso` handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_nonsep2d_aniso_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_nonsep2d_aniso_free(ev: *mut SmfNonSep2DAniso) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerNonSep2D>())) };
    }));
}

// Builders and validators — extracted to keep file ≤500 lines.
include!("nonsep_ffi_helpers.rs");
