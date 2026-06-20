//! FFI surface for `DriftReactionChernoff` and `ShiftChernoff1D`.
//!
//! ## Symbol names
//!
//! - `smf_drift_reaction_new`  / `smf_drift_reaction_evolve`  / `smf_drift_reaction_size`
//!   / `smf_drift_reaction_values`  / `smf_drift_reaction_free`
//! - `smf_shift1d_new`  / `smf_shift1d_evolve`  / `smf_shift1d_size`
//!   / `smf_shift1d_values`  / `smf_shift1d_free`
//!
//! ## Safety invariants
//!
//! Same as `diffusion_hi_ffi.rs`: null-check before `catch_panic!`;
//! `_free` is null-safe; `(ptr, len)` pairs are caller-guaranteed.
//!
//! ## Default coefficients
//!
//! | Engine                 | `b`  | `c`  | Note                        |
//! |------------------------|------|------|-----------------------------|
//! | `DriftReactionChernoff`| 0.5  | 0.0  | mirrors Python `DriftReaction1D` |
//! | `ShiftChernoff1D`      | 0.0  | 0.0  | `a = 0.5`, mirrors Python `Shift1D` |

#![allow(unsafe_code)]

use std::os::raw::c_double;

use semiflow_core::{
    BoundaryPolicy, ChernoffSemigroup, DriftReactionChernoff, Grid1D, GridFn1D, ShiftChernoff1D,
};

use crate::handle::validate_u0_finite;
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Constant fn-pointers
// ---------------------------------------------------------------------------

extern "Rust" fn zero_cdr(_: f64) -> f64 {
    0.0
}
extern "Rust" fn half_b_cdr(_: f64) -> f64 {
    0.5
}
extern "Rust" fn half_a_cdr(_: f64) -> f64 {
    0.5
}

// ---------------------------------------------------------------------------
// DriftReactionChernoff
// ---------------------------------------------------------------------------

/// Opaque handle to a `DriftReactionChernoff` evolver.
#[repr(C)]
pub struct SmfDriftReaction {
    _private: [u8; 0],
}

struct InnerDriftReaction {
    semigroup: ChernoffSemigroup<DriftReactionChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a `DriftReactionChernoff` 1-D evolver (`b = 0.5`, `c = 0`).
///
/// Solves `∂_t u = b(x)∂_x u + c(x)u` (RK2 characteristic flow, order 2).
/// Same preconditions and return codes as `smf_trunc_exp_new`.
///
/// # Safety
/// `u0` → `u0_len` readable `f64`s; `out` → valid `*mut *mut SmfDriftReaction`.
#[no_mangle]
pub unsafe extern "C" fn smf_drift_reaction_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfDriftReaction,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_drift_reaction(xmin, xmax, n, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfDriftReaction>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `SmfDriftReaction` state by time `t`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_drift_reaction_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_drift_reaction_evolve(
    ev: *mut SmfDriftReaction,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerDriftReaction>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_drift_reaction(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current `SmfDriftReaction` grid values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_drift_reaction_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_drift_reaction_values(
    ev: *const SmfDriftReaction,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerDriftReaction>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return `SmfDriftReaction` grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_drift_reaction_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_drift_reaction_size(ev: *const SmfDriftReaction) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerDriftReaction>() };
    inner.current.values.len()
}

/// Free a `SmfDriftReaction` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_drift_reaction_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_drift_reaction_free(ev: *mut SmfDriftReaction) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerDriftReaction>())) };
    }));
}

// ---------------------------------------------------------------------------
// ShiftChernoff1D
// ---------------------------------------------------------------------------

/// Opaque handle to a `ShiftChernoff1D` evolver.
#[repr(C)]
pub struct SmfShift1D {
    _private: [u8; 0],
}

struct InnerShift1D {
    semigroup: ChernoffSemigroup<ShiftChernoff1D<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a `ShiftChernoff1D` (formula-6 CDR) evolver (`a = 0.5`, `b = 0`, `c = 0`).
///
/// Solves `∂_t u = a(x)∂²u + b(x)∂u + c(x)u` (global order 1).
/// Same preconditions and return codes as `smf_trunc_exp_new`.
///
/// # Safety
/// `u0` → `u0_len` readable `f64`s; `out` → valid `*mut *mut SmfShift1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_shift1d_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfShift1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_shift1d(xmin, xmax, n, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfShift1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `SmfShift1D` state by time `t`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_shift1d_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_shift1d_evolve(
    ev: *mut SmfShift1D,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerShift1D>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_shift1d(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current `SmfShift1D` grid values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_shift1d_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_shift1d_values(
    ev: *const SmfShift1D,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerShift1D>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return `SmfShift1D` grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_shift1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_shift1d_size(ev: *const SmfShift1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerShift1D>() };
    inner.current.values.len()
}

/// Free a `SmfShift1D` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_shift1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_shift1d_free(ev: *mut SmfShift1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerShift1D>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builder helpers
// ---------------------------------------------------------------------------

fn build_drift_reaction(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<InnerDriftReaction, semiflow_core::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    // Default: b = 0.5, c = 0; c_norm_bound = 0.5.
    let func = DriftReactionChernoff::new(half_b_cdr, zero_cdr, 0.5, grid);
    let semigroup = ChernoffSemigroup::new(func, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerDriftReaction { semigroup, current })
}

fn build_shift1d(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<InnerShift1D, semiflow_core::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    // Default: a = 0.5, b = 0, c = 0; c_norm_bound = 0.5.
    let func = ShiftChernoff1D::new(half_a_cdr, zero_cdr, zero_cdr, 0.5, grid);
    let semigroup = ChernoffSemigroup::new(func, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerShift1D { semigroup, current })
}

// ---------------------------------------------------------------------------
// Evolve helpers
// ---------------------------------------------------------------------------

fn evolve_drift_reaction(
    inner: &mut InnerDriftReaction,
    t: f64,
) -> Result<(), semiflow_core::SemiflowError> {
    let result = inner.semigroup.evolve(t, &inner.current)?;
    inner.current.values = result.values;
    Ok(())
}

fn evolve_shift1d(
    inner: &mut InnerShift1D,
    t: f64,
) -> Result<(), semiflow_core::SemiflowError> {
    let result = inner.semigroup.evolve(t, &inner.current)?;
    inner.current.values = result.values;
    Ok(())
}
