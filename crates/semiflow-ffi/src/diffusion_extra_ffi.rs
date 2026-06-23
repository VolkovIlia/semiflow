//! FFI surface for `TruncatedExpDiffusionChernoff` and `TruncatedExp4thDiffusionChernoff`.
//!
//! ## Symbol names
//!
//! - `smf_trunc_exp_new`  / `smf_trunc_exp_evolve`  / `smf_trunc_exp_size`
//!   / `smf_trunc_exp_values`  / `smf_trunc_exp_free`
//! - `smf_trunc_exp4_new` / `smf_trunc_exp4_evolve` / `smf_trunc_exp4_size`
//!   / `smf_trunc_exp4_values` / `smf_trunc_exp4_free`
//!
//! ## Safety invariants
//!
//! Same as `diffusion_hi_ffi.rs`: null-check before `catch_panic!`;
//! `_free` is null-safe; `(ptr, len)` pairs are caller-guaranteed.
//!
//! ## Default coefficients (unit diffusion)
//!
//! Both engines use `a = 1`, `a' = 0`, `a'' = 0` (unit diffusion).

#![allow(unsafe_code)]

use std::os::raw::c_double;

use semiflow::{
    BoundaryPolicy, ChernoffSemigroup, Grid1D, GridFn1D, TruncatedExp4thDiffusionChernoff,
    TruncatedExpDiffusionChernoff,
};

use crate::handle::validate_u0_finite;
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Unit fn-pointers
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_ex(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_ex(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// TruncatedExpDiffusionChernoff
// ---------------------------------------------------------------------------

/// Opaque handle to a `TruncatedExpDiffusionChernoff` evolver.
#[repr(C)]
pub struct SmfTruncExp {
    _private: [u8; 0],
}

struct InnerTruncExp {
    semigroup: ChernoffSemigroup<TruncatedExpDiffusionChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a K=4 truncated-exp 1-D diffusion evolver (unit `a = 1`).
///
/// # Preconditions
/// `xmin < xmax`, `n >= 4`, `u0_len == n`, `n_chernoff >= 1`, ptrs non-null.
///
/// # Return values
/// `Ok`/`NullPtr`/`GridMismatch`/`NanInf`/`OutOfDomain`/`CflViolated`/`Panic`.
///
/// # Safety
/// `u0` → `u0_len` readable `f64`s; `out` → valid `*mut *mut SmfTruncExp`.
#[no_mangle]
pub unsafe extern "C" fn smf_trunc_exp_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfTruncExp,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_trunc_exp(xmin, xmax, n, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfTruncExp>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `SmfTruncExp` state by time `t`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_trunc_exp_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_trunc_exp_evolve(
    ev: *mut SmfTruncExp,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerTruncExp>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_trunc_exp(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current `SmfTruncExp` grid values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_trunc_exp_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_trunc_exp_values(
    ev: *const SmfTruncExp,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerTruncExp>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return `SmfTruncExp` grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_trunc_exp_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_trunc_exp_size(ev: *const SmfTruncExp) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerTruncExp>() };
    inner.current.values.len()
}

/// Free a `SmfTruncExp` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_trunc_exp_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_trunc_exp_free(ev: *mut SmfTruncExp) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerTruncExp>())) };
    }));
}

// ---------------------------------------------------------------------------
// TruncatedExp4thDiffusionChernoff
// ---------------------------------------------------------------------------

/// Opaque handle to a `TruncatedExp4thDiffusionChernoff` evolver.
#[repr(C)]
pub struct SmfTruncExp4 {
    _private: [u8; 0],
}

struct InnerTruncExp4 {
    semigroup: ChernoffSemigroup<TruncatedExp4thDiffusionChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a K=4 higher-resolution truncated-exp 1-D diffusion evolver.
///
/// Same preconditions and return codes as `smf_trunc_exp_new`.
///
/// # Safety
/// `u0` → `u0_len` readable `f64`s; `out` → valid `*mut *mut SmfTruncExp4`.
#[no_mangle]
pub unsafe extern "C" fn smf_trunc_exp4_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfTruncExp4,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_trunc_exp4(xmin, xmax, n, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfTruncExp4>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `SmfTruncExp4` state by time `t`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_trunc_exp4_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_trunc_exp4_evolve(
    ev: *mut SmfTruncExp4,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerTruncExp4>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_trunc_exp4(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current `SmfTruncExp4` grid values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_trunc_exp4_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_trunc_exp4_values(
    ev: *const SmfTruncExp4,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerTruncExp4>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return `SmfTruncExp4` grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_trunc_exp4_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_trunc_exp4_size(ev: *const SmfTruncExp4) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerTruncExp4>() };
    inner.current.values.len()
}

/// Free a `SmfTruncExp4` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_trunc_exp4_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_trunc_exp4_free(ev: *mut SmfTruncExp4) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerTruncExp4>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builder helpers
// ---------------------------------------------------------------------------

fn build_trunc_exp(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<InnerTruncExp, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let func = TruncatedExpDiffusionChernoff::new(unit_a_ex, zero_ex, zero_ex, 1.0, grid);
    let semigroup = ChernoffSemigroup::new(func, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerTruncExp { semigroup, current })
}

fn build_trunc_exp4(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<InnerTruncExp4, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let func = TruncatedExp4thDiffusionChernoff::new(unit_a_ex, zero_ex, zero_ex, 1.0, grid);
    let semigroup = ChernoffSemigroup::new(func, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerTruncExp4 { semigroup, current })
}

// ---------------------------------------------------------------------------
// Evolve helpers
// ---------------------------------------------------------------------------

fn evolve_trunc_exp(
    inner: &mut InnerTruncExp,
    t: f64,
) -> Result<(), semiflow::SemiflowError> {
    let result = inner.semigroup.evolve(t, &inner.current)?;
    inner.current.values = result.values;
    Ok(())
}

fn evolve_trunc_exp4(
    inner: &mut InnerTruncExp4,
    t: f64,
) -> Result<(), semiflow::SemiflowError> {
    let result = inner.semigroup.evolve(t, &inner.current)?;
    inner.current.values = result.values;
    Ok(())
}
