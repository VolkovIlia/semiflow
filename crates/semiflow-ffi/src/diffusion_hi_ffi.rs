//! FFI surface for higher-order 1-D diffusion engines (4th and 6th order).
//!
//! Exposes `Diffusion4thChernoff` and `Diffusion6thChernoff` with unit
//! diffusion coefficient `a = 1` via C-ABI opaque handles.
//!
//! ## Symbol names
//!
//! - `smf_heat1d_4th_new`   / `smf_heat1d_4th_evolve`   / `smf_heat1d_4th_size`
//!   / `smf_heat1d_4th_values`   / `smf_heat1d_4th_free`
//! - `smf_heat1d_6th_new`   / `smf_heat1d_6th_evolve`   / `smf_heat1d_6th_size`
//!   / `smf_heat1d_6th_values`   / `smf_heat1d_6th_free`
//!
//! ## Safety invariants
//!
//! 1. Null-check BEFORE `catch_panic!`.
//! 2. Handle pointer is always a live `Box<Inner*>`.
//! 3. `_free` is null-safe.
//! 4. `(ptr, len)` slice pairs are caller-guaranteed valid.
//!
//! ## Ownership
//!
//! `_new` allocates; ownership transfers to the caller.  Free with `_free`.
//!
//! ## Panic safety
//!
//! Every entry point wraps its body in `catch_panic!`.
//! Build with `--profile release-ffi` for `panic = "unwind"`.

#![allow(unsafe_code)]

use std::os::raw::c_double;

use semiflow::{
    BoundaryPolicy, ChernoffSemigroup, Diffusion4thChernoff, Diffusion6thChernoff, Grid1D,
    GridFn1D,
};

use crate::handle::validate_u0_finite;
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Unit-coefficient fn-pointers
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_hi(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_d_hi(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// 4th-order engine
// ---------------------------------------------------------------------------

/// Opaque handle to a `Diffusion4thChernoff` evolver.
#[repr(C)]
pub struct SmfHeat1D4th {
    _private: [u8; 0],
}

struct Inner4th {
    semigroup: ChernoffSemigroup<Diffusion4thChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a 4th-order 1-D heat evolver (unit diffusion, `a = 1`).
///
/// # Preconditions
/// - `xmin < xmax`, both finite; `n >= 4`; `u0_len == n`; `u0` non-null.
/// - `n_chernoff >= 1`; `out` non-null.
///
/// # Return values
/// - `Ok` (0) — `*out` set. `NullPtr` (5) / `GridMismatch` (1) /
///   `NanInf` (2) / `OutOfDomain` (3) / `Panic` (99) on error.
///
/// # Safety
/// `u0` must point to `u0_len` contiguous readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfHeat1D4th`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_4th_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfHeat1D4th,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_inner4(xmin, xmax, n, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfHeat1D4th>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve the 4th-order state by time `t`; write `n` values into `dst_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_heat1d_4th_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_4th_evolve(
    ev: *mut SmfHeat1D4th,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<Inner4th>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_4th(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current grid values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_heat1d_4th_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_4th_values(
    ev: *const SmfHeat1D4th,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<Inner4th>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return the grid size; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat1d_4th_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_4th_size(ev: *const SmfHeat1D4th) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<Inner4th>() };
    inner.current.values.len()
}

/// Free a handle from `smf_heat1d_4th_new`.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat1d_4th_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_4th_free(ev: *mut SmfHeat1D4th) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<Inner4th>())) };
    }));
}

// ---------------------------------------------------------------------------
// 6th-order engine
// ---------------------------------------------------------------------------

/// Opaque handle to a `Diffusion6thChernoff` evolver.
#[repr(C)]
pub struct SmfHeat1D6th {
    _private: [u8; 0],
}

struct Inner6th {
    semigroup: ChernoffSemigroup<Diffusion6thChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a 6th-order 1-D heat evolver (unit diffusion, `a = 1`).
///
/// Parameters mirror `smf_heat1d_4th_new`; same return codes apply.
///
/// # Safety
/// `u0` must point to `u0_len` contiguous readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfHeat1D6th`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_6th_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfHeat1D6th,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_inner6(xmin, xmax, n, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfHeat1D6th>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve the 6th-order state by time `t`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_heat1d_6th_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_6th_evolve(
    ev: *mut SmfHeat1D6th,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<Inner6th>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_6th(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current grid values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_heat1d_6th_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_6th_values(
    ev: *const SmfHeat1D6th,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<Inner6th>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return the grid size; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat1d_6th_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_6th_size(ev: *const SmfHeat1D6th) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<Inner6th>() };
    inner.current.values.len()
}

/// Free a handle from `smf_heat1d_6th_new`.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_heat1d_6th_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_6th_free(ev: *mut SmfHeat1D6th) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<Inner6th>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builder helpers
// ---------------------------------------------------------------------------

fn build_inner4(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<Inner4th, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let chernoff = Diffusion4thChernoff::new(unit_a_hi, zero_d_hi, zero_d_hi, 1.0, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Inner4th { semigroup, current })
}

fn build_inner6(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<Inner6th, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let chernoff = Diffusion6thChernoff::new(unit_a_hi, zero_d_hi, zero_d_hi, 1.0, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Inner6th { semigroup, current })
}

fn evolve_4th(
    inner: &mut Inner4th,
    t: f64,
) -> Result<(), semiflow::SemiflowError> {
    let result = inner.semigroup.evolve(t, &inner.current)?;
    inner.current.values = result.values;
    Ok(())
}

fn evolve_6th(
    inner: &mut Inner6th,
    t: f64,
) -> Result<(), semiflow::SemiflowError> {
    let result = inner.semigroup.evolve(t, &inner.current)?;
    inner.current.values = result.values;
    Ok(())
}
