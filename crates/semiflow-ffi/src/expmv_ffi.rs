//! FFI surface for `DiffusionExpmvChernoff` (ADR-0121, Al-Mohy & Higham 2011).
//!
//! ## Symbol names
//!
//! - `smf_expmv1d_new` / `smf_expmv1d_evolve` / `smf_expmv1d_values`
//!   / `smf_expmv1d_size` / `smf_expmv1d_free`
//!
//! ## Default coefficients
//!
//! Unit diffusion: `a(x) = 1.0`, `drift = 0.0`, `react = 0.0`, `a_norm_bound = 1.0`.
//! Tolerance-driven `(s, m)` selection; `order()` = `u32::MAX`.

#![allow(unsafe_code)]

use std::os::raw::c_double;

use semiflow_core::{
    BoundaryPolicy, ChernoffSemigroup, Diffusion4thChernoff, DiffusionExpmvChernoff, Grid1D,
    GridFn1D,
};

use crate::handle::validate_u0_finite;
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Static coefficient fn-pointers
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_expmv_ffi(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_expmv_ffi(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a `DiffusionExpmvChernoff` evolver.
#[repr(C)]
pub struct SmfExpmv1D {
    _private: [u8; 0],
}

struct InnerExpmv {
    semigroup: ChernoffSemigroup<DiffusionExpmvChernoff, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a `DiffusionExpmvChernoff` 1-D evolver (unit diffusion `a = 1`).
///
/// Solves `∂_t u = ∂²u` via tolerance-driven scaled truncated-Taylor (ADR-0121).
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64`s.
/// `out` must be a valid `*mut *mut SmfExpmv1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_expmv1d_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    u0: *const c_double,
    u0_len: usize,
    n_steps: usize,
    out: *mut *mut SmfExpmv1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_expmv(xmin, xmax, n, n_steps, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfExpmv1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

/// Evolve `SmfExpmv1D` state by time `t`, writing result into `dst_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_expmv1d_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_expmv1d_evolve(
    ev: *mut SmfExpmv1D,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerExpmv>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_expmv(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// Copy current `SmfExpmv1D` grid values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_expmv1d_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_expmv1d_values(
    ev: *const SmfExpmv1D,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerExpmv>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Size / Free
// ---------------------------------------------------------------------------

/// Return `SmfExpmv1D` grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_expmv1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_expmv1d_size(ev: *const SmfExpmv1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerExpmv>() };
    inner.current.values.len()
}

/// Free a `SmfExpmv1D` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_expmv1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_expmv1d_free(ev: *mut SmfExpmv1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerExpmv>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builder / evolve helpers
// ---------------------------------------------------------------------------

fn build_expmv(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    u0: &[f64],
) -> Result<InnerExpmv, semiflow_core::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let d4 = Diffusion4thChernoff::new(unit_a_expmv_ffi, zero_expmv_ffi, zero_expmv_ffi, 1.0, grid);
    let kernel = DiffusionExpmvChernoff::new(d4);
    let semigroup = ChernoffSemigroup::new(kernel, n_steps)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerExpmv { semigroup, current })
}

fn evolve_expmv(inner: &mut InnerExpmv, t: f64) -> Result<(), semiflow_core::SemiflowError> {
    let result = inner.semigroup.evolve(t, &inner.current)?;
    inner.current.values = result.values;
    Ok(())
}
