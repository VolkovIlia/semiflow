//! FFI surface for `DriftReactionZeta4Chernoff` (ADR-0127, math §35).
//!
//! ## Symbol names
//!
//! - `smf_drift_reaction_zeta4_new` / `smf_drift_reaction_zeta4_evolve`
//!   / `smf_drift_reaction_zeta4_values` / `smf_drift_reaction_zeta4_size`
//!   / `smf_drift_reaction_zeta4_free`
//!
//! ## Default coefficients
//!
//! `b(x) = 0.5`, `b'(x) = 0.0`, `c(x) = 0.0`. Variable-coefficient closure API
//! is deferred (separate architect task).

#![allow(unsafe_code)]

use std::os::raw::c_double;

use semiflow::{
    BoundaryPolicy, ChernoffSemigroup, Diffusion4thChernoff, DriftReactionZeta4Chernoff, Grid1D,
    GridFn1D,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

// ---------------------------------------------------------------------------
// Static coefficient fn-pointers
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_zeta4_ffi(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_zeta4_ffi(_: f64) -> f64 {
    0.0
}
extern "Rust" fn half_b_zeta4_ffi(_: f64) -> f64 {
    0.5
}

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a `DriftReactionZeta4Chernoff` evolver.
#[repr(C)]
pub struct SmfDriftReactionZeta4 {
    _private: [u8; 0],
}

struct InnerDrZeta4 {
    semigroup: ChernoffSemigroup<DriftReactionZeta4Chernoff, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a `DriftReactionZeta4Chernoff` 1-D evolver.
///
/// Solves `∂_t u = b(x)∂_x u + c(x)u` (order 4; palindromic `R_sym` ∘ K5 ∘ `R_sym`,
/// ADR-0127). Default: `b = 0.5`, `b' = 0.0`, `c = 0.0`.
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64`s.
/// `out` must be a valid `*mut *mut SmfDriftReactionZeta4`.
#[no_mangle]
pub unsafe extern "C" fn smf_drift_reaction_zeta4_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    u0: *const c_double,
    u0_len: usize,
    n_steps: usize,
    out: *mut *mut SmfDriftReactionZeta4,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_dr_zeta4(xmin, xmax, n, n_steps, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfDriftReactionZeta4>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

/// Evolve `SmfDriftReactionZeta4` state by time `t`, writing result into `dst_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_drift_reaction_zeta4_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_drift_reaction_zeta4_evolve(
    ev: *mut SmfDriftReactionZeta4,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerDrZeta4>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_dr_zeta4(inner, t) {
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

/// Copy current `SmfDriftReactionZeta4` grid values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_drift_reaction_zeta4_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_drift_reaction_zeta4_values(
    ev: *const SmfDriftReactionZeta4,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerDrZeta4>() };
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

/// Return `SmfDriftReactionZeta4` grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_drift_reaction_zeta4_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_drift_reaction_zeta4_size(ev: *const SmfDriftReactionZeta4) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerDrZeta4>() };
    inner.current.values.len()
}

/// Free a `SmfDriftReactionZeta4` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_drift_reaction_zeta4_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_drift_reaction_zeta4_free(ev: *mut SmfDriftReactionZeta4) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerDrZeta4>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builder / evolve helpers
// ---------------------------------------------------------------------------

fn build_dr_zeta4(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    u0: &[f64],
) -> Result<InnerDrZeta4, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let d4 = Diffusion4thChernoff::new(unit_a_zeta4_ffi, zero_zeta4_ffi, zero_zeta4_ffi, 1.0, grid);
    let kernel = DriftReactionZeta4Chernoff::new(
        d4,
        half_b_zeta4_ffi,
        zero_zeta4_ffi,
        zero_zeta4_ffi,
        0.5,
        grid,
    );
    let semigroup = ChernoffSemigroup::new(kernel, n_steps)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerDrZeta4 { semigroup, current })
}

fn evolve_dr_zeta4(inner: &mut InnerDrZeta4, t: f64) -> Result<(), semiflow::SemiflowError> {
    let result = inner.semigroup.evolve(t, &inner.current)?;
    inner.current.values = result.values;
    Ok(())
}
