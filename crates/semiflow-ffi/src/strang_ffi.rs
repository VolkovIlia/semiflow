//! FFI surface for `StrangSplit<DiffusionChernoff, DriftReactionChernoff>`.
//!
//! Mirrors the Python `Strang1D` class in `semiflow-py/src/diffusion_extra2.rs`:
//! palindromic operator splitting `D(τ/2) ∘ R(τ) ∘ D(τ/2)` (ADR-0006, math §9.4).
//!
//! ## Symbol names
//!
//! - `smf_strang1d_new`   / `smf_strang1d_evolve`   / `smf_strang1d_size`
//!   / `smf_strang1d_values`   / `smf_strang1d_free`
//!
//! ## Default wiring (mirrors Python `Strang1D`)
//!
//! | Sub-kernel    | a    | b    | c    |
//! |---------------|------|------|------|
//! | `DiffusionChernoff` (D) | 1.0 | 0.0 | 0.0 |
//! | `DriftReactionChernoff` (R) | — | 0.5 | 0.0 |
//!
//! ## Safety invariants
//!
//! Same as `diffusion_hi_ffi.rs`: null-check before `catch_panic!`;
//! `_free` is null-safe; `(ptr, len)` pairs are caller-guaranteed.

#![allow(unsafe_code)]

use std::os::raw::c_double;

use semiflow::{
    BoundaryPolicy, ChernoffSemigroup, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D,
    StrangSplit,
};

use crate::handle::validate_u0_finite;
use crate::status::SemiflowStatus;

// Concrete split type alias mirrors diffusion_extra2.rs.
type StrangConcrete = StrangSplit<DiffusionChernoff<f64>, DriftReactionChernoff<f64>>;

// ---------------------------------------------------------------------------
// Unit fn-pointers for default coefficients
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_st(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_st(_: f64) -> f64 {
    0.0
}
extern "Rust" fn half_b_st(_: f64) -> f64 {
    0.5
}

// ---------------------------------------------------------------------------
// Opaque handle + inner state
// ---------------------------------------------------------------------------

/// Opaque handle to a `StrangSplit<DiffusionChernoff, DriftReactionChernoff>` evolver.
#[repr(C)]
pub struct SmfStrang1D {
    _private: [u8; 0],
}

struct InnerStrang1D {
    semigroup: ChernoffSemigroup<StrangConcrete, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Public extern "C" API
// ---------------------------------------------------------------------------

/// Allocate a Strang-split 1-D evolver (unit diffusion `a=1`, drift `b=0.5`, `c=0`).
///
/// Solves `∂_t u = ∂²u + 0.5 ∂_x u` via palindromic Strang splitting (order 2).
///
/// # Preconditions
/// `xmin < xmax`, `n >= 4`, `u0_len == n`, `n_chernoff >= 1`, ptrs non-null.
///
/// # Return values
/// `Ok`/`NullPtr`/`GridMismatch`/`NanInf`/`OutOfDomain`/`Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` contiguous readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfStrang1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_strang1d_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfStrang1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_strang1d(xmin, xmax, n, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfStrang1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve `Strang1D` state by time `t`; write `n` values into `dst_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_strang1d_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_strang1d_evolve(
    ev: *mut SmfStrang1D,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerStrang1D>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_strang1d(inner, t) {
            return SemiflowStatus::from(&e);
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, n) };
        out.copy_from_slice(&inner.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current `Strang1D` grid values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_strang1d_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_strang1d_values(
    ev: *const SmfStrang1D,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerStrang1D>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return `Strang1D` grid size; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_strang1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_strang1d_size(ev: *const SmfStrang1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerStrang1D>() };
    inner.current.values.len()
}

/// Free a `Strang1D` handle from `smf_strang1d_new`.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_strang1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_strang1d_free(ev: *mut SmfStrang1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerStrang1D>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builder helper
// ---------------------------------------------------------------------------

fn build_strang1d(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<InnerStrang1D, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let diff = DiffusionChernoff::new(unit_a_st, zero_st, zero_st, 1.0, grid);
    // Mirror Python Strang1D default: b = 0.5, c_norm_bound = 0.5.
    let drift = DriftReactionChernoff::new(half_b_st, zero_st, 0.5, grid);
    let split = StrangSplit::new(diff, drift);
    let semigroup = ChernoffSemigroup::new(split, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerStrang1D { semigroup, current })
}

// ---------------------------------------------------------------------------
// Evolve helper
// ---------------------------------------------------------------------------

fn evolve_strang1d(inner: &mut InnerStrang1D, t: f64) -> Result<(), semiflow::SemiflowError> {
    let result = inner.semigroup.evolve(t, &inner.current)?;
    inner.current.values = result.values;
    Ok(())
}
