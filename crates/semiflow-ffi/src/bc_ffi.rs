//! FFI surface for BC kernels — Part 1: `Killing1D`, `Reflected1D` (Round 5).
//!
//! | C handle         | Core type                                          | Python class  |
//! |------------------|----------------------------------------------------|---------------|
//! | `SmfKilling1D`   | `KillingChernoff<DiffUnit, BoxRegion<f64,1>>`      | `Killing1D`   |
//! | `SmfReflected1D` | `ReflectedHeatChernoff<DiffUnit, HalfSpaceRegion>` | `Reflected1D` |
//!
//! Part 2 (`bc_ffi2.rs`): `Robin1D`, `Resolvent1D`, `KilledDir1D`.
//!
//! ## Safety contract (all entry points)
//!
//! - Null-check BEFORE `catch_panic!`.
//! - `(ptr, len)` pairs are caller-guaranteed valid for that length.
//! - `_free` is always null-safe.
//! - Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use std::os::raw::c_double;

use semiflow_core::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    killing::{BoxRegion, KillingChernoff},
    reflection::{HalfSpaceRegion, ReflectedHeatChernoff},
    ChernoffSemigroup,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

// ---------------------------------------------------------------------------
// Unit-diffusion fn-pointers
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_bc(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_bc(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type KillingKernel = KillingChernoff<DiffUnit, BoxRegion<f64, 1>, f64>;
type ReflectedKernel = ReflectedHeatChernoff<DiffUnit, HalfSpaceRegion<f64, 1>, f64>;

// ===========================================================================
// KillingChernoff — smf_killing1d_*
// ===========================================================================

/// Opaque handle to `KillingChernoff<DiffusionChernoff, BoxRegion>`.
#[repr(C)]
pub struct SmfKilling1D {
    _private: [u8; 0],
}

struct KillingState {
    semigroup: ChernoffSemigroup<KillingKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a killing-region 1D heat evolver.
///
/// Absorbs the solution outside `[lo, hi)` (Feynman-Kac, order 1).
/// Unit diffusion `a = 1`.
///
/// ## Preconditions
/// - `xmin < xmax`, both finite; `n_grid >= 4`; `lo < hi`, both finite.
/// - `n_chernoff >= 1`; `u0` non-null, `u0_len == n_grid`, all finite.
/// - `out` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfKilling1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_killing1d_new(
    xmin: c_double,
    xmax: c_double,
    n_grid: usize,
    n_chernoff: usize,
    lo: c_double,
    hi: c_double,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfKilling1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_killing(xmin, xmax, n_grid, n_chernoff, lo, hi, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfKilling1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Advance killing evolver by `t`; write `n` values into `dst`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_killing1d_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_killing1d_evolve(
    ev: *mut SmfKilling1D,
    t: c_double,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &mut *ev.cast::<KillingState>() };
        let n = s.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        match s.semigroup.evolve(t, &s.current) {
            Err(e) => return SemiflowStatus::from(&e),
            Ok(next) => s.current = next,
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, n) };
        out.copy_from_slice(&s.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current values into `out`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_killing1d_new`.
/// `out` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_killing1d_values(
    ev: *const SmfKilling1D,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<KillingState>() };
        let vals = &s.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out, vals.len()) };
        buf.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return grid size; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_killing1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_killing1d_size(ev: *const SmfKilling1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let s = unsafe { &*ev.cast::<KillingState>() };
    s.current.values.len()
}

/// Free a killing handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_killing1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_killing1d_free(ev: *mut SmfKilling1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<KillingState>())) };
    }));
}

fn build_killing(
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_chernoff: usize,
    lo: f64,
    hi: f64,
    u0: &[f64],
) -> Result<KillingState, semiflow_core::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n_grid)?;
    let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
    let region = BoxRegion::<f64, 1>::new([lo], [hi])?;
    let kernel = KillingChernoff::new(diff, region)?;
    let semigroup = ChernoffSemigroup::new(kernel, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(KillingState { semigroup, current })
}

// ===========================================================================
// ReflectedHeatChernoff — smf_reflected1d_*
// ===========================================================================

/// Opaque handle to `ReflectedHeatChernoff<DiffusionChernoff, HalfSpaceRegion>`.
#[repr(C)]
pub struct SmfReflected1D {
    _private: [u8; 0],
}

struct ReflectedState {
    semigroup: ChernoffSemigroup<ReflectedKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a Neumann-BC (image method, order 2) 1D heat evolver.
///
/// Reflecting boundary at `origin`. Unit diffusion `a = 1`.
///
/// ## Preconditions
/// - `xmin < xmax`, both finite; `n_grid >= 4`.
/// - `n_chernoff >= 1`; `u0` non-null, `u0_len == n_grid`, all finite.
/// - `out` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64` values.
/// `out` must be a valid writable `*mut *mut SmfReflected1D`.
#[no_mangle]
pub unsafe extern "C" fn smf_reflected1d_new(
    xmin: c_double,
    xmax: c_double,
    n_grid: usize,
    n_chernoff: usize,
    origin: c_double,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfReflected1D,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_reflected(xmin, xmax, n_grid, n_chernoff, origin, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(state) => {
                let raw = Box::into_raw(Box::new(state)).cast::<SmfReflected1D>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Advance reflected evolver by `t`; write values into `dst`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_reflected1d_new`.
/// `dst` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_reflected1d_evolve(
    ev: *mut SmfReflected1D,
    t: c_double,
    dst: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &mut *ev.cast::<ReflectedState>() };
        let n = s.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        match s.semigroup.evolve(t, &s.current) {
            Err(e) => return SemiflowStatus::from(&e),
            Ok(next) => s.current = next,
        }
        let out = unsafe { std::slice::from_raw_parts_mut(dst, n) };
        out.copy_from_slice(&s.current.values);
        SemiflowStatus::Ok
    })
}

/// Copy current values into `out`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_reflected1d_new`.
/// `out` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_reflected1d_values(
    ev: *const SmfReflected1D,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*ev.cast::<ReflectedState>() };
        let vals = &s.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(out, vals.len()) };
        buf.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return grid size; 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_reflected1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_reflected1d_size(ev: *const SmfReflected1D) -> usize {
    if ev.is_null() {
        return 0;
    }
    let s = unsafe { &*ev.cast::<ReflectedState>() };
    s.current.values.len()
}

/// Free a reflected handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_reflected1d_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_reflected1d_free(ev: *mut SmfReflected1D) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<ReflectedState>())) };
    }));
}

fn build_reflected(
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_chernoff: usize,
    origin: f64,
    u0: &[f64],
) -> Result<ReflectedState, semiflow_core::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n_grid)?;
    let diff = DiffusionChernoff::new(unit_a_bc, zero_bc, zero_bc, 1.0, grid);
    let region = HalfSpaceRegion::<f64, 1>::new([origin], [1.0])?;
    let kernel = ReflectedHeatChernoff::new(diff, region)?;
    let semigroup = ChernoffSemigroup::new(kernel, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(ReflectedState { semigroup, current })
}
