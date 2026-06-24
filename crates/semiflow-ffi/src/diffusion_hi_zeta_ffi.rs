//! FFI surface for zeta-lifted 1-D diffusion engines (ζ⁴, ζ⁶, ζ⁸).
//!
//! Exposes `Diffusion4thZeta4Chernoff`, `Diffusion6thZeta6Chernoff`, and
//! `Diffusion8thZeta8Chernoff` with unit diffusion `a = 1` via C-ABI opaque
//! handles following the same idiom as `diffusion_hi_ffi.rs`.
//!
//! Chain: D4 → Zeta4 → Zeta6 → Zeta8 (nested Richardson lifts).
//! Safety: null-check before `catch_panic!`; `_free` is null-safe.

#![allow(unsafe_code)]

use std::os::raw::c_double;

use semiflow::{
    BoundaryPolicy, ChernoffSemigroup, Diffusion4thChernoff, Diffusion4thZeta4Chernoff,
    Diffusion6thZeta6Chernoff, Diffusion8thZeta8Chernoff, Grid1D, GridFn1D,
};

use crate::{handle::validate_u0_finite, status::SemiflowStatus};

extern "Rust" fn unit_a_z(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_d_z(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// ζ⁴ engine — Diffusion4thZeta4Chernoff
// ---------------------------------------------------------------------------

/// Opaque handle to a `Diffusion4thZeta4Chernoff` evolver.
#[repr(C)]
pub struct SmfHeat1DZeta4 {
    _private: [u8; 0],
}

struct InnerZeta4 {
    semigroup: ChernoffSemigroup<Diffusion4thZeta4Chernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a ζ⁴ 1-D heat evolver (temporal order 4, unit `a = 1`).
///
/// # Preconditions
/// `xmin < xmax`, `n >= 4`, `u0_len == n`, `n_chernoff >= 1`, ptrs non-null.
///
/// # Return values
/// `Ok`/`NullPtr`/`GridMismatch`/`NanInf`/`OutOfDomain`/`Panic`.
///
/// # Safety
/// `u0` → `u0_len` readable `f64`s; `out` → valid `*mut *mut SmfHeat1DZeta4`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta4_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfHeat1DZeta4,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_zeta4(xmin, xmax, n, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfHeat1DZeta4>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve the ζ⁴ state by time `t`.
///
/// # Safety
/// `ev` → live `smf_heat1d_zeta4_new` handle; `dst_buf` → `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta4_evolve(
    ev: *mut SmfHeat1DZeta4,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerZeta4>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = run_evolve4(inner, t) {
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
/// `ev` → live handle; `out_buf` → `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta4_values(
    ev: *const SmfHeat1DZeta4,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerZeta4>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return the grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or a live `smf_heat1d_zeta4_new` handle.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta4_size(ev: *const SmfHeat1DZeta4) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<InnerZeta4>() }.current.values.len()
}

/// Free a ζ⁴ handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live `smf_heat1d_zeta4_new` handle.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta4_free(ev: *mut SmfHeat1DZeta4) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerZeta4>())) };
    }));
}

// ---------------------------------------------------------------------------
// ζ⁶ engine — Diffusion6thZeta6Chernoff
// ---------------------------------------------------------------------------

/// Opaque handle to a `Diffusion6thZeta6Chernoff` evolver.
#[repr(C)]
pub struct SmfHeat1DZeta6 {
    _private: [u8; 0],
}

struct InnerZeta6 {
    semigroup: ChernoffSemigroup<Diffusion6thZeta6Chernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a ζ⁶ 1-D heat evolver (temporal order 6, unit `a = 1`).
///
/// # Safety
/// `u0` → `u0_len` readable `f64`s; `out` → valid `*mut *mut SmfHeat1DZeta6`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta6_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfHeat1DZeta6,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_zeta6(xmin, xmax, n, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfHeat1DZeta6>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve the ζ⁶ state by time `t`.
///
/// # Safety
/// `ev` → live handle; `dst_buf` → `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta6_evolve(
    ev: *mut SmfHeat1DZeta6,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerZeta6>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = run_evolve6(inner, t) {
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
/// `ev` → live handle; `out_buf` → `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta6_values(
    ev: *const SmfHeat1DZeta6,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerZeta6>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return the grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or a live `smf_heat1d_zeta6_new` handle.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta6_size(ev: *const SmfHeat1DZeta6) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<InnerZeta6>() }.current.values.len()
}

/// Free a ζ⁶ handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live `smf_heat1d_zeta6_new` handle.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta6_free(ev: *mut SmfHeat1DZeta6) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerZeta6>())) };
    }));
}

// ---------------------------------------------------------------------------
// ζ⁸ engine — Diffusion8thZeta8Chernoff
// ---------------------------------------------------------------------------

/// Opaque handle to a `Diffusion8thZeta8Chernoff` evolver.
#[repr(C)]
pub struct SmfHeat1DZeta8 {
    _private: [u8; 0],
}

struct InnerZeta8 {
    semigroup: ChernoffSemigroup<Diffusion8thZeta8Chernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

/// Allocate a ζ⁸ 1-D heat evolver (temporal order 8, unit `a = 1`).
///
/// # Safety
/// `u0` → `u0_len` readable `f64`s; `out` → valid `*mut *mut SmfHeat1DZeta8`.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta8_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out: *mut *mut SmfHeat1DZeta8,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_zeta8(xmin, xmax, n, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfHeat1DZeta8>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

/// Evolve the ζ⁸ state by time `t`.
///
/// # Safety
/// `ev` → live handle; `dst_buf` → `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta8_evolve(
    ev: *mut SmfHeat1DZeta8,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerZeta8>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = run_evolve8(inner, t) {
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
/// `ev` → live handle; `out_buf` → `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta8_values(
    ev: *const SmfHeat1DZeta8,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerZeta8>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return the grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or a live `smf_heat1d_zeta8_new` handle.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta8_size(ev: *const SmfHeat1DZeta8) -> usize {
    if ev.is_null() {
        return 0;
    }
    unsafe { &*ev.cast::<InnerZeta8>() }.current.values.len()
}

/// Free a ζ⁸ handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live `smf_heat1d_zeta8_new` handle.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_zeta8_free(ev: *mut SmfHeat1DZeta8) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerZeta8>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private builder helpers
// ---------------------------------------------------------------------------

fn build_zeta4(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<InnerZeta4, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let d4 = Diffusion4thChernoff::new(unit_a_z, zero_d_z, zero_d_z, 1.0, grid);
    let zeta4 = Diffusion4thZeta4Chernoff::new(d4, Some(1.0_f64))?;
    let semigroup = ChernoffSemigroup::new(zeta4, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerZeta4 { semigroup, current })
}

fn build_zeta6(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<InnerZeta6, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let d4 = Diffusion4thChernoff::new(unit_a_z, zero_d_z, zero_d_z, 1.0, grid);
    let zeta4 = Diffusion4thZeta4Chernoff::new(d4, Some(1.0_f64))?;
    let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, Some(1.0_f64))?;
    let semigroup = ChernoffSemigroup::new(zeta6, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerZeta6 { semigroup, current })
}

fn build_zeta8(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<InnerZeta8, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let d4 = Diffusion4thChernoff::new(unit_a_z, zero_d_z, zero_d_z, 1.0, grid);
    let zeta4 = Diffusion4thZeta4Chernoff::new(d4, Some(1.0_f64))?;
    let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, Some(1.0_f64))?;
    let zeta8 = Diffusion8thZeta8Chernoff::new(zeta6, Some(1.0_f64))?;
    let semigroup = ChernoffSemigroup::new(zeta8, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerZeta8 { semigroup, current })
}

fn run_evolve4(i: &mut InnerZeta4, t: f64) -> Result<(), semiflow::SemiflowError> {
    i.current.values = i.semigroup.evolve(t, &i.current)?.values;
    Ok(())
}
fn run_evolve6(i: &mut InnerZeta6, t: f64) -> Result<(), semiflow::SemiflowError> {
    i.current.values = i.semigroup.evolve(t, &i.current)?.values;
    Ok(())
}
fn run_evolve8(i: &mut InnerZeta8, t: f64) -> Result<(), semiflow::SemiflowError> {
    i.current.values = i.semigroup.evolve(t, &i.current)?.values;
    Ok(())
}
