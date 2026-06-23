//! FFI surface for `Killing2ndChernoff` — soft-killing Feynman-Kac (ADR-0126, math §21.8).
//!
//! ## Symbol names
//!
//! - `smf_killing2nd_new` / `smf_killing2nd_evolve` / `smf_killing2nd_values`
//!   / `smf_killing2nd_size` / `smf_killing2nd_free`
//!
//! ## Default
//!
//! Unit diffusion (`a = 1`), constant `κ ≥ 0`.
//! Palindromic Strang `e^{−τκ/2} C(τ) e^{−τκ/2}`; order 2.

#![allow(unsafe_code)]

use std::os::raw::c_double;

use semiflow::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    killing_soft::{Killing2ndChernoff, KillingRate},
    BoundaryPolicy, ChernoffSemigroup,
};

use crate::handle::validate_u0_finite;
use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Constant killing rate
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ConstKappa(f64);

impl KillingRate<f64> for ConstKappa {
    fn kappa(&self, _x: f64) -> f64 {
        self.0
    }
}

type DiffUnit = DiffusionChernoff<f64>;
type Killing2ndUnit = Killing2ndChernoff<DiffUnit, ConstKappa, f64>;

// ---------------------------------------------------------------------------
// fn-pointer stubs
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_killing2nd(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_killing2nd(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a `Killing2ndChernoff` evolver.
#[repr(C)]
pub struct SmfKilling2nd {
    _private: [u8; 0],
}

struct InnerKilling2nd {
    semigroup: ChernoffSemigroup<Killing2ndUnit, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Allocate a `Killing2ndChernoff` 1-D evolver (unit diffusion, constant κ).
///
/// Solves `∂_t u = ∂²u − κ·u` via palindromic Strang; order 2 (ADR-0126).
///
/// # Safety
/// `u0` must point to `u0_len` readable `f64`s.
/// `out` must be a valid `*mut *mut SmfKilling2nd`.
#[no_mangle]
pub unsafe extern "C" fn smf_killing2nd_new(
    xmin: c_double,
    xmax: c_double,
    n: usize,
    kappa: c_double,
    u0: *const c_double,
    u0_len: usize,
    n_steps: usize,
    out: *mut *mut SmfKilling2nd,
) -> SemiflowStatus {
    if u0.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if !kappa.is_finite() || kappa < 0.0 {
        return SemiflowStatus::OutOfDomain;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_killing2nd(xmin, xmax, n, kappa, n_steps, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfKilling2nd>();
                unsafe { *out = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

/// Evolve `SmfKilling2nd` state by time `t`, writing result into `dst_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_killing2nd_new`.
/// `dst_buf` must be valid for `dst_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_killing2nd_evolve(
    ev: *mut SmfKilling2nd,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<InnerKilling2nd>() };
        let n = inner.current.values.len();
        if dst_len != n {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = evolve_killing2nd(inner, t) {
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

/// Copy current `SmfKilling2nd` grid values into `out_buf`.
///
/// # Safety
/// `ev` must be a live pointer from `smf_killing2nd_new`.
/// `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_killing2nd_values(
    ev: *const SmfKilling2nd,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<InnerKilling2nd>() };
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

/// Return `SmfKilling2nd` grid size; 0 if null.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_killing2nd_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_killing2nd_size(ev: *const SmfKilling2nd) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<InnerKilling2nd>() };
    inner.current.values.len()
}

/// Free a `SmfKilling2nd` handle.  Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_killing2nd_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_killing2nd_free(ev: *mut SmfKilling2nd) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<InnerKilling2nd>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn build_killing2nd(
    xmin: f64,
    xmax: f64,
    n: usize,
    kappa: f64,
    n_steps: usize,
    u0: &[f64],
) -> Result<InnerKilling2nd, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(BoundaryPolicy::Reflect);
    let diff = DiffusionChernoff::new(unit_a_killing2nd, zero_killing2nd, zero_killing2nd, 1.0, grid);
    let rate = ConstKappa(kappa);
    let kernel = Killing2ndUnit::new(diff, rate, grid)?;
    let semigroup = ChernoffSemigroup::new(kernel, n_steps)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(InnerKilling2nd { semigroup, current })
}

fn evolve_killing2nd(
    inner: &mut InnerKilling2nd,
    t: f64,
) -> Result<(), semiflow::SemiflowError> {
    let result = inner.semigroup.evolve(t, &inner.current)?;
    inner.current.values = result.values;
    Ok(())
}
