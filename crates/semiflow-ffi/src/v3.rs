//! v3.0 FFI surface (ADR-0076, Wave D, Approach A).
//!
//! Adds bare-name `_v3`-suffixed `extern "C"` functions exposing the v3.0
//! Rust API ([`Evolver`], [`Growth<f64>`], `apply_into`, zero-alloc) as an
//! **additive** companion to the existing v2.x surface in `ffi.rs`.
//!
//! ## Coexistence model
//!
//! The v2.x surface in `ffi.rs` (`smf_state_new_*`, `smf_evolve`,
//! `smf_state_values`, …) is PRESERVED UNCHANGED per ADR-0035 §9 12-month
//! deprecation cadence.  New C callers should use this v3 surface.
//!
//! ## v3 symbol naming
//!
//! Per ADR-0076 §Rationale "Why bare names for v3": the v3 surface IS the
//! long-term future surface; the `_v3` suffix in this additive (Approach A)
//! ship is a pragmatic compromise that avoids renaming the v2 symbols.
//!
//! ## Ownership model
//!
//! - `smf_evolver_new_heat_1d_unit_v3` allocates a `Box<EvolverInnerV3>` and
//!   transfers ownership to the caller as `*mut SmfEvolverV3`.
//! - The caller owns the handle until `smf_evolver_free_v3` is called.
//! - All `*const SmfEvolverV3` functions borrow for the call duration only.
//!
//! ## Panic safety
//!
//! Every entry point wraps its body in `catch_panic!` (see `panic.rs`).
//! Build with `--profile release-ffi` to ensure `panic = "unwind"`.
//!
//! ## Safety invariants
//!
//! 1. Null-check BEFORE `catch_panic!`.
//! 2. `*mut SmfEvolverV3` is always a live `Box<EvolverInnerV3>`.
//! 3. `smf_evolver_free_v3` is null-safe and idempotent.
//! 4. `(ptr, len)` slice pairs are caller-guaranteed valid.

#![allow(unsafe_code)]

use std::os::raw::c_double;

use semiflow::{
    ChernoffFunction, DiffusionChernoff, Evolver, Grid1D, GridFn1D, Growth, ScratchPool,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// C-visible struct: Growth bound mirror
// ---------------------------------------------------------------------------

/// C-ABI mirror of the Rust [`Growth<f64>`] struct (v3.0, ADR-0074).
///
/// Returned by [`smf_growth_v3`] by value; no heap allocation.
///
/// ```c
/// SmfGrowthV3 g = smf_growth_v3(ev);
/// printf("multiplier=%f omega=%f\n", g.multiplier, g.omega);
/// ```
#[repr(C)]
pub struct SmfGrowthV3 {
    /// M in `‖S(τ)‖ ≤ M · exp(ω · τ)`.  Always `≥ 1`.
    pub multiplier: c_double,
    /// ω in `‖S(τ)‖ ≤ M · exp(ω · τ)`.  Finite.
    pub omega: c_double,
}

// ---------------------------------------------------------------------------
// Opaque handle: SmfEvolverV3
// ---------------------------------------------------------------------------

/// Opaque handle to a v3.0 [`Evolver`] over `DiffusionChernoff<f64>`.
///
/// C callers receive a `SmfEvolverV3 *` from `smf_evolver_new_heat_1d_unit_v3`
/// and pass it to other `smf_evolver_*_v3` functions.  Free with
/// `smf_evolver_free_v3`.  Do not dereference or heap-allocate this struct.
#[repr(C)]
pub struct SmfEvolverV3 {
    _private: [u8; 0],
}

/// Inner data for `SmfEvolverV3` (Rust-private).
pub(crate) struct EvolverInnerV3 {
    pub evolver: Evolver<DiffusionChernoff<f64>>,
    pub current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Create a v3.0 [`Evolver`] for the unit-diffusion heat equation.
///
/// Solves `∂_t u = ∂_xx u` on `[xmin, xmax]` with `n_grid` nodes and
/// `n_chernoff` Chernoff iterations per call to `smf_evolver_evolve_into_v3`.
///
/// On success, `*out_ev` is set to a freshly allocated handle.
/// On any error, `*out_ev` is left unchanged.
///
/// ## Preconditions
/// - `xmin < xmax`; both finite.
/// - `n_grid >= 4`.
/// - `n_chernoff >= 1`.
/// - `u0` is non-null; `u0_len == n_grid`.
/// - All elements of `u0[0..u0_len]` must be finite.
/// - `out_ev` is a valid `*mut *mut SmfEvolverV3` (non-null).
///
/// ## Return values
/// - `Ok` (0) — success; `*out_ev` is set.
/// - `NullPtr` (5) — `u0` or `out_ev` is null.
/// - `GridMismatch` (1) — grid geometry invalid.
/// - `NanInf` (2) — non-finite element in `u0`.
/// - `OutOfDomain` (3) — `n_chernoff == 0`.
/// - `Panic` (99) — internal panic caught at FFI boundary.
///
/// # Safety
/// - `u0` must point to `u0_len` contiguous, readable `f64` values.
/// - `out_ev` must be a valid, writable `*mut *mut SmfEvolverV3`.
#[no_mangle]
pub unsafe extern "C" fn smf_evolver_new_heat_1d_unit_v3(
    xmin: c_double,
    xmax: c_double,
    n_grid: usize,
    n_chernoff: usize,
    u0: *const c_double,
    u0_len: usize,
    out_ev: *mut *mut SmfEvolverV3,
) -> SemiflowStatus {
    if u0.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_evolver_heat_unit(xmin, xmax, n_grid, n_chernoff, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfEvolverV3>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

/// Free an evolver handle created by `smf_evolver_new_*_v3`.
///
/// Null-safe: passing `NULL` is a no-op.  After this call the pointer is
/// dangling; do not use it again.
///
/// # Safety
/// - `ev` must be null or a live pointer from `smf_evolver_new_*_v3`.
#[no_mangle]
pub unsafe extern "C" fn smf_evolver_free_v3(ev: *mut SmfEvolverV3) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<EvolverInnerV3>())) };
    }));
}

// ---------------------------------------------------------------------------
// Evolution
// ---------------------------------------------------------------------------

/// Evolve the state by time `t` using the configured Chernoff iterations.
///
/// Writes `n_grid` evolved values into `dst_buf[0..dst_len]`.
/// The internal current state is updated.  Call again to chain time steps.
///
/// ## Preconditions
/// - `ev` is non-null and was obtained from `smf_evolver_new_*_v3`.
/// - `t >= 0.0` and finite.
/// - `dst_buf` is non-null and writable for `dst_len` `f64` values.
/// - `dst_len == n_grid` (the grid size passed to the constructor).
///
/// ## Return values
/// - `Ok` (0) — evolved values written to `dst_buf`.
/// - `NullPtr` (5) — `ev` or `dst_buf` is null.
/// - `GridMismatch` (1) — `dst_len != n_grid`.
/// - `OutOfDomain` (3) — `t < 0`, NaN, or Inf.
/// - `Panic` (99) — internal panic at FFI boundary.
///
/// # Safety
/// - `ev` must be a live pointer from `smf_evolver_new_*_v3`.
/// - `dst_buf` must be valid for `dst_len` writable, well-aligned `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_evolver_evolve_into_v3(
    ev: *mut SmfEvolverV3,
    t: c_double,
    dst_buf: *mut c_double,
    dst_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || dst_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<EvolverInnerV3>() };
        let expected = inner.current.values.len();
        if dst_len != expected {
            return SemiflowStatus::GridMismatch;
        }
        let mut dst = inner.current.clone();
        let mut scratch = ScratchPool::<f64>::new();
        if let Err(e) = inner
            .evolver
            .evolve_into(t, &inner.current, &mut dst, &mut scratch)
        {
            SemiflowStatus::from(&e)
        } else {
            inner.current = dst;
            let out = unsafe { std::slice::from_raw_parts_mut(dst_buf, dst_len) };
            out.copy_from_slice(&inner.current.values);
            SemiflowStatus::Ok
        }
    })
}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------

/// Copy the current grid values into `out_buf`.
///
/// Writes exactly `n_grid` `f64` values.  Must be called after construction
/// or `smf_evolver_evolve_into_v3` to read the current state.
///
/// ## Return values
/// - `Ok` (0) — values written.
/// - `NullPtr` (5) — `ev` or `out_buf` is null.
/// - `GridMismatch` (1) — `out_len < n_grid`.
/// - `Panic` (99) — internal panic at FFI boundary.
///
/// # Safety
/// - `ev` must be a live pointer from `smf_evolver_new_*_v3`.
/// - `out_buf` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_evolver_values_v3(
    ev: *const SmfEvolverV3,
    out_buf: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<EvolverInnerV3>() };
        let vals = &inner.current.values;
        if out_len < vals.len() {
            return SemiflowStatus::GridMismatch;
        }
        let out = unsafe { std::slice::from_raw_parts_mut(out_buf, vals.len()) };
        out.copy_from_slice(vals);
        SemiflowStatus::Ok
    })
}

/// Return the grid size for an evolver handle.
///
/// Returns `0` if `ev` is null.
///
/// # Safety
/// - `ev` must be null or a live pointer from `smf_evolver_new_*_v3`.
#[no_mangle]
pub unsafe extern "C" fn smf_evolver_size_v3(ev: *const SmfEvolverV3) -> usize {
    if ev.is_null() {
        return 0;
    }
    let inner = unsafe { &*ev.cast::<EvolverInnerV3>() };
    inner.current.values.len()
}

// ---------------------------------------------------------------------------
// Growth bound
// ---------------------------------------------------------------------------

/// Return the growth bound of the underlying [`DiffusionChernoff`] kernel.
///
/// Returns `SmfGrowthV3 { multiplier: 0.0, omega: 0.0 }` if `ev` is null.
///
/// For unit-diffusion, expected values: `multiplier = 1.0`, `omega = 0.0`
/// (contractive semigroup).
///
/// # Safety
/// - `ev` must be null or a live pointer from `smf_evolver_new_*_v3`.
#[no_mangle]
pub unsafe extern "C" fn smf_growth_v3(ev: *const SmfEvolverV3) -> SmfGrowthV3 {
    if ev.is_null() {
        return SmfGrowthV3 {
            multiplier: 0.0,
            omega: 0.0,
        };
    }
    let inner = unsafe { &*ev.cast::<EvolverInnerV3>() };
    let Growth { multiplier, omega } = inner.evolver.func.growth();
    SmfGrowthV3 { multiplier, omega }
}

// ---------------------------------------------------------------------------
// Private builder helper
// ---------------------------------------------------------------------------

/// Build an `EvolverInnerV3` for unit-diffusion heat on `[xmin, xmax]`.
fn build_evolver_heat_unit(
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_chernoff: usize,
    u0: &[f64],
) -> Result<EvolverInnerV3, semiflow::SemiflowError> {
    crate::handle::validate_u0_finite(u0)?;
    let grid = Grid1D::new(xmin, xmax, n_grid)?;
    let chernoff = DiffusionChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
    let evolver = Evolver::new(chernoff, n_chernoff)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(EvolverInnerV3 { evolver, current })
}

extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_d(_: f64) -> f64 {
    0.0
}
