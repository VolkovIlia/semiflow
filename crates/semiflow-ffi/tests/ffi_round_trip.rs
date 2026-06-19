//! Rust-side integration test: verifies all `semiflow_ffi` extern "C" functions.
//!
//! Compares FFI output against the pure-Rust `ChernoffSemigroup::evolve`
//! reference (rather than a closed-form oracle) to avoid hard-coding the
//! diffusion-convention factor.  The sup-error between the two paths must be
//! below f64 machine precision noise (< 1e-12).

#![allow(unsafe_code)]

use semiflow_core::{ChernoffSemigroup, DiffusionChernoff, Grid1D, GridFn1D};
use semiflow_ffi::{
    smf_evolve, smf_state_free, smf_state_new_heat_1d_unit,
    smf_state_new_with_closure, smf_state_size, smf_state_values, smf_status_str,
    smf_version, SemiflowState, SemiflowStatus,
};

// ---------------------------------------------------------------------------
// Shared scenario parameters
// ---------------------------------------------------------------------------

const XMIN: f64 = -10.0;
const XMAX: f64 = 10.0;
const N: usize = 1000;
const T: f64 = 1.0;
const N_STEPS: usize = 100;

fn make_u0() -> Vec<f64> {
    let grid = Grid1D::new(XMIN, XMAX, N).unwrap();
    (0..N).map(|i| (-grid.x_at(i).powi(2)).exp()).collect()
}

fn rust_reference(u0: &[f64]) -> Vec<f64> {
    extern "Rust" fn unit_a(_: f64) -> f64 {
        1.0
    }
    extern "Rust" fn zero_deriv(_: f64) -> f64 {
        0.0
    }
    let grid = Grid1D::new(XMIN, XMAX, N).unwrap();
    let chernoff = DiffusionChernoff::new(unit_a, zero_deriv, zero_deriv, 1.0, grid);
    let sg = ChernoffSemigroup::new(chernoff, N_STEPS).unwrap();
    let gf = GridFn1D::new(grid, u0.to_vec()).unwrap();
    sg.evolve(T, &gf).unwrap().values
}

// ---------------------------------------------------------------------------
// Happy path: construct, evolve, read back
// ---------------------------------------------------------------------------

#[test]
fn test_new_evolve_values() {
    let u0 = make_u0();
    let mut state_ptr: *mut SemiflowState = std::ptr::null_mut();

    let status = unsafe {
        smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), u0.len(), &mut state_ptr)
    };
    assert_eq!(status, SemiflowStatus::Ok, "new_heat_1d_unit failed");
    assert!(!state_ptr.is_null());

    let sz = unsafe { smf_state_size(state_ptr.cast_const()) };
    assert_eq!(sz, N);

    let ev_status = unsafe { smf_evolve(state_ptr, T, N_STEPS) };
    assert_eq!(ev_status, SemiflowStatus::Ok, "evolve failed");

    let mut out = vec![0.0f64; N];
    let rd_status =
        unsafe { smf_state_values(state_ptr.cast_const(), out.as_mut_ptr(), out.len()) };
    assert_eq!(rd_status, SemiflowStatus::Ok);

    let reference = rust_reference(&u0);
    let sup_err = out
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        sup_err < 1e-12,
        "FFI vs pure-Rust sup-error = {sup_err:.3e} (expected < 1e-12)"
    );

    unsafe { smf_state_free(state_ptr) };
}

// ---------------------------------------------------------------------------
// Null-pointer guards
// ---------------------------------------------------------------------------

#[test]
fn test_null_u0_returns_null_ptr() {
    let mut ptr = std::ptr::null_mut();
    let status =
        unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, N, std::ptr::null(), 0, &mut ptr) };
    assert_eq!(status, SemiflowStatus::NullPtr);
}

#[test]
fn test_null_out_state_returns_null_ptr() {
    let u0 = make_u0();
    let status = unsafe {
        smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), u0.len(), std::ptr::null_mut())
    };
    assert_eq!(status, SemiflowStatus::NullPtr);
}

#[test]
fn test_free_null_is_noop() {
    // Must not crash.
    unsafe { smf_state_free(std::ptr::null_mut()) };
}

#[test]
fn test_size_null_returns_zero() {
    let sz = unsafe { smf_state_size(std::ptr::null()) };
    assert_eq!(sz, 0);
}

#[test]
fn test_evolve_null_returns_null_ptr() {
    let status = unsafe { smf_evolve(std::ptr::null_mut(), 1.0, 10) };
    assert_eq!(status, SemiflowStatus::NullPtr);
}

#[test]
fn test_values_null_state_returns_null_ptr() {
    let mut buf = [0.0f64; 10];
    let status = unsafe { smf_state_values(std::ptr::null(), buf.as_mut_ptr(), buf.len()) };
    assert_eq!(status, SemiflowStatus::NullPtr);
}

// ---------------------------------------------------------------------------
// Error path: invalid grid
// ---------------------------------------------------------------------------

#[test]
fn test_invalid_n_returns_grid_mismatch() {
    let u0 = [1.0f64; 3]; // n=3 < 4 minimum
    let mut ptr = std::ptr::null_mut();
    let status =
        unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, 3, u0.as_ptr(), u0.len(), &mut ptr) };
    assert_ne!(status, SemiflowStatus::Ok);
    assert!(ptr.is_null());
}

#[test]
fn test_nan_in_u0_returns_error() {
    let mut u0 = make_u0();
    u0[42] = f64::NAN;
    let mut ptr = std::ptr::null_mut();
    let status =
        unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), u0.len(), &mut ptr) };
    assert_ne!(status, SemiflowStatus::Ok);
    assert!(ptr.is_null());
}

// ---------------------------------------------------------------------------
// Error path: xmin >= xmax
// ---------------------------------------------------------------------------

#[test]
fn test_xmin_ge_xmax_returns_grid_mismatch() {
    let u0 = vec![0.0_f64; 1000];
    let mut ptr = std::ptr::null_mut();
    let st = unsafe {
        smf_state_new_heat_1d_unit(10.0, -10.0, 1000, u0.as_ptr(), u0.len(), &mut ptr)
    };
    assert_eq!(st, SemiflowStatus::GridMismatch);
    assert!(ptr.is_null());
}

// ---------------------------------------------------------------------------
// with_closure: happy-path and null-guard (ADR-0034 S1.2)
// ---------------------------------------------------------------------------

/// Callback: `a(x) = 1.0` via `user_data` pointer to a constant.
unsafe extern "C" fn cb_const_a(_x: f64, ud: *mut ()) -> f64 {
    *(ud as *const f64)
}

/// Callback: `a'(x) = 0.0` (constant a).
unsafe extern "C" fn cb_zero(_x: f64, _ud: *mut ()) -> f64 {
    0.0
}

/// `with_closure` with `a = 1.0` via callback must match the unit-a path exactly.
///
/// Both paths call `DiffusionChernoff::with_closure` / `new` with `a(x) = 1.0`,
/// so the sup-error must be below f64 machine precision noise.
#[test]
fn test_with_closure_unit_a_matches_unit_path() {
    let u0 = make_u0();
    let a_value: f64 = 1.0;

    let mut state_ptr: *mut SemiflowState = std::ptr::null_mut();
    let status = unsafe {
        smf_state_new_with_closure(
            XMIN,
            XMAX,
            N,
            Some(cb_const_a),
            Some(cb_zero),
            Some(cb_zero),
            std::ptr::addr_of!(a_value).cast_mut().cast(),
            1.0,
            u0.as_ptr(),
            u0.len(),
            &mut state_ptr,
        )
    };
    assert_eq!(status, SemiflowStatus::Ok, "new_with_closure failed");
    assert!(!state_ptr.is_null());

    let ev_status = unsafe { smf_evolve(state_ptr, T, N_STEPS) };
    assert_eq!(ev_status, SemiflowStatus::Ok, "evolve failed");

    let mut out_closure = vec![0.0f64; N];
    let rd = unsafe { smf_state_values(state_ptr.cast_const(), out_closure.as_mut_ptr(), N) };
    assert_eq!(rd, SemiflowStatus::Ok);
    unsafe { smf_state_free(state_ptr) };

    let reference = rust_reference(&u0);
    let sup_err = out_closure
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        sup_err < 1e-12,
        "with_closure unit-a vs pure-Rust sup-error = {sup_err:.3e} (expected < 1e-12)"
    );
}

/// Null function pointer must return `NullPtr` without crashing.
#[test]
fn test_with_closure_null_fn_returns_null_ptr() {
    let u0 = make_u0();
    let mut ptr = std::ptr::null_mut();
    let st = unsafe {
        smf_state_new_with_closure(
            XMIN,
            XMAX,
            N,
            None,
            None,
            None,
            std::ptr::null_mut(),
            1.0,
            u0.as_ptr(),
            u0.len(),
            &mut ptr,
        )
    };
    assert_eq!(st, SemiflowStatus::NullPtr);
    assert!(ptr.is_null());
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[test]
fn test_status_str_ok() {
    let ptr = smf_status_str(SemiflowStatus::Ok);
    let s = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap();
    assert_eq!(s, "Ok");
}

#[test]
fn test_version_not_empty() {
    let ptr = smf_version();
    let s = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap();
    assert!(!s.is_empty(), "version string is empty");
    // Accept any semver X.Y.Z (not pinned to 0.x.y since crate is now >=1.0).
    assert!(
        s.chars().next().is_some_and(|c| c.is_ascii_digit()),
        "expected semver X.Y.Z, got: {s}"
    );
}
