//! Edge-case ABI tests for `semiflow-ffi`.
//!
//! Complements `ffi_round_trip.rs` (happy path) with adversarial inputs:
//! mismatched lengths, non-finite values, reversed bounds, zero-width grids,
//! and lifecycle invariants.
//!
//! ## Safety note
//!
//! The FFI contract requires callers to set pointers to NULL after
//! `smf_state_free`.  Double-free or use-after-free are UB; this suite
//! does NOT exercise them because such tests would risk crashing the harness
//! and would document invalid usage rather than correct behaviour.

#![allow(unsafe_code)]

use std::ffi::CStr;

use semiflow_ffi::{
    smf_evolve, smf_state_free, smf_state_new_heat_1d_unit, smf_state_values, smf_status_str,
    smf_version, SemiflowState, SemiflowStatus,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const XMIN: f64 = -10.0;
const XMAX: f64 = 10.0;
const N: usize = 1000;
const N_STEPS: usize = 100;

/// Allocate a valid state (`n=1000` Gaussian `u0`), panicking on failure.
fn make_state() -> *mut SemiflowState {
    #[allow(clippy::cast_precision_loss)]
    let u0: Vec<f64> = (0..N)
        .map(|i| {
            let x = XMIN + (XMAX - XMIN) * (i as f64) / ((N - 1) as f64);
            (-x * x).exp()
        })
        .collect();
    let mut ptr: *mut SemiflowState = std::ptr::null_mut();
    let s = unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), u0.len(), &mut ptr) };
    assert_eq!(s, SemiflowStatus::Ok, "make_state: construction failed");
    assert!(!ptr.is_null());
    ptr
}

// ---------------------------------------------------------------------------
// Constructor edge cases
// ---------------------------------------------------------------------------

/// EC-C1: `u0_len` one less than `n` — `GridFn1D` rejects mismatched lengths.
#[test]
fn test_u0_len_shorter_than_n() {
    let u0 = vec![1.0f64; N - 1];
    let mut ptr: *mut SemiflowState = std::ptr::null_mut();
    let s = unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), u0.len(), &mut ptr) };
    assert_ne!(s, SemiflowStatus::Ok, "expected error for u0_len < n");
    assert!(ptr.is_null(), "ptr must remain null on error");
}

/// EC-C2: `u0_len` one more than `n` — `GridFn1D` rejects mismatched lengths.
#[test]
fn test_u0_len_longer_than_n() {
    let u0 = vec![1.0f64; N + 1];
    let mut ptr: *mut SemiflowState = std::ptr::null_mut();
    let s = unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), u0.len(), &mut ptr) };
    assert_ne!(s, SemiflowStatus::Ok, "expected error for u0_len > n");
    assert!(ptr.is_null());
}

/// EC-C3: `u0_len = 0` with `n > 0` — degenerate, must not crash.
#[test]
fn test_u0_len_zero_with_positive_n() {
    let u0 = Vec::<f64>::new();
    let mut ptr: *mut SemiflowState = std::ptr::null_mut();
    let s = unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), 0, &mut ptr) };
    assert_ne!(s, SemiflowStatus::Ok);
    assert!(ptr.is_null());
}

/// EC-C4: `Inf` in `u0` — must return `NanInf` (non-finite value guard).
#[test]
fn test_inf_in_u0() {
    let mut u0 = vec![1.0f64; N];
    u0[0] = f64::INFINITY;
    let mut ptr: *mut SemiflowState = std::ptr::null_mut();
    let s = unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), u0.len(), &mut ptr) };
    assert_eq!(s, SemiflowStatus::NanInf, "Inf in u0 must return NanInf");
    assert!(ptr.is_null());
}

/// EC-C5: Negative `Inf` in `u0` — same `NanInf` guard as positive `Inf`.
#[test]
fn test_neg_inf_in_u0() {
    let mut u0 = vec![1.0f64; N];
    u0[0] = f64::NEG_INFINITY;
    let mut ptr: *mut SemiflowState = std::ptr::null_mut();
    let s = unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), u0.len(), &mut ptr) };
    assert_eq!(s, SemiflowStatus::NanInf);
    assert!(ptr.is_null());
}

/// EC-C6: Reversed bounds (`xmin > xmax`) — `Grid1D::new` must reject.
#[test]
fn test_reversed_bounds() {
    let u0 = vec![1.0f64; N];
    let mut ptr: *mut SemiflowState = std::ptr::null_mut();
    let s = unsafe { smf_state_new_heat_1d_unit(10.0, -10.0, N, u0.as_ptr(), u0.len(), &mut ptr) };
    assert_ne!(s, SemiflowStatus::Ok, "reversed bounds must be rejected");
    assert!(ptr.is_null());
}

/// EC-C7: Zero-width grid (`xmin == xmax`) — `Grid1D::new` must reject.
#[test]
fn test_zero_width_grid() {
    let u0 = vec![1.0f64; N];
    let mut ptr: *mut SemiflowState = std::ptr::null_mut();
    let s = unsafe { smf_state_new_heat_1d_unit(5.0, 5.0, N, u0.as_ptr(), u0.len(), &mut ptr) };
    assert_ne!(s, SemiflowStatus::Ok, "zero-width grid must be rejected");
    assert!(ptr.is_null());
}

/// EC-C8: Very large `n` (`usize::MAX`) — must fail gracefully before OOM.
///
/// Marked `#[ignore = "would OOM"]` because allocating `usize::MAX` f64 values
/// exhausts memory and kills the test runner on any real machine.
/// If the implementation ever adds a capacity guard this test can be
/// un-ignored and tightened.
#[test]
#[ignore = "would OOM on any real machine — kept as a documentation stub"]
fn test_very_large_n_does_not_segfault() {
    let u0 = [1.0f64; 4]; // stub — execution never reaches here
    let mut ptr: *mut SemiflowState = std::ptr::null_mut();
    let s = unsafe {
        smf_state_new_heat_1d_unit(XMIN, XMAX, usize::MAX, u0.as_ptr(), u0.len(), &mut ptr)
    };
    // If the impl ever adds an early-out it must return non-Ok, not panic.
    assert_ne!(s, SemiflowStatus::Ok);
    assert!(ptr.is_null());
}

// ---------------------------------------------------------------------------
// Inspection edge cases
// ---------------------------------------------------------------------------

/// EC-I1: `out_buf_len < n` — must return `GridMismatch` without OOB write.
#[test]
fn test_values_buf_too_short() {
    let state = make_state();
    let mut out = vec![0.0f64; N - 1];
    let s = unsafe { smf_state_values(state.cast_const(), out.as_mut_ptr(), out.len()) };
    assert_eq!(s, SemiflowStatus::GridMismatch);
    unsafe { smf_state_free(state) };
}

/// EC-I2: `out_buf_len > n` — succeeds; canary bytes beyond `n` must stay.
#[test]
fn test_values_buf_oversized_no_oob_write() {
    const TAIL: usize = 10;
    // Sentinel value written into the tail before the call.
    const CANARY: f64 = 57_005.0; // 0xDEAD as f64
    let state = make_state();
    let mut out = vec![CANARY; N + TAIL];
    let s = unsafe { smf_state_values(state.cast_const(), out.as_mut_ptr(), out.len()) };
    assert_eq!(s, SemiflowStatus::Ok);
    for (i, &v) in out[N..].iter().enumerate() {
        // Use abs-diff instead of strict `==` to avoid clippy::float_cmp.
        assert!(
            (v - CANARY).abs() < f64::EPSILON,
            "canary overwritten at tail[{i}]"
        );
    }
    unsafe { smf_state_free(state) };
}

/// EC-I3: `NULL` `out_buf` — must return `NullPtr`.
#[test]
fn test_values_null_out_buf() {
    let state = make_state();
    let s = unsafe { smf_state_values(state.cast_const(), std::ptr::null_mut(), N) };
    assert_eq!(s, SemiflowStatus::NullPtr);
    unsafe { smf_state_free(state) };
}

// ---------------------------------------------------------------------------
// Evolution edge cases
// ---------------------------------------------------------------------------

/// EC-E1: `t = 0.0` — `ChernoffSemigroup` accepts `t >= 0`; must return `Ok`.
///
/// Note: `t=0` applies `n_steps` Chernoff kernels each with `tau = 0/n`, so
/// the result is NOT guaranteed to be the identity (the kernel is applied with
/// a zero step but still executes).  We only assert `Ok` + finite output.
#[test]
fn test_evolve_t_zero() {
    let state = make_state();
    let s = unsafe { smf_evolve(state, 0.0, N_STEPS) };
    assert_eq!(s, SemiflowStatus::Ok, "t=0 must be accepted");

    let mut after = vec![0.0f64; N];
    let r = unsafe { smf_state_values(state.cast_const(), after.as_mut_ptr(), N) };
    assert_eq!(r, SemiflowStatus::Ok);
    assert!(
        after.iter().all(|v| v.is_finite()),
        "values after t=0 must be finite"
    );

    unsafe { smf_state_free(state) };
}

/// EC-E2: `t < 0.0` — must return `OutOfDomain` (not panic).
#[test]
fn test_evolve_negative_t() {
    let state = make_state();
    let s = unsafe { smf_evolve(state, -1.0, N_STEPS) };
    assert_eq!(
        s,
        SemiflowStatus::OutOfDomain,
        "negative t must yield OutOfDomain"
    );
    unsafe { smf_state_free(state) };
}

/// EC-E3: `n_steps = 0` — `OutOfDomain` per `ffi.rs` explicit guard (line ~102).
#[test]
fn test_evolve_n_steps_zero() {
    let state = make_state();
    let s = unsafe { smf_evolve(state, 1.0, 0) };
    assert_eq!(
        s,
        SemiflowStatus::OutOfDomain,
        "n_steps=0 must yield OutOfDomain"
    );
    unsafe { smf_state_free(state) };
}

/// EC-E4: `NaN` `t` — must return non-`Ok` status.
#[test]
fn test_evolve_nan_t() {
    let state = make_state();
    let s = unsafe { smf_evolve(state, f64::NAN, N_STEPS) };
    assert_ne!(s, SemiflowStatus::Ok, "NaN t must be rejected");
    unsafe { smf_state_free(state) };
}

/// EC-E5: `Inf` `t` — must return non-`Ok` status.
#[test]
fn test_evolve_inf_t() {
    let state = make_state();
    let s = unsafe { smf_evolve(state, f64::INFINITY, N_STEPS) };
    assert_ne!(s, SemiflowStatus::Ok, "Inf t must be rejected");
    unsafe { smf_state_free(state) };
}

/// EC-E6: Three successive evolve calls sum to approximately 1.0 time unit.
///
/// Tolerance is `5e-3` (loose) because Chernoff steps are truncation
/// approximations — they don't compose exactly across time-step boundaries.
#[test]
fn test_evolve_multiple_calls_compose() {
    #[allow(clippy::cast_precision_loss)]
    let u0: Vec<f64> = (0..N)
        .map(|i| {
            let x = XMIN + (XMAX - XMIN) * (i as f64) / ((N - 1) as f64);
            (-x * x).exp()
        })
        .collect();

    // Reference: single shot t=1.0.
    let mut ref_state: *mut SemiflowState = std::ptr::null_mut();
    let s_ref =
        unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), u0.len(), &mut ref_state) };
    assert_eq!(s_ref, SemiflowStatus::Ok);
    assert_eq!(
        unsafe { smf_evolve(ref_state, 1.0, N_STEPS) },
        SemiflowStatus::Ok
    );
    let mut ref_vals = vec![0.0f64; N];
    unsafe { smf_state_values(ref_state.cast_const(), ref_vals.as_mut_ptr(), N) };
    unsafe { smf_state_free(ref_state) };

    // Multi-step: 0.33 + 0.34 + 0.33 ≈ 1.0.
    let mut multi_state: *mut SemiflowState = std::ptr::null_mut();
    assert_eq!(
        unsafe {
            smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), u0.len(), &mut multi_state)
        },
        SemiflowStatus::Ok,
    );
    for &t in &[0.33_f64, 0.34, 0.33] {
        assert_eq!(
            unsafe { smf_evolve(multi_state, t, N_STEPS) },
            SemiflowStatus::Ok,
            "evolve({t}) failed",
        );
    }
    let mut multi_vals = vec![0.0f64; N];
    unsafe { smf_state_values(multi_state.cast_const(), multi_vals.as_mut_ptr(), N) };
    unsafe { smf_state_free(multi_state) };

    let sup_err = ref_vals
        .iter()
        .zip(multi_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        sup_err < 5e-3,
        "multi-step vs single-shot sup-error = {sup_err:.3e} (expected < 5e-3)"
    );
}

// ---------------------------------------------------------------------------
// Lifecycle edge cases
// ---------------------------------------------------------------------------

/// EC-L1: Evolve, read, evolve, read — state mutation persists across calls.
#[test]
fn test_evolve_read_evolve_read_state_is_persistent() {
    let state = make_state();

    assert_eq!(
        unsafe { smf_evolve(state, 0.5, N_STEPS) },
        SemiflowStatus::Ok
    );

    let mut vals1 = vec![0.0f64; N];
    assert_eq!(
        unsafe { smf_state_values(state.cast_const(), vals1.as_mut_ptr(), N) },
        SemiflowStatus::Ok,
    );

    assert_eq!(
        unsafe { smf_evolve(state, 0.5, N_STEPS) },
        SemiflowStatus::Ok
    );

    let mut vals2 = vec![0.0f64; N];
    assert_eq!(
        unsafe { smf_state_values(state.cast_const(), vals2.as_mut_ptr(), N) },
        SemiflowStatus::Ok,
    );

    // Second read must differ from first (further diffusion occurred).
    let changed = vals1
        .iter()
        .zip(vals2.iter())
        .any(|(a, b)| (a - b).abs() > 1e-15);
    assert!(changed, "second evolve must change the state");

    unsafe { smf_state_free(state) };
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/// EC-D1: Every `SemiflowStatus` variant maps to a non-empty string that
/// matches its Rust variant name exactly.
#[test]
fn test_status_str_all_variants() {
    let cases: &[(SemiflowStatus, &str)] = &[
        (SemiflowStatus::Ok, "Ok"),
        (SemiflowStatus::GridMismatch, "GridMismatch"),
        (SemiflowStatus::NanInf, "NanInf"),
        (SemiflowStatus::OutOfDomain, "OutOfDomain"),
        (SemiflowStatus::BoundaryFailure, "BoundaryFailure"),
        (SemiflowStatus::NullPtr, "NullPtr"),
        (SemiflowStatus::CflViolated, "CflViolated"),
        (SemiflowStatus::ConvergenceFailed, "ConvergenceFailed"),
        (SemiflowStatus::Unsupported, "Unsupported"),
        (SemiflowStatus::Panic, "Panic"),
    ];
    for &(variant, expected) in cases {
        let ptr = smf_status_str(variant);
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(s, expected, "status_str({expected:?}) mismatch");
    }
}

/// EC-D2: `version` string parses as semver (`MAJOR.MINOR.PATCH[-pre]`).
///
/// Accepts both stable (`1.2.3`) and pre-release (`2.1.0-rc.1`) forms.
/// The first three `.`-separated components must be numeric; any subsequent
/// text (after a `-`) is the pre-release identifier and is not validated.
#[test]
fn test_version_parses_as_semver() {
    let ptr = smf_version();
    let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
    assert!(!s.is_empty(), "version must not be empty");

    // Strip optional pre-release suffix (everything after the first `-`).
    let core = s.split('-').next().unwrap_or(s);
    let parts: Vec<&str> = core.split('.').collect();
    assert!(
        parts.len() >= 3,
        "semver requires at least 3 numeric parts before any pre-release suffix, got: {s}"
    );
    for part in &parts[..3] {
        assert!(
            part.chars().all(|c| c.is_ascii_digit()),
            "semver part {part:?} is not all-numeric in version: {s}"
        );
    }
}
