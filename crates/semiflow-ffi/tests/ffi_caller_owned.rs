//! Integration test: `smf_evolve_inplace` (Wave 5, ADR-0045 §2).
//!
//! Verifies that the new caller-owned buffer path produces byte-identical
//! results to the v1.0.0 `smf_evolve` + `smf_state_values` round-trip,
//! and that all documented error conditions return the correct status codes.

#![allow(unsafe_code)]

use semiflow_ffi::{
    smf_evolve, smf_evolve_inplace, smf_state_free, smf_state_new_heat_1d_unit,
    smf_state_values, SemiflowStatus,
};

// ---------------------------------------------------------------------------
// Test constants
// ---------------------------------------------------------------------------

const XMIN: f64 = -1.0;
const XMAX: f64 = 1.0;
const N: usize = 64;
const TAU: f64 = 0.01;
const N_STEPS: usize = 10;

/// Gaussian initial condition: `exp(-25 * x²)`.
fn gaussian_init() -> Vec<f64> {
    #[allow(clippy::cast_precision_loss)]
    let dx = (XMAX - XMIN) / (N as f64 - 1.0);
    (0..N)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let x = XMIN + i as f64 * dx;
            (-25.0 * x * x).exp()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main correctness test: caller-owned must match v1.0.0 evolve byte-for-byte
// ---------------------------------------------------------------------------

#[test]
fn caller_owned_matches_v1_evolve() {
    let u0 = gaussian_init();
    let mut buf_a = u0.clone();
    let mut buf_b = u0.clone();

    // Path A: v1.0.0 smf_evolve + smf_state_values
    unsafe {
        let mut state_a = std::ptr::null_mut();
        let rc = smf_state_new_heat_1d_unit(
            XMIN,
            XMAX,
            N,
            buf_a.as_ptr(),
            buf_a.len(),
            &mut state_a,
        );
        assert_eq!(rc, SemiflowStatus::Ok, "state_new Path A");
        let rc = smf_evolve(state_a, TAU, N_STEPS);
        assert_eq!(rc, SemiflowStatus::Ok, "evolve Path A");
        let rc = smf_state_values(state_a, buf_a.as_mut_ptr(), N);
        assert_eq!(rc, SemiflowStatus::Ok, "state_values Path A");
        smf_state_free(state_a);
    }

    // Path B: Wave 5 smf_evolve_inplace
    unsafe {
        let mut state_b = std::ptr::null_mut();
        let rc = smf_state_new_heat_1d_unit(
            XMIN,
            XMAX,
            N,
            buf_b.as_ptr(),
            buf_b.len(),
            &mut state_b,
        );
        assert_eq!(rc, SemiflowStatus::Ok, "state_new Path B");
        let rc = smf_evolve_inplace(state_b, buf_b.as_mut_ptr(), N, TAU, N_STEPS);
        assert_eq!(rc, SemiflowStatus::Ok, "evolve_inplace Path B");
        smf_state_free(state_b);
    }

    // Byte-for-byte equality (ADR-0045 §2 bit-equality requirement).
    assert_eq!(
        buf_a, buf_b,
        "caller-owned must match v1.0.0 evolve byte-for-byte"
    );
}

// ---------------------------------------------------------------------------
// Error condition tests
// ---------------------------------------------------------------------------

#[test]
fn null_state_returns_null_ptr() {
    let mut buf = gaussian_init();
    let rc =
        unsafe { smf_evolve_inplace(std::ptr::null_mut(), buf.as_mut_ptr(), N, TAU, N_STEPS) };
    assert_eq!(rc, SemiflowStatus::NullPtr);
}

#[test]
fn null_buf_returns_null_ptr() {
    let u0 = gaussian_init();
    unsafe {
        let mut state = std::ptr::null_mut();
        let rc = smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), N, &mut state);
        assert_eq!(rc, SemiflowStatus::Ok);
        let rc = smf_evolve_inplace(state, std::ptr::null_mut(), N, TAU, N_STEPS);
        assert_eq!(rc, SemiflowStatus::NullPtr);
        smf_state_free(state);
    }
}

#[test]
fn wrong_buf_len_returns_grid_mismatch() {
    let u0 = gaussian_init();
    let mut buf = u0.clone();
    unsafe {
        let mut state = std::ptr::null_mut();
        let rc = smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), N, &mut state);
        assert_eq!(rc, SemiflowStatus::Ok);
        // Pass buf_len = N - 1 (wrong)
        let rc = smf_evolve_inplace(state, buf.as_mut_ptr(), N - 1, TAU, N_STEPS);
        assert_eq!(rc, SemiflowStatus::GridMismatch);
        smf_state_free(state);
    }
}

#[test]
fn negative_tau_returns_out_of_domain() {
    let u0 = gaussian_init();
    let mut buf = u0.clone();
    unsafe {
        let mut state = std::ptr::null_mut();
        let rc = smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), N, &mut state);
        assert_eq!(rc, SemiflowStatus::Ok);
        let rc = smf_evolve_inplace(state, buf.as_mut_ptr(), N, -0.01, N_STEPS);
        assert_eq!(rc, SemiflowStatus::OutOfDomain);
        smf_state_free(state);
    }
}

#[test]
fn nan_tau_returns_out_of_domain() {
    let u0 = gaussian_init();
    let mut buf = u0.clone();
    unsafe {
        let mut state = std::ptr::null_mut();
        let rc = smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), N, &mut state);
        assert_eq!(rc, SemiflowStatus::Ok);
        let rc = smf_evolve_inplace(state, buf.as_mut_ptr(), N, f64::NAN, N_STEPS);
        assert_eq!(rc, SemiflowStatus::OutOfDomain);
        smf_state_free(state);
    }
}

#[test]
fn zero_n_steps_returns_out_of_domain() {
    let u0 = gaussian_init();
    let mut buf = u0.clone();
    unsafe {
        let mut state = std::ptr::null_mut();
        let rc = smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), N, &mut state);
        assert_eq!(rc, SemiflowStatus::Ok);
        let rc = smf_evolve_inplace(state, buf.as_mut_ptr(), N, TAU, 0);
        assert_eq!(rc, SemiflowStatus::OutOfDomain);
        smf_state_free(state);
    }
}
