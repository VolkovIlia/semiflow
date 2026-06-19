//! Smoke test for the v3.0 FFI surface (ADR-0076, Wave D).
//!
//! Verifies that:
//! 1. `smf_evolver_new_heat_1d_unit_v3` allocates and constructs an evolver.
//! 2. `smf_evolver_evolve_into_v3` produces the same evolved values as the
//!    v2.x path (`smf_state_new_heat_1d_unit` + `smf_evolve` +
//!    `smf_state_values`) — byte-identical (`memcmp == 0` gate per
//!    ADR-0076 §`G_binding_parity`).
//! 3. `smf_evolver_values_v3` and `smf_evolver_size_v3` return correct data.
//! 4. `smf_growth_v3` returns the expected unit-diffusion growth bound
//!    (`multiplier = 1.0, omega = 0.0`).
//! 5. All null-pointer guard paths return `NullPtr`.
//! 6. `smf_evolver_free_v3(NULL)` is a safe no-op.
//!
//! The test calls the `extern "C"` functions directly from Rust via raw
//! pointer manipulation — equivalent to a C caller, without a separate
//! C binary.

#![allow(unsafe_code)]
// Test: allows exact float comparisons for identity/sentinel checks.
#![allow(clippy::float_cmp)]

use semiflow_ffi::{
    smf_evolve, smf_state_free, smf_state_new_heat_1d_unit, smf_state_values,
    smf_evolver_evolve_into_v3, smf_evolver_free_v3, smf_evolver_new_heat_1d_unit_v3,
    smf_evolver_size_v3, smf_evolver_values_v3, smf_growth_v3, SemiflowStatus,
};

// ---------------------------------------------------------------------------
// Test constants (mirror ffi_caller_owned.rs to allow byte-identical comparison)
// ---------------------------------------------------------------------------

const XMIN: f64 = -1.0;
const XMAX: f64 = 1.0;
const N: usize = 64;
const T: f64 = 0.01;
const N_STEPS: usize = 10;

fn gaussian_u0() -> Vec<f64> {
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
// Test 1: happy path — v3 evolver constructs and evolves
// ---------------------------------------------------------------------------

#[test]
fn v3_evolver_constructs_and_evolves() {
    let u0 = gaussian_u0();
    let mut ev = std::ptr::null_mut();

    let rc =
        unsafe { smf_evolver_new_heat_1d_unit_v3(XMIN, XMAX, N, N_STEPS, u0.as_ptr(), N, &mut ev) };
    assert_eq!(rc, SemiflowStatus::Ok, "constructor failed: {rc:?}");
    assert!(!ev.is_null());

    let n = unsafe { smf_evolver_size_v3(ev) };
    assert_eq!(n, N);

    let mut out_v3 = vec![0.0f64; N];
    let rc = unsafe { smf_evolver_evolve_into_v3(ev, T, out_v3.as_mut_ptr(), N) };
    assert_eq!(rc, SemiflowStatus::Ok, "evolve_into_v3 failed: {rc:?}");

    // Sanity: all values finite and different from initial condition
    assert!(out_v3.iter().all(|v| v.is_finite()));
    assert!(out_v3 != u0, "evolution should change the state");

    unsafe { smf_evolver_free_v3(ev) };
}

// ---------------------------------------------------------------------------
// Test 2: byte-identical to v2 path (G_binding_parity gate)
// ---------------------------------------------------------------------------

#[test]
fn v3_byte_identical_to_v2() {
    let u0 = gaussian_u0();

    // --- v2 path ---
    let mut state_v2 = std::ptr::null_mut();
    let rc =
        unsafe { smf_state_new_heat_1d_unit(XMIN, XMAX, N, u0.as_ptr(), N, &mut state_v2) };
    assert_eq!(rc, SemiflowStatus::Ok);

    let rc = unsafe { smf_evolve(state_v2, T, N_STEPS) };
    assert_eq!(rc, SemiflowStatus::Ok);

    let mut out_v2 = vec![0.0f64; N];
    let rc = unsafe { smf_state_values(state_v2, out_v2.as_mut_ptr(), N) };
    assert_eq!(rc, SemiflowStatus::Ok);

    unsafe { smf_state_free(state_v2) };

    // --- v3 path ---
    let mut ev = std::ptr::null_mut();
    let rc =
        unsafe { smf_evolver_new_heat_1d_unit_v3(XMIN, XMAX, N, N_STEPS, u0.as_ptr(), N, &mut ev) };
    assert_eq!(rc, SemiflowStatus::Ok);

    let mut out_v3 = vec![0.0f64; N];
    let rc = unsafe { smf_evolver_evolve_into_v3(ev, T, out_v3.as_mut_ptr(), N) };
    assert_eq!(rc, SemiflowStatus::Ok);

    unsafe { smf_evolver_free_v3(ev) };

    // --- byte-identical comparison ---
    assert_eq!(
        out_v2, out_v3,
        "v3 output must be byte-identical to v2 output (G_binding_parity)"
    );
}

// ---------------------------------------------------------------------------
// Test 3: smf_evolver_values_v3 returns same data as evolve output
// ---------------------------------------------------------------------------

#[test]
fn v3_values_matches_evolve_output() {
    let u0 = gaussian_u0();
    let mut ev = std::ptr::null_mut();
    unsafe {
        smf_evolver_new_heat_1d_unit_v3(XMIN, XMAX, N, N_STEPS, u0.as_ptr(), N, &mut ev);
    }

    let mut out1 = vec![0.0f64; N];
    unsafe { smf_evolver_evolve_into_v3(ev, T, out1.as_mut_ptr(), N) };

    let mut out2 = vec![0.0f64; N];
    let rc = unsafe { smf_evolver_values_v3(ev, out2.as_mut_ptr(), N) };
    assert_eq!(rc, SemiflowStatus::Ok);

    assert_eq!(out1, out2, "values_v3 must match evolve_into output");

    unsafe { smf_evolver_free_v3(ev) };
}

// ---------------------------------------------------------------------------
// Test 4: growth bound for unit diffusion
// ---------------------------------------------------------------------------

#[test]
fn v3_growth_unit_diffusion() {
    let u0 = gaussian_u0();
    let mut ev = std::ptr::null_mut();
    unsafe {
        smf_evolver_new_heat_1d_unit_v3(XMIN, XMAX, N, N_STEPS, u0.as_ptr(), N, &mut ev);
    }

    let g = unsafe { smf_growth_v3(ev) };
    // Unit diffusion is contractive: M = 1.0, ω = 0.0
    assert!(
        (g.multiplier - 1.0).abs() < 1e-12,
        "expected multiplier ≈ 1.0, got {}",
        g.multiplier
    );
    assert!(
        g.omega.abs() < 1e-12,
        "expected omega ≈ 0.0, got {}",
        g.omega
    );

    unsafe { smf_evolver_free_v3(ev) };
}

// ---------------------------------------------------------------------------
// Test 5: null-pointer guards
// ---------------------------------------------------------------------------

#[test]
fn v3_null_guards() {
    let u0 = gaussian_u0();
    let mut ev = std::ptr::null_mut();

    // null u0
    let rc = unsafe {
        smf_evolver_new_heat_1d_unit_v3(XMIN, XMAX, N, N_STEPS, std::ptr::null(), N, &mut ev)
    };
    assert_eq!(rc, SemiflowStatus::NullPtr);

    // null out_ev
    let rc = unsafe {
        smf_evolver_new_heat_1d_unit_v3(
            XMIN,
            XMAX,
            N,
            N_STEPS,
            u0.as_ptr(),
            N,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, SemiflowStatus::NullPtr);

    // null ev for evolve
    let rc =
        unsafe { smf_evolver_evolve_into_v3(std::ptr::null_mut(), T, std::ptr::null_mut(), N) };
    assert_eq!(rc, SemiflowStatus::NullPtr);

    // null size — returns 0
    let n = unsafe { smf_evolver_size_v3(std::ptr::null()) };
    assert_eq!(n, 0);

    // null growth — returns zeros sentinel
    let g = unsafe { smf_growth_v3(std::ptr::null()) };
    assert_eq!(g.multiplier, 0.0);
    assert_eq!(g.omega, 0.0);

    // free(NULL) — safe no-op
    unsafe { smf_evolver_free_v3(std::ptr::null_mut()) };
}

// ---------------------------------------------------------------------------
// Test 6: error conditions
// ---------------------------------------------------------------------------

#[test]
fn v3_grid_mismatch_wrong_len() {
    let u0 = gaussian_u0();
    let mut ev = std::ptr::null_mut();
    let rc = unsafe {
        // u0_len != n_grid
        smf_evolver_new_heat_1d_unit_v3(XMIN, XMAX, N, N_STEPS, u0.as_ptr(), N - 1, &mut ev)
    };
    assert_ne!(rc, SemiflowStatus::Ok);
    assert!(ev.is_null());
}

#[test]
fn v3_out_of_domain_zero_chernoff() {
    let u0 = gaussian_u0();
    let mut ev = std::ptr::null_mut();
    let rc = unsafe { smf_evolver_new_heat_1d_unit_v3(XMIN, XMAX, N, 0, u0.as_ptr(), N, &mut ev) };
    // n_chernoff == 0 is a DomainViolation
    assert_ne!(rc, SemiflowStatus::Ok);
    assert!(ev.is_null());
}
