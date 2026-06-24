//! Smoke tests for the S³ carrier FFI bindings (ADR-0171, v9.2.0).
//!
//! Mirrors `crates/semiflow-wasm/tests/s3_smoke.rs`:
//! construct → evolve → observable round-trips for all three families.
//!
//! Tests:
//! - `tt_state_roundtrip`        — build `SmfTtState`, check `ndim` / `storage_size`.
//! - `tt_evolver_evolve`         — evolve a rank-1 `TtState` for t=0; state unchanged.
//! - `tt_coupled_topology_none`  — `couplingTag=0` constructs and evolves.
//! - `measure_state_roundtrip`   — build `SmfMeasureState`, check `n_diracs` / TV.
//! - `gridless_evolve_dirac`     — evolve single Dirac for t=0; TV preserved.

#![allow(unsafe_code)]
#![allow(clippy::float_cmp)]

use semiflow_ffi::{
    smf_gridless_evolve, smf_gridless_free, smf_gridless_new, smf_measurestate_free,
    smf_measurestate_n_diracs, smf_measurestate_new, smf_measurestate_total_variation,
    smf_tt_coupled_evolve, smf_tt_coupled_free, smf_tt_coupled_new, smf_tt_evolver_evolve,
    smf_tt_evolver_free, smf_tt_evolver_new, smf_ttstate_free, smf_ttstate_ndim,
    smf_ttstate_new_separable, smf_ttstate_storage_size, SemiflowStatus,
};

// ---------------------------------------------------------------------------
// TtState round-trip
// ---------------------------------------------------------------------------

#[test]
fn tt_state_roundtrip() {
    // Two axes: [1.0, 2.0, 3.0] and [4.0, 5.0]
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let offsets: Vec<usize> = vec![0, 3, 5];
    let mut state = std::ptr::null_mut();

    let rc = unsafe { smf_ttstate_new_separable(data.as_ptr(), offsets.as_ptr(), 2, &mut state) };
    assert_eq!(
        rc,
        SemiflowStatus::Ok,
        "smf_ttstate_new_separable failed: {rc:?}"
    );
    assert!(!state.is_null());

    let ndim = unsafe { smf_ttstate_ndim(state) };
    assert_eq!(ndim, 2);

    let sz = unsafe { smf_ttstate_storage_size(state) };
    assert!(sz > 0, "storage_size must be positive");

    unsafe { smf_ttstate_free(state) };
}

// ---------------------------------------------------------------------------
// TtEvolver zero-time evolve
// ---------------------------------------------------------------------------

#[test]
fn tt_evolver_evolve() {
    let data: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
    let offsets: Vec<usize> = vec![0, 2, 4];
    let mut state = std::ptr::null_mut();
    let rc = unsafe { smf_ttstate_new_separable(data.as_ptr(), offsets.as_ptr(), 2, &mut state) };
    assert_eq!(rc, SemiflowStatus::Ok);

    let a = [0.5f64, 0.5];
    let b = [0.0f64, 0.0];
    let dom_min = [-1.0f64, -1.0];
    let dom_max = [1.0f64, 1.0];
    let mut ev = std::ptr::null_mut();
    let rc = unsafe {
        smf_tt_evolver_new(
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            dom_min.as_ptr(),
            dom_max.as_ptr(),
            2,
            1e-8,
            &mut ev,
        )
    };
    assert_eq!(rc, SemiflowStatus::Ok, "smf_tt_evolver_new failed: {rc:?}");

    let rc = unsafe { smf_tt_evolver_evolve(ev, state, 0.0, 1) };
    assert_eq!(
        rc,
        SemiflowStatus::Ok,
        "smf_tt_evolver_evolve failed: {rc:?}"
    );

    let ndim = unsafe { smf_ttstate_ndim(state) };
    assert_eq!(ndim, 2);

    unsafe { smf_tt_evolver_free(ev) };
    unsafe { smf_ttstate_free(state) };
}

// ---------------------------------------------------------------------------
// TtCoupledEvolver coupling_tag=0 (None)
// ---------------------------------------------------------------------------

#[test]
fn tt_coupled_topology_none() {
    let data: Vec<f64> = vec![1.0, 0.0, 1.0, 0.0];
    let offsets: Vec<usize> = vec![0, 2, 4];
    let mut state = std::ptr::null_mut();
    let rc = unsafe { smf_ttstate_new_separable(data.as_ptr(), offsets.as_ptr(), 2, &mut state) };
    assert_eq!(rc, SemiflowStatus::Ok);

    let a = [0.5f64, 0.5];
    let b = [0.0f64, 0.0];
    let dom_min = [-1.0f64, -1.0];
    let dom_max = [1.0f64, 1.0];
    let mut ev = std::ptr::null_mut();
    // coupling_tag=0 (None), tridiag_rho ignored, empty pairs
    let rc = unsafe {
        smf_tt_coupled_new(
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            0,
            0.0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            dom_min.as_ptr(),
            dom_max.as_ptr(),
            2,
            1e-8,
            &mut ev,
        )
    };
    assert_eq!(rc, SemiflowStatus::Ok, "smf_tt_coupled_new failed: {rc:?}");

    let rc = unsafe { smf_tt_coupled_evolve(ev, state, 0.0, 1) };
    assert_eq!(
        rc,
        SemiflowStatus::Ok,
        "smf_tt_coupled_evolve failed: {rc:?}"
    );

    unsafe { smf_tt_coupled_free(ev) };
    unsafe { smf_ttstate_free(state) };
}

// ---------------------------------------------------------------------------
// MeasureState round-trip
// ---------------------------------------------------------------------------

#[test]
fn measure_state_roundtrip() {
    let pos: Vec<f64> = vec![0.0, 1.0, -1.0];
    let wts: Vec<f64> = vec![0.5, 0.25, 0.25];
    let mut state = std::ptr::null_mut();

    let rc = unsafe { smf_measurestate_new(pos.as_ptr(), wts.as_ptr(), 3, 1, &mut state) };
    assert_eq!(
        rc,
        SemiflowStatus::Ok,
        "smf_measurestate_new failed: {rc:?}"
    );

    let n = unsafe { smf_measurestate_n_diracs(state) };
    assert_eq!(n, 3);

    let mut tv = 0.0f64;
    let rc = unsafe { smf_measurestate_total_variation(state, &mut tv) };
    assert_eq!(rc, SemiflowStatus::Ok);
    assert!((tv - 1.0).abs() < 1e-10, "TV={tv}");

    unsafe { smf_measurestate_free(state) };
}

// ---------------------------------------------------------------------------
// GridlessEvolver zero-time Dirac
// ---------------------------------------------------------------------------

#[test]
fn gridless_evolve_dirac() {
    let pos: Vec<f64> = vec![0.0];
    let wts: Vec<f64> = vec![1.0];
    let mut state = std::ptr::null_mut();
    let rc = unsafe { smf_measurestate_new(pos.as_ptr(), wts.as_ptr(), 1, 1, &mut state) };
    assert_eq!(rc, SemiflowStatus::Ok);

    let a = [0.5f64];
    let b = [0.0f64];
    let mut ev = std::ptr::null_mut();
    // reduction_tag=0 (WeightedVoronoi), voronoi_cap=16
    let rc = unsafe { smf_gridless_new(a.as_ptr(), b.as_ptr(), 0.0, 1, 0, 16, &mut ev) };
    assert_eq!(rc, SemiflowStatus::Ok, "smf_gridless_new failed: {rc:?}");

    let rc = unsafe { smf_gridless_evolve(ev, 0.0, 1, state) };
    assert_eq!(rc, SemiflowStatus::Ok, "smf_gridless_evolve failed: {rc:?}");

    let mut tv = 0.0f64;
    let rc = unsafe { smf_measurestate_total_variation(state, &mut tv) };
    assert_eq!(rc, SemiflowStatus::Ok);
    assert!((tv - 1.0).abs() < 1e-6, "TV after evolve={tv}");

    unsafe { smf_gridless_free(ev) };
    unsafe { smf_measurestate_free(state) };
}
