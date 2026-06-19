//! Smoke tests for graph FFI entry points (v2.2 Wave C, ADR-0059).
//!
//! Verifies null-safety, basic constructors, and simple round-trip through
//! `smf_ghc_*` for a path graph. Does NOT exercise time evolution numerics.

#![allow(unsafe_code)]

use semiflow_ffi::{
    smf_ghc_current, smf_ghc_drop, smf_ghc_new, smf_graph_drop, smf_graph_n_nodes, smf_graph_path,
    smf_graphsig_drop, smf_graphsig_len, smf_graphsig_new, smf_graphsig_values, SemiflowStatus,
    SmfGhc, SmfGraph, SmfGraphSig,
};

// ---------------------------------------------------------------------------
// Graph constructors
// ---------------------------------------------------------------------------

#[test]
fn test_graph_path_creates_handle() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    let status = unsafe { smf_graph_path(8, &mut g) };
    assert_eq!(status, SemiflowStatus::Ok);
    assert!(!g.is_null());
    let n = unsafe { smf_graph_n_nodes(g) };
    assert_eq!(n, 8);
    unsafe { smf_graph_drop(g) };
}

#[test]
fn test_graph_path_null_out_returns_null_ptr() {
    let status = unsafe { smf_graph_path(8, std::ptr::null_mut()) };
    assert_eq!(status, SemiflowStatus::NullPtr);
}

#[test]
fn test_graph_path_zero_nodes_returns_out_of_domain() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    let status = unsafe { smf_graph_path(0, &mut g) };
    assert_eq!(status, SemiflowStatus::OutOfDomain);
    assert!(g.is_null());
}

#[test]
fn test_graph_n_nodes_null_returns_zero() {
    let n = unsafe { smf_graph_n_nodes(std::ptr::null()) };
    assert_eq!(n, 0);
}

#[test]
fn test_graph_drop_null_is_noop() {
    unsafe { smf_graph_drop(std::ptr::null_mut()) };
    // must not crash
}

// ---------------------------------------------------------------------------
// GraphSignal
// ---------------------------------------------------------------------------

#[test]
fn test_graphsig_new_and_values() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(4, &mut g) };

    let vals = [1.0_f64, 2.0, 3.0, 4.0];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    let status = unsafe { smf_graphsig_new(g, vals.as_ptr(), 4, &mut sig) };
    assert_eq!(status, SemiflowStatus::Ok);
    assert!(!sig.is_null());
    assert_eq!(unsafe { smf_graphsig_len(sig) }, 4);

    let mut out = [0.0_f64; 4];
    let s2 = unsafe { smf_graphsig_values(sig, out.as_mut_ptr(), 4) };
    assert_eq!(s2, SemiflowStatus::Ok);
    // Signal was constructed from exact `vals` — bit-identity is expected.
    #[allow(clippy::float_cmp)]
    for (a, b) in out.iter().zip(vals.iter()) {
        assert_eq!(a, b);
    }

    unsafe { smf_graphsig_drop(sig) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn test_graphsig_size_mismatch_returns_grid_mismatch() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(4, &mut g) };
    let vals = [1.0_f64, 2.0, 3.0];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    let status = unsafe { smf_graphsig_new(g, vals.as_ptr(), 3, &mut sig) };
    assert_eq!(status, SemiflowStatus::GridMismatch);
    assert!(sig.is_null());
    unsafe { smf_graph_drop(g) };
}

#[test]
fn test_graphsig_nan_input_returns_nan_inf() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(3, &mut g) };
    let vals = [1.0_f64, f64::NAN, 0.0];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    let status = unsafe { smf_graphsig_new(g, vals.as_ptr(), 3, &mut sig) };
    assert_eq!(status, SemiflowStatus::NanInf);
    unsafe { smf_graph_drop(g) };
}

// ---------------------------------------------------------------------------
// GHC constructors
// ---------------------------------------------------------------------------

#[test]
fn test_ghc_new_creates_handle() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(5, &mut g) };
    let vals = [1.0_f64; 5];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 5, &mut sig) };

    let mut state: *mut SmfGhc = std::ptr::null_mut();
    let status = unsafe { smf_ghc_new(g, sig, 10, &mut state) };
    assert_eq!(status, SemiflowStatus::Ok);
    assert!(!state.is_null());

    // Verify initial values read back
    let mut out = [0.0_f64; 5];
    let s2 = unsafe { smf_ghc_current(state, out.as_mut_ptr(), 5) };
    assert_eq!(s2, SemiflowStatus::Ok);
    for &v in &out {
        assert!((v - 1.0).abs() < 1e-15);
    }

    unsafe { smf_ghc_drop(state) };
    unsafe { smf_graphsig_drop(sig) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn test_ghc_new_null_graph_returns_null_ptr() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(3, &mut g) };
    let vals = [1.0_f64; 3];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 3, &mut sig) };

    let mut state: *mut SmfGhc = std::ptr::null_mut();
    let status = unsafe { smf_ghc_new(std::ptr::null(), sig, 10, &mut state) };
    assert_eq!(status, SemiflowStatus::NullPtr);

    unsafe { smf_graphsig_drop(sig) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn test_ghc_drop_null_is_noop() {
    unsafe { smf_ghc_drop(std::ptr::null_mut()) };
}
