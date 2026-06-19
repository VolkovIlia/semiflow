//! Null-pointer safety tests for graph FFI (v2.2 Wave C, ADR-0059).
//!
//! Every entry point must return `NullPtr` (5) when given a null argument,
//! and must NOT segfault or panic.

#![allow(unsafe_code)]

use semiflow_ffi::{
    smf_ghc_apply_into, smf_ghc_current, smf_ghc_new, smf_graph_path, smf_graphsig_new,
    smf_graphsig_values, smf_mghc_apply_into, smf_mghc_current, smf_mghc_new, SemiflowStatus,
    SmfGhc, SmfGraph, SmfGraphSig, SmfMghc,
};

// ---------------------------------------------------------------------------
// smf_ghc_new null variants
// ---------------------------------------------------------------------------

#[test]
fn test_ghc_new_null_sig_returns_null_ptr() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(3, &mut g) };
    let mut state: *mut SmfGhc = std::ptr::null_mut();
    let st = unsafe { smf_ghc_new(g, std::ptr::null(), 10, &mut state) };
    assert_eq!(st, SemiflowStatus::NullPtr);
    unsafe { semiflow_ffi::smf_graph_drop(g) };
}

#[test]
fn test_ghc_new_null_out_returns_null_ptr() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(3, &mut g) };
    let vals = [1.0_f64; 3];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 3, &mut sig) };
    let st = unsafe { smf_ghc_new(g, sig, 10, std::ptr::null_mut()) };
    assert_eq!(st, SemiflowStatus::NullPtr);
    unsafe { semiflow_ffi::smf_graphsig_drop(sig) };
    unsafe { semiflow_ffi::smf_graph_drop(g) };
}

// ---------------------------------------------------------------------------
// smf_ghc_apply_into null variant
// ---------------------------------------------------------------------------

#[test]
fn test_ghc_apply_null_state_returns_null_ptr() {
    let st = unsafe { smf_ghc_apply_into(std::ptr::null_mut(), 0.1, 10) };
    assert_eq!(st, SemiflowStatus::NullPtr);
}

// ---------------------------------------------------------------------------
// smf_ghc_current null variants
// ---------------------------------------------------------------------------

#[test]
fn test_ghc_current_null_state_returns_null_ptr() {
    let mut buf = [0.0_f64; 4];
    let st = unsafe { smf_ghc_current(std::ptr::null(), buf.as_mut_ptr(), 4) };
    assert_eq!(st, SemiflowStatus::NullPtr);
}

#[test]
fn test_ghc_current_null_buf_returns_null_ptr() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(3, &mut g) };
    let vals = [1.0_f64; 3];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 3, &mut sig) };
    let mut state: *mut SmfGhc = std::ptr::null_mut();
    unsafe { smf_ghc_new(g, sig, 5, &mut state) };
    let st = unsafe { smf_ghc_current(state, std::ptr::null_mut(), 3) };
    assert_eq!(st, SemiflowStatus::NullPtr);
    unsafe { semiflow_ffi::smf_ghc_drop(state) };
    unsafe { semiflow_ffi::smf_graphsig_drop(sig) };
    unsafe { semiflow_ffi::smf_graph_drop(g) };
}

// ---------------------------------------------------------------------------
// smf_graphsig_* null variants
// ---------------------------------------------------------------------------

#[test]
fn test_graphsig_new_null_graph_returns_null_ptr() {
    let vals = [1.0_f64; 3];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    let st = unsafe { smf_graphsig_new(std::ptr::null(), vals.as_ptr(), 3, &mut sig) };
    assert_eq!(st, SemiflowStatus::NullPtr);
}

#[test]
fn test_graphsig_new_null_vals_returns_null_ptr() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(3, &mut g) };
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    let st = unsafe { smf_graphsig_new(g, std::ptr::null(), 3, &mut sig) };
    assert_eq!(st, SemiflowStatus::NullPtr);
    unsafe { semiflow_ffi::smf_graph_drop(g) };
}

#[test]
fn test_graphsig_values_null_sig_returns_null_ptr() {
    let mut buf = [0.0_f64; 4];
    let st = unsafe { smf_graphsig_values(std::ptr::null(), buf.as_mut_ptr(), 4) };
    assert_eq!(st, SemiflowStatus::NullPtr);
}

#[test]
fn test_graphsig_values_null_buf_returns_null_ptr() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(3, &mut g) };
    let vals = [1.0_f64; 3];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 3, &mut sig) };
    let st = unsafe { smf_graphsig_values(sig, std::ptr::null_mut(), 3) };
    assert_eq!(st, SemiflowStatus::NullPtr);
    unsafe { semiflow_ffi::smf_graphsig_drop(sig) };
    unsafe { semiflow_ffi::smf_graph_drop(g) };
}

// ---------------------------------------------------------------------------
// smf_mghc_new null variants
// ---------------------------------------------------------------------------

unsafe extern "C" fn dummy_cb(_t: f64, _user_data: *mut (), out_graph: *mut *mut SmfGraph) -> i32 {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    semiflow_ffi::smf_graph_path(3, &mut g);
    unsafe { *out_graph = g };
    0
}

#[test]
fn test_mghc_new_null_graph_returns_null_ptr() {
    let mut out: *mut SmfMghc = std::ptr::null_mut();
    let vals = [0.5_f64; 3];
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(3, &mut g) };
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 3, &mut sig) };

    let st = unsafe {
        smf_mghc_new(
            std::ptr::null(),
            sig,
            Some(dummy_cb),
            std::ptr::null_mut(),
            10.0,
            0,
            &mut out,
        )
    };
    assert_eq!(st, SemiflowStatus::NullPtr);
    unsafe { semiflow_ffi::smf_graphsig_drop(sig) };
    unsafe { semiflow_ffi::smf_graph_drop(g) };
}

#[test]
fn test_mghc_new_null_callback_returns_null_ptr() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(3, &mut g) };
    let vals = [0.5_f64; 3];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 3, &mut sig) };

    let mut out: *mut SmfMghc = std::ptr::null_mut();
    let st = unsafe { smf_mghc_new(g, sig, None, std::ptr::null_mut(), 10.0, 0, &mut out) };
    assert_eq!(st, SemiflowStatus::NullPtr);
    unsafe { semiflow_ffi::smf_graphsig_drop(sig) };
    unsafe { semiflow_ffi::smf_graph_drop(g) };
}

#[test]
fn test_mghc_apply_null_returns_null_ptr() {
    let st = unsafe { smf_mghc_apply_into(std::ptr::null_mut(), 0.1, 5) };
    assert_eq!(st, SemiflowStatus::NullPtr);
}

#[test]
fn test_mghc_current_null_state_returns_null_ptr() {
    let mut buf = [0.0_f64; 4];
    let st = unsafe { smf_mghc_current(std::ptr::null(), buf.as_mut_ptr(), 4) };
    assert_eq!(st, SemiflowStatus::NullPtr);
}
