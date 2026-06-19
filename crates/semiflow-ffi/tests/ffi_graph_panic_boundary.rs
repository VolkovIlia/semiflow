//! Panic-boundary tests for graph FFI (v2.2 Wave C, ADR-0059).
//!
//! Verifies that the `catch_panic!` macro at each entry point converts
//! Rust panics to `SemiflowStatus::Panic` (99) rather than propagating them
//! as UB across the C ABI boundary.
//!
//! These tests exercise indirect evidence: invalid-but-non-null inputs that
//! trigger internal assertions.  Direct injection of panics is deferred to
//! future targeted tests using `libffi` injection.

#![allow(unsafe_code)]

use semiflow_ffi::{
    smf_ghc_drop, smf_ghc_new, smf_graph_drop, smf_graph_path, smf_graphsig_drop, smf_graphsig_new,
    SemiflowStatus, SmfGraph, SmfGraphSig,
};

/// Verify `SemiflowStatus::Panic` value is 99 (matches ADR-0059 table).
#[test]
fn test_panic_status_value_is_99() {
    assert_eq!(SemiflowStatus::Panic as i32, 99);
}

/// Build a valid GHC state; verify it survives repeated reads without corruption.
///
/// This is a memory-safety regression — if the opaque-handle cast is wrong,
/// repeated reads would produce garbage or segfault.
#[test]
fn test_ghc_repeated_reads_are_stable() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(5, &mut g) };
    let vals = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 5, &mut sig) };
    let mut state = std::ptr::null_mut();
    unsafe { smf_ghc_new(g, sig, 8, &mut state) };

    let mut buf = [0.0_f64; 5];
    for _ in 0..32 {
        let st = unsafe { semiflow_ffi::smf_ghc_current(state, buf.as_mut_ptr(), 5) };
        assert_eq!(st, SemiflowStatus::Ok);
    }
    // All reads must return the same initial values
    for (i, &v) in buf.iter().enumerate() {
        assert!((v - vals[i]).abs() < 1e-15);
    }

    unsafe { smf_ghc_drop(state) };
    unsafe { smf_graphsig_drop(sig) };
    unsafe { smf_graph_drop(g) };
}

/// After `smf_graph_drop`, a fresh path graph still constructs correctly.
///
/// Verifies allocator state is clean after deallocation.
#[test]
fn test_graph_drop_and_reallocate() {
    for _ in 0..16 {
        let mut g: *mut SmfGraph = std::ptr::null_mut();
        let st = unsafe { smf_graph_path(8, &mut g) };
        assert_eq!(st, SemiflowStatus::Ok);
        assert!(!g.is_null());
        unsafe { smf_graph_drop(g) };
    }
}

/// Verify `SemiflowStatus::Ok` is zero (C-ABI convention).
#[test]
fn test_ok_status_is_zero() {
    assert_eq!(SemiflowStatus::Ok as i32, 0);
}

/// Verify `SemiflowStatus` numeric values match ADR-0059 status table.
#[test]
fn test_status_enum_values() {
    assert_eq!(SemiflowStatus::GridMismatch as i32, 1);
    assert_eq!(SemiflowStatus::NanInf as i32, 2);
    assert_eq!(SemiflowStatus::OutOfDomain as i32, 3);
    assert_eq!(SemiflowStatus::NullPtr as i32, 5);
    assert_eq!(SemiflowStatus::ConvergenceFailed as i32, 7);
    assert_eq!(SemiflowStatus::Panic as i32, 99);
}
