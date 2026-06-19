//! Numerical evolve tests for graph FFI (v2.2 Wave C, ADR-0059).
//!
//! Compares `smf_ghc_apply_into` output against the pure-Rust
//! `ChernoffSemigroup::evolve` reference path. The sup-error must be < 1 ULP
//! (≤ 2e-16 for f64). This satisfies the 3-ULP gate from ADR-0059.

#![allow(unsafe_code)]

use std::sync::Arc;

use semiflow_core::{ChernoffSemigroup, Graph, GraphHeatChernoff, GraphSignal, Laplacian};
use semiflow_ffi::{
    smf_ghc_apply_into, smf_ghc_current, smf_ghc_drop, smf_ghc_new, smf_graph_drop, smf_graph_path,
    smf_graphsig_drop, smf_graphsig_new, SemiflowStatus, SmfGhc, SmfGraph, SmfGraphSig,
};

const N: usize = 8;
const TAU: f64 = 0.01;
const N_STEPS: u32 = 10;
/// 3 ULP gate for cross-binding identity (ADR-0059).
const ULP_3: f64 = 3.0 * 2.220_446_049_250_313e-16;

fn make_u0(n: usize) -> Vec<f64> {
    // n <= N = 8; precision loss is negligible for small index counts.
    #[allow(clippy::cast_precision_loss)]
    (0..n)
        .map(|i| (i as f64 * std::f64::consts::PI / n as f64).sin())
        .collect()
}

/// Pure-Rust reference path.
fn rust_reference(u0: &[f64], n: usize, tau: f64, n_steps: usize) -> Vec<f64> {
    let g = Arc::new(Graph::<f64>::path(n));
    let lap = Laplacian::assemble_combinatorial(&g);
    let chernoff = GraphHeatChernoff::from_owned(lap);
    let sg = ChernoffSemigroup::new(chernoff, n_steps).unwrap();
    let sig = GraphSignal::from_fn(Arc::clone(&g), |i| u0[i as usize]);
    let result = sg.evolve(tau, &sig).unwrap();
    result.values().to_vec()
}

#[test]
fn test_ghc_evolve_matches_rust_reference() {
    let n = N;
    let u0 = make_u0(n);
    let reference = rust_reference(&u0, n, TAU, N_STEPS as usize);

    // Build FFI path. n == N == 8; cast is safe.
    #[allow(clippy::cast_possible_truncation)]
    let n_u32 = n as u32;
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(n_u32, &mut g) };

    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, u0.as_ptr(), n_u32, &mut sig) };

    let mut state: *mut SmfGhc = std::ptr::null_mut();
    let st = unsafe { smf_ghc_new(g, sig, N_STEPS, &mut state) };
    assert_eq!(st, SemiflowStatus::Ok);

    let evolve_st = unsafe { smf_ghc_apply_into(state, TAU, N_STEPS) };
    assert_eq!(evolve_st, SemiflowStatus::Ok);

    let mut out = vec![0.0_f64; n];
    let read_st = unsafe { smf_ghc_current(state, out.as_mut_ptr(), n_u32) };
    assert_eq!(read_st, SemiflowStatus::Ok);

    let sup_err = out
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // 3-ULP gate from ADR-0059 (conservative; expect < 1e-15 in practice)
    assert!(
        sup_err < ULP_3,
        "sup_error={sup_err:.3e} exceeds 3-ULP gate {ULP_3:.3e}"
    );

    unsafe { smf_ghc_drop(state) };
    unsafe { smf_graphsig_drop(sig) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn test_ghc_apply_zero_tau_returns_out_of_domain() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(4, &mut g) };
    let vals = [1.0_f64; 4];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 4, &mut sig) };
    let mut state: *mut SmfGhc = std::ptr::null_mut();
    unsafe { smf_ghc_new(g, sig, 10, &mut state) };

    let st = unsafe { smf_ghc_apply_into(state, 0.0, 10) };
    assert_eq!(st, SemiflowStatus::OutOfDomain);

    unsafe { smf_ghc_drop(state) };
    unsafe { smf_graphsig_drop(sig) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn test_ghc_apply_negative_tau_returns_out_of_domain() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(4, &mut g) };
    let vals = [1.0_f64; 4];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 4, &mut sig) };
    let mut state: *mut SmfGhc = std::ptr::null_mut();
    unsafe { smf_ghc_new(g, sig, 10, &mut state) };

    let st = unsafe { smf_ghc_apply_into(state, -0.1, 10) };
    assert_eq!(st, SemiflowStatus::OutOfDomain);

    unsafe { smf_ghc_drop(state) };
    unsafe { smf_graphsig_drop(sig) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn test_ghc_apply_zero_steps_returns_out_of_domain() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(4, &mut g) };
    let vals = [1.0_f64; 4];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 4, &mut sig) };
    let mut state: *mut SmfGhc = std::ptr::null_mut();
    unsafe { smf_ghc_new(g, sig, 10, &mut state) };

    let st = unsafe { smf_ghc_apply_into(state, 0.1, 0) };
    assert_eq!(st, SemiflowStatus::OutOfDomain);

    unsafe { smf_ghc_drop(state) };
    unsafe { smf_graphsig_drop(sig) };
    unsafe { smf_graph_drop(g) };
}

#[test]
fn test_ghc_current_buf_too_small_returns_grid_mismatch() {
    let mut g: *mut SmfGraph = std::ptr::null_mut();
    unsafe { smf_graph_path(6, &mut g) };
    let vals = [1.0_f64; 6];
    let mut sig: *mut SmfGraphSig = std::ptr::null_mut();
    unsafe { smf_graphsig_new(g, vals.as_ptr(), 6, &mut sig) };
    let mut state: *mut SmfGhc = std::ptr::null_mut();
    unsafe { smf_ghc_new(g, sig, 10, &mut state) };

    let mut out = [0.0_f64; 4]; // too small
    let st = unsafe { smf_ghc_current(state, out.as_mut_ptr(), 4) };
    assert_eq!(st, SemiflowStatus::GridMismatch);

    unsafe { smf_ghc_drop(state) };
    unsafe { smf_graphsig_drop(sig) };
    unsafe { smf_graph_drop(g) };
}
