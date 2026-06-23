//! Unit tests for `VarCoefMagnusGraphHeatChernoff` (extracted from the lib
//! file to keep `src/varcoef_magnus_graph.rs` ≤ 500 `LoC` per constitution
//! Override #1).
//!
//! Covers: order metadata, zero-tau identity, negative-tau error,
//! Magnus-radius-violation error, and the CRITICAL "a ≡ 1 matches standard
//! Magnus K=4" invariant that proves the operator-form composition is
//! correct.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;

use semiflow::{
    error::SemiflowError,
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    magnus_graph::MagnusGraphHeatChernoff,
    state::State,
    varcoef_magnus_graph::{VarCoefMagnusGraphHeatChernoff, WeightAtTime},
    ChernoffFunction, LaplacianAtTime, ScratchPool,
};

fn make_path_mc(n: usize) -> VarCoefMagnusGraphHeatChernoff<f64> {
    let g = Arc::new(Graph::<f64>::path(n));
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    let a_at: WeightAtTime<f64> = Box::new(move |_t| vec![1.0_f64; n]);
    VarCoefMagnusGraphHeatChernoff::new(n, lap_at, a_at, 4.0, 1.0).unwrap()
}

#[test]
fn order_is_four() {
    assert_eq!(make_path_mc(8).order(), 4);
}

#[test]
fn zero_tau_returns_src() {
    let mc = make_path_mc(8);
    let g = Arc::new(Graph::<f64>::path(8));
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    mc.apply_into(0.0, &src, &mut dst, &mut pool).unwrap();
    let mut diff = dst.clone();
    diff.axpy_into(-1.0, &src);
    assert!(diff.norm_sup() < 1e-14);
}

#[test]
fn negative_tau_errors() {
    let mc = make_path_mc(4);
    let g = Arc::new(Graph::<f64>::path(4));
    let src = GraphSignal::zeros(Arc::clone(&g));
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    let r = mc.apply_into(-0.1, &src, &mut dst, &mut pool);
    assert!(matches!(r, Err(SemiflowError::DomainViolation { .. })));
}

#[test]
fn radius_violation_errors() {
    // rho_bar=4.0, a_sup=1.0, tau just above pi/8 => 4*1*tau > pi/2
    let mc = make_path_mc(4);
    let g = Arc::new(Graph::<f64>::path(4));
    let src = GraphSignal::zeros(Arc::clone(&g));
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    let tau_bad = core::f64::consts::FRAC_PI_8 + 0.01;
    let r = mc.apply_into(tau_bad, &src, &mut dst, &mut pool);
    assert!(matches!(r, Err(SemiflowError::OutOfMagnusRadius { .. })));
}

#[test]
fn constant_a_matches_magnus_when_a_equals_one() {
    // With a(t) ≡ 1, L_a = L_G, so this kernel MUST agree with the
    // standard MagnusGraphHeatChernoff on the same Laplacian. This is the
    // canonical proof that the operator-form composition is correct.
    let n = 8_usize;
    let g = Arc::new(Graph::<f64>::path(n));
    let g_mag = Arc::clone(&g);
    let lap_at_mag: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g_mag)));
    let mc_std = MagnusGraphHeatChernoff::new(Arc::clone(&g), lap_at_mag, 4.0_f64, true).unwrap();
    let mc_vc = make_path_mc(n);

    let src = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f64 * 0.3).sin());
    let mut dst_std = src.clone();
    let mut dst_vc = src.clone();
    let mut pool = ScratchPool::<f64>::new();

    let tau = 0.05_f64;
    mc_std
        .apply_into(tau, &src, &mut dst_std, &mut pool)
        .unwrap();
    mc_vc.apply_into(tau, &src, &mut dst_vc, &mut pool).unwrap();

    let mut diff = dst_vc.clone();
    diff.axpy_into(-1.0, &dst_std);
    let d = diff.norm_sup();
    // a_at_t(t) returns Vec of 1.0; sqrt of 1.0 is exact in f64; the only
    // round-off is from the two extra diag-mul-by-1.0 passes. Tolerance
    // ~ 1e-13 for safety.
    assert!(d < 1e-13, "a≡1 mismatch with standard Magnus: {d:.3e}");
}
