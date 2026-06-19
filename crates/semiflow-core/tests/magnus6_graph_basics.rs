//! Integration tests for `MagnusGraphHeat6thChernoff` — smoke + GL6 constants.
//!
//! Covers: `order()`, `growth()`, GL6 abscissae values, constructor validation,
//! and zero-τ preservation.
//!
//! See contract wave-b-advanced-semigroups.md §2 and ADR-0056.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;

use semiflow_core::{
    chernoff::ChernoffFunction,
    magnus6_graph::{MagnusGraphHeat6thChernoff, GL6_B1, GL6_B2, GL6_B3, GL6_C1, GL6_C2, GL6_C3},
    magnus_graph::LaplacianAtTime,
    Graph, GraphSignal, Laplacian, ScratchPool, State,
};

// ---------------------------------------------------------------------------
// Helper: build a time-dep path graph operator
// ---------------------------------------------------------------------------

fn make_magnus6(n: usize, rho_bar: f64) -> MagnusGraphHeat6thChernoff<f64> {
    let g = Arc::new(Graph::<f64>::path(n));
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    MagnusGraphHeat6thChernoff::new(g, lap_at, rho_bar, true).unwrap()
}

// ---------------------------------------------------------------------------
// order() and growth() — contract §2
// ---------------------------------------------------------------------------

#[test]
fn magnus6_order_is_six() {
    let mc = make_magnus6(8, 4.0);
    assert_eq!(
        mc.order(),
        6,
        "MagnusGraphHeat6thChernoff must report order 6"
    );
}

#[test]
fn magnus6_growth_reflects_rho_bar() {
    let rho_bar = 3.5_f64;
    let mc = make_magnus6(8, rho_bar);
    let g = mc.growth();
    let (m, w) = (g.multiplier, g.omega);
    assert!((m - 1.0).abs() < 1e-14, "growth M must be 1.0, got {m}");
    // w == rho_bar_max as stored.
    assert!(
        (w - rho_bar).abs() < 1e-14,
        "growth w must equal rho_bar, got {w}"
    );
}

// ---------------------------------------------------------------------------
// GL6 abscissae and weights — Gauss-Legendre 3-point on [0,1]
// (NORMATIVE — verified symbolically in T14N; here we check the constants
// at runtime to catch any accidental alteration)
// ---------------------------------------------------------------------------

#[test]
fn gl6_abscissae_match_gauss_legendre() {
    // c1 = (5 - sqrt(15)) / 10
    let c1_expected = (5.0 - 15.0_f64.sqrt()) / 10.0;
    // c2 = 0.5
    let c2_expected = 0.5_f64;
    // c3 = (5 + sqrt(15)) / 10
    let c3_expected = (5.0 + 15.0_f64.sqrt()) / 10.0;

    assert!(
        (GL6_C1 - c1_expected).abs() < 1e-14,
        "GL6_C1 = {GL6_C1}, expected {c1_expected}"
    );
    assert!(
        (GL6_C2 - c2_expected).abs() < 1e-14,
        "GL6_C2 = {GL6_C2}, expected {c2_expected}"
    );
    assert!(
        (GL6_C3 - c3_expected).abs() < 1e-14,
        "GL6_C3 = {GL6_C3}, expected {c3_expected}"
    );
}

#[test]
fn gl6_weights_sum_to_one() {
    let sum = GL6_B1 + GL6_B2 + GL6_B3;
    assert!(
        (sum - 1.0).abs() < 1e-14,
        "GL6 weights must sum to 1.0, got {sum}"
    );
}

#[test]
fn gl6_weights_match_five_over_eighteen() {
    // b1 = b3 = 5/18, b2 = 8/18
    assert!(
        (GL6_B1 - 5.0 / 18.0).abs() < 1e-14,
        "GL6_B1 wrong: {GL6_B1}"
    );
    assert!(
        (GL6_B2 - 8.0 / 18.0).abs() < 1e-14,
        "GL6_B2 wrong: {GL6_B2}"
    );
    assert!(
        (GL6_B3 - 5.0 / 18.0).abs() < 1e-14,
        "GL6_B3 wrong: {GL6_B3}"
    );
}

// ---------------------------------------------------------------------------
// Zero-τ preserves input
// ---------------------------------------------------------------------------

#[test]
fn magnus6_zero_tau_preserves_src() {
    let n = 8usize;
    let mc = make_magnus6(n, 4.0);
    let g = Arc::new(Graph::<f64>::path(n));
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    mc.apply_into(0.0, &src, &mut dst, &mut pool).unwrap();
    let mut diff = dst.clone();
    diff.axpy_into(-1.0, &src);
    assert!(
        diff.norm_sup() < 1e-14,
        "zero-tau must preserve src, diff = {}",
        diff.norm_sup()
    );
}

// ---------------------------------------------------------------------------
// Constructor validation: rejects non-positive rho_bar and empty graph
// ---------------------------------------------------------------------------

#[test]
fn magnus6_constructor_rejects_zero_rho() {
    let g = Arc::new(Graph::<f64>::path(4));
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    assert!(MagnusGraphHeat6thChernoff::new(g, lap_at, 0.0, true).is_err());
}

#[test]
fn magnus6_constructor_rejects_negative_rho() {
    let g = Arc::new(Graph::<f64>::path(4));
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    assert!(MagnusGraphHeat6thChernoff::new(g, lap_at, -1.0, true).is_err());
}

// ---------------------------------------------------------------------------
// Convergence radius check: rejects tau*rho_bar >= pi/2
// ---------------------------------------------------------------------------

#[test]
fn magnus6_radius_check_rejects_oversized_tau() {
    let n = 4usize;
    let mc = make_magnus6(n, 4.0);
    let g = Arc::new(Graph::<f64>::path(n));
    let src = GraphSignal::zeros(Arc::clone(&g));
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    // rho_bar=4.0; tau * 4.0 must be < pi/2 → tau_over triggers violation.
    let tau_over = core::f64::consts::FRAC_PI_2 / 4.0 + 0.01;
    let result = mc.apply_into(tau_over, &src, &mut dst, &mut pool);
    assert!(
        result.is_err(),
        "expected error for tau*rho_bar >= pi/2, but got Ok"
    );
}
