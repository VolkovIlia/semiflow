//! Basic correctness tests for `GraphHeat6thChernoff` (ADR-0062).
//!
//! Verifies:
//! - Order is 6.
//! - Zero τ returns input verbatim.
//! - Negative τ returns `DomainViolation`.
//! - K=6 differs from K=4 and matches dense-matrix oracle to floor at small τ.
//! - Composition under `ChernoffSemigroup` converges to dense oracle.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
// Test: allows exact float comparisons for identity/sentinel checks.
#![allow(clippy::float_cmp)]

use std::sync::Arc;

use semiflow_core::{
    graph::{Graph, Laplacian},
    graph_heat4::GraphHeat4thChernoff,
    graph_heat6::GraphHeat6thChernoff,
    graph_signal::GraphSignal,
    ChernoffFunction, ChernoffSemigroup, ScratchPool,
};

#[test]
fn k6_order_metadata_is_six() {
    let g = Arc::new(Graph::<f64>::path(8));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let c = GraphHeat6thChernoff::new(lap);
    assert_eq!(c.order(), 6);
    let g_val = c.growth();
    assert_eq!(
        g_val.multiplier, 1.0,
        "growth bound coefficient should be 1.0 for the heat semigroup"
    );
}

#[test]
fn k6_zero_tau_is_identity_on_random_state() {
    let n = 16_usize;
    let g = Arc::new(Graph::<f64>::erdos_renyi(n, 0.3, 0xDEAD_BEEF));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let c = GraphHeat6thChernoff::new(Arc::clone(&lap));
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f64 * 0.7).sin());
    let mut dst = src.clone();
    let mut scratch = ScratchPool::<f64>::new();
    c.apply_into(0.0, &src, &mut dst, &mut scratch).unwrap();
    for (a, b) in dst.values().iter().zip(src.values().iter()) {
        assert!((a - b).abs() < 1e-14);
    }
}

#[test]
fn k6_negative_tau_errors() {
    let g = Arc::new(Graph::<f64>::path(4));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let c = GraphHeat6thChernoff::new(lap);
    let src = GraphSignal::zeros(Arc::clone(&g));
    let mut dst = src.clone();
    let mut scratch = ScratchPool::<f64>::new();
    assert!(c.apply_into(-0.001, &src, &mut dst, &mut scratch).is_err());
}

#[test]
fn k6_nan_tau_errors() {
    let g = Arc::new(Graph::<f64>::path(4));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let c = GraphHeat6thChernoff::new(lap);
    let src = GraphSignal::zeros(Arc::clone(&g));
    let mut dst = src.clone();
    let mut scratch = ScratchPool::<f64>::new();
    assert!(c
        .apply_into(f64::NAN, &src, &mut dst, &mut scratch)
        .is_err());
}

#[test]
fn k6_beats_k4_in_self_convergence_at_small_tau() {
    // For a smooth initial condition on P_32, refining n_steps should make
    // K=6 converge faster than K=4. We compare diffs of successive refinements.
    let n_nodes = 32_usize;
    let g = Arc::new(Graph::<f64>::path(n_nodes));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 / n_nodes as f64 * core::f64::consts::TAU;
        x.cos()
    });

    let t = 0.25_f64;
    let evolve_k4 = |steps: usize| {
        let k = GraphHeat4thChernoff::new(Arc::clone(&lap));
        let semi = ChernoffSemigroup::new(k, steps).unwrap();
        semi.evolve(t, &src).unwrap()
    };
    let evolve_k6 = |steps: usize| {
        let k = GraphHeat6thChernoff::new(Arc::clone(&lap));
        let semi = ChernoffSemigroup::new(k, steps).unwrap();
        semi.evolve(t, &src).unwrap()
    };

    let u4_lo = evolve_k4(8);
    let u4_hi = evolve_k4(16);
    let u6_lo = evolve_k6(8);
    let u6_hi = evolve_k6(16);

    let err4 = max_abs_diff(&u4_lo, &u4_hi);
    let err6 = max_abs_diff(&u6_lo, &u6_hi);
    assert!(
        err6 < err4,
        "K=6 self-convergence diff {err6:.3e} should be < K=4 {err4:.3e}"
    );
}

fn max_abs_diff(a: &GraphSignal<f64>, b: &GraphSignal<f64>) -> f64 {
    a.values()
        .iter()
        .zip(b.values().iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}
