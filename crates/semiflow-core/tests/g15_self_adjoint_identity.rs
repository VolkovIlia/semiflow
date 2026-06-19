//! G15 slope gate — `AdjointChernoff::new_self_adjoint` identity (0 ULP).
//!
//! For a self-adjoint inner generator (symmetric graph Laplacian),
//! `AdjointChernoff::new_self_adjoint(inner).apply_into(τ, f)` must produce
//! a result **bit-equal** (0 ULP difference) to `inner.apply_into(τ, f)`.
//!
//! Both f64 and f32 sub-tests are required (contract §5.1).
//!
//! See wave-b-advanced-semigroups.md §5.1 and ADR-0055.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;

use semiflow_core::{
    graph_heat::GraphHeatChernoff, AdjointChernoff, ChernoffFunction, Graph, GraphSignal,
    Laplacian, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_sym_heat_f64(n: usize) -> (GraphHeatChernoff<f64>, Arc<Graph<f64>>) {
    let g = Arc::new(Graph::<f64>::path(n));
    let lap = Laplacian::assemble_combinatorial(&g);
    (GraphHeatChernoff::from_owned(lap), g)
}

fn make_sym_heat_f32(n: usize) -> (GraphHeatChernoff<f32>, Arc<Graph<f32>>) {
    let g = Arc::new(Graph::<f32>::path(n));
    let lap = Laplacian::assemble_combinatorial(&g);
    (GraphHeatChernoff::from_owned(lap), g)
}

// ---------------------------------------------------------------------------
// G15 f64 — 0 ULP for several τ values and input signals
// ---------------------------------------------------------------------------

#[test]
fn g15_self_adjoint_identity_f64() {
    let n = 32usize;
    let (inner, g) = make_sym_heat_f64(n);

    for (tau, label) in [
        (0.001_f64, "tau=0.001"),
        (0.01, "tau=0.01"),
        (0.1, "tau=0.1"),
    ] {
        for seed in [0u32, 7, 42] {
            // Deterministic pseudo-random signal.
            let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
                let x = (i as f64 * 0.31 + seed as f64 * 1.7) * core::f64::consts::PI;
                x.sin()
            });
            let mut dst_inner = src.clone();
            let mut dst_adj = src.clone();
            let mut pool = ScratchPool::<f64>::new();

            inner
                .apply_into(tau, &src, &mut dst_inner, &mut pool)
                .unwrap();

            let adj = AdjointChernoff::new_self_adjoint(make_sym_heat_f64(n).0);
            adj.apply_into(tau, &src, &mut dst_adj, &mut pool).unwrap();

            for (i, (&a, &b)) in dst_inner.values().iter().zip(dst_adj.values()).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "G15 f64 FAIL {label} seed={seed} node={i}: \
                     inner={a} vs adjoint={b} (bit mismatch)"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// G15 f32 — 0 ULP
// ---------------------------------------------------------------------------

#[test]
fn g15_self_adjoint_identity_f32() {
    let n = 32usize;
    let (inner, g) = make_sym_heat_f32(n);

    for (tau, label) in [
        (0.001_f32, "tau=0.001"),
        (0.01, "tau=0.01"),
        (0.1, "tau=0.1"),
    ] {
        for seed in [0u32, 7, 42] {
            let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
                let x = (i as f32 * 0.31_f32 + seed as f32 * 1.7_f32) * core::f32::consts::PI;
                x.sin()
            });
            let mut dst_inner = src.clone();
            let mut dst_adj = src.clone();
            let mut pool = ScratchPool::<f32>::new();

            inner
                .apply_into(tau, &src, &mut dst_inner, &mut pool)
                .unwrap();

            let adj = AdjointChernoff::new_self_adjoint(make_sym_heat_f32(n).0);
            adj.apply_into(tau, &src, &mut dst_adj, &mut pool).unwrap();

            for (i, (&a, &b)) in dst_inner.values().iter().zip(dst_adj.values()).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "G15 f32 FAIL {label} seed={seed} node={i}: \
                     inner={a} vs adjoint={b} (bit mismatch)"
                );
            }
        }
    }
}
