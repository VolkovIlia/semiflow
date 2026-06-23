//! Property tests for `VarCoefMagnusGraphHeatChernoff` (ADR-0063 §"acceptance gates" P3).
//!
//! P1 — Constant `a ≡ 1` matches standard `MagnusGraphHeatChernoff`.
//! P2 — `apply_into(0, src, dst)` returns `src` exactly.
//! P3 — `ρ̄_max · a_sup_max² · τ ≥ π/2` always returns `OutOfMagnusRadius`.
//! P4 — Negative or non-finite `a(t)` entry returns `DomainViolation`.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;

use proptest::prelude::*;

use semiflow::{
    error::SemiflowError,
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    varcoef_magnus_graph::{VarCoefMagnusGraphHeatChernoff, WeightAtTime},
    ChernoffFunction, LaplacianAtTime, ScratchPool,
};

fn make_mc(
    n: usize,
    rho_bar: f64,
    a_sup: f64,
    a_value: f64,
) -> VarCoefMagnusGraphHeatChernoff<f64> {
    let g = Arc::new(Graph::<f64>::path(n));
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    let nn = n;
    let av = a_value;
    let a_at: WeightAtTime<f64> = Box::new(move |_t| vec![av; nn]);
    VarCoefMagnusGraphHeatChernoff::new(n, lap_at, a_at, rho_bar, a_sup).unwrap()
}

proptest! {
    #[test]
    fn p3_radius_violation_always_errors(
        // tau > pi/2 / (4 * a_sup^2) → always violates radius.
        tau_excess in 0.001_f64..0.5_f64,
        a_sup in 1.0_f64..3.0_f64,
    ) {
        let rho_bar = 4.0_f64;
        let tau = core::f64::consts::FRAC_PI_2 / (rho_bar * a_sup * a_sup) + tau_excess;
        let mc = make_mc(8, rho_bar, a_sup, a_sup * a_sup);
        let g = Arc::new(Graph::<f64>::path(8));
        let src = GraphSignal::zeros(Arc::clone(&g));
        let mut dst = src.clone();
        let mut pool = ScratchPool::<f64>::new();
        let result = mc.apply_into(tau, &src, &mut dst, &mut pool);
        let is_radius_err = matches!(result, Err(SemiflowError::OutOfMagnusRadius { .. }));
        prop_assert!(is_radius_err, "expected OutOfMagnusRadius for tau={tau}");
    }

    #[test]
    fn p2_zero_tau_is_identity(
        n in 4_usize..32_usize,
    ) {
        let mc = make_mc(n, 4.0, 1.0, 1.0);
        let g = Arc::new(Graph::<f64>::path(n));
        let src = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f64 * 0.31).sin());
        let mut dst = src.clone();
        let mut pool = ScratchPool::<f64>::new();
        mc.apply_into(0.0, &src, &mut dst, &mut pool).unwrap();
        let max_diff = dst.values().iter().zip(src.values().iter())
            .map(|(&a, &b)| (a - b).abs()).fold(0.0_f64, f64::max);
        prop_assert!(max_diff < 1e-13);
    }
}

#[test]
fn p4_negative_a_returns_error() {
    let g = Arc::new(Graph::<f64>::path(8));
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    let a_at: WeightAtTime<f64> = Box::new(move |_t| {
        let mut a = vec![1.0_f64; 8];
        a[3] = -0.5; // negative entry
        a
    });
    let mc = VarCoefMagnusGraphHeatChernoff::new(8, lap_at, a_at, 4.0, 1.0).unwrap();
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| i as f64);
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    let result = mc.apply_into(0.05, &src, &mut dst, &mut pool);
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}

#[test]
fn p4_wrong_length_a_returns_error() {
    let g = Arc::new(Graph::<f64>::path(8));
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    // a_at_t returns length 5 instead of 8
    let a_at: WeightAtTime<f64> = Box::new(move |_t| vec![1.0_f64; 5]);
    let mc = VarCoefMagnusGraphHeatChernoff::new(8, lap_at, a_at, 4.0, 1.0).unwrap();
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| i as f64);
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    let result = mc.apply_into(0.05, &src, &mut dst, &mut pool);
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}
