//! R4 zero-alloc gate: `AdjointChernoff::apply_into` in steady state.
//!
//! Contract §1.4 (wave-b-advanced-semigroups.md): all work-buffers MUST come
//! from `ScratchPool`. Zero heap allocation per step in steady state.
//!
//! Strategy: warm-up once (populates pool), then measure with
//! `allocation_counter`. Tests both self-adjoint and general paths.
//!
//! See ADR-0055 R4 zero-alloc invariant.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;

use allocation_counter::{self, AllocationInfo};

use semiflow_core::{
    drift_reaction::DriftReactionChernoff, graph_heat::GraphHeatChernoff, AdjointChernoff,
    ChernoffFunction, Graph, GraphSignal, Grid1D, GridFn1D, Laplacian, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_path_heat(n: usize) -> (GraphHeatChernoff<f64>, Arc<Graph<f64>>) {
    let g = Arc::new(Graph::<f64>::path(n));
    let lap = Laplacian::assemble_combinatorial(&g);
    (GraphHeatChernoff::from_owned(lap), g)
}

// ---------------------------------------------------------------------------
// Self-adjoint path: 0 allocs (pure delegation, no correction scratch)
// ---------------------------------------------------------------------------

#[test]
fn adjoint_self_adjoint_apply_into_zero_alloc_steady_f64() {
    let n = 64usize;
    let (inner, g) = make_path_heat(n);
    let adj = AdjointChernoff::new_self_adjoint(inner);

    let tau = 0.01_f64;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 - n as f64 / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();

    // Warm-up: populates inner's scratch pool.
    let mut pool = ScratchPool::<f64>::new();
    adj.apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up");

    // Steady-state measurement.
    let info: AllocationInfo = allocation_counter::measure(|| {
        adj.apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state");
    });

    assert_eq!(
        info.count_total, 0,
        "AdjointChernoff (self-adjoint path) f64 allocated {} times (expected 0)",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// General (non-self-adjoint) path: 0 allocs
//
// Uses DriftReactionChernoff which implements AdjointApply<f64> (ADR-0114).
// The apply_adjoint_into path uses the negated-drift kernel (allocation-free).
// ---------------------------------------------------------------------------

#[test]
fn adjoint_general_apply_into_zero_alloc_steady_f64() {
    let n = 64usize;
    let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
    // DriftReactionChernoff implements AdjointApply<f64> — required for new_general.
    let inner = DriftReactionChernoff::new(|_| 0.3_f64, |_| 0.0_f64, 0.0, grid);
    let adj = AdjointChernoff::new_general(inner);

    let tau = 0.01_f64;
    let src = GridFn1D::from_fn(grid, |x| {
        // Compactly supported: near-zero at boundaries to avoid OobPolicy effects.
        let dx = x - 0.5;
        (-dx * dx / (2.0 * 0.1 * 0.1)).exp()
    });
    let mut dst = src.clone();

    // Warm-up: allocates inner's scratch buffers into pool.
    let mut pool = ScratchPool::<f64>::new();
    adj.apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up");

    // Steady-state measurement: pool already holds required capacity.
    let info: AllocationInfo = allocation_counter::measure(|| {
        adj.apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state");
    });

    assert_eq!(
        info.count_total, 0,
        "AdjointChernoff (general path) f64 allocated {} times (expected 0)",
        info.count_total
    );
}
