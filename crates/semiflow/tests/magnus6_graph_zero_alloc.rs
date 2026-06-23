//! R4 zero-alloc gate: `MagnusGraphHeat6thChernoff::apply_into` in steady state.
//!
//! Contract §2.4 (wave-b-advanced-semigroups.md): pre-allocate work-buffers in
//! first call; zero allocation per step in steady state.
//!
//! BCOR-6 corrected form (ADR-0114) uses 22 `ScratchPool` buffers (2 ping-pong +
//! 20 for the 2-commutator outer-bracket assembly), vs the old 7. The warm-up
//! call allocates 22; subsequent steady-state calls allocate 0.
//!
//! Strategy: warm-up once (populates pool with 22 scratch buffers), then measure
//! allocations for a second identical call with the same pre-warmed pool.
//!
//! See ADR-0056, ADR-0114, R4 zero-alloc invariant.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;

use allocation_counter::{self, AllocationInfo};

use semiflow_core::{
    magnus6_graph::MagnusGraphHeat6thChernoff, magnus_graph::LaplacianAtTime, ChernoffFunction,
    Graph, GraphSignal, Laplacian, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_magnus6_f64(n: usize) -> (MagnusGraphHeat6thChernoff<f64>, Arc<Graph<f64>>) {
    let g = Arc::new(Graph::<f64>::path(n));
    // Pre-compute the Laplacian once and clone the Arc in the closure — zero
    // allocation per call (only reference-count bump).
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let lap_at: LaplacianAtTime<f64> = Box::new(move |_t| Arc::clone(&lap));
    let mc = MagnusGraphHeat6thChernoff::new(Arc::clone(&g), lap_at, 4.0, true).unwrap();
    (mc, g)
}

// ---------------------------------------------------------------------------
// 0 allocs after warm-up
// ---------------------------------------------------------------------------

#[test]
fn magnus6_apply_into_zero_alloc_steady_f64() {
    let n = 64usize;
    let (mc, g) = make_magnus6_f64(n);

    let tau = 0.01_f64;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 - n as f64 / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();

    // Warm-up: allocates 22 scratch buffers into pool (BCOR-6 form; ADR-0114).
    let mut pool = ScratchPool::<f64>::new();
    mc.apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up");

    // Steady-state measurement: pool already holds the 22 required buffers.
    let info: AllocationInfo = allocation_counter::measure(|| {
        mc.apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state");
    });

    assert_eq!(
        info.count_total, 0,
        "MagnusGraphHeat6thChernoff f64 steady-state allocated {} times (expected 0)",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// 0 allocs after warm-up — apply_into_at variant
// ---------------------------------------------------------------------------

#[test]
fn magnus6_apply_into_at_zero_alloc_steady_f64() {
    let n = 64usize;
    let (mc, g) = make_magnus6_f64(n);

    let tau = 0.01_f64;
    let t_start = 0.0_f64;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 - n as f64 / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();

    // Warm-up.
    let mut pool = ScratchPool::<f64>::new();
    mc.apply_into_at(t_start, tau, &src, &mut dst, &mut pool)
        .expect("warm-up");

    // Steady-state measurement.
    let info: AllocationInfo = allocation_counter::measure(|| {
        mc.apply_into_at(t_start, tau, &src, &mut dst, &mut pool)
            .expect("steady-state");
    });

    assert_eq!(
        info.count_total, 0,
        "MagnusGraphHeat6thChernoff::apply_into_at f64 steady-state allocated {} times (expected 0)",
        info.count_total
    );
}
