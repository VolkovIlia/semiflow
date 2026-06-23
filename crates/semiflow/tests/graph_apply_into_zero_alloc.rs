//! Zero-alloc gate: `GraphHeatChernoff::apply_into` allocates 0 heap bytes per
//! step in steady state after a single warm-up call.
//!
//! ADR-0047 acceptance criterion AC-4: once the `ScratchPool` has been populated
//! by the first `apply_into` call, every subsequent call on the same pool must
//! perform zero heap allocations.
//!
//! Strategy: warm-up once (populates `ScratchPool` free-list), then measure
//! allocations for a second identical call with the same pre-warmed pool.
//! The `allocation-counter` crate replaces the global allocator (test-only).

#![allow(clippy::cast_precision_loss)] // test-only node index → float casts
#![allow(clippy::cast_lossless)] // u32 → f64 widening casts in initializers

use std::sync::Arc;

use allocation_counter::{self, AllocationInfo};

use semiflow_core::{
    graph::{Graph, Laplacian},
    graph_heat::GraphHeatChernoff,
    graph_heat4::GraphHeat4thChernoff,
    graph_signal::GraphSignal,
    strang_graph::StrangSplitGraph,
    ChernoffFunction, ScratchPool,
};

// ---------------------------------------------------------------------------
// GraphHeatChernoff f64: 0 allocs after warm-up
// ---------------------------------------------------------------------------

#[test]
fn graph_heat_zero_alloc_steady_f64() {
    let n = 64_usize;
    let g = Arc::new(Graph::<f64>::erdos_renyi(n, 0.15, 0xC0FF_EE42));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let chernoff = GraphHeatChernoff::new(Arc::clone(&lap));

    let tau = 0.01_f64;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 - (n as f64) / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();

    // Warm-up: allocates scratch buffers, filling the pool's free-list.
    let mut pool = ScratchPool::<f64>::new();
    chernoff
        .apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up apply_into must succeed");

    // Steady-state measurement: pool already holds the required capacity.
    let info: AllocationInfo = allocation_counter::measure(|| {
        chernoff
            .apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state apply_into must succeed");
    });

    assert_eq!(
        info.count_total, 0,
        "GraphHeatChernoff f64 steady-state allocated {} times (expected 0)",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// GraphHeatChernoff::with_zeta_a f64: 0 allocs after warm-up (Wave 2.1B)
// ---------------------------------------------------------------------------

#[test]
fn with_zeta_a_apply_into_zero_alloc_steady_f64() {
    let n = 64_usize;
    let g = Arc::new(Graph::<f64>::erdos_renyi(n, 0.15, 0xC0FF_EE43));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let chernoff = GraphHeatChernoff::with_zeta_a(Arc::clone(&lap));

    let tau = 0.01_f64;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 - (n as f64) / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();

    // Warm-up: populates pool free-list (allocates 2 scratch buffers).
    chernoff
        .apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up zeta_a apply_into must succeed");

    let info: AllocationInfo = allocation_counter::measure(|| {
        chernoff
            .apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state zeta_a apply_into must succeed");
    });

    assert_eq!(
        info.count_total, 0,
        "GraphHeatChernoff::with_zeta_a f64 steady-state allocated {} times (expected 0)",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// GraphHeat4thChernoff f64: 0 allocs after warm-up (Wave 2.1B)
// ---------------------------------------------------------------------------

#[test]
fn graph_heat4_apply_into_zero_alloc_steady_f64() {
    let n = 64_usize;
    let g = Arc::new(Graph::<f64>::erdos_renyi(n, 0.15, 0xC0FF_EE44));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let chernoff = GraphHeat4thChernoff::new(Arc::clone(&lap));

    let tau = 0.005_f64;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 - (n as f64) / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();

    // Warm-up: populates pool (2 scratch bufs for ping-pong).
    chernoff
        .apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up graph_heat4 apply_into must succeed");

    let info: AllocationInfo = allocation_counter::measure(|| {
        chernoff
            .apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state graph_heat4 apply_into must succeed");
    });

    assert_eq!(
        info.count_total, 0,
        "GraphHeat4thChernoff f64 steady-state allocated {} times (expected 0)",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// StrangSplitGraph (bipartite path) f64: 0 allocs after warm-up (Wave 2.1B R4)
// ---------------------------------------------------------------------------

#[test]
fn strang_bipartite_path_apply_into_zero_alloc_steady_f64() {
    let n = 64_usize;
    let g = Arc::new(Graph::<f64>::path(n));
    let strang = StrangSplitGraph::new_bipartite_path(&g).expect("path(64) builds OK");

    let tau = 0.01_f64;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 - (n as f64) / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();

    // Warm-up: allocates graph-signal arena + inner scratch bufs.
    strang
        .apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up strang apply_into must succeed");

    let info: AllocationInfo = allocation_counter::measure(|| {
        strang
            .apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state strang apply_into must succeed");
    });

    assert_eq!(
        info.count_total, 0,
        "StrangSplitGraph (bipartite path) f64 steady-state allocated {} times (expected 0) — R4 invariant violated",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// GraphHeatChernoff f32: 0 allocs after warm-up
// ---------------------------------------------------------------------------

#[test]
fn graph_heat_zero_alloc_steady_f32() {
    let n = 64_usize;
    let g = Arc::new(Graph::<f32>::erdos_renyi(n, 0.15, 0xC0FF_EE42));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let chernoff = GraphHeatChernoff::new(Arc::clone(&lap));

    let tau = 0.01_f32;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f32 - (n as f32) / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();

    let mut pool = ScratchPool::<f32>::new();
    chernoff
        .apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up apply_into f32 must succeed");

    let info: AllocationInfo = allocation_counter::measure(|| {
        chernoff
            .apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state apply_into f32 must succeed");
    });

    assert_eq!(
        info.count_total, 0,
        "GraphHeatChernoff f32 steady-state allocated {} times (expected 0)",
        info.count_total
    );
}
