//! Zero-alloc gate: `GraphHeat6thChernoff::apply_into` allocates 0 heap bytes
//! per step in steady state after a single warm-up call.
//!
//! Same contract as ADR-0047 / ADR-0042: once the `ScratchPool` holds the
//! required capacities, every subsequent `apply_into` reuses them without
//! heap traffic. The `allocation-counter` crate replaces the global allocator
//! (test-only) to count allocations.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;

use allocation_counter::{self, AllocationInfo};

use semiflow::{
    graph::{Graph, Laplacian},
    graph_heat6::GraphHeat6thChernoff,
    graph_signal::GraphSignal,
    ChernoffFunction, ScratchPool,
};

#[test]
fn graph_heat6_zero_alloc_steady_f64() {
    let n = 64_usize;
    let g = Arc::new(Graph::<f64>::erdos_renyi(n, 0.15, 0xC0FF_EE66));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let chernoff = GraphHeat6thChernoff::new(Arc::clone(&lap));

    let tau = 0.003_f64;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 - (n as f64) / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();

    // Warm-up: populates pool with 2 ping-pong scratch bufs.
    let mut pool = ScratchPool::<f64>::new();
    chernoff
        .apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up graph_heat6 apply_into must succeed");

    let info: AllocationInfo = allocation_counter::measure(|| {
        chernoff
            .apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state graph_heat6 apply_into must succeed");
    });

    assert_eq!(
        info.count_total, 0,
        "GraphHeat6thChernoff f64 steady-state allocated {} times (expected 0)",
        info.count_total
    );
}

#[test]
fn graph_heat6_zero_alloc_steady_f32() {
    let n = 64_usize;
    let g = Arc::new(Graph::<f32>::erdos_renyi(n, 0.15, 0xC0FF_EE67));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let chernoff = GraphHeat6thChernoff::new(Arc::clone(&lap));

    let tau = 0.003_f32;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f32 - (n as f32) / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();

    let mut pool = ScratchPool::<f32>::new();
    chernoff
        .apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up graph_heat6 f32 must succeed");

    let info: AllocationInfo = allocation_counter::measure(|| {
        chernoff
            .apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state graph_heat6 f32 must succeed");
    });

    assert_eq!(
        info.count_total, 0,
        "GraphHeat6thChernoff f32 steady-state allocated {} times (expected 0)",
        info.count_total
    );
}
