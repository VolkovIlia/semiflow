//! Zero-alloc gate: `MagnusGraphHeatChernoff::apply_into` allocates 0 heap
//! bytes per step in steady state after warm-up (R4 invariant, ADR-0051).
//!
//! Contract §"R4 zero-alloc invariant preserved": `apply_into` acquires 5
//! scratch buffers from `ScratchPool` via `take_vec`/`return_vec` and returns
//! them all before returning. After warm-up the pool free-list holds those 5
//! buffers, so the next call allocates 0 bytes.
//!
//! The `lap_at_t` closure used here captures a pre-built `Arc<Laplacian>` and
//! returns `Arc::clone` on every call — which does NOT allocate. This isolates
//! the R4 invariant to the kernel's own scratch-buffer bookkeeping, independent
//! of user-supplied closure cost.
//!
//! Strategy: warm-up 3 steps (fully populates the free-list), then measure
//! allocations for a subsequent identical call on the same pool.
//!
//! See `graph_apply_into_zero_alloc.rs` for the Wave 2.1A/B sibling tests.

#![allow(clippy::cast_precision_loss)] // node index → float cast in signal init
#![allow(clippy::cast_lossless)] // u32 → f64 widening

use std::sync::Arc;

use allocation_counter::{self, AllocationInfo};

use semiflow::{
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    magnus_graph::{LaplacianAtTime, MagnusGraphHeatChernoff},
    ChernoffFunction, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `MagnusGraphHeatChernoff` on `P_N` with a **constant** Laplacian.
///
/// The `lap_at_t` closure captures a pre-built `Arc<Laplacian<f64>>` and
/// returns `Arc::clone` of it on every call.  `Arc::clone` does not heap-
/// allocate, so this closure is alloc-free in steady state — which is exactly
/// what the R4 zero-alloc invariant tests for (the invariant covers only the
/// 5 scratch buffers the kernel itself acquires).
///
/// Returns `(mc, graph_arc)` so callers can construct `GraphSignal` values
/// without going through `mc.graph()`, which returns `&Graph<F>` not
/// `&Arc<Graph<F>>`.
fn make_path_mc(n: usize) -> (MagnusGraphHeatChernoff<f64>, Arc<Graph<f64>>) {
    let g = Arc::new(Graph::<f64>::path(n));
    let g2 = Arc::clone(&g);
    // Pre-build the Laplacian once; the closure just clones the Arc (no alloc).
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let lap_at: LaplacianAtTime<f64> = Box::new(move |_t: f64| Arc::clone(&lap));
    // rho_bar_max = 2.0 (path graph, all weights = 1); radius check enabled.
    let mc = MagnusGraphHeatChernoff::new(g, lap_at, 2.0_f64, true)
        .expect("valid path MC must construct");
    (mc, g2)
}

/// Gaussian-peak initial signal on an N-node graph.
fn gaussian_signal(g: &Arc<Graph<f64>>, n: usize) -> GraphSignal<f64> {
    GraphSignal::from_fn(Arc::clone(g), |i| {
        let x = i as f64 - (n as f64) / 2.0;
        (-x * x / 8.0).exp()
    })
}

// ---------------------------------------------------------------------------
// Zero-alloc test: path graph P_16, constant Laplacian (alloc-free closure)
// ---------------------------------------------------------------------------

/// R4 gate: `MagnusGraphHeatChernoff::apply_into` allocates 0 heap bytes in
/// steady state (3 warm-up steps to fully populate the pool free-list).
#[test]
fn magnus_apply_into_zero_alloc_steady_f64() {
    let n = 16_usize;
    let (mc, g) = make_path_mc(n);

    let tau = 0.005_f64;
    let src = gaussian_signal(&g, n);
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();

    // Warm-up: 3 steps ensure all 5 scratch buffers are in the free-list.
    for _ in 0..3 {
        mc.apply_into(tau, &src, &mut dst, &mut pool)
            .expect("warm-up apply_into must succeed");
    }

    // Steady-state measurement: pool holds all required capacity.
    let info: AllocationInfo = allocation_counter::measure(|| {
        mc.apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state apply_into must succeed");
    });

    assert_eq!(
        info.count_total, 0,
        "MagnusGraphHeatChernoff f64 steady-state allocated {} times (expected 0) — R4 invariant violated",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// Zero-alloc test: apply_into_at with explicit t_start, constant Laplacian
// ---------------------------------------------------------------------------

/// R4 gate via `apply_into_at`: explicit `t_start` tracking does not break
/// the zero-alloc invariant (same 5 scratch buffers, same pool reuse path).
#[test]
fn magnus_apply_into_at_zero_alloc_steady_f64() {
    let n = 16_usize;
    let (mc, g) = make_path_mc(n);

    let tau = 0.005_f64;
    let src = gaussian_signal(&g, n);
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();

    // Warm-up with apply_into_at (t_start = 0.0, 0.005, 0.010 …).
    for step in 0..3_u32 {
        let t = f64::from(step) * tau;
        mc.apply_into_at(t, tau, &src, &mut dst, &mut pool)
            .expect("warm-up apply_into_at must succeed");
    }

    // Steady-state measurement.
    let t_start = 0.015_f64;
    let info: AllocationInfo = allocation_counter::measure(|| {
        mc.apply_into_at(t_start, tau, &src, &mut dst, &mut pool)
            .expect("steady-state apply_into_at must succeed");
    });

    assert_eq!(
        info.count_total, 0,
        "MagnusGraphHeatChernoff::apply_into_at f64 steady-state allocated {} times (expected 0) — R4 invariant violated",
        info.count_total
    );
}
