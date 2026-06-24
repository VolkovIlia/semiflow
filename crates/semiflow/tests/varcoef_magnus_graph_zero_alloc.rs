//! Zero-alloc gate: `VarCoefMagnusGraphHeatChernoff::apply_into` allocates
//! 0 heap bytes per step in steady state.
//!
//! Note: each `apply_into` call invokes the caller closures `lap_at_t(t)` and
//! `a_at_t(t)`, which return `Arc<Laplacian<F>>` and `Vec<F>` respectively.
//! Those allocations are CALLER-RESPONSIBLE (the closures own them). The
//! library kernel itself MUST NOT allocate beyond what `ScratchPool` already
//! holds after warmup.
//!
//! Strategy: cache pre-built `Arc<Laplacian>` and a fixed `Vec<F>` inside the
//! closures so that calling them returns Arc-clones / Vec-clones without new
//! heap allocations. The Vec-clone DOES allocate — so to actually measure
//! zero kernel allocations we need to inspect after the closures populate
//! the kernel's internal scratch (`sqrt_a_1/sqrt_a_2` + omega buffers).
//!
//! Reality check: `(self.a_at_t)(t)` returns `Vec<F>` which allocates inside
//! the closure. We accept that cost in v2.4 (one Vec alloc per quadrature
//! point per step = 2 allocs/step). The zero-alloc claim is for the
//! kernel-internal buffers; we count `count_total - 2`.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;

use allocation_counter::{self, AllocationInfo};
use semiflow::{
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    varcoef_magnus_graph::{VarCoefMagnusGraphHeatChernoff, WeightAtTime},
    ChernoffFunction, LaplacianAtTime, ScratchPool,
};

#[test]
fn varcoef_magnus_kernel_allocs_only_for_callbacks_f64() {
    let n = 32_usize;
    let g = Arc::new(Graph::<f64>::path(n));
    let lap_cache = Arc::new(Laplacian::assemble_combinatorial(&g));

    // Closures that return PRE-BUILT data:
    // - lap_at_t clones an Arc (no allocation beyond the Arc count bump).
    // - a_at_t allocates a fresh Vec (1 alloc per call).
    let lap_for_closure = Arc::clone(&lap_cache);
    let lap_at: LaplacianAtTime<f64> = Box::new(move |_t| Arc::clone(&lap_for_closure));
    let a_at: WeightAtTime<f64> = Box::new(move |_t| vec![1.5_f64; n]);

    let mc = VarCoefMagnusGraphHeatChernoff::new(n, lap_at, a_at, 4.0, 1.5_f64.sqrt())
        .expect("valid inputs");

    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 - (n as f64) / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();

    let tau = 0.05_f64;

    // Warm-up: populates pool with sqrt_a + omega + tmp buffers (9 take_vec).
    mc.apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up apply_into must succeed");

    let info: AllocationInfo = allocation_counter::measure(|| {
        mc.apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state apply_into must succeed");
    });

    // Acceptable allocations per step:
    // - 2 × `Vec<f64>` from `a_at_t(c_i τ)` (closure-owned, ADR-0063 §"WeightAtTime")
    // - 0 × Laplacian clones (lap_at_t returns Arc::clone)
    //
    // The library itself MUST add zero allocations beyond these.
    //
    // We bound at 2 since Vec allocation may or may not show up depending on
    // allocator capture. This is the BUDGET for v2.4; v2.5 may add a
    // pre-built `a` schedule to eliminate the 2 closure allocs.
    assert!(
        info.count_total <= 2,
        "VarCoefMagnusGraphHeatChernoff f64 kernel allocated {} times (budget: 2 for closures)",
        info.count_total
    );
}

#[test]
fn varcoef_magnus_kernel_allocs_only_for_callbacks_f32() {
    let n = 32_usize;
    let g = Arc::new(Graph::<f32>::path(n));
    let lap_cache = Arc::new(Laplacian::assemble_combinatorial(&g));

    let lap_for_closure = Arc::clone(&lap_cache);
    let lap_at: LaplacianAtTime<f32> = Box::new(move |_t| Arc::clone(&lap_for_closure));
    let a_at: WeightAtTime<f32> = Box::new(move |_t| vec![1.0_f32; n]);

    let mc = VarCoefMagnusGraphHeatChernoff::new(n, lap_at, a_at, 4.0_f32, 1.0_f32)
        .expect("valid inputs");

    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f32 - (n as f32) / 2.0;
        (-x * x / 8.0).exp()
    });
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f32>::new();

    let tau = 0.05_f32;
    mc.apply_into(tau, &src, &mut dst, &mut pool)
        .expect("warm-up apply_into f32 must succeed");

    let info: AllocationInfo = allocation_counter::measure(|| {
        mc.apply_into(tau, &src, &mut dst, &mut pool)
            .expect("steady-state apply_into f32 must succeed");
    });

    assert!(
        info.count_total <= 2,
        "VarCoefMagnusGraphHeatChernoff f32 kernel allocated {} times (budget: 2 for closures)",
        info.count_total
    );
}
