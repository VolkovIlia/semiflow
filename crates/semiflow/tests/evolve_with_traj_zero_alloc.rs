//! R4 zero-alloc test for `MagnusGraphHeatChernoff::evolve_with_traj_into` via
//! `allocation-counter`.
//!
//! Verifies that the steady-state hot path (`evolve_with_traj_into` after warmup)
//! performs zero heap allocations across a multi-segment trajectory (K=3).
//!
//! The R4 zero-alloc invariant is satisfied because:
//! - The ping-pong `cur` buffer is allocated **once** outside the segment loop.
//! - `cur.copy_from(dst)` at each segment boundary uses existing capacity.
//! - `apply_magnus_k4_with_fn` borrows its 5 scratch vectors from `ScratchPool`.
//! - `ScratchPool::take_vec` returns previously returned vectors without new alloc.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use allocation_counter::{self, AllocationInfo};

use semiflow::{
    graph_traj::{GraphTraj, SegmentWeightFn},
    magnus_graph::MagnusGraphHeatChernoff,
    Graph, GraphSignal, Laplacian, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a uniform-weight Laplacian for `P_n` with edge weight `w`.
fn make_lap_uniform(n: usize, w: f64) -> Arc<Laplacian<f64>> {
    let edges = (0..n as u32 - 1).map(|i| (i, i + 1, w));
    let g = Graph::from_edges(n, edges).expect("valid path edges");
    Arc::new(Laplacian::assemble_combinatorial(&g))
}

/// Build a 3-segment trajectory on `P_n`.
///
/// Segments use piecewise-constant Laplacians with weights 1.0, 0.7, 1.2 so
/// that all three segments exercise distinct graph operators.
fn make_traj(n: usize) -> (GraphTraj<f64>, Arc<Graph<f64>>) {
    let g_ref = Arc::new(Graph::<f64>::path(n));
    let breakpoints = vec![0.0_f64, 0.1, 0.2, 0.3];

    let snapshots: Vec<Arc<Graph<f64>>> = (0..3).map(|_| Arc::clone(&g_ref)).collect();

    let weights = [1.0_f64, 0.7, 1.2];
    let weight_fns: Vec<SegmentWeightFn<f64>> = weights
        .iter()
        .copied()
        .map(|w| {
            let lap = make_lap_uniform(n, w);
            let wfn: SegmentWeightFn<f64> = Box::new(move |_t: f64| Arc::clone(&lap));
            wfn
        })
        .collect();

    let traj = GraphTraj::new(breakpoints, snapshots, weight_fns).expect("valid 3-segment traj");
    (traj, g_ref)
}

/// Build a `MagnusGraphHeatChernoff` whose `rho_bar_max` covers the max spectral
/// radius of the trajectory Laplacians.  `rho_bar = 3.0` covers max-degree 2 *
/// max-weight 1.2 = 2.4 with margin.
fn make_magnus(n: usize) -> MagnusGraphHeatChernoff<f64> {
    use semiflow::magnus_graph::LaplacianAtTime;
    let topology = Arc::new(Graph::<f64>::path(n));
    let lap0 = make_lap_uniform(n, 1.0);
    let lap_at: LaplacianAtTime<f64> = Box::new(move |_t: f64| Arc::clone(&lap0));
    MagnusGraphHeatChernoff::new(Arc::clone(&topology), lap_at, 3.0_f64, true)
        .expect("valid magnus")
}

// ---------------------------------------------------------------------------
// R4 zero-alloc test: steady-state after single warmup
// ---------------------------------------------------------------------------

#[test]
fn evolve_with_traj_into_zero_alloc_steady_state() {
    let n = 32usize;
    let mc = make_magnus(n);
    let (traj, g) = make_traj(n);
    let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| (f64::from(i) * 0.1).sin());
    let mut dst = f0.clone();
    let mut scratch = ScratchPool::<f64>::new();

    // Warmup: first call populates ScratchPool and allocates the ping-pong buffer.
    mc.evolve_with_traj_into(&traj, 4, &f0, &mut dst, &mut scratch)
        .expect("warmup ok");

    // Steady state: pool is fully loaded, ping-pong buffer is sized.
    // Rebuild traj so the second call uses fresh closures (avoids Arc refcount
    // allocation from clone-on-write) while reusing the same pool and dst.
    let (traj2, _) = make_traj(n);

    let info: AllocationInfo = allocation_counter::measure(|| {
        mc.evolve_with_traj_into(&traj2, 4, &f0, &mut dst, &mut scratch)
            .expect("evolve_with_traj_into ok");
    });

    assert_eq!(
        info.count_total, 0,
        "evolve_with_traj_into must be zero-alloc in steady state, got {} allocs",
        info.count_total
    );
}

// ---------------------------------------------------------------------------
// R4 zero-alloc test: two warmup calls, third must be zero-alloc
// ---------------------------------------------------------------------------

#[test]
fn evolve_with_traj_into_zero_alloc_repeated() {
    let n = 16usize;
    let mc = make_magnus(n);
    let (_, g) = make_traj(n);
    let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
    let mut dst = f0.clone();
    let mut scratch = ScratchPool::<f64>::new();

    // Two warmup calls.
    for _ in 0..2 {
        let (traj_w, _) = make_traj(n);
        mc.evolve_with_traj_into(&traj_w, 3, &f0, &mut dst, &mut scratch)
            .expect("warmup ok");
    }

    // Third call: must be zero-alloc.
    let (traj3, _) = make_traj(n);
    let info: AllocationInfo = allocation_counter::measure(|| {
        mc.evolve_with_traj_into(&traj3, 3, &f0, &mut dst, &mut scratch)
            .expect("ok");
    });

    assert_eq!(
        info.count_total, 0,
        "repeated evolve_with_traj_into after warmup must be zero-alloc, got {} allocs",
        info.count_total
    );
}
