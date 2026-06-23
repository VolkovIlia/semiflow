//! Criterion benchmark: graph heat Chernoff kernels.
//!
//! Compares three graph-PDE kernels on a fixed Erdős–Rényi random graph:
//!
//! - `GraphHeatChernoff` (order-1; Wave 2.1A)
//! - `VarCoefGraphHeatChernoff` (order-2 variable-a; ADR-0053)
//! - `MagnusGraphHeatChernoff` (Magnus K=4; ADR-0051)
//!
//! All three are serial-only (no parallel impl in v2.2). The bench measures
//! raw kernel throughput as a single-thread baseline. Thread-scaling for graph
//! kernels is deferred to v2.3+ (ADR-0060 §"Out of scope").
//!
//! Graph: N=2000-node Erdős–Rényi with edge probability p=0.008 (mean degree ~16),
//! seeded at 42. `n_chernoff=20` steps.  The graph is assembled once outside the
//! measurement loop and shared via `Arc`.
//!
//! Run (quick smoke):
//! ```sh
//! cargo bench -p semiflow-core --bench graph_heat -- --quick
//! ```
//! Results land in `target/criterion/graph_heat/`.

#![allow(missing_docs)]

use std::{sync::Arc, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use semiflow_core::{
    ChernoffSemigroup, Graph, GraphHeatChernoff, GraphSignal, Laplacian, LaplacianAtTime,
    MagnusGraphHeatChernoff, VarCoefGraphHeatChernoff,
};

/// Node count for the random graph.
const N_NODES: usize = 2_000;

/// Edge probability: ~p*N*(N-1)/2 edges (mean degree ≈ 16 for p=0.008, N=2000).
const EDGE_P: f64 = 0.008;

/// Fixed random seed (deterministic across machines and runs).
const SEED: u64 = 42;

/// Number of Chernoff steps in `ChernoffSemigroup`.
const N_STEPS: usize = 20;

/// Build a fixed Erdős–Rényi graph shared across all bench variants.
fn make_graph() -> Arc<Graph<f64>> {
    Arc::new(Graph::<f64>::erdos_renyi(N_NODES, EDGE_P, SEED))
}

/// Build initial signal: `f[i] = exp(-i/N)`.
fn make_signal(graph: &Arc<Graph<f64>>) -> GraphSignal<f64> {
    #[allow(clippy::cast_precision_loss)]
    let n = graph.n_nodes() as f64;
    GraphSignal::from_fn(Arc::clone(graph), |i| (-f64::from(i) / n).exp())
}

/// Build `GraphHeatChernoff` semigroup (order-1).
fn make_order1_semi(
    graph: &Arc<Graph<f64>>,
) -> ChernoffSemigroup<GraphHeatChernoff<f64>, GraphSignal<f64>> {
    let lap = Arc::new(Laplacian::assemble_combinatorial(graph));
    let kernel = GraphHeatChernoff::new(lap);
    ChernoffSemigroup::new(kernel, N_STEPS).unwrap()
}

/// Build `VarCoefGraphHeatChernoff` semigroup (order-2, variable-a).
///
/// CFL constraint: `tau * rho_bar * max(a)^2 <= 0.5`.
/// With `a[i] = 1.0` and `rho_bar` from the assembled Laplacian, the maximum
/// safe tau is `0.5 / rho_bar`. The semigroup uses `n_steps` steps over T=1.0,
/// so we need `n_steps >= 2 * rho_bar`.
fn make_var_coef_semi(
    graph: &Arc<Graph<f64>>,
) -> ChernoffSemigroup<VarCoefGraphHeatChernoff<f64>, GraphSignal<f64>> {
    let n = graph.n_nodes();
    let lap = Arc::new(Laplacian::assemble_combinatorial(graph));
    let rho_bar = lap.spectral_radius_bound();
    // Uniform conductivity a[i] = 1.0. CFL: tau <= 0.5 / rho_bar.
    // n_steps = ceil(2 * rho_bar) + 1 ensures tau = 1.0 / n_steps < 0.5 / rho_bar.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n_steps = (2.0 * rho_bar).ceil() as usize + 1;
    let a = vec![1.0_f64; n];
    let kernel = VarCoefGraphHeatChernoff::new(Arc::clone(graph), a, rho_bar).unwrap();
    ChernoffSemigroup::new(kernel, n_steps).unwrap()
}

/// Build `MagnusGraphHeatChernoff` semigroup (K=4).
///
/// Magnus radius constraint: `rho_bar_max * tau < π/2`.
/// `n_steps = ceil(2 * rho_bar / π) + 1` ensures `tau = 1.0 / n_steps < π/(2*rho_bar)`.
fn make_magnus_semi(
    graph: &Arc<Graph<f64>>,
) -> ChernoffSemigroup<MagnusGraphHeatChernoff<f64>, GraphSignal<f64>> {
    let lap = Arc::new(Laplacian::assemble_combinatorial(graph));
    let rho_bar = lap.spectral_radius_bound();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n_steps = (2.0 * rho_bar / core::f64::consts::PI).ceil() as usize + 1;
    // Time-independent closure: L_G(t) = constant combinatorial Laplacian.
    let lap_for_closure = Arc::clone(&lap);
    let lap_fn: LaplacianAtTime<f64> = Box::new(move |_t| Arc::clone(&lap_for_closure));
    let kernel = MagnusGraphHeatChernoff::new(Arc::clone(graph), lap_fn, rho_bar, true).unwrap();
    ChernoffSemigroup::new(kernel, n_steps).unwrap()
}

/// Benchmark: order-1 `GraphHeatChernoff`.
fn bench_graph_heat_order1(c: &mut Criterion) {
    let graph = make_graph();
    let f0 = make_signal(&graph);
    let semi = make_order1_semi(&graph);

    let mut group = c.benchmark_group("graph_heat/order1");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    group.bench_function(format!("N={N_NODES}/steps={N_STEPS}"), |b| {
        b.iter(|| semi.evolve(black_box(1.0), black_box(&f0)).expect("evolve"));
    });

    group.finish();
}

/// Benchmark: order-2 `VarCoefGraphHeatChernoff`.
///
/// Step count is computed from the graph's spectral radius to satisfy the CFL.
fn bench_graph_heat_var_coef(c: &mut Criterion) {
    let graph = make_graph();
    let f0 = make_signal(&graph);
    let semi = make_var_coef_semi(&graph);

    let mut group = c.benchmark_group("graph_heat/var_coef");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    group.bench_function(format!("N={N_NODES}"), |b| {
        b.iter(|| semi.evolve(black_box(1.0), black_box(&f0)).expect("evolve"));
    });

    group.finish();
}

/// Benchmark: Magnus K=4 `MagnusGraphHeatChernoff`.
///
/// Step count is computed from the graph's spectral radius to satisfy
/// the Magnus convergence-radius constraint.
fn bench_graph_heat_magnus(c: &mut Criterion) {
    let graph = make_graph();
    let f0 = make_signal(&graph);
    let semi = make_magnus_semi(&graph);

    let mut group = c.benchmark_group("graph_heat/magnus_k4");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    group.bench_function(format!("N={N_NODES}"), |b| {
        b.iter(|| semi.evolve(black_box(1.0), black_box(&f0)).expect("evolve"));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_graph_heat_order1,
    bench_graph_heat_var_coef,
    bench_graph_heat_magnus
);
criterion_main!(benches);
