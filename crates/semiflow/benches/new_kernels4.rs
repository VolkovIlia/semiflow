//! Criterion benchmark: Magnus-6 graph + state-adjoint / sensitivity kernels
//! + `ObstacleChernoff` cost-per-step (v6.3.0).
//!
//! Covers the v6.2.1–v6.2.2 graph-signal family plus obstacle kernels:
//!
//! | Kernel / operation                              | State/output        | Order |
//! |-------------------------------------------------|---------------------|-------|
//! | `MagnusGraphHeat6thChernoff` N=64               | `GraphSignal`         | 6     |
//! | `MagnusGraphHeat6thChernoff` N=256              | `GraphSignal`         | 6     |
//! | `MagnusGraphHeatChernoff::apply_state_adjoint`  | `GraphSignal`         | 4     |
//! | `magnus_step_jvp_into` (forward JVP, 1 param)  | slice               | —     |
//! | `adjoint_state_gradient` (n=4 steps, 2 params) | gradient vec        | —     |
//! | `ObstacleChernoff<DiffusionChernoff>` N=256     | `GridFn1D`            | 1     |
//! | `ObstacleChernoff<DiffusionChernoff>` N=1024    | `GridFn1D`            | 1     |
//!
//! Note on Magnus-6 cost: one `apply_into_at` triggers 3 Laplacian evaluations,
//! 8 `SpMV` for 2 commutators, plus a degree-6 Taylor exponentiation (≈18 `SpMV` total).
//!
//! Run:
//! ```sh
//! cargo bench -p semiflow-core --bench new_kernels4 --features simd
//! # Quick smoke:
//! cargo bench -p semiflow-core --bench new_kernels4 --features simd -- \
//!     --warm-up-time 1 --measurement-time 2 --sample-size 10
//! ```
//! Results land in `target/criterion/new_kernels4/`.

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]
// Integration test/bench/example: allows for numerical patterns.
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use semiflow::{
    chernoff::ChernoffFunction,
    graph_sensitivity::{adjoint_state_gradient, magnus_step_jvp_into, EdgeWeightSensitivity},
    magnus6_graph::MagnusGraphHeat6thChernoff,
    magnus_graph::MagnusGraphHeatChernoff,
    BoundaryPolicy, ConstantObstacle, DiffusionChernoff, Graph, GraphSignal, Grid1D, GridFn1D,
    Laplacian, ObstacleChernoff, ScratchPool,
};

// ---------------------------------------------------------------------------
// Bench configuration
// ---------------------------------------------------------------------------

/// Time step for all graph benchmarks.
const TAU: f64 = 0.01;
/// Warm-up duration.
const WARM_UP_SECS: u64 = 3;
/// Measurement duration.
const MEASUREMENT_SECS: u64 = 10;
/// Starting time for Magnus steps.
const T_START: f64 = 0.0;

// ---------------------------------------------------------------------------
// Phased-edge Laplacian builder (non-commuting; mirrors g17_magnus6_slope.rs)
// ---------------------------------------------------------------------------

/// Phase per edge: φᵢ = (i × 0.13) mod 1 — ensures non-commuting L(t₁), L(t₂).
fn edge_phase(i: usize) -> f64 {
    let raw = (i as f64) * 0.13;
    raw - raw.floor()
}

fn weight_at(t: f64, edge_idx: usize) -> f64 {
    let phi = edge_phase(edge_idx);
    1.0 + 0.3 * (core::f64::consts::PI * (t + phi)).sin()
}

/// Build a path-graph Laplacian with phased edge weights at time `t`.
fn laplacian_phased(n_nodes: usize, t: f64) -> Laplacian<f64> {
    let edges = (0..n_nodes as u32 - 1).map(|i| (i, i + 1, weight_at(t, i as usize)));
    let g = Graph::from_edges(n_nodes, edges).expect("valid path graph");
    Laplacian::assemble_combinatorial(&g)
}

/// Non-constant IC: `f_i` = sin(0.31·π·i/N) + 0.2·cos(1.7·i).
fn make_ic(graph_arc: Arc<Graph<f64>>) -> GraphSignal<f64> {
    let n_nodes = graph_arc.n_nodes();
    let pi = core::f64::consts::PI;
    GraphSignal::from_fn(graph_arc, |i| {
        (0.31 * pi * f64::from(i) / (n_nodes as f64)).sin() + 0.2 * (1.7 * f64::from(i)).cos()
    })
}

/// Build a path-graph `Arc<Graph>`.
fn make_path_graph(n_nodes: usize) -> Arc<Graph<f64>> {
    Arc::new(
        Graph::from_edges(
            n_nodes,
            (0..n_nodes as u32 - 1).map(|i| (i, i + 1, 1.0_f64)),
        )
        .expect("path graph"),
    )
}

/// Build a `MagnusGraphHeat6thChernoff` on a path of `n_nodes` nodes.
fn make_magnus6(n_nodes: usize) -> MagnusGraphHeat6thChernoff<f64> {
    let g = make_path_graph(n_nodes);
    let lap_at_t = Box::new(move |t: f64| Arc::new(laplacian_phased(n_nodes, t)));
    // Gershgorin bound: path graph, max degree 2, max weight ≈ 1.3 → ρ̄_max ≈ 2.6.
    MagnusGraphHeat6thChernoff::new(g, lap_at_t, 2.6_f64, false).expect("magnus6 kernel")
}

/// Build a `MagnusGraphHeatChernoff` (order 4) for adjoint benchmarks.
fn make_magnus4(n_nodes: usize) -> MagnusGraphHeatChernoff<f64> {
    let g = make_path_graph(n_nodes);
    let lap_at_t = Box::new(move |t: f64| Arc::new(laplacian_phased(n_nodes, t)));
    MagnusGraphHeatChernoff::new(g, lap_at_t, 2.6_f64, false).expect("magnus4 kernel")
}

// ---------------------------------------------------------------------------
// Group 1: MagnusGraphHeat6thChernoff per-step cost
// ---------------------------------------------------------------------------

fn bench_magnus6_per_step(c: &mut Criterion, n_nodes: usize) {
    let kernel = make_magnus6(n_nodes);
    let g = make_path_graph(n_nodes);
    let ic = make_ic(g);
    let mut dst = ic.clone();
    let mut scratch = ScratchPool::new();

    let mut group = c.benchmark_group(format!("new_kernels4/magnus6_per_step/N={n_nodes}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(n_nodes as u64));

    group.bench_with_input(
        BenchmarkId::new("magnus6_phased_edges", ""),
        &(),
        |b, ()| {
            b.iter(|| {
                kernel
                    .apply_into_at(
                        black_box(T_START),
                        black_box(TAU),
                        black_box(&ic),
                        &mut dst,
                        &mut scratch,
                    )
                    .expect("ok");
            });
        },
    );

    group.finish();
}

fn bench_magnus6_n64(c: &mut Criterion) {
    bench_magnus6_per_step(c, 64);
}
fn bench_magnus6_n256(c: &mut Criterion) {
    bench_magnus6_per_step(c, 256);
}

// ---------------------------------------------------------------------------
// Group 2: MagnusGraphHeatChernoff::apply_state_adjoint (order 4)
// ---------------------------------------------------------------------------

fn bench_state_adjoint(c: &mut Criterion, n_nodes: usize) {
    let kernel = make_magnus4(n_nodes);
    let g = make_path_graph(n_nodes);
    let src = make_ic(g);
    let mut dst = src.clone();
    let mut scratch = ScratchPool::new();

    let mut group = c.benchmark_group(format!("new_kernels4/state_adjoint/N={n_nodes}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(n_nodes as u64));

    group.bench_with_input(BenchmarkId::new("apply_state_adjoint", ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_state_adjoint_into_at(
                    black_box(T_START),
                    black_box(TAU),
                    black_box(&src),
                    &mut dst,
                    &mut scratch,
                )
                .expect("ok");
        });
    });

    group.finish();
}

fn bench_state_adjoint_n64(c: &mut Criterion) {
    bench_state_adjoint(c, 64);
}
fn bench_state_adjoint_n256(c: &mut Criterion) {
    bench_state_adjoint(c, 256);
}

// ---------------------------------------------------------------------------
// Group 3: magnus_step_jvp_into — forward JVP for one parameter
// ---------------------------------------------------------------------------

fn bench_jvp(c: &mut Criterion, n_nodes: usize) {
    // Two GL4 Laplacians at t_start + c1·τ and t_start + c2·τ.
    let lap1 = laplacian_phased(n_nodes, T_START + 0.211_324_865 * TAU);
    let lap2 = laplacian_phased(n_nodes, T_START + 0.788_675_135 * TAU);
    // delta Laplacians: rank-1 perturbation on edge (0,1) only (δw₀₁ = 0.01).
    // Build a single-edge graph — Graph::from_edges rejects zero/non-positive
    // weights, so we include ONLY the perturbed edge at its δw value.
    // The resulting Laplacian is the sparse matrix [δL]_{01} = δw · L_{single_01}.
    let dlap1 = {
        let g = Graph::from_edges(n_nodes, core::iter::once((0_u32, 1_u32, 0.01_f64)))
            .expect("dlap1 graph: single perturbed edge");
        Laplacian::assemble_combinatorial(&g)
    };
    let dlap2 = {
        let g = Graph::from_edges(n_nodes, core::iter::once((0_u32, 1_u32, 0.01_f64)))
            .expect("dlap2 graph: single perturbed edge");
        Laplacian::assemble_combinatorial(&g)
    };

    let g = make_path_graph(n_nodes);
    let ic = make_ic(g);
    let u: Vec<f64> = ic.values().to_vec();
    let mut out = vec![0.0_f64; n_nodes];
    let mut scratch = ScratchPool::new();

    let mut group = c.benchmark_group(format!("new_kernels4/magnus_jvp/N={n_nodes}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(n_nodes as u64));

    group.bench_with_input(BenchmarkId::new("step_jvp_1param", ""), &(), |b, ()| {
        b.iter(|| {
            magnus_step_jvp_into(
                black_box(&lap1),
                black_box(&lap2),
                black_box(&dlap1),
                black_box(&dlap2),
                black_box(TAU),
                black_box(&u),
                &mut out,
                &mut scratch,
            )
            .expect("ok");
        });
    });

    group.finish();
}

fn bench_jvp_n64(c: &mut Criterion) {
    bench_jvp(c, 64);
}
fn bench_jvp_n256(c: &mut Criterion) {
    bench_jvp(c, 256);
}

// ---------------------------------------------------------------------------
// Group 4: adjoint_state_gradient (full backward sweep, n=4 steps, 2 params)
// ---------------------------------------------------------------------------

fn bench_adjoint_gradient(c: &mut Criterion, n_nodes: usize) {
    let kernel = make_magnus4(n_nodes);
    let g = make_path_graph(n_nodes);
    let u0 = make_ic(g.clone());
    // Sensitivity: 2 edge-weight parameters (edges 0 and 1).
    let sens = EdgeWeightSensitivity {
        params: vec![(0, 1), (1, 2)],
        n_nodes,
    };
    let dj_du_n = u0.clone(); // arbitrary terminal adjoint state
    let n_steps = 4_usize;

    let mut group = c.benchmark_group(format!("new_kernels4/adjoint_gradient/N={n_nodes}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    // Cost scales as n_steps × n_nodes.
    group.throughput(Throughput::Elements((n_steps * n_nodes) as u64));

    let mut scratch = ScratchPool::new();

    group.bench_with_input(BenchmarkId::new("grad_2params_4steps", ""), &(), |b, ()| {
        b.iter(|| {
            let mut grad = [0.0_f64; 2];
            adjoint_state_gradient(
                black_box(&kernel),
                black_box(&u0),
                black_box(n_steps),
                black_box(TAU),
                black_box(&dj_du_n),
                black_box(&sens),
                &mut grad,
                &mut scratch,
            )
            .expect("ok");
            black_box(grad)
        });
    });

    group.finish();
}

fn bench_adjoint_gradient_n64(c: &mut Criterion) {
    bench_adjoint_gradient(c, 64);
}
fn bench_adjoint_gradient_n256(c: &mut Criterion) {
    bench_adjoint_gradient(c, 256);
}

// ---------------------------------------------------------------------------
// Group 5: ObstacleChernoff<DiffusionChernoff> cost-per-step (v6.3.0)
// ---------------------------------------------------------------------------

/// Per-step cost for `ObstacleChernoff<DiffusionChernoff, ConstantObstacle>`.
///
/// Kernel: heat `a=0.5`, zero-floor obstacle. Measures the overhead of the
/// post-projection `max(·, g)` on top of the base heat step.
fn bench_obstacle_per_step(c: &mut Criterion, n: usize) {
    let grid = Grid1D::new(0.0_f64, 1.0, n)
        .expect("grid")
        .with_boundary(BoundaryPolicy::ZeroExtend);
    let inner = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let obs = ConstantObstacle::new(0.0_f64).expect("obs");
    let kernel = ObstacleChernoff::new(inner, obs).expect("obstacle kernel");
    let ic = GridFn1D::from_fn(grid, |x| (-(x - 0.5).powi(2) / 0.05).exp());
    let mut dst = ic.zeroed_like();
    let mut scratch = ScratchPool::new();

    let mut group = c.benchmark_group(format!("new_kernels4/obstacle_per_step/N={n}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(n as u64));

    group.bench_with_input(BenchmarkId::new("obstacle_chernoff", ""), &(), |b, ()| {
        b.iter(|| {
            kernel
                .apply_into(black_box(TAU), black_box(&ic), &mut dst, &mut scratch)
                .expect("ok");
        });
    });

    group.finish();
}

fn bench_obstacle_n256(c: &mut Criterion) {
    bench_obstacle_per_step(c, 256);
}
fn bench_obstacle_n1024(c: &mut Criterion) {
    bench_obstacle_per_step(c, 1024);
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_magnus6_n64,
    bench_magnus6_n256,
    bench_state_adjoint_n64,
    bench_state_adjoint_n256,
    bench_jvp_n64,
    bench_jvp_n256,
    bench_adjoint_gradient_n64,
    bench_adjoint_gradient_n256,
    bench_obstacle_n256,
    bench_obstacle_n1024,
);
criterion_main!(benches);
