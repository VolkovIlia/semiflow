//! Criterion benchmark: lifted/geometric kernels — cost-per-step throughput.
//!
//! Covers the v2.7–v2.8 nonautonomous-lift and Riemannian-manifold family:
//!
//! | Kernel                     | State type            | Order |
//! |----------------------------|-----------------------|-------|
//! | `HowlandLift` `n_t=8`        | `HowlandState`<GridFn1D>| 1     |
//! | `HowlandLift` `n_t=32`       | `HowlandState`<GridFn1D>| 1     |
//! | `ManifoldChernoff<Sphere2>`     (no correction)  | `GridFn2D` | 1  |
//! | `ManifoldChernoff<Sphere2>`     (R/12 correction)| `GridFn2D` | 2  |
//! | `ManifoldChernoff<Hyperbolic2>` (no correction)  | `GridFn2D` | 1  |
//! | `ManifoldChernoff<Hyperbolic2>` (R/12 correction)| `GridFn2D` | 2  |
//!
//! Note on `HowlandLift` cost: one `apply_into` triggers `(n_t - 1)` inner
//! diffusion steps. The throughput metric is `n_t × N_grid` to reflect total `DoF`.
//!
//! Note on manifold grids: small 16×32 charts are used here (the expensive
//! GH-5 ⊗ GH-5 quadrature is O(25) evaluations per node). Larger charts are
//! benched by the slow-tests suite; not repeated here.
//!
//! Run:
//! ```sh
//! cargo bench -p semiflow-core --bench new_kernels2
//! # Quick smoke:
//! cargo bench -p semiflow-core --bench new_kernels2 -- \
//!     --warm-up-time 1 --measurement-time 2 --sample-size 10
//! ```
//! Results land in `target/criterion/new_kernels2/`.

#![allow(missing_docs)]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use semiflow_core::{
    howland::{HowlandLift, HowlandState},
    ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D, GridFn1D, GridFn2D, Hyperbolic2,
    ManifoldChernoff, ScratchPool, Sphere2,
};

// ---------------------------------------------------------------------------
// Bench configuration
// ---------------------------------------------------------------------------

/// Time step τ for manifold benchmarks (matches the G26 gate regime).
const TAU_MANIFOLD: f64 = 0.00125;
/// Warm-up duration.
const WARM_UP_SECS: u64 = 3;
/// Measurement duration.
const MEASUREMENT_SECS: u64 = 10;
/// Howland grid N (1D spatial).
const HOWLAND_N_GRID: usize = 64;
/// Howland time-slice counts to bench.
const HOWLAND_N_T_SMALL: usize = 8;
const HOWLAND_N_T_LARGE: usize = 32;
/// Time horizon for `HowlandLift`.
const HOWLAND_T_HORIZON: f64 = 0.5;
/// Manifold chart grid sizes (`n_theta` × `n_phi` = `N_CHART` × `2×N_CHART`).
const N_CHART: usize = 16;

// ---------------------------------------------------------------------------
// Howland benchmark helpers
// ---------------------------------------------------------------------------

/// Build a `HowlandLift` over `DiffusionChernoff` (autonomous bridge via marker impl).
fn make_howland_state(
    n_t: usize,
) -> (
    HowlandLift<DiffusionChernoff<f64>>,
    HowlandState<GridFn1D<f64>>,
) {
    let grid = Grid1D::new(-8.0_f64, 8.0, HOWLAND_N_GRID).expect("grid");
    let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.5, grid);
    let lift = HowlandLift::new(inner, HOWLAND_T_HORIZON, n_t).expect("howland lift");
    let ic = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let samples: Vec<GridFn1D<f64>> = (0..n_t).map(|_| ic.clone()).collect();
    let state = HowlandState::new(samples).expect("howland state");
    (lift, state)
}

/// One `HowlandLift::apply_into` call (costs `n_t - 1` inner diffusion steps).
fn bench_howland_kernels(c: &mut Criterion) {
    let n = HOWLAND_N_GRID;
    let mut group = c.benchmark_group("new_kernels2/howland_per_lift");
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    for &n_t in &[HOWLAND_N_T_SMALL, HOWLAND_N_T_LARGE] {
        group.throughput(Throughput::Elements((n_t * n) as u64));
        let (lift, src) = make_howland_state(n_t);
        let mut dst = src.clone();
        let mut scratch = ScratchPool::new();
        let delta_s = lift.delta_s();
        group.bench_with_input(
            BenchmarkId::new(format!("howland_n_t={n_t}"), ""),
            &(),
            |b, ()| {
                b.iter(|| {
                    lift.apply_into(
                        black_box(delta_s),
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(&mut scratch),
                    )
                    .expect("ok");
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Manifold benchmark helpers
// ---------------------------------------------------------------------------

fn sphere_chart_grid() -> Grid2D<f64> {
    let eps = 0.02_f64;
    let g_theta = Grid1D::new(eps, core::f64::consts::PI - eps, N_CHART).expect("theta grid");
    let g_phi = Grid1D::new(0.0, 2.0 * core::f64::consts::PI, 2 * N_CHART).expect("phi grid");
    Grid2D::new(g_theta, g_phi)
}

/// Poincaré-disk chart: both coords in (-0.9, 0.9) to stay inside |x|<1.
fn hyperbolic_chart_grid() -> Grid2D<f64> {
    let g_x = Grid1D::new(-0.85_f64, 0.85, N_CHART).expect("x grid");
    let g_y = Grid1D::new(-0.85_f64, 0.85, 2 * N_CHART).expect("y grid");
    Grid2D::new(g_x, g_y)
}

fn bench_sphere_cases(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    let grid = sphere_chart_grid();
    let u = GridFn2D::from_fn(grid, |theta, _phi| theta.cos());
    for (name, correction) in [("sphere2_base", false), ("sphere2_r12", true)] {
        let kernel = ManifoldChernoff::new(Sphere2::unit(), correction);
        let mut dst = u.clone();
        let mut scratch = ScratchPool::new();
        group.bench_with_input(BenchmarkId::new(name, ""), &(), |b, ()| {
            b.iter(|| {
                kernel
                    .apply_into(
                        black_box(TAU_MANIFOLD),
                        black_box(&u),
                        black_box(&mut dst),
                        black_box(&mut scratch),
                    )
                    .expect("ok");
            });
        });
    }
}

fn bench_hyperbolic_cases(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let grid = hyperbolic_chart_grid();
    let u = GridFn2D::from_fn(grid, |x, y| (-(x * x + y * y)).exp());
    for (name, correction) in [("hyperbolic2_base", false), ("hyperbolic2_r12", true)] {
        let kernel = ManifoldChernoff::new(Hyperbolic2::unit(), correction);
        let mut dst = u.clone();
        let mut scratch = ScratchPool::new();
        group.bench_with_input(BenchmarkId::new(name, ""), &(), |b, ()| {
            b.iter(|| {
                kernel
                    .apply_into(
                        black_box(TAU_MANIFOLD),
                        black_box(&u),
                        black_box(&mut dst),
                        black_box(&mut scratch),
                    )
                    .expect("ok");
            });
        });
    }
}

fn bench_manifold_kernels(c: &mut Criterion) {
    let n_nodes = N_CHART * (2 * N_CHART);
    let mut group = c.benchmark_group(format!(
        "new_kernels2/manifold_per_step/N={N_CHART}x{}",
        2 * N_CHART
    ));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(n_nodes as u64));
    bench_sphere_cases(&mut group);
    bench_hyperbolic_cases(&mut group);
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

criterion_group!(benches, bench_howland_kernels, bench_manifold_kernels);
criterion_main!(benches);
