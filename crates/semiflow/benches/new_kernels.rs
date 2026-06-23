//! Criterion benchmark: boundary-condition kernels — cost-per-step throughput.
//!
//! Covers the v2.6–v4.6 boundary/absorbing/reflecting/resolvent family:
//!
//! | Kernel                  | BC type  | Order |
//! |-------------------------|----------|-------|
//! | `KillingChernoff`       | Dirichlet| 1     |
//! | `ReflectedHeatChernoff` | Neumann  | 2     |
//! | `RobinHeatChernoff`     | Robin    | 1     |
//! | `LaplaceChernoffResolvent` n=64  | resolvent | — |
//! | `LaplaceChernoffResolvent` n=256 | resolvent | — |
//!
//! Run:
//! ```sh
//! cargo bench -p semiflow-core --bench new_kernels
//! # Quick smoke:
//! cargo bench -p semiflow-core --bench new_kernels -- \
//!     --warm-up-time 1 --measurement-time 2 --sample-size 10
//! ```
//! Results land in `target/criterion/new_kernels/`.

#![allow(missing_docs)]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use semiflow::{
    chernoff::ApplyChernoffExt,
    killing::{BoxRegion, KillingChernoff},
    reflection::{HalfSpaceRegion, ReflectedHeatChernoff},
    resolvent::{LaplaceChernoffResolvent, LaplaceQuadrature},
    robin::{HalfSpaceRobin, RobinHeatChernoff},
    BoundaryPolicy, DiffusionChernoff, Grid1D, GridFn1D, InterpKind,
};

// ---------------------------------------------------------------------------
// Bench configuration
// ---------------------------------------------------------------------------

/// Time step τ for the per-step benchmarks.
const TAU: f64 = 0.001;
/// Warm-up duration for bench groups.
const WARM_UP_SECS: u64 = 3;
/// Measurement duration for bench groups.
const MEASUREMENT_SECS: u64 = 10;
/// Resolvent eval truncation levels.
const RESOLVENT_N_SMALL: usize = 64;
const RESOLVENT_N_LARGE: usize = 256;
/// Resolvent λ parameter.
const RESOLVENT_LAMBDA: f64 = 1.0;

// ---------------------------------------------------------------------------
// Grid and IC construction
// ---------------------------------------------------------------------------

fn make_grid_halfline(n: usize) -> Grid1D<f64> {
    // Half-line [0, 10] for Neumann/Robin; all nodes inside R=[0,∞).
    Grid1D::new(0.0_f64, 10.0, n).expect("grid")
}

fn make_grid_full(n: usize) -> Grid1D<f64> {
    // Full line [-5, 5] for resolvent (Reflect BC suits GL32 quadrature).
    Grid1D::new(-5.0_f64, 5.0, n).expect("grid")
}

fn make_grid_killing(n: usize) -> Grid1D<f64> {
    // [0, 1] with ZeroExtend BC for Dirichlet killing.
    Grid1D::new(0.0_f64, 1.0, n)
        .expect("grid")
        .with_boundary(BoundaryPolicy::ZeroExtend)
}

fn gaussian_ic(grid: Grid1D<f64>) -> GridFn1D<f64> {
    GridFn1D::from_fn(grid, |x| {
        let c = (grid.xmin + grid.xmax) * 0.5;
        (-(x - c).powi(2)).exp()
    })
}

// ---------------------------------------------------------------------------
// Kernel constructors
// ---------------------------------------------------------------------------

fn make_killing(
    n: usize,
) -> (
    KillingChernoff<DiffusionChernoff<f64>, BoxRegion<f64, 1>>,
    GridFn1D<f64>,
) {
    let grid = make_grid_killing(n);
    let inner = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let region = BoxRegion::<f64, 1>::new([0.0], [1.0]).expect("box region");
    let kernel = KillingChernoff::new(inner, region).expect("killing");
    let ic = gaussian_ic(grid);
    (kernel, ic)
}

fn make_reflected(
    n: usize,
) -> (
    ReflectedHeatChernoff<DiffusionChernoff<f64>, HalfSpaceRegion<f64, 1>>,
    GridFn1D<f64>,
) {
    let grid = make_grid_halfline(n);
    let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let region = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).expect("half-space");
    let kernel = ReflectedHeatChernoff::new(inner, region).expect("reflected");
    let ic = GridFn1D::from_fn(grid, |x| (-(x - 2.0).powi(2)).exp());
    (kernel, ic)
}

fn make_robin(
    n: usize,
) -> (
    RobinHeatChernoff<DiffusionChernoff<f64>, HalfSpaceRobin<f64, 1>>,
    GridFn1D<f64>,
) {
    // CubicHermite: required because reflect_in_place calls sample_generic()
    // (the generic-F path), which does not support the f64-only SepticHermite.
    let grid = Grid1D::new(0.0_f64, 10.0, n)
        .expect("grid")
        .with_interp(InterpKind::CubicHermite);
    let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let region = HalfSpaceRobin::<f64, 1>::new([0.0], [1.0], 1.0, 1.0).expect("robin region");
    let kernel = RobinHeatChernoff::new(inner, region).expect("robin");
    let ic = GridFn1D::from_fn(grid, |x| (-(x - 2.0).powi(2)).exp());
    (kernel, ic)
}

fn make_resolvent(
    n: usize,
    chernoff_n: usize,
) -> (
    LaplaceChernoffResolvent<DiffusionChernoff<f64>>,
    GridFn1D<f64>,
) {
    let grid = make_grid_full(n);
    let diff = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let resolvent =
        LaplaceChernoffResolvent::new(diff, chernoff_n, LaplaceQuadrature::GaussLaguerre32)
            .expect("resolvent");
    let g = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    (resolvent, g)
}

// ---------------------------------------------------------------------------
// Bench groups
// ---------------------------------------------------------------------------

/// Per-step throughput for Killing/Reflected/Robin at grid size `n`.
fn bench_bc_kernels(c: &mut Criterion, n: usize) {
    let mut group = c.benchmark_group(format!("new_kernels/bc_per_step/N={n}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(n as u64));

    {
        let (kernel, ic) = make_killing(n);
        group.bench_with_input(BenchmarkId::new("killing_dirichlet", ""), &(), |b, ()| {
            b.iter(|| {
                kernel
                    .apply_chernoff(black_box(TAU), black_box(&ic))
                    .expect("ok")
            });
        });
    }

    {
        let (kernel, ic) = make_reflected(n);
        group.bench_with_input(BenchmarkId::new("reflected_neumann", ""), &(), |b, ()| {
            b.iter(|| {
                kernel
                    .apply_chernoff(black_box(TAU), black_box(&ic))
                    .expect("ok")
            });
        });
    }

    {
        let (kernel, ic) = make_robin(n);
        group.bench_with_input(BenchmarkId::new("robin_mixed", ""), &(), |b, ()| {
            b.iter(|| {
                kernel
                    .apply_chernoff(black_box(TAU), black_box(&ic))
                    .expect("ok")
            });
        });
    }

    group.finish();
}

/// `LaplaceChernoffResolvent::eval` at two truncation levels.
fn bench_resolvent(c: &mut Criterion, n: usize) {
    let mut group = c.benchmark_group(format!("new_kernels/resolvent/N={n}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(n as u64));

    {
        let (res, g) = make_resolvent(n, RESOLVENT_N_SMALL);
        group.bench_with_input(
            BenchmarkId::new(format!("gl32_n={RESOLVENT_N_SMALL}"), ""),
            &(),
            |b, ()| {
                b.iter(|| {
                    res.eval(black_box(RESOLVENT_LAMBDA), black_box(&g))
                        .expect("ok")
                });
            },
        );
    }

    {
        let (res, g) = make_resolvent(n, RESOLVENT_N_LARGE);
        group.bench_with_input(
            BenchmarkId::new(format!("gl32_n={RESOLVENT_N_LARGE}"), ""),
            &(),
            |b, ()| {
                b.iter(|| {
                    res.eval(black_box(RESOLVENT_LAMBDA), black_box(&g))
                        .expect("ok")
                });
            },
        );
    }

    group.finish();
}

fn bench_bc_256(c: &mut Criterion) {
    bench_bc_kernels(c, 256);
}

fn bench_bc_1024(c: &mut Criterion) {
    bench_bc_kernels(c, 1024);
}

fn bench_resolvent_64(c: &mut Criterion) {
    bench_resolvent(c, 64);
}

fn bench_resolvent_256(c: &mut Criterion) {
    bench_resolvent(c, 256);
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_bc_256,
    bench_bc_1024,
    bench_resolvent_64,
    bench_resolvent_256,
);
criterion_main!(benches);
