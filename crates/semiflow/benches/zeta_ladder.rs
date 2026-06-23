//! Criterion benchmark: zeta-ladder kernels — cost-per-step throughput.
//!
//! Measures the wall-clock cost of **one Chernoff step** (`apply_chernoff`)
//! for the headline accuracy kernels relative to the order-2 baseline.
//! All kernels operate on the same smooth MMS Gaussian IC on a uniform grid.
//!
//! Kernels benchmarked (and their internal K5 multiplier per outer step):
//!
//! | Kernel                       | Order | K5 calls/step |
//! |------------------------------|-------|---------------|
//! | `DiffusionChernoff`          | 2     | 1             |
//! | `Diffusion4thChernoff`       | 2†    | 1             |
//! | `Diffusion6thChernoff`       | 2†    | 1             |
//! | `Diffusion4thZeta4Chernoff`  | 4     | 3             |
//! | `Diffusion6thZeta6Chernoff`  | 6     | 9             |
//! | `Diffusion8thZeta8Chernoff`  | 8     | 27            |
//! | zeta4 + Chebyshev M=64       | 4     | 3 + spectral  |
//! | zeta6 + Chebyshev M=64       | 6     | 9 + spectral  |
//! | zeta8 + Chebyshev M=64       | 8     | 27 (default)  |
//!
//! † spatial order 4/6; temporal order 2 (no Richardson wrapping).
//!
//! Run:
//! ```sh
//! cargo bench -p semiflow-core --bench zeta_ladder
//! # Quick smoke (1 s warm-up, 2 s measurement, 10 samples):
//! cargo bench -p semiflow-core --bench zeta_ladder -- \
//!     --warm-up-time 1 --measurement-time 2 --sample-size 10
//! ```
//! Results land in `target/criterion/zeta_ladder/`.

#![allow(missing_docs)]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use semiflow::{
    chernoff::ApplyChernoffExt, Diffusion4thChernoff, Diffusion4thZeta4Chernoff,
    Diffusion6thChernoff, Diffusion6thZeta6Chernoff, Diffusion8thZeta8Chernoff, DiffusionChernoff,
    Grid1D, GridFn1D,
};

// ---------------------------------------------------------------------------
// Bench configuration
// ---------------------------------------------------------------------------

/// Time step τ — chosen small enough for all kernels to be stable.
const TAU: f64 = 0.005;

/// Chebyshev collocation node count for the Chebyshev-variant benchmarks.
const CHEB_M: usize = 64;

/// Warm-up duration for bench groups.
const WARM_UP_SECS: u64 = 3;

/// Measurement duration for bench groups.
const MEASUREMENT_SECS: u64 = 10;

// ---------------------------------------------------------------------------
// Smooth diffusion coefficient: a(x) = 1 + 0.5·tanh²(x)
// ---------------------------------------------------------------------------

fn a_fn(x: f64) -> f64 {
    1.0 + 0.5 * x.tanh().powi(2)
}

fn a_prime_fn(x: f64) -> f64 {
    let th = x.tanh();
    th * (1.0 - th * th)
}

fn a_double_prime_fn(x: f64) -> f64 {
    let th = x.tanh();
    let sech2 = 1.0 - th * th;
    sech2 * (1.0 - 3.0 * th * th)
}

/// Upper bound for ‖a‖_∞ on any finite domain (a ≤ 1.5 everywhere).
const A_NORM_BOUND: f64 = 1.5;

// ---------------------------------------------------------------------------
// IC: Gaussian MMS f₀(x) = exp(−x²) on [−10, 10]
// ---------------------------------------------------------------------------

fn make_grid(n: usize) -> Grid1D<f64> {
    Grid1D::new(-10.0, 10.0, n).expect("grid construction")
}

fn make_ic(grid: Grid1D<f64>) -> GridFn1D<f64> {
    GridFn1D::from_fn(grid, |x| (-x * x).exp())
}

// ---------------------------------------------------------------------------
// Kernel constructors — one per variant
// ---------------------------------------------------------------------------

fn make_dc(grid: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::new(a_fn, a_prime_fn, a_double_prime_fn, A_NORM_BOUND, grid)
}

fn make_d4(grid: Grid1D<f64>) -> Diffusion4thChernoff<f64> {
    Diffusion4thChernoff::new(a_fn, a_prime_fn, a_double_prime_fn, A_NORM_BOUND, grid)
}

fn make_d6(grid: Grid1D<f64>) -> Diffusion6thChernoff<f64> {
    Diffusion6thChernoff::new(a_fn, a_prime_fn, a_double_prime_fn, A_NORM_BOUND, grid)
}

fn make_zeta4(grid: Grid1D<f64>) -> Diffusion4thZeta4Chernoff<f64> {
    let k5 = make_d4(grid);
    Diffusion4thZeta4Chernoff::new(k5, Some(A_NORM_BOUND)).expect("zeta4 construction")
}

fn make_zeta4_cheb(grid: Grid1D<f64>) -> Diffusion4thZeta4Chernoff<f64> {
    make_zeta4(grid).with_chebyshev_sampling_m(CHEB_M)
}

fn make_zeta6(grid: Grid1D<f64>) -> Diffusion6thZeta6Chernoff<f64> {
    let zeta4 = make_zeta4(grid);
    Diffusion6thZeta6Chernoff::new(zeta4, Some(A_NORM_BOUND)).expect("zeta6 construction")
}

fn make_zeta6_cheb(grid: Grid1D<f64>) -> Diffusion6thZeta6Chernoff<f64> {
    make_zeta6(grid).with_chebyshev_sampling_m(CHEB_M)
}

fn make_zeta8(grid: Grid1D<f64>) -> Diffusion8thZeta8Chernoff<f64> {
    let zeta6 = make_zeta6(grid);
    // Chebyshev M=64 is ON by default in Diffusion8thZeta8Chernoff::new.
    Diffusion8thZeta8Chernoff::new(zeta6, Some(A_NORM_BOUND)).expect("zeta8 construction")
}

fn make_zeta8_cheb_m(grid: Grid1D<f64>, m: usize) -> Diffusion8thZeta8Chernoff<f64> {
    make_zeta8(grid).with_chebyshev_sampling_m(m)
}

// ---------------------------------------------------------------------------
// Macro-driven bench runner — avoids repeating the group boilerplate
// ---------------------------------------------------------------------------

/// Apply one Chernoff step and return the result (erases type for the closure).
macro_rules! bench_kernel {
    ($group:expr, $id:expr, $kernel:expr, $ic:expr) => {
        $group.bench_with_input(BenchmarkId::new($id, ""), &(), |b, _| {
            let k = $kernel;
            let f = $ic;
            b.iter(|| {
                k.apply_chernoff(black_box(TAU), black_box(&f))
                    .expect("apply_chernoff")
            });
        });
    };
}

// ---------------------------------------------------------------------------
// Benchmark: single Chernoff step at each grid size
// ---------------------------------------------------------------------------

/// Bench group: per-step cost across all kernels at one grid size `n`.
fn bench_per_step(c: &mut Criterion, n: usize) {
    let mut group = c.benchmark_group(format!("zeta_ladder/per_step/N={n}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements(n as u64));

    let grid = make_grid(n);
    let ic = make_ic(grid);

    bench_kernel!(group, "dc_order2", make_dc(grid), ic.clone());
    bench_kernel!(group, "d4_order2sp", make_d4(grid), ic.clone());
    bench_kernel!(group, "d6_order2sp", make_d6(grid), ic.clone());
    bench_kernel!(group, "zeta4_order4", make_zeta4(grid), ic.clone());
    bench_kernel!(group, "zeta6_order6", make_zeta6(grid), ic.clone());
    bench_kernel!(group, "zeta8_order8", make_zeta8(grid), ic.clone());
    bench_kernel!(group, "zeta4_cheb64", make_zeta4_cheb(grid), ic.clone());
    bench_kernel!(group, "zeta6_cheb64", make_zeta6_cheb(grid), ic.clone());
    bench_kernel!(
        group,
        "zeta8_cheb64",
        make_zeta8_cheb_m(grid, CHEB_M),
        ic.clone()
    );

    group.finish();
}

fn bench_per_step_256(c: &mut Criterion) {
    bench_per_step(c, 256);
}

fn bench_per_step_1024(c: &mut Criterion) {
    bench_per_step(c, 1024);
}

fn bench_per_step_4096(c: &mut Criterion) {
    bench_per_step(c, 4096);
}

// ---------------------------------------------------------------------------
// Benchmark: fixed-budget evolve (n = 50 steps) at N = 1024
// ---------------------------------------------------------------------------

/// Helper: evolve `n_steps` Chernoff steps at `tau` without `ChernoffSemigroup`.
///
/// We iterate `apply_chernoff` manually so that the bench works for all kernel
/// types without requiring a uniform `ChernoffSemigroup` wrapper (which would
/// need T as a type parameter).
fn evolve_steps<K>(kernel: &K, ic: &GridFn1D<f64>, n: usize) -> GridFn1D<f64>
where
    K: ApplyChernoffExt<f64, S = GridFn1D<f64>>,
{
    let mut state = ic.clone();
    for _ in 0..n {
        state = kernel
            .apply_chernoff(TAU, &state)
            .expect("apply_chernoff in evolve");
    }
    state
}

/// Number of steps for the fixed-budget evolve bench.
const N_EVOLVE: usize = 50;

fn bench_fixed_budget(c: &mut Criterion) {
    const N: usize = 1024;
    let mut group = c.benchmark_group(format!("zeta_ladder/evolve_{N_EVOLVE}steps/N={N}"));
    group.warm_up_time(Duration::from_secs(WARM_UP_SECS));
    group.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    group.throughput(Throughput::Elements((N * N_EVOLVE) as u64));

    let grid = make_grid(N);
    let ic = make_ic(grid);

    macro_rules! bench_evolve {
        ($id:expr, $kernel:expr) => {
            let k = $kernel;
            let f = ic.clone();
            group.bench_function($id, |b| {
                b.iter(|| evolve_steps(black_box(&k), black_box(&f), N_EVOLVE))
            });
        };
    }

    bench_evolve!("dc_order2", make_dc(grid));
    bench_evolve!("d4_order2sp", make_d4(grid));
    bench_evolve!("d6_order2sp", make_d6(grid));
    bench_evolve!("zeta4_order4", make_zeta4(grid));
    bench_evolve!("zeta6_order6", make_zeta6(grid));
    bench_evolve!("zeta8_order8", make_zeta8(grid));
    bench_evolve!("zeta4_cheb64", make_zeta4_cheb(grid));
    bench_evolve!("zeta6_cheb64", make_zeta6_cheb(grid));
    bench_evolve!("zeta8_cheb64", make_zeta8_cheb_m(grid, CHEB_M));

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_per_step_256,
    bench_per_step_1024,
    bench_per_step_4096,
    bench_fixed_budget,
);
criterion_main!(benches);
