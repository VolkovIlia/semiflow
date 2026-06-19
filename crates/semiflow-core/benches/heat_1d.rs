//! Criterion benchmark: Gaussian heat-kernel (1D, unit-a diffusion).
//!
//! Exercises `ShiftChernoff1D::evolve` at n=100 Chernoff steps, N=1000 grid points.
//! Tracked benchmark for the v1.0.0 perf commitment (S2.3).
//!
//! Run:
//! ```sh
//! cargo bench -p semiflow-core --bench heat_1d
//! ```
//! Results land in `target/criterion/heat_1d/`.

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use semiflow_core::{ChernoffSemigroup, Grid1D, GridFn1D, ShiftChernoff1D};

/// Build the semigroup and initial state for the benchmark.
fn make_state() -> (
    ChernoffSemigroup<ShiftChernoff1D<f64>, GridFn1D<f64>>,
    GridFn1D<f64>,
) {
    let grid = Grid1D::new(-10.0, 10.0, 1000).unwrap();
    let f0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let cher = ShiftChernoff1D::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.0, grid);
    let semi = ChernoffSemigroup::new(cher, 100).unwrap();
    (semi, f0)
}

/// Benchmark `ChernoffSemigroup::evolve` at n=100 steps, N=1000 grid points, T=1.0.
fn bench_heat_1d(c: &mut Criterion) {
    let (semi, f0) = make_state();
    c.bench_function("heat_1d/n=100/N=1000", |b| {
        b.iter(|| semi.evolve(black_box(1.0), black_box(&f0)).expect("evolve"));
    });
}

criterion_group!(benches, bench_heat_1d);
criterion_main!(benches);
