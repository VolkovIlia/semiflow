//! Criterion benchmark for `Heat1D::evolve` (ADR-0031 performance budget).
//!
//! Measures single-thread `evolve` throughput at persona-P2 scale:
//! `n_grid = 1000`, `n_steps = 100`, `t = 1.0` (mirror of the smoke-test
//! parameters).  The ≤2 % overhead budget vs v0.10.0 is gated in CI via
//! `cargo run -p xtask -- py-bench`.
//!
//! Run locally:
//! ```sh
//! cargo bench -p semiflow-py --profile release-ffi
//! ```
//! Results land in `target/criterion/Heat1D_evolve/`.

#![allow(missing_docs, clippy::cast_precision_loss)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use semiflow::{ChernoffSemigroup, DiffusionChernoff, Grid1D, GridFn1D};

// ---------------------------------------------------------------------------
// Shared helpers (no Python layer — benchmark pure Rust path)
// ---------------------------------------------------------------------------

fn unit_a(_: f64) -> f64 {
    1.0
}
fn zero_deriv(_: f64) -> f64 {
    0.0
}

/// Build the semigroup + initial state used in the benchmark.
fn make_state(
    n: usize,
    n_steps: usize,
) -> (
    ChernoffSemigroup<DiffusionChernoff<f64>, GridFn1D<f64>>,
    GridFn1D<f64>,
) {
    let grid = Grid1D::new(-10.0, 10.0, n).expect("grid");
    let chernoff = DiffusionChernoff::new(unit_a, zero_deriv, zero_deriv, 1.0, grid);
    let sg = ChernoffSemigroup::new(chernoff, n_steps).expect("semigroup");
    let u0: Vec<f64> = (0..n)
        .map(|i| {
            let x = -10.0 + 20.0 * (i as f64) / ((n - 1) as f64);
            (-x * x).exp()
        })
        .collect();
    let f = GridFn1D::new(grid, u0).expect("gridfn");
    (sg, f)
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

/// Benchmark `ChernoffSemigroup::evolve` at `n=1000`, `n_steps=100`, `t=1.0`.
///
/// Exercises the pure-Rust kernel that runs inside `py.detach`
/// in `Heat1D::evolve`.  The `PyO3` overhead (GIL acquire/release, Vec clone)
/// is not captured here; it is measured by the integration smoke run.
fn bench_evolve_n1000(c: &mut Criterion) {
    let (sg, f) = make_state(1000, 100);
    c.bench_function("Heat1D_evolve/n=1000/n_steps=100", |b| {
        b.iter(|| sg.evolve(black_box(1.0), black_box(&f)).expect("evolve"));
    });
}

criterion_group!(benches, bench_evolve_n1000);
criterion_main!(benches);
