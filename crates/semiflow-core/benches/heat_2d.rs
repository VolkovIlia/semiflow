//! Criterion benchmark: 2D isotropic heat equation via `Strang2D`.
//!
//! Tracks a single representative configuration (N=400×400, `n_chernoff=10`)
//! for the v1.0.0 perf commitment (S2.3).  Larger grids (800, 1600) are
//! covered by the `docs/perf-baseline-v0_7_0.md` historical record and the
//! `docs/perf-commitment-v1_0_0.md` methodology note.
//!
//! Run:
//! ```sh
//! cargo bench -p semiflow-core --bench heat_2d
//! ```
//! Results land in `target/criterion/heat_2d/`.

#![allow(missing_docs, clippy::type_complexity)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use semiflow_core::{ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid2D, GridFn2D, Strang2D};

type Semi2D =
    ChernoffSemigroup<Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>, GridFn2D<f64>>;

/// Build the semigroup and initial state for N×N at `n_chernoff` steps.
fn make_state(n: usize, n_chernoff: usize) -> (Semi2D, GridFn2D<f64>) {
    let gx = Grid1D::new(-4.0, 4.0, n).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, n).unwrap();
    let g2 = Grid2D::new(gx, gy);
    let f0 = GridFn2D::from_fn(g2, |x, y| (-x * x - y * y).exp());
    let dx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let dy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);
    let strang = Strang2D::new(dx, dy);
    let semi = ChernoffSemigroup::new(strang, n_chernoff).unwrap();
    (semi, f0)
}

/// Benchmark `Strang2D` evolve at N=400×400, `n_chernoff=10`, T=1.0.
///
/// `n_chernoff=10` keeps each criterion iteration under ~2 s on the reference
/// hardware (i7-12700K); the historical baseline at `n_chernoff=50` is recorded
/// in `docs/perf-baseline-v0_7_0.md`.
fn bench_heat_2d_400(c: &mut Criterion) {
    let (semi, f0) = make_state(400, 10);
    c.bench_function("heat_2d/N=400/n_chernoff=10", |b| {
        b.iter(|| semi.evolve(black_box(1.0), black_box(&f0)).expect("evolve"));
    });
}

criterion_group!(benches, bench_heat_2d_400);
criterion_main!(benches);
