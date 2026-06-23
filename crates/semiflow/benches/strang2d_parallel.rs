//! Criterion benchmark: `Strang2D::apply` thread-scaling sweep.
//!
//! Measures how `Strang2D` performance scales with thread count when
//! `--features parallel` is enabled.  Uses the `FORCE_THREADS` test hook
//! (ADR-0018) to pin each measurement to a specific thread count.
//!
//! Thread counts measured: `[1, 2, 4, 8, 16]`.  On machines with fewer
//! physical cores the OS over-subscribes; efficiency numbers will be
//! artificially low — always document hardware in `docs/perf/baseline-v2_2_0.json`.
//!
//! Grid: 800×800, `n_chernoff=4` (keeps each criterion iteration ≤ ~2 s on
//! the reference i7-12700K; enough steps to amortise thread-spawn overhead).
//!
//! Run (quick smoke):
//! ```sh
//! cargo bench -p semiflow-core --features parallel,simd \
//!     --bench strang2d_parallel -- --quick
//! ```
//! Results land in `target/criterion/strang2d_parallel/`.

#![allow(missing_docs, clippy::type_complexity)]

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use semiflow::{
    strang2d_parallel::FORCE_THREADS, ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid2D,
    GridFn2D, Strang2D,
};

/// Thread counts swept in each bench group.
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];

type Semi2D =
    ChernoffSemigroup<Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>, GridFn2D<f64>>;

/// Build the 800×800 semigroup and initial state.
fn make_state() -> (Semi2D, GridFn2D<f64>) {
    let gx = Grid1D::new(-4.0, 4.0, 800).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, 800).unwrap();
    let g2 = Grid2D::new(gx, gy);
    let f0 = GridFn2D::from_fn(g2, |x, y| (-x * x - y * y).exp());
    let dx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, gx);
    let dy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, gy);
    let strang = Strang2D::new(dx, dy);
    let semi = ChernoffSemigroup::new(strang, 4).unwrap();
    (semi, f0)
}

/// Sweep `THREAD_COUNTS` for a pre-built semigroup.
fn thread_sweep(group: &mut BenchmarkGroup<'_, WallTime>, semi: &Semi2D, f0: &GridFn2D<f64>) {
    for &n in THREAD_COUNTS {
        group.bench_function(format!("threads={n}"), |b| {
            FORCE_THREADS.with(|c| c.set(Some(n)));
            b.iter(|| semi.evolve(black_box(1.0), black_box(f0)).expect("evolve"));
            FORCE_THREADS.with(|c| c.set(None));
        });
    }
}

/// Benchmark `Strang2D` parallel scaling on an 800×800 grid.
fn bench_strang2d_parallel(c: &mut Criterion) {
    let (semi, f0) = make_state();

    let mut group = c.benchmark_group("strang2d_parallel/N=800/n_chernoff=4");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    thread_sweep(&mut group, &semi, &f0);

    group.finish();
}

criterion_group!(benches, bench_strang2d_parallel);
criterion_main!(benches);
