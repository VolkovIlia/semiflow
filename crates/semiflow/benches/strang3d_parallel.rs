//! Criterion benchmark: `Strang3D::apply` thread-scaling sweep.
//!
//! Mirrors `strang2d_parallel.rs` for 3D. Uses the `FORCE_THREADS_3D` test
//! hook (ADR-0024) to pin each measurement to a specific thread count.
//!
//! Thread counts measured: `[1, 2, 4, 8, 16]`.
//!
//! Grid: 120×120×120, `n_chernoff=2`.  A 120³ grid has ~1.7 M points;
//! chosen so single-thread iterations stay under ~3 s on the reference
//! i7-12700K while providing enough data to see thread-scaling.
//! Larger grids (200³) are recorded in `docs/perf/baseline-v2_2_0.json`
//! for reference.
//!
//! Run (quick smoke):
//! ```sh
//! cargo bench -p semiflow-core --features parallel,simd \
//!     --bench strang3d_parallel -- --quick
//! ```
//! Results land in `target/criterion/strang3d_parallel/`.

#![allow(missing_docs, clippy::type_complexity)]

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use semiflow::{
    strang3d_parallel::FORCE_THREADS_3D, ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid3D,
    GridFn3D, Strang3D,
};

/// Thread counts swept in each bench group.
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];

type Semi3D = ChernoffSemigroup<
    Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    GridFn3D<f64>,
>;

/// Build the 120×120×120 semigroup and initial state.
fn make_state() -> (Semi3D, GridFn3D<f64>) {
    let gx = Grid1D::new(-4.0, 4.0, 120).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, 120).unwrap();
    let gz = Grid1D::new(-4.0, 4.0, 120).unwrap();
    let g3 = Grid3D::new(gx, gy, gz).unwrap();
    let f0 = GridFn3D::from_fn(g3, |x, y, z| (-x * x - y * y - z * z).exp());
    let dx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, gx);
    let dy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, gy);
    let dz = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, gz);
    let strang = Strang3D::new(dx, dy, dz);
    let semi = ChernoffSemigroup::new(strang, 2).unwrap();
    (semi, f0)
}

/// Sweep `THREAD_COUNTS` for a pre-built semigroup.
fn thread_sweep(group: &mut BenchmarkGroup<'_, WallTime>, semi: &Semi3D, f0: &GridFn3D<f64>) {
    for &n in THREAD_COUNTS {
        group.bench_function(format!("threads={n}"), |b| {
            FORCE_THREADS_3D.with(|c| c.set(Some(n)));
            b.iter(|| semi.evolve(black_box(1.0), black_box(f0)).expect("evolve"));
            FORCE_THREADS_3D.with(|c| c.set(None));
        });
    }
}

/// Benchmark `Strang3D` parallel scaling on a 120×120×120 grid.
fn bench_strang3d_parallel(c: &mut Criterion) {
    let (semi, f0) = make_state();

    let mut group = c.benchmark_group("strang3d_parallel/N=120/n_chernoff=2");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    thread_sweep(&mut group, &semi, &f0);

    group.finish();
}

criterion_group!(benches, bench_strang3d_parallel);
criterion_main!(benches);
