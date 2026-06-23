// Phase 4 cache-aware A/B bench.
//
// Targets: Strang2D heat N ∈ {400, 800, 1600}, n_chernoff = 4.
//          Strang3D heat N ∈ {50, 100} — 3D is expensive; use small grids.
//
// Replicates: criterion default (100 samples, warm-up 3 s, measurement 5 s).
//
// A/B procedure (per CONTRACT-phase4-ab.md §3):
//   BEFORE:
//     taskset -c 0 cargo bench -p semiflow-core --bench cache_phase4 \
//       --features parallel,simd -- --save-baseline phase4-baseline
//   AFTER applying Candidate B / E:
//     taskset -c 0 cargo bench -p semiflow-core --bench cache_phase4 \
//       --features parallel,simd -- --baseline phase4-baseline
//
// The bench uses `evolve` (= apply_into × n_chernoff) so the steady-state
// zero-alloc ScratchPool path is exercised, not the cold `apply` path.

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use semiflow::{
    ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid2D, Grid3D, GridFn2D, GridFn3D, Strang2D,
    Strang3D,
};

// ---------------------------------------------------------------------------
// 2D helpers
// ---------------------------------------------------------------------------

type Semi2D =
    ChernoffSemigroup<Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>, GridFn2D<f64>>;

fn make_2d(n: usize, n_chernoff: usize) -> (Semi2D, GridFn2D<f64>) {
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

// ---------------------------------------------------------------------------
// 3D helpers
// ---------------------------------------------------------------------------

type Semi3D = ChernoffSemigroup<
    Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>,
    GridFn3D<f64>,
>;

fn make_3d(n: usize, n_chernoff: usize) -> (Semi3D, GridFn3D<f64>) {
    let gx = Grid1D::new(-2.0, 2.0, n).unwrap();
    let gy = Grid1D::new(-2.0, 2.0, n).unwrap();
    let gz = Grid1D::new(-2.0, 2.0, n).unwrap();
    let g3 = Grid3D::new(gx, gy, gz).unwrap();
    let f0 = GridFn3D::from_fn(g3, |x, y, z| (-x * x - y * y - z * z).exp());
    let dx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let dy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);
    let dz = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gz);
    let strang = Strang3D::new(dx, dy, dz);
    let semi = ChernoffSemigroup::new(strang, n_chernoff).unwrap();
    (semi, f0)
}

// ---------------------------------------------------------------------------
// Bench groups
// ---------------------------------------------------------------------------

/// `Strang2D` `apply_into` hot path across N ∈ {400, 800, 1600}.
///
/// `n_chernoff=4`: enough steps to saturate the `ScratchPool` reuse path while
/// keeping per-iteration time bounded at large N.
fn bench_strang2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("strang2d_apply_into");
    // Reduce sample count for large N to keep total bench time ≤ 5 min.
    group.sample_size(50);

    for &n in &[400usize, 800, 1600] {
        let (semi, f0) = make_2d(n, 4);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| semi.evolve(black_box(1.0), black_box(&f0)).expect("evolve"));
        });
    }
    group.finish();
}

/// `Strang3D` `apply_into` hot path across N ∈ {50, 100}.
///
/// 3D is CPU-heavy; N=100 gives a 100³ = 1 M cell grid.
fn bench_strang3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("strang3d_apply_into");
    group.sample_size(20);

    for &n in &[50usize, 100] {
        let (semi, f0) = make_3d(n, 4);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| semi.evolve(black_box(1.0), black_box(&f0)).expect("evolve"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_strang2d, bench_strang3d);
criterion_main!(benches);
