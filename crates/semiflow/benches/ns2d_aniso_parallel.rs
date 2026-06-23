//! Criterion benchmark: `NonSeparableMixedChernoff` (anisotropic 2D) thread-scaling.
//!
//! Measures the palindromic 5-leg Chernoff product for anisotropic non-separable
//! 2D diffusion (`NonSeparable2DAnisotropicChernoff` / `NonSeparableMixedChernoff`).
//!
//! Parallel path: the inner `Strang2D` sub-steps use `FORCE_THREADS` (ADR-0018).
//! The mixed-derivative leg `Φ_M(τ)` remains sequential; speedup curve is
//! partially saturated by that serial fraction (Amdahl's law).
//!
//! Thread counts measured: `[1, 2, 4, 8, 16]`.
//!
//! Grid: 400×400, `n_chernoff=4`, constant coupling `β(x,y) = 0.1`.
//! CFL: `4 · τ · 0.1 < dx · dy ≈ 0.0004` → `τ < 0.001`.
//! Each iter uses `T = 0.003` so `τ = T/4 = 0.00075` satisfies the CFL.
//!
//! Run (quick smoke):
//! ```sh
//! cargo bench -p semiflow-core --features parallel,simd \
//!     --bench ns2d_aniso_parallel -- --quick
//! ```
//! Results land in `target/criterion/ns2d_aniso_parallel/`.

#![allow(missing_docs, clippy::type_complexity)]

use std::time::Duration;

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use semiflow::{
    strang2d_parallel::FORCE_THREADS, ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid2D,
    GridFn2D, NonSeparable2DAnisotropicChernoff,
};

/// Thread counts swept in each bench group.
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];

type NS2D = NonSeparable2DAnisotropicChernoff<DiffusionChernoff<f64>, DiffusionChernoff<f64>>;
type Semi2D = ChernoffSemigroup<NS2D, GridFn2D<f64>>;

/// Build the 400×400 non-separable semigroup and initial state.
///
/// Uses `β(x,y) = 0.1` (constant anisotropic coupling).
/// CFL: `4 · τ · |β|_∞ < dx · dy`. With dx = dy = 8/400 = 0.02,
/// the constraint is `4 · τ · 0.1 < 0.0004` → `τ < 0.001`.
/// We use `n_chernoff=4` steps over `T = 0.003` (τ = 0.00075 < 0.001).
fn make_state() -> (Semi2D, GridFn2D<f64>) {
    let gx = Grid1D::new(-4.0, 4.0, 400).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, 400).unwrap();
    let g2 = Grid2D::new(gx, gy);
    let f0 = GridFn2D::from_fn(g2, |x, y| (-x * x - y * y).exp());
    let dx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, gx);
    let dy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, gy);
    // Constant anisotropic coupling β(x,y) = 0.1; norm bound 0.1.
    let ns = NS2D::with_beta(dx, dy, |_x, _y| 0.1_f64, 0.1, g2).unwrap();
    let semi = ChernoffSemigroup::new(ns, 4).unwrap();
    (semi, f0)
}

/// Sweep `THREAD_COUNTS` for a pre-built semigroup.
///
/// T=0.003 gives τ = T/4 = 0.00075 which satisfies `4·τ·0.1 < dx·dy ≈ 0.0004`.
fn thread_sweep(group: &mut BenchmarkGroup<'_, WallTime>, semi: &Semi2D, f0: &GridFn2D<f64>) {
    for &n in THREAD_COUNTS {
        group.bench_function(format!("threads={n}"), |b| {
            FORCE_THREADS.with(|c| c.set(Some(n)));
            b.iter(|| {
                semi.evolve(black_box(0.003), black_box(f0))
                    .expect("evolve")
            });
            FORCE_THREADS.with(|c| c.set(None));
        });
    }
}

/// Benchmark `NonSeparable2DAnisotropicChernoff` parallel scaling on 400×400.
fn bench_ns2d_aniso_parallel(c: &mut Criterion) {
    let (semi, f0) = make_state();

    let mut group = c.benchmark_group("ns2d_aniso_parallel/N=400/n_chernoff=4");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    thread_sweep(&mut group, &semi, &f0);

    group.finish();
}

criterion_group!(benches, bench_ns2d_aniso_parallel);
criterion_main!(benches);
