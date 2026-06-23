//! Criterion benchmark: `Diffusion6thChernoff::apply` — SIMD vs scalar (ADR-0019).
//!
//! Measures a single `apply` call at N=1024 with the `simd` feature.
//! Tracked for the v1.0.0 perf commitment (S2.3).
//!
//! Run with SIMD enabled:
//! ```sh
//! RUSTFLAGS="-C target-feature=+avx2" cargo bench -p semiflow-core --features simd --bench diffusion6_simd
//! ```
//! Run scalar baseline:
//! ```sh
//! cargo bench -p semiflow-core --no-default-features --bench diffusion6_simd
//! ```
//! Results land in `target/criterion/diffusion6_simd/`.

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use semiflow::{
    chernoff::ApplyChernoffExt, BoundaryPolicy, Diffusion6thChernoff, Grid1D, GridFn1D,
};

/// Deterministic LCG — no rand dep.
fn lcg_values(n: usize) -> Vec<f64> {
    let mut state: u64 = 0xdead_beef_cafe_babe;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let hi32 = (state >> 32) as u32;
            0.1 + f64::from(hi32) / f64::from(u32::MAX)
        })
        .collect()
}

/// Build the Diffusion6th operator and initial state for N=1024.
fn make_state() -> (Diffusion6thChernoff<f64>, GridFn1D<f64>) {
    let n = 1024_usize;
    let grid = Grid1D::new(-10.0, 10.0, n)
        .expect("grid")
        .with_boundary(BoundaryPolicy::Reflect);
    let dc = Diffusion6thChernoff::new(
        |x| 0.5 + 0.1 * x.sin(),
        |x| 0.1 * x.cos(),
        |x| -0.1 * x.sin(),
        0.6,
        grid,
    );
    let values = lcg_values(n);
    let f0 = GridFn1D::new(grid, values).expect("f0");
    (dc, f0)
}

/// Benchmark `Diffusion6thChernoff::apply` at N=1024, tau=0.01.
fn bench_diffusion6(c: &mut Criterion) {
    let (dc, f0) = make_state();
    let tau = 0.01_f64;
    c.bench_function("diffusion6_simd/N=1024", |b| {
        b.iter(|| {
            dc.apply_chernoff(black_box(tau), black_box(&f0))
                .expect("apply")
        });
    });
}

criterion_group!(benches, bench_diffusion6);
criterion_main!(benches);
