//! Informational SIMD speedup gate (ADR-0019).
//!
//! Measures the actual wall-clock ratio SIMD / scalar for
//! `Diffusion6thChernoff::apply` (fd9 hot path) at N=1024.
//!
//! Gate: `SIMD_HERMITE_SPEEDUP` in `contracts/semiflow-core.properties.yaml`.
//! Classification: **INFORMATIONAL** (non-blocking).
//!
//! The test PASSES regardless of speedup — it only prints the ratio so
//! it can be captured in CI logs and compared to the perf-baseline doc.
//! A speedup < 2.0× means Phase-3 cubic-Hermite vectorization is warranted.
//!
//! See also `benches/diffusion6_simd.rs` for standalone timing.

#![cfg(feature = "simd")]

use std::time::Instant;

use semiflow::{
    chernoff::ApplyChernoffExt, simd::with_force_scalar, BoundaryPolicy, Diffusion6thChernoff,
    Grid1D, GridFn1D,
};

/// LCG — deterministic, no rand dep.
fn lcg_values(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
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

fn measure_avg(
    force_scalar: bool,
    dc: &Diffusion6thChernoff,
    f0: &GridFn1D,
    tau: f64,
    n_iter: u32,
) -> std::time::Duration {
    // Warmup.
    if force_scalar {
        with_force_scalar(|| dc.apply_chernoff(tau, f0).expect("warmup"));
    } else {
        dc.apply_chernoff(tau, f0).expect("warmup");
    }

    let start = Instant::now();
    for _ in 0..n_iter {
        if force_scalar {
            std::hint::black_box(with_force_scalar(|| {
                dc.apply_chernoff(tau, f0).expect("apply")
            }));
        } else {
            std::hint::black_box(dc.apply_chernoff(tau, f0).expect("apply"));
        }
    }
    start.elapsed() / n_iter
}

/// Informational speedup measurement — always passes.
///
/// Prints SIMD vs scalar timing ratio for `Diffusion6thChernoff::apply`
/// at N=1024 with non-constant `a(x)` (exercises both fd9 and quintic-Hermite).
#[test]
fn diffusion6_simd_speedup_informational() {
    let n = 1024_usize;
    let n_iter = 20_u32;
    let tau = 0.01_f64;

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
    let values = lcg_values(n, 0xcafe_babe_dead_beef);
    let f0 = GridFn1D::new(grid, values).expect("f0");

    let t_scalar = measure_avg(true, &dc, &f0, tau, n_iter);
    let t_simd = measure_avg(false, &dc, &f0, tau, n_iter);

    let ratio = t_scalar.as_secs_f64() / t_simd.as_secs_f64();
    println!(
        "[SIMD_HERMITE_SPEEDUP] N={n}  scalar={t_scalar:?}  simd={t_simd:?}  \
         speedup={ratio:.2}x  (informational — gate PASSES always)"
    );

    // Always passes — informational only.
    assert!(
        ratio > 0.0,
        "ratio must be positive (measurement error otherwise)"
    );
}
