//! Bit-exact equality gate for SIMD hot paths (ADR-0019).
//!
//! Verifies that the SIMD-accelerated paths produce **byte-for-byte identical**
//! output to the scalar reference for every combination of:
//!
//! - `N ∈ {64, 256, 1024, 4096}` (grid nodes),
//! - `hot_path ∈ {diffusion6_apply, septic_hermite_sample}`,
//! - `boundary ∈ {Reflect, ZeroExtend, Periodic, LinearExtrapolate}`.
//!
//! Note: `quintic_hermite_sample` hot path removed at v7.0 (ADR-0109 removal clock).
//!
//! No tolerance — `Vec<f64>` `PartialEq` is byte-exact (covers signed-zero,
//! subnormals, NaN bit patterns).
//!
//! On failure: prints first divergent index, scalar bits (hex), simd bits (hex),
//! and ULP gap.
//!
//! Gate: `SIMD_BIT_EQUAL` in `contracts/semiflow-core.properties.yaml`.
//! Classification: **RELEASE-BLOCKING**.
//!
//! See `docs/adr/0019-simd-intrinsics.md`.

#![cfg(feature = "simd")]
// v7.0: QuinticHermite SIMD test removed (ADR-0109 removal clock fulfilled).

use semiflow::{
    chernoff::ApplyChernoffExt, simd::with_force_scalar, BoundaryPolicy, Diffusion6thChernoff,
    Grid1D, GridFn1D, InterpKind,
};

// ---------------------------------------------------------------------------
// Failure reporter — byte-level diagnostics on divergence.
// ---------------------------------------------------------------------------

fn assert_bit_equal_vecs(scalar: &[f64], simd: &[f64], label: &str) {
    assert_eq!(
        scalar.len(),
        simd.len(),
        "{label}: length mismatch {} vs {}",
        scalar.len(),
        simd.len()
    );
    for (k, (&s, &v)) in scalar.iter().zip(simd.iter()).enumerate() {
        if s.to_bits() != v.to_bits() {
            // Signed ULP distance via bitwise reinterpretation (no lossy cast).
            let sb = i64::from_ne_bytes(s.to_bits().to_ne_bytes());
            let vb = i64::from_ne_bytes(v.to_bits().to_ne_bytes());
            let ulp = sb.wrapping_sub(vb).unsigned_abs();
            panic!(
                "{label}: diverged at index {k}: \
                 scalar={s:.17e} (0x{:016x}), simd={v:.17e} (0x{:016x}), ULP gap={ulp}",
                s.to_bits(),
                v.to_bits(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic value generator — fixed LCG, no random deps.
// ---------------------------------------------------------------------------

/// Generate `n` f64 values in [0.1, 1.1) using a fixed LCG.
fn lcg_values(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // High 32 bits give enough entropy for test purposes.
            // Use u32→f64 cast (lossless) divided by u32::MAX+1.
            let hi32 = (state >> 32) as u32;
            let mantissa = f64::from(hi32) / f64::from(u32::MAX);
            0.1 + mantissa
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hot path 1: `Diffusion6thChernoff::apply` — exercises fd9 SIMD path.
// ---------------------------------------------------------------------------

fn check_diffusion6_bit_equal(n: usize, bnd: BoundaryPolicy) {
    let label = format!("diffusion6_apply N={n} bnd={bnd:?}");
    let grid = Grid1D::new(-10.0, 10.0, n)
        .expect("grid")
        .with_boundary(bnd);
    // Non-constant a(x) to exercise the τ²-correction term (fd9 code path).
    // Using sin approximation via f64::sin (simd feature implies std).
    let dc = Diffusion6thChernoff::new(
        |x| 0.5 + 0.1 * x.sin(),
        |x| 0.1 * x.cos(),
        |x| -0.1 * x.sin(),
        0.6,
        grid,
    );
    let values = lcg_values(n, 0xdead_beef_cafe_babe);
    let f0 = GridFn1D::new(grid, values).expect("f0");
    let tau = 0.01_f64;

    let scalar_out = with_force_scalar(|| dc.apply_chernoff(tau, &f0).expect("apply scalar"));
    let simd_out = dc.apply_chernoff(tau, &f0).expect("apply simd");

    assert_bit_equal_vecs(&scalar_out.values, &simd_out.values, &label);
}

#[test]
fn diffusion6_bit_equal_all() {
    let boundaries = [
        BoundaryPolicy::Reflect,
        BoundaryPolicy::ZeroExtend,
        BoundaryPolicy::Periodic,
        BoundaryPolicy::LinearExtrapolate,
    ];
    for &n in &[64usize, 256, 1024, 4096] {
        for &bnd in &boundaries {
            check_diffusion6_bit_equal(n, bnd);
        }
    }
}

// ---------------------------------------------------------------------------
// Hot path 2: SepticHermite via Grid1D::interp — exercises fd_scaled_prime SIMD.
// (QuinticHermite SIMD test removed at v7.0 — ADR-0109 removal clock fulfilled.)
// ---------------------------------------------------------------------------

fn check_septic_bit_equal(n: usize, bnd: BoundaryPolicy) {
    let label = format!("septic_hermite_sample N={n} bnd={bnd:?}");
    let grid = Grid1D::new(-5.0, 5.0, n)
        .expect("grid")
        .with_boundary(bnd)
        .with_interp(InterpKind::SepticHermite);
    let values = lcg_values(n, 0x1234_5678_9abc_def0);
    let f = GridFn1D::new(grid, values).expect("f");

    // Sample at 50 off-grid points spread across the domain.
    let sample_xs: Vec<f64> = (0_i32..50)
        .map(|k| grid.xmin + (f64::from(k) + 0.37) * grid.dx())
        .collect();

    let scalar_ys: Vec<f64> = with_force_scalar(|| {
        sample_xs
            .iter()
            .map(|&x| f.sample(x).expect("sample"))
            .collect()
    });
    let simd_ys: Vec<f64> = sample_xs
        .iter()
        .map(|&x| f.sample(x).expect("sample"))
        .collect();

    assert_bit_equal_vecs(&scalar_ys, &simd_ys, &label);
}

#[test]
fn septic_hermite_bit_equal_all() {
    let boundaries = [
        BoundaryPolicy::Reflect,
        BoundaryPolicy::ZeroExtend,
        BoundaryPolicy::Periodic,
        BoundaryPolicy::LinearExtrapolate,
    ];
    for &n in &[64usize, 256, 1024, 4096] {
        for &bnd in &boundaries {
            check_septic_bit_equal(n, bnd);
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel × SIMD composition (SIMD_BIT_EQUAL_PARALLEL, release-blocking).
// Requires --features parallel,simd,slow-tests.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "parallel", feature = "slow-tests"))]
mod parallel_composition {
    use semiflow::{
        strang2d_parallel::FORCE_THREADS, ChernoffSemigroup, Grid2D, GridFn2D, Strang2D,
    };

    use super::*;

    fn make_f0(grid: Grid2D) -> GridFn2D {
        GridFn2D::from_fn(grid, |x, y| (-(x * x + y * y) * 0.5).exp())
    }

    fn run_strang2d_d6(n: usize, thread_count: usize, force_scalar: bool) -> Vec<f64> {
        let gx = Grid1D::new(-10.0, 10.0, n).expect("gx");
        let gy = Grid1D::new(-10.0, 10.0, n).expect("gy");
        let grid = Grid2D::new(gx, gy);
        let f0 = make_f0(grid);

        let cx = Diffusion6thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
        let cy = Diffusion6thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);
        let phi = Strang2D::new(cx, cy);
        let semi = ChernoffSemigroup::new(phi, 2).expect("n_steps=2");

        FORCE_THREADS.with(|c| c.set(Some(thread_count)));
        let run = || semi.evolve(0.02, &f0).expect("evolve").values;
        let result = if force_scalar {
            with_force_scalar(run)
        } else {
            run()
        };
        FORCE_THREADS.with(|c| c.set(None));
        result
    }

    // simd_1t / simd_mt are numerical shorthand (1-thread / multi-thread).
    #[allow(clippy::similar_names)]
    #[test]
    fn simd_bit_equal_parallel() {
        for &n in &[64usize, 128, 256] {
            // Reference: single thread, scalar-forced.
            let scalar_1t = run_strang2d_d6(n, 1, true);
            // SIMD, single thread — must match scalar.
            let simd_1t = run_strang2d_d6(n, 1, false);
            assert_bit_equal_vecs(
                &scalar_1t,
                &simd_1t,
                &format!("parallel N={n}: scalar_1t vs simd_1t"),
            );
            // SIMD, multi-thread — must match single-thread SIMD.
            for &threads in &[2usize, 4, 8] {
                let simd_mt = run_strang2d_d6(n, threads, false);
                assert_bit_equal_vecs(
                    &simd_1t,
                    &simd_mt,
                    &format!("parallel N={n} threads={threads}"),
                );
            }
        }
    }
}
