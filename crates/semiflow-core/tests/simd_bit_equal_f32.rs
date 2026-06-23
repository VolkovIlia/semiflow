//! Bit-exact equality gate for f32 SIMD hot paths (ADR-0175, Phase 5b).
//!
//! Verifies that the f32 SIMD-accelerated paths produce **byte-for-byte identical**
//! output to the f32 scalar reference for every combination of:
//!
//! - `N ∈ {64, 256, 1024, 4096}` (grid nodes),
//! - `hot_path ∈ {diffusion6_apply_f32, catmull_rom_f32_sample}`,
//! - `boundary ∈ {Reflect, ZeroExtend, Periodic, LinearExtrapolate}`.
//!
//! No tolerance — comparison is byte-exact via `f32::to_bits()` (covers
//! signed-zero, subnormals, NaN bit patterns).
//!
//! On failure: prints first divergent index, scalar bits (hex), simd bits (hex),
//! and ULP gap.
//!
//! Gate: `SIMD_F32_BIT_EQUAL` in `contracts/semiflow-core.properties.yaml`.
//! Classification: **RELEASE-BLOCKING**.
//!
//! See `docs/adr/0175-f32-first-class-and-simd.md`.

#![cfg(feature = "simd")]

use semiflow_core::{
    chernoff::ApplyChernoffExt,
    diffusion6::Diffusion6thChernoff,
    simd::with_force_scalar,
    BoundaryPolicy, Grid1D, GridFn1D, InterpKind,
};

// ---------------------------------------------------------------------------
// Failure reporter — byte-level diagnostics on divergence.
// ---------------------------------------------------------------------------

fn assert_bit_equal_f32(scalar: &[f32], simd: &[f32], label: &str) {
    assert_eq!(
        scalar.len(),
        simd.len(),
        "{label}: length mismatch {} vs {}",
        scalar.len(),
        simd.len()
    );
    for (k, (&s, &v)) in scalar.iter().zip(simd.iter()).enumerate() {
        if s.to_bits() != v.to_bits() {
            let sb = i32::from_ne_bytes(s.to_bits().to_ne_bytes());
            let vb = i32::from_ne_bytes(v.to_bits().to_ne_bytes());
            let ulp = sb.wrapping_sub(vb).unsigned_abs();
            panic!(
                "{label}: diverged at index {k}: \
                 scalar={s:.10e} (0x{:08x}), simd={v:.10e} (0x{:08x}), ULP gap={ulp}",
                s.to_bits(),
                v.to_bits(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic value generator — fixed LCG, no random deps.
// ---------------------------------------------------------------------------

/// Generate `n` f32 values in [0.1, 1.1) using a fixed LCG.
fn lcg_values_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let hi32 = (state >> 32) as u32;
            // u32 → f64 (lossless), scale to [0,1), cast to f32.
            let mantissa = (hi32 as f64) / (u32::MAX as f64);
            (0.1 + mantissa) as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hot path: `Diffusion6thChernoff<f32>::apply_chernoff`
// Exercises both catmull_rom_f32 (baseline) and fd9_f32 (ζ⁶ correction).
// ---------------------------------------------------------------------------

fn check_diffusion6_f32_bit_equal(n: usize, bnd: BoundaryPolicy<f32>) {
    let label = format!("diffusion6_apply_f32 N={n} bnd={bnd:?}");
    let grid = Grid1D::<f32>::new_generic(-10.0_f32, 10.0_f32, n)
        .expect("grid f32")
        .with_boundary(bnd)
        .with_interp(InterpKind::CubicHermite);

    // Non-constant a(x) to exercise the τ²-correction term (fd9 code path).
    let dc = Diffusion6thChernoff::<f32>::new_generic(
        |x: f32| 0.5_f32 + 0.1_f32 * x.sin(),
        |x: f32| 0.1_f32 * x.cos(),
        |x: f32| -0.1_f32 * x.sin(),
        0.6,
        grid,
    );
    let values = lcg_values_f32(n, 0xdead_beef_cafe_babe);
    let f0 = GridFn1D::<f32>::new_generic(grid, values).expect("f0 f32");
    let tau = 0.01_f32;

    let scalar_out =
        with_force_scalar(|| dc.apply_chernoff(tau, &f0).expect("apply scalar f32"));
    let simd_out = dc.apply_chernoff(tau, &f0).expect("apply simd f32");

    assert_bit_equal_f32(&scalar_out.values, &simd_out.values, &label);
}

#[test]
fn diffusion6_f32_bit_equal_all() {
    let boundaries: [BoundaryPolicy<f32>; 4] = [
        BoundaryPolicy::Reflect,
        BoundaryPolicy::ZeroExtend,
        BoundaryPolicy::Periodic,
        BoundaryPolicy::LinearExtrapolate,
    ];
    for &n in &[64usize, 256, 1024, 4096] {
        for &bnd in &boundaries {
            check_diffusion6_f32_bit_equal(n, bnd);
        }
    }
}

// ---------------------------------------------------------------------------
// Hot path: Catmull-Rom f32 sampler via Diffusion6thChernoff<f32> (constant a).
// With constant a, the ζ⁶ correction is zero (a'=a''=0), so the result
// exercises ONLY the γ⁶-A baseline (catmull_rom_f32 sampling path).
// ---------------------------------------------------------------------------

fn check_catmull_f32_bit_equal(n: usize, bnd: BoundaryPolicy<f32>) {
    let label = format!("catmull_rom_f32_sample N={n} bnd={bnd:?}");
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, n)
        .expect("grid f32")
        .with_boundary(bnd)
        .with_interp(InterpKind::CubicHermite);

    // Constant a(x) = 0.5: a'=0, a''=0 → ζ⁶ correction = 0 exactly.
    // This isolates the catmull_rom_f32 path in gamma6_a_baseline_f32.
    let dc = Diffusion6thChernoff::<f32>::new_generic(
        |_: f32| 0.5_f32,
        |_: f32| 0.0_f32,
        |_: f32| 0.0_f32,
        0.5,
        grid,
    );
    let values = lcg_values_f32(n, 0x1234_5678_9abc_def0);
    let f0 = GridFn1D::<f32>::new_generic(grid, values).expect("f0 catmull");
    let tau = 0.01_f32;

    let scalar_out =
        with_force_scalar(|| dc.apply_chernoff(tau, &f0).expect("apply scalar catmull"));
    let simd_out = dc.apply_chernoff(tau, &f0).expect("apply simd catmull");

    assert_bit_equal_f32(&scalar_out.values, &simd_out.values, &label);
}

#[test]
fn catmull_rom_f32_bit_equal_all() {
    let boundaries: [BoundaryPolicy<f32>; 4] = [
        BoundaryPolicy::Reflect,
        BoundaryPolicy::ZeroExtend,
        BoundaryPolicy::Periodic,
        BoundaryPolicy::LinearExtrapolate,
    ];
    for &n in &[64usize, 256, 1024, 4096] {
        for &bnd in &boundaries {
            check_catmull_f32_bit_equal(n, bnd);
        }
    }
}
