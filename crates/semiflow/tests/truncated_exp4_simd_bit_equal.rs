//! Bit-equality gate for the Wave B3 SIMD path on `TruncatedExp4WithCache`
//! (v0.13.0, ADR-0019 Amendment 2).
//!
//! Gate name: `TEXP4_SIMD_BIT_EQUAL`.
//! Classification: **RELEASE-BLOCKING**.
//!
//! ## What is tested
//!
//! When `feature = "simd"` is active, `TruncatedExp4WithCache::apply` dispatches to
//! the `AVX2` (`x86_64`) or `NEON` (aarch64) 4-node parallel stencil path for interior
//! nodes.  The `simd::FORCE_SCALAR` hook forces the scalar fallback path.
//!
//! Both paths MUST be byte-for-byte identical because:
//!
//! 1. The SIMD kernels use the same constant factors (5.0, 0.25, 1.0/12.0) and
//!    arithmetic order as the scalar reference.
//! 2. FMA is FORBIDDEN in the SIMD kernel (ADR-0019 §`determinism_contract`).
//!    Only separate `mul` + `add`/`sub` are used, matching scalar rounding.
//! 3. Boundary nodes (first/last 2) always run the scalar path in both modes.
//!
//! ## Failure mode
//!
//! Divergence indicates FMA contamination or wrong arithmetic order.
//! The panic prints the first divergent index and ULP distance for diagnosis.
//!
//! ## Non-constant `a(x)` requirement
//!
//! Uses `a(x) = |sin(x)| + 0.5` (non-constant) so the stencil is non-trivial.

use semiflow::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, TruncatedExp4WithCache};

// ---------------------------------------------------------------------------
// Diagnostic reporter — byte-level diff on divergence.
// ---------------------------------------------------------------------------

fn assert_bit_equal(simd: &[f64], scalar: &[f64], label: &str) {
    assert_eq!(simd.len(), scalar.len(), "{label}: length mismatch");
    for (k, (&s, &r)) in simd.iter().zip(scalar.iter()).enumerate() {
        if s.to_bits() != r.to_bits() {
            let sb = i64::from_ne_bytes(s.to_bits().to_ne_bytes());
            let rb = i64::from_ne_bytes(r.to_bits().to_ne_bytes());
            let ulp = sb.wrapping_sub(rb).unsigned_abs();
            panic!(
                "{label}: diverged at index {k}: \
                 simd={s:.17e} (0x{:016x}), scalar={r:.17e} (0x{:016x}), ULP={ulp}",
                s.to_bits(),
                r.to_bits(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// TEXP4_SIMD_BIT_EQUAL — core gate.
// ---------------------------------------------------------------------------

/// SIMD path and scalar fallback must be bit-identical on the n=64 bench grid
/// with a non-constant variable coefficient.
///
/// `a(x) = |sin(x)| + 0.5` is strictly positive and non-constant, ensuring
/// the cache lookups and stencil arithmetic are non-trivial.
#[test]
fn texp4_simd_bit_equal_variable_a_n64() {
    fn a(x: f64) -> f64 {
        x.sin().abs() + 0.5
    }

    let grid = Grid1D::new(-4.0, 4.0, 64).expect("grid");
    // a_norm_bound >= sup|a| = 1.5; use 1.6 with margin.
    let a_norm_bound = 1.6_f64;
    // CFL: tau < 3*dx^2/(8*a_norm_bound).
    // dx = 8/63 ~ 0.127; dx^2 ~ 0.01608; limit ~ 0.00378.
    let tau = 0.001_f64;

    let u0 = GridFn1D::from_fn(grid, |x| (-(x * x)).exp());

    let me4c =
        TruncatedExp4WithCache::with_cached_coefficients(a, |_| 0.0, |_| 0.0, a_norm_bound, grid);

    // SIMD path (default when feature = "simd" and no FORCE_SCALAR).
    let out_simd = me4c.apply_chernoff(tau, &u0).expect("simd apply");

    // Scalar fallback path (forced via FORCE_SCALAR hook).
    let out_scalar =
        semiflow::simd::with_force_scalar(|| me4c.apply_chernoff(tau, &u0).expect("scalar apply"));

    assert_bit_equal(
        &out_simd.values,
        &out_scalar.values,
        "TEXP4_SIMD_BIT_EQUAL n=64 |sin(x)|+0.5",
    );
}

/// Bit-equality holds across multiple steps (accumulation test).
///
/// Confirms rounding stays consistent over 5 steps.
#[test]
fn texp4_simd_bit_equal_multistep() {
    fn a(x: f64) -> f64 {
        x.sin().abs() + 0.5
    }

    let grid = Grid1D::new(-4.0, 4.0, 64).expect("grid");
    let a_norm_bound = 1.6_f64;
    let tau = 0.001_f64;

    let me4c =
        TruncatedExp4WithCache::with_cached_coefficients(a, |_| 0.0, |_| 0.0, a_norm_bound, grid);

    let u0 = GridFn1D::from_fn(grid, |x| (core::f64::consts::PI * x).sin());
    let mut u_simd = u0.clone();
    let mut u_scalar = u0;

    for step in 0..5_usize {
        u_simd = me4c.apply_chernoff(tau, &u_simd).expect("simd step");
        u_scalar = semiflow::simd::with_force_scalar(|| {
            me4c.apply_chernoff(tau, &u_scalar).expect("scalar step")
        });
        assert_bit_equal(
            &u_simd.values,
            &u_scalar.values,
            &format!("TEXP4_SIMD_BIT_EQUAL multistep step={step}"),
        );
    }
}

/// Larger grid (n=256) exercises more SIMD chunks and a different tail size.
#[test]
fn texp4_simd_bit_equal_n256() {
    fn a(x: f64) -> f64 {
        x.sin().abs() + 0.5
    }

    let grid = Grid1D::new(-8.0, 8.0, 256).expect("grid");
    let a_norm_bound = 1.6_f64;
    // dx = 16/255 ~ 0.0627; dx^2 ~ 0.00393; CFL limit ~ 0.000921.
    let tau = 0.0001_f64;

    let u0 = GridFn1D::from_fn(grid, |x| (-(x * x / 4.0)).exp());
    let me4c =
        TruncatedExp4WithCache::with_cached_coefficients(a, |_| 0.0, |_| 0.0, a_norm_bound, grid);

    let out_simd = me4c.apply_chernoff(tau, &u0).expect("simd apply n256");
    let out_scalar = semiflow::simd::with_force_scalar(|| {
        me4c.apply_chernoff(tau, &u0).expect("scalar apply n256")
    });

    assert_bit_equal(
        &out_simd.values,
        &out_scalar.values,
        "TEXP4_SIMD_BIT_EQUAL n=256 |sin(x)|+0.5",
    );
}
