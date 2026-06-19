//! Bit-equality gate for the Wave B2 `HalfNodeCoeffCache` storage refactor
//! in `TruncatedExp4thDiffusionChernoff` (v0.13.0, ADR-0034 Amendment 1).
//!
//! Gate name: `TEXP4_CACHED_COEFF_BIT_EQUAL`.
//! Classification: **RELEASE-BLOCKING**.
//!
//! ## What is tested
//!
//! `TruncatedExp4WithCache::with_cached_coefficients` pre-evaluates `a(x)` at
//! all half-node points once at construction time.  The stencil hot-loop then
//! reads from the cache instead of calling the `a` function pointer.
//!
//! The two paths MUST be byte-for-byte identical because:
//!
//! 1. IEEE 754 addition and multiplication are deterministic: the exact same
//!    floating-point values enter each arithmetic operation in both paths.
//! 2. The cache is populated by the same function-pointer expression used in
//!    `apply_g4_at_node_f64`, so the stored value and the live value are equal.
//! 3. No rounding can occur between the two paths: cache stores the full f64,
//!    no truncation.
//!
//! ## Failure mode
//!
//! If any pair of bits diverges the panic prints: index, hex bits of both
//! values, and ULP distance — to enable root-cause analysis.
//!
//! ## Non-constant `a(x)` requirement
//!
//! Tests use `a(x) = |sin(x)| + 0.1` (non-constant) to ensure the cache
//! actually performs non-trivial work.  A constant `a` would give identical
//! results even if the cache were ignoring the stored values.

use semiflow_core::{
    chernoff::ApplyChernoffExt, Grid1D, GridFn1D, TruncatedExp4WithCache,
    TruncatedExp4thDiffusionChernoff,
};

// ---------------------------------------------------------------------------
// Diagnostic reporter — byte-level diff on divergence.
// ---------------------------------------------------------------------------

fn assert_bit_equal(cached: &[f64], reference: &[f64], label: &str) {
    assert_eq!(cached.len(), reference.len(), "{label}: length mismatch");
    for (k, (&c, &r)) in cached.iter().zip(reference.iter()).enumerate() {
        if c.to_bits() != r.to_bits() {
            let cb = i64::from_ne_bytes(c.to_bits().to_ne_bytes());
            let rb = i64::from_ne_bytes(r.to_bits().to_ne_bytes());
            let ulp = cb.wrapping_sub(rb).unsigned_abs();
            panic!(
                "{label}: diverged at index {k}: \
                 cached={c:.17e} (0x{:016x}), reference={r:.17e} (0x{:016x}), ULP={ulp}",
                c.to_bits(),
                r.to_bits(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// TEXP4_CACHED_COEFF_BIT_EQUAL — core gate.
// ---------------------------------------------------------------------------

/// Cache path must be bit-equal to closure path on the canonical n=64 bench grid
/// with a non-constant variable coefficient.
///
/// Non-constant `a(x) = |sin(x)| + 0.1` is used so the cache lookup is
/// non-trivial — a constant `a` would trivially agree regardless of cache.
#[test]
fn texp4_cached_coeff_bit_equal_variable_a_n64() {
    // a(x) = |sin(x)| + 0.1  — strictly positive, non-constant.
    fn a(x: f64) -> f64 {
        x.sin().abs() + 0.1
    }
    fn a_prime(_: f64) -> f64 {
        0.0
    }
    fn a_double_prime(_: f64) -> f64 {
        0.0
    }

    let grid = Grid1D::new(-4.0, 4.0, 64).expect("grid");
    // a_norm_bound ≥ sup|a| = 1.1; use 1.2 with margin.
    let a_norm_bound = 1.2_f64;

    // CFL: tau < 3*dx²/(8*a_norm_bound).
    // dx ≈ (8/63) ≈ 0.127; dx² ≈ 0.01608; limit ≈ 0.00753.
    let tau = 0.001_f64;

    // Non-trivial initial condition.
    let u0 = GridFn1D::from_fn(grid, |x| (-(x * x)).exp());

    // Reference path — unmodified closure-dispatch constructor.
    let me4_ref =
        TruncatedExp4thDiffusionChernoff::new(a, a_prime, a_double_prime, a_norm_bound, grid);

    // Cache path — Wave B2 constructor.
    let me4_cached = TruncatedExp4WithCache::with_cached_coefficients(
        a,
        a_prime,
        a_double_prime,
        a_norm_bound,
        grid,
    );

    let out_ref = me4_ref.apply_chernoff(tau, &u0).expect("reference apply");
    let out_cached = me4_cached.apply_chernoff(tau, &u0).expect("cached apply");

    assert_bit_equal(
        &out_cached.values,
        &out_ref.values,
        "TEXP4_CACHED_COEFF_BIT_EQUAL n=64 |sin(x)|+0.1",
    );
}

/// Bit-equality holds across multiple steps (accumulation test).
///
/// Confirms that the cache is not stalened or corrupted after repeated `apply`
/// calls.  Both paths are evolved for 3 steps; results must remain identical.
#[test]
fn texp4_cached_coeff_bit_equal_multistep() {
    fn a(x: f64) -> f64 {
        x.sin().abs() + 0.1
    }

    let grid = Grid1D::new(-4.0, 4.0, 64).expect("grid");
    let a_norm_bound = 1.2_f64;
    let tau = 0.001_f64;
    let mut u_ref = GridFn1D::from_fn(grid, |x| (core::f64::consts::PI * x).sin());
    let mut u_cached = u_ref.clone();

    let me4_ref = TruncatedExp4thDiffusionChernoff::new(a, |_| 0.0, |_| 0.0, a_norm_bound, grid);
    let me4_cached =
        TruncatedExp4WithCache::with_cached_coefficients(a, |_| 0.0, |_| 0.0, a_norm_bound, grid);

    for step in 0..3_usize {
        u_ref = me4_ref.apply_chernoff(tau, &u_ref).expect("ref step");
        u_cached = me4_cached
            .apply_chernoff(tau, &u_cached)
            .expect("cached step");
        assert_bit_equal(
            &u_cached.values,
            &u_ref.values,
            &format!("TEXP4_CACHED_COEFF_BIT_EQUAL multistep step={step}"),
        );
    }
}
