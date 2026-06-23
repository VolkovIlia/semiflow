//! Bit-equality gate for the Wave B1 wasted-sample bypass in
//! `TruncatedExp4thDiffusionChernoff` (v0.13.0, ADR-0019 Amendment 2).
//!
//! Gate name: `TEXP4_BYPASS_BIT_EQUAL`.
//! Classification: **RELEASE-BLOCKING**.
//!
//! ## What is tested
//!
//! When `b_for_conjugation.is_none()`, `apply` now uses
//! `apply_power_series_f64_at_node` (direct `values[i]` index) instead of
//! `g_grids[k].sample(x_mid)` (cubic-Hermite interpolation).
//!
//! The two paths must be byte-for-byte identical because:
//!   1. `x_mid = x_i` exactly (see `compute_x_mid_f64`, line 344-348).
//!   2. `catmull_rom(pm1, p0, p1, p2, 0.0) = p0` exactly in IEEE 754
//!      (scalar: `0.5 * 2·p0 = p0`; SIMD dot: coefficient for p0 is `2.0`).
//!   3. For the reference grid (n=64, [-4.0, 4.0]), `(x_i - xmin) / dx = i`
//!      exactly in f64 (verified by numerical experiment: all 64 nodes give
//!      `s = t_frac - floor(t_frac) == 0.0` exactly).
//!
//! ## Reference path
//!
//! The reference uses `.with_drift_conjugation(|_| 0.0)`:
//!   `x_mid = x_i - 0.5 * tau * 0.0 = x_i - 0.0 = x_i` (IEEE-exact).
//! So the reference also calls `sample(x_i)` which triggers full cubic-Hermite,
//! landing on `s == 0.0` and returning `values[i]` exactly — matching the bypass.
//!
//! ## Failure mode
//!
//! If any pair of bits diverges, the panic prints: index, hex bits of both
//! values, and ULP distance — to enable root-cause analysis.

use semiflow_core::{
    chernoff::ApplyChernoffExt, Grid1D, GridFn1D, TruncatedExp4thDiffusionChernoff,
};

// ---------------------------------------------------------------------------
// Diagnostic reporter — byte-level diff on divergence.
// ---------------------------------------------------------------------------

fn assert_bit_equal(bypass: &[f64], reference: &[f64], label: &str) {
    assert_eq!(bypass.len(), reference.len(), "{label}: length mismatch");
    for (k, (&b, &r)) in bypass.iter().zip(reference.iter()).enumerate() {
        if b.to_bits() != r.to_bits() {
            let bb = i64::from_ne_bytes(b.to_bits().to_ne_bytes());
            let rb = i64::from_ne_bytes(r.to_bits().to_ne_bytes());
            let ulp = bb.wrapping_sub(rb).unsigned_abs();
            panic!(
                "{label}: diverged at index {k}: \
                 bypass={b:.17e} (0x{:016x}), reference={r:.17e} (0x{:016x}), ULP={ulp}",
                b.to_bits(),
                r.to_bits(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// TEXP4_BYPASS_BIT_EQUAL — core gate (no drift, n=64 canonical bench grid).
// ---------------------------------------------------------------------------

/// Bypass path (`b_for_conjugation = None`) must be bit-equal to reference path
/// (b = |_| 0.0) on the canonical n=64 bench grid.
#[test]
fn texp4_bypass_bit_equal_no_drift_n64() {
    let grid = Grid1D::new(-4.0, 4.0, 64).expect("grid");
    let tau = 0.001_f64; // well within CFL for a=1.0, n=64

    // Non-trivial initial condition: sin(pi*x) on [-4, 4].
    let u0 = GridFn1D::from_fn(grid, |x| (core::f64::consts::PI * x).sin());

    // Bypass path: b_for_conjugation == None.
    let me4_bypass = TruncatedExp4thDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);

    // Reference path: zero drift -> x_mid = x_i exactly, sample(x_i) gives values[i].
    let me4_ref = TruncatedExp4thDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid)
        .with_drift_conjugation(|_| 0.0_f64);

    let out_bypass = me4_bypass.apply_chernoff(tau, &u0).expect("bypass apply");
    let out_ref = me4_ref.apply_chernoff(tau, &u0).expect("reference apply");

    assert_bit_equal(
        &out_bypass.values,
        &out_ref.values,
        "TEXP4_BYPASS_BIT_EQUAL n=64 sin(pi*x)",
    );
}

/// Non-zero drift path must NOT be bit-equal to no-drift result (correctness sanity).
///
/// Confirms the drift-conjugation branch is still exercised (non-trivial drift
/// shifts `x_mid` away from `x_i`, producing a genuinely different output).
#[test]
fn texp4_nonzero_drift_differs_from_no_drift() {
    let grid = Grid1D::new(-4.0, 4.0, 64).expect("grid");
    let tau = 0.001_f64;

    let u0 = GridFn1D::from_fn(grid, |x| (core::f64::consts::PI * x).sin());

    let me4_no_drift = TruncatedExp4thDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let me4_drift = TruncatedExp4thDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid)
        .with_drift_conjugation(|_| 0.5_f64);

    let out_no_drift = me4_no_drift
        .apply_chernoff(tau, &u0)
        .expect("no-drift apply");
    let out_drift = me4_drift.apply_chernoff(tau, &u0).expect("drift apply");

    // Non-zero drift must produce a different result (bypass branch not taken).
    let max_diff = out_no_drift
        .values
        .iter()
        .zip(out_drift.values.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 0.0,
        "Non-zero drift should produce different output (bypass must NOT fire for b!=0)"
    );
}
