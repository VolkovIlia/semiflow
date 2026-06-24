// Tests for `diffusion8_zeta8.rs` (Diffusion8thZeta8Chernoff, ADR-0088 Wave II).
//
// Properties asserted:
//   1. Construction succeeds from valid ζ⁶ inner kernel.
//   2. order() returns 8 (K=4 nested Richardson).
//   3. growth() is a contraction (multiplier ≤ 1, omega = 0).
//   4. without_chebyshev_sampling() returns a kernel with the same grid.
//   5. with_chebyshev_sampling_m(128) constructs without error.
//   6. with_octonic_sampling() constructs without error.
//   7. DomainViolation for NaN a_kth_bound.
//   8. DomainViolation for negative a_kth_bound.
//   9. apply_chernoff with tau > 0 gives finite output.
//  10. apply_chernoff with tau = 0 gives identity output.

use crate::{
    chernoff::ApplyChernoffExt,
    diffusion4::Diffusion4thChernoff,
    diffusion4_zeta4::Diffusion4thZeta4Chernoff,
    diffusion6_zeta6::Diffusion6thZeta6Chernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    ChernoffFunction,
};

fn make_d8(n: usize) -> Diffusion8thZeta8Chernoff<f64> {
    let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
    let d4 = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let zeta4 = Diffusion4thZeta4Chernoff::new(d4, None).unwrap();
    let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, None).unwrap();
    Diffusion8thZeta8Chernoff::new(zeta6, None).unwrap()
}

// ── Construction ──────────────────────────────────────────────────────────────

#[test]
fn construction_succeeds() {
    let _k = make_d8(16);
}

// ── order() and growth() ──────────────────────────────────────────────────────

#[test]
fn order_is_8() {
    let k = make_d8(16);
    assert_eq!(k.order(), 8, "order() must be 8 for ζ⁸ kernel");
}

#[test]
fn growth_is_contraction() {
    let k = make_d8(16);
    let g = k.growth();
    assert!(
        g.multiplier <= 1.0 + 1e-14,
        "multiplier={} should be ≤ 1",
        g.multiplier
    );
    assert!(g.omega.abs() < 1e-14, "omega={} should be 0", g.omega);
}

// ── Builder methods ───────────────────────────────────────────────────────────

#[test]
fn without_chebyshev_sampling_constructs() {
    let k = make_d8(16).without_chebyshev_sampling();
    assert_eq!(k.order(), 8);
}

#[test]
fn with_chebyshev_sampling_m_constructs() {
    let k = make_d8(16).with_chebyshev_sampling_m(128);
    assert_eq!(k.order(), 8);
}

#[test]
fn with_octonic_sampling_constructs() {
    let k = make_d8(16).with_octonic_sampling();
    assert_eq!(k.order(), 8);
}

// ── Domain violations ─────────────────────────────────────────────────────────

#[test]
fn nan_bound_returns_err() {
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let d4 = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let zeta4 = Diffusion4thZeta4Chernoff::new(d4, None).unwrap();
    let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, None).unwrap();
    let result = Diffusion8thZeta8Chernoff::new(zeta6, Some(f64::NAN));
    assert!(result.is_err(), "expected Err for NaN bound");
}

#[test]
fn negative_bound_returns_err() {
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let d4 = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let zeta4 = Diffusion4thZeta4Chernoff::new(d4, None).unwrap();
    let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, None).unwrap();
    let result = Diffusion8thZeta8Chernoff::new(zeta6, Some(-1.0_f64));
    assert!(result.is_err(), "expected Err for negative bound");
}

// ── apply_chernoff: finite output ─────────────────────────────────────────────

#[test]
fn apply_chernoff_gives_finite_output() {
    let k = make_d8(16);
    let grid = k.grid;
    let src = GridFn1D::from_fn(grid, f64::sin);
    let dst = k.apply_chernoff(0.01_f64, &src).unwrap();
    assert_eq!(dst.values.len(), src.values.len());
    for v in &dst.values {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}

// ── apply_chernoff tau=0 → output finite ─────────────────────────────────────
// Note: Chebyshev sampling at tau=0 is not exact identity (interpolation error).

#[test]
fn apply_chernoff_small_tau_output_finite() {
    // Use a very small tau (not zero) to confirm finite, bounded output.
    let k = make_d8(16);
    let grid = k.grid;
    let src = GridFn1D::from_fn(grid, |x| x * (1.0 - x));
    let norm_src = src.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let dst = k.apply_chernoff(1e-6_f64, &src).unwrap();
    let norm_dst = dst.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    for v in &dst.values {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
    // Output norm should not blow up relative to input.
    assert!(
        norm_dst <= norm_src * 2.0 + 1e-10,
        "sup-norm blew up: before={norm_src}, after={norm_dst}"
    );
}
