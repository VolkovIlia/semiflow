//! Subordinated Chernoff smoke tests (AC8 — non-slow, always-on gates).
//!
//! Covers all three Lévy backends: round-trips, known `laplace_exponent` values,
//! `SubordinatedChernoff` construction validation, and a single-step apply.
//!
//! **Gate**: `test-fast` (no `#[ignore]`, no feature gate).
//! **ADR**: 0103; **math**: §37; **properties.yaml**: `T_SUBORD` (symbolic oracle,
//! separate script `scripts/verify_subordinated_chernoff.py`).

use semiflow_core::{
    diffusion::DiffusionChernoff,
    error::SemiflowError,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
    subordinated::{
        GammaSubordinator, InverseGaussianSubordinator, LevySubordinator, StableSubordinator,
        SubordinatedChernoff,
    },
    ChernoffFunction,
};

// ─── StableSubordinator ───────────────────────────────────────────────────────

#[test]
fn stable_round_trip_and_phi_1_0() {
    // φ_{0.5}(1.0) = 1.0^0.5 = 1.0
    let sub = StableSubordinator::new(0.5_f64).unwrap();
    let phi = sub.laplace_exponent(1.0_f64);
    assert!(
        (phi - 1.0).abs() < 1e-13,
        "φ_0.5(1.0) expected 1.0; got {phi}"
    );
}

#[test]
fn stable_rejects_alpha_zero() {
    let err = StableSubordinator::<f64>::new(0.0).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

#[test]
fn stable_rejects_alpha_one() {
    let err = StableSubordinator::<f64>::new(1.0).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

#[test]
fn stable_rejects_alpha_nan() {
    let err = StableSubordinator::<f64>::new(f64::NAN).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

#[test]
fn stable_rejects_alpha_negative() {
    let err = StableSubordinator::<f64>::new(-0.3).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

// ─── GammaSubordinator ───────────────────────────────────────────────────────

#[test]
fn gamma_laplace_exponent_at_zero_is_zero() {
    // CBF property: φ(0) = 0.
    let sub = GammaSubordinator::new(2.0_f64).unwrap();
    let phi0 = sub.laplace_exponent(0.0_f64);
    assert!(phi0.abs() < 1e-14, "φ(0) should be 0; got {phi0}");
}

#[test]
fn gamma_laplace_exponent_log2() {
    // φ_{c=2}(2) = log(1 + 2/2) = log(2).
    let sub = GammaSubordinator::new(2.0_f64).unwrap();
    let phi = sub.laplace_exponent(2.0_f64);
    let expected = 2.0_f64.ln();
    assert!(
        (phi - expected).abs() < 1e-13,
        "φ_2(2) expected ln(2)={expected}; got {phi}"
    );
}

#[test]
fn gamma_rejects_zero_c() {
    let err = GammaSubordinator::<f64>::new(0.0).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

// ─── InverseGaussianSubordinator ─────────────────────────────────────────────

#[test]
fn ig_laplace_exponent_at_zero_is_zero() {
    // CBF property: φ(0) = √(c² + 0) − c = 0.
    let sub = InverseGaussianSubordinator::new(1.5_f64).unwrap();
    let phi0 = sub.laplace_exponent(0.0_f64);
    assert!(phi0.abs() < 1e-13, "φ_IG(0) should be 0; got {phi0}");
}

#[test]
fn ig_rejects_zero_c() {
    let err = InverseGaussianSubordinator::<f64>::new(0.0).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

#[test]
fn ig_rejects_negative_c() {
    let err = InverseGaussianSubordinator::<f64>::new(-0.1).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

// ─── SubordinatedChernoff construction gates ─────────────────────────────────

#[test]
fn with_n_nodes_zero_is_domain_violation() {
    let grid = Grid1D::new(-1.0_f64, 1.0, 8).unwrap();
    let base = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let sub = StableSubordinator::new(0.5_f64).unwrap();
    let result = SubordinatedChernoff::with_n_nodes(base, sub, 0);
    let err = result.err().expect("expected DomainViolation error");
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

#[test]
fn with_n_nodes_over_cap_is_domain_violation() {
    let grid = Grid1D::new(-1.0_f64, 1.0, 8).unwrap();
    let base = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let sub = StableSubordinator::new(0.5_f64).unwrap();
    let result = SubordinatedChernoff::with_n_nodes(base, sub, 33);
    let err = result.err().expect("expected DomainViolation error");
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

// ─── Single-step apply smoke ─────────────────────────────────────────────────

#[test]
fn subordinated_apply_stable_finite_non_negative() {
    // Heat semigroup is positivity-preserving; so is the subordinated version.
    let grid = Grid1D::new(-2.0_f64, 2.0, 32).unwrap();
    let base = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let sub = StableSubordinator::new(0.5_f64).unwrap();
    let wrapper = SubordinatedChernoff::new(base, sub);

    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut scratch = ScratchPool::new();
    wrapper
        .apply_into(0.01_f64, &u0, &mut dst, &mut scratch)
        .unwrap();

    assert!(
        dst.values.iter().all(|&v| v.is_finite()),
        "output has NaN/Inf"
    );
    assert!(
        dst.values.iter().all(|&v| v >= 0.0),
        "output has negative values"
    );
}

#[test]
fn subordinated_apply_rejects_negative_tau() {
    let grid = Grid1D::new(-1.0_f64, 1.0, 8).unwrap();
    let base = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let sub = StableSubordinator::new(0.5_f64).unwrap();
    let wrapper = SubordinatedChernoff::new(base, sub);
    let u0 = GridFn1D::from_fn(grid, |_| 1.0_f64);
    let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut scratch = ScratchPool::new();
    let err = wrapper
        .apply_into(-0.1_f64, &u0, &mut dst, &mut scratch)
        .unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}
