// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

use super::{
    GammaSubordinator, InverseGaussianSubordinator, LevySubordinator, StableSubordinator,
    SubordinatedChernoff,
};
use crate::{
    chernoff::ChernoffFunction, diffusion::DiffusionChernoff,
    drift_reaction::DriftReactionChernoff, error::SemiflowError, grid::Grid1D, grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// --- Laplace exponent spot-checks (CBF verification) ---

#[test]
fn stable_laplace_exponent_known_values() {
    let sub = StableSubordinator::new(0.5_f64).unwrap();
    assert!((sub.laplace_exponent(1.0_f64) - 1.0).abs() < 1e-14);
    assert!((sub.laplace_exponent(4.0_f64) - 2.0).abs() < 1e-14);
    assert!(sub.laplace_exponent(0.0_f64).abs() < 1e-14);
}

#[test]
fn gamma_laplace_exponent_known_values() {
    let sub1 = GammaSubordinator::new(1.0_f64).unwrap();
    assert!(sub1.laplace_exponent(0.0_f64).abs() < 1e-14);
    let sub2 = GammaSubordinator::new(2.0_f64).unwrap();
    assert!((sub2.laplace_exponent(2.0_f64) - 2.0_f64.ln()).abs() < 1e-14);
}

#[test]
fn inverse_gaussian_laplace_exponent_known_values() {
    let sub = InverseGaussianSubordinator::new(1.0_f64).unwrap();
    assert!(sub.laplace_exponent(0.0_f64).abs() < 1e-14);
    assert!((sub.laplace_exponent(0.5_f64) - (2.0_f64.sqrt() - 1.0)).abs() < 1e-14);
}

// --- Quadrature invariants ---

#[test]
fn stable_quadrature_node_count_and_weight_sum() {
    let sub = StableSubordinator::new(0.5_f64).unwrap();
    let (nodes, weights) = sub.quadrature(1.0_f64, 8);
    assert!(!nodes.is_empty());
    assert_eq!(nodes.len(), weights.len());
    let w_sum: f64 = weights.iter().sum();
    assert!((w_sum - 1.0_f64).abs() < 1e-10, "Σw={w_sum}");
}

#[test]
fn gamma_quadrature_node_count_and_positivity() {
    let sub = GammaSubordinator::new(1.0_f64).unwrap();
    let (nodes, weights) = sub.quadrature(1.0_f64, 16);
    assert_eq!(nodes.len(), 16);
    assert_eq!(weights.len(), 16);
    assert!(nodes.iter().all(|&s| s > 0.0_f64));
    assert!(weights.iter().all(|&w| w >= 0.0_f64));
    let w_sum: f64 = weights.iter().sum();
    assert!((w_sum - 1.0_f64).abs() < 1e-10, "Σw={w_sum}");
}

#[test]
fn ig_quadrature_node_count_positive() {
    let sub = InverseGaussianSubordinator::new(1.0_f64).unwrap();
    let (nodes, weights) = sub.quadrature(1.0_f64, 8);
    assert!(!nodes.is_empty());
    assert!(nodes.iter().all(|&s| s > 0.0_f64));
    let w_sum: f64 = weights.iter().sum();
    assert!((w_sum - 1.0_f64).abs() < 1e-6, "IG Σw={w_sum}");
}

// --- Construction + validation ---

#[test]
fn stable_rejects_invalid_alpha() {
    assert!(StableSubordinator::<f64>::new(0.0).is_err());
    assert!(StableSubordinator::<f64>::new(1.0).is_err());
    assert!(StableSubordinator::<f64>::new(-0.5).is_err());
    assert!(StableSubordinator::<f64>::new(f64::NAN).is_err());
}

#[test]
fn gamma_rejects_non_positive_c() {
    assert!(GammaSubordinator::<f64>::new(0.0).is_err());
    assert!(GammaSubordinator::<f64>::new(-1.0).is_err());
    assert!(GammaSubordinator::<f64>::new(f64::NAN).is_err());
}

#[test]
fn ig_rejects_non_positive_c() {
    assert!(InverseGaussianSubordinator::<f64>::new(0.0).is_err());
    assert!(InverseGaussianSubordinator::<f64>::new(-1.0).is_err());
}

#[test]
fn with_n_nodes_rejects_zero_and_over_cap() {
    let grid = Grid1D::new(-1.0_f64, 1.0, 8).unwrap();
    let base = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let sub = StableSubordinator::new(0.5_f64).unwrap();
    assert!(matches!(
        SubordinatedChernoff::with_n_nodes(base.clone(), sub, 0),
        Err(SemiflowError::DomainViolation { .. })
    ));
    assert!(matches!(
        SubordinatedChernoff::with_n_nodes(base, sub, 33),
        Err(SemiflowError::DomainViolation { .. })
    ));
}

// --- Smoke test: construct + apply on small grid ---

#[test]
fn subordinated_chernoff_smoke_stable() {
    let grid = Grid1D::new(-1.0_f64, 1.0, 16).unwrap();
    let base = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let sub = StableSubordinator::new(0.5_f64).unwrap();
    let wrapper = SubordinatedChernoff::new(base, sub);
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut scratch = ScratchPool::new();
    wrapper
        .apply_into(0.01_f64, &u0, &mut dst, &mut scratch)
        .unwrap();
    assert!(dst.values.iter().all(|&v| v.is_finite()));
}

// --- Anti-linearization regression gate (STEP 7, ADR-0103) ---

/// Always-on gate: verifies that `SubordinatedChernoff` does NOT linearize φ.
///
/// Setup: Gamma c=1, λ=4, small τ=0.01 (one step).
/// φ(4) = ln(5) ≈ 1.6094;  φ(1)·4 = ln(2)·4 ≈ 2.7726 (buggy linearization).
/// Correct step: one application ≈ 1 − τ·ln(5).  Wrong: 1 − τ·ln(2)·4.
///
/// Gate: |step − (1−τ·ln5)| < 0.1·|step − (1−τ·ln2·4)|.
#[test]
fn subordinated_does_not_linearize_phi() {
    const LAMBDA: f64 = 4.0;
    const C: f64 = 1.0;
    let tau = 0.01_f64;
    let grid = Grid1D::new(0.0_f64, 1.0, 4).unwrap();
    let base = DriftReactionChernoff::new(|_| 0.0_f64, |_| -LAMBDA, 1.0_f64, grid);
    let sub = GammaSubordinator::new(C).unwrap();
    let wrapper = SubordinatedChernoff::new(base, sub);
    let f = GridFn1D::from_fn(grid, |_| 1.0_f64);
    let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut scratch = ScratchPool::new();
    wrapper.apply_into(tau, &f, &mut dst, &mut scratch).unwrap();
    let step = dst.values[0];
    let correct_fo = 1.0 - tau * (1.0 + LAMBDA / C).ln();
    let wrong_fo = 1.0 - tau * (1.0 + 1.0 / C).ln() * LAMBDA;
    let err_correct = (step - correct_fo).abs();
    let err_wrong = (step - wrong_fo).abs();
    assert!(
        err_correct < 0.1 * err_wrong,
        "Linearization detected: step={step:.6}, correct_fo={correct_fo:.6}, \
         wrong_fo={wrong_fo:.6}, err_correct={err_correct:.2e}, err_wrong={err_wrong:.2e}. \
         Must satisfy err_correct < 0.1 * err_wrong."
    );
}
