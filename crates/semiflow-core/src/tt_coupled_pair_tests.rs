// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

use super::*;

/// `pair_eigen_check`: SPD for |ρ|<1 (d=2 block-disjoint).
#[test]
fn pair_eigen_check_spd_ok() {
    let (aj, ak, rho) = (0.8f64, 0.6f64, 0.6f64);
    let r = rho * (aj * ak).sqrt();
    let (lp, lm) = pair_eigen_check(aj, ak, r).expect("should be SPD");
    assert!(lp > 0.0 && lm > 0.0, "both eigenvalues must be positive");
    assert!(lp >= lm, "canonical: λ₊ ≥ λ₋");
    let tr_expected = aj + ak;
    let det_expected = aj * ak - r * r;
    assert!((lp + lm - tr_expected).abs() < 1e-12, "trace");
    assert!((lp * lm - det_expected).abs() < 1e-12, "det");
}

/// `pair_eigen_check`: rejects ρ=1 (singular block).
#[test]
fn pair_eigen_check_indefinite_rejected() {
    let (aj, ak, rho) = (0.5f64, 0.5f64, 1.0f64);
    let r = rho * (aj * ak).sqrt();
    assert!(pair_eigen_check(aj, ak, r).is_err(), "ρ=1 must fail");
}

/// `pair_eigen_check`: rejects shared-axis ρ=0.6 with cj=aj/2 (d≥4 interior).
#[test]
fn pair_eigen_check_shared_axis_threshold() {
    let aj = 0.5f64;
    let cj = aj / 2.0;
    let r = 0.6f64 * aj;
    assert!(
        pair_eigen_check(cj, cj, r).is_err(),
        "shared-axis ρ=0.6 should fail"
    );
}
