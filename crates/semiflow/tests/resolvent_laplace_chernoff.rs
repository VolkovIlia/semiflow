//! G24 acceptance tests — Laplace-Chernoff Resolvent (math.md §22.5, ADR-0069).
//!
//! Unit-level regression tests for [`LaplaceChernoffResolvent`] and [`Sampleable`].
//! Heavy convergence sweep (G24) is in a separate `#[cfg(feature = "slow-tests")]`
//! block; this file runs fast (< 1 s).

use semiflow::{
    boundary::InterpKind,
    diffusion::DiffusionChernoff,
    error::SemiflowError,
    grid::Grid1D,
    grid_fn::GridFn1D,
    resolvent::{LaplaceChernoffResolvent, LaplaceQuadrature, Sampleable},
    State,
};

fn make_resolvent(n: usize, grid_n: usize) -> LaplaceChernoffResolvent<DiffusionChernoff<f64>> {
    let grid = Grid1D::new(-5.0, 5.0, grid_n).unwrap();
    let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    LaplaceChernoffResolvent::new(diff, n, LaplaceQuadrature::GaussLaguerre32).unwrap()
}

#[test]
fn new_rejects_n_zero() {
    let grid = Grid1D::new(-5.0, 5.0, 32).unwrap();
    let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let result = LaplaceChernoffResolvent::new(diff, 0, LaplaceQuadrature::GaussLaguerre32);
    let err = result.err().expect("expected Err for n=0");
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

#[test]
fn new_rejects_trapezoid_nan_tmax() {
    let grid = Grid1D::new(-5.0, 5.0, 32).unwrap();
    let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let result = LaplaceChernoffResolvent::new(
        diff,
        1,
        LaplaceQuadrature::TrapezoidWithTail { t_max: f64::NAN },
    );
    let err = result.err().expect("expected Err for NaN t_max");
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

#[test]
fn new_rejects_trapezoid_negative_tmax() {
    let grid = Grid1D::new(-5.0, 5.0, 32).unwrap();
    let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let result = LaplaceChernoffResolvent::new(
        diff,
        1,
        LaplaceQuadrature::TrapezoidWithTail { t_max: -1.0_f64 },
    );
    let err = result.err().expect("expected Err for negative t_max");
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

#[test]
fn new_accepts_valid_gauss_laguerre() {
    let grid = Grid1D::new(-5.0, 5.0, 32).unwrap();
    let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let res = LaplaceChernoffResolvent::new(diff, 1, LaplaceQuadrature::GaussLaguerre32);
    assert!(res.is_ok());
}

#[test]
fn eval_rejects_nonpositive_lambda() {
    let resolvent = make_resolvent(1, 32);
    let grid = Grid1D::new(-5.0, 5.0, 32).unwrap();
    let g = GridFn1D::from_fn(grid, |x: f64| (-x * x).exp());
    let err = resolvent.eval(0.0_f64, &g).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

#[test]
fn eval_rejects_negative_lambda() {
    let resolvent = make_resolvent(1, 32);
    let grid = Grid1D::new(-5.0, 5.0, 32).unwrap();
    let g = GridFn1D::from_fn(grid, |x: f64| (-x * x).exp());
    let err = resolvent.eval(-1.0_f64, &g).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

/// Smoke test: `eval` with near-identity inner at `lambda = 1`, `n = 1`.
///
/// For a near-identity `C(tau) ≈ I`, `(C(tau))^1 g ≈ g`. Then
/// `R̃_1(1) g ≈ (1/lambda) · (sum_k w_k) · g`. With `sum_k w_k = 1`
/// (Gauss-Laguerre exactness on the constant integrand), the result is `≈ g`.
#[test]
fn eval_near_identity_inner_gauss_laguerre() {
    let grid = Grid1D::new(-2.0, 2.0, 16).unwrap();
    let diff = DiffusionChernoff::new(|_| 1e-10_f64, |_| 0.0, |_| 0.0, 1e-10, grid);
    let resolvent =
        LaplaceChernoffResolvent::new(diff, 1, LaplaceQuadrature::GaussLaguerre32).unwrap();
    let g = GridFn1D::from_fn(grid, |_: f64| 1.0_f64);
    let rg = resolvent.eval(1.0_f64, &g).unwrap();
    let sup = rg.norm_sup();
    assert!(
        (sup - 1.0).abs() < 0.01,
        "expected ≈ 1.0, got {sup:.6} (GL weight sum ≈ 1)"
    );
}

/// Test `Sampleable` impl for `GridFn1D`: `sample_at` and `fresh_from_fn`.
///
/// Pin to `CubicHermite`: `sample_at` → `sample_generic` → `interp_generic`,
/// which does not support `SepticHermite` (f64-specific kernel). v6.0 changed
/// `Grid1D::new` default to `SepticHermite` (ADR-0109); pin to `CubicHermite` so
/// this round-trip test works on the generic Sampleable<f64> path.
#[test]
fn sampleable_gridfn1d_round_trip() {
    let grid = Grid1D::new(-1.0, 1.0, 8)
        .unwrap()
        .with_interp(InterpKind::CubicHermite);
    let proto = GridFn1D::from_fn(grid, |x: f64| x * x);
    let rebuilt = proto
        .fresh_from_fn(&|coords: &[f64]| coords[0] * 2.0)
        .unwrap();
    let v = rebuilt.sample_at(&[0.5_f64]).unwrap();
    assert!((v - 1.0).abs() < 0.05, "expected ≈ 1.0, got {v}");
}

#[test]
fn sampleable_sample_at_empty_returns_error() {
    let grid = Grid1D::new(-1.0, 1.0, 8).unwrap();
    let g = GridFn1D::from_fn(grid, |_: f64| 0.0);
    let err = g.sample_at(&[]).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}
