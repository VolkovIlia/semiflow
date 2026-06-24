// Tests for `resolvent_residual.rs` (LaplaceChernoffResolventResidual, ADR-0083).
//
// Properties asserted:
//   1. Construction succeeds (budget stored correctly).
//   2. budget() returns the configured value.
//   3. verify_residual returns Err for lambda <= 0.
//   4. verify_residual returns Err for lambda = NaN.
//   5. verify_residual returns Ok and finite residual for valid lambda.
//   6. fresh_from_fn creates a GridFn1D with correct values.
//   7. sample_at returns Err for empty coordinate slice.
//   8. sample_at returns Ok for a valid interior point.

use crate::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    resolvent::{LaplaceChernoffResolvent, LaplaceQuadrature},
    resolvent_residual::{LaplaceChernoffResolventResidual, Sampleable},
};

fn make_residual_wrapper(
    n: usize,
) -> LaplaceChernoffResolventResidual<DiffusionChernoff<f64>, f64> {
    let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
    let kernel = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let resolvent = LaplaceChernoffResolvent::new(kernel, 4, LaplaceQuadrature::GaussLaguerre32)
        .expect("resolvent construction failed");
    LaplaceChernoffResolventResidual::new(resolvent, 1e-3_f64)
}

fn make_datum(n: usize) -> GridFn1D<f64> {
    let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
    GridFn1D::from_fn(grid, |x| (-(x - 0.5) * (x - 0.5) * 50.0).exp())
}

// ── Construction ──────────────────────────────────────────────────────────────

#[test]
fn construction_succeeds() {
    let wrapper = make_residual_wrapper(32);
    assert!((wrapper.budget() - 1e-3).abs() < 1e-15, "budget mismatch");
}

// ── budget() ─────────────────────────────────────────────────────────────────

#[test]
fn budget_returns_configured_value() {
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let kernel = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    let resolvent =
        LaplaceChernoffResolvent::new(kernel, 4, LaplaceQuadrature::GaussLaguerre32).unwrap();
    let wrapper = LaplaceChernoffResolventResidual::new(resolvent, 5e-4_f64);
    assert!((wrapper.budget() - 5e-4).abs() < 1e-15);
}

// ── verify_residual: Err for lambda <= 0 ─────────────────────────────────────

#[test]
fn verify_residual_rejects_nonpositive_lambda() {
    let wrapper = make_residual_wrapper(32);
    let f = make_datum(32);
    assert!(
        wrapper.verify_residual(0.0_f64, &f).is_err(),
        "expected Err for lambda=0"
    );
    assert!(
        wrapper.verify_residual(-1.0_f64, &f).is_err(),
        "expected Err for lambda=-1"
    );
}

// ── verify_residual: Err for lambda = NaN ────────────────────────────────────

#[test]
fn verify_residual_rejects_nan_lambda() {
    let wrapper = make_residual_wrapper(32);
    let f = make_datum(32);
    assert!(
        wrapper.verify_residual(f64::NAN, &f).is_err(),
        "expected Err for NaN lambda"
    );
}

// ── verify_residual: Ok for valid lambda ─────────────────────────────────────

#[test]
fn verify_residual_ok_for_positive_lambda() {
    let wrapper = make_residual_wrapper(32);
    let f = make_datum(32);
    let result = wrapper.verify_residual(1.0_f64, &f);
    assert!(result.is_ok(), "verify_residual failed: {:?}", result);
    let err = result.unwrap();
    assert!(err.is_finite(), "residual is non-finite: {err}");
    assert!(err >= 0.0, "residual is negative: {err}");
}

// ── Sampleable: fresh_from_fn ─────────────────────────────────────────────────

#[test]
fn fresh_from_fn_gives_correct_values() {
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let template = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let result = template.fresh_from_fn(&|x| x[0] * x[0]).unwrap();
    assert_eq!(result.values.len(), 16);
    for (i, &v) in result.values.iter().enumerate() {
        let x = grid.x_at(i);
        assert!((v - x * x).abs() < 1e-15, "at i={i}: expected {}, got {v}", x * x);
    }
}

// ── Sampleable: sample_at empty coords ───────────────────────────────────────

#[test]
fn sample_at_empty_coords_returns_err() {
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let gf = GridFn1D::from_fn(grid, |x| x.sin());
    let result = gf.sample_at(&[]);
    assert!(result.is_err(), "expected Err for empty coordinate slice");
}

// ── Sampleable: sample_at interior point ─────────────────────────────────────

#[test]
fn sample_at_valid_point_returns_ok() {
    let grid = Grid1D::new(0.0_f64, 1.0, 16).unwrap();
    let gf = GridFn1D::from_fn(grid, |x| x * 2.0);
    let result = gf.sample_at(&[0.5_f64]);
    assert!(result.is_ok(), "sample_at failed: {:?}", result);
    let v = result.unwrap();
    assert!(v.is_finite(), "sample_at returned non-finite: {v}");
}
