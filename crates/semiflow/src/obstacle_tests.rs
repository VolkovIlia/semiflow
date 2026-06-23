// Grid counts/lengths (usize) cast to f64 for error-reporting values; always ≪ 2^52.
#![allow(clippy::cast_precision_loss)]
// Exact float comparisons in tests verify projection identity and sentinel values.
#![allow(clippy::float_cmp)]

use super::{ClosureObstacle, ConstantObstacle, Obstacle, ObstacleChernoff};
use crate::{
    chernoff::ChernoffFunction, diffusion::DiffusionChernoff, error::SemiflowError, grid::Grid1D,
    grid_fn::GridFn1D, scratch::ScratchPool,
};

// --- ConstantObstacle construction ---

#[test]
fn constant_obstacle_finite_ok() {
    assert!(ConstantObstacle::<f64>::new(-0.5).is_ok());
}

#[test]
fn constant_obstacle_nan_err() {
    let err = ConstantObstacle::<f64>::new(f64::NAN).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

// --- projection = max(·, g) ---

#[test]
fn constant_projection_lifts_below_floor() {
    let grid = Grid1D::new(0.0_f64, 1.0, 5).unwrap();
    let mut u = GridFn1D::from_fn(grid, |x| x - 0.5); // ranges [-0.5, 0.5]
    let obs = ConstantObstacle::new(0.0_f64).unwrap();
    obs.project_in_place(&mut u).unwrap();
    // All values clamped up to >= 0.
    for &v in &u.values {
        assert!(v >= 0.0, "value {v} below obstacle 0");
    }
    // The node that was already above the floor (x=1.0 -> 0.5) is untouched.
    assert_eq!(u.values[4], 0.5);
}

#[test]
fn closure_projection_matches_pointwise_max() {
    let grid = Grid1D::new(0.0_f64, 1.0, 7).unwrap();
    let mut u = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let g = |x: f64| 0.2 * (x - 0.5); // signed ramp
    let obs = ClosureObstacle::new(g);
    obs.project_in_place(&mut u).unwrap();
    for i in 0..grid.n {
        let x = grid.x_at(i);
        assert_eq!(u.values[i], 0.0_f64.max(g(x)));
    }
}

// --- active set is the strict continuation set {w > g} ---

#[test]
fn active_set_is_strict_continuation_set() {
    let grid = Grid1D::new(0.0_f64, 1.0, 5).unwrap();
    let w = GridFn1D::from_fn(grid, |x| x); // 0, .25, .5, .75, 1
    let obs = ConstantObstacle::new(0.5_f64).unwrap();
    let mut active = [false; 5];
    obs.active_set_into(&w, &mut active).unwrap();
    // x in {0,.25,.5} -> w <= 0.5 -> inactive; x in {.75,1} -> active.
    assert_eq!(active, [false, false, false, true, true]);
}

#[test]
fn active_set_length_mismatch_err() {
    let grid = Grid1D::new(0.0_f64, 1.0, 5).unwrap();
    let w = GridFn1D::from_fn(grid, |_| 1.0_f64);
    let obs = ConstantObstacle::new(0.0_f64).unwrap();
    let mut active = [false; 3];
    let err = obs.active_set_into(&w, &mut active).unwrap_err();
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

// --- ObstacleChernoff: post-projection + order/lower-bound ---

#[test]
fn obstacle_chernoff_projects_after_step() {
    let grid = Grid1D::new(0.0_f64, 1.0, 9).unwrap();
    let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let obs = ConstantObstacle::new(0.3_f64).unwrap();
    let kernel = ObstacleChernoff::new(diff, obs).unwrap();
    let u0 = GridFn1D::from_fn(grid, |_| 0.1_f64); // everywhere below the floor
    let mut dst = u0.zeroed_like();
    let mut scratch = ScratchPool::new();
    kernel
        .apply_into(0.001, &u0, &mut dst, &mut scratch)
        .unwrap();
    // Post-projection guarantees the lower bound V >= g everywhere.
    for &v in &dst.values {
        assert!(v >= 0.3 - 1e-12, "value {v} below obstacle 0.3");
    }
    assert_eq!(kernel.order(), 1);
}

#[test]
fn obstacle_chernoff_growth_inherits_inner() {
    let grid = Grid1D::new(0.0_f64, 1.0, 5).unwrap();
    let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let inner_growth = diff.growth();
    let obs = ConstantObstacle::new(-1.0_f64).unwrap();
    let kernel = ObstacleChernoff::new(diff, obs).unwrap();
    assert_eq!(kernel.growth(), inner_growth);
}

#[test]
fn active_set_adjoint_unsupported_inner_errs() {
    // DiffusionChernoff does not override apply_adjoint_into -> UnsupportedOperation.
    let grid = Grid1D::new(0.0_f64, 1.0, 5).unwrap();
    let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let obs = ConstantObstacle::new(0.0_f64).unwrap();
    let kernel = ObstacleChernoff::new(diff, obs).unwrap();
    let w = GridFn1D::from_fn(grid, |_| 1.0_f64);
    let lam = GridFn1D::from_fn(grid, |_| 1.0_f64);
    let mut out = w.zeroed_like();
    let mut scratch = ScratchPool::new();
    let err = kernel
        .apply_active_set_adjoint_into(0.01, &w, &lam, &mut out, &mut scratch)
        .unwrap_err();
    assert!(matches!(err, SemiflowError::UnsupportedOperation { .. }));
}
