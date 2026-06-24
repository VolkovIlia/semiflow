// Tests for `nonseparable_mixed_closure.rs`.
//
// Properties asserted:
//   1. with_closure_c: constructs successfully with valid norm_bound.
//   2. with_closure_c: returns DomainViolation for negative/inf norm_bound.
//   3. with_closure_beta: constructs successfully.
//   4. with_closure_beta: returns DomainViolation for NaN norm_bound.
//   5. The constructed kernel applies without error on a small grid.
//   6. Zero coupling norm_bound (c=0 fast-path) still gives finite output.

use super::*;
use alloc::sync::Arc;
use crate::{
    chernoff::ApplyChernoffExt,
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid2d::Grid2D,
    grid_fn2d::GridFn2D,
};

fn make_1d_kernel(n: usize) -> DiffusionChernoff<f64> {
    let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
    DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid)
}

fn make_grid2d(nx: usize, ny: usize) -> Grid2D<f64> {
    let gx = Grid1D::new(0.0_f64, 1.0, nx).unwrap();
    let gy = Grid1D::new(0.0_f64, 1.0, ny).unwrap();
    Grid2D::new(gx, gy)
}

// ── with_closure_c: valid construction ───────────────────────────────────────

#[test]
fn with_closure_c_constructs_ok() {
    let x = make_1d_kernel(8);
    let y = make_1d_kernel(8);
    let grid = make_grid2d(8, 8);
    let c_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync> = Arc::new(|_x, _y| 0.1_f64);
    let result = with_closure_c(x, y, c_fn, 0.1, grid);
    assert!(result.is_ok(), "expected Ok for valid c norm_bound");
}

// ── with_closure_c: invalid norm_bound ───────────────────────────────────────

#[test]
fn with_closure_c_negative_norm_bound_errors() {
    let x = make_1d_kernel(8);
    let y = make_1d_kernel(8);
    let grid = make_grid2d(8, 8);
    let c_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync> = Arc::new(|_, _| 0.0_f64);
    let result = with_closure_c(x, y, c_fn, -1.0, grid);
    assert!(result.is_err(), "expected Err for negative norm_bound");
}

#[test]
fn with_closure_c_inf_norm_bound_errors() {
    let x = make_1d_kernel(8);
    let y = make_1d_kernel(8);
    let grid = make_grid2d(8, 8);
    let c_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync> = Arc::new(|_, _| 0.0_f64);
    let result = with_closure_c(x, y, c_fn, f64::INFINITY, grid);
    assert!(result.is_err(), "expected Err for inf norm_bound");
}

// ── with_closure_beta: valid construction ────────────────────────────────────

#[test]
fn with_closure_beta_constructs_ok() {
    let x = make_1d_kernel(8);
    let y = make_1d_kernel(8);
    let grid = make_grid2d(8, 8);
    let beta_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync> = Arc::new(|_, _| 0.05_f64);
    let result = with_closure_beta(x, y, beta_fn, 0.05, grid);
    assert!(result.is_ok(), "expected Ok for valid beta norm_bound");
}

#[test]
fn with_closure_beta_nan_norm_bound_errors() {
    let x = make_1d_kernel(8);
    let y = make_1d_kernel(8);
    let grid = make_grid2d(8, 8);
    let beta_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync> = Arc::new(|_, _| 0.0_f64);
    let result = with_closure_beta(x, y, beta_fn, f64::NAN, grid);
    assert!(result.is_err(), "expected Err for NaN norm_bound");
}

// ── kernel applies without error ──────────────────────────────────────────────

#[test]
fn closure_c_kernel_apply_ok() {
    let x = make_1d_kernel(8);
    let y = make_1d_kernel(8);
    let grid = make_grid2d(8, 8);
    let c_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync> = Arc::new(|_, _| 0.0_f64);
    let kernel = with_closure_c(x, y, c_fn, 0.0, grid).unwrap();
    let src = GridFn2D::from_fn(grid, |x, y| x + y);
    let dst = kernel.apply_chernoff(0.05, &src).unwrap();
    assert!(dst.values.iter().all(|v| v.is_finite()), "non-finite output");
}

// ── zero coupling fast-path ───────────────────────────────────────────────────

#[test]
fn zero_coupling_gives_finite_output() {
    let x = make_1d_kernel(8);
    let y = make_1d_kernel(8);
    let grid = make_grid2d(8, 8);
    let c_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync> = Arc::new(|_, _| 0.0_f64);
    // norm_bound = 0.0 triggers the is_zero fast-path.
    let kernel = with_closure_c(x, y, c_fn, 0.0, grid).unwrap();
    let src = GridFn2D::from_fn(grid, |x, _y| x.sin());
    let dst = kernel.apply_chernoff(0.01, &src).unwrap();
    for v in &dst.values {
        assert!(v.is_finite(), "non-finite output from zero-coupling path");
    }
}

// ── with_closure_beta: non-zero coupling applies ok ──────────────────────────

#[test]
fn beta_coupling_apply_gives_finite_output() {
    let x = make_1d_kernel(8);
    let y = make_1d_kernel(8);
    let grid = make_grid2d(8, 8);
    // Small non-zero beta coupling.
    let beta_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync> = Arc::new(|_, _| 0.05_f64);
    let kernel = with_closure_beta(x, y, beta_fn, 0.05, grid).unwrap();
    let src = GridFn2D::from_fn(grid, |x, y| x.sin() + y.cos());
    let dst = kernel.apply_chernoff(0.01, &src).unwrap();
    assert_eq!(dst.values.len(), src.values.len());
    for v in &dst.values {
        assert!(v.is_finite(), "non-finite output from beta coupling: {v}");
    }
}

// ── clone: Clone propagates through ClosureBetaCoupling ─────────────────────

#[test]
fn beta_coupling_clone_works() {
    let x = make_1d_kernel(8);
    let y = make_1d_kernel(8);
    let grid = make_grid2d(8, 8);
    let beta_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync> = Arc::new(|_, _| 0.02_f64);
    let kernel = with_closure_beta(x, y, beta_fn, 0.02, grid).unwrap();
    // Clone exercises clone_box() on ClosureBetaCoupling.
    let kernel2 = kernel.clone();
    let src = GridFn2D::from_fn(grid, |x, _y| x.sin());
    let dst = kernel2.apply_chernoff(0.01, &src).unwrap();
    for v in &dst.values {
        assert!(v.is_finite(), "non-finite from cloned beta kernel");
    }
}

// ── zero_coupling beta fast-path ──────────────────────────────────────────────

#[test]
fn zero_beta_coupling_gives_finite_output() {
    let x = make_1d_kernel(8);
    let y = make_1d_kernel(8);
    let grid = make_grid2d(8, 8);
    // norm_bound = 0.0 triggers is_zero fast-path for ClosureBetaCoupling.
    let beta_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync> = Arc::new(|_, _| 0.0_f64);
    let kernel = with_closure_beta(x, y, beta_fn, 0.0, grid).unwrap();
    let src = GridFn2D::from_fn(grid, |x, _y| x.sin());
    let dst = kernel.apply_chernoff(0.01, &src).unwrap();
    for v in &dst.values {
        assert!(v.is_finite(), "non-finite from zero-beta fast-path");
    }
}
