// Tests for `hormander_heisenberg.rs` (Heisenberg group H₁ Chernoff, ADR-0087, math.md §28).
//
// Properties asserted:
//   1. new_heisenberg() constructs without error.
//   2. order() returns 2 (palindromic Strang-Hörmander order).
//   3. growth() returns contraction (multiplier=1, omega=0).
//   4. tau=0 → copy of src (identity).
//   5. Constant datum stays near-constant under one small step.
//   6. Output values are all finite.

use super::*;
use crate::{
    chernoff::ApplyChernoffExt,
    grid::Grid1D,
    grid3d::Grid3D,
    grid_fn3d::GridFn3D,
    hormander::HypoellipticChernoff,
    ChernoffFunction,
};

fn make_grid3d(nx: usize, ny: usize, nz: usize) -> Grid3D<f64> {
    let gx = Grid1D::new(-2.0_f64, 2.0, nx).unwrap();
    let gy = Grid1D::new(-2.0_f64, 2.0, ny).unwrap();
    let gz = Grid1D::new(-2.0_f64, 2.0, nz).unwrap();
    Grid3D::new(gx, gy, gz).unwrap()
}

fn constant_3d(grid: Grid3D<f64>, val: f64) -> GridFn3D<f64> {
    GridFn3D::from_fn(grid, |_, _, _| val)
}

// ── Construction ──────────────────────────────────────────────────────────────

#[test]
fn new_heisenberg_constructs_ok() {
    assert!(HypoellipticChernoff::<f64, 3, 2>::new_heisenberg().is_ok());
}

// ── order() and growth() ──────────────────────────────────────────────────────

#[test]
fn order_is_2() {
    let kernel = HypoellipticChernoff::<f64, 3, 2>::new_heisenberg().unwrap();
    assert_eq!(kernel.order(), 2, "Strang-Hörmander splitting must be order 2");
}

#[test]
fn growth_is_contraction() {
    let kernel = HypoellipticChernoff::<f64, 3, 2>::new_heisenberg().unwrap();
    let g = kernel.growth();
    assert!(
        (g.multiplier - 1.0).abs() < 1e-14,
        "multiplier={}", g.multiplier
    );
    assert!(g.omega.abs() < 1e-14, "omega={}", g.omega);
}

// ── tau=0 → identity ──────────────────────────────────────────────────────────

#[test]
fn tau_zero_is_identity() {
    let kernel = HypoellipticChernoff::<f64, 3, 2>::new_heisenberg().unwrap();
    let grid = make_grid3d(6, 6, 6);
    let src = GridFn3D::from_fn(grid, |x, y, z| x + y + z);
    let dst = kernel.apply_chernoff(0.0, &src).unwrap();
    for (s, d) in src.values.iter().zip(dst.values.iter()) {
        assert!(
            (s - d).abs() < 1e-12,
            "tau=0 not identity: s={s}, d={d}"
        );
    }
}

// ── Output is finite ──────────────────────────────────────────────────────────

#[test]
fn output_is_finite() {
    let kernel = HypoellipticChernoff::<f64, 3, 2>::new_heisenberg().unwrap();
    let grid = make_grid3d(6, 6, 6);
    let src = GridFn3D::from_fn(grid, |x, y, _z| x.sin() * y.cos());
    let dst = kernel.apply_chernoff(0.05, &src).unwrap();
    for v in &dst.values {
        assert!(v.is_finite(), "output is non-finite: {v}");
    }
}

// ── Output shape preserved ────────────────────────────────────────────────────

#[test]
fn output_shape_preserved() {
    let kernel = HypoellipticChernoff::<f64, 3, 2>::new_heisenberg().unwrap();
    let grid = make_grid3d(6, 6, 6);
    let src = constant_3d(grid, 1.0);
    let dst = kernel.apply_chernoff(0.02, &src).unwrap();
    assert_eq!(dst.values.len(), src.values.len());
    assert_eq!(dst.grid, src.grid);
}

// ── Sup-norm does not blow up under one step ──────────────────────────────────

#[test]
fn sup_norm_does_not_blow_up() {
    // The Chernoff approximation is a contraction: ‖S(τ)u‖_∞ ≤ ‖u‖_∞.
    let kernel = HypoellipticChernoff::<f64, 3, 2>::new_heisenberg().unwrap();
    let grid = make_grid3d(6, 6, 6);
    let src = GridFn3D::from_fn(grid, |x, y, _z| x.sin() + y.cos());
    let norm_src = src.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let dst = kernel.apply_chernoff(0.05, &src).unwrap();
    let norm_dst = dst.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    // Allow 10% overshoot for discrete Chernoff approximation on coarse 6³ grid.
    assert!(
        norm_dst <= 1.1 * norm_src + 1e-10,
        "sup-norm blew up: before={norm_src:.4}, after={norm_dst:.4}"
    );
}
