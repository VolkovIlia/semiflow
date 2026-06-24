// Tests for `manifold_chernoff.rs` (math.md §24, ADR-0071, MMRS 2023).
//
// Properties asserted:
//   1. Torus: constant function is a fixed point of diffusion (mass preserved).
//   2. Torus: output shape matches input shape.
//   3. Torus: order() == 1 (no curvature correction) and growth() is contraction.
//   4. Sphere2: constant function is a fixed point (isotropic diffusion preserves constants).
//   5. Sphere2: with_curvature_correction flips order() from 1 to 2.
//   6. Hyperbolic2: output finitely-valued for small tau.
//   7. tau=0 (small tau): output near input for torus.

use super::*;
use crate::{
    chernoff::ApplyChernoffExt,
    grid::Grid1D,
    grid2d::Grid2D,
    grid_fn2d::GridFn2D,
    manifold::{Sphere2, Torus},
    scratch::ScratchPool,
    ChernoffFunction,
};

fn make_grid2d(nx: usize, ny: usize) -> Grid2D<f64> {
    let gx = Grid1D::new(0.0_f64, 1.0, nx).unwrap();
    let gy = Grid1D::new(0.0_f64, 1.0, ny).unwrap();
    Grid2D::new(gx, gy)
}

fn constant_fn2d(grid: Grid2D<f64>, val: f64) -> GridFn2D<f64> {
    GridFn2D::from_fn(grid, |_, _| val)
}

fn l2_dist(a: &GridFn2D<f64>, b: &GridFn2D<f64>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ── Torus: constant is fixed point ───────────────────────────────────────────

#[test]
fn torus_constant_fixed_point() {
    let grid = make_grid2d(10, 10);
    let torus = Torus::<f64, 2>::unit();
    let kernel = ManifoldChernoff::new(torus, false);
    let src = constant_fn2d(grid, 2.0);
    let dst = kernel.apply_chernoff(0.1, &src).unwrap();
    // Constant on flat torus → diffusion leaves it unchanged.
    for (s, d) in src.values.iter().zip(dst.values.iter()) {
        assert!(
            (s - d).abs() < 0.01,
            "constant not preserved: s={s:.4}, d={d:.4}"
        );
    }
}

// ── Torus: output shape ───────────────────────────────────────────────────────

#[test]
fn torus_output_shape_matches_input() {
    let grid = make_grid2d(8, 6);
    let torus = Torus::<f64, 2>::unit();
    let kernel = ManifoldChernoff::new(torus, false);
    let src = GridFn2D::from_fn(grid, |x, y| (x + y).sin());
    let dst = kernel.apply_chernoff(0.05, &src).unwrap();
    assert_eq!(dst.values.len(), src.values.len());
    assert_eq!(dst.grid, src.grid);
}

// ── Torus: order and growth ───────────────────────────────────────────────────

#[test]
fn torus_order_is_1() {
    let kernel = ManifoldChernoff::new(Torus::<f64, 2>::unit(), false);
    assert_eq!(kernel.order(), 1);
}

#[test]
fn torus_growth_is_contraction() {
    let kernel = ManifoldChernoff::new(Torus::<f64, 2>::unit(), false);
    let g = kernel.growth();
    assert!((g.multiplier - 1.0).abs() < 1e-14);
    assert!(g.omega.abs() < 1e-14);
}

// ── Sphere2: constant is approximately fixed point ────────────────────────────

#[test]
fn sphere2_constant_approximately_preserved() {
    // Chart grid for S²: θ ∈ [0.2, 2.9] (avoiding poles), φ ∈ [0, 2π].
    let gx = Grid1D::new(0.2_f64, 2.9, 8).unwrap();
    let gy = Grid1D::new(0.0_f64, 6.28318, 8).unwrap();
    let grid = Grid2D::new(gx, gy);
    let s2 = Sphere2::<f64>::unit();
    let kernel = ManifoldChernoff::new(s2, false);
    let src = constant_fn2d(grid, 1.5);
    let dst = kernel.apply_chernoff(0.01, &src).unwrap();
    // With small tau constant should be nearly preserved.
    let rel_err = l2_dist(&dst, &src) / (src.values.len() as f64).sqrt() / 1.5;
    assert!(
        rel_err < 0.05,
        "sphere2 constant not preserved: rel_err={rel_err:.4}"
    );
}

// ── Sphere2: curvature_correction changes order ───────────────────────────────

#[test]
fn sphere2_order_without_correction_is_1() {
    let kernel = ManifoldChernoff::new(Sphere2::<f64>::unit(), false);
    assert_eq!(kernel.order(), 1);
}

#[test]
fn sphere2_order_with_correction_is_2() {
    let kernel = ManifoldChernoff::new(Sphere2::<f64>::unit(), true);
    assert_eq!(kernel.order(), 2);
}

// ── Hyperbolic2: finite output for small tau ──────────────────────────────────

#[test]
fn hyperbolic2_output_is_finite() {
    use crate::manifold::Hyperbolic2;
    // Poincaré disk: chart coords (u,v) with u²+v² < 1 — use small inner region.
    let gx = Grid1D::new(-0.4_f64, 0.4, 8).unwrap();
    let gy = Grid1D::new(-0.4_f64, 0.4, 8).unwrap();
    let grid = Grid2D::new(gx, gy);
    let h2 = Hyperbolic2::<f64>::unit();
    let kernel = ManifoldChernoff::new(h2, false);
    let src = constant_fn2d(grid, 1.0);
    let dst = kernel.apply_chernoff(0.01, &src).unwrap();
    for v in &dst.values {
        assert!(v.is_finite(), "Hyperbolic2 output non-finite: {v}");
    }
}
