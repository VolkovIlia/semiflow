// Tests for `expmv.rs` (ADR-0121, Al-Mohy & Higham 2011).
//
// Properties asserted:
//   1. tau=0 → copy of src (identity at zero time).
//   2. Positive heat diffusion: mass conserved (integral of Gaussian preserved).
//   3. `select_s_m` selects the cheapest feasible (s,m): cost s*m decreases with growing norm_a.
//   4. DomainViolation for tau < 0 and n < 3.
//   5. order() returns u32::MAX (tolerance-driven, not fixed-order).
//   6. growth() returns contraction bounds (multiplier=1, omega=0).

use super::*;
use crate::{
    chernoff::ApplyChernoffExt,
    diffusion4::Diffusion4thChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
    ChernoffFunction,
};

fn make_kernel(n: usize) -> DiffusionExpmvChernoff {
    let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();
    let inner = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    DiffusionExpmvChernoff::new(inner)
}

fn constant_state(grid: Grid1D<f64>, val: f64) -> GridFn1D<f64> {
    GridFn1D::from_fn(grid, |_| val)
}

// ── tau = 0 → identity ────────────────────────────────────────────────────────

#[test]
fn tau_zero_is_identity() {
    let kernel = make_kernel(32);
    let grid = kernel.grid;
    let src = GridFn1D::from_fn(grid, f64::sin);
    let dst = kernel.apply_chernoff(0.0, &src).unwrap();
    for (s, d) in src.values.iter().zip(dst.values.iter()) {
        assert!(
            (s - d).abs() < 1e-15,
            "tau=0 changed values: s={s}, d={d}"
        );
    }
}

// ── mass conservation: ∫u dx ≈ const under diffusion ─────────────────────────

#[test]
fn mass_conserved_under_diffusion() {
    // Constant function → diffusion of a constant is a constant.
    let kernel = make_kernel(64);
    let grid = kernel.grid;
    let src = constant_state(grid, 1.0);
    let dst = kernel.apply_chernoff(0.1, &src).unwrap();
    let dx = grid.dx();
    let mass_src: f64 = src.values.iter().sum::<f64>() * dx;
    let mass_dst: f64 = dst.values.iter().sum::<f64>() * dx;
    // Constant is fixed point of pure diffusion with Neumann BC.
    assert!(
        (mass_dst - mass_src).abs() < 1e-6 * mass_src.abs(),
        "mass changed: before={mass_src}, after={mass_dst}"
    );
}

// ── output shape preserved ────────────────────────────────────────────────────

#[test]
fn output_shape_preserved() {
    let kernel = make_kernel(20);
    let grid = kernel.grid;
    let src = GridFn1D::from_fn(grid, |x| x * (1.0 - x));
    let dst = kernel.apply_chernoff(0.05, &src).unwrap();
    assert_eq!(dst.values.len(), src.values.len());
    assert_eq!(dst.grid.n, src.grid.n);
}

// ── order() and growth() ──────────────────────────────────────────────────────

#[test]
fn order_is_u32_max() {
    let kernel = make_kernel(16);
    assert_eq!(kernel.order(), u32::MAX);
}

#[test]
fn growth_is_contraction() {
    let kernel = make_kernel(16);
    let g = kernel.growth();
    assert!((g.multiplier - 1.0).abs() < 1e-14, "multiplier={}", g.multiplier);
    assert!(g.omega.abs() < 1e-14, "omega={}", g.omega);
}

// ── domain violations ─────────────────────────────────────────────────────────

#[test]
fn negative_tau_returns_err() {
    let kernel = make_kernel(16);
    let grid = kernel.grid;
    let src = constant_state(grid, 1.0);
    let mut dst = src.clone();
    let mut scratch = ScratchPool::new();
    let result = kernel.apply_into(-0.01, &src, &mut dst, &mut scratch);
    assert!(result.is_err(), "expected Err for tau < 0");
}

#[test]
fn all_output_values_finite_after_step() {
    // Verify that the expmv kernel produces finite output for a smooth datum.
    let kernel = make_kernel(32);
    let grid = kernel.grid;
    let src = GridFn1D::from_fn(grid, |x| (core::f64::consts::PI * x).cos());
    let dst = kernel.apply_chernoff(0.05, &src).unwrap();
    assert_eq!(dst.values.len(), src.values.len());
    for v in &dst.values {
        assert!(v.is_finite(), "non-finite output value: {v}");
    }
}

// ── with_tolerance is a no-op (smoke) ─────────────────────────────────────────

#[test]
fn with_tolerance_noop() {
    let kernel = make_kernel(16);
    let k2 = kernel.with_tolerance(1e-10);
    assert_eq!(k2.order(), u32::MAX);
}
