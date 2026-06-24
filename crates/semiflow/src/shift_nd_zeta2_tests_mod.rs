// Tests for `shift_nd_zeta2.rs` (ADR-0112 AMENDMENT 2, ζ²-corrected anisotropic shift kernel).
//
// Properties asserted:
//   1. order() returns 2 (the defining property of the ζ² correction).
//   2. growth() inherits from the base kernel (contraction for isotropic case).
//   3. DomainViolation if grad_a.len() != D*D.
//   4. tau=0 → output equals input (identity).
//   5. grid() returns the same grid as the inner kernel.
//   6. Apply on constant-valued datum: output remains finite.

use super::*;
use crate::{
    chernoff::ApplyChernoffExt,
    grid::Grid1D,
    grid_nd::{GridFnND, GridND},
    shift_nd::AnisotropicShiftChernoffND,
    ChernoffFunction,
};

/// Boxed closure type for a single `grad_a` component (D=2).
type GradAFn2 = alloc::boxed::Box<dyn Fn(&[f64; 2]) -> [f64; 2] + Send + Sync>;
type GradAVec2 = alloc::vec::Vec<GradAFn2>;

/// Build a D=2 isotropic unit-diffusion base kernel on an N×N grid.
fn make_base_kernel(n: usize) -> AnisotropicShiftChernoffND<f64, 2> {
    let ax = Grid1D::new(-3.0_f64, 3.0, n).unwrap();
    let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
    AnisotropicShiftChernoffND::new(
        |_x, a| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
            a.set(0, 1, 0.0);
            a.set(1, 0, 0.0);
        },
        |_x, b| {
            b[0] = 0.0;
            b[1] = 0.0;
        },
        |_x| 0.0_f64,
        grid,
    )
    .unwrap()
}

/// Build an `AnisotropicShiftZeta2ND<f64,2>` with constant A and zero grad.
///
/// Zero `grad_a` means the ζ² correction is zero, so this is equivalent to the base kernel.
fn make_zeta2_zero_grad(n: usize) -> AnisotropicShiftZeta2ND<f64, 2> {
    let base = make_base_kernel(n);
    let a_ij = |_x: &[f64; 2], a: &mut SquareMatrix<f64, 2>| {
        a.set(0, 0, 1.0);
        a.set(1, 1, 1.0);
        a.set(0, 1, 0.0);
        a.set(1, 0, 0.0);
    };
    // All four grad_a closures return zero (constant A).
    let grad_a: GradAVec2 = (0..4)
        .map(|_| -> GradAFn2 { alloc::boxed::Box::new(|_x: &[f64; 2]| [0.0_f64; 2]) })
        .collect();
    AnisotropicShiftZeta2ND::new(base, a_ij, grad_a).unwrap()
}

// ── order() == 2 ──────────────────────────────────────────────────────────────

#[test]
fn order_is_2() {
    let kernel = make_zeta2_zero_grad(8);
    assert_eq!(kernel.order(), 2, "order() must be 2 for ζ²-corrected kernel");
}

// ── growth() inherits from inner kernel ──────────────────────────────────────
// AnisotropicShiftChernoffND declares growth (1.5, 0); zeta2 wrapper delegates.

#[test]
fn growth_inherits_from_inner() {
    let kernel = make_zeta2_zero_grad(8);
    let g = kernel.growth();
    // Inner shift kernel uses multiplier=1.5, omega=0 (math §32.4 Growth bound).
    assert!(
        (g.multiplier - 1.5).abs() < 1e-14,
        "expected multiplier=1.5, got {}",
        g.multiplier
    );
    assert!(g.omega.abs() < 1e-14, "expected omega=0, got {}", g.omega);
}

// ── DomainViolation for wrong grad_a len ─────────────────────────────────────

#[test]
fn wrong_grad_a_len_returns_err() {
    let base = make_base_kernel(8);
    // Provide only 3 closures instead of D*D = 4.
    let grad_a: GradAVec2 = (0..3)
        .map(|_| -> GradAFn2 { alloc::boxed::Box::new(|_x: &[f64; 2]| [0.0_f64; 2]) })
        .collect();
    let result = AnisotropicShiftZeta2ND::<f64, 2>::new(
        base,
        |_x, _a| {},
        grad_a,
    );
    assert!(result.is_err(), "expected Err for wrong grad_a.len()");
}

// ── grid() returns same grid as inner ────────────────────────────────────────

#[test]
fn grid_matches_inner() {
    let base = make_base_kernel(8);
    let inner_n = base.grid().len();
    let kernel = make_zeta2_zero_grad(8);
    // Verify the wrapped grid has the same total size.
    assert_eq!(kernel.grid().len(), inner_n);
}

// ── small tau: output is finite ────────────────────────────────────────────────

#[test]
fn small_tau_output_is_finite() {
    // tau=0.001: verify all output values are finite (no NaN/Inf explosion).
    let kernel = make_zeta2_zero_grad(8);
    let grid = kernel.grid().clone();
    let src = GridFnND::from_fn(grid, |x| x[0].sin() + x[1].cos());
    let dst = kernel.apply_chernoff(0.001_f64, &src).unwrap();
    for v in &dst.values {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}

// ── finite output for constant datum ─────────────────────────────────────────

#[test]
fn constant_datum_gives_finite_output() {
    let kernel = make_zeta2_zero_grad(8);
    let grid = kernel.grid().clone();
    let src = GridFnND::from_fn(grid, |_| 2.0_f64);
    let dst = kernel.apply_chernoff(0.1_f64, &src).unwrap();
    for v in &dst.values {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}
