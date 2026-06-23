//! Unit tests for `NonSeparable2DChernoff`.
//!
//! 6 tests per architect spec (ADR-0016, math.md §10.7-bis).

use semiflow::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    DiffusionChernoff, Grid1D, Grid2D, GridFn2D, NonSeparable2DChernoff, SemiflowError,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_grid(n: usize) -> Grid2D {
    let g = Grid1D::new(-1.0, 1.0, n).unwrap();
    Grid2D::new(g, g)
}

fn make_inner(n: usize) -> (DiffusionChernoff, DiffusionChernoff) {
    let gx = Grid1D::new(-1.0, 1.0, n).unwrap();
    let ix = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx);
    let iy = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx);
    (ix, iy)
}

fn make_op(
    n: usize,
    c: fn(f64, f64) -> f64,
    c_norm: f64,
) -> NonSeparable2DChernoff<DiffusionChernoff, DiffusionChernoff> {
    let (ix, iy) = make_inner(n);
    NonSeparable2DChernoff::new(ix, iy, c, c_norm, make_grid(n)).unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// T1: `order()` == 2 (τ-axis, math.md §11.1.bis).
#[test]
fn order_is_two() {
    let op = make_op(16, |_, _| 0.0, 0.0);
    assert_eq!(op.order(), 2);
}

/// T2: `growth()` returns finite positive values.
#[test]
fn growth_finite_positive() {
    let op = make_op(16, |_, _| 0.05, 0.05);
    let g = op.growth();
    assert!(
        g.multiplier.is_finite() && g.multiplier > 0.0,
        "M must be finite positive, got {}",
        g.multiplier
    );
    assert!(g.omega.is_finite(), "omega must be finite, got {}", g.omega);
}

/// T3: `c_norm_bound` < 0 → `DomainViolation`.
#[test]
fn negative_c_norm_rejected() {
    let (ix, iy) = make_inner(8);
    let g2 = make_grid(8);
    let result = NonSeparable2DChernoff::new(ix, iy, |_, _| 0.0, -0.1, g2);
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}

/// T4: CFL violation → `CflViolated` error.
#[test]
fn cfl_violation_returned() {
    let op = make_op(8, |_, _| 1.0, 1.0);
    let grid = make_grid(8);
    let f = GridFn2D::from_fn(grid, |_, _| 0.0);
    // dx = 2/7, dy = 2/7; 4*1.0*1.0 >= (2/7)^2 ≈ 0.082 → violated
    assert!(matches!(
        op.apply_chernoff(1.0, &f),
        Err(SemiflowError::CflViolated { .. })
    ));
}

/// T5: tau=0 → output equals input (identity at zero step).
#[test]
fn apply_tau_zero_is_near_identity() {
    let op = make_op(16, |x, y| 0.05 * (x * y), 0.05);
    let grid = make_grid(16);
    let f = GridFn2D::from_fn(grid, |x, y| (-x * x - y * y).exp());
    let out = op.apply_chernoff(0.0, &f).unwrap();
    let nx = 16_usize;
    let ny = 16_usize;
    for j in 0..ny {
        for i in 0..nx {
            let k = j * nx + i;
            assert!(
                (out.values[k] - f.values[k]).abs() < 1e-12,
                "tau=0 mismatch at ({i},{j}): out={}, in={}",
                out.values[k],
                f.values[k]
            );
        }
    }
}

/// T6: shape (nx * ny) preserved after apply.
#[test]
fn apply_preserves_shape() {
    let op = make_op(12, |x, y| 0.01 * x * y, 0.01);
    let grid = make_grid(12);
    let n = 12_usize;
    let f = GridFn2D::from_fn(grid, |x, y| (x + y).sin());
    let tau = 1e-3;
    let out = op.apply_chernoff(tau, &f).unwrap();
    assert_eq!(out.values.len(), n * n, "shape mismatch");
}
