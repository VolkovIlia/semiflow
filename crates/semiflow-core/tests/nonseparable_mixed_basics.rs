//! Constructor smoke tests for `NonSeparableMixedChernoff`.
//!
//! Verifies:
//! - `with_scalar_c` and `with_beta` constructors succeed/fail correctly.
//! - Type aliases compile and behave identically.
//! - `new` (backwards-compat constructor) delegates to `with_scalar_c`.
//! - `order()` is 2.
//! - Growth bound is finite and > 1.
//! - CFL gate triggered at tau=1 on 8-node grid.
//! - Apply returns correct shape.
//!
//! See ADR-0058, math.md §10.7-ter, §18.

use semiflow_core::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    BoundaryPolicy, DiffusionChernoff, Grid1D, Grid2D, GridFn2D, NonSeparable2DAnisotropicChernoff,
    NonSeparable2DChernoff, NonSeparableMixedChernoff, SemiflowError,
};

fn diffusion_inner(n: usize) -> DiffusionChernoff {
    let gx = Grid1D::new(-1.0, 1.0, n).unwrap();
    DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx)
}

fn make_grid(n: usize) -> Grid2D<f64> {
    let g = Grid1D::new(-1.0, 1.0, n).unwrap();
    Grid2D::new(g, g)
}

// ---------------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------------

#[test]
fn with_scalar_c_zero_ok() {
    let op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(16),
        diffusion_inner(16),
        |_, _| 0.0,
        0.0,
        make_grid(16),
    );
    assert!(op.is_ok());
}

#[test]
fn with_scalar_c_nonzero_ok() {
    let op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(16),
        diffusion_inner(16),
        |_, _| 0.3,
        0.3,
        make_grid(16),
    );
    assert!(op.is_ok());
}

#[test]
fn with_beta_ok() {
    let op = NonSeparableMixedChernoff::with_beta(
        diffusion_inner(16),
        diffusion_inner(16),
        |x, _| 0.05 * (-x * x).exp(),
        0.05,
        make_grid(16),
    );
    assert!(op.is_ok());
}

#[test]
fn negative_norm_rejected_scalar() {
    let result = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(8),
        diffusion_inner(8),
        |_, _| 0.0_f64,
        -0.1,
        make_grid(8),
    );
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}

#[test]
fn negative_norm_rejected_beta() {
    let result = NonSeparableMixedChernoff::with_beta(
        diffusion_inner(8),
        diffusion_inner(8),
        |_, _| 0.0_f64,
        -1.0,
        make_grid(8),
    );
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}

#[test]
fn nan_norm_rejected() {
    let result = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(8),
        diffusion_inner(8),
        |_, _| 0.0_f64,
        f64::NAN,
        make_grid(8),
    );
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}

// ---------------------------------------------------------------------------
// ChernoffFunction trait methods
// ---------------------------------------------------------------------------

#[test]
fn order_is_two() {
    let op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(16),
        diffusion_inner(16),
        |_, _| 0.0,
        0.0,
        make_grid(16),
    )
    .unwrap();
    assert_eq!(op.order(), 2);
}

#[test]
fn growth_is_finite_and_above_one() {
    let op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(16),
        diffusion_inner(16),
        |_, _| 0.0,
        0.0,
        make_grid(16),
    )
    .unwrap();
    let g = op.growth();
    let (m, omega) = (g.multiplier, g.omega);
    assert!(m.is_finite() && m >= 1.0, "M={m}");
    assert!(omega.is_finite(), "ω={omega}");
}

#[test]
fn cfl_violation_at_tau_one() {
    let op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(8),
        diffusion_inner(8),
        |_, _| 1.0,
        1.0,
        make_grid(8),
    )
    .unwrap();
    let f = GridFn2D::from_fn(make_grid(8), |_, _| 0.0);
    // tau=1 → 4*1*1 >= dx*dy → CflViolated
    assert!(matches!(
        op.apply_chernoff(1.0, &f),
        Err(SemiflowError::CflViolated { .. })
    ));
}

#[test]
fn apply_preserves_shape() {
    let grid = make_grid(16);
    let op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(16),
        diffusion_inner(16),
        |_, _| 0.0,
        0.0,
        grid,
    )
    .unwrap();
    let f = GridFn2D::from_fn(grid, |x, y| (-x * x - y * y).exp());
    let out = op.apply_chernoff(0.01, &f).unwrap();
    assert_eq!(out.values.len(), 16 * 16);
}

// ---------------------------------------------------------------------------
// Type alias compilation and `new` backwards compat
// ---------------------------------------------------------------------------

#[test]
fn type_alias_scalar_new_v0_7() {
    // v0.7.0 call-site: NonSeparable2DChernoff::new(x, y, c, c_norm, grid)
    let _op: NonSeparable2DChernoff<DiffusionChernoff, DiffusionChernoff> =
        NonSeparable2DChernoff::new(
            diffusion_inner(16),
            diffusion_inner(16),
            |_, _| 0.3,
            0.3,
            make_grid(16),
        )
        .unwrap();
}

#[test]
fn type_alias_aniso_new_v0_9() {
    // v0.9.0 call-site: NonSeparable2DAnisotropicChernoff::new(x, y, β, β_norm, grid)
    let _op: NonSeparable2DAnisotropicChernoff<DiffusionChernoff, DiffusionChernoff> =
        NonSeparable2DAnisotropicChernoff::new(
            diffusion_inner(16),
            diffusion_inner(16),
            |x, _| 0.3 * x,
            1.5,
            make_grid(16),
        )
        .unwrap();
}

#[test]
fn apply_with_periodic_boundary() {
    let g = Grid1D::new(-1.0, 1.0, 16)
        .unwrap()
        .with_boundary(BoundaryPolicy::Periodic);
    let grid = Grid2D::new(g, g);
    let inner = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, g);
    let op = NonSeparableMixedChernoff::with_scalar_c(inner.clone(), inner, |_, _| 0.0, 0.0, grid)
        .unwrap();
    let f = GridFn2D::from_fn(grid, |x, _| x);
    let out = op.apply_chernoff(0.01, &f).unwrap();
    assert_eq!(out.values.len(), 16 * 16);
}
