//! Unit tests for `NonSeparableMixedChernoff`.
//!
//! Extracted from `nonseparable_mixed.rs` (mod.rs) to keep that file under 500 lines.

use super::*;
use crate::{diffusion::DiffusionChernoff, grid::Grid1D};

fn make_grid(n: usize) -> Grid2D<f64> {
    let g1 = Grid1D::new(-1.0, 1.0, n).unwrap();
    Grid2D::new(g1, g1)
}

fn diffusion_inner(n: usize) -> DiffusionChernoff {
    let gx = Grid1D::new(-1.0, 1.0, n).unwrap();
    DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx)
}

#[test]
fn scalar_zero_coupling_ok() {
    let op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(16),
        diffusion_inner(16),
        |_, _| 0.0,
        0.0,
        make_grid(16),
    );
    assert!(op.is_ok());
    assert!(op.unwrap().coupling.is_zero());
}

#[test]
fn beta_zero_coupling_ok() {
    let op = NonSeparableMixedChernoff::with_beta(
        diffusion_inner(16),
        diffusion_inner(16),
        |_, _| 0.0,
        0.0,
        make_grid(16),
    );
    assert!(op.is_ok());
    assert!(op.unwrap().coupling.is_zero());
}

#[test]
fn bad_norm_rejected() {
    let g = make_grid(8);
    assert!(NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(8),
        diffusion_inner(8),
        |_, _| 0.0_f64,
        -1.0,
        g,
    )
    .is_err());
}

#[test]
fn cfl_violation_returned() {
    use crate::chernoff::ApplyChernoffExt;
    let op = NonSeparableMixedChernoff::with_scalar_c(
        diffusion_inner(8),
        diffusion_inner(8),
        |_, _| 1.0,
        1.0,
        make_grid(8),
    )
    .unwrap();
    let grid = make_grid(8);
    let f = GridFn2D::from_fn(grid, |_, _| 0.0);
    assert!(matches!(
        op.apply_chernoff(1.0, &f),
        Err(SemiflowError::CflViolated { .. })
    ));
}

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
fn alias_nonsep2d_new_compiles() {
    // v0.7.0 call-site: NonSeparable2DChernoff::new(x, y, c, c_norm, grid)
    let _op: NonSeparable2DChernoff<DiffusionChernoff, DiffusionChernoff> =
        NonSeparable2DChernoff::with_scalar_c(
            diffusion_inner(16),
            diffusion_inner(16),
            |_, _| 0.3,
            0.3,
            make_grid(16),
        )
        .unwrap();
}

#[test]
fn alias_nonsep2d_aniso_new_compiles() {
    // v0.9.0 call-site: NonSeparable2DAnisotropicChernoff::new(x, y, β, β_norm, grid)
    let _op: NonSeparable2DAnisotropicChernoff<DiffusionChernoff, DiffusionChernoff> =
        NonSeparable2DAnisotropicChernoff::with_beta(
            diffusion_inner(16),
            diffusion_inner(16),
            |x, _| 0.3 * x,
            1.5,
            make_grid(16),
        )
        .unwrap();
}
