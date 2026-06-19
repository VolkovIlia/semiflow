// Tests for point_eval.rs — moved from point_eval.rs (batch H6).
use super::*;
use crate::{
    diffusion::DiffusionChernoff, grid::Grid1D, grid_fn::GridFn1D, shift1d::ShiftChernoff1D,
};

fn make_1d_grid() -> Grid1D<f64> {
    Grid1D::new(-5.0_f64, 5.0, 32).unwrap()
}

fn make_gaussian_1d(grid: Grid1D<f64>) -> GridFn1D<f64> {
    let values: alloc::vec::Vec<f64> = (0..grid.n)
        .map(|i| {
            let x = grid.x_at(i);
            (-x * x).exp()
        })
        .collect();
    GridFn1D { values, grid }
}

// Backend A: eval_at returns without error for n=1.
#[test]
fn diffusion_eval_at_n1_ok() {
    let grid = make_1d_grid();
    let kernel = DiffusionChernoff::new(
        |_: f64| 1.0_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        1.0_f64,
        grid,
    );
    let src = make_gaussian_1d(grid);
    let val = kernel.eval_at(0.01_f64, &src, &[0.0_f64], 1);
    assert!(val.is_ok(), "Backend A: eval_at n=1 must succeed");
}

// Backend B: eval_at returns without error for n=1.
#[test]
fn shift1d_eval_at_n1_ok() {
    let grid = make_1d_grid();
    // ShiftChernoff1D::new(a, b, c, c_norm_bound, grid) — 5 args.
    let kernel = ShiftChernoff1D::new(
        |_: f64| 1.0_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        0.0_f64,
        grid,
    );
    let src = make_gaussian_1d(grid);
    let val = kernel.eval_at(0.01_f64, &src, &[0.0_f64], 1);
    assert!(val.is_ok(), "Backend B: eval_at n=1 must succeed");
}

// Guard: n_steps=0 returns DomainViolation.
#[test]
fn eval_at_zero_steps_errors() {
    let grid = make_1d_grid();
    let kernel = DiffusionChernoff::new(
        |_: f64| 1.0_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        1.0_f64,
        grid,
    );
    let src = make_gaussian_1d(grid);
    let val = kernel.eval_at(0.01_f64, &src, &[0.0_f64], 0);
    assert!(
        matches!(val, Err(SemiflowError::DomainViolation { .. })),
        "n_steps=0 must return DomainViolation"
    );
}

// Guard: empty x slice returns DomainViolation for 1-D backend.
#[test]
fn eval_at_empty_x_errors() {
    let grid = make_1d_grid();
    let kernel = DiffusionChernoff::new(
        |_: f64| 1.0_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        1.0_f64,
        grid,
    );
    let src = make_gaussian_1d(grid);
    let val = kernel.eval_at(0.01_f64, &src, &[], 1);
    assert!(
        matches!(val, Err(SemiflowError::DomainViolation { .. })),
        "empty x must return DomainViolation"
    );
}

// G_BATCH_POINTEVAL: eval_at_batch is 0-ULP identical to mapping eval_at.
#[test]
fn g_batch_pointeval_byte_identical_to_scalar() {
    let grid = make_1d_grid();
    let kernel = DiffusionChernoff::new(
        |_: f64| 1.0_f64,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        1.0_f64,
        grid,
    );
    let src = make_gaussian_1d(grid);
    let tau = 0.01_f64;
    let n_steps = 4_u32;
    let xs: &[&[f64]] = &[&[-1.0], &[0.0], &[1.0]];
    let batch = kernel.eval_at_batch(tau, &src, xs, n_steps).unwrap();
    for (i, x) in xs.iter().enumerate() {
        let scalar = kernel.eval_at(tau, &src, x, n_steps).unwrap();
        assert_eq!(
            batch[i].to_bits(),
            scalar.to_bits(),
            "G_BATCH_POINTEVAL: index {i} batch {:.16e} != scalar {:.16e}",
            batch[i],
            scalar,
        );
    }
}

// sample_gridfn2d: centre node returns values[centre_idx] exactly.
#[test]
fn sample_gridfn2d_centre_node() {
    use crate::{grid2d::Grid2D, grid_fn2d::GridFn2D};
    let gx = Grid1D::new(-1.0_f64, 1.0, 4).unwrap();
    let gy = Grid1D::new(-1.0_f64, 1.0, 4).unwrap();
    let grid = Grid2D::new(gx, gy);
    let values: alloc::vec::Vec<f64> = (0..16).map(f64::from).collect();
    let state = GridFn2D { values, grid };
    // Centre of grid: x_at(1), y_at(1) → index 1*4+1 = 5 (value 5.0)
    let cx = gx.x_at(1);
    let cy = gy.x_at(1);
    let sampled = sample_gridfn2d(&state, cx, cy);
    assert_eq!(
        sampled.to_bits(),
        5.0_f64.to_bits(),
        "sample_gridfn2d at node centre must return exact node value"
    );
}
