// Tests for SchrodingerChernoff — moved from schrodinger.rs (batch H5).
use super::*;
use crate::{
    chernoff::ChernoffFunction, diffusion4::Diffusion4thChernoff, grid::Grid1D,
    grid_fn::GridFn1D, state::State,
};

fn make_schr_f64(n: usize) -> SchrodingerChernoff<f64> {
    let grid = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    // a = 0.5 (Laplacian), a' = 0, a'' = 0, norm_bound = 0.5.
    let kinetic = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    SchrodingerChernoff::new(kinetic, |x: f64| 0.5 * x * x).unwrap()
}

fn make_state_f64(_n: usize, schr: &SchrodingerChernoff<f64>) -> SchrodingerState<f64> {
    let grid = schr.kinetic.grid;
    let psi_re = GridFn1D::from_fn(grid, |x| (-(x - 1.0) * (x - 1.0) / 0.5).exp());
    let psi_im = GridFn1D::from_fn(grid, |_| 0.0);
    SchrodingerState { psi_re, psi_im }
}

#[test]
fn order_is_two() {
    let schr = make_schr_f64(32);
    assert_eq!(schr.order(), 2);
}

#[test]
fn growth_is_unitary() {
    let schr = make_schr_f64(32);
    let g = schr.growth();
    assert!((g.multiplier - 1.0).abs() < 1e-14);
    assert!((g.omega - 0.0).abs() < 1e-14);
}

#[test]
fn v_rotation_unit_angle_is_unitary() {
    // Check that iterated Chernoff steps approximately preserve ‖ψ‖².
    // At small τ the Chernoff approximation is near-unitary; tolerance is loose
    // for a smoke test (tight gate is in tests/g18_unitarity.rs).
    let n = 32;
    let schr = make_schr_f64(n);
    let state = make_state_f64(n, &schr);
    let norm_sq_before = state.norm_l2_sq();

    // Use many small steps (tau = 0.001) to stay in the regime where the
    // Chernoff approximation well approximates the unitary propagator.
    let tau = 0.001_f64;
    let n_steps = 10_usize;
    let mut current = state.clone();
    let mut next = state.clone();
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        schr.apply_into(tau, &current, &mut next, &mut pool)
            .unwrap();
        core::mem::swap(&mut current, &mut next);
    }
    let norm_sq_after = current.norm_l2_sq();

    // Tight unitarity is verified in tests/g18_unitarity.rs. This smoke test
    // only checks that the propagator runs without error and doesn't explode.
    assert!(
        norm_sq_after.is_finite(),
        "norm must remain finite after {n_steps} steps, got {norm_sq_after:.3e}"
    );
    let rel_change = (norm_sq_after - norm_sq_before).abs() / norm_sq_before.max(1e-30);
    assert!(
        rel_change < 0.5,
        "norm changed dramatically: rel_change = {rel_change:.3e}"
    );
}

#[test]
fn zero_tau_preserves_state() {
    let n = 16;
    let schr = make_schr_f64(n);
    let state = make_state_f64(n, &schr);
    let mut dst = state.clone();
    let mut pool = ScratchPool::<f64>::new();
    schr.apply_into(0.0, &state, &mut dst, &mut pool).unwrap();

    let mut diff = state.clone();
    diff.axpy_into(-1.0, &dst);
    assert!(
        diff.norm_sup() < 1e-14,
        "zero-tau should preserve state, sup_diff = {}",
        diff.norm_sup()
    );
}

#[test]
fn v_at_node_length_matches_grid() {
    let n = 32;
    let schr = make_schr_f64(n);
    assert_eq!(schr.v_at_node().len(), n);
}
