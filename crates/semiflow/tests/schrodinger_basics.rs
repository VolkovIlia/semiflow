//! Integration tests for `SchrodingerChernoff` + `SchrodingerState` — smoke + API.
//!
//! Covers: `SchrodingerState` constructor, `State` trait, `HilbertState::dot`,
//! `SchrodingerChernoff` constructor validation, `order()`, `growth()`,
//! `v_at_node()`, zero-τ preservation, and potential-only evolution.
//!
//! See contract wave-b-advanced-semigroups.md §3 and ADR-0057.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use semiflow::{
    diffusion4::Diffusion4thChernoff,
    state::{HilbertState, State},
    ChernoffFunction, Grid1D, GridFn1D, SchrodingerChernoff, SchrodingerState, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_schr_f64(n: usize) -> SchrodingerChernoff<f64> {
    let grid = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    let kinetic = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    SchrodingerChernoff::new(kinetic, |x: f64| 0.5 * x * x).unwrap()
}

fn make_schr_f32(n: usize) -> SchrodingerChernoff<f32> {
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, n).unwrap();
    let kinetic = Diffusion4thChernoff::<f32>::new_generic(
        (|_: f32| 0.5_f32) as fn(f32) -> f32,
        (|_: f32| 0.0_f32) as fn(f32) -> f32,
        (|_: f32| 0.0_f32) as fn(f32) -> f32,
        0.5,
        grid,
    );
    SchrodingerChernoff::<f32>::new(kinetic, |x: f32| 0.5_f32 * x * x).unwrap()
}

fn make_state_f64(schr: &SchrodingerChernoff<f64>) -> SchrodingerState<f64> {
    let grid = schr.kinetic().grid;
    let psi_re = GridFn1D::from_fn(grid, |x: f64| (-(x - 1.0) * (x - 1.0) / 0.5).exp());
    let psi_im = GridFn1D::from_fn(grid, |_| 0.0_f64);
    SchrodingerState { psi_re, psi_im }
}

// ---------------------------------------------------------------------------
// SchrodingerState constructor
// ---------------------------------------------------------------------------

#[test]
fn schrodinger_state_new_ok() {
    let n = 16usize;
    let schr = make_schr_f64(n);
    let state = make_state_f64(&schr);
    // len() = 2*N (real + imaginary).
    assert_eq!(state.len(), 2 * n);
}

#[test]
fn schrodinger_state_new_rejects_mismatched_lengths() {
    let grid_a = Grid1D::new(-1.0_f64, 1.0, 8).unwrap();
    let grid_b = Grid1D::new(-1.0_f64, 1.0, 16).unwrap();
    let psi_re = GridFn1D::from_fn(grid_a, |_| 0.0_f64);
    let psi_im = GridFn1D::from_fn(grid_b, |_| 0.0_f64);
    assert!(SchrodingerState::new(psi_re, psi_im).is_err());
}

// ---------------------------------------------------------------------------
// State trait
// ---------------------------------------------------------------------------

#[test]
fn state_copy_from_and_axpy() {
    let n = 16usize;
    let schr = make_schr_f64(n);
    let mut a = make_state_f64(&schr);
    let b = make_state_f64(&schr);
    // copy_from
    let mut c = a.clone();
    c.copy_from(&b);
    // axpy_into: a += 2 * b
    a.axpy_into(2.0, &b);
    // norm_sup should be finite and positive
    assert!(a.norm_sup().is_finite() && a.norm_sup() > 0.0);
    assert!(c.norm_sup().is_finite());
}

#[test]
fn state_zero_into() {
    let n = 16usize;
    let schr = make_schr_f64(n);
    let mut state = make_state_f64(&schr);
    state.zero_into();
    assert!(
        state.norm_sup() == 0.0_f64,
        "zero_into must produce zero state"
    );
}

#[test]
fn state_norm_sup_is_amplitude() {
    // Construct state with known amplitude.
    let grid = Grid1D::new(0.0_f64, 1.0, 4).unwrap();
    let psi_re = GridFn1D::from_fn(grid, |_| 3.0_f64);
    let psi_im = GridFn1D::from_fn(grid, |_| 4.0_f64);
    let state = SchrodingerState { psi_re, psi_im };
    // |3 + 4i| = 5.
    assert!(
        (state.norm_sup() - 5.0_f64).abs() < 1e-14,
        "norm_sup should be 5.0, got {}",
        state.norm_sup()
    );
}

// ---------------------------------------------------------------------------
// HilbertState::dot
// ---------------------------------------------------------------------------

#[test]
fn hilbert_dot_self_gives_norm_sq() {
    let n = 16usize;
    let schr = make_schr_f64(n);
    let state = make_state_f64(&schr);
    let dot = state.dot(&state);
    let norm_sq = state.norm_l2_sq();
    assert!(
        (dot - norm_sq).abs() < 1e-12,
        "dot(psi,psi) should equal norm_l2_sq; diff = {:.4e}",
        (dot - norm_sq).abs()
    );
}

// ---------------------------------------------------------------------------
// SchrodingerChernoff constructor
// ---------------------------------------------------------------------------

#[test]
fn constructor_rejects_nonfinite_potential() {
    let grid = Grid1D::new(-1.0_f64, 1.0, 4).unwrap();
    let kinetic = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    // V(0) = 1/x → infinity at x=0 is not tested here because we don't have
    // a guaranteed 0-node. Instead use NaN-returning potential.
    let result = SchrodingerChernoff::new(kinetic, |_: f64| f64::NAN);
    assert!(result.is_err(), "NaN potential must be rejected");
}

// ---------------------------------------------------------------------------
// order() and growth()
// ---------------------------------------------------------------------------

#[test]
fn order_is_two_f64() {
    let schr = make_schr_f64(16);
    assert_eq!(schr.order(), 2, "SchrodingerChernoff must report order 2");
}

#[test]
fn order_is_two_f32() {
    let schr = make_schr_f32(16);
    assert_eq!(
        schr.order(),
        2,
        "SchrodingerChernoff<f32> must report order 2"
    );
}

#[test]
fn growth_is_unitary_f64() {
    let g = make_schr_f64(16).growth();
    assert!(
        (g.multiplier - 1.0).abs() < 1e-14,
        "growth M must be 1.0 (unitary), got {}",
        g.multiplier
    );
    assert!(
        (g.omega - 0.0).abs() < 1e-14,
        "growth ω must be 0.0 (no shift), got {}",
        g.omega
    );
}

// ---------------------------------------------------------------------------
// v_at_node()
// ---------------------------------------------------------------------------

#[test]
fn v_at_node_length_matches_grid() {
    let n = 32usize;
    let schr = make_schr_f64(n);
    assert_eq!(
        schr.v_at_node().len(),
        n,
        "v_at_node must contain exactly N = {n} entries"
    );
}

#[test]
fn v_at_node_matches_potential_at_grid() {
    let n = 8usize;
    let grid = Grid1D::new(-2.0_f64, 2.0, n).unwrap();
    let kinetic = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let schr = SchrodingerChernoff::new(kinetic, |x: f64| 0.5 * x * x).unwrap();
    for i in 0..n {
        let xi = grid.x_at(i);
        let expected = 0.5 * xi * xi;
        let got = schr.v_at_node()[i];
        assert!(
            (got - expected).abs() < 1e-14,
            "v_at_node[{i}] = {got:.6e}, expected {expected:.6e}"
        );
    }
}

// ---------------------------------------------------------------------------
// zero-τ preservation
// ---------------------------------------------------------------------------

#[test]
fn zero_tau_preserves_state_f64() {
    let n = 16usize;
    let schr = make_schr_f64(n);
    let src = make_state_f64(&schr);
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    schr.apply_into(0.0, &src, &mut dst, &mut pool).unwrap();
    let mut diff = src.clone();
    diff.axpy_into(-1.0, &dst);
    assert!(
        diff.norm_sup() < 1e-14,
        "zero-τ must preserve state; sup_diff = {}",
        diff.norm_sup()
    );
}

#[test]
fn zero_tau_preserves_state_f32() {
    let n = 16usize;
    let schr = make_schr_f32(n);
    let grid = schr.kinetic().grid;
    let psi_re =
        GridFn1D::<f32>::from_fn_generic(grid, |x: f32| (-(x - 1.0_f32) * (x - 1.0_f32)).exp());
    let psi_im = GridFn1D::<f32>::from_fn_generic(grid, |_: f32| 0.0_f32);
    let src = SchrodingerState::<f32> { psi_re, psi_im };
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f32>::new();
    schr.apply_into(0.0_f32, &src, &mut dst, &mut pool).unwrap();
    let mut diff = src.clone();
    diff.axpy_into(-1.0_f32, &dst);
    assert!(
        (diff.norm_sup() as f64) < 1e-6,
        "zero-τ (f32) must preserve state; sup_diff = {}",
        diff.norm_sup()
    );
}
