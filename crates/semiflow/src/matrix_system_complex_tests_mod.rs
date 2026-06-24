// Tests for `matrix_system_complex.rs` (ADR-0128, complex matrix diffusion Chernoff).
//
// Properties asserted:
//   1. new() rejects grid.n < 5.
//   2. new() succeeds for n >= 5.
//   3. MatrixGridFnComplex1D::new constructs zero-valued state.
//   4. MatrixGridFnComplex1D::from_fn populates values.
//   5. point_view / set_point round-trip.
//   6. norm_sup() returns 0 for zero state, positive for non-zero.
//   7. order() == 2.
//   8. apply_into: DomainViolation for negative tau.
//   9. apply_into: finite output for small tau.
//  10. in_subspace: true for n>=5 finite, false for NaN value.

use crate::approximation::ApproximationSubspace;
use crate::chernoff::ApplyChernoffExt;
use crate::grid::Grid1D;
use crate::state::State;
use crate::ChernoffFunction;
use num_complex::Complex;
type C64 = Complex<f64>;

fn c(re: f64, im: f64) -> C64 {
    Complex::new(re, im)
}

fn make_grid(n: usize) -> Grid1D<f64> {
    Grid1D::new(0.0_f64, 1.0, n).unwrap()
}

// Identity coefficient matrix for M=2.
fn identity_a(_x: f64, a: &mut [[C64; 2]; 2]) {
    a[0][0] = c(1.0, 0.0);
    a[0][1] = c(0.0, 0.0);
    a[1][0] = c(0.0, 0.0);
    a[1][1] = c(1.0, 0.0);
}
fn zero_b(_x: f64, b: &mut [[C64; 2]; 2]) {
    for row in b.iter_mut() {
        for v in row.iter_mut() {
            *v = c(0.0, 0.0);
        }
    }
}

// ── Constructor validation ────────────────────────────────────────────────────

#[test]
fn new_rejects_n_less_than_5() {
    let grid = make_grid(4);
    let result =
        MatrixDiffusionChernoffComplex::<C64, 2>::new(identity_a, zero_b, zero_b, grid);
    assert!(result.is_err(), "expected Err for n=4");
}

#[test]
fn new_ok_for_n_ge_5() {
    let grid = make_grid(8);
    let result =
        MatrixDiffusionChernoffComplex::<C64, 2>::new(identity_a, zero_b, zero_b, grid);
    assert!(result.is_ok(), "expected Ok for n=8");
}

// ── MatrixGridFnComplex1D ─────────────────────────────────────────────────────

#[test]
fn state_new_is_zero() {
    let grid = make_grid(8);
    let state = MatrixGridFnComplex1D::<C64, 2>::new(grid);
    assert_eq!(state.values.len(), 8 * 2, "wrong len");
    for v in &state.values {
        assert_eq!(*v, c(0.0, 0.0), "initial value non-zero");
    }
}

#[test]
fn state_from_fn_populates_values() {
    let grid = make_grid(8);
    let state =
        MatrixGridFnComplex1D::<C64, 2>::from_fn(grid, |x| [c(x, 0.0), c(0.0, x)]);
    assert_eq!(state.values.len(), 8 * 2);
    // Component 0 at point 0 should be x_at(0).
    let x0 = grid.x_at(0);
    assert!(
        (state.values[0].re - x0).abs() < 1e-14,
        "component 0 wrong: {}",
        state.values[0]
    );
}

#[test]
fn point_view_set_point_roundtrip() {
    let grid = make_grid(8);
    let mut state = MatrixGridFnComplex1D::<C64, 2>::new(grid);
    let val = [c(1.5, 2.3), c(-0.7, 0.1)];
    state.set_point(3, &val);
    let got = state.point_view(3);
    assert!((got[0] - val[0]).norm() < 1e-15, "component 0 wrong");
    assert!((got[1] - val[1]).norm() < 1e-15, "component 1 wrong");
}

// ── State trait ───────────────────────────────────────────────────────────────

#[test]
fn norm_sup_zero_state_is_zero() {
    let grid = make_grid(8);
    let state = MatrixGridFnComplex1D::<C64, 2>::new(grid);
    assert!(state.norm_sup() < 1e-15, "norm_sup of zero state non-zero");
}

#[test]
fn norm_sup_nonzero_state_is_positive() {
    let grid = make_grid(8);
    let state = MatrixGridFnComplex1D::<C64, 2>::from_fn(grid, |x| {
        [c(x, 0.0), c(0.0, 1.0)]
    });
    assert!(state.norm_sup() > 0.0, "norm_sup should be positive");
}

// ── ChernoffFunction properties ───────────────────────────────────────────────

#[test]
fn order_is_2() {
    let grid = make_grid(8);
    let kernel =
        MatrixDiffusionChernoffComplex::<C64, 2>::new(identity_a, zero_b, zero_b, grid)
            .unwrap();
    assert_eq!(kernel.order(), 2);
}

#[test]
fn apply_negative_tau_errors() {
    let grid = make_grid(8);
    let kernel =
        MatrixDiffusionChernoffComplex::<C64, 2>::new(identity_a, zero_b, zero_b, grid)
            .unwrap();
    let src = MatrixGridFnComplex1D::<C64, 2>::from_fn(grid, |x| [c(x, 0.0), c(0.0, 0.0)]);
    let mut dst = MatrixGridFnComplex1D::<C64, 2>::new(grid);
    let mut scratch = crate::scratch::ScratchPool::new();
    assert!(kernel.apply_into(-0.01, &src, &mut dst, &mut scratch).is_err());
}

#[test]
fn apply_finite_output_for_small_tau() {
    let grid = make_grid(8);
    let kernel =
        MatrixDiffusionChernoffComplex::<C64, 2>::new(identity_a, zero_b, zero_b, grid)
            .unwrap();
    let src = MatrixGridFnComplex1D::<C64, 2>::from_fn(grid, |x| {
        [c((x * core::f64::consts::PI).sin(), 0.0), c(0.0, (x * core::f64::consts::PI).cos())]
    });
    let dst = kernel.apply_chernoff(0.01, &src).unwrap();
    for v in &dst.values {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}

// ── ApproximationSubspace<2> ──────────────────────────────────────────────────

#[test]
fn in_subspace_true_for_valid_state() {
    let grid = make_grid(8);
    let kernel =
        MatrixDiffusionChernoffComplex::<C64, 2>::new(identity_a, zero_b, zero_b, grid)
            .unwrap();
    let f = MatrixGridFnComplex1D::<C64, 2>::from_fn(grid, |x| [c(x, 0.0), c(0.0, 0.0)]);
    assert!(ApproximationSubspace::<2, f64>::in_subspace(&kernel, &f));
}

#[test]
fn in_subspace_false_for_nan_value() {
    let grid = make_grid(8);
    let kernel =
        MatrixDiffusionChernoffComplex::<C64, 2>::new(identity_a, zero_b, zero_b, grid)
            .unwrap();
    let mut f = MatrixGridFnComplex1D::<C64, 2>::from_fn(grid, |x| [c(x, 0.0), c(0.0, 0.0)]);
    f.values[2] = c(f64::NAN, 0.0);
    assert!(!ApproximationSubspace::<2, f64>::in_subspace(&kernel, &f));
}
