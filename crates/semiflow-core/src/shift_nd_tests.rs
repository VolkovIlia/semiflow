//! Unit tests for `AnisotropicShiftChernoffND`.
//!
//! Extracted from `shift_nd.rs` to keep that file under 500 lines.

use super::*;
use crate::grid::Grid1D;

fn make_grid_d2(n: usize) -> GridND<f64, 2> {
    let ax = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    GridND::new([ax, ax]).unwrap()
}

/// Build isotropic unit-diffusion kernel for D=2 (a = I, b = 0, c = 0).
fn unit_kernel_d2(n: usize) -> AnisotropicShiftChernoffND<f64, 2> {
    let grid = make_grid_d2(n);
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

#[test]
fn constructor_ok_d2() {
    let _ = unit_kernel_d2(8);
}

#[test]
fn constructor_err_too_few_nodes() {
    // 4^2 = 16 < 5^2 = 25 → must error
    let ax = Grid1D::new(-5.0_f64, 5.0, 4).unwrap();
    let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
    let result = AnisotropicShiftChernoffND::<f64, 2>::new(
        |_x, a| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
        },
        |_x, _b| {},
        |_x| 0.0_f64,
        grid,
    );
    assert!(result.is_err(), "4x4 grid should fail min-node check");
}

#[test]
fn constructor_err_not_spd() {
    let grid = make_grid_d2(8);
    // Singular matrix: det = 1*1 - 1*1 = 0
    let result = AnisotropicShiftChernoffND::<f64, 2>::new(
        |_x, a| {
            a.set(0, 0, 1.0);
            a.set(0, 1, 1.0);
            a.set(1, 0, 1.0);
            a.set(1, 1, 1.0);
        },
        |_x, _b| {},
        |_x| 0.0_f64,
        grid,
    );
    assert!(result.is_err(), "singular matrix should fail SPD check");
}

#[test]
fn order_is_1() {
    // ADR-0112: honest order-1 (variable-A per-step mismatch is O(τ²)).
    let k = unit_kernel_d2(8);
    assert_eq!(k.order(), 1);
}

#[test]
fn gauss_hermite_weight_sum() {
    // 1-D: sum of weights ≈ √π
    let gh = GaussHermiteTensor::<f64, 1>::new();
    let sum: f64 = (0..5).map(|i| gh.weights[i]).sum();
    let sqrt_pi = std::f64::consts::PI.sqrt();
    assert!(
        (sum - sqrt_pi).abs() < 1e-12,
        "weights sum {sum} != √π {sqrt_pi}"
    );
}

#[test]
fn gauss_hermite_weight_sum_d2() {
    // 2-D: sum of product weights ≈ π^(2/2) = π
    let gh = GaussHermiteTensor::<f64, 2>::new();
    let sum: f64 = (0..gh.n_nodes()).map(|q| gh.weight(q)).sum();
    let pi = std::f64::consts::PI;
    assert!((sum - pi).abs() < 1e-10, "2-D weight sum {sum} != π {pi}");
}

#[test]
fn apply_into_smoke_d2() {
    let k = unit_kernel_d2(8);
    let f0 = GridFnND::from_fn(k.grid.clone(), |x: &[f64; 2]| {
        (-x[0] * x[0] - x[1] * x[1]).exp()
    });
    let mut dst = f0.clone();
    let mut pool = ScratchPool::<f64>::new();
    k.apply_into(0.01, &f0, &mut dst, &mut pool).unwrap();
    // All values finite and positive (Gaussian IC → Gaussian output)
    assert!(
        dst.values.iter().all(|&v| v.is_finite() && v >= 0.0),
        "apply_into smoke: non-finite or negative output"
    );
}

#[test]
fn in_subspace_d2_ok() {
    use crate::approximation::ApproximationSubspace;
    let k = unit_kernel_d2(8);
    let f0 = GridFnND::from_fn(k.grid.clone(), |x: &[f64; 2]| {
        (-x[0] * x[0] - x[1] * x[1]).exp()
    });
    assert!(k.in_subspace(&f0), "Gaussian IC should be in_subspace");
}

#[test]
fn cholesky_identity_is_identity() {
    let mut a: SquareMatrix<f64, 2> = SquareMatrix::identity();
    let mut l: SquareMatrix<f64, 2> = SquareMatrix::zero();
    cholesky_factor(&a, &mut l).unwrap();
    // L of identity is identity
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (l.get(i, j) - expected).abs() < 1e-12,
                "L[{i},{j}]={} != {expected}",
                l.get(i, j)
            );
        }
    }
    // Suppress unused warning
    a.set(0, 0, a.get(0, 0));
}
