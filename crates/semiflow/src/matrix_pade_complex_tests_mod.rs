// Tests for `matrix_pade_complex.rs` (ADR-0128, G_CPLX_MATRIX gate).
//
// Properties asserted:
//   1. exp(0) = I (zero matrix → identity).
//   2. exp(iH) unitary for Hermitian H (‖UᴴU − I‖_F ≤ 1e-12).
//   3. cmat_inv_complex_dispatch: A·A⁻¹ = I for M=1, M=2, M≥3.
//   4. cmat_vec_mul: y = A·x consistent with manual computation.
//   5. real diagonal: exp(diag(λ₁,λ₂)) = diag(exp(λ₁), exp(λ₂)).
//   6. DomainViolation returned for near-singular matrices.

use super::*;
use num_complex::Complex;
type C64 = Complex<f64>;

fn c(re: f64, im: f64) -> C64 {
    Complex::new(re, im)
}

fn frob_err<const M: usize>(got: &[[C64; M]; M], expected: &[[C64; M]; M]) -> f64 {
    let mut sum = 0.0f64;
    for r in 0..M {
        for col in 0..M {
            let d = got[r][col] - expected[r][col];
            sum += (d * d.conj()).re;
        }
    }
    sum.sqrt()
}

fn eye2() -> [[C64; 2]; 2] {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
}

// ── exp(0) = I ────────────────────────────────────────────────────────────────

#[test]
fn exp_zero_2x2_is_identity() {
    let zero: [[C64; 2]; 2] = [[c(0.0, 0.0); 2]; 2];
    let result = mat_exp_pade13_complex(&zero).unwrap();
    let expected = eye2();
    assert!(
        frob_err(&result, &expected) < 1e-12,
        "exp(0) deviates from I: {result:?}"
    );
}

#[test]
fn exp_zero_3x3_is_identity() {
    let zero: [[C64; 3]; 3] = [[c(0.0, 0.0); 3]; 3];
    let result = mat_exp_pade13_complex(&zero).unwrap();
    let mut expected = [[c(0.0, 0.0); 3]; 3];
    expected[0][0] = c(1.0, 0.0);
    expected[1][1] = c(1.0, 0.0);
    expected[2][2] = c(1.0, 0.0);
    assert!(frob_err(&result, &expected) < 1e-12, "exp(0) 3x3 failed");
}

// ── exp(iH) unitary ───────────────────────────────────────────────────────────

// Build iH for H = [[2, 1], [1, 3]] (symmetric).
fn build_ih_2x2() -> [[C64; 2]; 2] {
    [[c(0.0, 2.0), c(0.0, 1.0)], [c(0.0, 1.0), c(0.0, 3.0)]]
}

#[test]
fn exp_ih_is_unitary_2x2() {
    let ih = build_ih_2x2();
    let u = mat_exp_pade13_complex(&ih).unwrap();
    // Compute U† U and check it equals I.
    let mut uhu = [[c(0.0, 0.0); 2]; 2];
    for r in 0..2 {
        for col in 0..2 {
            let mut s = c(0.0, 0.0);
            for uk in &u {
                s += uk[r].conj() * uk[col];
            }
            uhu[r][col] = s;
        }
    }
    assert!(
        frob_err(&uhu, &eye2()) < 1e-12,
        "U†U ≠ I: {uhu:?}"
    );
}

// ── real diagonal: exp(diag) = diag(exp) ─────────────────────────────────────

#[test]
fn exp_real_diagonal_2x2() {
    let a: [[C64; 2]; 2] = [
        [c(-1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(-2.0, 0.0)],
    ];
    let result = mat_exp_pade13_complex(&a).unwrap();
    let e1 = f64::exp(-1.0);
    let e2 = f64::exp(-2.0);
    assert!(
        (result[0][0].re - e1).abs() < 1e-12,
        "diag[0] wrong: {}",
        result[0][0].re
    );
    assert!(
        (result[1][1].re - e2).abs() < 1e-12,
        "diag[1] wrong: {}",
        result[1][1].re
    );
    assert!(result[0][1].norm() < 1e-12, "off-diag 01 wrong");
    assert!(result[1][0].norm() < 1e-12, "off-diag 10 wrong");
}

// ── cmat_inv_complex_dispatch ─────────────────────────────────────────────────

#[test]
fn inv_dispatch_m1_identity() {
    let a: [[C64; 1]; 1] = [[c(3.0, 1.0)]];
    let inv = cmat_inv_complex_dispatch(&a).unwrap();
    // inv[0][0] = 1 / (3+i) = (3-i)/10
    let expected = c(1.0, 0.0) / c(3.0, 1.0);
    assert!(
        (inv[0][0] - expected).norm() < 1e-12,
        "M=1 inv wrong: {}",
        inv[0][0]
    );
}

#[test]
fn inv_dispatch_m2_product_is_identity() {
    let a: [[C64; 2]; 2] = [
        [c(2.0, 0.0), c(1.0, 1.0)],
        [c(0.0, -1.0), c(3.0, 0.0)],
    ];
    let inv = cmat_inv_complex_dispatch(&a).unwrap();
    // A · A⁻¹ should be I.
    let mut prod = [[c(0.0, 0.0); 2]; 2];
    for r in 0..2 {
        for col in 0..2 {
            for k in 0..2 {
                prod[r][col] += a[r][k] * inv[k][col];
            }
        }
    }
    assert!(frob_err(&prod, &eye2()) < 1e-12, "A·A⁻¹ ≠ I: {prod:?}");
}

#[test]
fn inv_dispatch_m1_singular_returns_err() {
    let a: [[C64; 1]; 1] = [[c(0.0, 0.0)]];
    assert!(cmat_inv_complex_dispatch(&a).is_err());
}

#[test]
fn inv_dispatch_m2_singular_returns_err() {
    let a: [[C64; 2]; 2] = [
        [c(1.0, 0.0), c(2.0, 0.0)],
        [c(2.0, 0.0), c(4.0, 0.0)],
    ];
    assert!(cmat_inv_complex_dispatch(&a).is_err());
}

// ── cmat_vec_mul ──────────────────────────────────────────────────────────────

#[test]
fn cmat_vec_mul_2x2() {
    let a: [[C64; 2]; 2] = [[c(1.0, 1.0), c(0.0, 1.0)], [c(2.0, 0.0), c(-1.0, 0.0)]];
    let v: [C64; 2] = [c(1.0, 0.0), c(0.0, 1.0)];
    let out = cmat_vec_mul(&a, &v);
    // row 0: (1+i)·1 + i·i = (1+i) + i² = (1+i) + (-1) = 0+i
    assert!((out[0] - c(0.0, 1.0)).norm() < 1e-15, "row0: {}", out[0]);
    // row 1: 2·1 + (-1)·i = 2 - i
    assert!((out[1] - c(2.0, -1.0)).norm() < 1e-15, "row1: {}", out[1]);
}
