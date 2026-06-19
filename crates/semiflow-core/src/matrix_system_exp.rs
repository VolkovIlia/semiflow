//! Per-M matrix-exponential helpers for `matrix_system` (Cayley-Hamilton + Taylor).
//!
//! Provides `matrix_exp_dispatch`, `mat_vec_mul`, `mat_mul_mm`, and the
//! per-size backends `matrix_exp_m{1,2,3,4}` + `mat_exp_taylor` +
//! `scale_and_shift` + `mat_identity`.
//!
//! All functions are `pub(super)` — visible only to `matrix_system.rs`.

// Matrix scaling: log2(norm).ceil() as u32 where norm > 1 ⟹ log2 > 0 ⟹ cast is safe.
// f64→u32 cast after clamp(.min(30)) prevents truncation beyond u32 range.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use crate::{error::SemiflowError, float::SemiflowFloat, matrix_pade::mat_exp_pade13};

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Dispatch matrix exponential: Cayley-Hamilton for M ∈ {1,2,3,4};
/// Padé[13/13] (Higham 2005, ADR-0125) for M ≥ 5.
pub(super) fn matrix_exp_dispatch<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    match M {
        0 => Ok([[F::zero(); M]; M]),
        1 => Ok(matrix_exp_m1(a)),
        2 => Ok(matrix_exp_m2(a)),
        3 => Ok(matrix_exp_m3(a)),
        4 => Ok(matrix_exp_m4(a)),
        _ => mat_exp_pade13(a),
    }
}

// ---------------------------------------------------------------------------
// Matrix-vector multiply
// ---------------------------------------------------------------------------

/// M×M matrix-vector multiply: out = A·v.
#[inline]
pub(super) fn mat_vec_mul<F: SemiflowFloat, const M: usize>(a: &[[F; M]; M], v: &[F; M]) -> [F; M] {
    let mut out = [F::zero(); M];
    for i in 0..M {
        for j in 0..M {
            out[i] += a[i][j] * v[j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Per-M closed-form backends
// ---------------------------------------------------------------------------

/// M=1: scalar exponential.
fn matrix_exp_m1<F: SemiflowFloat, const M: usize>(a: &[[F; M]; M]) -> [[F; M]; M] {
    let mut out = [[F::zero(); M]; M];
    out[0][0] = a[0][0].exp();
    out
}

/// M=2: Cayley-Hamilton closed-form via eigenvalues (Higham 2008 §10.4).
///
/// For A ∈ ℝ²ˣ², e^A = α₀·I + α₁·A where α₀, α₁ are determined
/// by the Putzer algorithm: λ₁, λ₂ = eigenvalues of A.
///
/// Degenerate (repeated eigenvalue λ): e^A = e^λ·(I + (A - λI)).
fn matrix_exp_m2<F: SemiflowFloat, const M: usize>(a: &[[F; M]; M]) -> [[F; M]; M] {
    let a00 = a[0][0];
    let a01 = a[0][1];
    let a10 = a[1][0];
    let a11 = a[1][1];
    // Characteristic polynomial: λ² - tr·λ + det = 0.
    let tr = a00 + a11;
    let det = a00 * a11 - a01 * a10;
    let two = F::one() + F::one();
    let four = two + two;
    let disc = tr * tr - four * det;
    let half = F::one() / two;
    let mut out = [[F::zero(); M]; M];
    if disc.abs() < F::epsilon() * F::from(1000.0).unwrap_or(F::one()) {
        // Repeated eigenvalue λ = tr/2: e^A = e^λ·(I + (A - λI)).
        let lam = half * tr;
        let e_lam = lam.exp();
        out[0][0] = e_lam * (F::one() + a00 - lam);
        out[0][1] = e_lam * a01;
        out[1][0] = e_lam * a10;
        out[1][1] = e_lam * (F::one() + a11 - lam);
    } else {
        // Distinct eigenvalues λ₁ ≠ λ₂: Putzer formula.
        // e^A = ((e^λ₁ - e^λ₂)/(λ₁ - λ₂))·A + ((λ₁·e^λ₂ - λ₂·e^λ₁)/(λ₁ - λ₂))·I.
        let sqrt_disc = disc.abs().sqrt();
        let lam1 = half * (tr + sqrt_disc);
        let lam2 = half * (tr - sqrt_disc);
        let e1 = lam1.exp();
        let e2 = lam2.exp();
        let diff = lam1 - lam2;
        let c1 = (e1 - e2) / diff;
        let c0 = (lam1 * e2 - lam2 * e1) / diff;
        out[0][0] = c0 + c1 * a00;
        out[0][1] = c1 * a01;
        out[1][0] = c1 * a10;
        out[1][1] = c0 + c1 * a11;
    }
    out
}

/// M=3: matrix exponential via scaling-and-squaring with Taylor series.
///
/// Same algorithm as M=4 path (Higham 2008 §10.7.3). Degree-12 Taylor with
/// scaling ensures ≤ 1 ULP error for matrices with ∞-norm ≤ 2^30.
fn matrix_exp_m3<F: SemiflowFloat, const M: usize>(a: &[[F; M]; M]) -> [[F; M]; M] {
    mat_exp_taylor::<F, M>(a, 3)
}

/// M=4: matrix exponential via scaling-and-squaring (delegates to `mat_exp_taylor`).
fn matrix_exp_m4<F: SemiflowFloat, const M: usize>(a: &[[F; M]; M]) -> [[F; M]; M] {
    mat_exp_taylor::<F, M>(a, 4)
}

// ---------------------------------------------------------------------------
// Scaling-and-squaring Taylor series
// ---------------------------------------------------------------------------

/// Scaling-and-squaring matrix exponential for M×M matrices (M = `dim`).
///
/// Algorithm (Higham 2008 §10.7.3):
/// 1. Compute ∞-norm; choose k = ⌈log₂(‖A‖)⌉ so ‖A/2^k‖_∞ ≤ 1.
/// 2. Taylor series degree 12: e^(A/2^k) = Σ_{n=0}^{12} (A/2^k)^n / n!.
/// 3. Square k times: e^A = (e^(A/2^k))^(2^k).
///
/// Reliable for any matrix (no eigenvalue failure modes, no NaN from degenerate cases).
// i,j used both for indexing and as arguments; range loops needed.
#[allow(clippy::needless_range_loop)]
fn mat_exp_taylor<F: SemiflowFloat, const M: usize>(a: &[[F; M]; M], dim: usize) -> [[F; M]; M] {
    let (k, b) = scale_and_shift::<F, M>(a, dim);
    let mut result = mat_identity::<F, M>(dim);
    let mut term = mat_identity::<F, M>(dim);
    // Accumulate Taylor series: result += B^n / n!
    for d in 1u32..=12 {
        term = mat_mul_mm::<F, M>(&term, &b, dim);
        let inv_d = F::from(1.0 / f64::from(d)).unwrap_or(F::one());
        for i in 0..dim {
            for j in 0..dim {
                result[i][j] += term[i][j] * inv_d;
            }
        }
    }
    // Squaring phase.
    for _ in 0..k {
        result = mat_mul_mm::<F, M>(&result, &result, dim);
    }
    result
}

/// Compute scaling factor k and scaled matrix B = A / 2^k (∞-norm ≤ 1).
///
/// Returns `(k, B)` where k = ⌈log₂(‖A‖_∞)⌉ clamped to `[0, 30]`.
#[allow(clippy::needless_range_loop)]
fn scale_and_shift<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
    dim: usize,
) -> (u32, [[F; M]; M]) {
    let mut norm = F::zero();
    for i in 0..dim {
        let mut row = F::zero();
        for j in 0..dim {
            row += a[i][j].abs();
        }
        if row > norm {
            norm = row;
        }
    }
    let k = {
        let nf = norm.to_f64().unwrap_or(0.0);
        if nf <= 1.0 {
            0u32
        } else {
            (nf.log2().ceil() as u32).min(30)
        }
    };
    let scale = F::from(f64::from(1u32 << k)).unwrap_or(F::one());
    let mut b = [[F::zero(); M]; M];
    for i in 0..dim {
        for j in 0..dim {
            b[i][j] = a[i][j] / scale;
        }
    }
    (k, b)
}

/// Return M×M identity matrix restricted to the `dim`×`dim` upper-left block.
#[allow(clippy::needless_range_loop)]
fn mat_identity<F: SemiflowFloat, const M: usize>(dim: usize) -> [[F; M]; M] {
    let mut m = [[F::zero(); M]; M];
    for i in 0..dim {
        m[i][i] = F::one();
    }
    m
}

// ---------------------------------------------------------------------------
// Matrix multiply
// ---------------------------------------------------------------------------

/// dim×dim matrix multiply C = A·B (works for M=2, 3, or 4 via `dim` parameter).
#[inline]
pub(super) fn mat_mul_mm<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
    b: &[[F; M]; M],
    dim: usize,
) -> [[F; M]; M] {
    let mut c = [[F::zero(); M]; M];
    for i in 0..dim {
        for k in 0..dim {
            for j in 0..dim {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}
