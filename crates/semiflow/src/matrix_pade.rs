//! Padé[13/13] matrix exponential for M×M matrices (M ≥ 5).
//!
//! Algorithm: Higham 2005, *The Scaling and Squaring Method for the Matrix
//! Exponential Revisited*, SIAM J. Matrix Anal. Appl. 26:4 pp. 1179–1193.
//!
//! - θ₁₃ = 5.3719: backward-error radius (Higham 2005 Table 2.3).
//! - `s = ⌈log₂(‖A‖_∞ / θ₁₃)⌉` squarings; A scaled to `‖A/2ˢ‖ ≤ θ₁₃`.
//! - U/V even/odd Padé split (Table 1); R = (V−U)⁻¹(V+U) via in-tree LU.
//! - Square R s times.
//!
//! Gate: `G_MATRIX_PADE_M5` — relative Frobenius error ≤ 1e-12 for symmetric
//! reaction matrices with `‖τC/2‖_∞ ≤ 10` (the physical half-step regime per
//! §33.8 Para 2, contracts/semiflow-core.math.md).

use crate::{error::SemiflowError, float::SemiflowFloat, matrix_inv::mat_inv_lu};

// ---------------------------------------------------------------------------
// Padé[13/13] numerator/denominator coefficients (Higham 2005 Table 2.3).
// pade_b[0]..pade_b[13]: shared even/odd coefficients.
// ---------------------------------------------------------------------------

/// Higham 2005 Table 2.3 backward-error radius for Padé[13/13].
const THETA13: f64 = 5.371_920_351_148_152;

/// Padé[13/13] coefficients `pade_b[0]..pade_b[13]` (Higham 2005 Table 2.3).
const PADE_B: [f64; 14] = [
    64_764_752_532_480_000.0,
    32_382_376_266_240_000.0,
    7_771_770_303_897_600.0,
    1_187_353_796_428_800.0,
    129_060_195_264_000.0,
    10_559_470_521_600.0,
    670_442_572_800.0,
    33_522_128_640.0,
    1_323_241_920.0,
    40_840_800.0,
    960_960.0,
    16_380.0,
    182.0,
    1.0,
];

// ---------------------------------------------------------------------------
// Internal matrix helpers (stack-allocated, no_std + alloc)
// ---------------------------------------------------------------------------

/// M×M matrix multiply: `mat_c = mat_a · mat_b`.
#[inline]
fn mm_mul<F: SemiflowFloat, const M: usize>(
    mat_a: &[[F; M]; M],
    mat_b: &[[F; M]; M],
) -> [[F; M]; M] {
    let mut mat_c = [[F::zero(); M]; M];
    for (ridx, row_c) in mat_c.iter_mut().enumerate() {
        for mid in 0..M {
            let a_val = mat_a[ridx][mid];
            for (cidx, c_elem) in row_c.iter_mut().enumerate() {
                *c_elem += a_val * mat_b[mid][cidx];
            }
        }
    }
    mat_c
}

/// `mat_c = alpha·mat_a + beta·mat_b` (element-wise).
#[inline]
fn mm_axpby<F: SemiflowFloat, const M: usize>(
    alpha: F,
    mat_a: &[[F; M]; M],
    beta: F,
    mat_b: &[[F; M]; M],
) -> [[F; M]; M] {
    let mut mat_c = [[F::zero(); M]; M];
    for (row_c, (row_a, row_b)) in mat_c.iter_mut().zip(mat_a.iter().zip(mat_b.iter())) {
        for (c_elem, (&a_elem, &b_elem)) in row_c.iter_mut().zip(row_a.iter().zip(row_b.iter())) {
            *c_elem = alpha * a_elem + beta * b_elem;
        }
    }
    mat_c
}

/// Identity matrix.
#[inline]
fn mm_eye<F: SemiflowFloat, const M: usize>() -> [[F; M]; M] {
    let mut mat = [[F::zero(); M]; M];
    for (d, row) in mat.iter_mut().enumerate() {
        row[d] = F::one();
    }
    mat
}

/// Solve `mat_a·X = mat_b` via `mat_a⁻¹·mat_b` using the in-tree LU inverse.
fn mm_solve<F: SemiflowFloat, const M: usize>(
    mat_a: &[[F; M]; M],
    mat_b: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    let a_inv = mat_inv_lu::<F, M>(mat_a)?;
    Ok(mm_mul::<F, M>(&a_inv, mat_b))
}

// ---------------------------------------------------------------------------
// Padé[13/13] U polynomial (odd part): U = A·(A⁶·W₁ + W₂)
// ---------------------------------------------------------------------------

fn pade13_u<F: SemiflowFloat, const M: usize>(
    mat_scaled: &[[F; M]; M],
    sq2: &[[F; M]; M], // A²
    sq4: &[[F; M]; M], // A⁴
    sq6: &[[F; M]; M], // A⁶
    identity: &[[F; M]; M],
    coeff: impl Fn(usize) -> F,
) -> [[F; M]; M] {
    // W₁ = b₁₃·A⁶ + b₁₁·A⁴ + b₉·A²
    let tmp1 = mm_axpby(coeff(13), sq6, coeff(11), sq4);
    let w1 = mm_axpby(F::one(), &tmp1, coeff(9), sq2);
    // W₂ = b₇·A⁶ + b₅·A⁴ + b₃·A² + b₁·I
    let tmp2 = mm_axpby(coeff(7), sq6, coeff(5), sq4);
    let tmp3 = mm_axpby(coeff(3), sq2, coeff(1), identity);
    let w2 = mm_axpby(F::one(), &tmp2, F::one(), &tmp3);
    // U = A_scaled · (A⁶·W₁ + W₂)
    let aw1 = mm_mul::<F, M>(sq6, &w1);
    let inner = mm_axpby(F::one(), &aw1, F::one(), &w2);
    mm_mul::<F, M>(mat_scaled, &inner)
}

// ---------------------------------------------------------------------------
// Padé[13/13] V polynomial (even part): V = A⁶·V₁ + V₂
// ---------------------------------------------------------------------------

fn pade13_v<F: SemiflowFloat, const M: usize>(
    sq2: &[[F; M]; M], // A²
    sq4: &[[F; M]; M], // A⁴
    sq6: &[[F; M]; M], // A⁶
    identity: &[[F; M]; M],
    coeff: impl Fn(usize) -> F,
) -> [[F; M]; M] {
    // V₁ = b₁₂·A⁶ + b₁₀·A⁴ + b₈·A²
    let tmp1 = mm_axpby(coeff(12), sq6, coeff(10), sq4);
    let v1 = mm_axpby(F::one(), &tmp1, coeff(8), sq2);
    // V₂ = b₆·A⁶ + b₄·A⁴ + b₂·A² + b₀·I
    let tmp2 = mm_axpby(coeff(6), sq6, coeff(4), sq4);
    let tmp3 = mm_axpby(coeff(2), sq2, coeff(0), identity);
    let v2 = mm_axpby(F::one(), &tmp2, F::one(), &tmp3);
    // V = A⁶·V₁ + V₂
    let av1 = mm_mul::<F, M>(sq6, &v1);
    mm_axpby(F::one(), &av1, F::one(), &v2)
}

// ---------------------------------------------------------------------------
// Padé[13/13] scaling-and-squaring matrix exponential
// ---------------------------------------------------------------------------

/// Padé[13/13] matrix exponential for M×M matrices (ADR-0125, §33.8 Para 2).
///
/// Uses the Higham 2005 algorithm:
/// 1. Compute ∞-norm; scale A ← A/2ˢ so `‖A‖_∞ ≤ θ₁₃`.
/// 2. Evaluate the Padé U, V polynomials (even/odd degree-13 split).
/// 3. Solve R = (V−U)⁻¹(V+U) via in-tree partial-pivoting LU.
/// 4. Square R s times.
///
/// # Regime
/// Relative Frobenius error ≤ 1e-12 for symmetric matrices with
/// `‖A‖_∞ ≤ 10` (physical half-step regime). At `‖A‖_∞ ≈ 20` it
/// dips to ~1.9e-12 (squaring hump). Gate `G_MATRIX_PADE_M5` enforces
/// the ≤ 10 regime.
///
/// # Errors
/// Returns `Err` only if (V−U) is near-singular (not expected
/// in the physical regime for symmetric reaction matrices).
pub(crate) fn mat_exp_pade13<F: SemiflowFloat, const M: usize>(
    mat_a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    let n_sq = compute_squarings::<F, M>(mat_a);
    // Compute 1/2^n_sq by halving n_sq times (avoids f64→i32 cast).
    let inv_scale = (0..n_sq).fold(F::one(), |acc, _| {
        acc * F::from(0.5_f64).unwrap_or(F::one())
    });
    let scaled = mat_scale_entries(mat_a, inv_scale);

    let sq2 = mm_mul::<F, M>(&scaled, &scaled);
    let sq4 = mm_mul::<F, M>(&sq2, &sq2);
    let sq6 = mm_mul::<F, M>(&sq2, &sq4);
    let identity = mm_eye::<F, M>();
    let coeff = |idx: usize| -> F { F::from(PADE_B[idx]).unwrap_or(F::zero()) };

    let mat_u = pade13_u::<F, M>(&scaled, &sq2, &sq4, &sq6, &identity, &coeff);
    let mat_v = pade13_v::<F, M>(&sq2, &sq4, &sq6, &identity, &coeff);

    let numerator = mm_axpby(F::one(), &mat_v, F::one(), &mat_u); // V + U
    let denominator = mm_axpby(F::one(), &mat_v, -F::one(), &mat_u); // V - U
    let mut result = mm_solve::<F, M>(&denominator, &numerator)?;

    for _ in 0..n_sq {
        result = mm_mul::<F, M>(&result, &result);
    }
    Ok(result)
}

/// Compute number of squarings needed so `‖A/2ˢ‖_∞ ≤ θ₁₃`.
///
/// Uses integer doubling loop to avoid float→integer casts.
fn compute_squarings<F: SemiflowFloat, const M: usize>(mat_a: &[[F; M]; M]) -> u32 {
    let norm: F = mat_a.iter().fold(F::zero(), |best, row| {
        let row_sum = row.iter().fold(F::zero(), |acc, &val| acc + val.abs());
        if row_sum > best {
            row_sum
        } else {
            best
        }
    });
    let theta = F::from(THETA13).unwrap_or(F::one());
    if norm <= theta {
        return 0;
    }
    // Count squarings by doubling theta until it exceeds norm (max 63).
    let mut s = 1u32;
    let mut thresh = theta + theta; // 2 * theta after 1 squaring
    while thresh < norm && s < 63 {
        thresh = thresh + thresh;
        s += 1;
    }
    s
}

/// Scale all entries: `out[i][j] = alpha * mat_a[i][j]`.
fn mat_scale_entries<F: SemiflowFloat, const M: usize>(
    mat_a: &[[F; M]; M],
    alpha: F,
) -> [[F; M]; M] {
    let mut out = [[F::zero(); M]; M];
    for (out_row, in_row) in out.iter_mut().zip(mat_a.iter()) {
        for (o, &v) in out_row.iter_mut().zip(in_row.iter()) {
            *o = alpha * v;
        }
    }
    out
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    include!("matrix_pade_tests_mod.rs");
}
