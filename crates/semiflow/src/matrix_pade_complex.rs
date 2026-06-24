//! Padé[13/13] matrix exponential for M×M **complex** matrices (ADR-0128).
//!
//! Identical algorithm to `matrix_pade.rs`; scalar type changed from
//! `F: SemiflowFloat` to `C: SemiflowComplex`. The Padé coefficients `PADE_B`
//! are **real** — they are embedded into `C` via `C::from_real(...)` and
//! applied to the complex matrix argument unchanged (Higham 2005 §4; the
//! rational approximant structure is independent of whether the argument is
//! real or complex).
//!
//! ## Differences from the real path (`matrix_pade.rs`)
//!
//! | Aspect | Real path | Complex path |
//! |--------|-----------|--------------|
//! | Element type | `F: SemiflowFloat` | `C: SemiflowComplex` |
//! | Inf-norm | `val.abs()` (f64 abs) | `val.abs()` (complex modulus) |
//! | Coeff embed | `F::from(PADE_B[i])` | `C::from_real(F::from(PADE_B[i]))` |
//! | LU solve | `matrix_inv::mat_inv_lu` | `mat_inv_lu_complex` (below) |
//!
//! The real path in `matrix_pade.rs` is **byte-identical** after this change
//! (zero shared code paths; purely additive).
//!
//! Gate: `G_CPLX_MATRIX` — relative Frobenius error ≤ 1e-12 for non-Hermitian
//! complex M×M matrices (M ∈ {5,6,8}) with `‖A‖_∞ ≤ 10`. Unitarity:
//! `‖Uᴴ U − I‖_F ≤ 1e-12` for `U = exp(iH)`, Hermitian H.
//!
//! ADR-0128; contracts/semiflow-core.math.md §33.8 Para 3.

use num_traits::{Float, NumCast, Zero};

use crate::{complex::SemiflowComplex, error::SemiflowError};

/// Convert f64 to `F: Float + Zero`. Mirrors `float::from_f64`.
#[inline]
pub(crate) fn real_from_f64_cplx<F: Float + Zero>(v: f64) -> F {
    <F as NumCast>::from(v).unwrap_or_else(F::zero)
}

// Private alias.
#[inline]
fn real_from_f64<F: Float + Zero>(v: f64) -> F {
    real_from_f64_cplx(v)
}

// Re-use the real constants from the sibling module.
const THETA13: f64 = 5.371_920_351_148_152;

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
// Complex M×M matrix helpers (stack-allocated, no_std + alloc)
// ---------------------------------------------------------------------------

#[inline]
fn cmm_mul<C: SemiflowComplex, const M: usize>(a: &[[C; M]; M], b: &[[C; M]; M]) -> [[C; M]; M] {
    let mut c = [[C::zero(); M]; M];
    for (ridx, row_c) in c.iter_mut().enumerate() {
        for mid in 0..M {
            let a_val = a[ridx][mid];
            for (cidx, elem) in row_c.iter_mut().enumerate() {
                *elem += a_val * b[mid][cidx];
            }
        }
    }
    c
}

/// `c = alpha·a + beta·b` where `alpha`, `beta` are complex scalars.
#[inline]
fn cmm_axpby<C: SemiflowComplex, const M: usize>(
    alpha: C,
    a: &[[C; M]; M],
    beta: C,
    b: &[[C; M]; M],
) -> [[C; M]; M] {
    let mut c = [[C::zero(); M]; M];
    for (rc, (ra, rb)) in c.iter_mut().zip(a.iter().zip(b.iter())) {
        for (ec, (&ea, &eb)) in rc.iter_mut().zip(ra.iter().zip(rb.iter())) {
            *ec = alpha * ea + beta * eb;
        }
    }
    c
}

#[inline]
fn cmm_eye<C: SemiflowComplex, const M: usize>() -> [[C; M]; M] {
    let mut m = [[C::zero(); M]; M];
    for (d, row) in m.iter_mut().enumerate() {
        row[d] = C::one();
    }
    m
}

// ---------------------------------------------------------------------------
// Complex partial-pivoting LU + solve (no new deps)
// ---------------------------------------------------------------------------

/// Complex M×M matrix inversion via partial-pivoting LU.
///
/// Mirrors `matrix_inv::mat_inv_lu` but with `C: SemiflowComplex`.
/// Pivot selection uses complex modulus for the "largest" pivot.
fn mat_inv_lu_complex<C: SemiflowComplex, const M: usize>(
    mat_a: &[[C; M]; M],
) -> Result<[[C; M]; M], SemiflowError> {
    let (lu, piv) = lu_factor_complex(mat_a)?;
    let mut inv = [[C::zero(); M]; M];
    for cidx in 0..M {
        let xcol = lu_solve_complex(&lu, &piv, cidx);
        for (row, &val) in inv.iter_mut().zip(xcol.iter()) {
            row[cidx] = val;
        }
    }
    Ok(inv)
}

/// Matrix ∞-norm: max row sum of |entries|.
fn mat_inf_norm_complex<C: SemiflowComplex, const M: usize>(lu: &[[C; M]; M]) -> C::Real {
    lu.iter().fold(C::Real::zero(), |best, row| {
        let rs = row.iter().fold(C::Real::zero(), |acc, &v| {
            let av = v.abs();
            if av > acc {
                av
            } else {
                acc
            }
        });
        if rs > best {
            rs
        } else {
            best
        }
    })
}

/// Build the identity pivot array `[0, 1, …, M-1]`.
#[inline]
fn identity_pivot<const M: usize>() -> [usize; M] {
    let mut p = [0usize; M];
    for (i, v) in p.iter_mut().enumerate() {
        *v = i;
    }
    p
}

fn lu_factor_complex<C: SemiflowComplex, const M: usize>(
    mat_a: &[[C; M]; M],
) -> Result<([[C; M]; M], [usize; M]), SemiflowError> {
    let mut lu = *mat_a;
    let mut piv: [usize; M] = identity_pivot();
    // Singularity threshold: eps_machine * inf-norm * 64.
    let inf_norm: C::Real = mat_inf_norm_complex(&lu);
    let eps: C::Real = Float::epsilon();
    let thresh = eps * inf_norm * real_from_f64(64.0_f64);
    for pkc in 0..M {
        // Find row with largest complex modulus in column pkc.
        let max_row = (pkc..M)
            .max_by(|&r1, &r2| {
                lu[r1][pkc]
                    .abs()
                    .partial_cmp(&lu[r2][pkc].abs())
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap_or(pkc);
        if lu[max_row][pkc].abs() < thresh {
            return Err(SemiflowError::DomainViolation {
                what: "mat_inv_lu_complex: near-singular matrix",
                value: 0.0,
            });
        }
        if max_row != pkc {
            lu.swap(pkc, max_row);
            piv.swap(pkc, max_row);
        }
        // Elimination.
        let inv_pivot = C::one() / lu[pkc][pkc];
        let mut pivot_row = [C::zero(); M];
        pivot_row[..M].copy_from_slice(&lu[pkc]);
        for row in lu.iter_mut().skip(pkc + 1) {
            row[pkc] *= inv_pivot;
            let mult = row[pkc];
            for (ec, &pv) in pivot_row.iter().enumerate().skip(pkc + 1) {
                row[ec] -= mult * pv;
            }
        }
    }
    Ok((lu, piv))
}

fn lu_solve_complex<C: SemiflowComplex, const M: usize>(
    lu: &[[C; M]; M],
    piv: &[usize; M],
    rhs_col: usize,
) -> [C; M] {
    // Forward: L·y = P·e_{rhs_col}.
    let mut y = [C::zero(); M];
    for fwd in 0..M {
        y[fwd] = if piv[fwd] == rhs_col {
            C::one()
        } else {
            C::zero()
        };
        for sub in 0..fwd {
            let sub_val = lu[fwd][sub] * y[sub];
            y[fwd] -= sub_val;
        }
    }
    // Back: U·x = y.
    let mut x = [C::zero(); M];
    for bwd in (0..M).rev() {
        x[bwd] = y[bwd];
        for sub in (bwd + 1)..M {
            let sv = lu[bwd][sub] * x[sub];
            x[bwd] -= sv;
        }
        x[bwd] /= lu[bwd][bwd];
    }
    x
}

fn cmm_solve<C: SemiflowComplex, const M: usize>(
    a: &[[C; M]; M],
    b: &[[C; M]; M],
) -> Result<[[C; M]; M], SemiflowError> {
    let a_inv = mat_inv_lu_complex::<C, M>(a)?;
    Ok(cmm_mul::<C, M>(&a_inv, b))
}

// ---------------------------------------------------------------------------
// Padé[13/13] U and V polynomials (complex scalar embeddings)
// ---------------------------------------------------------------------------

fn pade13_u_complex<C: SemiflowComplex, const M: usize>(
    scaled: &[[C; M]; M],
    sq2: &[[C; M]; M],
    sq4: &[[C; M]; M],
    sq6: &[[C; M]; M],
    eye: &[[C; M]; M],
    cb: impl Fn(usize) -> C,
) -> [[C; M]; M] {
    // W1 = b13·A6 + b11·A4 + b9·A2
    let tmp1 = cmm_axpby(cb(13), sq6, cb(11), sq4);
    let w1 = cmm_axpby(C::one(), &tmp1, cb(9), sq2);
    // W2 = b7·A6 + b5·A4 + b3·A2 + b1·I
    let tmp2 = cmm_axpby(cb(7), sq6, cb(5), sq4);
    let tmp3 = cmm_axpby(cb(3), sq2, cb(1), eye);
    let w2 = cmm_axpby(C::one(), &tmp2, C::one(), &tmp3);
    // U = A_scaled · (A6·W1 + W2)
    let aw1 = cmm_mul::<C, M>(sq6, &w1);
    let inner = cmm_axpby(C::one(), &aw1, C::one(), &w2);
    cmm_mul::<C, M>(scaled, &inner)
}

fn pade13_v_complex<C: SemiflowComplex, const M: usize>(
    sq2: &[[C; M]; M],
    sq4: &[[C; M]; M],
    sq6: &[[C; M]; M],
    eye: &[[C; M]; M],
    cb: impl Fn(usize) -> C,
) -> [[C; M]; M] {
    // V1 = b12·A6 + b10·A4 + b8·A2
    let tmp1 = cmm_axpby(cb(12), sq6, cb(10), sq4);
    let v1 = cmm_axpby(C::one(), &tmp1, cb(8), sq2);
    // V2 = b6·A6 + b4·A4 + b2·A2 + b0·I
    let tmp2 = cmm_axpby(cb(6), sq6, cb(4), sq4);
    let tmp3 = cmm_axpby(cb(2), sq2, cb(0), eye);
    let v2 = cmm_axpby(C::one(), &tmp2, C::one(), &tmp3);
    // V = A6·V1 + V2
    let av1 = cmm_mul::<C, M>(sq6, &v1);
    cmm_axpby(C::one(), &av1, C::one(), &v2)
}

// ---------------------------------------------------------------------------
// Number of squarings: same formula, complex modulus for row sums
// ---------------------------------------------------------------------------

fn compute_squarings_complex<C: SemiflowComplex, const M: usize>(mat_a: &[[C; M]; M]) -> u32 {
    // ‖A‖_∞ via complex-modulus row sums.
    let norm: C::Real = mat_a.iter().fold(C::Real::zero(), |best, row| {
        let rs = row.iter().fold(C::Real::zero(), |acc, &v| {
            let av = v.abs();
            if av > acc {
                av
            } else {
                acc
            }
        });
        if rs > best {
            rs
        } else {
            best
        }
    });
    let theta: C::Real = real_from_f64(THETA13);
    if norm <= theta {
        return 0;
    }
    let mut s = 1u32;
    let mut thresh = theta + theta;
    while thresh < norm && s < 63 {
        thresh = thresh + thresh;
        s += 1;
    }
    s
}

fn cmat_scale<C: SemiflowComplex, const M: usize>(mat_a: &[[C; M]; M], alpha: C) -> [[C; M]; M] {
    let mut out = [[C::zero(); M]; M];
    for (out_row, in_row) in out.iter_mut().zip(mat_a.iter()) {
        for (o, &v) in out_row.iter_mut().zip(in_row.iter()) {
            *o = alpha * v;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Public: Padé[13/13] complex matrix exponential
// ---------------------------------------------------------------------------

/// Padé[13/13] matrix exponential for complex M×M matrices (ADR-0128, §33.8 Para 3).
///
/// Identical algorithm to `mat_exp_pade13` in `matrix_pade.rs`; argument type
/// changed to `C: SemiflowComplex`. The real Padé coefficients `PADE_B` are
/// embedded via `C::from_real(...)` — the rational approximant is the same.
///
/// # Regime
/// Relative Frobenius error ≤ 1e-12 for `‖A‖_∞ ≤ 10`; matched by
/// `mat_exp_pade13` on the real path (§33.8 Para 3).
///
/// # Errors
/// `Err` if `(V−U)` is near-singular (not expected in the physical regime).
pub(crate) fn mat_exp_pade13_complex<C: SemiflowComplex, const M: usize>(
    mat_a: &[[C; M]; M],
) -> Result<[[C; M]; M], SemiflowError> {
    let n_sq = compute_squarings_complex::<C, M>(mat_a);
    // 1/2^n_sq by repeated halving.
    let half = C::from_real(real_from_f64::<C::Real>(0.5_f64));
    let inv_scale = (0..n_sq).fold(C::one(), |acc, _| acc * half);
    let scaled = cmat_scale(mat_a, inv_scale);

    let sq2 = cmm_mul::<C, M>(&scaled, &scaled);
    let sq4 = cmm_mul::<C, M>(&sq2, &sq2);
    let sq6 = cmm_mul::<C, M>(&sq2, &sq4);
    let eye = cmm_eye::<C, M>();

    // Embed real coefficient as complex real part.
    let cb = |idx: usize| -> C { C::from_real(real_from_f64::<C::Real>(PADE_B[idx])) };
    let mat_u = pade13_u_complex::<C, M>(&scaled, &sq2, &sq4, &sq6, &eye, &cb);
    let mat_v = pade13_v_complex::<C, M>(&sq2, &sq4, &sq6, &eye, &cb);

    let numerator = cmm_axpby(C::one(), &mat_v, C::one(), &mat_u); // V + U
    let denominator = cmm_axpby(C::one(), &mat_v, -C::one(), &mat_u); // V - U
    let mut result = cmm_solve::<C, M>(&denominator, &numerator)?;

    for _ in 0..n_sq {
        result = cmm_mul::<C, M>(&result, &result);
    }
    Ok(result)
}

/// Dispatch complex M×M matrix inverse (Cramer for M≤4; LU for M≥5).
///
/// Used by both `mat_exp_pade13_complex` (Padé solve) and
/// `matrix_system_complex::complex_block_cn_diff_step` (block-Thomas pivot).
pub(crate) fn cmat_inv_complex_dispatch<C: SemiflowComplex, const M: usize>(
    a: &[[C; M]; M],
) -> Result<[[C; M]; M], SemiflowError> {
    match M {
        1 => cmat_inv_1::<C, M>(a),
        2 => cmat_inv_2::<C, M>(a),
        _ => mat_inv_lu_complex::<C, M>(a),
    }
}

fn cmat_inv_1<C: SemiflowComplex, const M: usize>(
    a: &[[C; M]; M],
) -> Result<[[C; M]; M], SemiflowError> {
    let eps: C::Real = Float::epsilon();
    if a[0][0].abs() < eps {
        return Err(SemiflowError::DomainViolation {
            what: "cmat_inv_1: singular matrix",
            value: 0.0,
        });
    }
    let mut out = [[C::zero(); M]; M];
    out[0][0] = C::one() / a[0][0];
    Ok(out)
}

fn cmat_inv_2<C: SemiflowComplex, const M: usize>(
    a: &[[C; M]; M],
) -> Result<[[C; M]; M], SemiflowError> {
    let eps: C::Real = Float::epsilon();
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < eps {
        return Err(SemiflowError::DomainViolation {
            what: "cmat_inv_2: singular matrix",
            value: 0.0,
        });
    }
    let inv_det = C::one() / det;
    let mut out = [[C::zero(); M]; M];
    out[0][0] = a[1][1] * inv_det;
    out[0][1] = -(a[0][1] * inv_det);
    out[1][0] = -(a[1][0] * inv_det);
    out[1][1] = a[0][0] * inv_det;
    Ok(out)
}

/// M×M complex matrix-vector multiply: `out = A·v`.
///
/// (test hook exposed as `pub(crate)` — used by `matrix_pade_complex_tests_mod.rs`)
#[inline]
pub(crate) fn cmat_vec_mul<C: SemiflowComplex, const M: usize>(
    a: &[[C; M]; M],
    v: &[C; M],
) -> [C; M] {
    let mut out = [C::zero(); M];
    for i in 0..M {
        for j in 0..M {
            out[i] += a[i][j] * v[j];
        }
    }
    out
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    include!("matrix_pade_complex_tests_mod.rs");
}
