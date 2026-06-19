//! Dense matrix exponential and one-sided Jacobi SVD.
//!
//! The `dense_expm` / dense matrix primitives are retained as the INDEPENDENT
//! reference for `d2_exactness_self_check` in `tt_coupled.rs` (they confirm the
//! spectral apply produces the same result as LU-Padé expm).
//! The production coupling path now uses `tt_spectral` (solver-free spectral apply).
//! `one_sided_jacobi_svd` is still used on the production path for the TT-SVD split.
//!
//! ## Padé[6/6] for expm
//! Coefficients from Higham (2008) Table 10.2. Scaling & squaring for `‖A‖_∞ > 0.5`.
//! Accurate to ~1e-15 for `‖A‖_∞ ≤ 50` (7 squarings max).
//!
//! ## One-sided Jacobi SVD
//! Directly on `A` (NOT the Gram `A^T*A`). Achieves high relative accuracy for
//! ALL singular values, including the small tail (pair-op-rank ~6, SV7 ≈ 1e-9).
//! The Gram approach would square the condition number (5e9 → 2.5e19 >> `1/eps_mach`).
#![allow(dead_code)] // dense_expm + dense matrix primitives used only by tests (independent reference)

extern crate alloc;
use alloc::{vec, vec::Vec};

use crate::float::SemiflowFloat;

// ═══════════════════════════════════════════════════════════════════════════
// §A — Dense matrix primitives (row-major, m×m)
// ═══════════════════════════════════════════════════════════════════════════

/// Dense matrix-vector: `out = A · v`.
pub(crate) fn mat_vec<F: SemiflowFloat>(a: &[F], v: &[F], m: usize) -> Vec<F> {
    let mut out = vec![F::zero(); m];
    for i in 0..m {
        let mut s = F::zero();
        for j in 0..m {
            s += a[i * m + j] * v[j];
        }
        out[i] = s;
    }
    out
}

/// Dense matrix-matrix: `C = A · B`.
pub(crate) fn mat_mat<F: SemiflowFloat>(a: &[F], b: &[F], m: usize) -> Vec<F> {
    let mut c = vec![F::zero(); m * m];
    for i in 0..m {
        for k in 0..m {
            if a[i * m + k] == F::zero() {
                continue;
            }
            let aik = a[i * m + k];
            for j in 0..m {
                c[i * m + j] += aik * b[k * m + j];
            }
        }
    }
    c
}

/// Infinity norm.
pub(crate) fn mat_norm_inf<F: SemiflowFloat>(a: &[F], m: usize) -> F {
    let mut norm = F::zero();
    for i in 0..m {
        let mut row_sum = F::zero();
        for j in 0..m {
            row_sum += a[i * m + j].abs();
        }
        if row_sum > norm {
            norm = row_sum;
        }
    }
    norm
}

/// Scale in-place: `A ← s·A`.
pub(crate) fn mat_scale<F: SemiflowFloat>(a: &mut [F], s: F) {
    for v in a.iter_mut() {
        *v *= s;
    }
}

/// Add: `A ← A + B`.
pub(crate) fn mat_add<F: SemiflowFloat>(a: &mut [F], b: &[F]) {
    for (ai, &bi) in a.iter_mut().zip(b.iter()) {
        *ai += bi;
    }
}

/// Identity matrix.
pub(crate) fn mat_eye<F: SemiflowFloat>(m: usize) -> Vec<F> {
    let mut e = vec![F::zero(); m * m];
    for i in 0..m {
        e[i * m + i] = F::one();
    }
    e
}

/// LU solve `A·x = b` in-place with partial pivoting.
pub(crate) fn lu_solve_inplace<F: SemiflowFloat>(a_lu: &mut [F], b: &mut [F], m: usize) {
    for col in 0..m {
        let mut max_val = a_lu[col * m + col].abs();
        let mut max_row = col;
        for row in (col + 1)..m {
            let v = a_lu[row * m + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..m {
                a_lu.swap(col * m + j, max_row * m + j);
            }
            b.swap(col, max_row);
        }
        let pivot = a_lu[col * m + col];
        if pivot.abs() < F::from(1e-300).unwrap() {
            continue;
        }
        let inv_pivot = F::one() / pivot;
        for row in (col + 1)..m {
            let factor = a_lu[row * m + col] * inv_pivot;
            a_lu[row * m + col] = F::zero();
            for j in (col + 1)..m {
                let a_cj = a_lu[col * m + j];
                a_lu[row * m + j] -= factor * a_cj;
            }
            b[row] -= factor * b[col];
        }
    }
    for row in (0..m).rev() {
        for col in (row + 1)..m {
            b[row] -= a_lu[row * m + col] * b[col];
        }
        let d = a_lu[row * m + row];
        if d.abs() > F::from(1e-300).unwrap() {
            b[row] /= d;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §B — Padé[6/6] matrix exponential
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Padé(6,6) numerator V and denominator factor U.
/// V = c0·I + c2·A² + c4·A⁴ + c6·A⁶;  U = A·(c1·I + c3·A² + c5·A⁴).
fn pade_numerator_denominator<F: SemiflowFloat>(
    a_s: &[F],
    a2: &[F],
    a4: &[F],
    a6: &[F],
    m: usize,
    coeff: &impl Fn(usize) -> F,
) -> (Vec<F>, Vec<F>) {
    let mut v = mat_eye(m);
    mat_scale(&mut v, coeff(0));
    let mut t = a2.to_vec();
    mat_scale(&mut t, coeff(2));
    mat_add(&mut v, &t);
    let mut t = a4.to_vec();
    mat_scale(&mut t, coeff(4));
    mat_add(&mut v, &t);
    let mut t = a6.to_vec();
    mat_scale(&mut t, coeff(6));
    mat_add(&mut v, &t);
    let mut inner = mat_eye(m);
    mat_scale(&mut inner, coeff(1));
    let mut t = a2.to_vec();
    mat_scale(&mut t, coeff(3));
    mat_add(&mut inner, &t);
    let mut t = a4.to_vec();
    mat_scale(&mut t, coeff(5));
    mat_add(&mut inner, &t);
    let u = mat_mat(a_s, &inner, m);
    (v, u)
}

/// `expm(A)` via Padé(6,6) + scaling & squaring (heap, no new deps).
///
/// Coefficients: Higham 2008 Table 10.2, `c_k` = (12-k)!6!/(12!k!(6-k)!).
// a, m, s, p, q, u, v: standard matrix-expm names from Higham 2008; match the paper.
#[allow(clippy::many_single_char_names)]
pub(crate) fn dense_expm<F: SemiflowFloat>(a: &[F], m: usize) -> Vec<F> {
    let norm = mat_norm_inf(a, m);
    let half = F::from(0.5).unwrap();
    let mut s = 0u32;
    let mut thresh = half;
    while norm > thresh && s < 30 {
        s += 1;
        thresh *= F::from(2.0).unwrap();
    }
    let scale = (0..s).fold(F::one(), |acc, _| acc * half);
    let mut a_s = a.to_vec();
    mat_scale(&mut a_s, scale);

    // Padé(6,6) numerator coefficients c[0..=6] (Higham 2008 Table 10.2)
    let pade_c: [f64; 7] = [
        1.0,
        0.5,
        5.0 / 44.0,
        1.0 / 66.0,
        1.0 / 792.0,
        1.0 / 15_840.0,
        1.0 / 665_280.0,
    ];
    let coeff = |k: usize| F::from(pade_c[k]).unwrap();

    let a2 = mat_mat(&a_s, &a_s, m);
    let a4 = mat_mat(&a2, &a2, m);
    let a6 = mat_mat(&a2, &a4, m);
    let (v, u) = pade_numerator_denominator(&a_s, &a2, &a4, &a6, m, &coeff);

    // expm_s = (V−U)^{-1}·(V+U) via column-by-column LU solves.
    let mut expm_s = pade_lu_solve(&v, &u, m);
    for _ in 0..s {
        expm_s = mat_mat(&expm_s, &expm_s, m);
    }
    expm_s
}

/// Compute `q^{-1} p` where `p = v + u`, `q = v − u`, via column-by-column LU.
// p, q, u, v, m, col, row: standard Higham 2008 / linear-algebra names.
#[allow(clippy::many_single_char_names)]
fn pade_lu_solve<F: SemiflowFloat>(v: &[F], u: &[F], m: usize) -> Vec<F> {
    let mut p = v.to_vec();
    mat_add(&mut p, u);
    let mut q = v.to_vec();
    for (qi, &ui) in q.iter_mut().zip(u.iter()) {
        *qi -= ui;
    }
    let mut result = vec![F::zero(); m * m];
    for col in 0..m {
        let mut rhs: Vec<F> = (0..m).map(|row| p[row * m + col]).collect();
        let mut q_col = q.clone();
        lu_solve_inplace(&mut q_col, &mut rhs, m);
        for row in 0..m {
            result[row * m + col] = rhs[row];
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — One-sided Jacobi SVD (numerically stable, no Gram squaring)
// ═══════════════════════════════════════════════════════════════════════════

/// Truncated SVD of `A` (`m×n` row-major) via right one-sided Jacobi rotations.
///
/// Returns `(U, s, V)`: U is `m×r`, s is `r`, V is `n×r` (all row-major),
/// sorted descending, `r = count(σ ≥ eps * σ_max)`.
///
/// High relative accuracy for ALL singular values (condition not squared).
pub(crate) fn one_sided_jacobi_svd<F: SemiflowFloat>(
    a_mat: &[F],
    m: usize,
    n: usize,
    eps: F,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    if m == 0 || n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    // Column-major work copy: a_work[col*m+row] = a_mat[row*n+col]
    let mut a_work = vec![F::zero(); m * n];
    for row in 0..m {
        for col in 0..n {
            a_work[col * m + row] = a_mat[row * n + col];
        }
    }
    // V: column-major identity n×n
    let mut v_mat = vec![F::zero(); n * n];
    for i in 0..n {
        v_mat[i * n + i] = F::one();
    }

    jacobi_sweep_all(&mut a_work, &mut v_mat, m, n);
    jacobi_extract_usvt(&a_work, &v_mat, m, n, eps)
}

/// Run up to 100 Jacobi sweeps over all (p,q) pairs, rotating `a_work` and `v_mat`.
///
/// Terminates early when all off-diagonal correlations are below `tol = 1e-14`.
/// Both `a_work` and `v_mat` are column-major (`col*dim+row` layout).
fn jacobi_sweep_all<F: SemiflowFloat>(a_work: &mut [F], v_mat: &mut [F], m: usize, n: usize) {
    let two = F::from(2.0).unwrap();
    let tol = F::from(1e-14).unwrap();
    for _sweep in 0..100 {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let (mut alpha, mut bp, mut bq) = (F::zero(), F::zero(), F::zero());
                for row in 0..m {
                    let ap = a_work[p * m + row];
                    let aq = a_work[q * m + row];
                    alpha += ap * aq;
                    bp += ap * ap;
                    bq += aq * aq;
                }
                let sq = (bp * bq).sqrt();
                if sq <= F::zero() || (alpha / sq).abs() <= tol {
                    continue;
                }
                converged = false;
                apply_jacobi_rotation(a_work, v_mat, m, n, p, q, two, alpha, bp, bq);
            }
        }
        if converged {
            break;
        }
    }
}

/// Compute and apply a single Jacobi rotation to columns `(p, q)` of `a_work` and `v_mat`.
// two/alpha/bp/bq are F: SemiflowFloat (likely Copy); pass by value is idiomatic for small scalars.
// a_work, v_mat, m, n, p, q, two, alpha, bp, bq: all 10 required for the Jacobi step.
#[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
fn apply_jacobi_rotation<F: SemiflowFloat>(
    a_work: &mut [F],
    v_mat: &mut [F],
    m: usize,
    n: usize,
    p: usize,
    q: usize,
    two: F,
    alpha: F,
    bp: F,
    bq: F,
) {
    let zeta = (bq - bp) / (two * alpha);
    let tan_t = if zeta >= F::zero() {
        F::one() / (zeta + (F::one() + zeta * zeta).sqrt())
    } else {
        F::one() / (zeta - (F::one() + zeta * zeta).sqrt())
    };
    let cos_t = F::one() / (F::one() + tan_t * tan_t).sqrt();
    let sin_t = cos_t * tan_t;
    // Rotate columns p,q of A_work
    for row in 0..m {
        let ap = a_work[p * m + row];
        let aq = a_work[q * m + row];
        a_work[p * m + row] = cos_t * ap - sin_t * aq;
        a_work[q * m + row] = sin_t * ap + cos_t * aq;
    }
    // Rotate columns p,q of V
    for row in 0..n {
        let vp = v_mat[p * n + row];
        let vq = v_mat[q * n + row];
        v_mat[p * n + row] = cos_t * vp - sin_t * vq;
        v_mat[q * n + row] = sin_t * vp + cos_t * vq;
    }
}

/// Extract `(U, sv, V)` from the converged column-major `a_work` and `v_mat`.
///
/// Computes column norms (singular values), sorts descending, truncates at
/// `r = count(σ ≥ eps * σ_max)`, and builds row-major output matrices.
fn jacobi_extract_usvt<F: SemiflowFloat>(
    a_work: &[F],
    v_mat: &[F],
    m: usize,
    n: usize,
    eps: F,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let (idx, sv_sorted, r) = sort_sv_descending(a_work, m, n, eps);

    // U (m×r): normalized columns of A_work
    let mut u_out = vec![F::zero(); m * r];
    for (nc, &oc) in idx[..r].iter().enumerate() {
        let sv = sv_sorted[nc];
        let inv = if sv > F::zero() {
            F::one() / sv
        } else {
            F::zero()
        };
        for row in 0..m {
            u_out[row * r + nc] = a_work[oc * m + row] * inv;
        }
    }

    // V (n×r): reordered V columns
    let mut v_out = vec![F::zero(); n * r];
    for (nc, &oc) in idx[..r].iter().enumerate() {
        for row in 0..n {
            v_out[row * r + nc] = v_mat[oc * n + row];
        }
    }

    (u_out, sv_sorted[..r].to_vec(), v_out)
}

/// Compute column norms of column-major `a_work` (`m×n`), sort descending,
/// and return `(sorted_indices, sorted_sv, rank_r)` where `r = count(σ ≥ eps·σ_max)`.
fn sort_sv_descending<F: SemiflowFloat>(
    a_work: &[F],
    m: usize,
    n: usize,
    eps: F,
) -> (Vec<usize>, Vec<F>, usize) {
    let sv_all: Vec<F> = (0..n)
        .map(|col| {
            let mut ns = F::zero();
            for row in 0..m {
                let v = a_work[col * m + row];
                ns += v * v;
            }
            ns.sqrt()
        })
        .collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        sv_all[b]
            .partial_cmp(&sv_all[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let sv_sorted: Vec<F> = idx.iter().map(|&i| sv_all[i]).collect();
    let threshold = if sv_sorted[0] > F::zero() {
        eps * sv_sorted[0]
    } else {
        F::zero()
    };
    let r = sv_sorted
        .iter()
        .filter(|&&s| s > threshold)
        .count()
        .max(1)
        .min(n);
    (idx, sv_sorted, r)
}
