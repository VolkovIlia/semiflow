//! M×M matrix inversion: Cramer's rule for M ∈ {1,2,3,4}, partial-pivoting LU
//! for M ≥ 5 (ADR-0125).
//!
//! Used by `matrix_strang::block_thomas_solve` for the per-block LU step and
//! by `matrix_pade::mat_exp_pade13` for the Padé (V−U)·X = (V+U) solve.
//!
//! References: Horn-Johnson §0.8 (Cramer), Golub-Van Loan §4.5.1, Higham 2005.

use crate::{error::SemiflowError, float::SemiflowFloat};

/// Dispatch M×M matrix inverse: Cramer's rule for M ∈ {1,2,3,4};
/// partial-pivoting LU for M ≥ 5 (ADR-0125, no new deps).
pub(crate) fn mat_inv_dispatch<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    match M {
        1 => mat_inv_m1(a),
        2 => mat_inv_m2(a),
        3 => mat_inv_m3(a),
        4 => mat_inv_m4(a),
        _ => mat_inv_lu::<F, M>(a),
    }
}

/// M×M matrix inverse via partial-pivoting LU (Golub-Van Loan §4.5.1).
///
/// No heap allocation: LU factorisation stored in a stack-allocated `[[F;M];M]`.
/// Inversion is done column-by-column by solving A·X = I.
///
/// Used for M ≥ 5 (block-Thomas + Padé solve, ADR-0125).
pub(crate) fn mat_inv_lu<F: SemiflowFloat, const M: usize>(
    mat_a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    let (lu, piv) = lu_factor(mat_a)?;
    let mut inv = [[F::zero(); M]; M];
    // Solve A·x_col = e_col for each column of the identity matrix.
    for cidx in 0..M {
        let xcol = lu_solve_col(&lu, &piv, cidx);
        // Write column cidx into inv (inv[row][cidx] = xcol[row]).
        write_column(&mut inv, cidx, &xcol);
    }
    Ok(inv)
}

/// Write `src` vector into column `cidx` of `mat`.
fn write_column<F: SemiflowFloat, const M: usize>(mat: &mut [[F; M]; M], cidx: usize, src: &[F; M]) {
    for (row, &val) in mat.iter_mut().zip(src.iter()) {
        row[cidx] = val;
    }
}

/// LU factorisation with partial pivoting. Returns `(lu, piv)`.
fn lu_factor<F: SemiflowFloat, const M: usize>(
    mat_a: &[[F; M]; M],
) -> Result<([[F; M]; M], [usize; M]), SemiflowError> {
    let mut lu = *mat_a;
    let mut piv = [0usize; M];
    for (idx, slot) in piv.iter_mut().enumerate() {
        *slot = idx;
    }
    let thresh = lu_inf_norm_eps::<F, M>(&lu);
    for pkc in 0..M {
        let max_row = find_pivot_row(&lu, pkc);
        if lu[max_row][pkc].abs() < thresh {
            return Err(SemiflowError::DomainViolation {
                what: "mat_inv_lu: near-singular matrix",
                value: 0.0,
            });
        }
        if max_row != pkc {
            lu.swap(pkc, max_row);
            piv.swap(pkc, max_row);
        }
        apply_lu_column(&mut lu, pkc);
    }
    Ok((lu, piv))
}

/// Find row with largest absolute value in column `pkc`, searching rows `pkc..M`.
fn find_pivot_row<F: SemiflowFloat, const M: usize>(lu: &[[F; M]; M], pkc: usize) -> usize {
    lu[pkc..M]
        .iter()
        .enumerate()
        .max_by(|(_, ra), (_, rb)| {
            ra[pkc]
                .abs()
                .partial_cmp(&rb[pkc].abs())
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .map_or(pkc, |(offset, _)| pkc + offset)
}

/// Eliminate column `pkc` of the LU matrix (scale multipliers + update submatrix).
fn apply_lu_column<F: SemiflowFloat, const M: usize>(lu: &mut [[F; M]; M], pkc: usize) {
    let inv_pivot = F::one() / lu[pkc][pkc];
    // Copy pivot row to avoid simultaneous mut + shared borrows.
    let mut pivot_row = [F::zero(); M];
    pivot_row[..M].copy_from_slice(&lu[pkc]);
    for row in lu.iter_mut().skip(pkc + 1) {
        row[pkc] *= inv_pivot;
        let mult = row[pkc];
        for (ec, &pivot_val) in pivot_row.iter().enumerate().skip(pkc + 1) {
            row[ec] -= mult * pivot_val;
        }
    }
}

/// Forward + back substitution for one column of the identity (col index `rhs_col`).
fn lu_solve_col<F: SemiflowFloat, const M: usize>(
    lu: &[[F; M]; M],
    piv: &[usize; M],
    rhs_col: usize,
) -> [F; M] {
    // Forward substitution: L·y = P·e_{rhs_col}.
    let mut yvec = [F::zero(); M];
    for fwd in 0..M {
        yvec[fwd] = if piv[fwd] == rhs_col {
            F::one()
        } else {
            F::zero()
        };
        for sub in 0..fwd {
            yvec[fwd] -= lu[fwd][sub] * yvec[sub];
        }
    }
    // Back substitution: U·x = y.
    let mut xvec = [F::zero(); M];
    for bwd in (0..M).rev() {
        xvec[bwd] = yvec[bwd];
        for sub in (bwd + 1)..M {
            xvec[bwd] -= lu[bwd][sub] * xvec[sub];
        }
        xvec[bwd] /= lu[bwd][bwd];
    }
    xvec
}

/// Singularity threshold: ε · ‖A‖_∞ · 64 (covers all M ≤ 64 const-generic).
fn lu_inf_norm_eps<F: SemiflowFloat, const M: usize>(lu: &[[F; M]; M]) -> F {
    let a_norm: F = lu.iter().fold(F::zero(), |acc, row| {
        let row_sum = row.iter().fold(F::zero(), |rs, &val| rs + val.abs());
        if row_sum > acc {
            row_sum
        } else {
            acc
        }
    });
    // Use 64 as upper bound on M (avoids usize→float cast precision warning).
    F::epsilon() * a_norm * F::from(64.0_f64).unwrap_or(F::one())
}

/// 1×1 inverse: `[[1/a[0][0]]]`.
fn mat_inv_m1<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    let d = a[0][0];
    if d == F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "mat_inv_m1: singular matrix (det=0)",
            value: 0.0,
        });
    }
    let mut out = [[F::zero(); M]; M];
    out[0][0] = F::one() / d;
    Ok(out)
}

/// 2×2 inverse via Cramer's rule.
fn mat_inv_m2<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det == F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "mat_inv_m2: singular matrix (det=0)",
            value: 0.0,
        });
    }
    let inv_det = F::one() / det;
    let mut out = [[F::zero(); M]; M];
    out[0][0] = a[1][1] * inv_det;
    out[0][1] = -(a[0][1] * inv_det);
    out[1][0] = -(a[1][0] * inv_det);
    out[1][1] = a[0][0] * inv_det;
    Ok(out)
}

/// 3×3 inverse via cofactor expansion (Cramer's rule).
fn mat_inv_m3<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
    let c01 = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]);
    let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
    let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
    if det == F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "mat_inv_m3: singular matrix (det=0)",
            value: 0.0,
        });
    }
    let inv_det = F::one() / det;
    let c10 = -(a[0][1] * a[2][2] - a[0][2] * a[2][1]);
    let c11 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
    let c12 = -(a[0][0] * a[2][1] - a[0][1] * a[2][0]);
    let c20 = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    let c21 = -(a[0][0] * a[1][2] - a[0][2] * a[1][0]);
    let c22 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    // Inverse = (1/det) * adjugate = (1/det) * cofactor_matrix^T.
    let mut out = [[F::zero(); M]; M];
    out[0][0] = c00 * inv_det;
    out[0][1] = c10 * inv_det;
    out[0][2] = c20 * inv_det;
    out[1][0] = c01 * inv_det;
    out[1][1] = c11 * inv_det;
    out[1][2] = c21 * inv_det;
    out[2][0] = c02 * inv_det;
    out[2][1] = c12 * inv_det;
    out[2][2] = c22 * inv_det;
    Ok(out)
}

/// Build the complement index set (indices 0..4 excluding `skip`).
#[inline]
fn complement3(skip: usize) -> [usize; 3] {
    let mut out = [0usize; 3];
    let mut idx = 0;
    for k in 0..4usize {
        if k != skip {
            out[idx] = k;
            idx += 1;
        }
    }
    out
}

/// 4×4 inverse via cofactor expansion (Cramer's rule using 3×3 cofactors).
fn mat_inv_m4<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
) -> Result<[[F; M]; M], SemiflowError> {
    let det2 = |r0: usize, r1: usize, c0: usize, c1: usize| -> F {
        a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]
    };
    let det3 = |r0: usize, r1: usize, r2: usize, c0: usize, c1: usize, c2: usize| -> F {
        a[r0][c0] * det2(r1, r2, c1, c2) - a[r0][c1] * det2(r1, r2, c0, c2)
            + a[r0][c2] * det2(r1, r2, c0, c1)
    };
    let cf = |row: usize, col: usize| -> F {
        let rows = complement3(row);
        let cols = complement3(col);
        let minor = det3(rows[0], rows[1], rows[2], cols[0], cols[1], cols[2]);
        if (row + col) % 2 == 0 {
            minor
        } else {
            -minor
        }
    };
    let det = (0..4).fold(F::zero(), |acc, j| acc + a[0][j] * cf(0, j));
    if det == F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "mat_inv_m4: singular matrix (det=0)",
            value: 0.0,
        });
    }
    let inv_det = F::one() / det;
    // Adjugate^T: inv[i][j] = cf(j,i)/det; i,j used as transposed args.
    let mut out = [[F::zero(); M]; M];
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = cf(j, i) * inv_det;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
// i,j used for both indexing and conditional `if i == j`; range loop is clearest.
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::matrix_strang::block_mat_mul;

    #[test]
    fn mat_inv_m1_identity_check() {
        const M: usize = 1;
        let a: [[f64; M]; M] = [[3.7]];
        let inv = mat_inv_dispatch::<f64, M>(&a).unwrap();
        let prod = block_mat_mul::<f64, M>(&a, &inv);
        assert!(
            (prod[0][0] - 1.0).abs() < 1e-14,
            "mat_inv_m1 A*A^-1 != I: {prod:?}"
        );
    }

    #[test]
    fn mat_inv_m2_identity_check() {
        const M: usize = 2;
        let a: [[f64; M]; M] = [[4.0, 1.5], [1.5, 3.0]];
        let inv = mat_inv_dispatch::<f64, M>(&a).unwrap();
        let prod = block_mat_mul::<f64, M>(&a, &inv);
        for i in 0..M {
            for j in 0..M {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[i][j] - expected).abs() < 1e-12,
                    "mat_inv_m2 A*A^-1 != I at ({i},{j}): got {val:.4e}",
                    val = prod[i][j]
                );
            }
        }
    }

    #[test]
    fn mat_inv_m3_identity_check() {
        const M: usize = 3;
        let a: [[f64; M]; M] = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.8], [0.5, 0.8, 2.5]];
        let inv = mat_inv_dispatch::<f64, M>(&a).unwrap();
        let prod = block_mat_mul::<f64, M>(&a, &inv);
        for i in 0..M {
            for j in 0..M {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[i][j] - expected).abs() < 1e-11,
                    "mat_inv_m3 A*A^-1 != I at ({i},{j}): got {val:.4e}",
                    val = prod[i][j]
                );
            }
        }
    }

    #[test]
    fn mat_inv_m4_identity_check() {
        const M: usize = 4;
        let a: [[f64; M]; M] = [
            [5.0, 1.0, 0.5, 0.2],
            [1.0, 4.0, 0.8, 0.3],
            [0.5, 0.8, 3.5, 0.6],
            [0.2, 0.3, 0.6, 2.5],
        ];
        let inv = mat_inv_dispatch::<f64, M>(&a).unwrap();
        let prod = block_mat_mul::<f64, M>(&a, &inv);
        for i in 0..M {
            for j in 0..M {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[i][j] - expected).abs() < 1e-10,
                    "mat_inv_m4 A*A^-1 != I at ({i},{j}): got {val:.4e}",
                    val = prod[i][j]
                );
            }
        }
    }
}
