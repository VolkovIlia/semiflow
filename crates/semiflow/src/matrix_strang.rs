//! Block-Thomas solver + block arithmetic helpers for `MatrixDiffusionChernoff`.
//!
//! Phase 2 block Crank-Nicolson Cayley map per ADR-0082 AMENDMENT 2.
//!
//! ## Algorithm
//!
//! Per spatial point k, solve:
//! ```text
//! (I − τ/2 · L^h_k) · u^(2)_k = (I + τ/2 · L^h_k) · u^(1)_k
//! ```
//! where `L^h_k = A(x_k)·D²_x + B(x_k)·D_x` is the M×M-matrix-valued spatial
//! operator at grid point k. The full block-tridiagonal system for all N grid
//! points is solved via block-Thomas (block-LU forward sweep + back substitution).
//!
//! ## Block structure
//!
//! For the LHS operator `I − τ/2 · L^h` applied to the flattened state
//! `u ∈ ℝ^{N·M}`, the M×M block at (k, k−1), (k, k), (k, k+1) is:
//! - Sub-diagonal:    `−τ/2 · (a/dx² − b/(2dx))`
//! - Main-diagonal:   `I + τ/2 · (2a/dx²)` (note: `L^h_k_diag = −2a/dx²`, negated)
//! - Super-diagonal:  `−τ/2 · (a/dx² + b/(2dx))`
//!
//! Boundary rows (k=0, k=N−1): Neumann ghost-cell (`u_{−1}=u_0`, `u_N=u_{N−1}`).
//!
//! ## Citations
//!
//! - Golub-Van Loan §4.5.1 (block-Thomas algorithm)
//! - Hochbruck-Lubich 2010 Acta Numerica §3.4 (A-stability of Cayley map)
//! - `matrix_inv.rs` (Cramer's-rule M×M inverse for M ∈ {1,2,3,4})

use alloc::vec;
use alloc::vec::Vec;

use crate::{error::SemiflowError, float::SemiflowFloat, matrix_inv::mat_inv_dispatch};

// ---------------------------------------------------------------------------
// Block arithmetic helpers
// ---------------------------------------------------------------------------

/// M×M matrix add: `c = a + b`.
#[inline]
pub(crate) fn block_add<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
    b: &[[F; M]; M],
) -> [[F; M]; M] {
    let mut c = [[F::zero(); M]; M];
    for i in 0..M {
        for j in 0..M {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

/// M×M matrix subtract: `c = a - b`.
#[inline]
pub(crate) fn block_sub<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
    b: &[[F; M]; M],
) -> [[F; M]; M] {
    let mut c = [[F::zero(); M]; M];
    for i in 0..M {
        for j in 0..M {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

/// M×M matrix multiply: `c = a · b`.
#[inline]
pub(crate) fn block_mat_mul<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
    b: &[[F; M]; M],
) -> [[F; M]; M] {
    let mut c = [[F::zero(); M]; M];
    for i in 0..M {
        for k in 0..M {
            for j in 0..M {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// M×M matrix times M-vector: `out = a · v`.
#[inline]
pub(crate) fn block_mat_vec_mul<F: SemiflowFloat, const M: usize>(
    a: &[[F; M]; M],
    v: &[F; M],
) -> [F; M] {
    let mut out = [F::zero(); M];
    for i in 0..M {
        for j in 0..M {
            out[i] += a[i][j] * v[j];
        }
    }
    out
}

/// Scale M×M matrix by scalar: `c = s * a`.
#[inline]
fn block_scale<F: SemiflowFloat, const M: usize>(s: F, a: &[[F; M]; M]) -> [[F; M]; M] {
    let mut c = [[F::zero(); M]; M];
    for i in 0..M {
        for j in 0..M {
            c[i][j] = s * a[i][j];
        }
    }
    c
}

/// Identity M×M matrix.
#[inline]
fn block_identity<F: SemiflowFloat, const M: usize>() -> [[F; M]; M] {
    let mut id = [[F::zero(); M]; M];
    // i indexes both dimensions of the diagonal; range loop is clearest here.
    #[allow(clippy::needless_range_loop)]
    for i in 0..M {
        id[i][i] = F::one();
    }
    id
}

// ---------------------------------------------------------------------------
// Block-Thomas solver
// ---------------------------------------------------------------------------

/// Solve block-tridiagonal system `A · x = rhs` for M-vector unknowns at N points.
///
/// `sub[i]`  = block at row i, column i−1 (i = 1..N-1)
/// `main[i]` = block at row i, column i   (i = 0..N-1)
/// `sup[i]`  = block at row i, column i+1 (i = 0..N-2)
/// `rhs[i]`  = right-hand-side M-vector at row i
/// `x[i]`    = solution M-vector at row i (output)
///
/// Uses block-LU: forward sweep (modify main/rhs in-place), back substitution.
/// Cost: O(N · M³) for block inversions.
pub(crate) fn block_thomas_solve<F: SemiflowFloat, const M: usize>(
    sub: &[[[F; M]; M]],
    main: &[[[F; M]; M]],
    sup: &[[[F; M]; M]],
    rhs: &[[F; M]],
    x: &mut [[F; M]],
    n: usize,
) -> Result<(), SemiflowError> {
    let mut c_prime: Vec<[[F; M]; M]> = vec![[[F::zero(); M]; M]; n];
    let mut d_prime: Vec<[F; M]> = vec![[F::zero(); M]; n];

    // Forward sweep: eliminate sub-diagonals.
    // Row 0: c'[0] = main[0]^{-1} * sup[0], d'[0] = main[0]^{-1} * rhs[0].
    {
        let inv_m0 = mat_inv_dispatch::<F, M>(&main[0])?;
        if n > 1 {
            c_prime[0] = block_mat_mul::<F, M>(&inv_m0, &sup[0]);
        }
        d_prime[0] = block_mat_vec_mul::<F, M>(&inv_m0, &rhs[0]);
    }
    for i in 1..n {
        // w = main[i] - sub[i] * c'[i-1].
        let sc = block_mat_mul::<F, M>(&sub[i - 1], &c_prime[i - 1]);
        let w = block_sub::<F, M>(&main[i], &sc);
        let inv_w = mat_inv_dispatch::<F, M>(&w)?;
        if i < n - 1 {
            c_prime[i] = block_mat_mul::<F, M>(&inv_w, &sup[i]);
        }
        // d'[i] = inv_w * (rhs[i] - sub[i] * d'[i-1])
        let sd = block_mat_vec_mul::<F, M>(&sub[i - 1], &d_prime[i - 1]);
        let mut rhs_i = rhs[i];
        for k in 0..M {
            rhs_i[k] -= sd[k];
        }
        d_prime[i] = block_mat_vec_mul::<F, M>(&inv_w, &rhs_i);
    }

    // Back substitution.
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        let cp_x = block_mat_vec_mul::<F, M>(&c_prime[i], &x[i + 1]);
        for k in 0..M {
            x[i][k] = d_prime[i][k] - cp_x[k];
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// block_cn_diff_step — Phase 2 Crank-Nicolson Cayley-map entry point
// ---------------------------------------------------------------------------

/// Phase 2 block Crank-Nicolson Cayley map for `MatrixDiffusionChernoff`.
///
/// Solves `(I − τ/2 · L^h) · u_out = (I + τ/2 · L^h) · u_in` where `L^h` is
/// the block-tridiagonal discrete diffusion operator with M×M blocks.
///
/// Both LHS and RHS use the same Neumann ghost-cell boundary (mirror):
/// `u_{-1} = u_0`, `u_N = u_{N-1}`. This ensures the Cayley map is
/// consistent and preserves the Chernoff-function identity F(0) = I.
///
/// Boundary block structure (Neumann ghost-cell modified stencil):
/// - k=0:   main = `I + τ/2·(a/dx²+b/(2dx))`;   sup = `−τ/2·(a/dx²+b/(2dx))`
/// - k=N-1: main = `I + τ/2·(a/dx²−b/(2dx))`;   sub = `−τ/2·(a/dx²−b/(2dx))`
///
/// Interior: main = `I + τ·a/dx²`, sub = `−τ/2·(a/dx²−b/(2dx))`,
///           sup = `−τ/2·(a/dx²+b/(2dx))`.
// sub_blk/sup_blk and neg_sub/neg_sup are standard tridiagonal stencil names.
// τ, A, B, u_in, u_out, n, dx — all required for CN block step; struct adds noise.
#[allow(clippy::similar_names, clippy::too_many_arguments)]
pub(crate) fn block_cn_diff_step<F: SemiflowFloat, const M: usize>(
    tau: F,
    a_ij_at_k: &[[[F; M]; M]],
    b_ij_at_k: &[[[F; M]; M]],
    u_in: &[F],
    u_out: &mut [F],
    n: usize,
    dx: F,
) -> Result<(), SemiflowError> {
    let two = F::one() + F::one();
    let half = F::one() / two;
    let dx2 = dx * dx;
    let two_dx = two * dx;
    let half_tau = half * tau;

    let (sub_blk, main_blk, sup_blk) =
        build_cn_stencil::<F, M>(a_ij_at_k, b_ij_at_k, tau, n, dx2, two_dx, half_tau);

    let rhs_vecs =
        assemble_cn_rhs::<F, M>(a_ij_at_k, b_ij_at_k, u_in, n, dx2, two_dx, two, half_tau);

    let mut x_vecs: Vec<[F; M]> = vec![[F::zero(); M]; n];
    block_thomas_solve::<F, M>(&sub_blk, &main_blk, &sup_blk, &rhs_vecs, &mut x_vecs, n)?;
    for k in 0..n {
        for i in 0..M {
            u_out[k * M + i] = x_vecs[k][i];
        }
    }
    Ok(())
}

/// Block-tridiagonal stencil row type alias.
type BlockRow<F, const M: usize> = Vec<[[F; M]; M]>;

/// Build the CN block-tridiagonal stencil (LHS matrices `sub`, `main`, `sup`).
///
/// Boundary rows use Neumann ghost-cell; interior uses standard 3-pt stencil.
#[allow(clippy::similar_names, clippy::too_many_arguments)]
fn build_cn_stencil<F: SemiflowFloat, const M: usize>(
    a_ij_at_k: &[[[F; M]; M]],
    b_ij_at_k: &[[[F; M]; M]],
    tau: F,
    n: usize,
    dx2: F,
    two_dx: F,
    half_tau: F,
) -> (BlockRow<F, M>, BlockRow<F, M>, BlockRow<F, M>) {
    let mut sub_blk: Vec<[[F; M]; M]> = vec![[[F::zero(); M]; M]; n];
    let mut main_blk: Vec<[[F; M]; M]> = vec![[[F::zero(); M]; M]; n];
    let mut sup_blk: Vec<[[F; M]; M]> = vec![[[F::zero(); M]; M]; n];
    for k in 0..n {
        let a = &a_ij_at_k[k];
        let b = &b_ij_at_k[k];
        let id = block_identity::<F, M>();
        let hta_dx2 = block_scale::<F, M>(half_tau / dx2, a);
        let htb_2dx = block_scale::<F, M>(half_tau / two_dx, b);
        if k == 0 {
            let off = block_add::<F, M>(&hta_dx2, &htb_2dx);
            main_blk[0] = block_add::<F, M>(&id, &off);
            sup_blk[0] = block_scale::<F, M>(-F::one(), &off);
        } else if k == n - 1 {
            let off = block_sub::<F, M>(&hta_dx2, &htb_2dx);
            main_blk[n - 1] = block_add::<F, M>(&id, &off);
            sub_blk[n - 2] = block_scale::<F, M>(-F::one(), &off);
        } else {
            main_blk[k] = block_add::<F, M>(&id, &block_scale::<F, M>(tau / dx2, a));
            sub_blk[k - 1] = block_scale::<F, M>(-F::one(), &block_sub::<F, M>(&hta_dx2, &htb_2dx));
            sup_blk[k] = block_scale::<F, M>(-F::one(), &block_add::<F, M>(&hta_dx2, &htb_2dx));
        }
    }
    (sub_blk, main_blk, sup_blk)
}

/// Assemble RHS vector: `rhs_k = (I + τ/2 L^h) u_in` (Neumann ghost-cell).
///
/// For boundary nodes the ghost is the boundary value itself (`u_{-1}=u_0`, `u_N=u_{N-1}`).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn assemble_cn_rhs<F: SemiflowFloat, const M: usize>(
    a_ij_at_k: &[[[F; M]; M]],
    b_ij_at_k: &[[[F; M]; M]],
    u_in: &[F],
    n: usize,
    dx2: F,
    two_dx: F,
    two: F,
    half_tau: F,
) -> Vec<[F; M]> {
    let mut rhs_vecs: Vec<[F; M]> = vec![[F::zero(); M]; n];
    for k in 0..n {
        let a = &a_ij_at_k[k];
        let b = &b_ij_at_k[k];
        let mut lhu = [F::zero(); M];
        for i in 0..M {
            for j in 0..M {
                let uj_prev = if k > 0 {
                    u_in[(k - 1) * M + j]
                } else {
                    u_in[j]
                };
                let uj_curr = u_in[k * M + j];
                let uj_next = if k + 1 < n {
                    u_in[(k + 1) * M + j]
                } else {
                    u_in[(n - 1) * M + j]
                };
                let d2 = (uj_next - two * uj_curr + uj_prev) / dx2;
                let d1 = (uj_next - uj_prev) / two_dx;
                lhu[i] = lhu[i] + a[i][j] * d2 + b[i][j] * d1;
            }
        }
        let mut point = [F::zero(); M];
        for i in 0..M {
            point[i] = u_in[k * M + i] + half_tau * lhu[i];
        }
        rhs_vecs[k] = point;
    }
    rhs_vecs
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Block-Thomas on a trivial 3-point diagonal-only system (M=2).
    /// Main blocks = I (identity), no sub/sup → solution x = rhs.
    #[test]
    fn block_thomas_solves_diagonal_m2() {
        const M: usize = 2;
        const N: usize = 3;
        let id: [[f64; M]; M] = [[1.0, 0.0], [0.0, 1.0]];
        let zero: [[f64; M]; M] = [[0.0; M]; M];
        let sub = vec![zero; N];
        let main = vec![id; N];
        let sup = vec![zero; N];
        let rhs: Vec<[f64; M]> = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut x = vec![[0.0f64; M]; N];
        block_thomas_solve::<f64, M>(&sub, &main, &sup, &rhs, &mut x, N).unwrap();
        for i in 0..N {
            for j in 0..M {
                assert!(
                    (x[i][j] - rhs[i][j]).abs() < 1e-14,
                    "diagonal block-thomas M=2 FAILED at ({i},{j}): got {val} want {want}",
                    val = x[i][j],
                    want = rhs[i][j]
                );
            }
        }
    }

    /// Block-Thomas on a coupled 3-point tridiagonal system (M=2).
    /// Verifies A*x = rhs residual < 1e-13.
    #[test]
    fn block_thomas_solves_coupled_m2() {
        const M: usize = 2;
        const N: usize = 3;
        let main: [[f64; M]; M] = [[2.0, 0.0], [0.0, 2.0]];
        let off: [[f64; M]; M] = [[-0.5, 0.0], [0.0, -0.5]];
        let sub = vec![off; N];
        let main_v = vec![main; N];
        let sup = vec![off; N];
        let rhs: Vec<[f64; M]> = vec![[1.0, 1.0]; N];
        let mut x = vec![[0.0f64; M]; N];
        block_thomas_solve::<f64, M>(&sub, &main_v, &sup, &rhs, &mut x, N).unwrap();
        for i in 0..N {
            for c in 0..M {
                let ax0 = if i > 0 { off[c][c] * x[i - 1][c] } else { 0.0 };
                let ax1 = main[c][c] * x[i][c];
                let ax2 = if i < N - 1 {
                    off[c][c] * x[i + 1][c]
                } else {
                    0.0
                };
                let residual = (ax0 + ax1 + ax2 - rhs[i][c]).abs();
                assert!(
                    residual < 1e-13,
                    "coupled block-thomas FAILED residual {residual:.2e} at ({i},{c})"
                );
            }
        }
    }
}
