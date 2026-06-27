//! φ-function actions via the augmented-matrix construction (ADR-0189 §58.2).
//!
//! ## Summary
//!
//! `φ_k(z) = Σ_{n≥0} zⁿ / (n+k)!` with `φ_0(z) = eᶻ`.
//!
//! [`phi_action`] computes `φ_k(τA)·v` for a single `k ≤ PHI_MAX`.
//! [`phi_action_batched`] computes all `φ_0…φ_p` in a single call
//! (each at the same `τ` and `v`).
//!
//! ## Algorithm
//!
//! Build the `(n+3) × (n+3)` augmented operator
//! `Ã = [[τA, v·e₁ᵀ], [0, J₃]]`.
//! A Horner sweep on `(1/s)·Ã` with `(s,m)` from
//! Al-Mohy–Higham Algorithm 3.2 yields `exp(Ã)·z_init`; the top-n rows
//! equal `φ_k(τA)·v` for the appropriate initial vector.
//!
//! ## References
//! - ADR-0189; math.md §58 (NORMATIVE).
//! - Al-Mohy & Higham (2011) SIAM J. Sci. Comput. 33:488–511.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::{
    error::SemiflowError,
    expmv::select_s_m,
    float::SemiflowFloat,
    generator_action::GeneratorAction,
    graph_krylov::MAX_DENSE_N,
    matrix_pade::mat_exp_pade13,
    phi_action_helpers::aug_horner_outer,
    scratch::ScratchPool,
};

/// Maximum supported φ index.
pub const PHI_MAX: usize = 3;

/// Norm-tightening factor for `select_s_m` in the φ-action.
///
/// `THETA_M` is calibrated for the exponential BACKWARD error; the φ-extraction
/// from the augmented block requires FORWARD accuracy.  Multiplying the norm
/// estimate by this factor nudges `select_s_m` to choose a higher Taylor degree
/// (m=18 instead of m=13 at the canonical z≈2 test point), reducing the per-substep
/// truncation from ~9e-9 to ~8e-14 without adding extra squarings.
const PHI_NORM_TIGHTEN: f64 = 2.0;

// ---------------------------------------------------------------------------
// norm helpers
// ---------------------------------------------------------------------------

/// `‖Ã‖` bound: `τ·‖A‖ + ‖v‖_∞ + 1`.
fn aug_norm_bound<F: SemiflowFloat>(op_norm: f64, tau: F, v: &[F]) -> f64 {
    let tau_f64 = tau.to_f64().unwrap_or(0.0);
    let v_inf = v.iter()
        .map(|x| x.abs().to_f64().unwrap_or(0.0))
        .fold(0.0_f64, f64::max);
    tau_f64 * op_norm + v_inf + 1.0
}

// ---------------------------------------------------------------------------
// phi_action_batched
// ---------------------------------------------------------------------------

/// Compute `φ_k(τA)·v` for all `k = 0 … p` simultaneously.
///
/// # Arguments
/// - `op`: linear generator providing `A`-matvec.
/// - `p`: max φ index (must be `≤ PHI_MAX = 3`).
/// - `tau`: time step.
/// - `v`: input vector, length `op.dim()`.
/// - `out`: output buffer, length `(p+1) * op.dim()`.
///   Slice `out[k*n .. (k+1)*n]` receives `φ_k(τA)·v`.
/// - `scratch`: reusable allocation pool.
///
/// # Errors
/// Returns `DomainViolation` if `p > PHI_MAX`.
#[allow(clippy::many_single_char_names)]
pub fn phi_action_batched<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    p: usize,
    tau: F,
    v: &[F],
    out: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if p > PHI_MAX {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "phi_action_batched: p > PHI_MAX (3)",
            value: p as f64,
        });
    }
    let n = op.dim();
    let dim_aug = n + PHI_MAX; // always n+3; uniform across all k

    // (s,m) from augmented-operator norm (one computation, shared).
    // PHI_NORM_TIGHTEN promotes to a higher Taylor degree for forward accuracy.
    let norm_aug = aug_norm_bound(op.norm_bound().to_f64().unwrap_or(0.0), tau, v);
    let (s, m) = select_s_m(norm_aug * PHI_NORM_TIGHTEN, 1.0);

    // Scratch: y (n+3), w (n+3), av (n)
    let mut y_aug = scratch.take_vec(dim_aug);
    let mut w_aug = scratch.take_vec(dim_aug);
    let mut av_buf = scratch.take_vec(n);

    for k in 0..=p {
        init_aug_vector(v, n, PHI_MAX, k, &mut y_aug);

        for _ in 0..s {
            aug_horner_outer(op, v, tau, s, m, &mut y_aug, &mut w_aug, &mut av_buf)?;
        }

        // Extract top-n into out[k*n .. (k+1)*n]
        out[k * n..(k + 1) * n].copy_from_slice(&y_aug[..n]);
    }

    scratch.return_vec(av_buf);
    scratch.return_vec(w_aug);
    scratch.return_vec(y_aug);
    Ok(())
}

// ---------------------------------------------------------------------------
// phi_action (single k)
// ---------------------------------------------------------------------------

/// Compute `φ_k(τA)·v` for a single index `k ≤ PHI_MAX`.
///
/// Equivalent to `phi_action_batched` restricted to one k.
///
/// # Errors
/// Returns `DomainViolation` if `k > PHI_MAX`.
#[allow(clippy::many_single_char_names)]
pub fn phi_action<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    k: usize,
    tau: F,
    v: &[F],
    out: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if k > PHI_MAX {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "phi_action: k > PHI_MAX (3)",
            value: k as f64,
        });
    }
    let n = op.dim();
    let dim_aug = n + PHI_MAX;

    // PHI_NORM_TIGHTEN promotes to a higher Taylor degree for forward accuracy.
    let norm_aug = aug_norm_bound(op.norm_bound().to_f64().unwrap_or(0.0), tau, v);
    let (s, m) = select_s_m(norm_aug * PHI_NORM_TIGHTEN, 1.0);

    let mut y_aug = scratch.take_vec(dim_aug);
    let mut w_aug = scratch.take_vec(dim_aug);
    let mut av_buf = scratch.take_vec(n);

    init_aug_vector(v, n, PHI_MAX, k, &mut y_aug);

    for _ in 0..s {
        aug_horner_outer(op, v, tau, s, m, &mut y_aug, &mut w_aug, &mut av_buf)?;
    }

    out[..n].copy_from_slice(&y_aug[..n]);

    scratch.return_vec(av_buf);
    scratch.return_vec(w_aug);
    scratch.return_vec(y_aug);
    Ok(())
}

// ---------------------------------------------------------------------------
// Initial-vector setup
// ---------------------------------------------------------------------------

/// Fill `y_aug` (size `n+p`) with the initial augmented vector for `φ_k`.
///
/// - `k = 0`: `[v; 0…0]` → `exp(Ã)·y_aug` top-n = `φ_0(τA)·v = e^{τA}·v`.
/// - `k ≥ 1`: `[0…0; e_{k-1}]` → top-n = `φ_k(τA)·v`.
fn init_aug_vector<F: SemiflowFloat>(v: &[F], n: usize, p: usize, k: usize, y_aug: &mut [F]) {
    // Zero everything first
    for yi in y_aug.iter_mut() {
        *yi = F::zero();
    }
    if k == 0 {
        // [v; 0 … 0]
        y_aug[..n].copy_from_slice(&v[..n]);
    } else {
        // [0; e_{k-1} in R^p] — 1 at position n + (k-1)
        let idx = n + k - 1;
        if idx < n + p {
            y_aug[idx] = F::one();
        }
    }
}

// ---------------------------------------------------------------------------
// Dense Padé-13 oracle (gate tests only)
// ---------------------------------------------------------------------------

/// Fill the `MAX_DENSE_N×MAX_DENSE_N` matrix for the augmented φ-oracle.
///
/// Materialises τA column-by-column (top-left n×n block), sets V=v in column n,
/// and sets the J₃ superdiagonal in rows `n..n+PHI_MAX-1`.
fn fill_aug_dense_mat(
    gen: &dyn GeneratorAction<f64>,
    tau: f64,
    v: &[f64],
    n: usize,
) -> [[f64; MAX_DENSE_N]; MAX_DENSE_N] {
    let mut mat = [[0.0_f64; MAX_DENSE_N]; MAX_DENSE_N];
    let mut e_j = vec![0.0_f64; n];
    let mut col_j = vec![0.0_f64; n];
    for j in 0..n {
        e_j[j] = 1.0;
        gen.apply_generator(&e_j, &mut col_j);
        for i in 0..n { mat[i][j] = tau * col_j[i]; }
        e_j[j] = 0.0;
    }
    for i in 0..n { mat[i][n] = v[i]; }
    for i in n..n + PHI_MAX - 1 { mat[i][i + 1] = 1.0; }
    mat
}

/// Dense Padé-13 oracle for the `G_PHI_AUG_DENSE` gate test.
///
/// Builds the `(n+PHI_MAX)×(n+PHI_MAX)` augmented matrix
/// `Ã = [[τA, v·e₁ᵀ], [0, J₃]]` (zero-padded to `MAX_DENSE_N = 12`),
/// exponentiates via `mat_exp_pade13`, and extracts `φ_k(τA)·v`
/// for `k = 0 … PHI_MAX`.
///
/// # Errors
/// `DomainViolation` if `n + PHI_MAX > MAX_DENSE_N = 12`.
pub fn dense_phi_aug_ref(
    gen: &dyn GeneratorAction<f64>,
    tau: f64,
    v: &[f64],
) -> Result<Vec<Vec<f64>>, SemiflowError> {
    let n = gen.dim();
    let dim_aug = n + PHI_MAX;
    if dim_aug > MAX_DENSE_N {
        return Err(SemiflowError::DomainViolation {
            what: "dense_phi_aug_ref: n + PHI_MAX > MAX_DENSE_N (12)",
            #[allow(clippy::cast_precision_loss)]
            value: dim_aug as f64,
        });
    }
    let mat = fill_aug_dense_mat(gen, tau, v, n);
    let exp_mat = mat_exp_pade13::<f64, MAX_DENSE_N>(&mat)?;
    let mut out = Vec::with_capacity(PHI_MAX + 1);
    // φ_0: exp_mat[0:n, 0:n] · v.
    out.push((0..n).map(|i| (0..n).map(|j| exp_mat[i][j] * v[j]).sum()).collect());
    // φ_k (k = 1 … PHI_MAX): column n+(k−1) of exp_mat, top n rows.
    for k in 1..=PHI_MAX { out.push((0..n).map(|i| exp_mat[i][n + k - 1]).collect()); }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Unit tests (compile-time only; slow gate is in tests/g_phi_aug_dense.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
include!("phi_action_tests_mod.rs");
