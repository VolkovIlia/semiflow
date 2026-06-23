//! TT-core data structures and SVD rounding kernel (`no_std`+alloc, no new deps).
//!
//! # Tensor-Train state
//!
//! A `TtState<F>` is a d-mode tensor stored as a chain of 3-index cores:
//! `u(i₁,…,i_d) = G₁[i₁] · G₂[i₂] · … · G_d[i_d]`
//! where `G_k` is an `r_{k-1} × n_k × r_k` array (row-major, mode index
//! is the middle index).  Bond ranks `r_0 = r_d = 1` (boundary condition).
//!
//! Storage: `O(d · n · r²)` — polynomial in d when `r` is polynomial in d.
//! This is the fundamental curse-escape of the TT format.
//!
//! # TT-rounding
//!
//! `tt_round` applies a left-to-right sweep of truncated SVDs to compress
//! all bonds to tolerance `eps`.  Each bond is compressed by unfolding the
//! left core as `(r_{k-1}·n_k) × r_k`, applying the one-sided Jacobi SVD
//! (deterministic, no LAPACK), truncating to the numerical rank at `eps`,
//! and absorbing the right factor into the next core.
//!
//! Quasi-optimality (Oseledets 2011): the rounding error is ≤ `√(d-1)·eps`
//! in Frobenius norm.
//!
//! # One-sided Jacobi SVD
//!
//! The `jacobi_svd_trunc` function computes a truncated SVD of an `m × n`
//! matrix via the Gram matrix (`nᵀn`, n×n), diagonalised by Jacobi sweeps.
//! Returns `(U_r, S_r, V_r)` with `r` kept columns (those with `σ ≥ eps·σ_max`).
//! Algorithm: Golub & Van Loan §8.3.4 (one-sided, small n per bond).
//!
//! For `n ≤ max_bond_rank ≤ 16` (typical), each SVD is tiny (≤16×16 Gram).
//!
//! # Scope
//! Linear/constant diagonal-A (Gaussian class): rank bounded poly-in-d.
//! Non-Gaussian / off-diagonal-A / variable coefs: rank not capped — research track.

extern crate alloc;
use alloc::vec::Vec;

use crate::float::SemiflowFloat;

// ═══════════════════════════════════════════════════════════════════════════
// §A — TT core geometry
// ═══════════════════════════════════════════════════════════════════════════

/// A single 3-index core of shape `r_left × n × r_right` (row-major).
///
/// Layout: `data[i_left * n * r_right + i_mode * r_right + i_right]`
#[derive(Clone, Debug)]
pub struct TtCore<F: SemiflowFloat> {
    /// Number of columns from the left bond.
    pub r_left: usize,
    /// Mode size (number of grid points on this axis).
    pub n: usize,
    /// Number of rows for the right bond.
    pub r_right: usize,
    /// Raw data, row-major: index `(il, im, ir)` → `il*n*r_right + im*r_right + ir`.
    pub data: Vec<F>,
}

impl<F: SemiflowFloat> TtCore<F> {
    /// Allocate a zero core of shape `r_left × n × r_right`.
    #[must_use]
    pub fn zeros(r_left: usize, n: usize, r_right: usize) -> Self {
        Self {
            r_left,
            n,
            r_right,
            data: vec![F::zero(); r_left * n * r_right],
        }
    }

    /// Element access `(il, im, ir)`.
    #[must_use]
    #[inline]
    pub fn get(&self, il: usize, im: usize, ir: usize) -> F {
        self.data[il * self.n * self.r_right + im * self.r_right + ir]
    }

    /// Element set `(il, im, ir)`.
    #[inline]
    pub fn set(&mut self, il: usize, im: usize, ir: usize, v: F) {
        let idx = il * self.n * self.r_right + im * self.r_right + ir;
        self.data[idx] = v;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §B — One-sided Jacobi SVD (no LAPACK, deterministic, ~130 LoC)
// ═══════════════════════════════════════════════════════════════════════════

/// Truncated SVD of `A` (`m × n`, row-major) via one-sided Jacobi on `AᵀA`.
///
/// Returns `(U, s, V)` where `U` is `m × r`, `s` is length `r`,
/// `V` is `n × r`, columns sorted descending; `r` = number of singular values
/// `σ ≥ eps * σ_max` (at least 1 if any nonzero).
///
/// `U = A · V · diag(1/s)`.  For tiny matrices (bond rank ≤ 16): fast, exact.
pub(crate) fn jacobi_svd_trunc<F: SemiflowFloat>(
    a: &[F],
    m: usize,
    n: usize,
    eps: F,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    if m == 0 || n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let (sv, v_mat) = jacobi_gram_ev(a, m, n);
    if sv.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let sigma_max = sv[0];
    let thr = if sigma_max > F::zero() {
        eps * sigma_max
    } else {
        F::zero()
    };
    let r = sv.iter().filter(|&&s| s > thr).count().max(1).min(n);

    // V_r: first r columns (n×r, column-major means row i, col j = v_mat[i*n + j])
    // Build v_r: rows of v_mat, first r columns only (n×r)
    let mut v_r: Vec<F> = Vec::with_capacity(n * r);
    for i in 0..n {
        for j in 0..r {
            v_r.push(v_mat[i * n + j]);
        }
    }

    // s_r: first r singular values
    let s_r: Vec<F> = sv[..r].to_vec();

    // U = A · V_r · diag(1/s_r)  [m × r]
    let u_r = compute_u(a, m, n, &v_r, &s_r, r);

    (u_r, s_r, v_r)
}

/// One-sided Jacobi: Gram matrix `G = AᵀA` (`n×n`), then Jacobi diagonalisation.
/// Returns (singular values descending, eigenvector matrix V n×n row-major).
#[allow(clippy::many_single_char_names)]
fn jacobi_gram_ev<F: SemiflowFloat>(a: &[F], m: usize, n: usize) -> (Vec<F>, Vec<F>) {
    let mut g = vec![F::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = F::zero();
            for k in 0..m {
                s += a[k * n + i] * a[k * n + j];
            }
            g[i * n + j] = s;
        }
    }
    let mut vmat = eye_f::<F>(n);
    // Jacobi sweeps until convergence
    for _sweep in 0..50 {
        let mut off_sq = F::zero();
        for i in 0..n {
            for j in (i + 1)..n {
                off_sq += two::<F>() * g[i * n + j] * g[i * n + j];
            }
        }
        if off_sq < small_f::<F>() {
            break;
        }
        jacobi_sweep(&mut g, &mut vmat, n);
    }
    // Extract eigenvalues as singular values σ = sqrt(λ_i)
    let mut sv: Vec<F> = (0..n)
        .map(|i| (g[i * n + i]).max(F::zero()).sqrt())
        .collect();
    // Sort descending and reorder V columns
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        sv[b]
            .partial_cmp(&sv[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let sv_sorted: Vec<F> = idx.iter().map(|&i| sv[i]).collect();
    let v_old = vmat.clone();
    for (new_col, &old_col) in idx.iter().enumerate() {
        for r in 0..n {
            vmat[r * n + new_col] = v_old[r * n + old_col];
        }
    }
    sv = sv_sorted;
    (sv, vmat)
}

#[allow(clippy::many_single_char_names)]
fn jacobi_sweep<F: SemiflowFloat>(g: &mut [F], v: &mut [F], n: usize) {
    for p in 0..n {
        for q in (p + 1)..n {
            let gpq = g[p * n + q];
            if gpq * gpq < small_f::<F>() {
                continue;
            }
            let gpp = g[p * n + p];
            let gqq = g[q * n + q];
            let tau = (gqq - gpp) / (two::<F>() * gpq);
            let t = if tau >= F::zero() {
                F::one() / (tau + (F::one() + tau * tau).sqrt())
            } else {
                F::one() / (tau - (F::one() + tau * tau).sqrt())
            };
            let c = F::one() / (F::one() + t * t).sqrt();
            let s = t * c;
            apply_jacobi_rotation(g, v, n, p, q, c, s);
        }
    }
}

#[allow(clippy::many_single_char_names, clippy::too_many_arguments)]
fn apply_jacobi_rotation<F: SemiflowFloat>(
    g: &mut [F],
    v: &mut [F],
    n: usize,
    p: usize,
    q: usize,
    c: F,
    s: F,
) {
    for r in 0..n {
        let grp = g[r * n + p];
        let grq = g[r * n + q];
        g[r * n + p] = c * grp - s * grq;
        g[r * n + q] = s * grp + c * grq;
    }
    for r in 0..n {
        let gpr = g[p * n + r];
        let gqr = g[q * n + r];
        g[p * n + r] = c * gpr - s * gqr;
        g[q * n + r] = s * gpr + c * gqr;
    }
    g[p * n + q] = F::zero();
    g[q * n + p] = F::zero();
    for r in 0..n {
        let vrp = v[r * n + p];
        let vrq = v[r * n + q];
        v[r * n + p] = c * vrp - s * vrq;
        v[r * n + q] = s * vrp + c * vrq;
    }
}

/// Compute `U = A · V_r · diag(1/s_r)` [`m × r`].
#[allow(clippy::many_single_char_names)]
fn compute_u<F: SemiflowFloat>(
    a: &[F],
    m: usize,
    n: usize,
    v_r: &[F],
    s_r: &[F],
    r: usize,
) -> Vec<F> {
    let mut u = vec![F::zero(); m * r];
    for i in 0..m {
        for j in 0..r {
            let mut dot = F::zero();
            for k in 0..n {
                dot += a[i * n + k] * v_r[k * r + j];
            }
            let sig = s_r[j];
            u[i * r + j] = if sig > small_f::<F>() {
                dot / sig
            } else {
                F::zero()
            };
        }
    }
    u
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — TT rounding
// ═══════════════════════════════════════════════════════════════════════════

/// Round all bonds in `cores` to tolerance `eps` (left-to-right sweep).
///
/// After rounding: `||u_original - u_rounded||_F ≤ √(d-1)·eps·||u_original||_F`
/// (Oseledets 2011 quasi-optimality).
///
/// Algorithm: for k = 0..d-2:
///   1. Unfold `G_k` to matrix `M` of shape `(r_{k-1}·n_k) × r_k`.
///   2. Compute truncated SVD: `M ≈ U_r · diag(s_r) · V_rᵀ`, rank r.
///   3. Replace `G_k` with `U_r` reshaped to `r_{k-1} × n_k × r`.
///   4. Absorb `diag(s_r)·V_rᵀ` into `G_{k+1}` from the left.
pub fn tt_round<F: SemiflowFloat>(cores: &mut [TtCore<F>], eps: F) {
    let d = cores.len();
    if d <= 1 {
        return;
    }
    for k in 0..(d - 1) {
        let rl = cores[k].r_left;
        let n = cores[k].n;
        let rr = cores[k].r_right;
        // M = (rl*n) × rr
        let m_rows = rl * n;
        let (u_r, s_r, v_r) = jacobi_svd_trunc::<F>(&cores[k].data, m_rows, rr, eps);
        let r_new = s_r.len();
        if r_new == 0 {
            continue;
        }
        // Update core k: shape rl × n × r_new
        let mut new_core_k = TtCore::zeros(rl, n, r_new);
        new_core_k.data.copy_from_slice(&u_r);
        cores[k] = new_core_k;
        // B = diag(s_r) · V_rᵀ: shape r_new × rr
        // Then absorb B into core k+1 from the left.
        let b = diag_times_vt(&s_r, &v_r, r_new, rr);
        absorb_left(&b, &mut cores[k + 1], r_new);
    }
}

/// Compute `diag(s) · Vᵀ` where `V` is `rr × r_new` (`V_r` stored col-major: `V[i*r+j] = v[i,j]`).
/// Returns matrix `r_new × rr`.
fn diag_times_vt<F: SemiflowFloat>(s: &[F], v_r: &[F], r_new: usize, rr: usize) -> Vec<F> {
    let mut b = vec![F::zero(); r_new * rr];
    for i in 0..r_new {
        for j in 0..rr {
            // V stored: v_r[j * r_new + i] = V[j, i] (n×r where n=rr here)
            b[i * rr + j] = s[i] * v_r[j * r_new + i];
        }
    }
    b
}

/// Absorb `b` (`r_new × old_r_left`) into `core` from the left.
/// Core shape changes: `(old_r_left × n × r_right)` → `(r_new × n × r_right)`.
fn absorb_left<F: SemiflowFloat>(b: &[F], core: &mut TtCore<F>, r_new: usize) {
    let old_rl = core.r_left;
    let n = core.n;
    let rr = core.r_right;
    let mut new_data = vec![F::zero(); r_new * n * rr];
    for il in 0..r_new {
        for im in 0..n {
            for ir in 0..rr {
                let mut val = F::zero();
                for k in 0..old_rl {
                    val += b[il * old_rl + k] * core.get(k, im, ir);
                }
                new_data[il * n * rr + im * rr + ir] = val;
            }
        }
    }
    core.r_left = r_new;
    core.data = new_data;
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — Small numeric helpers
// ═══════════════════════════════════════════════════════════════════════════

fn eye_f<F: SemiflowFloat>(n: usize) -> Vec<F> {
    let mut m = vec![F::zero(); n * n];
    for i in 0..n {
        m[i * n + i] = F::one();
    }
    m
}

#[inline]
fn two<F: SemiflowFloat>() -> F {
    F::from(2.0).unwrap()
}
#[inline]
fn small_f<F: SemiflowFloat>() -> F {
    F::from(1e-30).unwrap()
}
