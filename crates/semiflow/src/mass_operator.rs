//! Generalized `(M, K)` entry point for lumped and consistent mass matrices (ADR-0186, §55.3–§55.4).
//!
//! - [`TriangularFactor`]: dense Cholesky factor `R` s.t. M = Rᵀ R.
//! - [`MassKOperator`]: symmetric `Â = R⁻ᵀ K R⁻¹`; wraps `(K, R)` for Krylov.
//! - [`mass_lumped_evolve`]: fast path for diagonal mass matrices.
//! - [`dense_massk_expmv_ref`]: dense Padé oracle for gate tests.

use alloc::{vec, vec::Vec};

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    graph_krylov::{graph_expmv_krylov, KrylovPath, MAX_DENSE_N},
    matrix_pade::mat_exp_pade13,
    scratch::ScratchPool,
    symmetric_operator::{SymmetricLinearOp, SymmetricOperator},
};

// ── TriangularFactor ──────────────────────────────────────────────────────────

/// Dense upper-triangular Cholesky factor `R` such that `M = Rᵀ R`.
///
/// Row-major storage: `r[i * n + j]` is `R[i, j]` (valid only for `j ≥ i`).
pub struct TriangularFactor<F: SemiflowFloat> {
    n: usize,
    /// Row-major n×n dense upper-triangular matrix.
    r: Vec<F>,
}

impl<F: SemiflowFloat> TriangularFactor<F> {
    /// Compute Cholesky factor `R` of the dense symmetric positive-definite matrix `m_dense`.
    ///
    /// `m_dense` must be row-major (length `n*n`).
    ///
    /// # Errors
    ///
    /// [`SemiflowError::DomainViolation`] if a diagonal pivot `≤ 0` is encountered.
    pub fn dense_cholesky_spd(m_dense: &[F], n: usize) -> Result<Self, SemiflowError> {
        let mut r = vec![F::zero(); n * n];
        for i in 0..n {
            // Diagonal pivot: R[i,i] = sqrt(M[i,i] - Σ_{k<i} R[k,i]²)
            let mut s = m_dense[i * n + i];
            for k in 0..i {
                s -= r[k * n + i] * r[k * n + i];
            }
            if s <= F::zero() {
                return Err(SemiflowError::DomainViolation {
                    what: "TriangularFactor: mass matrix is not positive-definite (pivot ≤ 0)",
                    value: s.to_f64().unwrap_or(f64::NAN),
                });
            }
            r[i * n + i] = s.sqrt();
            // Off-diagonal: R[i,j] = (M[i,j] - Σ_{k<i} R[k,i]·R[k,j]) / R[i,i]
            for j in i + 1..n {
                let mut sj = m_dense[i * n + j];
                for k in 0..i {
                    sj -= r[k * n + i] * r[k * n + j];
                }
                r[i * n + j] = sj / r[i * n + i];
            }
        }
        Ok(Self { n, r })
    }

    /// Operator dimension.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Back-substitution: solve `R x = b`.
    pub fn solve_r(&self, b: &[F], x: &mut [F]) {
        let n = self.n;
        for i in (0..n).rev() {
            // x[..=i] and x[i+1..] are disjoint — split to satisfy borrow checker.
            let (x_lo, x_hi) = x.split_at_mut(i + 1);
            let r_row = &self.r[i * n + i + 1..i * n + n]; // R[i, i+1..n]
            let mut s = b[i];
            for (&r_ij, &x_j) in r_row.iter().zip(x_hi.iter()) {
                s -= r_ij * x_j;
            }
            x_lo[i] = s / self.r[i * n + i]; // x[i]
        }
    }

    /// Forward-substitution: solve `Rᵀ x = b`.
    pub fn solve_rt(&self, b: &[F], x: &mut [F]) {
        let n = self.n;
        for i in 0..n {
            let mut s = b[i];
            // Rᵀ[i,j] = R[j,i] = self.r[j*n+i]; column access — stride n, not contiguous.
            for (j, &x_j) in x[..i].iter().enumerate() {
                s -= self.r[j * n + i] * x_j;
            }
            x[i] = s / self.r[i * n + i];
        }
    }

    /// Matvec: `out ← R · x`.
    pub fn apply_r(&self, x: &[F], out: &mut [F]) {
        let n = self.n;
        for (i, o) in out.iter_mut().enumerate() {
            // Upper-triangular row i: self.r[i*n+i..i*n+n], paired with x[i..n].
            let r_row = &self.r[i * n + i..i * n + n];
            let x_tail = &x[i..n];
            *o = r_row
                .iter()
                .zip(x_tail.iter())
                .map(|(&r, &v)| r * v)
                .fold(F::zero(), core::ops::Add::add);
        }
    }
}

// ── MassKOperator ─────────────────────────────────────────────────────────────

/// Symmetric operator `Â = R⁻ᵀ K R⁻¹` for consistent-mass generalized problem (§55.4).
///
/// Matvec chain for `Â x`: `x → R⁻¹x → K(R⁻¹x) → R⁻ᵀ(...)`.
pub struct MassKOperator<F: SemiflowFloat> {
    k: SymmetricOperator<F>,
    r: TriangularFactor<F>,
    lambda_max_bound: F,
}

impl<F: SemiflowFloat> MassKOperator<F> {
    /// Build from stiffness operator `k` and Cholesky factor `r` of mass matrix M.
    ///
    /// The spectral bound is RIGOROUS: `λ_max(Â) ≤ ρ̄(K) · ‖R⁻¹‖₁ · ‖R⁻¹‖_∞`, with
    /// `‖R⁻¹‖₂² ≤ ‖R⁻¹‖₁‖R⁻¹‖_∞` and both norms bounded through the comparison matrix
    /// of `R` in `O(n²)` (see [`inverse_norm_bounds`]). The Chebyshev path requires a
    /// true upper bound: the previous 5-step inverse-power estimate of `λ_min(M)`
    /// started from the constant vector — an eigenvector of the consistent P1 mass
    /// matrix with its LARGEST eigenvalue — so it under-estimated `λ_max(Â)` by up to
    /// 3× and the Chebyshev series diverged (relative error `1.7e11` at `n = 12`).
    ///
    /// A bound that overflows is stored as `∞`; [`graph_expmv_krylov`] then returns
    /// `DomainViolation` instead of a silently wrong schedule (issue #44).
    #[must_use]
    pub fn new(k: SymmetricOperator<F>, r: TriangularFactor<F>) -> Self {
        let (inv_r_1, inv_r_inf) = inverse_norm_bounds(&r);
        #[allow(clippy::cast_precision_loss)]
        // n is a matrix dimension; only scales a safety factor
        let rounding =
            F::one() + F::epsilon() * F::from(4.0 * (r.n() as f64 + 2.0)).unwrap_or(F::one());
        let lambda_max_bound = k.lambda_max_bound() * inv_r_1 * inv_r_inf * rounding;
        Self {
            k,
            r,
            lambda_max_bound,
        }
    }

    /// Evolve `out ← e^{−τ M⁻¹K} v` using the Krylov solver on `Â = R⁻ᵀKR⁻¹`.
    ///
    /// # Errors
    ///
    /// Propagates from [`graph_expmv_krylov`].
    #[allow(clippy::too_many_arguments)]
    pub fn evolve(
        &self,
        tau: F,
        v: &[F],
        out: &mut [F],
        path: KrylovPath,
        tol: F,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        graph_expmv_krylov(self, tau, v, out, path, tol, scratch)
    }
}

impl<F: SemiflowFloat> SymmetricLinearOp<F> for MassKOperator<F> {
    fn n(&self) -> usize {
        self.k.n()
    }

    fn lambda_max_bound(&self) -> F {
        self.lambda_max_bound
    }

    /// `out ← Â x = R⁻ᵀ K R⁻¹ x`.
    ///
    /// Allocates two O(n) buffers per call (acceptable for N ≤ 10 gate tests).
    fn apply_into_slice(&self, x: &[F], out: &mut [F]) {
        let n = self.k.n();
        let mut w1 = vec![F::zero(); n]; // w1 = R⁻¹ x
        let mut w2 = vec![F::zero(); n]; // w2 = K w1
        self.r.solve_r(x, &mut w1);
        self.k.apply_into_slice(&w1, &mut w2);
        self.r.solve_rt(&w2, out);
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Rigorous upper bounds `(‖R⁻¹‖₁, ‖R⁻¹‖_∞)` for the upper-triangular factor `R`.
///
/// With the comparison matrix `C(R)` (`|rᵢᵢ|` on the diagonal, `−|rᵢⱼ|` off it),
/// `|R⁻¹| ≤ C(R)⁻¹` entrywise and `C(R)⁻¹ ≥ 0` (Higham, *Accuracy and Stability*,
/// §8.3), so `‖R⁻¹‖_∞ ≤ ‖C(R)⁻¹ e‖_∞` and `‖R⁻¹‖₁ ≤ ‖C(R)⁻ᵀ e‖_∞` for the all-ones
/// `e`. Both are one triangular solve with non-negative data: `O(n²)`, no
/// cancellation. Exact for bidiagonal `R` with positive entries (P1 mass in 1D).
#[allow(clippy::many_single_char_names)] // r, n, a, x, y, s: triangular-solve notation
fn inverse_norm_bounds<F: SemiflowFloat>(r: &TriangularFactor<F>) -> (F, F) {
    let n = r.n();
    let a = |i: usize, j: usize| r.r[i * n + j].abs();
    // ‖C⁻¹ e‖_∞: back substitution  x_i = (1 + Σ_{j>i} |r_ij| x_j) / |r_ii|.
    let mut x = vec![F::zero(); n];
    let mut inf_norm = F::zero();
    for i in (0..n).rev() {
        let s = ((i + 1)..n).fold(F::one(), |acc, j| acc + a(i, j) * x[j]);
        x[i] = s / a(i, i);
        inf_norm = inf_norm.max(x[i]);
    }
    // ‖C⁻ᵀ e‖_∞: forward substitution  y_j = (1 + Σ_{i<j} |r_ij| y_i) / |r_jj|.
    let mut y = vec![F::zero(); n];
    let mut one_norm = F::zero();
    for j in 0..n {
        let s = (0..j).fold(F::one(), |acc, i| acc + a(i, j) * y[i]);
        y[j] = s / a(j, j);
        one_norm = one_norm.max(y[j]);
    }
    (one_norm, inf_norm)
}

// ── Lumped-mass fast path ─────────────────────────────────────────────────────

/// Compute `e^{−τ M⁻¹K} v` for diagonal mass matrix `M = diag(masses)` (§55.3).
///
/// Maps to `D^{−½} e^{−τ Â} (D^{½} v)` where `Â = D^{−½} K D^{−½}`.
///
/// # Errors
///
/// [`SemiflowError::DomainViolation`] if `masses` has wrong length, contains non-positive
/// values, or if the underlying Krylov solve fails.
#[allow(clippy::too_many_arguments)]
pub fn mass_lumped_evolve<F: SemiflowFloat>(
    k: &SymmetricOperator<F>,
    masses: &[F],
    tau: F,
    v: &[F],
    out: &mut [F],
    path: KrylovPath,
    tol: F,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let a_hat = k.lumped_congruence(masses)?;
    let n = a_hat.n();
    // Pre-scale: w0 = D^{½} v
    let mut w0 = vec![F::zero(); n];
    let mut w1 = vec![F::zero(); n];
    for i in 0..n {
        w0[i] = v[i] * masses[i].sqrt();
    }
    // Evolve: w1 = e^{−τÂ} w0
    graph_expmv_krylov(&a_hat, tau, &w0, &mut w1, path, tol, scratch)?;
    // Post-scale: out = D^{−½} w1
    for i in 0..n {
        out[i] = w1[i] / masses[i].sqrt();
    }
    Ok(())
}

// ── Dense oracle (gate tests) ─────────────────────────────────────────────────

/// Dense `mat_exp_pade13` oracle for `MassKOperator` gate tests.
///
/// Builds `Â` column-by-column via `apply_into_slice`, then Padé-13 + matvec.
/// Restricted to `op.n() ≤ MAX_DENSE_N = 12`.
///
/// # Errors
///
/// [`SemiflowError::DomainViolation`] if `op.n() > MAX_DENSE_N`.
pub fn dense_massk_expmv_ref<F: SemiflowFloat>(
    op: &MassKOperator<F>,
    tau: F,
    src: &[F],
    dst: &mut [F],
) -> Result<(), SemiflowError> {
    let n = op.n();
    if n > MAX_DENSE_N {
        return Err(SemiflowError::DomainViolation {
            what: "dense_massk_expmv_ref: n > MAX_DENSE_N (12)",
            #[allow(clippy::cast_precision_loss)]
            value: n as f64,
        });
    }
    // Build -τ Â as MAX_DENSE_N×MAX_DENSE_N matrix (zero-padded).
    let mut a_mat = [[F::zero(); MAX_DENSE_N]; MAX_DENSE_N];
    let mut ej = vec![F::zero(); n];
    let mut col = vec![F::zero(); n];
    for j in 0..n {
        ej[j] = F::one();
        op.apply_into_slice(&ej, &mut col); // writes Â e_j → col (no pre-zero needed)
        for i in 0..n {
            a_mat[i][j] = -tau * col[i];
        }
        ej[j] = F::zero();
    }
    let exp_a = mat_exp_pade13::<F, MAX_DENSE_N>(&a_mat)?;
    for i in 0..n {
        let mut s = F::zero();
        for j in 0..n {
            s += exp_a[i][j] * src[j];
        }
        dst[i] = s;
    }
    Ok(())
}
