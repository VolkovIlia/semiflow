//! Generic symmetric positive-semidefinite operator entry point (ADR-0186, §55).
//!
//! Provides [`SymmetricLinearOp<F>`] — minimal trait used by Krylov helpers — and
//! [`SymmetricOperator<F>`] — a validated externally-assembled sparse operator.

use alloc::{sync::Arc, vec::Vec};

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    graph::{Laplacian, LaplacianKind},
    graph_krylov::{GraphKrylovChernoff, KrylovPath, MAX_DENSE_N},
    matrix_pade::mat_exp_pade13,
    spd_resolvent::{SpdResolvent, SpdSolver},
};

// ── SymmetricLinearOp ─────────────────────────────────────────────────────────

/// Minimal abstraction over a symmetric linear operator for Krylov helpers (ADR-0186 §55.1).
///
/// Implementors must ensure `n()` is consistent with `apply_into_slice` lengths.
pub trait SymmetricLinearOp<F: SemiflowFloat>: Send + Sync {
    /// Operator dimension.
    fn n(&self) -> usize;

    /// Gershgorin upper bound on the spectral radius `ρ̄ ≥ ρ(A)`.
    fn lambda_max_bound(&self) -> F;

    /// `dst ← A · src`.  Both slices must have length `self.n()`.
    fn apply_into_slice(&self, src: &[F], dst: &mut [F]);
}

impl<F: SemiflowFloat> SymmetricLinearOp<F> for Laplacian<F> {
    fn n(&self) -> usize {
        self.n_nodes()
    }

    fn lambda_max_bound(&self) -> F {
        self.spectral_radius_bound()
    }

    fn apply_into_slice(&self, src: &[F], dst: &mut [F]) {
        // Inherent `Laplacian::apply_into_slice` (same name — use UFCS to be unambiguous).
        Laplacian::apply_into_slice(self, src, dst);
    }
}

// ── SymmetricOperator ─────────────────────────────────────────────────────────

/// Externally-assembled symmetric positive-semidefinite sparse operator (ADR-0186 D1).
///
/// [`from_csr`](Self::from_csr) validates finiteness, diagonal ≥ 0, and symmetry.
/// Internal layout reuses [`Laplacian`]'s CSR storage and Gershgorin bound.
#[derive(Clone)]
pub struct SymmetricOperator<F: SemiflowFloat = f64> {
    pub(crate) inner: Arc<Laplacian<F>>,
}

impl<F: SemiflowFloat> SymmetricOperator<F> {
    /// Build from raw CSR slices (copies once for internal storage).
    ///
    /// # Errors
    ///
    /// [`SemiflowError::DomainViolation`] if finiteness, diagonal-≥0, or symmetry check fails.
    pub fn from_csr(
        n: usize,
        row_ptr: &[usize],
        col_idx: &[u32],
        vals: &[F],
        sym_tol: F,
    ) -> Result<Self, SemiflowError> {
        check_finite(vals)?;
        // CSR shape + Gershgorin bound; owned vecs passed to Laplacian.
        let inner = Laplacian::from_csr_parts(
            n,
            row_ptr.to_vec(),
            col_idx.to_vec(),
            vals.to_vec(),
            LaplacianKind::GeneralSymmetric,
        )?;
        check_diag_nonneg(n, row_ptr, col_idx, vals)?;
        check_sym_matrix(n, row_ptr, col_idx, vals, sym_tol)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Operator dimension.
    #[must_use]
    pub fn n(&self) -> usize {
        self.inner.n_nodes()
    }

    /// Gershgorin spectral-radius upper bound.
    #[must_use]
    pub fn lambda_max_bound(&self) -> F {
        self.inner.spectral_radius_bound()
    }

    /// Borrowed CSR view `(row_ptr, col_idx, vals)`; zero copies.
    ///
    /// Round trip: `from_csr(n, csr().0, csr().1, csr().2, 0)` reproduces identical arrays.
    #[must_use]
    pub fn csr(&self) -> (&[usize], &[u32], &[F]) {
        (
            self.inner.row_ptr(),
            self.inner.col_idx(),
            self.inner.vals(),
        )
    }

    /// `A + diag(c)` (ADR-0202 D2, §62.3). Inserts structurally missing diagonal
    /// entries; columns stay sorted.
    ///
    /// # Errors
    ///
    /// [`SemiflowError::DomainViolation`] if `c.len() != n`, any `cᵢ < 0` / non-finite, or an
    /// assembled entry overflows to `±∞`.
    pub fn with_diagonal(&self, c: &[F]) -> Result<Self, SemiflowError> {
        let n = self.n();
        if c.len() != n {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "with_diagonal: c.len() != n",
                value: c.len() as f64,
            });
        }
        if let Some(&bad) = c.iter().find(|&&v| !v.is_finite() || v < F::zero()) {
            return Err(SemiflowError::DomainViolation {
                what: "with_diagonal: c entries must be finite and >= 0",
                value: bad.to_f64().unwrap_or(f64::NAN),
            });
        }
        let (rp, ci, va) = self.csr();
        let (nrp, nci, nva) = add_diagonal(n, rp, ci, va, c);
        if let Some(&bad) = nva.iter().find(|v| !v.is_finite()) {
            return Err(SemiflowError::DomainViolation {
                what: "with_diagonal: A + diag(c) overflows (non-finite assembled entry)",
                value: bad.to_f64().unwrap_or(f64::NAN),
            });
        }
        let inner = Laplacian::from_csr_parts(n, nrp, nci, nva, LaplacianKind::GeneralSymmetric)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Factor-once / solve-many resolvent `(λ·diag(mass) + A)⁻¹` (ADR-0202 D1, §62).
    ///
    /// # Errors
    ///
    /// See [`SpdResolvent::new`].
    pub fn resolvent(
        &self,
        lambda: F,
        mass: Option<&[F]>,
        solver: SpdSolver,
        tol: F,
    ) -> Result<SpdResolvent<F>, SemiflowError> {
        SpdResolvent::new(self, lambda, mass, solver, tol)
    }

    /// Build a [`GraphKrylovChernoff`] solver backed by this operator's CSR matrix.
    ///
    /// # Errors
    ///
    /// Propagates from [`GraphKrylovChernoff::new`].
    pub fn krylov(
        &self,
        path: KrylovPath,
        tol: F,
    ) -> Result<GraphKrylovChernoff<F>, SemiflowError> {
        GraphKrylovChernoff::new(Arc::clone(&self.inner), path, tol)
    }

    /// Scale each CSR entry `A[i,j]` by `1 / √(m_i · m_j)` to form
    /// `Â = D^{−½} A D^{−½}` (§55.3, lumped-mass congruence).
    ///
    /// # Errors
    ///
    /// [`SemiflowError::DomainViolation`] if `masses.len() != n` or any mass is `≤ 0`.
    ///
    /// # Panics
    ///
    /// Panics if the derived CSR is structurally invalid (impossible: source was validated).
    pub fn lumped_congruence(&self, masses: &[F]) -> Result<Self, SemiflowError> {
        let n = self.inner.n_nodes();
        if masses.len() != n {
            return Err(SemiflowError::DomainViolation {
                what: "lumped_congruence: masses.len() != n",
                #[allow(clippy::cast_precision_loss)]
                value: masses.len() as f64,
            });
        }
        for &m in masses {
            if !m.is_finite() || m <= F::zero() {
                return Err(SemiflowError::DomainViolation {
                    what: "lumped_congruence: non-positive or non-finite mass",
                    value: m.to_f64().unwrap_or(f64::NAN),
                });
            }
        }
        let sqrt_m: Vec<F> = masses.iter().map(|&m| m.sqrt()).collect();
        let rp = self.inner.row_ptr().to_vec();
        let ci = self.inner.col_idx().to_vec();
        let ov = self.inner.vals();
        let mut nv = Vec::with_capacity(ov.len());
        for i in 0..n {
            for k in rp[i]..rp[i + 1] {
                let j = ci[k] as usize;
                nv.push(ov[k] / (sqrt_m[i] * sqrt_m[j]));
            }
        }
        let inner = Laplacian::from_csr_parts(n, rp, ci, nv, LaplacianKind::GeneralSymmetric)
            .expect("lumped_congruence: derived CSR is always valid (validated source)");
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

impl<F: SemiflowFloat> SymmetricLinearOp<F> for SymmetricOperator<F> {
    fn n(&self) -> usize {
        self.inner.n_nodes()
    }

    fn lambda_max_bound(&self) -> F {
        self.inner.spectral_radius_bound()
    }

    fn apply_into_slice(&self, src: &[F], dst: &mut [F]) {
        Laplacian::apply_into_slice(&self.inner, src, dst);
    }
}

/// CSR of `A + diag(c)`: copies each row, adding `c[i]` at (or inserting it as) `(i, i)`.
fn add_diagonal<F: SemiflowFloat>(
    n: usize,
    rp: &[usize],
    ci: &[u32],
    va: &[F],
    c: &[F],
) -> (Vec<usize>, Vec<u32>, Vec<F>) {
    let mut nrp = Vec::with_capacity(n + 1);
    let mut nci = Vec::with_capacity(ci.len() + n);
    let mut nva = Vec::with_capacity(ci.len() + n);
    nrp.push(0_usize);
    for i in 0..n {
        #[allow(clippy::cast_possible_truncation)]
        let diag_col = i as u32;
        let mut placed = false;
        for k in rp[i]..rp[i + 1] {
            if !placed && ci[k] >= diag_col {
                placed = true;
                nci.push(diag_col);
                if ci[k] == diag_col {
                    nva.push(va[k] + c[i]);
                    continue;
                }
                nva.push(c[i]);
            }
            nci.push(ci[k]);
            nva.push(va[k]);
        }
        if !placed {
            nci.push(diag_col);
            nva.push(c[i]);
        }
        nrp.push(nci.len());
    }
    (nrp, nci, nva)
}

// ── Validation helpers ─────────────────────────────────────────────────────────

fn check_finite<F: SemiflowFloat>(vals: &[F]) -> Result<(), SemiflowError> {
    for &v in vals {
        if !v.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "SymmetricOperator::from_csr: non-finite entry",
                value: v.to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}

fn check_diag_nonneg<F: SemiflowFloat>(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[F],
) -> Result<(), SemiflowError> {
    #[allow(clippy::cast_possible_truncation)]
    for i in 0..n {
        for k in row_ptr[i]..row_ptr[i + 1] {
            if col_idx[k] == i as u32 && vals[k] < F::zero() {
                return Err(SemiflowError::DomainViolation {
                    what: "SymmetricOperator::from_csr: diagonal entry < 0",
                    value: vals[k].to_f64().unwrap_or(f64::NAN),
                });
            }
        }
    }
    Ok(())
}

/// Per-row binary search in sorted CSR (invariant I3) to verify `A[j,i]` exists for `A[i,j]`.
fn check_sym_matrix<F: SemiflowFloat>(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[F],
    tol: F,
) -> Result<(), SemiflowError> {
    #[allow(clippy::cast_possible_truncation)]
    for i in 0..n {
        for k in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[k] as usize;
            if j == i {
                continue;
            }
            let row_j = &col_idx[row_ptr[j]..row_ptr[j + 1]];
            let pos = row_j.partition_point(|&c| c < i as u32);
            if pos >= row_j.len() || row_j[pos] != i as u32 {
                return Err(SemiflowError::DomainViolation {
                    what: "SymmetricOperator::from_csr: no (j,i) entry for existing (i,j)",
                    #[allow(clippy::cast_precision_loss)]
                    value: i as f64,
                });
            }
            let ij = vals[k];
            let ji = vals[row_ptr[j] + pos];
            let scale = ij.abs().max(ji.abs());
            if scale > F::zero() && (ij - ji).abs() > tol * scale {
                return Err(SemiflowError::DomainViolation {
                    what: "SymmetricOperator::from_csr: asymmetric entry exceeds sym_tol",
                    value: (ij - ji).abs().to_f64().unwrap_or(f64::NAN),
                });
            }
        }
    }
    Ok(())
}

// ── Dense oracle (gate tests) ──────────────────────────────────────────────────

/// Dense `mat_exp_pade13` oracle for [`SymmetricOperator`] gate tests.
///
/// Builds the dense matrix `−τ·A` from raw CSR, applies Padé-13, and writes
/// `e^{−τA} · src` into `dst`.  Restricted to `op.n() ≤ MAX_DENSE_N = 12`.
///
/// # Errors
///
/// [`SemiflowError::DomainViolation`] if `op.n() > MAX_DENSE_N`.
pub fn dense_csr_expmv_ref<F: SemiflowFloat>(
    op: &SymmetricOperator<F>,
    tau: F,
    src: &[F],
    dst: &mut [F],
) -> Result<(), SemiflowError> {
    let n = op.n();
    if n > MAX_DENSE_N {
        return Err(SemiflowError::DomainViolation {
            what: "dense_csr_expmv_ref: n > MAX_DENSE_N (12)",
            #[allow(clippy::cast_precision_loss)]
            value: n as f64,
        });
    }
    let mut a_mat = [[F::zero(); MAX_DENSE_N]; MAX_DENSE_N];
    let rp = op.inner.row_ptr();
    let ci = op.inner.col_idx();
    let vs = op.inner.vals();
    for i in 0..n {
        for k in rp[i]..rp[i + 1] {
            a_mat[i][ci[k] as usize] = -tau * vs[k];
        }
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
