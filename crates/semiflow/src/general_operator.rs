//! [`GeneralOperator`] — externally-assembled **possibly non-symmetric** CSR
//! operator, and its scaled-truncated-Taylor action (ADR-0195, Issue #24).
//!
//! `SymmetricOperator::from_csr` validates symmetry, so the whole Krylov surface
//! (`evolve_batched` with `chebyshev`/`lanczos`, `phi_action`, `Etdrk4`,
//! `mass_lumped_evolve`) is closed to non-self-adjoint generators. Two concrete
//! quant-finance operators are shut out by that: **drifted Fokker–Planck**
//! `∂_t p = ∂_x(D(x)∂_x p) − ∂_x(μ(x)p)`, whose drift term breaks symmetry, and
//! **inventory-ladder** generators (Cartea–Jaimungal), whose Hopf–Cole system
//! `dω/dt = Aω` has upper-bidiagonal `A` — structurally non-symmetric, tiny
//! (`N ~ 100`), and a natural `expmv` workload.
//!
//! ## Why Taylor and not Arnoldi
//!
//! The issue asks for "an Arnoldi-based path or a `GeneralOperator::from_csr`".
//! This ships the second, because the engine it needs **already exists and is
//! already symmetry-agnostic**: `expmv.rs` implements Al-Mohy & Higham's scaled
//! truncated-Taylor `expmv` (matvec-only), `phi_action` reuses it, and math.md
//! §58.2 already records that the method needs "only a matvec and a real
//! spectral interval — NOT symmetry". Adding a carrier is ~200 lines against
//! Arnoldi's full MGS + Hessenberg + happy-breakdown + restart policy, and it
//! introduces no new numerical claim to gate.
//!
//! It is also the safer engine for the two named targets. The inventory ladder
//! is nilpotent-plus-diagonal — maximally non-normal — where Arnoldi's
//! residual-based stopping is unreliable and unrestarted Arnoldi can stagnate,
//! while Al-Mohy–Higham's backward-error bound holds for *any* `A` and the
//! `s`-scaling bounds `‖(τ/s)A‖ ≤ θ_m` inside every substep.
//!
//! The price is stated rather than hidden: cost is `Θ(τ‖A‖_∞)` matvecs —
//! **linear in the depth, not flat**. See §"Honest limits" in ADR-0195.

extern crate alloc;

use alloc::{sync::Arc, vec, vec::Vec};

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph_signal::GraphSignal,
    scratch::ScratchPool,
    state::State,
};

/// Externally-assembled general (possibly non-symmetric) sparse operator.
///
/// Validates **finiteness and CSR structure only** — no symmetry check, no
/// diagonal-sign check. The caller owns the mathematical preconditions.
#[derive(Clone)]
pub struct GeneralOperator<F: SemiflowFloat = f64> {
    n: usize,
    row_ptr: Arc<Vec<usize>>,
    col_idx: Arc<Vec<u32>>,
    vals: Arc<Vec<F>>,
    /// Transposed CSR, built once so `Aᵀ·v` is a real transpose rather than the
    /// self-adjoint default `GeneratorAction` would otherwise silently use.
    t_row_ptr: Arc<Vec<usize>>,
    t_col_idx: Arc<Vec<u32>>,
    t_vals: Arc<Vec<F>>,
    norm_inf: f64,
}

impl<F: SemiflowFloat> GeneralOperator<F> {
    /// Build from CSR triples. Columns need not be sorted; `A` need not be symmetric.
    ///
    /// # Errors
    /// `DomainViolation` if the CSR shape is inconsistent or any entry is
    /// non-finite.
    pub fn from_csr(
        n: usize,
        row_ptr: &[usize],
        col_idx: &[u32],
        vals: &[F],
    ) -> Result<Self, SemiflowError> {
        validate_csr(n, row_ptr, col_idx, vals)?;
        let norm_inf = row_sum_norm(n, row_ptr, vals);
        let (t_row_ptr, t_col_idx, t_vals) = transpose_csr(n, row_ptr, col_idx, vals);
        Ok(Self {
            n,
            row_ptr: Arc::new(row_ptr.to_vec()),
            col_idx: Arc::new(col_idx.to_vec()),
            vals: Arc::new(vals.to_vec()),
            t_row_ptr: Arc::new(t_row_ptr),
            t_col_idx: Arc::new(t_col_idx),
            t_vals: Arc::new(t_vals),
            norm_inf,
        })
    }

    /// Operator dimension.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Gershgorin row-sum bound `‖A‖_∞ = maxᵢ Σₖ |aᵢₖ| ≥ ρ(A)`.
    ///
    /// Deliberately **not** named `lambda_max_bound`: for a non-symmetric `A`
    /// this is an induced *norm* bound, not a spectral interval. Over-estimation
    /// only raises the Taylor scaling `s` (more, cheaper substeps) and never
    /// harms correctness — the ADR-0121 rationale, unchanged.
    #[must_use]
    pub fn norm_inf_bound(&self) -> f64 {
        self.norm_inf
    }

    /// `dst ← A · src`.
    pub fn apply_into_slice(&self, src: &[F], dst: &mut [F]) {
        csr_matvec(&self.row_ptr, &self.col_idx, &self.vals, src, dst);
    }

    /// `dst ← Aᵀ · src`.
    pub fn apply_transpose_into_slice(&self, src: &[F], dst: &mut [F]) {
        csr_matvec(&self.t_row_ptr, &self.t_col_idx, &self.t_vals, src, dst);
    }

    /// Build the `e^{−τA}·v` kernel for this operator.
    ///
    /// The sign convention matches `SymmetricOperator`/`GraphKrylovChernoff`
    /// (`e^{−τA}`), NOT `expmv.rs`'s `e^{+τA}` — so this type drops into
    /// `graph_batched::evolve_batched` with the same semantics the Python
    /// surface already has for the symmetric path.
    #[must_use]
    pub fn expmv(&self) -> CsrExpmvChernoff<F> {
        CsrExpmvChernoff { op: self.clone() }
    }
}

/// Validate CSR structure and finiteness.
fn validate_csr<F: SemiflowFloat>(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[F],
) -> Result<(), SemiflowError> {
    #[allow(clippy::cast_precision_loss)]
    if n == 0 || row_ptr.len() != n + 1 {
        return Err(SemiflowError::DomainViolation {
            what: "GeneralOperator::from_csr: row_ptr.len() must be n + 1, n >= 1",
            value: row_ptr.len() as f64,
        });
    }
    if col_idx.len() != vals.len() || row_ptr[n] != vals.len() {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "GeneralOperator::from_csr: col_idx/vals length mismatch with row_ptr[n]",
            value: vals.len() as f64,
        });
    }
    for i in 0..n {
        if row_ptr[i] > row_ptr[i + 1] {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "GeneralOperator::from_csr: row_ptr is not non-decreasing",
                value: i as f64,
            });
        }
    }
    for (k, &c) in col_idx.iter().enumerate() {
        if c as usize >= n {
            return Err(SemiflowError::DomainViolation {
                what: "GeneralOperator::from_csr: column index out of range",
                value: f64::from(c),
            });
        }
        if !vals[k].is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "GeneralOperator::from_csr: non-finite entry",
                value: vals[k].to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}

/// `‖A‖_∞ = maxᵢ Σₖ |aᵢₖ|`.
fn row_sum_norm<F: SemiflowFloat>(n: usize, row_ptr: &[usize], vals: &[F]) -> f64 {
    let mut best = 0.0_f64;
    for i in 0..n {
        let mut acc = 0.0_f64;
        for v in &vals[row_ptr[i]..row_ptr[i + 1]] {
            acc += v.to_f64().unwrap_or(f64::NAN).abs();
        }
        if acc > best {
            best = acc;
        }
    }
    best
}

/// Build the transposed CSR by counting sort over columns.
fn transpose_csr<F: SemiflowFloat>(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[F],
) -> (Vec<usize>, Vec<u32>, Vec<F>) {
    let nnz = vals.len();
    let mut counts = vec![0_usize; n + 1];
    for &c in col_idx {
        counts[c as usize + 1] += 1;
    }
    for i in 0..n {
        counts[i + 1] += counts[i];
    }
    let t_row_ptr = counts.clone();
    let mut cursor = counts;
    let mut t_col_idx = vec![0_u32; nnz];
    let mut t_vals = vec![F::zero(); nnz];
    for i in 0..n {
        for k in row_ptr[i]..row_ptr[i + 1] {
            let c = col_idx[k] as usize;
            let pos = cursor[c];
            cursor[c] += 1;
            #[allow(clippy::cast_possible_truncation)]
            {
                t_col_idx[pos] = i as u32;
            }
            t_vals[pos] = vals[k];
        }
    }
    (t_row_ptr, t_col_idx, t_vals)
}

/// `dst ← M · src` for CSR triples.
fn csr_matvec<F: SemiflowFloat>(
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[F],
    src: &[F],
    dst: &mut [F],
) {
    for (i, d) in dst.iter_mut().enumerate() {
        let mut acc = F::zero();
        for k in row_ptr[i]..row_ptr[i + 1] {
            acc += vals[k] * src[col_idx[k] as usize];
        }
        *d = acc;
    }
}

// ---------------------------------------------------------------------------
// CsrExpmvChernoff
// ---------------------------------------------------------------------------

/// `e^{−τA}·v` for a general CSR operator via scaled truncated Taylor (§45).
///
/// Implements [`ChernoffFunction`] over [`GraphSignal`] so it composes with the
/// existing batched and channel-parallel layer (`graph_batched`) unchanged.
#[derive(Clone)]
pub struct CsrExpmvChernoff<F: SemiflowFloat = f64> {
    op: GeneralOperator<F>,
}

impl<F: SemiflowFloat> CsrExpmvChernoff<F> {
    /// The underlying operator.
    #[must_use]
    pub fn operator(&self) -> &GeneralOperator<F> {
        &self.op
    }

    /// `dst ← e^{−τA}·src` on raw slices.
    ///
    /// # Errors
    /// `DomainViolation` on a length mismatch or a non-finite result.
    pub fn action_into_slice(
        &self,
        tau: F,
        src: &[F],
        dst: &mut [F],
    ) -> Result<(), SemiflowError> {
        let n = self.op.n;
        if src.len() != n || dst.len() != n {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "CsrExpmvChernoff: slice length != operator dimension",
                value: src.len() as f64,
            });
        }
        let tau_f = tau.to_f64().unwrap_or(f64::NAN);
        if !tau_f.is_finite() || tau_f < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "CsrExpmvChernoff: tau must be finite and >= 0",
                value: tau_f,
            });
        }
        let (n_sub, degree) = select_s_m_tight(self.op.norm_inf, tau_f);
        let tau_s = from_f64::<F>(tau_f / f64::from(n_sub));
        let mut y = src.to_vec();
        let mut work = vec![F::zero(); n];
        let mut av = vec![F::zero(); n];
        for _ in 0..n_sub {
            horner_substep(&self.op, &mut y, &mut work, &mut av, tau_s, degree);
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(SemiflowError::DomainViolation {
                what: "CsrExpmvChernoff: non-finite result (operator norm too large?)",
                value: self.op.norm_inf,
            });
        }
        dst.copy_from_slice(&y);
        Ok(())
    }
}

/// Scaling/degree selection for this kernel.
///
/// Delegates to [`crate::expmv::select_s_m`], the shared Al-Mohy–Higham
/// selector. It did NOT always do so: this kernel originally carried its own
/// derived criterion (`z ≤ (u·(m+1)!)^{1/(m+1)} = 1.1894` at `m = 18`) because
/// the shared θ table was optimistic when fed a *tight* norm — at `arg = 7.02`
/// it returned `(s, m) = (1, 18)`, whose measured truncation error on the
/// drifted-Fokker–Planck datum was `1.6e−4`, not double precision. Existing
/// callers never saw it because they pass deliberately loose bounds
/// (`DiffusionExpmvChernoff` uses `4·a_norm_bound/dx²`) that inflate `s` enough
/// to compensate; `GeneralOperator` passes the tight `‖A‖_∞`.
///
/// ADR-0198 traced that to a mis-transcribed table and corrected it. The derived
/// forward-error radius (1.1894) and the corrected backward-error radius
/// (`θ_18 = 1.091`) agree to within 10%, so the duplication no longer buys
/// anything and the shared selector — which also offers degrees up to 30, and so
/// fewer substeps — is used instead.
fn select_s_m_tight(norm_a: f64, tau: f64) -> (u32, u32) {
    crate::expmv::select_s_m(norm_a, tau)
}

/// One Horner substep of `e^{−τ_s A}` truncated at degree `m`.
///
/// Note the sign: the term factor is `−τ_s/k`, giving `e^{−τ_s A}`, matching the
/// graph/symmetric convention rather than `expmv.rs`'s `e^{+τA}`.
fn horner_substep<F: SemiflowFloat>(
    op: &GeneralOperator<F>,
    y: &mut [F],
    w: &mut [F],
    av: &mut [F],
    tau_s: F,
    m: u32,
) {
    w.copy_from_slice(y);
    for k in 1..=m {
        op.apply_into_slice(w, av);
        let factor = F::zero() - tau_s / from_f64::<F>(f64::from(k));
        for (wi, &avi) in w.iter_mut().zip(av.iter()) {
            *wi = factor * avi;
        }
        for (yi, &wi) in y.iter_mut().zip(w.iter()) {
            *yi += wi;
        }
    }
}

impl<F: SemiflowFloat> ChernoffFunction<F> for CsrExpmvChernoff<F> {
    type S = GraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let mut out = vec![F::zero(); self.op.n];
        self.action_into_slice(tau, src.values(), &mut out)?;
        dst.zero_into();
        dst.axpy_into_slice(F::one(), &out);
        Ok(())
    }

    /// Tolerance-driven, not a fixed-order Chernoff function.
    ///
    /// `u32::MAX` is the honest answer: the Taylor degree `m` and scaling `s` are
    /// selected per call from `τ‖A‖_∞`, so there is no fixed consistency order
    /// to report. Callers that schedule PI gains from `order()` must not use
    /// this kernel.
    fn order(&self) -> u32 {
        u32::MAX
    }

    fn growth(&self) -> Growth<F> {
        Growth::new(F::one(), from_f64::<F>(self.op.norm_inf))
    }
}

/// Taylor scaling/degree `(s, m)` this kernel selects for `‖A‖` at time `tau`.
///
/// Exposed so the ADVISORY cost gate can measure that this path's work is
/// `Θ(τ‖A‖)` — linear, not depth-flat. Making the anti-claim measurable is the
/// point; `G_GRAPH_EXPMV_DEPTH_FLAT` must never be extended to this path.
#[must_use]
pub fn expmv_cost_probe(norm_a: f64, tau: f64) -> (u32, u32) {
    select_s_m_tight(norm_a, tau)
}
