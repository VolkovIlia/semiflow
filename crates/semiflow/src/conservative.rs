//! `ConservativeDiffusionChernoff` — order-2 conservative (FV divergence-form)
//! variable-coefficient diffusion (ADR-0187 D2, §56).
//!
//! Generator: `L_k u = ∂_x(k(x) ∂_x u)` with harmonic-mean faces (§56.1).
//! Chernoff step: Crank–Nicolson `(I − ½τL_k)⁻¹(I + ½τL_k)` via O(n) Thomas solve.

use alloc::{sync::Arc, vec, vec::Vec};

use crate::{
    boundary::BoundaryPolicy,
    chernoff::{ChernoffFunction, Growth},
    conservative_assemble::build_faces,
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
    symmetric_operator::SymmetricOperator,
};

// ── Struct ────────────────────────────────────────────────────────────────────

/// Conservative (divergence-form) variable-coefficient diffusion Chernoff function.
///
/// Implements `L_k u = ∂_x(k(x) ∂_x u)` with harmonic-mean face conductivities
/// and optional per-face contact resistance (§56.1).  State: `GridFn1D<F>`.
///
/// ## Chernoff step (order 2, A-stable)
///
/// `S(τ) = (I − ½τL_k)⁻¹(I + ½τL_k)` — one O(n) Thomas tridiagonal solve per step.
/// No CFL constraint. Boundary: Neumann (zero-flux) or Dirichlet (constant).
///
/// ## Bridge to Krylov (§56.2)
///
/// [`Self::to_symmetric_operator`] assembles `A = −L_k` as symmetric PSD CSR
/// consumable by `SymmetricOperator::from_csr` + §54 Krylov (exact, stable).
#[derive(Clone)]
pub struct ConservativeDiffusionChernoff<F: SemiflowFloat = f64> {
    /// Pre-computed face transmissibilities `T_{i+½}` (length n−1, §56.1.b).
    faces: Arc<[F]>,
    /// Reference grid.
    pub grid: Grid1D<F>,
    /// Boundary policy applied at both endpoints.
    pub boundary: BoundaryPolicy<F>,
}

// ── Constructors ──────────────────────────────────────────────────────────────

impl<F: SemiflowFloat> ConservativeDiffusionChernoff<F> {
    /// Node-sampled conductivities (length `grid.n`).
    ///
    /// # Errors
    ///
    /// `DomainViolation` if any `k_i ≤ 0`, length mismatch, non-finite entry,
    /// or invalid `r_contact`.
    pub fn from_k_array(
        grid: Grid1D<F>,
        k_nodes: &[F],
        r_contact: Option<&[F]>,
        boundary: BoundaryPolicy<F>,
    ) -> Result<Self, SemiflowError> {
        let n = grid.n;
        if k_nodes.len() != n {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "ConservativeDiffusionChernoff: k_nodes.len() != grid.n",
                value: k_nodes.len() as f64,
            });
        }
        let dx = grid.dx();
        let faces_vec = build_faces(k_nodes, dx, r_contact)?;
        Ok(Self { faces: faces_vec.into(), grid, boundary })
    }

    /// Sample `k(x)` at grid nodes and delegate to [`Self::from_k_array`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::from_k_array`].
    pub fn from_k_closure<C: Fn(F) -> F>(
        grid: Grid1D<F>,
        k: C,
        boundary: BoundaryPolicy<F>,
    ) -> Result<Self, SemiflowError> {
        let k_nodes: Vec<F> = (0..grid.n).map(|i| k(grid.x_at(i))).collect();
        Self::from_k_array(grid, &k_nodes, None, boundary)
    }

    /// Assemble `A = −L_k` as a `SymmetricOperator` for Krylov propagation (§56.2).
    ///
    /// Uses Neumann BCs for the CSR carrier (natural choice for Krylov).
    ///
    /// # Errors
    ///
    /// Propagates from `SymmetricOperator::from_csr`.
    pub fn to_symmetric_operator(&self) -> Result<SymmetricOperator<F>, SemiflowError> {
        let (row_ptr, col_idx, vals) =
            build_1d_csr_from_faces(self.grid.n, &self.faces, self.grid.dx());
        let sym_tol = F::from(1e-10_f64).unwrap_or(F::zero());
        SymmetricOperator::from_csr(self.grid.n, &row_ptr, &col_idx, &vals, sym_tol)
    }
}

// ── CSR from pre-computed faces ───────────────────────────────────────────────

/// Build CSR for `A = −L_k` from stored face transmissibilities (Neumann BCs).
///
/// Off-diagonal: `A[i,i±1] = −T_{i±½}/dx`. Diagonal: `A[i,i] = (T_{i-½}+T_{i+½})/dx`.
pub(crate) fn build_1d_csr_from_faces<F: SemiflowFloat>(
    n: usize,
    faces: &[F],
    dx: F,
) -> (Vec<usize>, Vec<u32>, Vec<F>) {
    let capacity = 2 + 3 * n.saturating_sub(2);
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx: Vec<u32> = Vec::with_capacity(capacity.max(4));
    let mut vals: Vec<F> = Vec::with_capacity(capacity.max(4));
    row_ptr.push(0usize);
    for i in 0..n {
        let t_l = if i > 0 { faces[i - 1] / dx } else { F::zero() };
        let t_r = if i + 1 < n { faces[i] / dx } else { F::zero() };
        if i > 0 {
            #[allow(clippy::cast_possible_truncation)]
            col_idx.push((i - 1) as u32);
            vals.push(-t_l);
        }
        #[allow(clippy::cast_possible_truncation)]
        col_idx.push(i as u32);
        vals.push(t_l + t_r);
        if i + 1 < n {
            #[allow(clippy::cast_possible_truncation)]
            col_idx.push((i + 1) as u32);
            vals.push(-t_r);
        }
        row_ptr.push(col_idx.len());
    }
    (row_ptr, col_idx, vals)
}

// ── CN step ───────────────────────────────────────────────────────────────────

/// Crank–Nicolson step: `dst ← (I − ½τL_k)⁻¹(I + ½τL_k) src`.
///
/// Stores explicit sub/super-diagonals (not exploiting symmetry) so that Dirichlet
/// boundary overrides are applied correctly without row-coupling issues.
#[allow(clippy::similar_names)] // sub_d / sup_d are standard tridiagonal names
fn cn_step<F: SemiflowFloat>(
    cd: &ConservativeDiffusionChernoff<F>,
    tau: F,
    src: &GridFn1D<F>,
    dst: &mut GridFn1D<F>,
) -> Result<(), SemiflowError> {
    let n = cd.grid.n;
    let dx = cd.grid.dx();
    let half_tau = tau / (F::one() + F::one());

    let mut rhs: Vec<F> = vec![F::zero(); n];
    let mut sub_d: Vec<F> = vec![F::zero(); n];
    let mut diag: Vec<F> = vec![F::zero(); n];
    let mut sup_d: Vec<F> = vec![F::zero(); n];

    for i in 0..n {
        let t_l = if i > 0 { cd.faces[i - 1] / dx } else { F::zero() };
        let t_r = if i + 1 < n { cd.faces[i] / dx } else { F::zero() };
        let src_l = if i > 0 { src.values[i - 1] } else { F::zero() };
        let src_r = if i + 1 < n { src.values[i + 1] } else { F::zero() };
        let lk_src = t_l * src_l - (t_l + t_r) * src.values[i] + t_r * src_r;
        rhs[i] = src.values[i] + half_tau * lk_src;
        sub_d[i] = -half_tau * t_l;
        diag[i] = F::one() + half_tau * (t_l + t_r);
        sup_d[i] = -half_tau * t_r;
    }

    if let BoundaryPolicy::Dirichlet { value } = cd.boundary {
        apply_dirichlet(&mut sub_d, &mut diag, &mut sup_d, &mut rhs, n, value);
    }
    // Neumann and all other variants: natural zero-flux BCs from construction.

    dst.values.resize(n, F::zero());
    thomas_solve(n, &sub_d, &diag, &sup_d, &rhs, &mut dst.values)?;
    dst.grid = cd.grid;
    Ok(())
}

/// Pin both endpoints to `value`: absorb LHS coupling into RHS, then identity rows.
#[allow(clippy::similar_names)] // sub_d / sup_d are standard tridiagonal names
fn apply_dirichlet<F: SemiflowFloat>(
    sub_d: &mut [F],
    diag: &mut [F],
    sup_d: &mut [F],
    rhs: &mut [F],
    n: usize,
    value: F,
) {
    // Left BC: move row-1 sub coupling to RHS, then make row 0 identity.
    if n > 1 {
        rhs[1] -= sub_d[1] * value;
        sub_d[1] = F::zero();
    }
    sub_d[0] = F::zero();
    diag[0] = F::one();
    sup_d[0] = F::zero();
    rhs[0] = value;
    // Right BC: move row-(n-2) sup coupling to RHS, then make row n-1 identity.
    if n > 1 {
        rhs[n - 2] -= sup_d[n - 2] * value;
        sup_d[n - 2] = F::zero();
    }
    sub_d[n - 1] = F::zero();
    diag[n - 1] = F::one();
    sup_d[n - 1] = F::zero();
    rhs[n - 1] = value;
}

// ── Thomas algorithm ──────────────────────────────────────────────────────────

/// O(n) Thomas solver for tridiagonal `(sub, diag, sup) x = rhs`.
///
/// `sub[0]` and `sup[n-1]` are ignored (first/last row have no left/right coupling).
///
/// # Errors
///
/// `DomainViolation` if a pivot is zero (degenerate system).
///
/// # Panics
///
/// Never.
pub(crate) fn thomas_solve<F: SemiflowFloat>(
    n: usize,
    sub: &[F],
    diag: &[F],
    sup: &[F],
    rhs: &[F],
    x: &mut [F],
) -> Result<(), SemiflowError> {
    let mut c_prime: Vec<F> = vec![F::zero(); n];
    let mut d_prime: Vec<F> = vec![F::zero(); n];
    if diag[0] == F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "thomas_solve: zero pivot at row 0",
            value: 0.0,
        });
    }
    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];
    for i in 1..n {
        let w = diag[i] - sub[i] * c_prime[i - 1];
        if w == F::zero() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "thomas_solve: zero pivot",
                value: i as f64,
            });
        }
        c_prime[i] = if i + 1 < n { sup[i] / w } else { F::zero() };
        d_prime[i] = (rhs[i] - sub[i] * d_prime[i - 1]) / w;
    }
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Ok(())
}

// ── ChernoffFunction impls ────────────────────────────────────────────────────

impl ChernoffFunction<f64> for ConservativeDiffusionChernoff<f64> {
    type S = GridFn1D<f64>;

    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }

    /// CN step: `(I − ½τL_k)⁻¹(I + ½τL_k)`.
    ///
    /// # Errors
    ///
    /// `DomainViolation` if `tau < 0` or non-finite.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        _scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        validate_tau(tau)?;
        cn_step(self, tau, src, dst)
    }
}

impl ChernoffFunction<f32> for ConservativeDiffusionChernoff<f32> {
    type S = GridFn1D<f32>;

    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<f32> {
        Growth::contraction()
    }

    fn apply_into(
        &self,
        tau: f32,
        src: &GridFn1D<f32>,
        dst: &mut GridFn1D<f32>,
        _scratch: &mut ScratchPool<f32>,
    ) -> Result<(), SemiflowError> {
        validate_tau(f64::from(tau))?;
        cn_step(self, tau, src, dst)
    }
}

// ── Validation ────────────────────────────────────────────────────────────────

fn validate_tau(tau: f64) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "ConservativeDiffusionChernoff: tau must be finite and >= 0",
            value: tau,
        });
    }
    Ok(())
}

// ── Steady-state solver (public utility for gate tests) ───────────────────────

/// Solve the steady-state 1-D conservative diffusion `L_k u = 0` with Dirichlet BCs.
///
/// Returns the temperature profile `u[0..n]` where `u[0] = t_left`, `u[n-1] = t_right`.
/// Uses O(n) Thomas algorithm — exact for this tridiagonal system.
///
/// The analytic solution is the series-resistance network (§56.4): flux
/// `q = (t_left - t_right) / R_tot` with `R_tot = Σ dx/k_i + Σ R_c`.
///
/// # Errors
///
/// `DomainViolation` on `k_nodes` validation failure or singular system.
///
/// # Panics
///
/// Never.
#[allow(clippy::similar_names)] // sub_d / sup_d are standard tridiagonal names
pub fn steady_state_dirichlet_1d<F: SemiflowFloat>(
    k_nodes: &[F],
    r_contact: Option<&[F]>,
    dx: F,
    t_left: F,
    t_right: F,
) -> Result<Vec<F>, SemiflowError> {
    use crate::conservative_assemble::build_faces;
    let n = k_nodes.len();
    let faces = build_faces(k_nodes, dx, r_contact)?;
    let mut sub_d: Vec<F> = vec![F::zero(); n];
    let mut diag: Vec<F> = vec![F::zero(); n];
    let mut sup_d: Vec<F> = vec![F::zero(); n];
    let mut rhs: Vec<F> = vec![F::zero(); n];
    for i in 0..n {
        let t_l = if i > 0 { faces[i - 1] / dx } else { F::zero() };
        let t_r = if i + 1 < n { faces[i] / dx } else { F::zero() };
        sub_d[i] = -t_l;
        diag[i] = t_l + t_r;
        sup_d[i] = -t_r;
    }
    // Left BC: row 0 = identity with t_left; correct row 1's sub coupling.
    if n > 1 {
        rhs[1] -= sub_d[1] * t_left;
        sub_d[1] = F::zero();
    }
    sub_d[0] = F::zero();
    diag[0] = F::one();
    sup_d[0] = F::zero();
    rhs[0] = t_left;
    // Right BC: row n-1 = identity with t_right; correct row n-2's sup coupling.
    if n > 1 {
        rhs[n - 2] -= sup_d[n - 2] * t_right;
        sup_d[n - 2] = F::zero();
    }
    sub_d[n - 1] = F::zero();
    diag[n - 1] = F::one();
    sup_d[n - 1] = F::zero();
    rhs[n - 1] = t_right;
    let mut u = vec![F::zero(); n];
    thomas_solve(n, &sub_d, &diag, &sup_d, &rhs, &mut u)?;
    Ok(u)
}
