//! Adjoint-state parameter-sensitivity for graph / Magnus kernels (Issue #1, ADR-0115).
//!
//! In-core MATH primitives for `∂(S(τ)u)/∂θ` (forward JVP) and the discrete
//! adjoint-state gradient `∂J/∂θ` (math.md §43, NORMATIVE).
//!
//! No autograd / torch types — those stay in `revssm` (ADR-0115 boundary).
//!
//! ## Key formulae (math.md §43)
//!
//! ```text
//! (∂L_G/∂w_{ij}) v = (e_i−e_j)(e_i−e_j)ᵀ v        (§43.2, 4 nonzeros)
//! δΩ₄ = (τ/2)(δA₁+δA₂) + (√3τ²/12)([δA₂,A₁]+[A₂,δA₁])   (§43.3)
//! δ(S u) = Σ_{m=1..4}(1/m!)Σ_{p=0}^{m-1} Ω₄^p δΩ₄ Ω₄^{m-1-p} u
//! ∂J/∂θ = Σ_k ⟨λ_{k+1}, (∂S_k/∂θ)u_k⟩,  λ_k = S_k⋆ λ_{k+1}   (§43.4)
//! ```

use alloc::vec::Vec;

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph::Laplacian,
    graph_sensitivity_helpers::{d_omega4_lap, fwd_traj, grad_backward, jvp_neumann},
    graph_signal::GraphSignal,
    magnus_graph::{apply_omega4, MagnusGraphHeatChernoff},
    scratch::ScratchPool,
};

#[allow(clippy::approx_constant)]
pub(crate) const SQRT3_12: f64 = 0.144_337_567_297_406_43; // √3 / 12

// ---------------------------------------------------------------------------
// GeneratorSensitivity trait
// ---------------------------------------------------------------------------

/// Provider for `(∂A/∂θ_k) · v` — perturbation of generator `A = −L` in
/// direction of parameter `θ_k`.
pub trait GeneratorSensitivity<F: SemiflowFloat> {
    /// Total number of sensitivity parameters.
    fn n_params(&self) -> usize;

    /// Apply `(∂A/∂θ_k) · v → out`.  `out.len() == v.len()`.
    ///
    /// # Errors
    ///
    /// `DomainViolation` if `k >= self.n_params()`.
    fn apply_param_deriv(
        &self,
        k: usize,
        t: F,
        v: &[F],
        out: &mut [F],
    ) -> Result<(), SemiflowError>;

    /// `grad[k] += w · ⟨a, (∂A/∂θ_k) b⟩` for EVERY parameter `k` (ADR-0203, §63.6).
    ///
    /// This is the contraction of one Duhamel node of
    /// [`graph_expmv_frechet`](crate::graph_expmv_frechet): `a`, `b` are the two
    /// propagated vectors, `w` the quadrature weight on the `σ`-scale (the
    /// weights of a call sum to `t`; do NOT multiply by `t` again). Contract:
    /// `a.len() == b.len()`, `grad.len() == n_params()`; no allocation per
    /// parameter (borrow work vectors from `scratch`).
    ///
    /// The provided default loops [`Self::apply_param_deriv`] and costs
    /// `O(n · n_params)` per call: correct for any implementor, slow for many
    /// parameters. The in-tree sensitivities override it with `O(nnz)` forms.
    ///
    /// # Errors
    ///
    /// `DomainViolation` on a length mismatch or an invalid parameter.
    #[allow(clippy::too_many_arguments)] // §63.8 contract: self + t, w, a, b, grad, scratch
    fn accumulate_bilinear(
        &self,
        t: F,
        w: F,
        a: &[F],
        b: &[F],
        grad: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        check_bilinear_args(a.len(), b.len(), grad.len(), self.n_params())?;
        let mut tmp = scratch.take_vec(b.len());
        let status = bilinear_by_loop(self, (t, w), (a, b), grad, &mut tmp);
        scratch.return_vec(tmp);
        status
    }
}

/// Default contraction: one `apply_param_deriv` + dot product per parameter.
fn bilinear_by_loop<F: SemiflowFloat, S: GeneratorSensitivity<F> + ?Sized>(
    sens: &S,
    (t, w): (F, F),
    (a, b): (&[F], &[F]),
    grad: &mut [F],
    tmp: &mut [F],
) -> Result<(), SemiflowError> {
    for (k, gk) in grad.iter_mut().enumerate() {
        sens.apply_param_deriv(k, t, b, tmp)?;
        let dot = a
            .iter()
            .zip(tmp.iter())
            .fold(F::zero(), |acc, (&x, &y)| acc + x * y);
        *gk += w * dot;
    }
    Ok(())
}

/// Shared argument check of the `accumulate_bilinear` family (§63.8 contract).
pub(crate) fn check_bilinear_args(
    a_len: usize,
    b_len: usize,
    grad_len: usize,
    n_params: usize,
) -> Result<(), SemiflowError> {
    if a_len != b_len {
        return Err(SemiflowError::DomainViolation {
            what: "accumulate_bilinear: a.len() != b.len()",
            #[allow(clippy::cast_precision_loss)]
            value: b_len as f64,
        });
    }
    if grad_len != n_params {
        return Err(SemiflowError::DomainViolation {
            what: "accumulate_bilinear: grad.len() != n_params()",
            #[allow(clippy::cast_precision_loss)]
            value: grad_len as f64,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// apply_edge_weight_deriv — §43.2 rank-1 stencil
// ---------------------------------------------------------------------------

/// Apply `(e_i − e_j)(e_i − e_j)ᵀ v` → `out` (4 nonzeros; §43.2 NORMATIVE).
///
/// This is `(∂L_G/∂w_{ij}) v`.
///
/// # Errors
///
/// `DomainViolation` if indices out of range or `i == j`.
pub fn apply_edge_weight_deriv<F: SemiflowFloat>(
    i: usize,
    j: usize,
    v: &[F],
    out: &mut [F],
) -> Result<(), SemiflowError> {
    let n = v.len();
    if i >= n || j >= n {
        return Err(SemiflowError::DomainViolation {
            what: "apply_edge_weight_deriv: index out of range",
            #[allow(clippy::cast_precision_loss)]
            value: i.max(j) as f64,
        });
    }
    if i == j {
        return Err(SemiflowError::DomainViolation {
            what: "apply_edge_weight_deriv: self-loop (i==j)",
            #[allow(clippy::cast_precision_loss)]
            value: i as f64,
        });
    }
    let diff = v[i] - v[j];
    for x in out.iter_mut() {
        *x = F::zero();
    }
    out[i] = diff;
    out[j] = -diff;
    Ok(())
}

// ---------------------------------------------------------------------------
// magnus_step_jvp_into — §43.3, Laplacian-object variant
// ---------------------------------------------------------------------------

/// Forward-mode JVP of ONE Magnus K=4 step: `δ(S·u)` for a perturbation
/// `δA_i = −δL_i` at the two GL nodes (math.md §43.3). Zero-alloc.
///
/// # Errors
///
/// `DomainViolation` if `tau <= 0`.
#[allow(clippy::too_many_arguments)]
pub fn magnus_step_jvp_into<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    dlap1: &Laplacian<F>,
    dlap2: &Laplacian<F>,
    tau: F,
    u: &[F],
    out: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    crate::magnus_graph::validate_tau(tau)?;
    let n = u.len();
    let mut ta = scratch.take_vec(n);
    let mut tb = scratch.take_vec(n);
    let mut tc = scratch.take_vec(n);
    let mut pw1 = scratch.take_vec(n);
    let mut pw2 = scratch.take_vec(n);
    let mut pw3 = scratch.take_vec(n);
    compute_lap_powers(
        lap1, lap2, tau, u, &mut pw1, &mut pw2, &mut pw3, &mut ta, &mut tb, &mut tc,
    );
    jvp_lap_main(
        lap1, lap2, dlap1, dlap2, tau, u, out, &pw1, &pw2, &pw3, &mut ta, &mut tb, &mut tc, scratch,
    )?;
    scratch.return_vec(pw3);
    scratch.return_vec(pw2);
    scratch.return_vec(pw1);
    scratch.return_vec(tc);
    scratch.return_vec(tb);
    scratch.return_vec(ta);
    Ok(())
}

/// Compute Ω u, Ω²u, Ω³u into pw1/pw2/pw3.
#[allow(clippy::too_many_arguments)]
fn compute_lap_powers<F: SemiflowFloat>(
    l1: &Laplacian<F>,
    l2: &Laplacian<F>,
    tau: F,
    u: &[F],
    pw1: &mut [F],
    pw2: &mut [F],
    pw3: &mut [F],
    ta: &mut [F],
    tb: &mut [F],
    tc: &mut [F],
) {
    let one = F::one();
    apply_omega4(l1, l2, tau, one, u, pw1, ta, tb, tc);
    apply_omega4(l1, l2, tau, one, pw1, pw2, ta, tb, tc);
    apply_omega4(l1, l2, tau, one, pw2, pw3, ta, tb, tc);
}

/// m=1 term + Neumann higher terms for the Lap-variant JVP.
#[allow(clippy::too_many_arguments)]
fn jvp_lap_main<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    dlap1: &Laplacian<F>,
    dlap2: &Laplacian<F>,
    tau: F,
    u: &[F],
    out: &mut [F],
    pw1: &[F],
    pw2: &[F],
    pw3: &[F],
    ta: &mut [F],
    tb: &mut [F],
    tc: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    // m=1: δ(S u) = δΩ u (+ higher).
    d_omega4_lap(lap1, lap2, dlap1, dlap2, tau, u, out, ta, tb, tc);
    let (l1, l2, d1, d2) = (lap1, lap2, dlap1, dlap2);
    let df = |w: &[F], o: &mut [F], a: &mut [F], b: &mut [F], c: &mut [F]| -> Result<_, _> {
        d_omega4_lap(l1, l2, d1, d2, tau, w, o, a, b, c);
        Ok(())
    };
    jvp_neumann(
        lap1, lap2, tau, u, pw1, pw2, pw3, &df, ta, tb, tc, out, scratch,
    )
}

// ---------------------------------------------------------------------------
// adjoint_state_gradient — §43.4 assembled gradient
// ---------------------------------------------------------------------------

/// Assembled adjoint-state gradient `∂J/∂θ` over an `n`-step trajectory
/// (math.md §43.4). Uses `apply_state_adjoint_into_at` (Issue #2) for λ.
///
/// # Errors
///
/// `DomainViolation` if `grad_theta.len() != param_deriv.n_params()`.
#[allow(clippy::too_many_arguments)]
pub fn adjoint_state_gradient<F, P>(
    mc: &MagnusGraphHeatChernoff<F>,
    u0: &GraphSignal<F>,
    n_steps: usize,
    tau: F,
    dj_du_n: &GraphSignal<F>,
    param_deriv: &P,
    grad_theta: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    if grad_theta.len() != param_deriv.n_params() {
        return Err(SemiflowError::DomainViolation {
            what: "adjoint_state_gradient: grad_theta.len() != n_params()",
            #[allow(clippy::cast_precision_loss)]
            value: grad_theta.len() as f64,
        });
    }
    for g in grad_theta.iter_mut() {
        *g = F::zero();
    }
    if n_steps == 0 {
        return Ok(());
    }
    let traj = fwd_traj(mc, u0, n_steps, tau, scratch)?;
    let mut lam = dj_du_n.clone();
    let mut lam_next = GraphSignal::zeros(u0.graph_arc());
    grad_backward(
        mc,
        &traj,
        &mut lam,
        &mut lam_next,
        n_steps,
        tau,
        param_deriv,
        grad_theta,
        scratch,
    )
}

// ---------------------------------------------------------------------------
// EdgeWeightSensitivity — rank-1 §43.2
// ---------------------------------------------------------------------------

/// Edge-weight sensitivity: `(∂A/∂w_{ij}) v = −(e_i−e_j)(e_i−e_j)ᵀ v`.
pub struct EdgeWeightSensitivity {
    /// One `(i, j)` pair per tracked edge weight parameter.
    pub params: Vec<(usize, usize)>,
    /// Number of graph nodes.
    pub n_nodes: usize,
}

impl<F: SemiflowFloat> GeneratorSensitivity<F> for EdgeWeightSensitivity {
    fn n_params(&self) -> usize {
        self.params.len()
    }

    fn apply_param_deriv(
        &self,
        k: usize,
        _t: F,
        v: &[F],
        out: &mut [F],
    ) -> Result<(), SemiflowError> {
        if k >= self.params.len() {
            return Err(SemiflowError::DomainViolation {
                what: "EdgeWeightSensitivity: k out of range",
                #[allow(clippy::cast_precision_loss)]
                value: k as f64,
            });
        }
        let (i, j) = self.params[k];
        apply_edge_weight_deriv(i, j, v, out)?;
        for x in out.iter_mut() {
            *x = -*x;
        }
        Ok(())
    }

    /// `O(n_params)` contraction (§63.6.b): `grad[k] −= w·(a_i − a_j)(b_i − b_j)`.
    fn accumulate_bilinear(
        &self,
        _t: F,
        w: F,
        a: &[F],
        b: &[F],
        grad: &mut [F],
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        check_bilinear_args(a.len(), b.len(), grad.len(), self.params.len())?;
        let n = a.len();
        for (gk, &(i, j)) in grad.iter_mut().zip(&self.params) {
            if i >= n || j >= n || i == j {
                return Err(SemiflowError::DomainViolation {
                    what: "EdgeWeightSensitivity: invalid edge (index out of range or i == j)",
                    #[allow(clippy::cast_precision_loss)]
                    value: i.max(j) as f64,
                });
            }
            *gk -= w * (a[i] - a[j]) * (b[i] - b[j]);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// NodeTimescaleSensitivity — corrected §43.2 form
// ---------------------------------------------------------------------------

/// Node-timescale sensitivity (CORRECTED §43.2):
///
/// ```text
/// (∂A/∂a_k) v = −½ a_k^{-1/2} (E_k L D + D L E_k) v
/// ```
///
/// `L` = bare combinatorial Laplacian, `D = diag(√a)`.
/// NOT the old `½ a_k^{-1/2}(E_k L_a + L_a E_k)` (double-counts D).
pub struct NodeTimescaleSensitivity<F: SemiflowFloat> {
    /// `sqrt_a[i] = √a_i`.
    pub sqrt_a: Vec<F>,
    /// Bare combinatorial Laplacian (without √a scaling).
    pub bare_lap: Laplacian<F>,
}

impl<F: SemiflowFloat> GeneratorSensitivity<F> for NodeTimescaleSensitivity<F> {
    fn n_params(&self) -> usize {
        self.sqrt_a.len()
    }

    fn apply_param_deriv(
        &self,
        k: usize,
        _t: F,
        v: &[F],
        out: &mut [F],
    ) -> Result<(), SemiflowError> {
        let n = v.len();
        if k >= n {
            return Err(SemiflowError::DomainViolation {
                what: "NodeTimescaleSensitivity: k >= n_nodes",
                #[allow(clippy::cast_precision_loss)]
                value: k as f64,
            });
        }
        let half = from_f64::<F>(0.5);
        let pre = half / self.sqrt_a[k]; // ½ a_k^{-1/2}
                                         // Term 1: E_k (L D v) — entry k only.
        let dv: Vec<F> = (0..n).map(|i| self.sqrt_a[i] * v[i]).collect();
        let mut ldv = alloc::vec![F::zero(); n];
        self.bare_lap.apply_into_slice(&dv, &mut ldv);
        for x in out.iter_mut() {
            *x = F::zero();
        }
        out[k] = pre * ldv[k];
        // Term 2: D (L E_k v).
        let mut ekv = alloc::vec![F::zero(); n];
        ekv[k] = v[k];
        let mut lekv = alloc::vec![F::zero(); n];
        self.bare_lap.apply_into_slice(&ekv, &mut lekv);
        for i in 0..n {
            out[i] += pre * self.sqrt_a[i] * lekv[i];
        }
        // ∂A/∂a_k = −∂L_a/∂a_k.
        for x in out.iter_mut() {
            *x = -*x;
        }
        Ok(())
    }

    /// Two `SpMV`s per call (§63.6.b): with `p = L(D b)`, `s = L(D a)`,
    /// `grad[k] −= w·½a_k^{-1/2}·(a_k p_k + b_k s_k)`, using `(E_k L D)ᵀ = D L E_k`.
    fn accumulate_bilinear(
        &self,
        _t: F,
        w: F,
        a: &[F],
        b: &[F],
        grad: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        check_bilinear_args(a.len(), b.len(), grad.len(), self.sqrt_a.len())?;
        let n = a.len();
        if n != self.sqrt_a.len() {
            return Err(SemiflowError::DomainViolation {
                what: "NodeTimescaleSensitivity::accumulate_bilinear: a.len() != n_nodes",
                #[allow(clippy::cast_precision_loss)]
                value: n as f64,
            });
        }
        let mut da = scratch.take_vec(n);
        let mut db = scratch.take_vec(n);
        let mut lda = scratch.take_vec(n);
        let mut ldb = scratch.take_vec(n);
        for i in 0..n {
            da[i] = self.sqrt_a[i] * a[i];
            db[i] = self.sqrt_a[i] * b[i];
        }
        self.bare_lap.apply_into_slice(&db, &mut ldb);
        self.bare_lap.apply_into_slice(&da, &mut lda);
        let half = from_f64::<F>(0.5);
        for node in 0..n {
            let pre = half / self.sqrt_a[node];
            grad[node] -= w * pre * (a[node] * ldb[node] + b[node] * lda[node]);
        }
        scratch.return_vec(ldb);
        scratch.return_vec(lda);
        scratch.return_vec(db);
        scratch.return_vec(da);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

// Tests are in the `graph_sensitivity_tests` sub-module in `lib.rs` (std-only).
// Moved to keep this file within the 500-line suckless limit.
