//! Depth-independent graph-semigroup action `e^{-tL_G}·v` via Krylov methods.
//!
//! Implements `GraphKrylovChernoff<F>` with two paths (§54, ADR-0185, ADR-0205):
//! - **Chebyshev** (default): ONE expansion of degree `m ≈ √(2z·ln(1/tol))`,
//!   `z = τλ_max/2`, with exponentially scaled Bessel coefficients
//!   (`crate::cheb_coeffs`); four work vectors, no Krylov basis. Cost `∝ √(τλ)`.
//! - **Lanczos**: `s` steps of an `m`-dimensional Krylov space (`m ≤ m_max`),
//!   `(s, m)` from the Hochbruck–Lubich a-priori bound, `e^{−hT_m}` by a
//!   tridiagonal eigen-decomposition; O(m·N) memory.
//!
//! `order()` returns `u32::MAX` (tolerance-driven; NOT fixed-order).

use alloc::{sync::Arc, vec::Vec};

// `f64` has inherent math methods only when `std` is linked; otherwise they come
// from `num_traits::Float` (libm). Test builds link `std` even without the
// feature (harness, dev-dependencies), hence the `allow`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{
    cheb_coeffs::{exp_chebyshev_coefficients, exp_chebyshev_degree, MAX_CHEB_DEGREE},
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    graph::Laplacian,
    graph_signal::GraphSignal,
    lanczos_sched::lanczos_schedule,
    matrix_pade::mat_exp_pade13,
    pcg::implicit_euler_action,
    scratch::ScratchPool,
    state::State,
    symmetric_operator::SymmetricLinearOp,
};

// ── Constants ────────────────────────────────────────────────────────────────

/// Maximum graph size for `dense_graph_expmv_ref` (gate test helper).
pub const MAX_DENSE_N: usize = 12;

// ── KrylovPath ───────────────────────────────────────────────────────────────

/// Algorithm variant for [`GraphKrylovChernoff`].
#[derive(Copy, Clone, Debug, Default)]
pub enum KrylovPath {
    /// Chebyshev expansion — one series of degree `≈ √(2z·ln(1/tol))`,
    /// `z = τλ_max/2`; four work vectors. Default.
    #[default]
    Chebyshev,
    /// Lanczos — `m`-dim Krylov basis, steps from the Hochbruck–Lubich bound.
    Lanczos {
        /// Maximum Krylov dimension per step (clamped to `[1, 512]` and to `n`).
        m_max: usize,
    },
    /// Implicit backward-Euler shift-invert (§59, ADR-0190).
    ///
    /// Computes `e^{-τÂ}v ≈ (I + Δt·Â)^{-n_steps} v` where `Δt = τ/n_steps`.
    /// Each sub-step solves `(I + Δt·Â) x = b` by preconditioned CG.
    /// Cost is `O(n_steps · √κ · N)` vs `O(λ_max · τ · N)` for explicit paths.
    /// L-stable — damps stiff modes unlike A-stable Crank–Nicolson.
    ImplicitEuler {
        /// Number of backward-Euler sub-steps. Must be ≥ 1.
        n_steps: usize,
        /// Optional CG iteration cap per sub-step.
        ///
        /// `None` (default) uses the theory-derived bound
        /// `ceil(√κ(S)·ln(2/tol))` where `κ(S) ≤ 1 + Δt·λ_max` (§59.4, fix #18).
        /// Set `Some(m)` to override, e.g. when `λ_max_bound` is a loose Gershgorin
        /// estimate and a tighter user-known bound suffices.
        cg_max_iter: Option<usize>,
    },
}

// ── GraphKrylovChernoff ───────────────────────────────────────────────────────

/// Depth-independent graph-semigroup action `e^{-τL_G}·v` (A1, §54, ADR-0185).
///
/// # Boundary (D5)
/// Symmetric `L_G` only (`Combinatorial` and `SymNormalized` Laplacians).
/// Non-symmetric (directed) → [`SemiflowError::Unsupported`] in a future variant.
#[derive(Clone)]
pub struct GraphKrylovChernoff<F: SemiflowFloat = f64> {
    laplacian: Arc<Laplacian<F>>,
    /// Cached Gershgorin bound `λ_max(L_G)` from `spectral_radius_bound()`.
    lambda_max: F,
    path: KrylovPath,
    /// Target accuracy ε for degree / Krylov-dimension selection.
    tol: F,
}

impl<F: SemiflowFloat> GraphKrylovChernoff<F> {
    /// Construct from a symmetric Laplacian and tolerance `tol`.
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `tol ≤ 0` or not finite, or the
    /// Laplacian's Gershgorin bound is non-finite (issue #44: finite entries such
    /// as a 2×2 block of `1e308` overflow it; every degree / substep schedule is
    /// derived from it).
    pub fn new(
        laplacian: Arc<Laplacian<F>>,
        path: KrylovPath,
        tol: F,
    ) -> Result<Self, SemiflowError> {
        if !tol.is_finite() || tol <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "GraphKrylovChernoff: tol must be finite and positive",
                value: tol.to_f64().unwrap_or(f64::NAN),
            });
        }
        let lambda_max = laplacian.spectral_radius_bound();
        check_schedule_arg(lambda_max, F::zero())?;
        Ok(Self {
            laplacian,
            lambda_max,
            path,
            tol,
        })
    }

    /// Convenience constructor: Chebyshev path, tol = 1e-10.
    ///
    /// # Panics
    /// Panics if `F::from(1e-10_f64)` returns `None` (only possible for exotic
    /// `F` implementations that cannot represent 1e-10; all standard floats are fine).
    #[must_use]
    pub fn default_cheb(laplacian: Arc<Laplacian<F>>) -> Self {
        let lambda_max = laplacian.spectral_radius_bound();
        Self {
            lambda_max,
            laplacian,
            path: KrylovPath::Chebyshev,
            tol: F::from(1e-10_f64).unwrap(),
        }
    }

    /// Cached Gershgorin bound `ρ̄ ≥ λ_max(L_G)` the polynomial paths are scaled with.
    ///
    /// Input of `graph_expmv_frechet_plan` and of the mesh of §63.3 (ADR-0203).
    #[must_use]
    pub fn lambda_max_bound(&self) -> F {
        self.lambda_max
    }

    /// Propagator tolerance `tol` (also the decay-skip threshold of §63.5.b).
    #[must_use]
    pub fn tol(&self) -> F {
        self.tol
    }

    /// Algorithm variant this solver runs.
    #[must_use]
    pub fn path(&self) -> KrylovPath {
        self.path
    }

    /// `out ← e^{−τL}·v` on plain slices (no `GraphSignal`, no allocation of domain objects).
    ///
    /// Same polynomial/Krylov selection as `apply_into`; used by the §63.5 sweep.
    pub(crate) fn expmv_slice(
        &self,
        tau: F,
        v: &[F],
        out: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        graph_expmv_krylov(&*self.laplacian, tau, v, out, self.path, self.tol, scratch)
    }

    /// Number of nodes in the underlying graph.  Used by A2 (`graph_expmv_frechet`).
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.laplacian.n_nodes()
    }
}

// ── ChernoffFunction impl ─────────────────────────────────────────────────────

impl<F: SemiflowFloat> ChernoffFunction<F> for GraphKrylovChernoff<F> {
    type S = GraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        validate_tau(tau)?;
        check_schedule_arg(self.lambda_max, tau)?;
        let (op, lambda, tol, path) = (&*self.laplacian, self.lambda_max, self.tol, self.path);
        via_slices(src, dst, scratch, |v, out, sc| match path {
            KrylovPath::Chebyshev => expmv_chebyshev(op, lambda, tau, v, out, tol, sc),
            KrylovPath::Lanczos { m_max } => expmv_lanczos(op, lambda, tau, v, out, m_max, tol, sc),
            KrylovPath::ImplicitEuler {
                n_steps,
                cg_max_iter,
            } => implicit_euler_action(
                op as &dyn SymmetricLinearOp<F>,
                v,
                out,
                tau,
                n_steps,
                tol,
                cg_max_iter,
                sc,
            ),
        })
    }

    /// `u32::MAX`: tolerance-driven, no fixed polynomial order (same as `DiffusionExpmvChernoff`).
    fn order(&self) -> u32 {
        u32::MAX
    }

    fn growth(&self) -> Growth<F> {
        // L_G is PSD ⇒ e^{-τL_G} is a contraction.
        Growth::contraction()
    }
}

// ── Public instrumentation (depth-flat gate) ──────────────────────────────────

/// Returns `(s, m)` where `s` = substep count and `m` = degree/Krylov dimension.
///
/// Chebyshev: `s = 1` and `m ≈ √(2z·ln(1/tol))`, `z = τλ_max/2` (`s > 1` only
/// beyond `MAX_CHEB_DEGREE`, `z ≳ 1.5·10¹⁰`); `(1, 0)` for `z = 0`.
/// Lanczos: `(s, m ≤ m_max)` from the Hochbruck–Lubich bound (`lanczos_sched`),
/// with `n = m_max` (the operator size is not known here; the kernel also caps
/// `m` at `n`). `ImplicitEuler`: `(n_steps, 0)`.
/// Total `SpMVs` = `s × m`. Used by `G_GRAPH_EXPMV_DEPTH_FLAT` and the Fréchet
/// planner. Saturates at `u32::MAX`.
#[must_use]
pub fn graph_expmv_matvec_count<F: SemiflowFloat>(
    lambda_max: F,
    tau: F,
    tol: F,
    path: &KrylovPath,
) -> (u32, u32) {
    let sat = |x: u64| u32::try_from(x).unwrap_or(u32::MAX);
    let tol_f = tol.to_f64().unwrap_or(1e-10);
    match path {
        KrylovPath::Chebyshev => {
            let z = (tau * lambda_max).to_f64().unwrap_or(f64::NAN) * 0.5;
            let (s, m) = chebyshev_schedule(z, tol_f);
            (s, sat(m as u64))
        }
        KrylovPath::Lanczos { m_max } => {
            let (s, m) = lanczos_schedule(
                lambda_max.to_f64().unwrap_or(f64::NAN),
                tau.to_f64().unwrap_or(f64::NAN),
                tol_f,
                *m_max,
                *m_max,
            );
            (sat(s), sat(m as u64))
        }
        KrylovPath::ImplicitEuler { n_steps, .. } => (sat(*n_steps as u64), 0),
    }
}

/// Dense reference `e^{-τ·L_G}·v` via `mat_exp_pade13` for `N ≤ MAX_DENSE_N = 12`.
///
/// Extracts the N×N dense Laplacian from CSR, scales by `−τ`, and exponentiates.
/// Used by the `G_GRAPH_EXPMV_DENSE` gate test.
///
/// # Errors
/// [`SemiflowError::DomainViolation`] if `n_nodes > MAX_DENSE_N`.
pub fn dense_graph_expmv_ref<F: SemiflowFloat>(
    laplacian: &Laplacian<F>,
    tau: F,
    src: &[F],
    dst: &mut [F],
) -> Result<(), SemiflowError> {
    let n = laplacian.n_nodes();
    if n > MAX_DENSE_N {
        // n is a node count — precision loss is impossible in practice (n ≤ MAX_DENSE_N = 12).
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "dense_graph_expmv_ref: n_nodes > MAX_DENSE_N (12)",
            value: n as f64,
        });
    }
    // Build -τ·L_G as a MAX_DENSE_N×MAX_DENSE_N matrix (zero-padded).
    let mut mat = [[F::zero(); MAX_DENSE_N]; MAX_DENSE_N];
    let mut unit = [F::zero(); MAX_DENSE_N];
    let mut col = [F::zero(); MAX_DENSE_N];
    for j in 0..n {
        unit[j] = F::one();
        laplacian.apply_into_slice(&unit[..n], &mut col[..n]);
        for i in 0..n {
            mat[i][j] = -tau * col[i];
        }
        unit[j] = F::zero();
    }
    let exp_mat = mat_exp_pade13::<F, MAX_DENSE_N>(&mat)?;
    // dst = exp_mat (upper-left n×n block) · src
    for i in 0..n {
        dst[i] = (0..n)
            .map(|j| exp_mat[i][j] * src[j])
            .fold(F::zero(), |s, x| s + x);
    }
    Ok(())
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Reject a schedule argument `τ·λ_max` that is non-finite or negative (issue #44).
///
/// `λ_max` comes from `SymmetricLinearOp::lambda_max_bound`, which external
/// implementors (and derived operators such as `MassKOperator`) may compute
/// themselves; an `∞` there used to select a single, silently inaccurate step.
fn check_schedule_arg<F: SemiflowFloat>(lambda_max: F, tau: F) -> Result<(), SemiflowError> {
    let arg = tau * lambda_max;
    if lambda_max.is_finite() && lambda_max >= F::zero() && arg.is_finite() {
        Ok(())
    } else {
        Err(SemiflowError::DomainViolation {
            what: "Krylov expmv: lambda_max bound must be finite and >= 0 (tau*lambda_max finite)",
            value: lambda_max.to_f64().unwrap_or(f64::NAN),
        })
    }
}

fn validate_tau<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "GraphKrylovChernoff: tau must be finite and non-negative",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

// Private Chebyshev / Lanczos helpers — include! keeps them in module scope.
include!("graph_krylov_helpers.rs");

// ── Slice-based helpers (graph_expmv_krylov) ─────────────────────────────────
// Included at module scope: has full access to chebyshev_accumulate, lanczos_step_inner, etc.
include!("graph_krylov_slice_helpers.rs");

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    include!("graph_krylov_tests_mod.rs");
}
