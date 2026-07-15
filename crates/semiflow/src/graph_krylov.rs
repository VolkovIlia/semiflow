//! Depth-independent graph-semigroup action `e^{-tL_G}·v` via Krylov methods.
//!
//! Implements `GraphKrylovChernoff<F>` with two paths (§54, ADR-0185):
//! - **Chebyshev** (default): degree-m Chebyshev expansion on `[0, λ_max]`,
//!   two work vectors, no Krylov basis stored. O(1) memory.
//! - **Lanczos** (adaptive): m-dim Krylov basis + tridiagonal Padé, O(m·N) memory.
//!
//! `order()` returns `u32::MAX` (tolerance-driven; NOT fixed-order).

use alloc::sync::Arc;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    graph::Laplacian,
    graph_signal::GraphSignal,
    matrix_pade::mat_exp_pade13,
    pcg::implicit_euler_action,
    scratch::ScratchPool,
    state::State,
    symmetric_operator::SymmetricLinearOp,
};

// ── Constants ────────────────────────────────────────────────────────────────

/// Maximum Krylov dimension for the Lanczos path (matches `THETA_M` max m).
const MAX_LANCZOS_DIM: usize = 18;
/// Minimum Chebyshev degree (safety floor — never return m=0 or m=1).
const MIN_CHEB_DEGREE: usize = 3;
/// Maximum Chebyshev degree cap.
const MAX_CHEB_DEGREE: usize = 200;
/// Maximum graph size for `dense_graph_expmv_ref` (gate test helper).
pub const MAX_DENSE_N: usize = 12;
/// Maximum `z = τ·λ_max/2` per Chebyshev substep such that all Bessel coefficients
/// `c_k = e^{-z}·I_k(z)` are finite in f64 (e^{-200} ≈ 1.4e-87, representable).
/// Stiff operators (`z_total > Z_SAFE`) use `s = ⌈z_total / Z_SAFE⌉` substeps.
const Z_SAFE: f64 = 200.0;

/// Al-Mohy–Higham 2011 Table 3.1 (`m`, `θ_m`).  Mirror of `expmv.rs::THETA_M`.
const THETA_M: &[(u32, f64)] = &[
    (1, 2.29e-16),
    (2, 2.58e-8),
    (4, 3.40e-3),
    (5, 1.44e-1),
    (8, 1.44),
    (10, 2.74),
    (13, 4.74),
    (18, 8.84),
];

// ── KrylovPath ───────────────────────────────────────────────────────────────

/// Algorithm variant for [`GraphKrylovChernoff`].
#[derive(Copy, Clone, Debug, Default)]
pub enum KrylovPath {
    /// Chebyshev expansion — two work vectors, degree from Bessel decay. Default.
    #[default]
    Chebyshev,
    /// Lanczos — m-dim Krylov basis + Padé on `T_m`.
    Lanczos {
        /// Maximum Krylov dimension per outer step. Must be ≤ `MAX_LANCZOS_DIM = 18`.
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
    /// [`SemiflowError::DomainViolation`] if `tol ≤ 0` or not finite.
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
        match &self.path {
            KrylovPath::Chebyshev => chebyshev_action(
                &*self.laplacian,
                src,
                dst,
                tau,
                self.lambda_max,
                self.tol,
                scratch,
            ),
            KrylovPath::Lanczos { m_max } => lanczos_action(
                &*self.laplacian,
                src,
                dst,
                tau,
                self.lambda_max,
                *m_max,
                scratch,
            ),
            KrylovPath::ImplicitEuler { n_steps, cg_max_iter } => implicit_euler_gk_action(
                &*self.laplacian,
                src,
                dst,
                tau,
                *n_steps,
                self.tol,
                *cg_max_iter,
                scratch,
            ),
        }
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
/// Chebyshev: `s = ⌈z_total / Z_SAFE⌉` (1 for non-stiff), `m = chebyshev_degree(z_sub, tol)`.
/// Lanczos: `(s, m)` from `THETA_M`, `m` capped at `m_max`.
/// Total `SpMVs` = `s × m`. Used by `G_GRAPH_EXPMV_DEPTH_FLAT`.
///
/// # Panics
/// Panics if `F::from(2.0_f64)` returns `None` (not possible for `f32` or `f64`).
pub fn graph_expmv_matvec_count<F: SemiflowFloat>(
    lambda_max: F,
    tau: F,
    tol: F,
    path: &KrylovPath,
) -> (u32, u32) {
    match path {
        KrylovPath::Chebyshev => {
            let z_total = tau * lambda_max / F::from(2.0_f64).unwrap();
            let s = cheb_substep_count(z_total);
            let step_tau = tau / F::from(f64::from(s)).unwrap();
            let z_sub = step_tau * lambda_max / F::from(2.0_f64).unwrap();
            // chebyshev_degree is bounded by MAX_CHEB_DEGREE = 200 — fits u32.
            #[allow(clippy::cast_possible_truncation)]
            let m = chebyshev_degree(z_sub, tol) as u32;
            (s, m)
        }
        KrylovPath::Lanczos { m_max } => {
            let (s, m) = lanczos_select_s_m(lambda_max, tau);
            // m_max is bounded by MAX_LANCZOS_DIM = 18 — fits u32.
            #[allow(clippy::cast_possible_truncation)]
            let m_max_u32 = *m_max as u32;
            (s, m.min(m_max_u32))
        }
        KrylovPath::ImplicitEuler { n_steps, .. } => {
            // n_steps backward-Euler sub-steps; 0 = no Chebyshev/Lanczos degree (§59.4).
            // n_steps bounded by u32::MAX above — cast is exact.
            #[allow(clippy::cast_possible_truncation)]
            let s = (*n_steps).min(u32::MAX as usize) as u32;
            (s, 0)
        }
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

fn validate_tau<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "GraphKrylovChernoff: tau must be finite and non-negative",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

// ── Chebyshev degree selection ────────────────────────────────────────────────

/// Minimum degree m such that `e^{-z} · I_{m+1}(z) ≤ tol/4` (Bessel tail bound).
fn chebyshev_degree<F: SemiflowFloat>(z: F, tol: F) -> usize {
    let threshold = tol / F::from(4.0_f64).unwrap();
    let em_z = (-z).exp();
    let mut m = MIN_CHEB_DEGREE;
    while m < MAX_CHEB_DEGREE {
        if em_z * bessel_i_k(m + 1, z) <= threshold {
            break;
        }
        m += 1;
    }
    m
}

// ── Chebyshev action (§54.3) ─────────────────────────────────────────────────
//
// e^{-τL_G}v = Σ_{k=0}^m c_k · T_k(B)v,  B = (2/λ_max)·L_G − I,  z = τλ_max/2.
// c_0 = e^{-z}·I_0(z),  c_k = 2·e^{-z}·(−1)^k·I_k(z)  (k ≥ 1).
// Recurrence: T_{k+1}(B)v = 2·B·T_k(B)v − T_{k-1}(B)v.

// Private Chebyshev and Lanczos helpers — include! keeps them in module scope.
include!("graph_krylov_helpers.rs");

// ── Lanczos selection ─────────────────────────────────────────────────────────

fn lanczos_select_s_m<F: SemiflowFloat>(lambda_max: F, tau: F) -> (u32, u32) {
    let arg = tau.to_f64().unwrap_or(1.0) * lambda_max.to_f64().unwrap_or(1.0);
    let mut best: Option<(u32, u32, u64)> = None;
    for &(m, theta) in THETA_M {
        if m as usize > MAX_LANCZOS_DIM {
            break;
        }
        let s_raw = (arg / theta).ceil();
        if s_raw > 1.0e14 {
            continue;
        }
        let s = if s_raw < 1.0 {
            1u32
        } else {
            // s_raw ≥ 1 (guarded above) and ≤ 1e14 — fits u32; ceil preserves sign.
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            {
                s_raw as u32
            }
        };
        let cost = u64::from(s) * u64::from(m);
        if best.map_or(true, |(_, _, pc)| cost < pc) {
            best = Some((s, m, cost));
        }
    }
    // MAX_LANCZOS_DIM = 18 — fits u32.
    #[allow(clippy::cast_possible_truncation)]
    let fallback_m = MAX_LANCZOS_DIM as u32;
    best.map_or((1, fallback_m), |(s, m, _)| (s, m))
}

fn build_exp_tridiag<F: SemiflowFloat>(
    alpha: &[F; MAX_LANCZOS_DIM],
    beta: &[F; MAX_LANCZOS_DIM],
    tau: F,
    m: usize,
) -> Result<[[F; MAX_LANCZOS_DIM]; MAX_LANCZOS_DIM], SemiflowError> {
    let mut t_mat = [[F::zero(); MAX_LANCZOS_DIM]; MAX_LANCZOS_DIM];
    for k in 0..m {
        t_mat[k][k] = -tau * alpha[k];
        if k + 1 < m {
            t_mat[k][k + 1] = -tau * beta[k + 1];
            t_mat[k + 1][k] = -tau * beta[k + 1];
        }
    }
    mat_exp_pade13::<F, MAX_LANCZOS_DIM>(&t_mat)
}

// 7 args by necessity — 4 LaplacianAction state vars + tau/lambda_max/m_max/scratch.
#[allow(clippy::too_many_arguments)]
fn lanczos_action<F: SemiflowFloat>(
    op: &impl SymmetricLinearOp<F>,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    tau: F,
    lambda_max: F,
    m_max: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let (s, m) = lanczos_select_s_m(lambda_max, tau);
    let m = (m as usize).min(m_max);
    // s is u32 — f64::from(s) is exact and infallible.
    let step_tau = tau / F::from(f64::from(s)).unwrap();
    let n = src.len();

    let mut current = scratch.take_vec(n);
    let mut next = scratch.take_vec(n);
    current.copy_from_slice(src.values());

    for _ in 0..s {
        lanczos_step_inner(op, &current, &mut next, step_tau, m, scratch)?;
        core::mem::swap(&mut current, &mut next);
    }
    dst.zero_into();
    dst.axpy_into_slice(F::one(), &current);

    scratch.return_vec(current);
    scratch.return_vec(next);
    Ok(())
}

// ── Slice-based helpers (graph_expmv_krylov) ─────────────────────────────────
// Included at module scope: has full access to chebyshev_accumulate, lanczos_step_inner, etc.
include!("graph_krylov_slice_helpers.rs");

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    include!("graph_krylov_tests_mod.rs");
}
