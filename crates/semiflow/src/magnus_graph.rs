//! Fourth-order Magnus expansion for time-dependent graph heat:
//! `∂_t u = −L_G(t) u` on a fixed-topology weighted graph.
//!
//! Two-point Gauss-Legendre quadrature (GL₄) + first commutator term:
//!
//! ```text
//! Ω₄(τ) = (τ/2)·(A₁ + A₂) + (√3·τ²/12) · [A₂, A₁]
//! ```
//!
//! with `A_i = −L_G(c_i · τ)`, `c₁ = (3−√3)/6`, `c₂ = (3+√3)/6`.
//!
//! The Magnus map applied to `f` is then evaluated by degree-4 Taylor
//! truncation of `exp(Ω₄)`:
//!
//! ```text
//! S₄(τ) f ≈ Σ_{k=0..4} (Ω₄(τ))^k · f / k!
//! ```
//!
//! **This is the FIRST genuine Magnus expansion in `semiflow-core`.**
//! The pre-existing `TruncatedExpDiffusionChernoff` /
//! `TruncatedExp4thDiffusionChernoff` types were renamed from `Magnus*`
//! in v0.7.0 (audit finding D2) because they truncate `exp(τG)` for a
//! frozen `G`, not a Magnus expansion.
//!
//! # Citations
//!
//! - A. Iserles, H. Z. Munthe-Kaas, S. P. Nørsett, A. Zanna,
//!   *Lie-group methods*, **Acta Numerica** **9** (2000) 215–365.
//!   DOI 10.1017/S0962492900002154. (§5 establishes the fourth-order Magnus
//!   method; eq. (5.10) gives the two-point GL₄ formula used here.)
//! - S. Blanes, F. Casas, J. A. Oteo, J. Ros,
//!   *The Magnus expansion and some of its applications*,
//!   **Physics Reports** **470** (2009) 151–238.
//!   DOI 10.1016/j.physrep.2008.11.001. (Tables 5–6 list fourth-order
//!   Magnus weights; §3 Theorem 1 gives the convergence-radius condition.)
//! - M. Hochbruck, A. Ostermann, *Exponential integrators*,
//!   **Acta Numerica** **19** (2010) 209–286.
//!   DOI 10.1017/S0962492910000048. (§3 reviews `exp(Ω)·v` evaluation
//!   on bounded operators via Taylor truncation.)
//!
//! See math.md §12.9 (NORMATIVE library policy) and ADR-0051 (design).
//!
//! # Zero-alloc steady state
//!
//! `apply_into` acquires 5 scratch buffers via `ScratchPool::take_vec` and
//! returns them all before returning — R4 zero-alloc invariant preserved.

use alloc::sync::Arc;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    graph::Graph,
    graph_signal::GraphSignal,
    graph_traj::GraphTraj,
    scratch::ScratchPool,
    state::State,
};

// Free-function helpers (validation, trajectory, kernels) — re-exported so
// that external callers (magnus_graph_adjoint, tests) keep the canonical
// `crate::magnus_graph::*` paths unchanged.
pub(crate) use crate::magnus_graph_helpers::{
    apply_magnus_k4_into_at, apply_omega4, run_all_segments, validate_magnus_radius, validate_rho,
    validate_tau, validate_traj_inputs,
};

// ---------------------------------------------------------------------------
// GL₄ constants (NORMATIVE — DO NOT CHANGE; see contract §3)
// ---------------------------------------------------------------------------

/// GL₄ first abscissa: `c₁ = (3 − √3) / 6 ≈ 0.211324865405187`.
///
/// Source: Iserles+ 2000 *Acta Numerica* §5.5; Numerical Recipes 3e §4.6.1.
// NORMATIVE constant (contract §3): extra digits beyond f64 ULP are kept for
// documentation and to match the Sympy reference; clippy::excessive_precision
// is suppressed because the value is verified symbolically in T12.
#[allow(clippy::excessive_precision)]
pub(crate) const GL4_C1_F64: f64 = 0.211_324_865_405_187_134;

/// GL₄ second abscissa: `c₂ = (3 + √3) / 6 ≈ 0.788675134594813`.
///
/// Source: Iserles+ 2000 *Acta Numerica* §5.5; Numerical Recipes 3e §4.6.1.
#[allow(clippy::excessive_precision)]
pub(crate) const GL4_C2_F64: f64 = 0.788_675_134_594_812_866;

/// Commutator coefficient: `√3 / 12 ≈ 0.144337567297406`.
///
/// Appears in `Ω₄(τ) = (τ/2)(A₁+A₂) + (√3τ²/12)[A₂,A₁]`.
/// Source: Iserles+ 2000 eq. (5.10); Blanes+ 2009 Table 5.
#[allow(clippy::excessive_precision)]
pub(crate) const SQRT3_OVER_12_F64: f64 = 0.144_337_567_297_406_433;

// ---------------------------------------------------------------------------
// Type alias
// ---------------------------------------------------------------------------

/// Caller-supplied closure mapping a time point `t` to the Laplacian
/// `L_G(t)` valid at that time.
///
/// # Contract
///
/// - **Pure**: same `t` → equal output (no side effects, no global state).
/// - **Topology fixed**: `row_ptr` and `col_idx` of every returned
///   `Laplacian` MUST equal those of the topology graph passed to
///   [`MagnusGraphHeatChernoff::new`]. The library enforces this via
///   `debug_assert!` in debug builds; release builds skip the check.
/// - **Send + Sync + 'static**: closure may be shared across threads.
/// - **C² regularity**: `lap_at_t(·)` must be twice continuously
///   differentiable in `t` (required by GL₄ quadrature error analysis).
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use semiflow_core::{Graph, Laplacian, LaplacianAtTime};
///
/// let topology = Arc::new(Graph::<f64>::path(8));
/// let lap_at: LaplacianAtTime<f64> = {
///     let topo = Arc::clone(&topology);
///     Box::new(move |_t: f64| {
///         Arc::new(Laplacian::assemble_combinatorial(&topo))
///     })
/// };
/// ```
/// Type alias for [`graph_traj::SegmentWeightFn`] — identical closure type,
/// unified to remove duplication (ADR Wave 2.2A §8).
pub type LaplacianAtTime<F> = crate::graph_traj::SegmentWeightFn<F>;

// ---------------------------------------------------------------------------
// MagnusGraphHeatChernoff<F>
// ---------------------------------------------------------------------------

/// Fourth-order Magnus Chernoff for `∂_t u = −L_G(t) u` on a fixed-topology
/// weighted graph with time-varying edge weights.
///
/// Implements [`ChernoffFunction<F, S = GraphSignal<F>>`] with `order() == 4`.
/// Uses two-point Gauss-Legendre quadrature (GL₄) + first commutator term:
///
/// ```text
/// Ω₄(τ) = (τ/2)(A₁ + A₂) + (√3τ²/12)[A₂,A₁]
/// ```
///
/// with `A_i = −L_G(t₀ + c_i · τ)`, `c₁ = (3−√3)/6`, `c₂ = (3+√3)/6`.
///
/// **Time-tracking note**: [`ChernoffFunction::apply_into`] only receives `τ`
/// (step size), not the absolute time `t₀`. When used via
/// [`crate::ChernoffSemigroup`], all steps sample from `t ∈ [0, τ]`, giving
/// globally correct results only for time-independent or slowly varying `L_G`.
/// For accurate time-varying evolution, use [`apply_into_at`](Self::apply_into_at)
/// with explicit `t_start`, or drive a manual loop as in the G11 gate test.
///
/// **This is the FIRST genuine Magnus expansion in semiflow-core** — see
/// module-level documentation for the naming history.
///
/// # Convergence
///
/// Global error `‖(S₄(t/n))^n f − u_exact(t)‖₂ = O(1/n⁴)` by Iserles+
/// 2000 §5 Theorem 5.2 + Chernoff product formula.
///
/// # Convergence-radius check
///
/// Each `apply_into` call validates `ρ̄_max · τ < π/2` (50% safety margin
/// vs. theoretical convergence radius `< π`). On violation it returns
/// [`SemiflowError::OutOfMagnusRadius`]. Caller must reduce `τ` or supply a
/// tighter `rho_bar_max`.
///
/// # Zero-alloc steady state (R4 invariant)
///
/// `apply_into` acquires 5 scratch buffers from `ScratchPool` via
/// `take_vec` and returns them all before returning. No heap allocations
/// in the steady-state hot path.
///
/// # Topology invariant
///
/// `row_ptr` and `col_idx` of every Laplacian returned by `lap_at_t(t)` MUST
/// equal those of the topology graph passed to [`new`](Self::new). Debug
/// builds enforce this via `debug_assert!`; release builds skip the check.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use semiflow_core::{
///     Graph, Laplacian, GraphSignal, LaplacianAtTime,
///     MagnusGraphHeatChernoff, ChernoffSemigroup,
/// };
///
/// let topology = Arc::new(Graph::<f64>::path(16));
/// let topo2 = Arc::clone(&topology);
/// let lap_at: LaplacianAtTime<f64> = Box::new(move |_t| {
///     Arc::new(Laplacian::assemble_combinatorial(&topo2))
/// });
/// let rho_bar = 4.0_f64; // conservative Gershgorin bound
/// let mc = MagnusGraphHeatChernoff::new(Arc::clone(&topology), lap_at, rho_bar, true)
///     .expect("valid inputs");
/// let f0 = GraphSignal::from_fn(Arc::clone(&topology), |i| (i as f64 * 0.1).sin());
/// let semi = ChernoffSemigroup::new(mc, 50).expect("n >= 1");
/// let _u = semi.evolve(0.5, &f0).expect("evolve");
/// ```
pub struct MagnusGraphHeatChernoff<F: SemiflowFloat = f64> {
    /// Topology graph. Fixed across all sampled `L_G(t)`.
    pub(crate) graph: Arc<Graph<F>>,
    /// `t ↦ Arc<Laplacian<F>>` — caller-supplied edge-weight sampler.
    pub(crate) lap_at_t: LaplacianAtTime<F>,
    /// Peak Gershgorin radius bound `ρ̄_max` over `t ∈ [0, t_horizon]`.
    pub(crate) rho_bar_max: F,
    /// If `true`, every `apply_into` validates `ρ̄_max · τ < π/2`.
    pub(crate) convergence_radius_check: bool,
}

impl<F: SemiflowFloat> MagnusGraphHeatChernoff<F> {
    /// Construct from topology + time-to-Laplacian closure.
    ///
    /// # Parameters
    ///
    /// - `graph`: fixed-topology graph (Wave 2.1A).
    /// - `lap_at_t`: closure `t ↦ Arc<Laplacian<F>>`. MUST be pure and
    ///   MUST preserve `graph.row_ptr()` and `graph.col_idx()` across all `t`.
    /// - `rho_bar_max`: caller-supplied upper bound for
    ///   `max_{t} ρ̄(L_G(t))`. Used for `growth()` and for the Magnus
    ///   convergence-radius check. For a path graph with max edge weight `w`,
    ///   `ρ̄ ≤ 2 · w` suffices.
    /// - `convergence_radius_check`: if `true`, each `apply_into` rejects `τ`
    ///   with `rho_bar_max · τ ≥ π/2`. Recommended: `true`.
    ///
    /// # Errors
    ///
    /// - [`SemiflowError::DomainViolation`] if `rho_bar_max` is not strictly
    ///   positive and finite, or if `graph.n_nodes() == 0`.
    pub fn new(
        graph: Arc<Graph<F>>,
        lap_at_t: LaplacianAtTime<F>,
        rho_bar_max: F,
        convergence_radius_check: bool,
    ) -> Result<Self, SemiflowError> {
        validate_rho(rho_bar_max)?;
        if graph.n_nodes() == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "MagnusGraphHeatChernoff: graph must have at least one node",
                value: 0.0,
            });
        }
        Ok(Self {
            graph,
            lap_at_t,
            rho_bar_max,
            convergence_radius_check,
        })
    }

    /// Borrow the topology graph.
    #[must_use]
    pub fn graph(&self) -> &Graph<F> {
        &self.graph
    }

    /// Sampled Laplacian at time `t` (clones the `Arc`).
    #[must_use]
    pub fn laplacian_at(&self, t: F) -> Arc<crate::graph::Laplacian<F>> {
        (self.lap_at_t)(t)
    }

    /// Apply one Magnus K=4 step starting at absolute time `t_start`.
    ///
    /// GL₄ quadrature nodes are `t_start + c_i · τ` (correct for time-varying
    /// generators). This is the accurate entry-point when driving a manual
    /// time-tracking loop.
    ///
    /// [`ChernoffFunction::apply_into`] uses `t_start = F::zero()`, which is
    /// correct for time-independent generators.
    ///
    /// # Errors
    /// Same as [`ChernoffFunction::apply_into`].
    pub fn apply_into_at(
        &self,
        t_start: F,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>
    where
        GraphSignal<F>: Clone,
    {
        apply_magnus_k4_into_at(self, t_start, tau, src, dst, scratch)
    }

    /// Evolve `f0 ↦ u(T_horizon)` along a piecewise-smooth trajectory.
    ///
    /// # Errors
    /// - `DomainViolation` if `n_steps_per_segment == 0`.
    /// - `DomainViolation` if `traj.breakpoints()[0] != 0`.
    /// - `DomainViolation` if `f0.n_nodes() != traj.snapshot(0).n_nodes()`.
    /// - `DomainViolation` if segment vertex counts differ.
    /// - `OutOfMagnusRadius` if `self.rho_bar_max * tau >= π/2`.
    pub fn evolve_with_traj(
        &self,
        traj: &GraphTraj<F>,
        n_steps_per_segment: usize,
        f0: &GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<GraphSignal<F>, SemiflowError>
    where
        GraphSignal<F>: Clone,
    {
        let mut dst = f0.clone();
        self.evolve_with_traj_into(traj, n_steps_per_segment, f0, &mut dst, scratch)?;
        Ok(dst)
    }

    /// Same as [`evolve_with_traj`](Self::evolve_with_traj) but writes into
    /// caller-supplied `dst`. Zero allocation in the steady-state loop.
    ///
    /// # Errors
    /// Same as [`evolve_with_traj`](Self::evolve_with_traj).
    pub fn evolve_with_traj_into(
        &self,
        traj: &GraphTraj<F>,
        n_steps_per_segment: usize,
        f0: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        validate_traj_inputs(traj, n_steps_per_segment, f0)?;
        dst.copy_from(f0);
        let n = f0.len();
        let graph_arc = f0.graph_arc();
        let cur_buf = scratch.take_graph_buf(n);
        let mut cur = GraphSignal::from_pool_buf(Arc::clone(&graph_arc), cur_buf);
        let result = run_all_segments(self, traj, n_steps_per_segment, dst, &mut cur, scratch);
        let (reclaim, _) = cur.into_pool_buf();
        scratch.return_graph_buf(reclaim);
        result
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> ChernoffFunction<F> for MagnusGraphHeatChernoff<F> {
    type S = GraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        // t_start = 0: consistent with ChernoffSemigroup which tracks no
        // absolute time.  For time-varying L_G(t), use apply_into_at instead.
        apply_magnus_k4_into_at(self, F::zero(), tau, src, dst, scratch)
    }

    /// Returns `4`: global Chernoff convergence rate `O(1/n⁴)`.
    fn order(&self) -> u32 {
        4
    }

    /// Returns `Growth { multiplier: 1, omega: ρ̄_max }`.
    fn growth(&self) -> Growth<F> {
        Growth {
            multiplier: F::one(),
            omega: self.rho_bar_max,
        }
    }
}
// NOTE: `apply_adjoint_into` is intentionally NOT overridden here.
// Each instantaneous L_G(t) is symmetric, but the Magnus exponent Ω₄(τ)
// accumulates an antisymmetric commutator term ~τ²·[L_i, L_j], so
// exp(Ω₄(τ))ᵀ ≠ exp(Ω₄(τ)) at O(τ²).  The default trait impl returns
// `SemiflowError::UnsupportedOperation`, which is the honest behaviour.
