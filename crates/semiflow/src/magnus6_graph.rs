//! Sixth-order Magnus expansion for time-dependent graph heat:
//! `∂_t u = −L_G(t) u` on a fixed-topology weighted graph.
//!
//! Three-point Gauss-Legendre quadrature (GL₆) + **2-commutator BCOR form**
//! (Blanes-Casas-Oteo-Ros 2009, canonical order-6 Magnus integrator):
//!
//! ```text
//! B₁ = τ · A₂
//! B₂ = τ · (√15/3) · (A₃ − A₁)
//! B₃ = τ · (10/3)  · (A₃ − 2·A₂ + A₁)
//!
//! C₁ = [B₁, B₂]
//! C₂ = −(1/60) · [B₁, 2·B₃ + C₁]
//!
//! Ω₆(τ) = B₁ + (1/12)·B₃ + (1/240)·[−20·B₁ − B₃ + C₁ , B₂ + C₂]
//! ```
//!
//! Local truncation error O(τ⁷); global order 6 on non-commuting A(t).
//! Only 2 commutator evaluations; cheaper than the old 4-commutator form.
//!
//! `A_i = −L_G(t₀ + c_i·τ)` at GL₆ abscissae:
//! `c₁ = (5−√15)/10,  c₂ = 1/2,  c₃ = (5+√15)/10`.
//!
//! **f64 ONLY** — `impl ChernoffFunction<f32>` is intentionally absent.
//! See ADR-0056 §"f32 instability rationale".
//!
//! # References
//!
//! - S. Blanes, F. Casas, J. A. Oteo, J. Ros, *Phys. Rep.* **470** (2009)
//!   §4 and Table 5 — canonical 2-commutator order-6 Magnus method.
//! - A. Iserles, H. Z. Munthe-Kaas, S. P. Nørsett, A. Zanna, *Acta Numerica*
//!   **9** (2000) §6, Table III — GL₆ quadrature abscissae and weights.
//! - M. Hochbruck, A. Ostermann, *Acta Numerica* **19** (2010) §3.
//!
//! See `contracts/semiflow-core.math.md` §16 (NORMATIVE) and ADR-0056, ADR-0114.
//!
//! # Zero-alloc steady state
//!
//! `apply_into` acquires **22 scratch buffers** via `ScratchPool::take_vec`
//! and returns them all before returning (R4 zero-alloc invariant, ADR-0114).

use alloc::sync::Arc;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    magnus_graph::LaplacianAtTime,
    scratch::ScratchPool,
    state::State,
};

#[path = "magnus6_graph_helpers.rs"]
mod helpers;

// ---------------------------------------------------------------------------
// GL₆ constants (NORMATIVE — DO NOT CHANGE; see contracts §16.1)
// ---------------------------------------------------------------------------

/// GL₆ first abscissa: `c₁ = (5 − √15) / 10 ≈ 0.11270166537925831`.
///
/// Source: Iserles+ 2000 *Acta Numerica* Table III; Blanes+ 2009 Table 6.
pub const GL6_C1: f64 = 0.112_701_665_379_258_31;

/// GL₆ second abscissa: `c₂ = 1/2`.
pub const GL6_C2: f64 = 0.5;

/// GL₆ third abscissa: `c₃ = (5 + √15) / 10 ≈ 0.88729833462074169`.
#[allow(clippy::excessive_precision)]
pub const GL6_C3: f64 = 0.887_298_334_620_741_7;

/// GL₆ first weight: `b₁ = 5/18`.
pub const GL6_B1: f64 = 5.0 / 18.0;

/// GL₆ second weight: `b₂ = 8/18`.
pub const GL6_B2: f64 = 8.0 / 18.0;

/// GL₆ third weight: `b₃ = 5/18`.
pub const GL6_B3: f64 = 5.0 / 18.0;

// ---------------------------------------------------------------------------
// BCOR-6 operator constants (NORMATIVE — ADR-0114 / math.md §16.2)
// ---------------------------------------------------------------------------

/// `√15 / 3` — scale for B₂ = τ·(√15/3)·(A₃ − A₁).
#[allow(clippy::excessive_precision)]
pub(crate) const SQRT15_OVER_3: f64 = 1.290_994_448_735_805_6;

/// `10/3` — scale for B₃ = τ·(10/3)·(A₃ − 2·A₂ + A₁).
pub(crate) const TEN_OVER_3: f64 = 10.0 / 3.0;

/// `1/12` — coefficient of B₃ in Ω₆.
pub(crate) const ONE_OVER_12: f64 = 1.0 / 12.0;

/// `1/60` — coefficient in C₂ = −(1/60)·[B₁, 2·B₃ + C₁].
pub(crate) const ONE_OVER_60: f64 = 1.0 / 60.0;

/// `1/240` — outer bracket coefficient in Ω₆.
pub(crate) const ONE_OVER_240: f64 = 1.0 / 240.0;

// ---------------------------------------------------------------------------
// MagnusGraphHeat6thChernoff<F>
// ---------------------------------------------------------------------------

/// Order-6 Magnus Chernoff for `∂_t u = −L_G(t) u` on a fixed-topology
/// weighted graph with time-varying edge weights.
///
/// Uses the canonical Blanes-Casas-Oteo-Ros 2-commutator form (BCOR-6,
/// Phys. Rep. 470 2009 §4) with GL₆ quadrature. Local truncation error
/// O(τ⁷); global order 6 on genuinely non-commuting `L_G(t)`.
///
/// **f64 ONLY** — `impl ChernoffFunction<f32>` intentionally absent (ADR-0056).
pub struct MagnusGraphHeat6thChernoff<F: SemiflowFloat = f64> {
    /// Fixed-topology graph.
    graph: Arc<Graph<F>>,
    /// `t ↦ Arc<Laplacian<F>>` — caller-supplied sampler.
    lap_at_t: LaplacianAtTime<F>,
    /// Gershgorin radius bound `ρ̄_max`.
    rho_bar_max: F,
    /// If `true`, each `apply_into` checks `ρ̄_max · τ < π/2`.
    convergence_radius_check: bool,
}

impl<F: SemiflowFloat> MagnusGraphHeat6thChernoff<F> {
    /// Construct from topology + time-to-Laplacian closure.
    ///
    /// # Errors
    ///
    /// [`SemiflowError::DomainViolation`] if `rho_bar_max` is not finite and
    /// strictly positive, or if `graph.n_nodes() == 0`.
    pub fn new(
        graph: Arc<Graph<F>>,
        lap_at_t: LaplacianAtTime<F>,
        rho_bar_max: F,
        convergence_radius_check: bool,
    ) -> Result<Self, SemiflowError> {
        validate_rho(rho_bar_max)?;
        if graph.n_nodes() == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "MagnusGraphHeat6thChernoff: graph must have at least one node",
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

    /// Return the Laplacian sampled at time `t` (clones the `Arc`).
    #[must_use]
    pub fn laplacian_at(&self, t: F) -> Arc<Laplacian<F>> {
        (self.lap_at_t)(t)
    }

    /// Apply one Magnus K=6 step starting at absolute time `t_start`.
    ///
    /// GL₆ nodes are `t_start + c_i · τ` — correct for time-varying generators.
    ///
    /// # Errors
    ///
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
        apply_magnus_k6_into_at(self, t_start, tau, src, dst, scratch)
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<f64> impl — f64 ONLY (no f32 impl; see ADR-0056)
// ---------------------------------------------------------------------------

/// **f64 ONLY** — building `MagnusGraphHeat6thChernoff::<f32>` as a
/// `ChernoffFunction<f32>` is intentionally not supported (ADR-0056).
impl ChernoffFunction<f64> for MagnusGraphHeat6thChernoff<f64> {
    type S = GraphSignal<f64>;

    /// Apply one Magnus K=6 step from `t_start = 0`.
    ///
    /// **WARNING**: hardcodes `t_start = 0` — for time-varying `L(t)` call
    /// `apply_into_at` instead (see ADR-0114).
    fn apply_into(
        &self,
        tau: f64,
        src: &GraphSignal<f64>,
        dst: &mut GraphSignal<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        apply_magnus_k6_into_at(self, 0.0_f64, tau, src, dst, scratch)
    }

    fn order(&self) -> u32 {
        6
    }

    fn growth(&self) -> Growth<f64> {
        Growth {
            multiplier: 1.0_f64,
            omega: self.rho_bar_max,
        }
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_rho<F: SemiflowFloat>(rho: F) -> Result<(), SemiflowError> {
    let v = rho.to_f64().unwrap_or(f64::NAN);
    if !v.is_finite() || v <= 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "MagnusGraphHeat6thChernoff: rho_bar_max must be finite and > 0",
            value: v,
        });
    }
    Ok(())
}

fn validate_tau<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    let v = tau.to_f64().unwrap_or(f64::NAN);
    if !v.is_finite() || v < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "MagnusGraphHeat6thChernoff: tau must be finite and >= 0",
            value: v,
        });
    }
    Ok(())
}

fn validate_magnus_radius<F: SemiflowFloat>(rho_bar_max: F, tau: F) -> Result<(), SemiflowError> {
    let radius = rho_bar_max * tau;
    let half_pi = from_f64::<F>(core::f64::consts::FRAC_PI_2);
    if radius >= half_pi {
        return Err(SemiflowError::OutOfMagnusRadius {
            tau: tau.to_f64().unwrap_or(f64::NAN),
            rho_estimate: rho_bar_max.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core kernel
// ---------------------------------------------------------------------------

/// Apply `exp(Ω₆(t_start, τ)) · src` into `dst`.
fn apply_magnus_k6_into_at<F: SemiflowFloat>(
    mc: &MagnusGraphHeat6thChernoff<F>,
    t_start: F,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    validate_tau(tau)?;
    if mc.convergence_radius_check {
        validate_magnus_radius(mc.rho_bar_max, tau)?;
    }
    let n = src.len();
    debug_assert_eq!(dst.len(), n, "apply_magnus_k6_into_at: dst len mismatch");
    let c1 = from_f64::<F>(GL6_C1);
    let c2 = from_f64::<F>(GL6_C2);
    let c3 = from_f64::<F>(GL6_C3);
    let lap1 = (mc.lap_at_t)(t_start + c1 * tau);
    let lap2 = (mc.lap_at_t)(t_start + c2 * tau);
    let lap3 = (mc.lap_at_t)(t_start + c3 * tau);
    #[cfg(debug_assertions)]
    {
        debug_assert_eq!(
            lap1.n_nodes(),
            mc.graph.n_nodes(),
            "MagnusGraphHeat6thChernoff: topology drift at c1"
        );
        debug_assert_eq!(
            lap2.n_nodes(),
            mc.graph.n_nodes(),
            "MagnusGraphHeat6thChernoff: topology drift at c2"
        );
        debug_assert_eq!(
            lap3.n_nodes(),
            mc.graph.n_nodes(),
            "MagnusGraphHeat6thChernoff: topology drift at c3"
        );
    }
    helpers::apply_exp_omega6_kernel(&lap1, &lap2, &lap3, tau, src, dst, scratch);
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit smoke tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "magnus6_graph_tests.rs"]
mod tests;
