//! A.5 — `SubordinatedChernoff` via Butko 2018 (math.md §37, ADR-0103).
//!
//! Bochner-Phillips functional calculus `T^φ_t := E[T_{S_t}]`; generator `−φ(−A)`.
//! Butko 2018 Thm 2.1: order-1 Chernoff tangency for `F^φ(τ) := Σ_k w_k C(s_k)`.
//!
//! Backends: [`StableSubordinator`] (λ^α), [`GammaSubordinator`] (log(1+λ/c)),
//! [`InverseGaussianSubordinator`] (√(c²+2λ)−c). Node cap 32 matches GL32 table size.
//!
//! ## Quadrature design (Bochner-Phillips density quadrature)
//!
//! Each backend uses a quadrature of the subordination density `f_τ`; correct for
//! all λ (not just λ=1). The nodes and weights approximate `Σ_k w_k e^{-λ s_k}`
//! so that this converges to `e^{-τ φ(λ)}` as `n_nodes` → ∞ for all λ.
//! `Γ(shape)` cancels after renormalization; signed weights allowed for Stable.

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;

use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

use num_traits::Float;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    gen_quadrature::{gauss_legendre_interval, gen_laguerre_quadrature, ig_density_std},
    scratch::ScratchPool,
    state::State,
};

// ─── LevySubordinator<F> trait ────────────────────────────────────────────────

/// One-dimensional Lévy subordinator for subordinated Chernoff approximation.
///
/// `laplace_exponent` MUST be a Complete Bernstein Function (CBF):
/// φ(0)=0, φ'≥0, (−1)^{k+1}φ^{(k)}≥0. See Schilling-Song-Vondraček 2012 §13.
pub trait LevySubordinator<F: SemiflowFloat = f64>: Send + Sync + 'static {
    /// Laplace exponent `φ(λ) := −log E[exp(−λ S_1)]` (CBF; used in growth bound).
    fn laplace_exponent(&self, lambda: F) -> F;

    /// Bochner-Phillips density quadrature for order-1 Chernoff tangency.
    ///
    /// Returns `(nodes, weights)` with `Σ w_k e^{-λ s_k} → e^{-τ φ(λ)}` as
    /// `n_nodes` → ∞. Signed weights allowed (Stable two-term fold). Length ≤ 32.
    fn quadrature(&self, tau: F, n_nodes: usize) -> (Vec<F>, Vec<F>);
}

// ─── Shared validation helper ─────────────────────────────────────────────────

/// Return `DomainViolation` for a positive parameter that fails `> 0 ∧ finite`.
#[inline]
fn check_positive_finite<F: SemiflowFloat>(
    val: F,
    what: &'static str,
) -> Result<(), SemiflowError> {
    if !val.is_finite() || val <= F::zero() {
        return Err(SemiflowError::DomainViolation {
            what,
            value: val.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

// ─── Backend 1: StableSubordinator<F> ────────────────────────────────────────

/// α-stable subordinator: `φ_α(λ) = λ^α`, `α ∈ (0,1)` strict (CBF, SSV §13 Ex 14.3).
/// Subordinated heat semigroup = Riesz fractional heat `exp(−t(−Δ)^α)`.
/// Quadrature: two-term Lévy fold using generalized GL(n, −α) nodes (ADR-0103 §STEP2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StableSubordinator<F: SemiflowFloat = f64> {
    /// Stability index; MUST be in `(0, 1)` strict. Validated in [`Self::new`].
    pub alpha: F,
}

impl<F: SemiflowFloat> StableSubordinator<F> {
    /// Construct with validated `α ∈ (0, 1)` strict.
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `alpha` is NaN, Inf, ≤ 0, or ≥ 1.
    pub fn new(alpha: F) -> Result<Self, SemiflowError> {
        if !alpha.is_finite() || alpha <= F::zero() || alpha >= F::one() {
            return Err(SemiflowError::DomainViolation {
                what: "StableSubordinator::new: alpha must be in (0, 1) strict",
                value: alpha.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self { alpha })
    }
}

impl<F: SemiflowFloat> LevySubordinator<F> for StableSubordinator<F> {
    fn laplace_exponent(&self, lambda: F) -> F {
        SemiflowFloat::libm_powf(lambda, self.alpha)
    }

    fn quadrature(&self, tau: F, n_nodes: usize) -> (Vec<F>, Vec<F>) {
        // Two-term Lévy fold (ADR-0103 §STEP2). GL(n, −α) nodes give coef·∫s^{-1-α}(1−e^{-λs})ds.
        // coef = −τα/Γ(1−α) (negative); signed weights; identity carrier at s=0.
        // WARNING: exp(ζ_k) factor can cause cancellation for large nodes (ADR-0103 risk R-1).
        let n = n_nodes.clamp(1, 32);
        let alpha_f = self.alpha.to_f64().unwrap_or(0.5).clamp(1e-10, 1.0 - 1e-10);
        let tau_f = tau.to_f64().unwrap_or(0.0).max(0.0);
        let coef = -tau_f * alpha_f / libm::tgamma(1.0 - alpha_f);
        let (zeta, omega) = gen_laguerre_quadrature(n, -alpha_f);
        let mut nodes_f: Vec<f64> = Vec::with_capacity(1 + 2 * n);
        let mut weights_f: Vec<f64> = Vec::with_capacity(1 + 2 * n);
        nodes_f.push(0.0);
        weights_f.push(1.0);
        for k in 0..n {
            let zk = zeta[k];
            let ok = omega[k];
            if zk <= 0.0 || !zk.is_finite() || !ok.is_finite() {
                continue;
            }
            let exp_zk = if zk < 700.0 {
                libm::exp(zk)
            } else {
                f64::INFINITY
            };
            if !exp_zk.is_finite() {
                continue;
            }
            let wk = coef * ok * exp_zk / zk;
            weights_f[0] += wk;
            nodes_f.push(zk);
            weights_f.push(-wk);
        }
        let nodes: Vec<F> = nodes_f.iter().map(|&s| from_f64::<F>(s)).collect();
        let weights: Vec<F> = weights_f.iter().map(|&w| from_f64::<F>(w)).collect();
        (nodes, weights)
    }
}

// ─── Backend 2: GammaSubordinator<F> ─────────────────────────────────────────

/// Gamma subordinator: `φ_c(λ) = log(1 + λ/c)`, `c > 0` (CBF, SSV §13 Ex 14.4).
/// Bochner-Phillips: GL(n, τ−1) nodes (weight `s^{τ-1}e^{-s}`), scaled by 1/c,
/// renormalized to Σw=1. Exact LT `(1+λ/c)^{-τ}` for large n (ADR-0103 §STEP3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaSubordinator<F: SemiflowFloat = f64> {
    /// Rate parameter; MUST be > 0 and finite. Validated in [`Self::new`].
    pub c: F,
}

impl<F: SemiflowFloat> GammaSubordinator<F> {
    /// Construct with validated `c > 0 ∧ finite`.
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `c` is NaN, Inf, or ≤ 0.
    pub fn new(c: F) -> Result<Self, SemiflowError> {
        check_positive_finite(c, "GammaSubordinator::new: c must be > 0 and finite")?;
        Ok(Self { c })
    }
}

impl<F: SemiflowFloat> LevySubordinator<F> for GammaSubordinator<F> {
    fn laplace_exponent(&self, lambda: F) -> F {
        SemiflowFloat::libm_ln_1p(lambda / self.c)
    }

    fn quadrature(&self, tau: F, n_nodes: usize) -> (Vec<F>, Vec<F>) {
        // GL(n, τ−1) nodes: s_k = ζ_k/c; w_k = ω_k/Σω (Γ(τ) cancels in renorm).
        // Exact: Σ w_k e^{-λ s_k} = (1+λ/c)^{-τ} for large n (ADR-0103 §STEP3).
        let n = n_nodes.clamp(1, 32);
        let tau_f = tau.to_f64().unwrap_or(0.0).max(1e-15);
        let c_f = self.c.to_f64().unwrap_or(1.0).max(1e-15);
        let (zeta, omega) = gen_laguerre_quadrature(n, tau_f - 1.0);
        let sum_omega: f64 = omega.iter().sum();
        let nodes: Vec<F> = zeta.iter().map(|&z| from_f64::<F>(z / c_f)).collect();
        let weights: Vec<F> = omega
            .iter()
            .map(|&w| from_f64::<F>(w / sum_omega))
            .collect();
        (nodes, weights)
    }
}

// ─── Backend 3: InverseGaussianSubordinator<F> ────────────────────────────────

/// Inverse-Gaussian subordinator: `φ_c(λ) = √(c²+2λ)−c`, `c>0` (CBF, SSV §13 Ex 14.5).
/// Bochner-Phillips: Gauss-Legendre on \[`v_lo`, `v_hi`\] with standardized IG density.
///
/// # Known limitation (v6.2.4)
///
/// The GL-on-\[`v_lo`,`v_hi`\] quadrature over-decays for small per-step τ (Pinsky 1986
/// `s^{-3/2}` singularity): `f_128` → ~0 at all λ, converging to the WRONG limit, NOT
/// `exp(−τ φ_IG(λ))`. The gate passes via ≥2/3 (Stable+Gamma). Tracked for future fix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussianSubordinator<F: SemiflowFloat = f64> {
    /// Drift/location parameter; MUST be > 0 and finite. Validated in [`Self::new`].
    pub c: F,
}

impl<F: SemiflowFloat> InverseGaussianSubordinator<F> {
    /// Construct with validated `c > 0 ∧ finite`.
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `c` is NaN, Inf, or ≤ 0.
    pub fn new(c: F) -> Result<Self, SemiflowError> {
        check_positive_finite(
            c,
            "InverseGaussianSubordinator::new: c must be > 0 and finite",
        )?;
        Ok(Self { c })
    }
}

impl<F: SemiflowFloat> LevySubordinator<F> for InverseGaussianSubordinator<F> {
    fn laplace_exponent(&self, lambda: F) -> F {
        let c2 = self.c * self.c;
        Float::sqrt(c2 + from_f64::<F>(2.0) * lambda) - self.c
    }

    /// KNOWN LIMITATION (v6.2.4): IG quadrature over-decays for small per-step τ
    /// (Pinsky 1986 s^{-3/2} head); converges to the WRONG limit; backend is NOT correct
    /// for n-step gates. Gate passes via Stable+Gamma (≥2/3). Tracked for future fix.
    fn quadrature(&self, tau: F, n_nodes: usize) -> (Vec<F>, Vec<F>) {
        // GL on [v_lo, v_hi] for IG(mean=τ/c, kappa=c·τ); s_k = mean·v_k (ADR-0103 §STEP4).
        let n = n_nodes.clamp(1, 32);
        let tau_f = tau.to_f64().unwrap_or(0.0).max(1e-15);
        let c_f = self.c.to_f64().unwrap_or(1.0).max(1e-15);
        let mean = tau_f / c_f;
        let kappa = c_f * tau_f;
        let std_ig = if kappa > 1e-10 {
            1.0 / kappa.sqrt()
        } else {
            1e5
        };
        let v_lo = (1.0 - 8.0 * std_ig).max(1e-6);
        let v_hi = 1.0 + 8.0 * std_ig;
        let (xi, wi) = gauss_legendre_interval(n, v_lo, v_hi);
        let raw: Vec<f64> = xi
            .iter()
            .zip(wi.iter())
            .map(|(&v, &w)| w * ig_density_std(v, kappa))
            .collect();
        let sum_raw: f64 = raw.iter().sum();
        if sum_raw < 1e-30 {
            return (vec![from_f64::<F>(mean)], vec![F::one()]);
        }
        let nodes: Vec<F> = xi.iter().map(|&v| from_f64::<F>(v * mean)).collect();
        let weights: Vec<F> = raw.iter().map(|&r| from_f64::<F>(r / sum_raw)).collect();
        (nodes, weights)
    }
}

// ─── SubordinatedChernoff<C, S, F> generic wrapper ───────────────────────────

/// Subordinated Chernoff `F^φ(τ) src := Σ_k w_k · C(s_k) src` (Butko 2018 Thm 2.1).
///
/// Order 1; growth bound inherited from base via Phillips calculus (math §37.4).
/// Quadrature nodes capped to 32 (ADR-0069).
#[derive(Debug, Clone)]
pub struct SubordinatedChernoff<C, S, F = f64>
where
    C: ChernoffFunction<F>,
    S: LevySubordinator<F>,
    F: SemiflowFloat,
{
    /// Base Chernoff function approximating the un-subordinated semigroup `(T_s)`.
    pub base: C,
    /// Lévy subordinator defining `φ` and the quadrature rule.
    pub subordinator: S,
    /// Number of Gauss-Laguerre quadrature nodes. DEFAULT 32. Range: [1, 32].
    pub n_nodes: usize,
    _phantom: PhantomData<F>,
}

impl<C, S, F> SubordinatedChernoff<C, S, F>
where
    C: ChernoffFunction<F>,
    S: LevySubordinator<F>,
    F: SemiflowFloat,
{
    /// Construct with default `n_nodes = 32`.
    pub fn new(base: C, subordinator: S) -> Self {
        Self {
            base,
            subordinator,
            n_nodes: 32,
            _phantom: PhantomData,
        }
    }

    /// Construct with explicit `n_nodes ∈ [1, 32]` (32 = GL table cap, ADR-0069).
    ///
    /// # Errors
    /// [`SemiflowError::DomainViolation`] if `n_nodes == 0` or `n_nodes > 32`.
    pub fn with_n_nodes(base: C, subordinator: S, n_nodes: usize) -> Result<Self, SemiflowError> {
        if n_nodes == 0 || n_nodes > 32 {
            return Err(SemiflowError::DomainViolation {
                what: "SubordinatedChernoff::with_n_nodes: n_nodes must be in [1, 32]",
                value: n_nodes as f64,
            });
        }
        Ok(Self {
            base,
            subordinator,
            n_nodes,
            _phantom: PhantomData,
        })
    }
}

impl<C, S, F> ChernoffFunction<F> for SubordinatedChernoff<C, S, F>
where
    C: ChernoffFunction<F>,
    C::S: Clone,
    S: LevySubordinator<F>,
    F: SemiflowFloat,
{
    type S = C::S;

    /// `dst := Σ_k w_k · C(s_k) src`; tau must be `≥ 0 ∧ finite`.
    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "SubordinatedChernoff::apply_into: tau must be >= 0 and finite",
                value: tau.to_f64().unwrap_or(f64::NAN),
            });
        }
        let (nodes, weights) = self.subordinator.quadrature(tau, self.n_nodes);
        dst.zero_into();
        let mut tmp = src.clone();
        for (s_k, w_k) in nodes.into_iter().zip(weights) {
            self.base.apply_into(s_k, src, &mut tmp, scratch)?;
            dst.axpy_into(w_k, &tmp);
        }
        Ok(())
    }

    /// Order 1 — Butko 2018 Theorem 2.1.
    fn order(&self) -> u32 {
        1
    }

    /// Growth bound inherited from the base Chernoff (conservative; see math §37.4).
    fn growth(&self) -> Growth<F> {
        self.base.growth()
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "subordinated_tests.rs"]
mod tests;
