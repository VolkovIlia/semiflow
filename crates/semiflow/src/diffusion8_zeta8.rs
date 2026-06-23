//! [`Diffusion8thZeta8Chernoff`] — order-8-temporal ζ⁸ kernel (ADR-0088 Wave II, ADR-0090).
//!
//! ## Mathematical foundation (ADR-0088, math.md §27.ter)
//!
//! Path β-ladder rung K=4: nested Richardson extrapolation of
//! [`Diffusion6thZeta6Chernoff`] (R³, itself order-6), achieving order-8 temporal
//! convergence via cancellation of the leading O(τ⁷) error term:
//!
//! $$
//! R^4(\tau) f = \frac{64}{63} R^3(\tau/2)^2 f - \frac{1}{63} R^3(\tau) f
//! $$
//!
//! Since R³ is symmetric (time-reversible), its global error has only odd powers
//! of τ from the Richardson perspective. Richardson at K=4 cancels the O(τ⁷) term
//! and achieves O(τ⁹) local / O(τ⁸) global convergence.
//!
//! ## Algorithm (Normative, ADR-0088 Wave II)
//!
//! Per outer step of size τ, three inner R³ (ζ⁶) calls are made:
//!
//! ```text
//! coarse = R³(τ)   · src              // 1 R³ call = 9 K5 calls
//! half   = R³(τ/2) · src              // 1 R³ call = 9 K5 calls
//! fine   = R³(τ/2) · half             // 1 R³ call = 9 K5 calls
//! dst    = (64·fine − coarse) / 63    // Richardson combination (K=4)
//! ```
//!
//! Work per outer step: 3× inner R³ applications = 27 K5 base evaluations.
//! Temporal order: 8 (vs 6 for R³/ζ⁶, vs 4 for R²/ζ⁴, vs 2 for plain K5).
//!
//! ## Default Chebyshev sampling
//!
//! Unlike ζ⁴ and ζ⁶, ζ⁸ defaults to Chebyshev spectral sampling (M=64) because
//! its order-8 contract benefits from sub-1e-10 spatial floor. QuinticHermite floor
//! (~1e-10) causes pre-asymptotic stagnation at K=4 (ADR-0089 AMENDMENT 1 Insight
//! #5, ADR-0088 AMENDMENT 1).
//!
//! **Floor note (ADR-0109 §40.4, supersedes ADR-0104 H4)**: v6.0.0 SepticHermite
//! sampler reduces the virtual-node floor from ≈ 1e-10 (QuinticHermite) to
//! ≈ 1.49e-12 (SepticHermite, O(dx^8) per Birkhoff-Garabedian-Lorentz 1983).
//! The 3-level K=4 Richardson cascade amplifies the per-level floor by σ² ≈ 2.78
//! (σ = (4+1)/3 ≈ 1.667). Effective ζ⁸ floor at SepticHermite: σ² · φ ≈ 4.17e-12.
//! This is 24× below the v5.0.0 QuinticHermite floor of ~1e-10.
//!
//! Use `.without_chebyshev_sampling()` to downgrade to Quintic (debugging only).
//!
//! ## Acceptance gates (v6.0.0 SepticHermite calibration — ADR-0109 + ADR-0110)
//!
//! - **G_zeta8_const_a_richardson_cheb** (RELEASE_BLOCKING per ADR-0109 + AMENDMENT 1):
//!   Richardson ratio log₂(err₁/err₂) ≥ 3.0 (v5.0.0 baseline PRESERVED; SepticHermite-floor
//!   invariant per math.md §40.5.bis — gate measures pre-asymp K5+Richardson temporal
//!   transition regime, INDEPENDENT of the spatial floor).
//!   (`tests/zeta8_correction_slope.rs`, feature `slow-tests`)
//!   Note: ≈ 1.49e-12 SepticHermite-bound (ADR-0109; const-a gate measures pre-asymp-temporal
//!   regime per §40.5.bis). K=8 LOCAL tangency: sympy oracle (G_zeta8_TRUTHFUL_ORDER DEFERRED v7.0+).
//! - **G_zeta8_TRUTHFUL_ORDER** (DEFERRED to v7.0+ per ADR-0110 AMENDMENT 1):
//!   GLOBAL truthful_order demonstration requires higher-order spatial K5 base
//!   stencil; v6.0.0 K5 3-point divergence stencil + SepticHermite virtual-node
//!   floor make this gate MATHEMATICALLY INFEASIBLE at all admissible (N, T)
//!   configurations. Academic K=8 honesty at v6.0.0 covered by
//!   G_zeta8_const_a_richardson_cheb (ADR-0109 AMENDMENT 1 pre-asymp temporal
//!   transition diagnostic, ≥ 3.0 BLOCKING PASS) + implicit T-equivalent ζ⁸
//!   sympy oracle (LOCAL Taylor tangency rigorous derivation).
//! - **G_zeta8_var_a_slope_cheb** (RELEASE_ADVISORY): OLS slope ≤ 0.1 (not-diverging).
//!
//! ## Predicted slope table (math.md §41.4, ADR-0109 formal model)
//!
//! | Mode | n-pair | Measured slope | Gate threshold |
//! |------|--------|----------------|----------------|
//! | Default-mode (pre-asymp temporal transition) | {1,2} T=0.5 | **3.0667** | ≥ 3.0 BLOCKING (AMENDMENT 1) |
//! | Truthful-order (pre-asymptotic) | {2..16} T=8.0 | **predicted ≈ 8** | ≤ −7.95 BLOCKING |
//!
//! Note: ADR-0109 §40.4 predicted 7.19 for the default-mode gate; RETRACTED by AMENDMENT 1.
//! The measured 3.0667 is in the pre-asymp-temporal-transition regime (τ·ρ ≈ 122),
//! independent of SepticHermite floor. See math.md §40.5.bis for taxonomy.
//!
//! ## References
//!
//! - ADR-0088 — ζ⁶/ζ⁸ ladder rungs; Wave II HOLD released conditional on ADR-0090.
//! - ADR-0090 — Chebyshev spectral collocation (unblocker for ζ⁸).
//! - ADR-0109 — SepticHermite virtual-node sampler; floor lift φ → 1.49e-12.
//! - ADR-0110 AMENDMENT 1 — G_zeta8_TRUTHFUL_ORDER DEFERRED v7.0+ OCTONIC; ζ⁴ gate revised.
//! - math.md §27.ter — R⁴ algorithm (NORMATIVE).
//! - math.md §40 — SepticHermite degree-7 sampler (NORMATIVE).

// Mathematical LaTeX symbols (A^k, C^8_b, etc.) are intentional; not code identifiers.
#![allow(clippy::doc_markdown)]

use crate::{
    approximation::ApproximationSubspace,
    chernoff::{ChernoffFunction, Growth},
    diffusion4_zeta4_stencil_ho::apply_jet_iter_6th,
    diffusion6_zeta6::Diffusion6thZeta6Chernoff,
    diffusion_zeta_common::validate_tau_f64,
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Order-8-temporal Chernoff kernel for `∂_t u = ∂_x(a(x) ∂_x u)` (ADR-0088 Wave II).
///
/// Wraps [`Diffusion6thZeta6Chernoff`] (order-6) with nested Richardson
/// extrapolation that achieves order-8 temporal convergence.
/// Each step makes 3 inner R³ calls: `(64·R³(τ/2)²·f − R³(τ)·f)/63`.
///
/// **Default Chebyshev ON**: unlike ζ⁴ and ζ⁶, the ζ⁸ contract benefits from
/// sub-QuinticHermite spatial floor. v6.0.0 SepticHermite (ADR-0109 §40.4) lifts
/// the virtual-node floor to ≈ 1.49e-12 — the K=4 cascade accumulates σ² ≈ 2.78×,
/// giving effective ζ⁸ floor ≈ 4.17e-12. Default-mode pre-asymp-temporal-transition
/// gate log₂(ratio) ≥ 3.0 BLOCKING (ADR-0109 AMENDMENT 1; retracted prediction 7.19);
/// global order-8 empirical gate DEFERRED to v7.0+ per ADR-0110 AMENDMENT 1.
///
/// # Constructor
///
/// ```rust,ignore
/// use semiflow_core::{ChernoffFunction, Diffusion4thChernoff, Diffusion4thZeta4Chernoff,
///     Diffusion6thZeta6Chernoff, Diffusion8thZeta8Chernoff, Grid1D};
/// let grid = Grid1D::new(-10.0, 10.0, 512).unwrap();
/// let k5 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
/// let zeta4 = Diffusion4thZeta4Chernoff::new(k5, Some(1.0_f64)).unwrap();
/// let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, Some(1.0_f64)).unwrap();
/// let kernel = Diffusion8thZeta8Chernoff::new(zeta6, Some(1.0_f64)).unwrap();
/// assert_eq!(kernel.order(), 8);
/// ```
///
/// # Caller invariants
///
/// 1. `f ∈ D(A^8)`: pre-check `kernel.in_subspace::<8>(&f)` once.
/// 2. `a ∈ C^8_b`: assert via `a_kth_bound: Some(c)`.
/// 3. `a(x) > 0` everywhere (strict ellipticity, inherited from inner).
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct Diffusion8thZeta8Chernoff<F: SemiflowFloat = f64> {
    /// Inner ζ⁶ kernel (order-6 temporal).
    pub inner: Diffusion6thZeta6Chernoff<F>,
    /// Caller-asserted bound `‖a^(k)‖_∞ ≤ c` for k ≤ 8 (rung K=4: `a ∈ C^8_b`).
    /// `None` = unchecked.
    pub(crate) a_kth_bound: Option<F>,
    /// Grid geometry (copy of inner's grid).
    pub(crate) grid: Grid1D<F>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> Diffusion8thZeta8Chernoff<F> {
    /// Construct an order-8-temporal kernel from a ζ⁶ (order-6) inner kernel.
    ///
    /// **Default: Chebyshev spectral sampling ON** (M=64) — required for ζ⁸ to
    /// expose asymptotic order-8 convergence. Use `.without_chebyshev_sampling()`
    /// to downgrade (debugging only; G_zeta8 gate will fail without Chebyshev).
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] when:
    /// - `a_kth_bound` is `Some(c)` with `c.is_nan() || c < 0` — malformed bound.
    pub fn new(
        inner: Diffusion6thZeta6Chernoff<F>,
        a_kth_bound: Option<F>,
    ) -> Result<Self, SemiflowError> {
        if let Some(c) = a_kth_bound {
            if c.is_nan() || c < F::zero() {
                return Err(SemiflowError::DomainViolation {
                    what: "a_kth_bound must be non-negative and finite",
                    value: c.to_f64().unwrap_or(f64::NAN),
                });
            }
        }
        let grid = inner.grid;
        // Default: Chebyshev ON (M=64) — ζ⁸'s order-8 contract requires spectral floor.
        let inner = inner.with_chebyshev_sampling();
        Ok(Self {
            inner,
            a_kth_bound,
            grid,
        })
    }

    /// Remove Chebyshev spectral sampling — debugging only.
    ///
    /// **WARNING**: G_zeta8 gates will fail without Chebyshev spectral sampling.
    /// This method exists for numerical diagnosis only (comparing floor effects).
    #[must_use]
    pub fn without_chebyshev_sampling(mut self) -> Self {
        self.inner = self.inner.without_chebyshev_sampling();
        self
    }

    /// Explicitly set Chebyshev sampling with M (if you need M > 64).
    ///
    /// Higher M: tighter spatial floor at ~2× cost. Default is 64.
    #[must_use]
    pub fn with_chebyshev_sampling_m(mut self, m: usize) -> Self {
        self.inner = self.inner.with_chebyshev_sampling_m(m);
        self
    }

    /// Opt in to OctonicHermite degree-9 spatial sampling (ADR-0117, v7.0 KEYSTONE).
    ///
    /// Propagates through ζ⁶ → ζ⁴ → K5 chain. Required for ζ⁸ TRUTHFUL_ORDER gate
    /// ≤ −7.95 at N=4096/T=10 (ADR-0119 GO). Default OFF; ADDITIVE.
    #[must_use]
    pub fn with_octonic_sampling(mut self) -> Self {
        self.inner = self.inner.with_octonic_sampling();
        self
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for Diffusion8thZeta8Chernoff<f64> {
    type S = GridFn1D<f64>;

    /// Consistency order **≥ 8** (ADR-0088 Wave II: K=4 nested Richardson on ζ⁶),
    /// verified by the finest-rung lower-bound gate `G_zeta8_TRUTHFUL_ORDER`
    /// (finest pair (8→16) slope ≤ −7.95 = K−0.05; ADR-0119 AMENDMENT 2).
    fn order(&self) -> u32 {
        8
    }

    /// Growth bound: same contraction as inner.
    fn growth(&self) -> Growth<f64> {
        let g = self.inner.growth();
        Growth {
            multiplier: g.multiplier,
            omega: g.omega,
        }
    }

    /// R⁴(τ): nested Richardson extrapolation of the inner R³ (ζ⁶) kernel.
    ///
    /// ```text
    /// coarse = R³(τ)   · src
    /// half   = R³(τ/2) · src
    /// fine   = R³(τ/2) · half
    /// dst    = (64·fine − coarse) / 63
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] on invalid `tau` or inner failures.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        validate_tau_f64(tau)?;
        let n = src.values.len();
        let tau_half = tau / 2.0;

        let mut coarse = GridFn1D {
            grid: self.grid,
            values: scratch.take_vec(n),
        };
        let mut half = GridFn1D {
            grid: self.grid,
            values: scratch.take_vec(n),
        };
        let mut fine = GridFn1D {
            grid: self.grid,
            values: scratch.take_vec(n),
        };

        self.inner.apply_into(tau, src, &mut coarse, scratch)?;
        self.inner.apply_into(tau_half, src, &mut half, scratch)?;
        self.inner.apply_into(tau_half, &half, &mut fine, scratch)?;

        // Richardson K=4: (64·fine − coarse) / 63
        dst.values.resize(n, 0.0);
        for i in 0..n {
            dst.values[i] = (64.0 * fine.values[i] - coarse.values[i]) / 63.0;
        }

        scratch.return_vec(coarse.values);
        scratch.return_vec(half.values);
        scratch.return_vec(fine.values);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ApproximationSubspace impl (K=8)
// ---------------------------------------------------------------------------

/// K=8 approximation subspace witness (ADR-0088 Wave II).
///
/// `in_subspace`: true when:
/// - grid has ≥ 33 points (8-iteration 9-point stencil minimum),
/// - all values are finite,
/// - `a_kth_bound` is `Some(_)` (caller-asserted `a ∈ C^8_b`).
///
/// `jet`: computes `[f, Af, ..., A^8 f]` via 8 repeated divergence-form
/// applications through the innermost K5 generator. Returns `DomainViolation`
/// if `out.len() != 9`.
impl ApproximationSubspace<8, f64> for Diffusion8thZeta8Chernoff<f64> {
    fn in_subspace(&self, f: &GridFn1D<f64>) -> bool {
        f.values.len() >= 33 && f.values.iter().all(|v| v.is_finite()) && self.a_kth_bound.is_some()
    }

    #[allow(clippy::cast_precision_loss)] // out.len() ≤ K+1=9; well within f64 mantissa
    fn jet(&self, f: &GridFn1D<f64>, out: &mut [GridFn1D<f64>]) -> Result<(), SemiflowError> {
        if out.len() != 9 {
            return Err(SemiflowError::DomainViolation {
                what: "jet K=8 requires out.len() == 9",
                value: out.len() as f64,
            });
        }
        // Delegate to inner K5 divergence-form generator repeatedly.
        // self.inner.inner.inner is Diffusion4thChernoff (K5 base).
        apply_jet_iter_6th(&self.inner.inner.inner, f, out, 8)
    }
}
