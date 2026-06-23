//! [`Diffusion4thZeta4Chernoff`] — order-4-temporal ζ⁴ kernel (ADR-0086 Path β resolution).
//!
//! ## Mathematical foundation (ADR-0086, math.md §27 AMENDMENT)
//!
//! Path β: single-step Richardson extrapolation of [`Diffusion4thChernoff`] (K5),
//! achieving order-4 temporal convergence via cancellation of the leading O(τ³) error term:
//!
//! $$
//! F_\beta(\tau) f = \frac{4}{3} K_5(\tau/2)^2 f - \frac{1}{3} K_5(\tau) f
//! $$
//!
//! Since K5 is a symmetric (time-reversible) semigroup approximation, its global
//! error has only odd powers of τ:
//! `K5(τ)^n f = exp(tA)f + c₂τ²Ε + c₄τ⁴E' + O(τ⁶)` (n=t/τ).
//! Richardson cancels the leading `c₂τ²` term and jumps to `O(τ⁴)` global accuracy.
//!
//! This is unconditionally stable (each K5 step is contractive) and works at any N.
//! The 4-term Taylor expansion of A⊕ was explored but fails at N=512 because the
//! divergence-form stencil spectral radius (~3916) makes τ·ρ ≈ 122 at n=16, causing
//! catastrophic overflow. Richardson avoids this entirely.
//!
//! ## Algorithm (Normative)
//!
//! For each outer step of size τ, three inner K5 calls are made:
//!
//! ```text
//! coarse = K5(τ) · f                        // one coarse step
//! half   = K5(τ/2) · f                      // first half-step
//! fine   = K5(τ/2) · half                   // second half-step
//! F_β(τ) f = (4·fine − coarse) / 3          // Richardson combination
//! ```
//!
//! Work per outer step: 3× inner K5 applications (vs 1× for plain K5).
//! Temporal order: 4 (via Richardson; vs 2 for plain K5).
//!
//! ## Acceptance gates
//!
//! - **G_zeta4_const_a_richardson_ratio** (RELEASE_BLOCKING per ADR-0086 AMENDMENT 1):
//!   log₂(err₄/err₈) ≥ 3.5 in constant-a regime (Richardson order-gain detector).
//!   (`tests/zeta4_correction_slope.rs` `g_zeta4_const_a_richardson_ratio`, feature `slow-tests`)
//!   Gate params: N_SPATIAL=512, n-pair {4,8}, T=0.5, analytic oracle. Measured ≈ 3.585.
//!   *Note*: a prior −3.9 OLS gate was superseded by ADR-0086 AMENDMENT 1; no test
//!   enforces −3.9. The var-a OLS gate ≤ −2.5 is ADVISORY (not BLOCKING).
//! - **G_zeta4_const_a_richardson_cheb** (RELEASE_BLOCKING per ADR-0109 + AMENDMENT 1):
//!   Richardson ratio log₂(err_4/err_8) ≥ 3.1 (v5.0.0 baseline PRESERVED; SepticHermite-floor
//!   invariant per math.md §40.5.bis — gate measures pre-asymp K5+Richardson temporal
//!   transition regime, INDEPENDENT of the spatial floor).
//!   (`tests/zeta4_correction_slope_cheb.rs`, feature `slow-tests`)
//!   Note: ≈ 1.49e-12 SepticHermite-bound (ADR-0109; const-a gate measures pre-asymp-temporal
//!   regime per §40.5.bis). K=4 order proven by G_zeta4_TRUTHFUL_ORDER (ADR-0110).
//! - **G_zeta4_TRUTHFUL_ORDER** (RELEASE_BLOCKING per ADR-0110 AMENDMENT 1):
//!   OLS slope ≤ −3.5 in pre-asymptotic regime. T=2.0, N_STEPS={2,4,8,16},
//!   Chebyshev M=64. (`tests/zeta4_truthful_order.rs`, feature `slow-tests`)
//!   Demonstrates honest GLOBAL order-4 via middle-pair (4→8) clean signal.
//!   AMENDMENT 1 revises original -3.95 (per-step formula model) to -3.5 (GLOBAL
//!   model + OLS tolerance for boundary anomalies); ζ⁶/ζ⁸ TRUTHFUL_ORDER siblings
//!   DEFERRED to v7.0+ OCTONIC per AMENDMENT 1 §"Path D".
//! - **T23N** (NORMATIVE): 4 sub-checks — Taylor coefficient, Hermite tangency,
//!   rate constant, τ² coefficient verification.
//!   (`scripts/verify_zeta4_correction.py`)
//!
//! ## Caller invariants
//!
//! 1. `f ∈ D(A^4)`: pre-check `kernel.in_subspace::<4>(&f)` before iterating.
//! 2. `a ∈ C^4_b`: assert via `a_kth_bound: Some(c)` at construction.
//! 3. `a(x) > 0` everywhere (strict ellipticity, inherited from inner).
//!
//! ## References
//!
//! - Galkin, Remizov (2025) *Israel J. Math.* — Theorem 3.1 (m=4 Taylor tangency).
//! - Remizov (2025) *Vladikavkaz Math. J.* 27:4 — Theorem 6 (foundational).
//! - ADR-0086 — G_zeta4 resolution via Path β (supersedes ADR-0085 deferral).
//! - math.md §27 AMENDMENT — Path β normative algorithm spec.

// Mathematical LaTeX symbols (A^k, C^4_b, etc.) are intentional; not code identifiers.
#![allow(clippy::doc_markdown)]

use crate::{
    approximation::ApproximationSubspace,
    chernoff::{ChernoffFunction, Growth},
    diffusion4::Diffusion4thChernoff,
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Order-4-temporal Chernoff kernel for `∂_t u = ∂_x(a(x) ∂_x u)` (ADR-0086 Path β).
///
/// Wraps [`Diffusion4thChernoff`] (order-2 temporal, order-4 spatial) with Richardson
/// extrapolation that achieves genuine order-4 temporal convergence per
/// Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 (m=4 specialisation).
/// Each step makes 3 inner K5 calls: `(4·K5(τ/2)²·f − K5(τ)·f)/3`.
///
/// # Constructor
///
/// ```rust
/// use semiflow::{ChernoffFunction, Diffusion4thChernoff, Diffusion4thZeta4Chernoff, Grid1D};
/// let grid = Grid1D::new(-10.0, 10.0, 512).unwrap();
/// let inner = Diffusion4thChernoff::new(
///     |x: f64| 1.0 + 0.5 * x.tanh().powi(2),
///     |x: f64| x.tanh() * (1.0 - x.tanh().powi(2)),
///     |x: f64| (1.0 - x.tanh().powi(2)).powi(2) - 2.0 * x.tanh().powi(2) * (1.0 - x.tanh().powi(2)),
///     2.5,
///     grid,
/// );
/// let kernel = Diffusion4thZeta4Chernoff::new(inner, Some(2.5_f64)).unwrap();
/// assert_eq!(kernel.order(), 4); // v4.1: Path β achieves what v3.0 promised
/// ```
///
/// # Caller invariants
///
/// 1. `f ∈ D(A^4)`: pre-check `kernel.in_subspace::<4>(&f)` once.
/// 2. `a ∈ C^4_b`: assert via `a_kth_bound: Some(c)`.
/// 3. `a(x) > 0` everywhere (strict ellipticity).
///
/// Failure to meet invariant 3 causes a `DomainViolation` error from the inner kernel.
/// Failure to meet invariants 1 or 2 does NOT cause a panic but may degrade convergence
/// below order 4. See math.md §27 AMENDMENT.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct Diffusion4thZeta4Chernoff<F: SemiflowFloat = f64> {
    /// Inner v0.6.0 4th-order spatial kernel (order-2 in time, baseline A operator).
    pub inner: Diffusion4thChernoff<F>,
    /// Caller-asserted bound `‖a^(k)‖_∞ ≤ c` for k ≤ 4 (Path β semantics: `a ∈ C^4_b`).
    /// `None` = unchecked (K=4 witness returns `false`).
    pub(crate) a_kth_bound: Option<F>,
    /// Grid geometry (copy of inner's grid for direct access).
    pub(crate) grid: Grid1D<F>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> Diffusion4thZeta4Chernoff<F> {
    /// Construct an order-4-temporal kernel from a v0.6.0 order-2 inner kernel.
    ///
    /// `a_kth_bound: Some(c)` asserts `‖a^(k)‖_∞ ≤ c` for k ≤ 4 (Path β: `a ∈ C^4_b`).
    /// `None` opts out (the K=4 witness returns `false` in that case).
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] when:
    /// - `inner.order() < 2` — the inner kernel must be at least order-2.
    /// - `a_kth_bound` is `Some(c)` with `c.is_nan() || c < 0` — malformed bound.
    pub fn new(
        inner: Diffusion4thChernoff<F>,
        a_kth_bound: Option<F>,
    ) -> Result<Self, SemiflowError> {
        if inner.order_val() < 2 {
            return Err(SemiflowError::DomainViolation {
                what: "inner kernel must be order >= 2 for ζ⁴ correction",
                value: f64::from(inner.order_val()),
            });
        }
        if let Some(c) = a_kth_bound {
            if c.is_nan() || c < F::zero() {
                return Err(SemiflowError::DomainViolation {
                    what: "a_kth_bound must be non-negative and finite",
                    value: c.to_f64().unwrap_or(f64::NAN),
                });
            }
        }
        let grid = inner.grid;
        Ok(Self {
            inner,
            a_kth_bound,
            grid,
        })
    }

    /// Opt in to Chebyshev spectral sampling with default M=64 (ADR-0090).
    ///
    /// Propagates to the inner K5 kernel. Default OFF for ζ⁴.
    /// Use `.with_chebyshev_sampling()` to reach theoretical order-4 in the
    /// const-a regime (see ADR-0090 AC8 / G_zeta4_const_a_richardson_cheb gate).
    ///
    /// **Floor note (ADR-0109 §40.4)**: v6.0.0 SepticHermite sampler floor ≈ 1.49e-12.
    /// Chebyshev M=64 barycentric combines these SepticHermite samples; the floor
    /// propagates with the Lebesgue constant Λ_M=64 ≈ 3.3 (Berrut-Trefethen 2004).
    ///
    /// ## Predicted slope table (math.md §41.4, ADR-0109 formal model)
    ///
    /// | Mode | n-pair | Predicted slope | Gate threshold |
    /// |------|--------|-----------------|----------------|
    /// | Default-mode (floor-saturated) | {4,8} T=0.5 | **4.84** | ≥ 4.84 BLOCKING |
    /// | Truthful-order (pre-asymptotic) | {2..16} T=2.0 | **≥ 3.95** | ≥ 3.95 BLOCKING |
    ///
    /// v5.0.0 gate was ≥ 3.1 (QuinticHermite-bound). v6.0.0 raises to ≥ 4.84 (SepticHermite).
    /// Pre-asymptotic gate `G_zeta4_TRUTHFUL_ORDER` demonstrates true order-4 per ADR-0110.
    #[must_use]
    pub fn with_chebyshev_sampling(mut self) -> Self {
        self.inner = self.inner.with_chebyshev_sampling();
        self
    }

    /// Remove Chebyshev spectral sampling (debugging only).
    ///
    /// Propagates to the inner K5 kernel. **WARNING**: gates calibrated for
    /// Chebyshev will fail after this call.
    #[must_use]
    pub fn without_chebyshev_sampling(mut self) -> Self {
        self.inner = self.inner.without_chebyshev_sampling();
        self
    }

    /// Opt in to Chebyshev spectral sampling with explicit M (ADR-0090).
    ///
    /// M ∈ {8, 16, 32, 64, 128, 256, 512}. Default M=64 via `.with_chebyshev_sampling()`.
    #[must_use]
    pub fn with_chebyshev_sampling_m(mut self, m: usize) -> Self {
        self.inner = self.inner.with_chebyshev_sampling_m(m);
        self
    }

    /// Opt in to OctonicHermite degree-9 spatial sampling (ADR-0117, v7.0 KEYSTONE).
    ///
    /// Propagates to the inner K5 kernel. Required for ζ⁶/ζ⁸ TRUTHFUL_ORDER gates
    /// at N=4096/T=10 (ADR-0119 GO). Default OFF; ADDITIVE — no existing gates affected.
    #[must_use]
    pub fn with_octonic_sampling(mut self) -> Self {
        self.inner = self.inner.with_octonic_sampling();
        self
    }
}

// ---------------------------------------------------------------------------
// Private helper: apply the divergence-form A = ∂_x(a(x) ∂_x) stencil
// ---------------------------------------------------------------------------

/// Apply 3-point divergence-form `A = ∂_x(a(x)·∂_x)` with Neumann BCs.
///
/// `(Af)_i = [a(x_{i+½})(f_{i+1}-f_i) - a(x_{i-½})(f_i-f_{i-1})] / dx²`
///
/// Reused from `diffusion4_zeta4_data.rs` (now deleted per ADR-0086 AC4).
/// This is a `pub(crate)` helper so `ApproximationSubspace` jet impls can call it.
#[allow(clippy::cast_precision_loss)] // n ≤ grid size; well within f64 mantissa
pub(crate) fn apply_div_form(
    dc: &Diffusion4thChernoff<f64>,
    f: &GridFn1D<f64>,
    out: &mut GridFn1D<f64>,
) -> Result<(), SemiflowError> {
    let n = f.values.len();
    if n < 3 {
        return Err(SemiflowError::DomainViolation {
            what: "divergence-form stencil requires >= 3 grid points",
            value: n as f64,
        });
    }
    let dx = dc.grid.dx();
    let dx2 = dx * dx;
    out.values.resize(n, 0.0);
    for i in 0..n {
        let x_i = dc.grid.x_at(i);
        let a_pos = dc.eval_a(x_i + 0.5 * dx);
        let a_neg = dc.eval_a(x_i - 0.5 * dx);
        let f_pos = if i + 1 < n {
            f.values[i + 1]
        } else {
            f.values[n - 1]
        };
        let f_neg = if i > 0 { f.values[i - 1] } else { f.values[0] };
        let f_i = f.values[i];
        out.values[i] = (a_pos * (f_pos - f_i) - a_neg * (f_i - f_neg)) / dx2;
    }
    Ok(())
}

/// Compute K-jet `[f, Af, ..., A^K f]` via K iterations of `apply_div_form`.
///
/// `out` must have length K+1. `out[0] = f` (identity).
pub(crate) fn apply_jet_iter(
    dc: &Diffusion4thChernoff<f64>,
    f: &GridFn1D<f64>,
    out: &mut [GridFn1D<f64>],
    k: usize,
) -> Result<(), SemiflowError> {
    out[0].values.clone_from(&f.values);
    apply_div_form(dc, f, &mut out[1])?;
    for j in 1..k {
        let prev = out[j].clone();
        apply_div_form(dc, &prev, &mut out[j + 1])?;
    }
    Ok(())
}

/// Validate tau: finite, non-negative (f64).
#[inline]
fn validate_tau(tau: f64) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for Diffusion4thZeta4Chernoff<f64> {
    type S = GridFn1D<f64>;

    /// Consistency order **4** (Path β: 4-term Taylor expansion per ADR-0086).
    ///
    /// v3.0 (ADR-0075) claimed order 4 via BCH correction — empirically falsified
    /// in v3.1 Wave D (slope ≈ −1.0). v4.0 (ADR-0085) corrected to order 2.
    /// v4.1 (ADR-0086) restores order 4 via Path β (4-term Taylor; slope −4.06
    /// empirically confirmed in v3.1 Wave D with exact A^k applications).
    fn order(&self) -> u32 {
        4
    }

    /// Growth bound: same contraction as inner (`multiplier=1.0, ω=0.0`).
    ///
    /// Path β's correction terms are bounded by `‖f‖_{D(A^3)} · e^{|τω|}`;
    /// the inner growth bound applies without inflation. Reduced from v3.0's 1.5×
    /// factor (which was justified by the unverified BCH correction claim).
    fn growth(&self) -> Growth<f64> {
        let g = self.inner.growth();
        Growth {
            multiplier: g.multiplier,
            omega: g.omega,
        }
    }

    /// Path β: Richardson extrapolation of the inner K5 kernel (ADR-0086).
    ///
    /// Algorithm (3 inner K5 applications per outer step):
    ///
    /// ```text
    /// coarse = K5(τ) · src
    /// half   = K5(τ/2) · src
    /// fine   = K5(τ/2) · half
    /// dst    = (4·fine − coarse) / 3
    /// ```
    ///
    /// Unconditionally stable: each K5 step is a contractive semigroup approximation.
    /// Order-4 temporal: K5 is symmetric (Catmull-Rom baseline), so its error has only
    /// odd τ powers. Richardson cancels the leading O(τ³) global error term, jumping
    /// to O(τ⁵) local / O(τ⁴) global convergence.
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] on:
    /// - Invalid `tau` (NaN, negative, infinite).
    /// - Inner K5 apply_into failure (propagated unchanged).
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        validate_tau(tau)?;

        let n = src.values.len();
        let tau_half = tau / 2.0;

        // Scratch buffers: coarse, half, fine.
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

        // Step 1: coarse = K5(τ) · src  (one coarse step)
        self.inner.apply_into(tau, src, &mut coarse, scratch)?;

        // Step 2: half = K5(τ/2) · src  (first half-step)
        self.inner.apply_into(tau_half, src, &mut half, scratch)?;

        // Step 3: fine = K5(τ/2) · half  (second half-step)
        self.inner.apply_into(tau_half, &half, &mut fine, scratch)?;

        // Step 4: dst = (4·fine − coarse) / 3  (Richardson combination)
        // dst[i] = (4·fine[i] - coarse[i]) / 3
        dst.values.resize(n, 0.0);
        for i in 0..n {
            dst.values[i] = (4.0 * fine.values[i] - coarse.values[i]) / 3.0;
        }

        // Return scratch to pool.
        scratch.return_vec(coarse.values);
        scratch.return_vec(half.values);
        scratch.return_vec(fine.values);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ApproximationSubspace impl (K=4 only; K=6 retired per ADR-0086 AC9)
// ---------------------------------------------------------------------------

/// K=4 approximation subspace witness (ADR-0073, ADR-0086 §AC9).
///
/// Path β requires `D(A^4)` — the K=4 witness is necessary and sufficient.
/// The K=6 witness (`ApproximationSubspace<6, F>`) is REMOVED per ADR-0086 AC9:
/// Path β does not require the strict `D(A^6)` core. Callers who relied on the
/// K=6 witness must downgrade to K=4 (strictly more permissive — any K=4-OK datum
/// passes K=4 trivially). See `docs/migration/v3-to-v4.md` §"v4.1 G_zeta4 resolution".
///
/// `in_subspace`: true when:
/// - grid has ≥ 17 points (4-iteration 9-point stencil minimum),
/// - all values are finite,
/// - `a_kth_bound` is `Some(_)` (caller-asserted `a ∈ C^4_b`).
impl ApproximationSubspace<4, f64> for Diffusion4thZeta4Chernoff<f64> {
    fn in_subspace(&self, f: &GridFn1D<f64>) -> bool {
        f.values.len() >= 17 && f.values.iter().all(|v| v.is_finite()) && self.a_kth_bound.is_some()
    }

    #[allow(clippy::cast_precision_loss)] // out.len() ≤ K+1=5; well within f64 mantissa
    fn jet(&self, f: &GridFn1D<f64>, out: &mut [GridFn1D<f64>]) -> Result<(), SemiflowError> {
        if out.len() != 5 {
            return Err(SemiflowError::DomainViolation {
                what: "jet K=4 requires out.len() == 5",
                value: out.len() as f64,
            });
        }
        apply_jet_iter(&self.inner, f, out, 4)
    }
}

// ---------------------------------------------------------------------------
// Unit tests (fast, no feature gate) — extracted to sibling file per ≤500-line cap.
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "diffusion4_zeta4_tests.rs"]
mod tests;
