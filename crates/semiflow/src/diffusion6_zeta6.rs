//! [`Diffusion6thZeta6Chernoff`] — order-6-temporal ζ⁶ kernel (ADR-0088 Wave I, ADR-0086 Path β-ladder rung K=3).
//!
//! ## Mathematical foundation (ADR-0088, math.md §27.bis)
//!
//! Path β-ladder rung K=3: nested Richardson extrapolation of
//! [`Diffusion4thZeta4Chernoff`] (R², itself order-4), achieving order-6 temporal
//! convergence via cancellation of the leading O(τ⁵) error term:
//!
//! $$
//! R^3(\tau) f = \frac{16}{15} R^2(\tau/2)^2 f - \frac{1}{15} R^2(\tau) f
//! $$
//!
//! where $R^2 = $ `Diffusion4thZeta4Chernoff` (order-4 Richardson on K5).
//!
//! Since R² is a symmetric (time-reversible) approximation, its global error has
//! only odd powers of τ from the perspective of Richardson iteration.
//! Richardson at K=3 cancels the leading `c₄τ⁴` term and jumps to O(τ⁵) local /
//! O(τ⁶) global convergence.
//!
//! ## Algorithm (Normative, ADR-0088 Wave I)
//!
//! Per outer step of size τ, **three** inner R² (ζ⁴) calls are made:
//!
//! ```text
//! coarse = R²(τ)   · src              // 1 R² call = 3 K5 calls
//! half   = R²(τ/2) · src              // 1 R² call = 3 K5 calls
//! fine   = R²(τ/2) · half             // 1 R² call = 3 K5 calls
//! dst    = (16·fine − coarse) / 15    // Richardson combination (K=3)
//! ```
//!
//! Work per outer step: 3× inner R² applications = 9 K5 base evaluations.
//! Temporal order: 6 (vs 4 for R²/ζ⁴, vs 2 for plain K5).
//!
//! ## Acceptance gates
//!
//! - **G_zeta6_const_a_richardson** (RELEASE_BLOCKING per ADR-0088):
//!   Richardson ratio log₂(err₄/err₈) ≥ 5.5.
//!   (`tests/zeta6_correction_slope.rs`, feature `slow-tests`)
//! - **G_zeta6_const_a_richardson_cheb** (RELEASE_BLOCKING per ADR-0109 + AMENDMENT 1):
//!   Richardson ratio log₂(err_1/err_2) ≥ 3.8 (v5.0.0 baseline PRESERVED; SepticHermite-floor
//!   invariant per math.md §40.5.bis — gate measures pre-asymp K5+Richardson temporal
//!   transition regime, INDEPENDENT of the spatial floor).
//!   (`tests/zeta6_correction_slope_cheb.rs`, feature `slow-tests`)
//!   Note: ≈ 1.49e-12 SepticHermite-bound (ADR-0109; const-a gate measures pre-asymp-temporal
//!   regime per §40.5.bis). K=6 LOCAL tangency: T23N_zeta6 sympy (G_zeta6_TRUTHFUL_ORDER DEFERRED v7.0+).
//! - **G_zeta6_TRUTHFUL_ORDER** (DEFERRED v7.0+ OCTONIC per ADR-0110 AMENDMENT 1):
//!   GLOBAL gate INFEASIBLE at v6.0.0 K5 3-point stencil + SepticHermite floor.
//!   K=6 honesty covered by G_zeta6_const_a_richardson_cheb (≥ 3.8 BLOCKING PASS,
//!   ADR-0109 AMENDMENT 1) + T23N_zeta6 sympy LOCAL Taylor tangency oracle.
//! - **G_zeta6_var_a_slope** (RELEASE_ADVISORY per ADR-0088):
//!   OLS slope ≤ −4.5 against K5 reference at n_ref=8192.
//!   (`tests/zeta6_correction_slope.rs`, feature `slow-tests`)
//! - **T23N_zeta6** (NORMATIVE): 4 sub-checks — Taylor to τ⁵, Hermite tangency
//!   to τ⁶, rate constant C_R^(K=3) ≤ 1/126, leading τ⁴ coefficient inheritance.
//!   (`scripts/verify_zeta6_correction.py`)
//!
//! ## Caller invariants
//!
//! 1. `f ∈ D(A^6)`: pre-check `kernel.in_subspace::<6>(&f)` before iterating.
//! 2. `a ∈ C^6_b`: assert via `a_kth_bound: Some(c)` at construction.
//! 3. `a(x) > 0` everywhere (strict ellipticity, inherited from inner).
//!
//! ## References
//!
//! - Galkin, Remizov (2025) *Israel J. Math.* — Theorem 3.1 (m=6 Taylor tangency).
//! - ADR-0088 — ζ⁶/ζ⁸ ladder rungs via nested Richardson on K5.
//! - ADR-0086 — Path β resolution (ζ⁴ foundation).
//! - math.md §27.bis — R³ algorithm (NORMATIVE).

// Mathematical LaTeX symbols (A^k, C^6_b, etc.) are intentional; not code identifiers.
#![allow(clippy::doc_markdown)]

use crate::{
    approximation::ApproximationSubspace,
    chernoff::{ChernoffFunction, Growth},
    diffusion4_zeta4::Diffusion4thZeta4Chernoff,
    diffusion4_zeta4_stencil_ho::apply_jet_iter_6th,
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

/// Order-6-temporal Chernoff kernel for `∂_t u = ∂_x(a(x) ∂_x u)` (ADR-0088 Wave I).
///
/// Wraps [`Diffusion4thZeta4Chernoff`] (order-4) with nested Richardson
/// extrapolation that achieves order-6 temporal convergence per
/// Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 (m=6 specialisation).
/// Each step makes 3 inner R² calls: `(16·R²(τ/2)²·f − R²(τ)·f)/15`.
///
/// # Constructor
///
/// ```rust,ignore
/// use semiflow_core::{ChernoffFunction, Diffusion4thChernoff, Diffusion4thZeta4Chernoff,
///     Diffusion6thZeta6Chernoff, Grid1D};
/// let grid = Grid1D::new(-10.0, 10.0, 512).unwrap();
/// let inner = Diffusion4thChernoff::new(
///     |x: f64| 1.0 + 0.5 * x.tanh().powi(2),
///     |x: f64| x.tanh() * (1.0 - x.tanh().powi(2)),
///     |x: f64| (1.0 - x.tanh().powi(2)).powi(2) - 2.0 * x.tanh().powi(2) * (1.0 - x.tanh().powi(2)),
///     2.5,
///     grid,
/// );
/// let zeta4 = Diffusion4thZeta4Chernoff::new(inner, Some(2.5_f64)).unwrap();
/// let kernel = Diffusion6thZeta6Chernoff::new(zeta4, Some(2.5_f64)).unwrap();
/// assert_eq!(kernel.order(), 6);
/// ```
///
/// # Caller invariants
///
/// 1. `f ∈ D(A^6)`: pre-check `kernel.in_subspace::<6>(&f)` once.
/// 2. `a ∈ C^6_b`: assert via `a_kth_bound: Some(c)`.
/// 3. `a(x) > 0` everywhere (strict ellipticity).
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct Diffusion6thZeta6Chernoff<F: SemiflowFloat = f64> {
    /// Inner ζ⁴ kernel (order-4 temporal, wraps Diffusion4thChernoff).
    /// `pub` for test-access to verify Path ε direct K5 Quintic wiring (ADR-0089 AMENDMENT 1).
    pub inner: Diffusion4thZeta4Chernoff<F>,
    /// Caller-asserted bound `‖a^(k)‖_∞ ≤ c` for k ≤ 6 (path β-ladder: `a ∈ C^6_b`).
    /// `None` = unchecked (K=6 witness returns `false`).
    pub(crate) a_kth_bound: Option<F>,
    /// Grid geometry (copy of inner's grid for direct access).
    pub(crate) grid: Grid1D<F>,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> Diffusion6thZeta6Chernoff<F> {
    /// Construct an order-6-temporal kernel from a ζ⁴ (order-4) inner kernel.
    ///
    /// `a_kth_bound: Some(c)` asserts `‖a^(k)‖_∞ ≤ c` for k ≤ 6 (rung K=3: `a ∈ C^6_b`).
    /// `None` opts out (the K=6 witness returns `false` in that case).
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] when:
    /// - `a_kth_bound` is `Some(c)` with `c.is_nan() || c < 0` — malformed bound.
    pub fn new(
        inner: Diffusion4thZeta4Chernoff<F>,
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
        Ok(Self {
            inner,
            a_kth_bound,
            grid,
        })
    }

    /// Opt in to Chebyshev spectral sampling with default M=64 (ADR-0090).
    ///
    /// Propagates to the inner ζ⁴ → K5 chain. Default OFF for ζ⁶.
    /// Use to opt in to the Chebyshev spectral path for the ζ⁶ spatial sampling.
    ///
    /// **Floor note (ADR-0109 §40.4)**: SepticHermite floor ≈ 1.49e-12. Cascade σ² ≈ 1.33.
    /// Predicted default-mode slope 5.98 (floor-saturated; n={1,2} T=0.5); GLOBAL
    /// truthful-order gate DEFERRED v7.0+ per ADR-0110 AMENDMENT 1. v5.0 gate was ≥ 3.8.
    #[must_use]
    pub fn with_chebyshev_sampling(mut self) -> Self {
        self.inner = self.inner.with_chebyshev_sampling();
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

    /// Remove Chebyshev spectral sampling — debugging only.
    ///
    /// Propagates the downgrade through the inner ζ⁴ kernel.
    /// **WARNING**: G_zeta6 truthful-order gate will fail without Chebyshev.
    #[must_use]
    pub fn without_chebyshev_sampling(mut self) -> Self {
        self.inner = self.inner.without_chebyshev_sampling();
        self
    }

    /// Opt in to OctonicHermite degree-9 spatial sampling (ADR-0117, v7.0 KEYSTONE).
    ///
    /// Propagates through ζ⁴ → K5 chain. Required for ζ⁶ TRUTHFUL_ORDER gate ≤ −5.95
    /// at N=4096/T=10 (ADR-0119 GO). Default OFF; ADDITIVE — existing gates unaffected.
    #[must_use]
    pub fn with_octonic_sampling(mut self) -> Self {
        self.inner = self.inner.with_octonic_sampling();
        self
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for Diffusion6thZeta6Chernoff<f64> {
    type S = GridFn1D<f64>;

    /// Consistency order **≥ 6** (ADR-0088 Wave I: K=3 nested Richardson on ζ⁴),
    /// verified by the finest-rung lower-bound gate `G_zeta6_TRUTHFUL_ORDER`
    /// (finest pair (8→16) slope ≤ −5.95 = K−0.05; ADR-0119 AMENDMENT 2).
    ///
    /// Richardson at K=3 on R² (order-4 base) cancels the leading O(τ⁵) error
    /// term, achieving O(τ⁷) local / O(τ⁶) global convergence.
    fn order(&self) -> u32 {
        6
    }

    /// Growth bound: same contraction as inner (`multiplier=1.0, ω=0.0`).
    ///
    /// Nested Richardson combination over inner is bounded by `‖f‖_{D(A^5)}`;
    /// the inner growth bound applies without inflation.
    fn growth(&self) -> Growth<f64> {
        let g = self.inner.growth();
        Growth {
            multiplier: g.multiplier,
            omega: g.omega,
        }
    }

    /// R³(τ): nested Richardson extrapolation of the inner R² (ζ⁴) kernel.
    ///
    /// Algorithm (3 inner R² applications per outer step):
    ///
    /// ```text
    /// coarse = R²(τ)   · src          // one coarse ζ⁴ step
    /// half   = R²(τ/2) · src          // first half ζ⁴ step
    /// fine   = R²(τ/2) · half         // second half ζ⁴ step
    /// dst    = (16·fine − coarse) / 15 // Richardson K=3 combination
    /// ```
    ///
    /// Unconditionally stable: each R² step is contractive.
    /// Order-6 temporal: R² is symmetric, so its global error has only odd τ
    /// powers. Richardson cancels the leading O(τ⁵) term, achieving O(τ⁶) global.
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] on:
    /// - Invalid `tau` (NaN, negative, infinite).
    /// - Inner R² apply_into failure (propagated unchanged).
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

        // Step 1: coarse = R²(τ) · src  (one coarse ζ⁴ step)
        self.inner.apply_into(tau, src, &mut coarse, scratch)?;

        // Step 2: half = R²(τ/2) · src  (first half ζ⁴ step)
        self.inner.apply_into(tau_half, src, &mut half, scratch)?;

        // Step 3: fine = R²(τ/2) · half  (second half ζ⁴ step → R²(τ/2)² · src)
        self.inner.apply_into(tau_half, &half, &mut fine, scratch)?;

        // Step 4: dst = (16·fine − coarse) / 15  (Richardson K=3 combination)
        dst.values.resize(n, 0.0);
        for i in 0..n {
            dst.values[i] = (16.0 * fine.values[i] - coarse.values[i]) / 15.0;
        }

        // Return scratch to pool.
        scratch.return_vec(coarse.values);
        scratch.return_vec(half.values);
        scratch.return_vec(fine.values);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ApproximationSubspace impl (K=6)
// ---------------------------------------------------------------------------

/// K=6 approximation subspace witness (ADR-0073, ADR-0088 Wave I).
///
/// Rung K=3 requires `D(A^6)` — the K=6 witness is necessary and sufficient.
/// `in_subspace`: true when:
/// - grid has ≥ 25 points (6-iteration 9-point stencil minimum: 4 K5 pts × 6),
/// - all values are finite,
/// - `a_kth_bound` is `Some(_)` (caller-asserted `a ∈ C^6_b`).
impl ApproximationSubspace<6, f64> for Diffusion6thZeta6Chernoff<f64> {
    fn in_subspace(&self, f: &GridFn1D<f64>) -> bool {
        f.values.len() >= 25 && f.values.iter().all(|v| v.is_finite()) && self.a_kth_bound.is_some()
    }

    #[allow(clippy::cast_precision_loss)] // out.len() ≤ K+1=7; well within f64 mantissa
    fn jet(&self, f: &GridFn1D<f64>, out: &mut [GridFn1D<f64>]) -> Result<(), SemiflowError> {
        if out.len() != 7 {
            return Err(SemiflowError::DomainViolation {
                what: "jet K=6 requires out.len() == 7",
                value: out.len() as f64,
            });
        }
        apply_jet_iter_6th(&self.inner.inner, f, out, 6)
    }
}

// ---------------------------------------------------------------------------
// Unit tests (fast, no feature gate)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Diffusion4thChernoff, Diffusion4thZeta4Chernoff, Grid1D, GridFn1D};

    fn make_kernel(n: usize) -> Diffusion6thZeta6Chernoff<f64> {
        let grid = Grid1D::new(-4.0, 4.0, n).expect("grid");
        let inner = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let zeta4 = Diffusion4thZeta4Chernoff::new(inner, Some(1.0_f64)).expect("zeta4");
        Diffusion6thZeta6Chernoff::new(zeta4, Some(1.0_f64)).expect("kernel")
    }

    #[test]
    fn constructor_validates_bound() {
        let grid = Grid1D::new(-4.0, 4.0, 32).expect("grid");
        let inner = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let zeta4 = Diffusion4thZeta4Chernoff::new(inner.clone(), Some(1.0_f64)).expect("zeta4");
        assert!(Diffusion6thZeta6Chernoff::new(zeta4.clone(), Some(-1.0_f64)).is_err());
        assert!(Diffusion6thZeta6Chernoff::new(zeta4, Some(f64::NAN)).is_err());
        // Also verify with None (should succeed)
        let inner2 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let zeta4b = Diffusion4thZeta4Chernoff::new(inner2, None).expect("zeta4");
        assert!(Diffusion6thZeta6Chernoff::new(zeta4b, None).is_ok());
    }

    #[test]
    fn order_is_6() {
        let k = make_kernel(32);
        assert_eq!(k.order(), 6);
    }

    #[test]
    fn growth_multiplier_is_1p0() {
        let k = make_kernel(32);
        let g = k.growth();
        assert!((g.multiplier - 1.0).abs() < 1e-12);
        assert!(g.omega.abs() < f64::EPSILON, "omega must be zero");
    }

    #[test]
    fn apply_into_produces_finite_output() {
        let k = make_kernel(64);
        let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
        let mut dst = f.zeroed_like();
        let mut scratch = ScratchPool::new();
        k.apply_into(0.01, &f, &mut dst, &mut scratch)
            .expect("apply_into should succeed");
        assert!(
            dst.values.iter().all(|v| v.is_finite()),
            "output must be finite"
        );
    }

    #[test]
    fn apply_into_rejects_negative_tau() {
        let k = make_kernel(32);
        let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
        let mut dst = f.zeroed_like();
        let mut scratch = ScratchPool::new();
        assert!(k.apply_into(-0.01, &f, &mut dst, &mut scratch).is_err());
    }

    /// tau=0 should return f unchanged (all correction terms vanish).
    #[test]
    fn apply_into_tau_zero_returns_src() {
        let k = make_kernel(64);
        let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
        let mut dst = f.zeroed_like();
        let mut scratch = ScratchPool::new();
        k.apply_into(0.0, &f, &mut dst, &mut scratch)
            .expect("tau=0 must succeed");
        let max_diff = f
            .values
            .iter()
            .zip(dst.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-14,
            "tau=0 must return src unchanged; max_diff={max_diff:.2e}"
        );
    }

    #[test]
    fn k6_in_subspace_true_for_large_grid() {
        let k = make_kernel(64);
        let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
        assert!(<Diffusion6thZeta6Chernoff<f64> as ApproximationSubspace<
            6,
            f64,
        >>::in_subspace(&k, &f));
    }

    #[test]
    fn k6_in_subspace_false_without_bound() {
        let grid = Grid1D::new(-4.0, 4.0, 64).expect("grid");
        let inner = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
        let zeta4 = Diffusion4thZeta4Chernoff::new(inner, None).expect("zeta4");
        let k = Diffusion6thZeta6Chernoff::new(zeta4, None).expect("kernel");
        let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
        assert!(!<Diffusion6thZeta6Chernoff<f64> as ApproximationSubspace<
            6,
            f64,
        >>::in_subspace(&k, &f));
    }

    #[test]
    fn k6_jet_finite_on_gaussian() {
        let k = make_kernel(64);
        let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
        let mut out: [GridFn1D<f64>; 7] = core::array::from_fn(|_| f.zeroed_like());
        <Diffusion6thZeta6Chernoff<f64> as ApproximationSubspace<6, f64>>::jet(&k, &f, &mut out)
            .expect("jet K=6 must succeed");
        for (j, slice) in out.iter().enumerate() {
            assert!(
                slice.values.iter().all(|v| v.is_finite()),
                "jet[{j}] has non-finite values"
            );
        }
    }

    #[test]
    fn k6_jet_wrong_len_errors() {
        let k = make_kernel(64);
        let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
        let mut out: [GridFn1D<f64>; 5] = core::array::from_fn(|_| f.zeroed_like());
        assert!(
            <Diffusion6thZeta6Chernoff<f64> as ApproximationSubspace<6, f64>>::jet(
                &k, &f, &mut out,
            )
            .is_err()
        );
    }
}
