//! `Killing2ndChernoff<C, K, F>` — order-2 soft killing via symmetric Strang split.
//!
//! Approximates the Feynman-Kac semigroup `e^{t(L−κ)}` via:
//!
//! ```text
//! F(τ)f = e^{−τκ/2} · C(τ) · e^{−τκ/2}
//! ```
//!
//! where `κ(x) ≥ 0` is a smooth bounded killing-rate field and `C(τ)` is a
//! Chernoff function for the generator `L`. The palindrome achieves **global
//! order 2** when `[L, κ] ≠ 0` (verified in `scripts/verify_killing_order2_preflight.py`,
//! 2026-06-05; ADR-0126; math.md §21.8).
//!
//! ## Contrast with `KillingChernoff` (v2.6, §21.3)
//!
//! This type and `KillingChernoff` solve DIFFERENT problems:
//!
//! - `Killing2ndChernoff` — **soft killing**, `L_κ = L − κ`, order-2.
//!   Rate field `κ(x) ≥ 0`, killed generator is `L − κ` on all of `ℝ`.
//! - `KillingChernoff` — **hard absorbing wall**, order-1.
//!   Indicator `𝟙_R` (post-multiply); forced `u|_{∂R} = 0` Dirichlet BC.
//!
//! Users needing absorbing Dirichlet boundaries MUST use `KillingChernoff`.
//! `Killing2ndChernoff` does NOT implement the absorbing-wall problem at
//! higher order (see ADR-0126 §Decision for why the hard wall is irreducibly
//! order-1).
//!
//! ## Validity guard
//!
//! `new` verifies `κ(x) ≥ 0` at every grid node. A negative rate corresponds
//! to mass creation (not killing) and violates the Feynman-Kac formula;
//! the constructor returns `Err(DomainViolation)` fail-closed.
//!
//! ## Mathematical references
//!
//! - Butko 2018, §5 — open conjecture that motivated the rate-formulation reading.
//! - Strang 1968, SIAM J. Numer. Anal. 5:3 — palindromic operator splitting.
//! - `scripts/verify_killing_order2_preflight.py` — sympy verification of
//!   τ¹, τ² coefficient cancellation for `[L, κ] ≠ 0`.

use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{half, SemiflowFloat},
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// KillingRate<F> — trait
// ---------------------------------------------------------------------------

/// Smooth bounded killing-rate field `κ(x) ≥ 0`.
///
/// Required by `Killing2ndChernoff<C, K, F>`.
///
/// ## Contract
///
/// Implementations MUST return `κ(x) ≥ 0` for all `x`. A negative value
/// corresponds to mass creation (not killing) and the caller SHOULD validate
/// at construction time. `Killing2ndChernoff::new` validates at every grid node
/// and returns `Err(DomainViolation)` if any node has `κ < 0`.
///
/// ## Implementing for constant rates
///
/// The simplest implementation wraps a constant `c ≥ 0`:
///
/// ```rust
/// use semiflow::killing_soft::KillingRate;
///
/// struct ConstRate(f64);
///
/// impl KillingRate<f64> for ConstRate {
///     fn kappa(&self, _x: f64) -> f64 { self.0 }
/// }
/// ```
pub trait KillingRate<F: SemiflowFloat = f64> {
    /// Killing rate at position `x`. Must return `≥ 0`.
    fn kappa(&self, x: F) -> F;
}

// ---------------------------------------------------------------------------
// ClosureKillingRate<F> — blanket impl for closures
// ---------------------------------------------------------------------------

/// `KillingRate<F>` adapter for `Fn(F) -> F` closures.
///
/// Allows constructing `Killing2ndChernoff` directly from a closure without
/// defining a named struct.
///
/// # Example
///
/// ```rust
/// use semiflow::killing_soft::{ClosureKillingRate, KillingRate};
///
/// let rate = ClosureKillingRate::new(|x: f64| 0.3 * x * x);
/// assert!((rate.kappa(1.0) - 0.3).abs() < 1e-15);
/// ```
#[derive(Clone, Copy)]
pub struct ClosureKillingRate<F, Kfn>
where
    F: SemiflowFloat,
    Kfn: Fn(F) -> F,
{
    kfn: Kfn,
    _f: PhantomData<F>,
}

impl<F: SemiflowFloat, Kfn: Fn(F) -> F> ClosureKillingRate<F, Kfn> {
    /// Wrap a closure as a `KillingRate<F>`.
    pub fn new(kfn: Kfn) -> Self {
        Self {
            kfn,
            _f: PhantomData,
        }
    }
}

impl<F: SemiflowFloat, Kfn: Fn(F) -> F> KillingRate<F> for ClosureKillingRate<F, Kfn> {
    fn kappa(&self, x: F) -> F {
        (self.kfn)(x)
    }
}

// ---------------------------------------------------------------------------
// Killing2ndChernoff<C, K, F> — wrapper
// ---------------------------------------------------------------------------

/// Order-2 soft-killing Chernoff function (ADR-0126, math.md §21.8).
///
/// `F(τ)f = e^{−τκ/2} · C(τ)f · e^{−τκ/2}` (pointwise scalar factors).
///
/// ## Type parameters
///
/// - `C`: inner `ChernoffFunction<F>` for the generator `L` (e.g.
///   `DiffusionChernoff<F>`). Must have `S = GridFn1D<F>`.
/// - `K`: killing-rate field, implementing `KillingRate<F>`.
/// - `F`: float type (`f64` default).
///
/// ## Construction
///
/// Use `Killing2ndChernoff::new(inner, rate, grid)`. The constructor validates
/// `κ(x_i) ≥ 0` at every node `i = 0..grid.n`.
///
/// ## Growth bound
///
/// `e^{−τκ/2} ≤ 1` pointwise (κ ≥ 0, τ ≥ 0) ⟹ `‖F(τ)‖ ≤ ‖C(τ)‖`. Growth
/// is inherited from the inner Chernoff.
#[derive(Clone)]
pub struct Killing2ndChernoff<C, K, F: SemiflowFloat = f64> {
    inner: C,
    rate: K,
    grid: Grid1D<F>,
    _f: PhantomData<F>,
}

impl<C, K, F> Killing2ndChernoff<C, K, F>
where
    F: SemiflowFloat,
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    K: KillingRate<F>,
{
    /// Construct a `Killing2ndChernoff` with validated non-negative rate field.
    ///
    /// Validates `κ(x_i) ≥ 0` at every grid node `i = 0..grid.n`. Returns
    /// `Err(DomainViolation)` at the first negative value found.
    ///
    /// # Errors
    ///
    /// - `DomainViolation` if `κ(x_i) < 0` for any `i`.
    /// - `DomainViolation` if `κ(x_i)` is non-finite for any `i`.
    pub fn new(inner: C, rate: K, grid: Grid1D<F>) -> Result<Self, SemiflowError> {
        for i in 0..grid.n {
            let x = grid.x_at(i);
            let k = rate.kappa(x);
            if !k.is_finite() {
                return Err(SemiflowError::DomainViolation {
                    what: "Killing2ndChernoff: kappa(x) must be finite",
                    value: k.to_f64().unwrap_or(f64::NAN),
                });
            }
            if k < F::zero() {
                return Err(SemiflowError::DomainViolation {
                    what:
                        "Killing2ndChernoff: kappa(x) must be >= 0 (negative rate = mass creation)",
                    value: k.to_f64().unwrap_or(f64::NAN),
                });
            }
        }
        Ok(Self {
            inner,
            rate,
            grid,
            _f: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction<F> impl
// ---------------------------------------------------------------------------

impl<C, K, F> ChernoffFunction<F> for Killing2ndChernoff<C, K, F>
where
    F: SemiflowFloat,
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    K: KillingRate<F>,
{
    type S = GridFn1D<F>;

    /// Symmetric Strang split: `e^{−τκ/2} · C(τ) · e^{−τκ/2}`.
    ///
    /// Three stages:
    /// 1. Half-step kill: `tmp[i] = src[i] · e^{−τ·κ(xᵢ)/2}`.
    /// 2. Inner Chernoff step: `mid = C(τ) tmp`.
    /// 3. Half-step kill: `dst[i] = mid[i] · e^{−τ·κ(xᵢ)/2}`.
    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let half_tau = half::<F>() * tau;
        let n = self.grid.n;

        // Stage 1: half-step kill → tmp
        let mut tmp = src.zeroed_like();
        for i in 0..n {
            let x = self.grid.x_at(i);
            let w = (-half_tau * self.rate.kappa(x)).exp();
            tmp.values[i] = src.values[i] * w;
        }

        // Stage 2: inner Chernoff step → mid
        let mut mid = src.zeroed_like();
        self.inner.apply_into(tau, &tmp, &mut mid, scratch)?;

        // Stage 3: half-step kill → dst
        for i in 0..n {
            let x = self.grid.x_at(i);
            let w = (-half_tau * self.rate.kappa(x)).exp();
            dst.values[i] = mid.values[i] * w;
        }
        Ok(())
    }

    /// Order-2 (Strang palindrome, ADR-0126 / math.md §21.8).
    ///
    /// The symmetric split cancels the τ¹ and τ² commutator terms in
    /// `F(τ) − e^{τ(L−κ)}` when `[L, κ] ≠ 0` (pre-flight sympy, 2026-06-05).
    fn order(&self) -> u32 {
        2
    }

    /// Growth bound inherited from inner (soft killing is sub-Markov: `e^{−τκ} ≤ 1`).
    fn growth(&self) -> Growth<F> {
        self.inner.growth()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::{ClosureKillingRate, KillingRate};
    use crate::{
        chernoff::ChernoffFunction, diffusion::DiffusionChernoff, error::SemiflowError,
        grid::Grid1D, grid_fn::GridFn1D, killing_soft::Killing2ndChernoff as K2,
        scratch::ScratchPool,
    };

    // --- KillingRate contract ---

    #[test]
    fn closure_killing_rate_basic() {
        let r = ClosureKillingRate::new(|x: f64| x.abs());
        assert!((r.kappa(2.0) - 2.0).abs() < 1e-15);
        assert!((r.kappa(0.0)).abs() < 1e-15);
    }

    // --- Constructor validation ---

    #[test]
    fn negative_kappa_is_rejected() {
        let grid = Grid1D::new(0.0_f64, 1.0, 8).unwrap();
        let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        // κ(x) = -0.1 everywhere → DomainViolation
        let rate = ClosureKillingRate::new(|_: f64| -0.1_f64);
        let result = K2::new(diff, rate, grid);
        assert!(result.is_err(), "expected Err for κ < 0");
        match result {
            Err(SemiflowError::DomainViolation { .. }) => {}
            Err(other) => panic!("expected DomainViolation, got {other:?}"),
            Ok(_) => unreachable!(),
        }
    }

    #[test]
    fn zero_kappa_returns_inner() {
        // κ = 0 everywhere → F(τ) = C(τ), so Killing2ndChernoff == DiffusionChernoff.
        let grid = Grid1D::new(-1.0_f64, 1.0, 16).unwrap();
        let diff_ref = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        let rate = ClosureKillingRate::new(|_: f64| 0.0_f64);
        let k2 = K2::new(diff, rate, grid).unwrap();

        let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
        let mut dst_k2 = u0.zeroed_like();
        let mut dst_ref = u0.zeroed_like();
        let mut scratch = ScratchPool::new();

        k2.apply_into(0.01, &u0, &mut dst_k2, &mut scratch).unwrap();
        diff_ref
            .apply_into(0.01, &u0, &mut dst_ref, &mut scratch)
            .unwrap();

        let max_diff = dst_k2
            .values
            .iter()
            .zip(dst_ref.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-14,
            "zero-kappa: K2 and DiffusionChernoff should be bit-level close, diff={max_diff:.3e}"
        );
    }

    #[test]
    fn positive_kappa_attenuates() {
        // κ > 0 must reduce the sup-norm vs κ = 0 (mass is killed, not created).
        let grid = Grid1D::new(-2.0_f64, 2.0, 32).unwrap();
        let diff_k2 = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        let diff_ref = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        let rate = ClosureKillingRate::new(|_: f64| 1.0_f64);
        let k2 = K2::new(diff_k2, rate, grid).unwrap();

        let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
        let mut dst_killed = u0.zeroed_like();
        let mut dst_plain = u0.zeroed_like();
        let mut scratch = ScratchPool::new();

        k2.apply_into(0.1, &u0, &mut dst_killed, &mut scratch)
            .unwrap();
        diff_ref
            .apply_into(0.1, &u0, &mut dst_plain, &mut scratch)
            .unwrap();

        let norm_killed: f64 = dst_killed
            .values
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let norm_plain: f64 = dst_plain
            .values
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            norm_killed < norm_plain,
            "killing must reduce sup-norm: killed={norm_killed:.6}, plain={norm_plain:.6}"
        );
    }

    #[test]
    fn order_is_two() {
        let grid = Grid1D::new(0.0_f64, 1.0, 8).unwrap();
        let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        let rate = ClosureKillingRate::new(|_: f64| 0.5_f64);
        let k2 = K2::new(diff, rate, grid).unwrap();
        assert_eq!(k2.order(), 2);
    }
}
