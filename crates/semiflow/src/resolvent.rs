//! A1 — Laplace-Chernoff Resolvent (math.md §22, ADR-0069).
//!
//! Computes `(λI − A)⁻¹ g` via the Hille-Yosida Laplace representation:
//!
//! ```text
//! (λI − A)⁻¹ g = ∫₀^∞ e^{−λt} S(t) g dt
//! ```
//!
//! Substituting `s = λt`:
//!
//! ```text
//! ≈ (1/λ) Σ_k w_k · (C(s_k/(λn)))^n g    (Gauss-Laguerre 32-pt)
//! ```
//!
//! UNIQUE TO REMIZOV: no Trotter-Kato analog for the resolvent.
//! Cite: Remizov 2025 *Vladikavkaz Math. J.* 27(4) Theorem 3.

// Grid size n cast to f64 for error reporting; n ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

use core::marker::PhantomData;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    resolvent_quad::{GL32_NODES, GL32_WEIGHTS},
    scratch::ScratchPool,
    state::State,
};

/// Number of trapezoid nodes for `TrapezoidWithTail` quadrature.
const TRAPEZOID_N: usize = 256;

/// Quadrature strategy for the Laplace-Chernoff resolvent integral.
///
/// Two variants ship in v2.7 (ADR-0069, math §22.3 + §22.4):
///
/// - [`GaussLaguerre32`](Self::GaussLaguerre32): 32-point Gauss-Laguerre
///   quadrature. Const-array nodes/weights; zero allocation. Recommended
///   when `Re(λ) ≫ ω` (well-conditioned regime).
/// - [`TrapezoidWithTail`](Self::TrapezoidWithTail): uniform trapezoid on
///   `[0, t_max]` with 256 nodes + analytical tail bound. Use when
///   `Re(λ) → ω⁺` (marginal regime where Gauss-Laguerre stalls).
///
/// Marked `#[non_exhaustive]` for forward compatibility.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaplaceQuadrature<F: SemiflowFloat = f64> {
    /// 32-point Gauss-Laguerre; recommended default.
    GaussLaguerre32,
    /// Uniform trapezoid on `[0, t_max]` plus analytical tail.
    TrapezoidWithTail {
        /// Upper integration limit. Must be finite and > 0.
        /// Suggestion: `t_max = 50 / lambda` → tail ≈ exp(-50).
        t_max: F,
    },
}

/// Minimal trait: sample the state at a point and reconstruct from a closure.
///
/// Required by [`LaplaceChernoffResolvent::eval_at_point`].
/// Only `GridFn1D<F>` implements this in v2.7 (ADR-0069 §"Limitations").
pub trait Sampleable<F: SemiflowFloat>: State<F> + Sized {
    /// Sample state at point `x`.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] if `x` is out of domain.
    fn sample_at(&self, x: &[F]) -> Result<F, SemiflowError>;
    /// Build a fresh same-shape state from closure `f`.
    ///
    /// # Errors
    /// Returns [`SemiflowError::Unsupported`] if not yet implemented for this type.
    fn fresh_from_fn(&self, f: &dyn Fn(&[F]) -> F) -> Result<Self, SemiflowError>;
}

/// Wrapper that computes `(λI − A)⁻¹ g` via Laplace-Chernoff quadrature.
///
/// See math.md §22 and ADR-0069 for full mathematical derivation.
///
/// # Usage
///
/// ```rust
/// use semiflow_core::{
///     DiffusionChernoff, Grid1D, GridFn1D,
///     resolvent::{LaplaceChernoffResolvent, LaplaceQuadrature},
/// };
///
/// let grid = Grid1D::new(-5.0, 5.0, 64).unwrap();
/// let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
/// let resolvent = LaplaceChernoffResolvent::new(
///     diff, 32, LaplaceQuadrature::GaussLaguerre32
/// ).unwrap();
/// let g = GridFn1D::from_fn(grid, |x: f64| (-x * x).exp());
/// let rg = resolvent.eval(1.0_f64, &g).unwrap();
/// assert_eq!(rg.values.len(), 64);
/// ```
#[derive(Debug, Clone)]
pub struct LaplaceChernoffResolvent<C, F = f64>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
{
    /// Inner Chernoff function approximating `exp(tA)`.
    pub inner: C,
    /// Chernoff truncation level `n`. Invariant: `n ≥ 1`.
    pub n: usize,
    /// Quadrature strategy.
    pub quadrature: LaplaceQuadrature<F>,
    _f: PhantomData<F>,
}

impl<C, F> LaplaceChernoffResolvent<C, F>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
{
    /// Construct and validate.
    ///
    /// # Errors
    ///
    /// - `DomainViolation` if `n == 0`.
    /// - `DomainViolation` if `quadrature == TrapezoidWithTail { t_max }` and
    ///   `t_max` is not finite or `≤ 0`.
    pub fn new(inner: C, n: usize, quadrature: LaplaceQuadrature<F>) -> Result<Self, SemiflowError> {
        if n == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "LaplaceChernoffResolvent n must be >= 1",
                value: 0.0,
            });
        }
        if let LaplaceQuadrature::TrapezoidWithTail { t_max } = quadrature {
            if !t_max.is_finite() || t_max <= F::zero() {
                return Err(SemiflowError::DomainViolation {
                    what: "TrapezoidWithTail t_max must be finite and > 0",
                    value: t_max.to_f64().unwrap_or(f64::NAN),
                });
            }
        }
        Ok(Self {
            inner,
            n,
            quadrature,
            _f: PhantomData,
        })
    }

    /// Evaluate `R̃_n(λ) g`, returning a fresh state.
    ///
    /// Validates `lambda.is_finite() && lambda > 0` (real-λ contract;
    /// complex `λ` is deferred to v4.0 B6 `SemiflowComplex`).
    ///
    /// # Errors
    ///
    /// - `DomainViolation` if `lambda ≤ 0` or not finite.
    /// - Propagates errors from `inner.apply_into`.
    pub fn eval(&self, lambda: F, g: &C::S) -> Result<C::S, SemiflowError>
    where
        C::S: Clone,
    {
        validate_lambda(lambda)?;
        match self.quadrature {
            LaplaceQuadrature::GaussLaguerre32 => self.eval_gauss_laguerre(lambda, g),
            LaplaceQuadrature::TrapezoidWithTail { t_max } => self.eval_trapezoid(lambda, g, t_max),
        }
    }

    /// Evaluate `[R̃_n(λ) g](x0)` at a single spatial point, returning a scalar.
    ///
    /// # Errors
    ///
    /// `DomainViolation` if `lambda ≤ 0`, not finite, or `x0` is empty.
    /// Propagates errors from `inner.apply_into` or `Sampleable::sample_at`.
    pub fn eval_at_point(
        &self,
        lambda: F,
        x0: &[F],
        g_at: &dyn Fn(&[F]) -> F,
    ) -> Result<F, SemiflowError>
    where
        C::S: Clone + Sampleable<F>,
    {
        validate_lambda(lambda)?;
        if x0.is_empty() {
            return Err(SemiflowError::DomainViolation {
                what: "eval_at_point: x0 must not be empty",
                value: 0.0,
            });
        }
        match self.quadrature {
            LaplaceQuadrature::GaussLaguerre32 => self.eval_ap_gl32(lambda, x0, g_at),
            LaplaceQuadrature::TrapezoidWithTail { t_max } => {
                self.eval_ap_trap(lambda, x0, g_at, t_max)
            }
        }
    }

    fn eval_gauss_laguerre(&self, lambda: F, g: &C::S) -> Result<C::S, SemiflowError>
    where
        C::S: Clone,
    {
        let mut acc = g.clone();
        acc.zero_into();
        let mut scratch: ScratchPool<F> = ScratchPool::new();
        let mut buf_a = g.clone();
        let mut buf_b = g.clone();
        let n_f = n_to_f::<F>(self.n);
        for k in 0..32 {
            let s_k = from_f64::<F>(GL32_NODES[k]);
            let w_k = from_f64::<F>(GL32_WEIGHTS[k]);
            let tau = s_k / (lambda * n_f);
            run_n_steps(
                &self.inner,
                tau,
                g,
                &mut buf_a,
                &mut buf_b,
                self.n,
                &mut scratch,
            )?;
            State::axpy_into(&mut acc, w_k / lambda, &buf_a);
        }
        Ok(acc)
    }

    fn eval_trapezoid(&self, lambda: F, g: &C::S, t_max: F) -> Result<C::S, SemiflowError>
    where
        C::S: Clone,
    {
        let mut acc = g.clone();
        acc.zero_into();
        let mut scratch: ScratchPool<F> = ScratchPool::new();
        let mut buf_a = g.clone();
        let mut buf_b = g.clone();
        let m = TRAPEZOID_N;
        let dt = t_max / n_to_f::<F>(m);
        let n_f = n_to_f::<F>(self.n);
        for j in 0..=m {
            let t_j = n_to_f::<F>(j) * dt;
            let tau = t_j / n_f;
            let e_lam = exp_neg(lambda * t_j);
            let weight = trapezoid_weight(j, m, dt);
            run_n_steps(
                &self.inner,
                tau,
                g,
                &mut buf_a,
                &mut buf_b,
                self.n,
                &mut scratch,
            )?;
            State::axpy_into(&mut acc, e_lam * weight, &buf_a);
        }
        Ok(acc)
    }

    // Point-eval: Gauss-Laguerre path (scalar accumulation).
    // Uses buf_b as grid prototype for fresh_from_fn after the first node.
    #[allow(clippy::too_many_arguments)]
    fn eval_ap_gl32(&self, lambda: F, x0: &[F], g_at: &dyn Fn(&[F]) -> F) -> Result<F, SemiflowError>
    where
        C::S: Clone + Sampleable<F>,
    {
        let n_f = n_to_f::<F>(self.n);
        let mut scratch: ScratchPool<F> = ScratchPool::new();
        let mut acc = F::zero();

        let s0 = from_f64::<F>(GL32_NODES[0]);
        let w0 = from_f64::<F>(GL32_WEIGHTS[0]);
        let tau0 = s0 / (lambda * n_f);
        let g0 = err_unsupported_g()?;
        let mut buf_a: C::S = g0;
        let mut buf_b = buf_a.clone();
        run_n_steps(
            &self.inner,
            tau0,
            &buf_a.fresh_from_fn(g_at)?,
            &mut buf_a,
            &mut buf_b,
            self.n,
            &mut scratch,
        )?;
        acc += (w0 / lambda) * buf_a.sample_at(x0)?;

        for k in 1..32 {
            let s_k = from_f64::<F>(GL32_NODES[k]);
            let w_k = from_f64::<F>(GL32_WEIGHTS[k]);
            let tau = s_k / (lambda * n_f);
            let gk = buf_b.fresh_from_fn(g_at)?;
            run_n_steps(
                &self.inner,
                tau,
                &gk,
                &mut buf_a,
                &mut buf_b,
                self.n,
                &mut scratch,
            )?;
            acc += (w_k / lambda) * buf_a.sample_at(x0)?;
        }
        Ok(acc)
    }

    // Point-eval: TrapezoidWithTail path (scalar accumulation).
    fn eval_ap_trap(
        &self,
        lambda: F,
        x0: &[F],
        g_at: &dyn Fn(&[F]) -> F,
        t_max: F,
    ) -> Result<F, SemiflowError>
    where
        C::S: Clone + Sampleable<F>,
    {
        let m = TRAPEZOID_N;
        let dt = t_max / n_to_f::<F>(m);
        let n_f = n_to_f::<F>(self.n);
        let mut scratch: ScratchPool<F> = ScratchPool::new();
        let mut acc = F::zero();
        let g0 = err_unsupported_g()?;
        let mut buf_a: C::S = g0;
        let mut buf_b = buf_a.clone();
        for j in 0..=m {
            let t_j = n_to_f::<F>(j) * dt;
            let tau = t_j / n_f;
            let e_lam = exp_neg(lambda * t_j);
            let weight = trapezoid_weight(j, m, dt);
            let gj = buf_b.fresh_from_fn(g_at)?;
            run_n_steps(
                &self.inner,
                tau,
                &gj,
                &mut buf_a,
                &mut buf_b,
                self.n,
                &mut scratch,
            )?;
            acc += e_lam * weight * buf_a.sample_at(x0)?;
        }
        Ok(acc)
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

fn validate_lambda<F: SemiflowFloat>(lambda: F) -> Result<(), SemiflowError> {
    if !lambda.is_finite() || lambda <= F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "lambda must be finite and > 0 (real-λ contract; complex λ deferred v4.0)",
            value: lambda.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

/// Stub: `eval_at_point` cannot build `C::S` without grid geometry embedded in C.
///
/// Callers should use [`LaplaceChernoffResolvent::eval`] + `sample_at()`.
/// Generic `g_at`-based construction deferred to v2.8 (ADR-0069 §"Limitations").
fn err_unsupported_g<T>() -> Result<T, SemiflowError> {
    Err(SemiflowError::Unsupported {
        feature: "eval_at_point: requires grid embedded in C; \
                  use eval() + sample_at() instead; \
                  generic g_at-construction deferred to v2.8 (ADR-0069)",
    })
}

/// Convert a `usize` index to `F`. Index counts ≤ 2^53 so the f64 cast is exact.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn n_to_f<F: SemiflowFloat>(n: usize) -> F {
    from_f64::<F>(n as f64)
}

/// Trapezoid weight: half-step at endpoints, full-step elsewhere.
#[inline]
fn trapezoid_weight<F: SemiflowFloat>(j: usize, m: usize, dt: F) -> F {
    if j == 0 || j == m {
        from_f64::<F>(0.5) * dt
    } else {
        dt
    }
}

/// Compute `exp(-x)`.
#[inline]
fn exp_neg<F: SemiflowFloat>(x: F) -> F {
    (-x).exp()
}

/// Run `n` Chernoff steps `(C(tau))^n g`, ping-ponging `buf_a`/`buf_b`.
///
/// After the call `buf_a` holds the result.
#[allow(clippy::too_many_arguments)]
fn run_n_steps<C, F>(
    inner: &C,
    tau: F,
    g: &C::S,
    buf_a: &mut C::S,
    buf_b: &mut C::S,
    n: usize,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
    C::S: Clone,
{
    buf_a.copy_from(g);
    for _ in 0..n {
        inner.apply_into(tau, buf_a, buf_b, scratch)?;
        core::mem::swap(buf_a, buf_b);
    }
    Ok(())
}

// Residual gate-wrapper and Sampleable<GridFn1D> split to resolvent_residual.rs.
pub use crate::resolvent_residual::LaplaceChernoffResolventResidual;
