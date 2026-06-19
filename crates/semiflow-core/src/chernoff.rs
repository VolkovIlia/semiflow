//! Core Chernoff abstraction: [`ChernoffFunction`] trait and [`Evolver`] executor.
//!
//! The Chernoff product formula (Chernoff 1968,
//! DOI [10.1016/0022-1236(68)90020-7](https://doi.org/10.1016/0022-1236(68)90020-7))
//! guarantees `(S(t/n))^n f → e^{tA} f` strongly when `S` is a Chernoff
//! function for the generator `A`. [`Evolver::evolve_into`] implements
//! this iteration.
//!
//! ## v3.0 BREAKING (ADR-0074)
//!
//! - `apply` REMOVED from the trait (moves to per-impl inherent `apply_chernoff`).
//! - `Clone` bound on `Self::S` REMOVED from the trait (opt-in at call site).
//! - `growth()` now returns [`Growth<F>`] (was `(f64, f64)` tuple).
//! - `order()` REQUIRED, no default (Wave A audit confirmed all impls explicit).
//! - `ChernoffSemigroup<C>` RENAMED to [`Evolver<C, F>`] (v2.x alias hard-removed at v4.0, ADR-0084).
//!
//! ## Usage pattern
//!
//! 1. Choose a [`ChernoffFunction`] implementor for your PDE operator
//!    (e.g. [`crate::DiffusionChernoff`], [`crate::StrangSplit`]).
//! 2. Wrap it in [`Evolver::new`] with an iteration count `n`.
//! 3. Call [`Evolver::evolve_into`]`(t, &u0, &mut dst, &mut scratch)`.
//!
//! For one-shot allocating use, call `func.apply_chernoff(tau, &f)` directly
//! on any type whose state is `Clone`.
//!
//! See `contracts/semiflow-core.math.md` §5–6 for the approximation-order
//! analysis (Theorem 6, inequality (9)).

use core::marker::PhantomData;

use crate::{error::SemiflowError, float::SemiflowFloat, scratch::ScratchPool, state::State};

// ---------------------------------------------------------------------------
// Growth<F>
// ---------------------------------------------------------------------------

/// Growth bound `‖F(τ)‖ ≤ multiplier · exp(omega · τ)`.
///
/// v3.0 BREAKING (ADR-0074): replaces the v2.x `(f64, f64)` tuple return of
/// `ChernoffFunction::growth`. Named fields prevent the tuple-order footgun
/// (`growth().0` vs `growth().1`). Generic over `F: SemiflowFloat = f64` per
/// ADR-0025; for f64-only impls use `Growth::new(m, o)` (the default).
///
/// ## Composition
///
/// For a composed kernel such as `Strang2D<X, Y, F>`, the growth is the
/// product of the per-axis growth bounds plus a step-scaling factor.
/// Consult `contracts/semiflow-core.tensor.yaml` §3 for the exact formula.
///
/// ## Migration from v2.x
///
/// ```rust,no_run
/// // v2.x (tuple return — hard-removed at v4.0, ADR-0084):
/// // let (m, om) = func.growth();
///
/// // v3.0:
/// // let g = func.growth();
/// // let (m, om) = (g.multiplier, g.omega);
/// // or: let Growth { multiplier: m, omega: om } = func.growth();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Growth<F: SemiflowFloat = f64> {
    /// M in `‖F(τ)‖ ≤ M · exp(ω · τ)`. Invariant: `M ≥ 1`.
    pub multiplier: F,
    /// ω in `‖F(τ)‖ ≤ M · exp(ω · τ)`. Finite.
    pub omega: F,
}

impl<F: SemiflowFloat> Growth<F> {
    /// Construct from named fields.
    #[inline]
    pub fn new(multiplier: F, omega: F) -> Self {
        Self { multiplier, omega }
    }

    /// Contraction bound `(1, 0)` — positivity-preserving step.
    #[inline]
    #[must_use]
    pub fn contraction() -> Self {
        Self {
            multiplier: F::one(),
            omega: F::zero(),
        }
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction trait
// ---------------------------------------------------------------------------

/// A Chernoff function `S: [0, ∞) → ℒ(X)` approximating `e^{tA}`.
///
/// ## Mathematical contract
///
/// An operator family `S(τ)` is a Chernoff function for generator `A` when:
/// - `S(0) = I` (identity at zero step),
/// - `τ ↦ S(τ) f` is strongly continuous,
/// - `S'(0) f = A f` for all `f` in a core `D ⊂ D(A)`.
///
/// Under these conditions the Chernoff product formula guarantees
/// `‖(S(t/n))^n f − e^{tA} f‖ = O(1/n)` (Theorem 6, Remizov 2025).
///
/// ## v3.0 surface (ADR-0074)
///
/// Three required methods:
///
/// - [`apply_into`](Self::apply_into) — zero-alloc single step (the primary hot-path method).
/// - [`order`](Self::order) — declared consistency order m ≥ 1. REQUIRED, no default.
/// - [`growth`](Self::growth) — growth bound `(M, ω)`, returns [`Growth<F>`].
///
/// The v2.x `apply` method is NOT a trait method in v3.0. It moves to a
/// per-impl inherent method `apply_chernoff(τ, &src)` (available wherever
/// `Self::S: Clone`). The v2.x `apply` shim was hard-removed at v4.0 (ADR-0084).
///
/// ## Generics (ADR-0025, v0.9.0 Block D Wave 4)
///
/// `ChernoffFunction<F: SemiflowFloat = f64>` — the `= f64` default keeps
/// existing call-sites compiling unchanged. The `Clone` bound on `Self::S`
/// is NOT required by the trait (v3.0 cleanup); add it only at call sites
/// that need the allocating convenience.
///
/// ## Example
///
/// ```rust
/// use semiflow_core::{Grid1D, GridFn1D, DiffusionChernoff, ChernoffFunction};
/// let grid = Grid1D::new(-5.0, 5.0, 64).unwrap();
/// let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// // v3.0 allocating convenience (apply_chernoff on the inherent impl):
/// let u1 = diff.apply_chernoff(0.01, &u0).unwrap();
/// assert_eq!(u1.values.len(), 64);
/// ```
#[allow(clippy::module_name_repetitions)]
pub trait ChernoffFunction<F: SemiflowFloat = f64> {
    /// The state space type on which `S(tau)` acts.
    ///
    /// v3.0: no `Clone` bound here (removed per ADR-0074). Add `where Self::S: Clone`
    /// only at call sites that need the allocating convenience.
    type S: State<F>;

    /// Zero-alloc apply: `dst := S(tau) src`. Caller provides pre-allocated
    /// `dst` and a scratch pool for hot-path temporaries (ADR-0041 Wave 1).
    ///
    /// This is the SOLE apply method on the v3.0 trait.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0`, `tau` is `NaN`/`Inf`,
    ///   or if `src` contains `NaN`/`Inf`.
    /// - [`SemiflowError::Unsupported`] from the underlying interpolation policy.
    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;

    /// Declared consistency order m ≥ 1: `‖S(τ) f − exp(τA) f‖ = O(τ^{m+1})`.
    ///
    /// REQUIRED method; no default (ADR-0074 Wave A audit confirmed all v2.x impls
    /// already declare an explicit body).
    fn order(&self) -> u32;

    /// Growth bound `(M, ω)` with `‖S(τ)‖ ≤ M · exp(ω · τ)`.
    ///
    /// v3.0: returns [`Growth<F>`] (was `(f64, f64)` in v2.x).
    /// Invariant: `M >= 1`, `ω` finite.
    fn growth(&self) -> Growth<F>;

    /// Apply the TRANSPOSE (adjoint) semigroup `exp(τ Aᵀ) src` into `dst`.
    ///
    /// Default: returns [`SemiflowError::UnsupportedOperation`].
    ///
    /// Types that can expose their transpose action SHOULD NOT rely on this
    /// default — they should implement [`crate::adjoint::AdjointApply<F>`]
    /// (supertrait with no default) AND override this method. The supertrait
    /// provides compile-time guarantees that the implementation is genuine.
    ///
    /// This default exists to satisfy [`AdjointChernoff::apply_into`]'s
    /// dispatch without requiring specialization. See ADR-0114.
    ///
    /// # Errors
    ///
    /// Default returns [`SemiflowError::UnsupportedOperation`].
    /// Overrides MUST document their own error conditions.
    #[allow(unused_variables)]
    fn apply_adjoint_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        Err(crate::error::SemiflowError::UnsupportedOperation {
            what: "apply_adjoint_into: no transpose-apply primitive for this type; \
                   implement AdjointApply<F> or use new_self_adjoint for symmetric inners \
                   (ADR-0114)",
        })
    }
}

// ---------------------------------------------------------------------------
// apply_chernoff inherent blanket (per ADR-0074)
// ---------------------------------------------------------------------------

/// Extension trait providing the allocating `apply_chernoff` convenience.
///
/// Available on any `ChernoffFunction<F>` whose `S: Clone`. This replaces the
/// v2.x trait method `apply(τ, &f)` as a blanket impl. Not part of the
/// `ChernoffFunction` trait surface; the `Clone` bound is opt-in at call site.
///
/// Prefer [`ChernoffFunction::apply_into`] on hot paths.
pub trait ApplyChernoffExt<F: SemiflowFloat>: ChernoffFunction<F>
where
    Self::S: Clone,
{
    /// Allocating single-step apply: returns `S(tau) src` as a fresh state.
    ///
    /// Convenience for non-hot-path callers (one-shot evaluations, tests,
    /// REPL exploration). Hot-path callers MUST use `apply_into`.
    ///
    /// # Errors
    /// Same conditions as [`ChernoffFunction::apply_into`].
    fn apply_chernoff(&self, tau: F, src: &Self::S) -> Result<Self::S, SemiflowError> {
        let mut dst = src.clone();
        let mut scratch = ScratchPool::<F>::new();
        self.apply_into(tau, src, &mut dst, &mut scratch)?;
        Ok(dst)
    }
}

impl<C, F> ApplyChernoffExt<F> for C
where
    C: ChernoffFunction<F>,
    C::S: Clone,
    F: SemiflowFloat,
{
}

// ---------------------------------------------------------------------------
// Evolver<C, F>
// ---------------------------------------------------------------------------

/// Iterates a [`ChernoffFunction`] `n` times with step `τ = t / n`,
/// producing the Chernoff approximant `(S(t/n))^n f` of `exp(tA) f`.
///
/// RENAMED from v2.x `ChernoffSemigroup<C>` (ADR-0074). The v2.x type alias
/// `ChernoffSemigroup<C> = Evolver<C, f64>` was hard-removed at v4.0 (ADR-0084).
///
/// ## Why the rename?
///
/// "Semigroup" is mathematically inaccurate: the type evolves an n-step
/// Chernoff iterate `(S(t/n))^n`, which CONVERGES to the semigroup `exp(tA)`
/// but is itself NOT a semigroup. "Evolver" aligns with v2.6/v2.7/v2.8
/// single-noun wrapper naming (`KillingChernoff`, `HowlandLift`,
/// `LaplaceChernoffResolvent`, `ReflectedHeatChernoff`, `ManifoldChernoff`).
///
/// ## Example
///
/// ```rust
/// use semiflow_core::{Grid1D, GridFn1D, DiffusionChernoff, Evolver, State, ScratchPool};
/// let grid = Grid1D::new(-8.0, 8.0, 200).unwrap();
/// let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
/// let evolver = Evolver::new(diff, 100).unwrap();
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// let mut dst = u0.clone();
/// let mut scratch = ScratchPool::new();
/// evolver.evolve_into(1.0, &u0, &mut dst, &mut scratch).unwrap();
/// ```
#[allow(clippy::module_name_repetitions)]
pub struct Evolver<C, F: SemiflowFloat = f64>
where
    C: ChernoffFunction<F>,
{
    /// The Chernoff function to iterate. Public for inspection.
    pub func: C,
    /// Number of iterations. Invariant: `n >= 1`.
    pub n: usize,
    _phantom: PhantomData<F>,
}

impl<C, F> Evolver<C, F>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
{
    /// Construct with validation.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `n == 0`.
    pub fn new(func: C, n: usize) -> Result<Self, SemiflowError> {
        if n == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "Evolver n must be >= 1",
                value: 0.0,
            });
        }
        // Pre-warm parallel singletons (allocation-free after first call — ADR-0041 AC-4).
        #[cfg(feature = "parallel")]
        {
            let _ = crate::parallel1d::min_points_per_thread();
            let _ = crate::parallel1d::available_parallelism_cached();
        }
        Ok(Self {
            func,
            n,
            _phantom: PhantomData,
        })
    }

    /// Zero-alloc `(S(t/n))^n f` via ping-pong scratch (ADR-0041 Wave 1).
    ///
    /// `dst` receives the final state; caller manages the `ScratchPool`.
    /// Requires `C::S: Clone` for the ping-pong buffer allocation at the
    /// call site — explicit opt-in (ADR-0074 Clone cleanup).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `t < 0` or non-finite.
    /// - Errors from `apply_into` propagate unchanged.
    pub fn evolve_into(
        &self,
        t: F,
        src: &C::S,
        dst: &mut C::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>
    where
        C::S: Clone,
    {
        validate_t(t)?;
        #[allow(clippy::cast_precision_loss)]
        let tau = t / crate::float::from_f64::<F>(self.n as f64);
        let mut buf_a: C::S = src.clone();
        let mut buf_b: C::S = src.clone();
        buf_b.zero_into();
        let mut src_is_a = true;
        for _ in 0..self.n {
            if src_is_a {
                self.func.apply_into(tau, &buf_a, &mut buf_b, scratch)?;
            } else {
                self.func.apply_into(tau, &buf_b, &mut buf_a, scratch)?;
            }
            src_is_a = !src_is_a;
        }
        let result = if src_is_a { &buf_a } else { &buf_b };
        dst.copy_from(result);
        Ok(())
    }

    /// Allocating `(S(t/n))^n f` — convenience for non-hot-path callers.
    ///
    /// Returns the final state as a freshly cloned buffer. Hot-path callers
    /// MUST use [`evolve_into`](Self::evolve_into).
    ///
    /// # Errors
    /// Same as [`evolve_into`](Self::evolve_into).
    pub fn evolve(&self, t: F, src: &C::S) -> Result<C::S, SemiflowError>
    where
        C::S: Clone,
    {
        let mut dst = src.clone();
        let mut scratch = ScratchPool::<F>::new();
        self.evolve_into(t, src, &mut dst, &mut scratch)?;
        Ok(dst)
    }
}

// ---------------------------------------------------------------------------
// ChernoffSemigroup backward-compat alias (v3.0 — thin wrapper over Evolver)
// ---------------------------------------------------------------------------

/// Chernoff executor with public `func` and `n` fields.
///
/// Two-parameter form `ChernoffSemigroup<C, S>` preserving the
/// `pub func: C` field access pattern used throughout the binding crates.
/// For the generic-over-F single-parameter form, see [`Evolver<C, F>`].
pub struct ChernoffSemigroup<C, S>
where
    C: ChernoffFunction<f64, S = S>,
    S: State<f64> + Clone,
{
    /// The Chernoff function (public field for direct access by binding crates).
    pub func: C,
    /// Number of iterations. Invariant: `n >= 1`.
    pub n: usize,
    _state: PhantomData<S>,
}

impl<C, S> ChernoffSemigroup<C, S>
where
    C: ChernoffFunction<f64, S = S>,
    S: State<f64> + Clone,
{
    /// Construct with validation (v2.x API).
    ///
    /// Pre-warms parallel singletons (ADR-0041 AC-4).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `n == 0`.
    pub fn new(func: C, n: usize) -> Result<Self, SemiflowError> {
        if n == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "ChernoffSemigroup n must be >= 1",
                value: 0.0,
            });
        }
        #[cfg(feature = "parallel")]
        {
            let _ = crate::parallel1d::min_points_per_thread();
            let _ = crate::parallel1d::available_parallelism_cached();
        }
        Ok(Self {
            func,
            n,
            _state: PhantomData,
        })
    }

    /// Compute `(S(t/n))^n f` — v2.x `evolve` API.
    ///
    /// Ping-pong zero-alloc iteration (ADR-0041 Wave 1).
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `t < 0` or non-finite.
    /// - Errors from `apply_into` propagate unchanged.
    pub fn evolve(&self, t: f64, f: &S) -> Result<S, SemiflowError> {
        if !t.is_finite() || t < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "t must be finite and >= 0",
                value: t,
            });
        }
        #[allow(clippy::cast_precision_loss)]
        let tau = t / self.n as f64;
        let mut buf_a: S = f.clone();
        let mut buf_b: S = f.clone();
        buf_b.zero_into();
        let mut scratch: ScratchPool<f64> = ScratchPool::new();
        let mut src_is_a = true;
        for _ in 0..self.n {
            if src_is_a {
                self.func
                    .apply_into(tau, &buf_a, &mut buf_b, &mut scratch)?;
            } else {
                self.func
                    .apply_into(tau, &buf_b, &mut buf_a, &mut scratch)?;
            }
            src_is_a = !src_is_a;
        }
        Ok(if src_is_a { buf_a } else { buf_b })
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

#[inline]
fn validate_t<F: SemiflowFloat>(t: F) -> Result<(), SemiflowError> {
    if !t.is_finite() || t < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "t must be finite and >= 0",
            value: t.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}
