//! [`ChernoffSemigroup`]: the two-parameter executor kept for the binding crates
//! (a thin wrapper over the [`super::Evolver`] iteration).

use core::marker::PhantomData;

use super::ChernoffFunction;
use crate::{error::SemiflowError, scratch::ScratchPool, state::State};

// ---------------------------------------------------------------------------
// ChernoffSemigroup backward-compat alias (v3.0 — thin wrapper over Evolver)
// ---------------------------------------------------------------------------

/// Chernoff executor with public `func` and `n` fields.
///
/// Two-parameter form `ChernoffSemigroup<C, S>` preserving the
/// `pub func: C` field access pattern used throughout the binding crates.
/// For the generic-over-F single-parameter form, see [`Evolver<C, F>`](super::Evolver).
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
        // ADR-0215: one τ-keyed plan for all n steps (bit-identical). Dropped with
        // the local pool, not returned to it.
        let mut plan = crate::step_plan::PencilPlan::with_uses(self.n);
        for _ in 0..self.n {
            if src_is_a {
                plan.apply(&self.func, tau, &buf_a, &mut buf_b, &mut scratch)?;
            } else {
                plan.apply(&self.func, tau, &buf_b, &mut buf_a, &mut scratch)?;
            }
            src_is_a = !src_is_a;
        }
        drop(plan);
        Ok(if src_is_a { buf_a } else { buf_b })
    }
}
