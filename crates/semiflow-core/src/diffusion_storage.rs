//! Internal coefficient storage for [`crate::DiffusionChernoff`] and siblings.
//!
//! [`Storage<F>`] holds three coefficient functions `a`, `a'`, `a''`.
//! [`Storage2<F>`] holds two coefficient functions (e.g. `b`, `c` for drift-reaction).
//! Each enum has:
//! - zero-heap fn-pointers (legacy `new` path, v0.3.0+),
//! - `Arc`-wrapped `dyn Fn` closures (`with_closure` path, v0.12.0+).
//!
//! Keeping these types in their own file satisfies the 500-line suckless cap on
//! the caller modules without altering any public API or runtime behaviour.

use alloc::sync::Arc;

use crate::float::SemiflowFloat;

/// Internal storage for the three coefficient functions of `DiffusionChernoff`.
///
/// - `FnPtr` — zero-heap, branch-predictor-friendly (legacy `new` path, v0.3.0+).
/// - `Closure` — `Arc`-wrapped `dyn Fn + Send + Sync` (`with_closure` path, v0.12.0+).
/// - `ConstA` — constant diffusion coefficient; `a'≡0, a''≡0` by definition so
///   the S-shift inner loop and the ζ-A correction are both skipped (D1, v0.13.0).
///
/// `Clone` is cheap: `FnPtr` copies pointers; `Closure` increments Arc reference counts;
/// `ConstA` copies a scalar.
///
/// ## WASM note (ADR-0034 §"Per-binding plan")
///
/// WASM single-thread path: the `semiflow-wasm` crate wraps `js_sys::Function` in a
/// newtype with `unsafe impl Send + Sync` (permitted there by `#![allow(unsafe_code)]`)
/// then calls `DiffusionChernoff::with_closure`. `semiflow-core` does not provide a
/// non-`Send+Sync` variant because `#![deny(unsafe_code)]` prevents explicit unsafe impls
/// here; the WASM crate carries the safety contract.
pub(crate) enum Storage<F: SemiflowFloat> {
    /// Zero-heap fn-pointer variant (legacy `new` path).
    FnPtr {
        /// Coefficient function `a(x)`.
        a: fn(F) -> F,
        /// First derivative `a'(x)`.
        a_prime: fn(F) -> F,
        /// Second derivative `a''(x)`.
        a_double_prime: fn(F) -> F,
    },
    // Arc lets Clone cheaply share the closure without requiring Fn: Clone.
    /// Heap-owned closure variant (`with_closure` path).
    Closure {
        /// Coefficient function `a(x)`.
        a: Arc<dyn Fn(F) -> F + Send + Sync + 'static>,
        /// First derivative `a'(x)`.
        a_prime: Arc<dyn Fn(F) -> F + Send + Sync + 'static>,
        /// Second derivative `a''(x)`.
        a_double_prime: Arc<dyn Fn(F) -> F + Send + Sync + 'static>,
    },
    /// Constant-a fast path (v0.13.0, D1): stores only the scalar value.
    ///
    /// Because `a'(x) ≡ 0` and `a''(x) ≡ 0`, the S-shift inner loop in the
    /// γ-A baseline reduces to `x_pre = x` (no shift), and the entire ζ-A
    /// τ²-correction evaluates to exactly zero. Both reductions are implemented
    /// in `diffusion.rs` by pattern-matching on this variant.
    ConstA {
        /// Constant diffusion coefficient value.
        a_value: F,
    },
}

impl<F: SemiflowFloat> Storage<F> {
    /// Evaluate `a(x)` from whichever variant is active.
    #[inline]
    pub(crate) fn eval_a(&self, x: F) -> F {
        match self {
            Self::FnPtr { a, .. } => a(x),
            Self::Closure { a, .. } => a(x),
            Self::ConstA { a_value } => *a_value,
        }
    }

    /// Evaluate `a'(x)` from whichever variant is active.
    #[inline]
    pub(crate) fn eval_ap(&self, x: F) -> F {
        match self {
            Self::FnPtr { a_prime, .. } => a_prime(x),
            Self::Closure { a_prime, .. } => a_prime(x),
            Self::ConstA { .. } => F::zero(),
        }
    }

    /// Evaluate `a''(x)` from whichever variant is active.
    #[inline]
    pub(crate) fn eval_app(&self, x: F) -> F {
        match self {
            Self::FnPtr { a_double_prime, .. } => a_double_prime(x),
            Self::Closure { a_double_prime, .. } => a_double_prime(x),
            Self::ConstA { .. } => F::zero(),
        }
    }
}

impl<F: SemiflowFloat> Clone for Storage<F> {
    fn clone(&self) -> Self {
        match self {
            Self::FnPtr {
                a,
                a_prime,
                a_double_prime,
            } => Self::FnPtr {
                a: *a,
                a_prime: *a_prime,
                a_double_prime: *a_double_prime,
            },
            Self::Closure {
                a,
                a_prime,
                a_double_prime,
            } => Self::Closure {
                a: Arc::clone(a),
                a_prime: Arc::clone(a_prime),
                a_double_prime: Arc::clone(a_double_prime),
            },
            Self::ConstA { a_value } => Self::ConstA { a_value: *a_value },
        }
    }
}

// Storage<F>: Send + Sync is auto-derived by the compiler:
// - FnPtr variant: `fn(F) -> F` is always Send + Sync.
// - Closure variant: `Arc<dyn Fn(F) -> F + Send + Sync>` is Send + Sync by
//   construction (bounds enforced at `with_closure`'s call-site).
// - ConstA variant: `F: SemiflowFloat` implies `F: Send + Sync`.
// No explicit unsafe impl needed.

// ---------------------------------------------------------------------------
// Storage2<F> — two-coefficient storage (used by DriftReactionChernoff).
// ---------------------------------------------------------------------------

/// Internal storage for two coefficient functions (e.g. `b`, `c`).
///
/// Used by [`crate::DriftReactionChernoff::with_closure`] to store heap-owned
/// closures alongside the existing zero-heap fn-pointer `new` path.
///
/// `Clone` is cheap: `FnPtr` copies pointers; `Closure` increments Arc ref counts.
pub(crate) enum Storage2<F: SemiflowFloat> {
    /// Zero-heap fn-pointer variant (legacy `new` path).
    FnPtr {
        /// First coefficient (e.g. `b(x)`).
        f0: fn(F) -> F,
        /// Second coefficient (e.g. `c(x)`).
        f1: fn(F) -> F,
    },
    /// Heap-owned closure variant (`with_closure` path).
    Closure {
        /// First coefficient closure.
        f0: Arc<dyn Fn(F) -> F + Send + Sync + 'static>,
        /// Second coefficient closure.
        f1: Arc<dyn Fn(F) -> F + Send + Sync + 'static>,
    },
}

impl<F: SemiflowFloat> Storage2<F> {
    /// Evaluate first coefficient at `x`.
    #[inline]
    pub(crate) fn eval0(&self, x: F) -> F {
        match self {
            Self::FnPtr { f0, .. } => f0(x),
            Self::Closure { f0, .. } => f0(x),
        }
    }

    /// Evaluate second coefficient at `x`.
    #[inline]
    pub(crate) fn eval1(&self, x: F) -> F {
        match self {
            Self::FnPtr { f1, .. } => f1(x),
            Self::Closure { f1, .. } => f1(x),
        }
    }
}

impl<F: SemiflowFloat> Clone for Storage2<F> {
    fn clone(&self) -> Self {
        match self {
            Self::FnPtr { f0, f1 } => Self::FnPtr { f0: *f0, f1: *f1 },
            Self::Closure { f0, f1 } => Self::Closure {
                f0: Arc::clone(f0),
                f1: Arc::clone(f1),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Storage3<F> — three-coefficient storage (used by ShiftChernoff1D).
// ---------------------------------------------------------------------------

/// Internal storage for three independent coefficient functions (a, b, c).
///
/// Used by [`crate::ShiftChernoff1D::with_closure`] to store heap-owned
/// closures alongside the existing zero-heap fn-pointer `new` path.
pub(crate) enum Storage3<F: SemiflowFloat> {
    /// Zero-heap fn-pointer variant.
    FnPtr {
        /// First coefficient (e.g. `a(x)`).
        f0: fn(F) -> F,
        /// Second coefficient (e.g. `b(x)`).
        f1: fn(F) -> F,
        /// Third coefficient (e.g. `c(x)`).
        f2: fn(F) -> F,
    },
    /// Heap-owned closure variant.
    Closure {
        /// First coefficient closure.
        f0: Arc<dyn Fn(F) -> F + Send + Sync + 'static>,
        /// Second coefficient closure.
        f1: Arc<dyn Fn(F) -> F + Send + Sync + 'static>,
        /// Third coefficient closure.
        f2: Arc<dyn Fn(F) -> F + Send + Sync + 'static>,
    },
}

impl<F: SemiflowFloat> Storage3<F> {
    /// Evaluate first coefficient at `x`.
    #[inline]
    pub(crate) fn eval0(&self, x: F) -> F {
        match self {
            Self::FnPtr { f0, .. } => f0(x),
            Self::Closure { f0, .. } => f0(x),
        }
    }

    /// Evaluate second coefficient at `x`.
    #[inline]
    pub(crate) fn eval1(&self, x: F) -> F {
        match self {
            Self::FnPtr { f1, .. } => f1(x),
            Self::Closure { f1, .. } => f1(x),
        }
    }

    /// Evaluate third coefficient at `x`.
    #[inline]
    pub(crate) fn eval2(&self, x: F) -> F {
        match self {
            Self::FnPtr { f2, .. } => f2(x),
            Self::Closure { f2, .. } => f2(x),
        }
    }
}

impl<F: SemiflowFloat> Clone for Storage3<F> {
    fn clone(&self) -> Self {
        match self {
            Self::FnPtr { f0, f1, f2 } => Self::FnPtr {
                f0: *f0,
                f1: *f1,
                f2: *f2,
            },
            Self::Closure { f0, f1, f2 } => Self::Closure {
                f0: Arc::clone(f0),
                f1: Arc::clone(f1),
                f2: Arc::clone(f2),
            },
        }
    }
}
