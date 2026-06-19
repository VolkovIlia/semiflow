//! Private coupling trait and concrete impls for `NonSeparableMixedChernoff`.
//!
//! Extracted from `nonseparable_mixed.rs` to keep that file under 500 lines.

extern crate alloc;

use alloc::boxed::Box;

use crate::{float::SemiflowFloat, grid2d::Grid2D, grid_fn2d::GridFn2D};

use super::stencil::{cross_stencil_beta, cross_stencil_scalar};

// ---------------------------------------------------------------------------
// Private coupling trait — INTERNAL to nonseparable_mixed
// ---------------------------------------------------------------------------

/// Abstraction over the mixed-derivative coupling `coupling(x,y) · ∂_x∂_y`.
///
/// Two concrete impls live below:
/// - [`ScalarCoupling`]: constant `c` (v0.7.0 surface).
/// - [`BetaCoupling`]: position-dep `β(x,y)` (v0.9.0 surface).
///
/// Private to `nonseparable_mixed` for v2.2; promoted to `pub(crate)` in v2.3
/// for closure-backed coupling support (ADR-0058 §"Risks" → resolved v2.3).
pub(crate) trait MixedDerivOp<F: SemiflowFloat>: Send + Sync {
    /// Sup-norm bound on the coupling coefficient.
    fn norm_bound(&self) -> f64;
    /// `true` iff coupling is identically zero (fast-path to `Strang2D`).
    fn is_zero(&self) -> bool;
    /// Apply `M·f = coupling(x,y)·(∂_x∂_y f)` weighted stencil into `dst`.
    fn apply_mixed_into(&self, src: &GridFn2D<F>, dst: &mut GridFn2D<F>, grid: &Grid2D<F>);
    /// Clone to a `Box<dyn MixedDerivOp<F>>`.
    fn clone_box(&self) -> Box<dyn MixedDerivOp<F>>;
    /// Debug label (for `Debug` impl on the outer struct).
    fn debug_label(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// ScalarCoupling — constant c
// ---------------------------------------------------------------------------

/// Constant scalar coupling: `M = c · ∂_x∂_y`.
pub(crate) struct ScalarCoupling<F: SemiflowFloat> {
    pub(crate) c: fn(F, F) -> F,
    pub(crate) c_norm_bound: f64,
    pub(crate) is_zero: bool,
}

impl<F: SemiflowFloat> MixedDerivOp<F> for ScalarCoupling<F> {
    fn norm_bound(&self) -> f64 {
        self.c_norm_bound
    }

    fn is_zero(&self) -> bool {
        self.is_zero
    }

    fn apply_mixed_into(&self, src: &GridFn2D<F>, dst: &mut GridFn2D<F>, grid: &Grid2D<F>) {
        cross_stencil_scalar(src, dst, grid, self.c);
    }

    fn clone_box(&self) -> Box<dyn MixedDerivOp<F>> {
        Box::new(ScalarCoupling {
            c: self.c,
            c_norm_bound: self.c_norm_bound,
            is_zero: self.is_zero,
        })
    }

    fn debug_label(&self) -> &'static str {
        "ScalarCoupling"
    }
}

// ---------------------------------------------------------------------------
// BetaCoupling — position-dependent β(x,y)
// ---------------------------------------------------------------------------

/// Position-dependent coupling: `M_β = β(x,y) · ∂_x∂_y`.
pub(crate) struct BetaCoupling<F: SemiflowFloat> {
    pub(crate) beta: fn(F, F) -> F,
    pub(crate) beta_norm_bound: f64,
    pub(crate) is_zero: bool,
}

impl<F: SemiflowFloat> MixedDerivOp<F> for BetaCoupling<F> {
    fn norm_bound(&self) -> f64 {
        self.beta_norm_bound
    }

    fn is_zero(&self) -> bool {
        self.is_zero
    }

    fn apply_mixed_into(&self, src: &GridFn2D<F>, dst: &mut GridFn2D<F>, grid: &Grid2D<F>) {
        cross_stencil_beta(src, dst, grid, self.beta);
    }

    fn clone_box(&self) -> Box<dyn MixedDerivOp<F>> {
        Box::new(BetaCoupling {
            beta: self.beta,
            beta_norm_bound: self.beta_norm_bound,
            is_zero: self.is_zero,
        })
    }

    fn debug_label(&self) -> &'static str {
        "BetaCoupling"
    }
}
