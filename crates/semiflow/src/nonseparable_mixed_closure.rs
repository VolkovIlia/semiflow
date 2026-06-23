//! Closure-backed coupling implementations for [`NonSeparableMixedChernoff`].
//!
//! Provides [`ClosureScalarCoupling`] and [`ClosureBetaCoupling`] that store
//! `Arc<dyn Fn(F, F) -> F + Send + Sync>` coupling functions.  These are used
//! by the Python bindings (v2.3 Phase 4) where sampled arrays must be captured
//! by a closure — bare function pointers are insufficient.
//!
//! Free-function constructors [`with_closure_c`] and [`with_closure_beta`] wrap
//! `NonSeparableMixedChernoff::with_coupling` (the `pub(crate)` escape hatch
//! added in v2.3, ADR-0058 §"v2.3 promotions").

extern crate alloc;

use alloc::boxed::Box;
use alloc::sync::Arc;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::SemiflowFloat,
    grid2d::Grid2D,
    grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
    nonseparable_mixed::{cross_stencil_scalar, MixedDerivOp, NonSeparableMixedChernoff},
};

// ---------------------------------------------------------------------------
// ClosureScalarCoupling — Arc<dyn Fn> scalar variant
// ---------------------------------------------------------------------------

/// Arc-closure-backed scalar coupling `M = c(x,y) · ∂_x∂_y`.
///
/// Unlike `ScalarCoupling` (which stores a bare `fn` pointer), this variant
/// captures sampled array data via an `Arc`-wrapped heap closure.
#[doc(hidden)]
pub struct ClosureScalarCoupling<F: SemiflowFloat> {
    /// Coupling function `c(x, y)`.
    c: Arc<dyn Fn(F, F) -> F + Send + Sync + 'static>,
    /// Sup-norm bound on `|c|`.
    c_norm_bound: f64,
    /// True when `c ≡ 0` (fast-path to `Strang2D`).
    is_zero: bool,
}

impl<F: SemiflowFloat> MixedDerivOp<F> for ClosureScalarCoupling<F> {
    fn norm_bound(&self) -> f64 {
        self.c_norm_bound
    }

    fn is_zero(&self) -> bool {
        self.is_zero
    }

    fn apply_mixed_into(&self, src: &GridFn2D<F>, dst: &mut GridFn2D<F>, grid: &Grid2D<F>) {
        let c_ref = Arc::clone(&self.c);
        cross_stencil_scalar(src, dst, grid, move |x, y| c_ref(x, y));
    }

    fn clone_box(&self) -> Box<dyn MixedDerivOp<F>> {
        Box::new(ClosureScalarCoupling {
            c: Arc::clone(&self.c),
            c_norm_bound: self.c_norm_bound,
            is_zero: self.is_zero,
        })
    }

    fn debug_label(&self) -> &'static str {
        "ClosureScalarCoupling"
    }
}

// ---------------------------------------------------------------------------
// ClosureBetaCoupling — Arc<dyn Fn> position-dependent variant
// ---------------------------------------------------------------------------

/// Arc-closure-backed position-dependent coupling `M_β = β(x,y) · ∂_x∂_y`.
#[doc(hidden)]
pub struct ClosureBetaCoupling<F: SemiflowFloat> {
    /// Anisotropy function `β(x, y)`.
    beta: Arc<dyn Fn(F, F) -> F + Send + Sync + 'static>,
    /// Sup-norm bound on `|β|`.
    beta_norm_bound: f64,
    /// True when `β ≡ 0` (fast-path to `Strang2D`).
    is_zero: bool,
}

impl<F: SemiflowFloat> MixedDerivOp<F> for ClosureBetaCoupling<F> {
    fn norm_bound(&self) -> f64 {
        self.beta_norm_bound
    }

    fn is_zero(&self) -> bool {
        self.is_zero
    }

    fn apply_mixed_into(&self, src: &GridFn2D<F>, dst: &mut GridFn2D<F>, grid: &Grid2D<F>) {
        let beta_ref = Arc::clone(&self.beta);
        cross_stencil_scalar(src, dst, grid, move |x, y| beta_ref(x, y));
    }

    fn clone_box(&self) -> Box<dyn MixedDerivOp<F>> {
        Box::new(ClosureBetaCoupling {
            beta: Arc::clone(&self.beta),
            beta_norm_bound: self.beta_norm_bound,
            is_zero: self.is_zero,
        })
    }

    fn debug_label(&self) -> &'static str {
        "ClosureBetaCoupling"
    }
}

// ---------------------------------------------------------------------------
// pub(crate) constructors
// ---------------------------------------------------------------------------

/// Build a `NonSeparableMixedChernoff` with a scalar closure coupling `c(x,y)`.
///
/// The closure is stored via `Arc` for cheap `Clone` (used by `ChernoffSemigroup`).
///
/// # Errors
/// Returns [`SemiflowError::DomainViolation`] if `c_norm_bound` is not finite or
/// is negative.
pub fn with_closure_c<X, Y, F>(
    x_inner: X,
    y_inner: Y,
    c: Arc<dyn Fn(F, F) -> F + Send + Sync + 'static>,
    c_norm_bound: f64,
    grid: Grid2D<F>,
) -> Result<NonSeparableMixedChernoff<X, Y, F>, SemiflowError>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    F: SemiflowFloat,
{
    validate_norm_bound(c_norm_bound, "c_norm_bound")?;
    let coupling: Box<dyn MixedDerivOp<F>> = Box::new(ClosureScalarCoupling {
        c,
        c_norm_bound,
        is_zero: c_norm_bound == 0.0,
    });
    Ok(NonSeparableMixedChernoff::with_coupling(
        x_inner, y_inner, grid, coupling,
    ))
}

/// Build a `NonSeparableMixedChernoff` with an anisotropy closure `β(x,y)`.
///
/// # Errors
/// Returns [`SemiflowError::DomainViolation`] if `beta_norm_bound` is not finite
/// or is negative.
pub fn with_closure_beta<X, Y, F>(
    x_inner: X,
    y_inner: Y,
    beta: Arc<dyn Fn(F, F) -> F + Send + Sync + 'static>,
    beta_norm_bound: f64,
    grid: Grid2D<F>,
) -> Result<NonSeparableMixedChernoff<X, Y, F>, SemiflowError>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    F: SemiflowFloat,
{
    validate_norm_bound(beta_norm_bound, "beta_norm_bound")?;
    let coupling: Box<dyn MixedDerivOp<F>> = Box::new(ClosureBetaCoupling {
        beta,
        beta_norm_bound,
        is_zero: beta_norm_bound == 0.0,
    });
    Ok(NonSeparableMixedChernoff::with_coupling(
        x_inner, y_inner, grid, coupling,
    ))
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Validate that a norm bound is finite and non-negative.
fn validate_norm_bound(val: f64, name: &'static str) -> Result<(), SemiflowError> {
    if !val.is_finite() || val < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: name,
            value: val,
        });
    }
    Ok(())
}
