//! `NonSeparableMixedChernoff<X, Y, F, S>` — generic non-separable 2D Chernoff operator.
//!
//! Unifies `NonSeparable2DChernoff` (v0.7.0, scalar coupling `c`) and
//! `NonSeparable2DAnisotropicChernoff` (v0.9.0, anisotropic `β(x,y)`) under a
//! single generic over `S: Discrete<F>`.
//!
//! ## Composition (math.md §10.7-ter + §18)
//!
//! ```text
//!   S_5(τ) = AxisLift::X(L_x)(τ/2) ∘ AxisLift::Y(L_y)(τ/2)
//!          ∘ Φ_M(τ)
//!          ∘ AxisLift::Y(L_y)(τ/2) ∘ AxisLift::X(L_x)(τ/2)
//! ```
//!
//! Palindromic 5-leg with K=2 truncated Taylor mixed leg:
//! `Φ_M(τ) = I + τ·M + (τ²/2)·M²`, `M = coupling·∂_x∂_y`.
//!
//! Order: τ-axis 2 (math.md §11.1.bis); spatial O(dx² + dy²)
//! (4-point centred cross-stencil).
//!
//! CFL gate: `4·τ·‖coupling‖_∞ < dx·dy`. Violation returns
//! `SemiflowError::CflViolated`.
//!
//! ## Backwards compatibility
//!
//! Type aliases preserve v0.7.0 and v0.9.0 call sites unchanged:
//! - `NonSeparable2DChernoff<X, Y, F>` (scalar coupling `c`)
//! - `NonSeparable2DAnisotropicChernoff<X, Y, F>` (position-dep coupling `β(x,y)`)
//!
//! ## ADR-0058 (v2.2 Wave C — SUPERSEDES ADR-0033 keep-both)
//!
//! v2.2 collapses to `S = GridFn2D<F>` only. The `S` type parameter is
//! reserved for `v2.3+` graph-mixed extensions (`NonSeparableMixedGraphChernoff`).
//!
//! See math.md §10.7-ter (Theorem 7-bis), §18 (refactor pointer), ADR-0058.

extern crate alloc;

use alloc::boxed::Box;

use crate::{
    axis::{Axis, AxisLift},
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, half, SemiflowFloat},
    grid2d::Grid2D,
    grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
    scratch::ScratchPool,
    strang2d::apply_strang2d_into,
};

mod coupling;
mod stencil;

pub(crate) use coupling::MixedDerivOp;
use coupling::{BetaCoupling, ScalarCoupling};
pub(crate) use stencil::cross_stencil_scalar;

const CFL_NUMER: f64 = 4.0;

// ---------------------------------------------------------------------------
// NonSeparableMixedChernoff
// ---------------------------------------------------------------------------

/// Unified non-separable mixed-derivative 2D Chernoff operator.
///
/// Type aliases preserve v0.7.0 / v0.9.0 call sites unchanged:
/// - [`NonSeparable2DChernoff`]: `with_scalar_c` (constant `c`).
/// - [`NonSeparable2DAnisotropicChernoff`]: `with_beta` (anisotropic `β(x,y)`).
///
/// See math.md §10.7-ter (Theorem 7-bis) and ADR-0058.
///
/// ## Type Parameters
/// - `X`: Inner Chernoff function for `L_x`, where `S = GridFn1D<F>`.
/// - `Y`: Inner Chernoff function for `L_y`, where `S = GridFn1D<F>`.
/// - `F`: Float type (default `f64`; `SemiflowFloat` bound).
/// - `S`: State type (default `GridFn2D<F>`; reserved for v2.3+ graph extension).
///
/// ## CFL constraint
/// Caller must ensure `4·τ·coupling_norm_bound < dx·dy`; violated →
/// [`SemiflowError::CflViolated`] (with `dx_squared = dx*dy`, `a_norm_bound = coupling_norm_bound`).
///
/// ## Generic-over-Float (ADR-0025 / ADR-0026)
/// The `= f64` default on `F` keeps all existing call sites compiling unchanged.
pub struct NonSeparableMixedChernoff<X, Y, F: SemiflowFloat = f64, S = GridFn2D<F>>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    /// Lifted x-axis operator.
    pub x: AxisLift<X, F>,
    /// Lifted y-axis operator.
    pub y: AxisLift<Y, F>,
    /// 2D grid for spatial indexing.
    pub grid: Grid2D<F>,
    /// Mixed-derivative coupling (private — use constructors).
    pub(super) coupling: Box<dyn MixedDerivOp<F>>,
    /// Phantom for `S` — reserved for v2.3+ graph extensions.
    _state: core::marker::PhantomData<S>,
}

impl<X, Y, F: SemiflowFloat, S> Clone for NonSeparableMixedChernoff<X, Y, F, S>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.clone(),
            grid: self.grid,
            coupling: self.coupling.clone_box(),
            _state: core::marker::PhantomData,
        }
    }
}

impl<X, Y, F: SemiflowFloat, S> core::fmt::Debug for NonSeparableMixedChernoff<X, Y, F, S>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + core::fmt::Debug,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone + core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NonSeparableMixedChernoff")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("coupling", &self.coupling.debug_label())
            .field("grid", &format_args!("Grid2D<{}>", self.grid.nx()))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Constructors (GridFn2D<F> specialisation)
// ---------------------------------------------------------------------------

impl<X, Y, F: SemiflowFloat> NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    /// Backwards-compatible `new` constructor.
    ///
    /// Accepted by both `NonSeparable2DChernoff::new` (v0.7.0) and
    /// `NonSeparable2DAnisotropicChernoff::new` (v0.9.0) call sites.
    /// Internally routes to [`with_scalar_c`][Self::with_scalar_c]; both
    /// aliases produce byte-identical results since the coupling function
    /// pointer is stored and invoked identically regardless of which alias
    /// was used at the call site.
    ///
    /// For new code, prefer [`with_scalar_c`][Self::with_scalar_c] or
    /// [`with_beta`][Self::with_beta] which are more explicit.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] if `norm_bound` is
    /// non-finite or negative.
    pub fn new(
        x_inner: X,
        y_inner: Y,
        coupling_fn: fn(F, F) -> F,
        norm_bound: f64,
        grid: Grid2D<F>,
    ) -> Result<Self, SemiflowError> {
        Self::with_scalar_c(x_inner, y_inner, coupling_fn, norm_bound, grid)
    }

    /// Constant scalar coupling constructor.
    ///
    /// Mirrors v0.7.0 `NonSeparable2DChernoff::new`. Use the
    /// [`NonSeparable2DChernoff`] type alias for the v0.7.0 call-site syntax.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] if `c_norm_bound` is
    /// non-finite or negative.
    pub fn with_scalar_c(
        x_inner: X,
        y_inner: Y,
        c: fn(F, F) -> F,
        c_norm_bound: f64,
        grid: Grid2D<F>,
    ) -> Result<Self, SemiflowError> {
        if !c_norm_bound.is_finite() || c_norm_bound < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "c_norm_bound must be finite and >= 0",
                value: c_norm_bound,
            });
        }
        let coupling = Box::new(ScalarCoupling {
            c,
            c_norm_bound,
            is_zero: c_norm_bound == 0.0,
        });
        Ok(Self {
            x: AxisLift::new(x_inner, Axis::X),
            y: AxisLift::new(y_inner, Axis::Y),
            grid,
            coupling,
            _state: core::marker::PhantomData,
        })
    }

    /// Position-dependent coupling constructor.
    ///
    /// Mirrors v0.9.0 `NonSeparable2DAnisotropicChernoff::new`. Use the
    /// [`NonSeparable2DAnisotropicChernoff`] type alias for the v0.9.0 call-site syntax.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] if `beta_norm_bound` is
    /// non-finite or negative.
    pub fn with_beta(
        x_inner: X,
        y_inner: Y,
        beta: fn(F, F) -> F,
        beta_norm_bound: f64,
        grid: Grid2D<F>,
    ) -> Result<Self, SemiflowError> {
        if !beta_norm_bound.is_finite() || beta_norm_bound < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "beta_norm_bound must be finite and >= 0",
                value: beta_norm_bound,
            });
        }
        let coupling = Box::new(BetaCoupling {
            beta,
            beta_norm_bound,
            is_zero: beta_norm_bound == 0.0,
        });
        Ok(Self {
            x: AxisLift::new(x_inner, Axis::X),
            y: AxisLift::new(y_inner, Axis::Y),
            grid,
            coupling,
            _state: core::marker::PhantomData,
        })
    }

    /// Construct from a `Box<dyn MixedDerivOp<F>>` (v2.3, ADR-0058 §"v2.3 promotions").
    pub(crate) fn with_coupling(
        x_inner: X,
        y_inner: Y,
        grid: Grid2D<F>,
        coupling: Box<dyn MixedDerivOp<F>>,
    ) -> Self {
        Self {
            x: AxisLift::new(x_inner, Axis::X),
            y: AxisLift::new(y_inner, Axis::Y),
            grid,
            coupling,
            _state: core::marker::PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl (sequential, non-parallel)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "parallel"))]
impl<X, Y, F: SemiflowFloat> ChernoffFunction<F> for NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    type S = GridFn2D<F>;

    /// Temporal order: 2 (τ-axis; math.md §11.1.bis).
    fn order(&self) -> u32 {
        2
    }

    /// Semigroup growth bound `(M, ω)`.
    fn growth(&self) -> Growth<F> {
        compute_growth(&self.x, &self.y)
    }

    /// Allocation-free apply (writes into `dst`).
    ///
    /// Zero-coupling fast path: calls `apply_strang2d_into` (the `ScratchPool`-backed
    /// serial ping-pong kernel shared with `Strang2D::apply_into`) — achieves **0**
    /// heap allocations per step in steady state after the first warm-up call.
    ///
    /// Non-zero coupling path: `apply_five_leg` allocates `GridFn2D`
    /// intermediates per step (f1, f2, `Φ_M` temporaries). 2D-state sized buffers
    /// cannot be stored in `ScratchPool<F>` (pool holds 1D row/col scratch only).
    fn apply_into(
        &self,
        tau: F,
        src: &GridFn2D<F>,
        dst: &mut GridFn2D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if self.coupling.is_zero() {
            return apply_strang2d_into(tau, src, dst, &self.x, &self.y, scratch);
        }
        validate_cfl(tau, self.coupling.norm_bound(), &self.grid)?;
        *dst = apply_five_leg(&self.x, &self.y, &*self.coupling, tau, src, &self.grid)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl (parallel feature — f64 only, extra Send+Sync bounds)
// ---------------------------------------------------------------------------

/// `NonSeparableMixedChernoff` [`ChernoffFunction`] impl
/// (`parallel` feature: extra `Send + Sync` bounds, f64-only).
#[cfg(feature = "parallel")]
impl<X, Y> ChernoffFunction<f64> for NonSeparableMixedChernoff<X, Y, f64, GridFn2D<f64>>
where
    X: ChernoffFunction<f64, S = GridFn1D<f64>> + Clone + Send + Sync,
    Y: ChernoffFunction<f64, S = GridFn1D<f64>> + Clone + Send + Sync,
{
    type S = GridFn2D<f64>;

    /// Temporal order: 2 (τ-axis; math.md §11.1.bis).
    fn order(&self) -> u32 {
        2
    }

    /// Semigroup growth bound `(M, ω)`.
    fn growth(&self) -> Growth<f64> {
        compute_growth(&self.x, &self.y)
    }

    /// Allocation-free apply (writes into `dst`).
    ///
    /// Zero-coupling fast path: calls `apply_strang2d_into` (the `ScratchPool`-backed
    /// serial ping-pong kernel) — achieves **0** heap allocations per step in steady
    /// state after the first warm-up call.
    ///
    /// Non-zero coupling path: allocates `GridFn2D` intermediates per step
    /// (2D-state sized; not `ScratchPool`-able).
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn2D<f64>,
        dst: &mut GridFn2D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        if self.coupling.is_zero() {
            return apply_strang2d_into(tau, src, dst, &self.x, &self.y, scratch);
        }
        validate_cfl(tau, self.coupling.norm_bound(), &self.grid)?;
        *dst = apply_five_leg(&self.x, &self.y, &*self.coupling, tau, src, &self.grid)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Backwards-compatibility type aliases (ADR-0058 §"Migration")
// ---------------------------------------------------------------------------

/// Palindromic 5-leg Chernoff product for non-separable 2D diffusion with
/// scalar coupling coefficient `c(x, y)` (v0.7.0 surface).
///
/// Type alias for `NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>`.
/// Use [`NonSeparableMixedChernoff::with_scalar_c`] or
/// the shim `NonSeparable2DChernoff::new` constructor (unchanged call site).
///
/// See math.md §10.7-bis, ADR-0016, ADR-0058.
pub type NonSeparable2DChernoff<X, Y, F = f64> = NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>;

/// Palindromic 5-leg Chernoff product for anisotropic non-separable 2D diffusion
/// with position-dependent coupling coefficient `β(x, y)` (v0.9.0 surface).
///
/// Type alias for `NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>`.
/// Use [`NonSeparableMixedChernoff::with_beta`] or
/// the shim `NonSeparable2DAnisotropicChernoff::new` constructor (unchanged call site).
///
/// See math.md §10.7-ter, ADR-0023, ADR-0058.
pub type NonSeparable2DAnisotropicChernoff<X, Y, F = f64> =
    NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// CFL gate: `4·τ·‖coupling‖_∞ < dx·dy`. Returns error on violation.
fn validate_cfl<F: SemiflowFloat>(
    tau: F,
    norm_bound: f64,
    grid: &Grid2D<F>,
) -> Result<(), SemiflowError> {
    let tau_f64 = tau.to_f64().unwrap_or(f64::NAN);
    if !tau_f64.is_finite() || tau_f64 < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau_f64,
        });
    }
    let dx_dy = grid.x.dx().to_f64().unwrap_or(f64::NAN) * grid.y.dx().to_f64().unwrap_or(f64::NAN);
    if CFL_NUMER * tau_f64 * norm_bound >= dx_dy {
        return Err(SemiflowError::CflViolated {
            tau: tau_f64,
            dx_squared: dx_dy,
            a_norm_bound: norm_bound,
        });
    }
    Ok(())
}

/// Growth bound shared between sequential and parallel impls.
fn compute_growth<X, Y, F: SemiflowFloat>(x: &AxisLift<X, F>, y: &AxisLift<Y, F>) -> Growth<F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    let gx = x.growth();
    let gy = y.growth();
    let m_mixed = from_f64::<F>(1.0 + 0.25 + 0.5 * 0.25 * 0.25);
    Growth {
        multiplier: gx.multiplier * gy.multiplier * m_mixed * gy.multiplier * gx.multiplier,
        omega: gx.omega + gy.omega,
    }
}

/// Apply the palindromic 5-leg composition `S_5(τ)`.
///
/// `S_5(τ) = X(τ/2) ∘ Y(τ/2) ∘ Φ_M(τ) ∘ Y(τ/2) ∘ X(τ/2)`
///
/// where `Φ_M(τ) = I + τ·M + (τ²/2)·M²` is the K=2 truncated Taylor
/// mixed leg and `M = coupling·∂_x∂_y`.
#[allow(clippy::too_many_lines)]
fn apply_five_leg<X, Y, F: SemiflowFloat>(
    x: &AxisLift<X, F>,
    y: &AxisLift<Y, F>,
    coupling: &dyn MixedDerivOp<F>,
    tau: F,
    f: &GridFn2D<F>,
    grid: &Grid2D<F>,
) -> Result<GridFn2D<F>, SemiflowError>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    use crate::chernoff::ApplyChernoffExt;
    let half_tau = half::<F>() * tau;
    let f1 = x.apply_chernoff(half_tau, f)?;
    let f2 = y.apply_chernoff(half_tau, &f1)?;
    let f3 = phi_mixed_step(coupling, tau, &f2, grid);
    let f4 = y.apply_chernoff(half_tau, &f3)?;
    x.apply_chernoff(half_tau, &f4)
}

/// K=2 truncated Taylor mixed leg: `Φ_M(τ) = I + τ·M + (τ²/2)·M²`.
///
/// Uses `coupling.apply_mixed_into` for `M·f` and `M²·f` steps.
/// No heap allocation beyond the two `GridFn2D` clones for the M-step
/// intermediates (consistent with v0.7.0 / v0.9.0 behaviour).
fn phi_mixed_step<F: SemiflowFloat>(
    coupling: &dyn MixedDerivOp<F>,
    tau: F,
    f: &GridFn2D<F>,
    grid: &Grid2D<F>,
) -> GridFn2D<F> {
    let mut mf = f.clone();
    coupling.apply_mixed_into(f, &mut mf, grid);
    let mut m2f = f.clone();
    coupling.apply_mixed_into(&mf, &mut m2f, grid);
    let nx = grid.nx();
    let ny = grid.ny();
    let half_v = half::<F>();
    let mut out = f.clone();
    for j in 0..ny {
        for i in 0..nx {
            let k = grid.idx(i, j);
            out.values[k] = out.values[k] + tau * mf.values[k] + half_v * tau * tau * m2f.values[k];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
