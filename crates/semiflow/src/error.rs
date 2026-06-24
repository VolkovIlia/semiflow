//! Error taxonomy for `semiflow-core`.
//!
//! See `contracts/semiflow-core.errors.yaml` for the authoritative listing.
//! Every fallible operation returns `Result<T, SemiflowError>`; the library
//! never panics in release mode (errors-as-values, suckless principle).
//!
//! ## Recovery patterns
//!
//! | Variant | Recovery |
//! |---------|----------|
//! | [`SemiflowError::DomainViolation`] | fix the offending input value |
//! | [`SemiflowError::GridUnderresolved`] | use `suggested_n` nodes |
//! | [`SemiflowError::CflViolated`] | use `tau_safe = 0.45 * dx² / a_norm_bound` |
//! | [`SemiflowError::AdaptiveStepRejected`] | loosen `tol_rel`/`tol_abs`, coarsen grid |
//! | [`SemiflowError::ConvergenceFailed`] | reserved; not returned in v0.11.0 |
//! | [`SemiflowError::Unsupported`] | enable the named feature flag |

use core::fmt;

/// Errors returned by `semiflow-core` operations.
///
/// `#[non_exhaustive]` — new variants may be added in minor versions without
/// a semver break. Match with a catch-all `_ => {}` arm when forward-compat
/// is required.
///
/// All `f64` payload fields (`value`, `tau`, etc.) may be `NaN` / `Inf` when
/// the offending input was itself non-finite.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[allow(clippy::module_name_repetitions)]
pub enum SemiflowError {
    /// Caller passed input that violates a documented precondition.
    ///
    /// Examples: `n < 4` in `Grid1D::new`, `tau < 0` in `apply`,
    /// `xmin >= xmax`.
    DomainViolation {
        /// Static label for the violated invariant (stable, grep-friendly).
        what: &'static str,
        /// Offending numeric value. `NaN`/`Inf` permitted.
        value: f64,
    },

    /// The Chernoff shift exceeds the grid spacing — accuracy may degrade.
    ///
    /// v0.1.0 policy: soft signal only; `apply` still returns `Ok`.
    GridUnderresolved {
        /// Recommended grid size to resolve the shift.
        suggested_n: usize,
        /// `(max shift) / dx` ratio that triggered the warning (> 1.0).
        shift_dx_ratio: f64,
    },

    /// Iterative solver did not converge within the iteration cap.
    ///
    /// Reserved for v0.3+ resolvent; **never returned** in v0.1.0.
    ConvergenceFailed {
        /// Final residual norm when the cap was hit.
        last_residual: f64,
        /// Iteration cap that was reached.
        max_iter: usize,
    },

    /// A feature is declared in the public API but not yet implemented.
    ///
    /// v0.1.0: returned by `BoundaryPolicy::{ZeroExtend, Periodic,
    /// LinearExtrapolate}` and `InterpKind::Linear` (without the
    /// `linear-interp` feature).
    Unsupported {
        /// Which optional feature is unsupported. Stable label.
        feature: &'static str,
    },

    /// CFL bound violated for the truncated-exp K=4 power series.
    ///
    /// Returned by `TruncatedExpDiffusionChernoff::apply` when
    /// `2·tau·a_norm_bound ≥ dx²`. The K=4 partial sum diverges in
    /// operator norm when this bound is violated (BCOR 2009 §3.1).
    ///
    /// Caller pattern: retry with `tau_safe = 0.45 * dx_squared / a_norm_bound`.
    ///
    /// **v0.7.0 Block C note (ADR-0016, math.md §10.7-bis.4)**: for
    /// `NonSeparable2DChernoff`, `dx_squared` semantically holds `dx · dy`
    /// (cell area) and `a_norm_bound` holds `‖c‖_∞`. The variant is
    /// REUSED — no new variant was added.
    CflViolated {
        /// The Chernoff step that violated the CFL bound.
        tau: f64,
        /// `dx² = grid.dx() * grid.dx()` at the point of failure.
        /// For `NonSeparable2DChernoff`: `dx · dy` (cell area).
        dx_squared: f64,
        /// The `a_norm_bound` of the `TruncatedExpDiffusionChernoff` instance.
        /// For `NonSeparable2DChernoff`: `‖c‖_∞`.
        a_norm_bound: f64,
    },

    /// Adaptive PI step controller failed to converge within `max_substeps`.
    ///
    /// Returned exclusively by [`crate::AdaptivePI::evolve_adaptive`] when
    /// `steps_attempted >= max_substeps` (runaway protection). This indicates
    /// a pathological tolerance mismatch — the requested `tol_rel` / `tol_abs`
    /// is tighter than what the underlying `ChernoffFunction` can achieve for
    /// the given grid and time step configuration.
    ///
    /// Caller pattern: increase `tol_rel`, coarsen the grid, or reduce `t`.
    ///
    /// See: ADR-0014, `contracts/semiflow-core.errors.yaml`.
    AdaptiveStepRejected {
        /// Last substep size attempted before the cap was hit.
        last_tau: f64,
        /// Richardson error estimate on the last attempted substep.
        last_err: f64,
        /// Total substeps attempted (accepted + rejected); equals `max_substeps`.
        steps_attempted: usize,
    },

    /// Operation requires a primitive not yet available for this inner type.
    ///
    /// Returned by [`crate::adjoint::AdjointChernoff`] `apply_into` when called via
    /// `new_general` on an inner that does not implement
    /// [`crate::adjoint::AdjointApply`] (no transpose-apply primitive).
    ///
    /// Caller options: (a) use `new_self_adjoint` for symmetric inners;
    /// (b) use an inner that implements `AdjointApply` (e.g.,
    /// `DriftReactionChernoff`). See ADR-0114.
    UnsupportedOperation {
        /// What operation was requested and why it cannot be fulfilled.
        what: &'static str,
    },

    /// Magnus convergence radius violated: `ρ̄_max · τ ≥ π/2`.
    ///
    /// Returned by [`crate::magnus_graph::MagnusGraphHeatChernoff`] `apply_into` when
    /// `rho_bar_max * tau >= π/2` (50% safety margin vs. theoretical `< π`).
    ///
    /// Caller MUST reduce `τ` or supply a tighter `rho_bar_max` bound.
    ///
    /// See: ADR-0051, math.md §12.9 (NORMATIVE library policy).
    OutOfMagnusRadius {
        /// The Chernoff step `τ` that violated the convergence radius.
        tau: f64,
        /// The caller-supplied Gershgorin spectral-radius estimate `ρ̄_max`.
        rho_estimate: f64,
    },

    /// Input lies outside the proven operator class for an S³ POC evolver.
    ///
    /// Returned by the `S3*` constructor family (ADR-0169, v9.2.0) when the
    /// supplied parameters cannot represent an in-class operator — for example,
    /// a diffusion coefficient `a[j] ≤ 0` (non-parabolic), a CP-rank coefficient
    /// that violates parabolicity on the grid, or `u0` outside `(0,1)` for
    /// [`crate::S3BurgersColeHopf`] logistic reaction.
    ///
    /// `detail` is a stable, grep-friendly static label naming the violated class
    /// boundary.  See the `## Proven boundary` stanza on each S³ type for the
    /// full class description.
    ///
    /// **Recovery:** adjust the input to lie within the proven class, or choose
    /// a different evolver that covers the desired operator.
    #[cfg(feature = "s3-poc")]
    S3OutOfClass {
        /// Stable label for the violated class boundary.
        detail: &'static str,
    },

    /// Input lies outside the proven operator class for [`crate::VarCoefTt`].
    ///
    /// Returned by [`crate::VarCoefTt::new`] when the supplied parameters
    /// cannot represent an in-class additive-separable variable-coefficient
    /// operator — for example, `a_axis[j][i] ≤ 0` (non-parabolic), shape
    /// mismatch between axes, or `d == 0`.
    ///
    /// `detail` is a stable, grep-friendly static label naming the violated
    /// class boundary.
    ///
    /// **Recovery:** adjust the input to lie within the additive-separable
    /// parabolic class described in [`crate::VarCoefTt`].
    VarCoefOutOfClass {
        /// Stable label for the violated class boundary.
        detail: &'static str,
    },
}

/// Format helpers — each formats one variant, keeping `fmt` under 50 lines.
fn fmt_grid_underresolved(
    f: &mut fmt::Formatter<'_>,
    suggested_n: usize,
    shift_dx_ratio: f64,
) -> fmt::Result {
    write!(
        f,
        "grid under-resolved: shift/dx = {shift_dx_ratio:.3}, \
         suggested N >= {suggested_n}"
    )
}

fn fmt_cfl_violated(
    f: &mut fmt::Formatter<'_>,
    tau: f64,
    dx_squared: f64,
    a_norm_bound: f64,
) -> fmt::Result {
    write!(
        f,
        "CFL violated for truncated-exp K=4: tau = {tau:.3e}, \
         dx\u{00B2} = {dx_squared:.3e}, \
         \u{2016}a\u{2016}_\u{221E} = {a_norm_bound:.3e} \
         (need 2\u{00B7}tau\u{00B7}\u{2016}a\u{2016}_\u{221E} < dx\u{00B2})"
    )
}

fn fmt_adaptive_rejected(
    f: &mut fmt::Formatter<'_>,
    last_tau: f64,
    last_err: f64,
    steps_attempted: usize,
) -> fmt::Result {
    write!(
        f,
        "adaptive PI: step rejected after {steps_attempted} substeps; \
         last_tau = {last_tau:.3e}, last_err = {last_err:.3e} \
         (increase tol_rel/tol_abs or reduce t)"
    )
}

fn fmt_out_of_magnus_radius(
    f: &mut fmt::Formatter<'_>,
    tau: f64,
    rho_estimate: f64,
) -> fmt::Result {
    write!(
        f,
        "Magnus convergence radius violated: \
         rho_bar_max = {rho_estimate:.3e}, tau = {tau:.3e}, \
         product = {:.3e} >= pi/2 \
         (reduce tau or supply tighter rho_bar_max)",
        rho_estimate * tau
    )
}

impl fmt::Display for SemiflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DomainViolation { what, value } => {
                write!(f, "domain violation: {what} (value = {value})")
            }
            Self::GridUnderresolved {
                suggested_n,
                shift_dx_ratio,
            } => fmt_grid_underresolved(f, *suggested_n, *shift_dx_ratio),
            Self::ConvergenceFailed {
                last_residual,
                max_iter,
            } => {
                write!(
                    f,
                    "convergence failed: last_residual = {last_residual:.3e}, \
                     max_iter = {max_iter}"
                )
            }
            Self::Unsupported { feature } => {
                write!(f, "feature not supported in this build: {feature}")
            }
            Self::UnsupportedOperation { what } => {
                write!(f, "unsupported operation: {what}")
            }
            Self::CflViolated {
                tau,
                dx_squared,
                a_norm_bound,
            } => fmt_cfl_violated(f, *tau, *dx_squared, *a_norm_bound),
            Self::AdaptiveStepRejected {
                last_tau,
                last_err,
                steps_attempted,
            } => fmt_adaptive_rejected(f, *last_tau, *last_err, *steps_attempted),
            Self::OutOfMagnusRadius { tau, rho_estimate } => {
                fmt_out_of_magnus_radius(f, *tau, *rho_estimate)
            }
            #[cfg(feature = "s3-poc")]
            Self::S3OutOfClass { detail } => {
                write!(f, "S³ out-of-class: {detail}")
            }
            Self::VarCoefOutOfClass { detail } => {
                write!(f, "VarCoefTt out-of-class: {detail}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SemiflowError {}
