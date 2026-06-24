//! Shared validation helpers for `Diffusion4thChernoff`, `Diffusion6thChernoff`,
//! and `Diffusion8thZeta8Chernoff`.
//!
//! Only genuinely byte-identical logic lives here: the two `validate_tau` and two
//! `validate_a_x` functions.  All order-specific code (stencil arrays, K-kernel
//! weights, FD-apply routines, baseline fns, delta formulas) is intentionally kept
//! in the per-order modules — they differ in width (7-pt vs 9-pt), SIMD layout,
//! and stencil-step formula.

use crate::{error::SemiflowError, float::SemiflowFloat};

/// Validate `tau`: must be finite and non-negative (f64).
#[inline]
pub(crate) fn validate_tau_f64(tau: f64) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau,
        });
    }
    Ok(())
}

/// Validate `a(x_pre) ≥ 0` and finite (f64, strict ellipticity for `sqrt`).
#[inline]
pub(crate) fn validate_a_x_f64(a_x: f64, x: f64) -> Result<(), SemiflowError> {
    if !a_x.is_finite() || a_x < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "a(x) must be finite and >= 0 (strict ellipticity required for sqrt)",
            value: x,
        });
    }
    Ok(())
}

/// Validate `tau`: must be finite and non-negative (generic).
#[inline]
pub(crate) fn validate_tau_generic<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

/// Validate `a(x_pre) ≥ 0` and finite (generic, strict ellipticity for `sqrt`).
#[inline]
pub(crate) fn validate_a_x_generic<F: SemiflowFloat>(a_x: F, x: F) -> Result<(), SemiflowError> {
    if !a_x.is_finite() || a_x < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "a(x) must be finite and >= 0 (strict ellipticity required for sqrt)",
            value: x.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}
