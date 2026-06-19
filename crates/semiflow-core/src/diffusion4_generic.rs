//! Private generic helpers for `Diffusion4thChernoff`.
//!
//! Declared as `#[path = "diffusion4_generic.rs"] mod helpers_generic;` inside
//! `diffusion4.rs` — this file is a child of that module, so `super::` works.

use num_traits::Float;

use crate::{
    error::SemiflowError,
    float::{from_f64, half, SemiflowFloat},
    grid_fn::GridFn1D,
};

use super::{Diffusion4thChernoff, C1, C2, C3, W0, W1, W2};

/// Validate `tau`: must be finite and non-negative (generic).
#[inline]
pub(super) fn validate_tau_generic<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

/// Validate `a(x_pre) ≥ 0` and finite (generic).
#[inline]
pub(super) fn validate_a_x_generic<F: SemiflowFloat>(a_x: F, x: F) -> Result<(), SemiflowError> {
    if !a_x.is_finite() || a_x < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "a(x) must be finite and >= 0 (strict ellipticity required for sqrt)",
            value: x.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

/// γ-A inner-Strang baseline (generic, scalar path).
#[inline]
pub(super) fn gamma_a_baseline_generic<F: SemiflowFloat>(
    dc: &Diffusion4thChernoff<F>,
    tau: F,
    f: &GridFn1D<F>,
    x: F,
) -> Result<F, SemiflowError> {
    let s_half = half::<F>() * tau;
    let two = from_f64::<F>(2.0);
    let three = from_f64::<F>(3.0);
    let w0 = from_f64::<F>(W0);
    let w1 = from_f64::<F>(W1);
    let w2 = from_f64::<F>(W2);

    let x_pre = x + s_half * (dc.a_prime)(x);
    let a_at_pre = (dc.a)(x_pre);
    validate_a_x_generic(a_at_pre, x_pre)?;

    let h0 = two * Float::sqrt(a_at_pre * tau);
    let h0_3 = two * Float::sqrt(three * a_at_pre * tau);

    let center_pos = x_pre + s_half * (dc.a_prime)(x_pre);

    let near_p_raw = x_pre + h0;
    let near_p_pos = near_p_raw + s_half * (dc.a_prime)(near_p_raw);

    let near_neg_raw = x_pre - h0;
    let near_neg_pos = near_neg_raw + s_half * (dc.a_prime)(near_neg_raw);

    let far_p_raw = x_pre + h0_3;
    let far_p_pos = far_p_raw + s_half * (dc.a_prime)(far_p_raw);

    let far_neg_raw = x_pre - h0_3;
    let far_neg_pos = far_neg_raw + s_half * (dc.a_prime)(far_neg_raw);

    let center = w0 * f.sample_generic(center_pos)?;
    let near = w1 * (f.sample_generic(near_p_pos)? + f.sample_generic(near_neg_pos)?);
    let far = w2 * (f.sample_generic(far_p_pos)? + f.sample_generic(far_neg_pos)?);

    Ok(center + near + far)
}

/// Apply 7-point Fornberg FD stencil for `f^(deriv)` at `x` (generic).
#[inline]
pub(super) fn fd7_generic<F: SemiflowFloat>(
    f: &GridFn1D<F>,
    x: F,
    delta: F,
    coeffs: &[f64; 7],
    deriv: i32,
) -> Result<F, SemiflowError> {
    let ks: [f64; 7] = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let mut sum = F::zero();
    for j in 0..7 {
        let c = from_f64::<F>(coeffs[j]);
        let k = from_f64::<F>(ks[j]);
        sum += c * f.sample_generic(x + k * delta)?;
    }
    let denom = Float::powi(delta, deriv);
    Ok(sum / denom)
}

/// ζ⁴ τ²-correction (generic scalar path).
#[inline]
pub(super) fn zeta4_correction_generic<F: SemiflowFloat>(
    dc: &Diffusion4thChernoff<F>,
    tau: F,
    f: &GridFn1D<F>,
    x: F,
) -> Result<F, SemiflowError> {
    let three = from_f64::<F>(3.0);
    let half_v = half::<F>();
    let quarter = half_v * half_v;
    let delta = Float::max(three * dc.grid.dx(), Float::powf(tau, from_f64::<F>(0.75)));

    let a_x = (dc.a)(x);
    let a_prime_x = (dc.a_prime)(x);
    let app_x = (dc.a_double_prime)(x);

    // Short-circuit: constant-a path.
    if a_prime_x == F::zero() && app_x == F::zero() {
        return Ok(F::zero());
    }

    let f1 = fd7_generic(f, x, delta, &C1, 1)?;
    let f2 = fd7_generic(f, x, delta, &C2, 2)?;
    let f3 = fd7_generic(f, x, delta, &C3, 3)?;

    Ok(tau
        * tau
        * (a_x * a_prime_x * f3 + (a_x * app_x * half_v) * f2 + (a_prime_x * app_x * quarter) * f1))
}

/// Apply ζ⁴ at a single grid node `i` (generic path).
#[inline]
pub(super) fn apply_at_node_generic<F: SemiflowFloat>(
    dc: &Diffusion4thChernoff<F>,
    tau: F,
    f: &GridFn1D<F>,
    i: usize,
) -> Result<F, SemiflowError> {
    let x = dc.grid.x_at(i);
    Ok(gamma_a_baseline_generic(dc, tau, f, x)? + zeta4_correction_generic(dc, tau, f, x)?)
}
