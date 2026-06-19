//! Private generic helpers for `Diffusion6thChernoff`.
//!
//! Declared as `#[path = "diffusion6_generic.rs"] mod helpers_generic;` inside
//! `diffusion6.rs` — this file is a child of that module, so `super::` works.

use num_traits::Float;

use crate::{
    error::SemiflowError,
    float::{from_f64, half, SemiflowFloat},
    grid_fn::GridFn1D,
};

use super::{Diffusion6thChernoff, C1_9, C2_9, C3_9, K7_P, K7_W0, K7_W1, K7_W2, K7_W3};

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

/// γ⁶-A baseline (generic, scalar path).
#[inline]
pub(super) fn gamma6_a_baseline_generic<F: SemiflowFloat>(
    dc: &Diffusion6thChernoff<F>,
    tau: F,
    f: &GridFn1D<F>,
    x: F,
) -> Result<F, SemiflowError> {
    let s_half = half::<F>() * tau;
    let two = from_f64::<F>(2.0);
    let three = from_f64::<F>(3.0);
    let five = from_f64::<F>(K7_P);
    let w0 = from_f64::<F>(K7_W0);
    let w1 = from_f64::<F>(K7_W1);
    let w2 = from_f64::<F>(K7_W2);
    let w3 = from_f64::<F>(K7_W3);

    let x_pre = x + s_half * (dc.a_prime)(x);
    let a_pre = (dc.a)(x_pre);
    validate_a_x_generic(a_pre, x_pre)?;

    let h = two * Float::sqrt(a_pre * tau);
    let h_3 = two * Float::sqrt(three * a_pre * tau);
    let j_5 = two * Float::sqrt(five * a_pre * tau);

    let post = |x_raw: F| x_raw + s_half * (dc.a_prime)(x_raw);

    let v_center = f.sample_generic(post(x_pre))?;
    let v_near_p = f.sample_generic(post(x_pre + h))?;
    let v_near_n = f.sample_generic(post(x_pre - h))?;
    let v_far_p = f.sample_generic(post(x_pre + h_3))?;
    let v_far_n = f.sample_generic(post(x_pre - h_3))?;
    let v_ext_p = f.sample_generic(post(x_pre + j_5))?;
    let v_ext_n = f.sample_generic(post(x_pre - j_5))?;

    Ok(w0 * v_center
        + w1 * (v_near_p + v_near_n)
        + w2 * (v_far_p + v_far_n)
        + w3 * (v_ext_p + v_ext_n))
}

/// Apply 9-point Fornberg FD stencil for `f^(deriv)` at `x` (generic).
#[inline]
pub(super) fn fd9_generic<F: SemiflowFloat>(
    f: &GridFn1D<F>,
    x: F,
    delta: F,
    coeffs: &[f64; 9],
    deriv: i32,
) -> Result<F, SemiflowError> {
    let ks: [f64; 9] = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let mut sum = F::zero();
    for j in 0..9 {
        let c = from_f64::<F>(coeffs[j]);
        let k = from_f64::<F>(ks[j]);
        sum += c * f.sample_generic(x + k * delta)?;
    }
    let denom = Float::powi(delta, deriv);
    Ok(sum / denom)
}

/// ζ⁶ τ²-correction (generic scalar path).
#[allow(clippy::similar_names)]
#[inline]
pub(super) fn zeta6_correction_generic<F: SemiflowFloat>(
    dc: &Diffusion6thChernoff<F>,
    tau: F,
    f: &GridFn1D<F>,
    x: F,
) -> Result<F, SemiflowError> {
    let four = from_f64::<F>(4.0);
    let half_v = half::<F>();
    let quarter = half_v * half_v;
    let delta = Float::max(four * dc.grid.dx(), Float::sqrt(tau));

    let a_val = (dc.a)(x);
    let a_prime_val = (dc.a_prime)(x);
    let a_dbl_val = (dc.a_double_prime)(x);

    if a_prime_val == F::zero() && a_dbl_val == F::zero() {
        return Ok(F::zero());
    }

    let fd1 = fd9_generic(f, x, delta, &C1_9, 1)?;
    let fd2 = fd9_generic(f, x, delta, &C2_9, 2)?;
    let fd3 = fd9_generic(f, x, delta, &C3_9, 3)?;

    Ok(tau
        * tau
        * (a_val * a_prime_val * fd3
            + (a_val * a_dbl_val * half_v) * fd2
            + (a_prime_val * a_dbl_val * quarter) * fd1))
}

/// Apply ζ⁶ at a single grid node `i` (generic path).
#[inline]
pub(super) fn apply_at_node_generic<F: SemiflowFloat>(
    dc: &Diffusion6thChernoff<F>,
    tau: F,
    f: &GridFn1D<F>,
    i: usize,
) -> Result<F, SemiflowError> {
    let x = dc.grid.x_at(i);
    Ok(gamma6_a_baseline_generic(dc, tau, f, x)? + zeta6_correction_generic(dc, tau, f, x)?)
}
