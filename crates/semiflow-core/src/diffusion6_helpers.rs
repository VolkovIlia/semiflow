//! Private f64-specific helpers for `Diffusion6thChernoff`.
//!
//! Declared as `#[path = "diffusion6_helpers.rs"] mod helpers_f64;` inside
//! `diffusion6.rs` — this file is a child of that module, so `super::` works.

use crate::{error::SemiflowError, grid_fn::GridFn1D};

#[cfg(feature = "simd")]
use crate::simd::{F64x4, SimdF64x4};

use super::{Diffusion6thChernoff, C1_9, C2_9, C3_9, K7_P, K7_W0, K7_W1, K7_W2, K7_W3};

/// Validate `tau`: must be finite and non-negative (f64).
#[inline]
pub(super) fn validate_tau_f64(tau: f64) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau,
        });
    }
    Ok(())
}

/// Validate `a(x_pre) ≥ 0` and finite (strict ellipticity for `sqrt`).
#[inline]
pub(super) fn validate_a_x_f64(a_x: f64, x: f64) -> Result<(), SemiflowError> {
    if !a_x.is_finite() || a_x < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "a(x) must be finite and >= 0 (strict ellipticity required for sqrt)",
            value: x,
        });
    }
    Ok(())
}

/// γ⁶-A baseline: `D_γ⁶(τ) = S(τ/2) ∘ K7(τ;a) ∘ S(τ/2)` (f64, SIMD path).
#[inline]
pub(super) fn gamma6_a_baseline_f64(
    dc: &Diffusion6thChernoff<f64>,
    tau: f64,
    f: &GridFn1D<f64>,
    x: f64,
) -> Result<f64, SemiflowError> {
    let s_half = 0.5 * tau;

    let x_pre = x + s_half * dc.eval_ap(x);
    let a_pre = dc.eval_a(x_pre);
    validate_a_x_f64(a_pre, x_pre)?;

    let h = 2.0 * libm::sqrt(a_pre * tau);
    let h_3 = 2.0 * libm::sqrt(3.0 * a_pre * tau);
    let j_5 = 2.0 * libm::sqrt(K7_P * a_pre * tau);

    let dc_ref = dc;
    let post = |x_raw: f64| x_raw + s_half * dc_ref.eval_ap(x_raw);

    let v_center = f.sample(post(x_pre))?;
    let v_near_p = f.sample(post(x_pre + h))?;
    let v_near_n = f.sample(post(x_pre - h))?;
    let v_far_p = f.sample(post(x_pre + h_3))?;
    let v_far_n = f.sample(post(x_pre - h_3))?;
    let v_ext_p = f.sample(post(x_pre + j_5))?;
    let v_ext_n = f.sample(post(x_pre - j_5))?;

    Ok(K7_W0 * v_center
        + K7_W1 * (v_near_p + v_near_n)
        + K7_W2 * (v_far_p + v_far_n)
        + K7_W3 * (v_ext_p + v_ext_n))
}

/// Apply 9-point Fornberg FD stencil for `f^(deriv)` at `x` (scalar path, f64).
// used under #[cfg(not(feature = "simd"))] fallback path in fd9_f64
#[allow(dead_code)]
#[inline]
pub(super) fn fd9_scalar(
    f: &GridFn1D<f64>,
    x: f64,
    delta: f64,
    coeffs: &[f64; 9],
    deriv: u32,
) -> Result<f64, SemiflowError> {
    let ks: [f64; 9] = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let mut sum = 0.0_f64;
    for j in 0..9 {
        sum += coeffs[j] * f.sample(x + ks[j] * delta)?;
    }
    let denom = libm::pow(delta, f64::from(deriv));
    Ok(sum / denom)
}

/// SIMD 9-pt stencil: 4+4+1 split (f64).
#[cfg(feature = "simd")]
#[allow(clippy::similar_names)]
#[inline]
pub(super) fn fd9_simd(
    f: &GridFn1D<f64>,
    x: f64,
    delta: f64,
    coeffs: &[f64; 9],
    deriv: u32,
) -> Result<f64, SemiflowError> {
    let fa0 = f.sample(x - 4.0 * delta)?;
    let fa1 = f.sample(x - 3.0 * delta)?;
    let fa2 = f.sample(x - 2.0 * delta)?;
    let fa3 = f.sample(x - 1.0 * delta)?;
    let fb0 = f.sample(x + 1.0 * delta)?;
    let fb1 = f.sample(x + 2.0 * delta)?;
    let fb2 = f.sample(x + 3.0 * delta)?;
    let fb3 = f.sample(x + 4.0 * delta)?;

    let vals_a = [fa0, fa1, fa2, fa3];
    let vals_b = [fb0, fb1, fb2, fb3];
    let wa = [coeffs[0], coeffs[1], coeffs[2], coeffs[3]];
    let wb = [coeffs[5], coeffs[6], coeffs[7], coeffs[8]];

    let va = F64x4::load_unaligned(&vals_a);
    let vb = F64x4::load_unaligned(&vals_b);
    let wa_v = F64x4::load_unaligned(&wa);
    let wb_v = F64x4::load_unaligned(&wb);

    let sum_a = va.mul(wa_v).horizontal_sum();
    let sum_b = vb.mul(wb_v).horizontal_sum();
    let tail = coeffs[4] * f.sample(x)?;

    let denom = libm::pow(delta, f64::from(deriv));
    Ok(((sum_a + sum_b) + tail) / denom)
}

/// Apply 9-point Fornberg FD stencil (f64, SIMD dispatch).
#[inline]
pub(super) fn fd9_f64(
    f: &GridFn1D<f64>,
    x: f64,
    delta: f64,
    coeffs: &[f64; 9],
    deriv: u32,
) -> Result<f64, SemiflowError> {
    #[cfg(feature = "simd")]
    {
        if cfg!(test) && crate::simd::FORCE_SCALAR.with(core::cell::Cell::get) {
            return fd9_scalar(f, x, delta, coeffs, deriv);
        }
        fd9_simd(f, x, delta, coeffs, deriv)
    }
    #[cfg(not(feature = "simd"))]
    fd9_scalar(f, x, delta, coeffs, deriv)
}

/// ζ⁶ τ²-correction with 9-point Fornberg FD (math.md §9.2.6, NORMATIVE, f64).
#[allow(clippy::similar_names)]
#[inline]
pub(super) fn zeta6_correction_f64(
    dc: &Diffusion6thChernoff<f64>,
    tau: f64,
    f: &GridFn1D<f64>,
    x: f64,
) -> Result<f64, SemiflowError> {
    let delta = (4.0 * dc.grid.dx()).max(libm::sqrt(tau));

    let a_val = dc.eval_a(x);
    let a_prime_val = dc.eval_ap(x);
    let a_dbl_val = dc.eval_app(x);

    if a_prime_val == 0.0 && a_dbl_val == 0.0 {
        return Ok(0.0);
    }

    let fd1 = fd9_f64(f, x, delta, &C1_9, 1)?;
    let fd2 = fd9_f64(f, x, delta, &C2_9, 2)?;
    let fd3 = fd9_f64(f, x, delta, &C3_9, 3)?;

    Ok(tau
        * tau
        * (a_val * a_prime_val * fd3
            + (a_val * a_dbl_val / 2.0) * fd2
            + (a_prime_val * a_dbl_val / 4.0) * fd1))
}

/// Apply ζ⁶ at a single grid node `i` (f64 path).
#[inline]
pub(super) fn apply_at_node_f64(
    dc: &Diffusion6thChernoff<f64>,
    tau: f64,
    f: &GridFn1D<f64>,
    i: usize,
) -> Result<f64, SemiflowError> {
    let x = dc.grid.x_at(i);
    Ok(gamma6_a_baseline_f64(dc, tau, f, x)? + zeta6_correction_f64(dc, tau, f, x)?)
}
