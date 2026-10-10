//! Private f64-specific helpers for `Diffusion6thChernoff`.
//!
//! Declared as `#[path = "diffusion6_helpers.rs"] mod helpers_f64;` inside
//! `diffusion6.rs` — this file is a child of that module, so `super::` works.

pub(super) use diffusion_zeta_common::{validate_a_x_f64, validate_tau_f64};

use super::{Diffusion6thChernoff, C1_9, C2_9, C3_9, K7_P, K7_W0, K7_W1, K7_W2, K7_W3};
use crate::simd::{F64x4, SimdF64x4};
use crate::{diffusion_zeta_common, error::SemiflowError, sample_table::Sample1D};

/// γ⁶-A baseline: `D_γ⁶(τ) = S(τ/2) ∘ K7(τ;a) ∘ S(τ/2)` (f64, SIMD path).
#[inline]
pub(super) fn gamma6_a_baseline_f64<S: Sample1D>(
    dc: &Diffusion6thChernoff<f64>,
    tau: f64,
    f: &S,
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

/// ζ⁶ stencil step `Δ = max(4·dx, √τ)` and the divisors `Δ¹, Δ², Δ³`.
///
/// Constant over a step: computed once per `apply_into` instead of once per node
/// (ADR-0204). Same functions and arguments, so the values are bit-identical.
#[derive(Clone, Copy)]
pub(super) struct Zeta6Step {
    delta: f64,
    denom: [f64; 3],
}

impl Zeta6Step {
    pub(super) fn new(dc: &Diffusion6thChernoff<f64>, tau: f64) -> Self {
        let delta = (4.0 * dc.grid.dx()).max(libm::sqrt(tau));
        let denom = [1_u32, 2, 3].map(|d| libm::pow(delta, f64::from(d)));
        Self { delta, denom }
    }
}

/// 9-point Fornberg sum — scalar path, reduction order `j = 0..9`.
#[allow(dead_code)] // used by the test force-scalar hook
#[inline]
fn fd9_sum_scalar(samples: &[f64; 9], coeffs: &[f64; 9], denom: f64) -> f64 {
    let mut sum = 0.0_f64;
    for j in 0..9 {
        sum += coeffs[j] * samples[j];
    }
    sum / denom
}

/// 9-point Fornberg sum — lane path: 4+4 lanes plus the centre tap.
#[inline]
fn fd9_sum_simd(samples: &[f64; 9], coeffs: &[f64; 9], denom: f64) -> f64 {
    let vals_a = [samples[0], samples[1], samples[2], samples[3]];
    let vals_b = [samples[5], samples[6], samples[7], samples[8]];
    let wa = [coeffs[0], coeffs[1], coeffs[2], coeffs[3]];
    let wb = [coeffs[5], coeffs[6], coeffs[7], coeffs[8]];

    let sum_a = F64x4::load_unaligned(&vals_a)
        .mul(F64x4::load_unaligned(&wa))
        .horizontal_sum();
    let sum_b = F64x4::load_unaligned(&vals_b)
        .mul(F64x4::load_unaligned(&wb))
        .horizontal_sum();
    let tail = coeffs[4] * samples[4];
    ((sum_a + sum_b) + tail) / denom
}

/// 9-point Fornberg sum (lane path; scalar under the test force-scalar hook).
#[inline]
fn fd9_sum(samples: &[f64; 9], coeffs: &[f64; 9], denom: f64) -> f64 {
    #[cfg(test)]
    if crate::simd::FORCE_SCALAR.with(core::cell::Cell::get) {
        return fd9_sum_scalar(samples, coeffs, denom);
    }
    fd9_sum_simd(samples, coeffs, denom)
}

/// ζ⁶ τ²-correction with 9-point Fornberg FD (math.md §9.2.6, NORMATIVE, f64).
///
/// The three derivative stencils share their nine sample points, so `f` is
/// sampled once per point (three times before ADR-0204; samples are
/// deterministic, so the result is unchanged bit for bit).
#[allow(clippy::similar_names)]
#[inline]
pub(super) fn zeta6_correction_f64<S: Sample1D>(
    dc: &Diffusion6thChernoff<f64>,
    tau: f64,
    f: &S,
    x: f64,
    step: Zeta6Step,
) -> Result<f64, SemiflowError> {
    let a_val = dc.eval_a(x);
    let a_prime_val = dc.eval_ap(x);
    let a_dbl_val = dc.eval_app(x);

    if a_prime_val == 0.0 && a_dbl_val == 0.0 {
        return Ok(0.0);
    }

    let ks: [f64; 9] = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let mut samples = [0.0_f64; 9];
    for (v, k) in samples.iter_mut().zip(ks) {
        *v = f.sample(x + k * step.delta)?;
    }
    let fd1 = fd9_sum(&samples, &C1_9, step.denom[0]);
    let fd2 = fd9_sum(&samples, &C2_9, step.denom[1]);
    let fd3 = fd9_sum(&samples, &C3_9, step.denom[2]);

    Ok(tau
        * tau
        * (a_val * a_prime_val * fd3
            + (a_val * a_dbl_val / 2.0) * fd2
            + (a_prime_val * a_dbl_val / 4.0) * fd1))
}

/// Apply ζ⁶ at a single grid node `i` (f64 path).
#[inline]
pub(super) fn apply_at_node_f64<S: Sample1D>(
    dc: &Diffusion6thChernoff<f64>,
    tau: f64,
    f: &S,
    i: usize,
    step: Zeta6Step,
) -> Result<f64, SemiflowError> {
    let x = dc.grid.x_at(i);
    Ok(gamma6_a_baseline_f64(dc, tau, f, x)? + zeta6_correction_f64(dc, tau, f, x, step)?)
}
