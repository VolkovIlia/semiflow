//! Private f64-specific helpers for `Diffusion4thChernoff`.
//!
//! Declared as `#[path = "diffusion4_helpers.rs"] mod helpers_f64;` inside
//! `diffusion4.rs` — this file is a child of that module, so `super::` works.

pub(super) use diffusion_zeta_common::{validate_a_x_f64, validate_tau_f64};

use super::{Diffusion4thChernoff, C1, C2, C3, W0, W1, W2};
use crate::{
    boundary::InterpKind, diffusion_zeta_common, error::SemiflowError, grid::Grid1D,
    grid_fn::GridFn1D, sample_table::Sample1D,
};

/// γ-A inner-Strang baseline: `D_γ(τ) = S(τ/2) ∘ K(τ;a) ∘ S(τ/2)`.
///
/// BIT-EQUAL to v0.5.0 `DiffusionChernoff` — do NOT reorder operations.
/// Math.md §9.2.3.A pseudocode.
#[inline]
pub(super) fn gamma_a_baseline_f64<S: Sample1D>(
    dc: &Diffusion4thChernoff<f64>,
    tau: f64,
    f: &S,
    x: f64,
) -> Result<f64, SemiflowError> {
    let s_half = 0.5 * tau;

    let x_pre = x + s_half * dc.eval_ap(x);
    let a_at_pre = dc.eval_a(x_pre);
    validate_a_x_f64(a_at_pre, x_pre)?;

    let h0 = 2.0 * libm::sqrt(a_at_pre * tau);
    let h0_3 = 2.0 * libm::sqrt(3.0 * a_at_pre * tau);

    let center_pos = x_pre + s_half * dc.eval_ap(x_pre);

    let near_p_raw = x_pre + h0;
    let near_p_pos = near_p_raw + s_half * dc.eval_ap(near_p_raw);

    let near_neg_raw = x_pre - h0;
    let near_neg_pos = near_neg_raw + s_half * dc.eval_ap(near_neg_raw);

    let far_p_raw = x_pre + h0_3;
    let far_p_pos = far_p_raw + s_half * dc.eval_ap(far_p_raw);

    let far_neg_raw = x_pre - h0_3;
    let far_neg_pos = far_neg_raw + s_half * dc.eval_ap(far_neg_raw);

    let center = W0 * f.sample(center_pos)?;
    let near = W1 * (f.sample(near_p_pos)? + f.sample(near_neg_pos)?);
    let far = W2 * (f.sample(far_p_pos)? + f.sample(far_neg_pos)?);

    Ok(center + near + far)
}

/// ζ⁴ stencil step `Δ = max(3·dx, τ^{3/4})` and the divisors `Δ¹, Δ², Δ³`.
///
/// Constant over a step: computed once per `apply_into` instead of once per node
/// (`libm::pow` ×4 per node before ADR-0204). Same functions, same arguments,
/// so the values are bit-identical to the per-node ones.
#[derive(Clone, Copy)]
pub(super) struct Zeta4Step {
    delta: f64,
    denom: [f64; 3],
}

impl Zeta4Step {
    pub(super) fn new(dc: &Diffusion4thChernoff<f64>, tau: f64) -> Self {
        let delta = (3.0 * dc.grid.dx()).max(libm::pow(tau, 0.75));
        let denom = [1_u32, 2, 3].map(|d| libm::pow(delta, f64::from(d)));
        Self { delta, denom }
    }
}

/// `Σⱼ coeffs[j]·samples[j] / denom` — the 7-point Fornberg sum in its fixed order.
#[inline]
fn fd7_sum(samples: &[f64; 7], coeffs: &[f64; 7], denom: f64) -> f64 {
    let mut sum = 0.0_f64;
    for j in 0..7 {
        sum += coeffs[j] * samples[j];
    }
    sum / denom
}

/// ζ⁴ τ²-correction with 7-point Fornberg FD (math.md §9.2.4, NORMATIVE).
///
/// Stencil step: `Δ = max(3·dx, τ^{3/4})`.
/// When `a' ≡ 0 ∧ a'' ≡ 0`, all three terms vanish → correction = 0 (Z⁴_const-a).
///
/// The three derivative stencils share their seven sample points, so `f` is
/// sampled once per point (three times before ADR-0204; samples are
/// deterministic, so the result is unchanged bit for bit).
#[inline]
pub(super) fn zeta4_correction_f64<S: Sample1D>(
    dc: &Diffusion4thChernoff<f64>,
    tau: f64,
    f: &S,
    x: f64,
    step: Zeta4Step,
) -> Result<f64, SemiflowError> {
    let a_x = dc.eval_a(x);
    let a_prime_x = dc.eval_ap(x);
    let app_x = dc.eval_app(x);

    // Short-circuit when correction is exactly zero (constant-a path).
    if a_prime_x == 0.0 && app_x == 0.0 {
        return Ok(0.0);
    }

    let ks: [f64; 7] = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let mut samples = [0.0_f64; 7];
    for (v, k) in samples.iter_mut().zip(ks) {
        *v = f.sample(x + k * step.delta)?;
    }
    let f1 = fd7_sum(&samples, &C1, step.denom[0]); // f'   O(Δ⁶)
    let f2 = fd7_sum(&samples, &C2, step.denom[1]); // f''  O(Δ⁶)
    let f3 = fd7_sum(&samples, &C3, step.denom[2]); // f''' O(Δ⁴)

    // τ²·[a·a'·f''' + ½·a·a''·f'' + ¼·a'·a''·f'] (§9.2.4 box formula).
    Ok(tau
        * tau
        * (a_x * a_prime_x * f3 + (a_x * app_x / 2.0) * f2 + (a_prime_x * app_x / 4.0) * f1))
}

/// Apply ζ⁴ at a single grid node `i`: γ-A baseline + ζ⁴ correction (f64).
#[inline]
pub(super) fn apply_at_node_f64<S: Sample1D>(
    dc: &Diffusion4thChernoff<f64>,
    tau: f64,
    f: &S,
    i: usize,
    step: Zeta4Step,
) -> Result<f64, SemiflowError> {
    let x = dc.grid.x_at(i);
    Ok(gamma_a_baseline_f64(dc, tau, f, x)? + zeta4_correction_f64(dc, tau, f, x, step)?)
}

/// Grid through which `apply_into` samples `src` (ADR-0117 / ADR-0090 / ADR-0104).
///
/// Octonic and Chebyshev sampling re-view the values through the KERNEL grid
/// with that interpolant (the octonic flag is checked first, as it always was);
/// otherwise `src` is sampled through its own grid.
pub(super) fn sample_grid(dc: &Diffusion4thChernoff<f64>, src: &GridFn1D<f64>) -> Grid1D<f64> {
    if dc.octonic_sampling {
        dc.grid.with_interp(InterpKind::OctonicHermite)
    } else if dc.chebyshev_sampling {
        dc.grid.with_interp(InterpKind::ChebyshevSpectralWithBC {
            m: dc.chebyshev_m,
            oob_policy: crate::boundary::OobPolicy::Inherit,
        })
    } else {
        src.grid
    }
}
