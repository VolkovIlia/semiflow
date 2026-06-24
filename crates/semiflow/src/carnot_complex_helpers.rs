//! Complex-diffusion helpers for `carnot_complex.rs`.
//!
//! Extracted to keep `carnot_complex.rs` ≤ 500 `LoC` (suckless default).
//!
//! Contents:
//! - `cplx_diffuse_x1`: apply `exp(σ·X₁²)` with complex σ to a `CplxGridFn5`.
//! - `cplx_diffuse_x2`: apply `exp(σ·X₂²)` with complex σ to a `CplxGridFn5`.
//! - `sample_5d_cplx`: 5D multilinear interpolation for complex grids.
//! - `apply_im_correction`: first-order imaginary-displacement correction.

extern crate alloc;

use num_complex::Complex;

use crate::{
    carnot_complex::CplxGridFn5,
    carnot_stepk_helpers::lerp_idx_1d_f64,
    error::SemiflowError,
    grid_nd::GridND,
    hormander_engel_helpers::{GH32_NODES_ENGEL, GH32_WEIGHTS_ENGEL},
};

/// Apply `exp(sigma·X₁²)` with complex sigma to a complex grid function.
///
/// X₁ = ∂_{x₁} — trivial flow, no coupling. The displacement along x₁ is
/// `s = √(2σ)·ξₖ` (complex). We evaluate `f(x₁−s, x₂,…,x₅)` via bilinear
/// interp in x₁ at the real part of the displacement, with a first-order
/// imaginary correction:
///   `f(x₁−s) ≈ f(x₁−Re(s)) − i·Im(s)·∂_{x₁}f(x₁−Re(s))`
///
/// The GH32 quadrature: `(exp(σ·X₁²)f)(x) = π^{-1/2} Σ wₖ f(x₁−√(2σ)·ξₖ,…)`.
// Result kept for API symmetry with filiform5 variants and future error paths.
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn cplx_diffuse_x1(
    src: &CplxGridFn5,
    dst: &mut CplxGridFn5,
    sigma: Complex<f64>,
) -> Result<(), SemiflowError> {
    if sigma.re == 0.0 && sigma.im == 0.0 {
        dst.values.copy_from_slice(&src.values);
        return Ok(());
    }
    let g = &src.grid;
    let [n0, n1, n2, n3, n4] = [
        g.axes[0].n,
        g.axes[1].n,
        g.axes[2].n,
        g.axes[3].n,
        g.axes[4].n,
    ];
    let sqrt2s = (sigma + sigma).sqrt();
    let pi_inv_sqrt = 1.0 / libm::sqrt(core::f64::consts::PI);
    for i4 in 0..n4 {
        for i3 in 0..n3 {
            for i2 in 0..n2 {
                for i1 in 0..n1 {
                    for i0 in 0..n0 {
                        let x0 = g.axes[0].x_at(i0);
                        let mut val = Complex::new(0.0, 0.0);
                        for (&xi, &wi) in GH32_NODES_ENGEL.iter().zip(GH32_WEIGHTS_ENGEL.iter()) {
                            let disp = sqrt2s * xi;
                            let x0_src_re = x0 - disp.re;
                            let (k0, k1, alpha) = lerp_idx_1d_f64(x0_src_re, &g.axes[0]);
                            let f0 = src.values[g.flat_idx(&[k0, i1, i2, i3, i4])];
                            let f1 = src.values[g.flat_idx(&[k1, i1, i2, i3, i4])];
                            let fv = f0 * (1.0 - alpha) + f1 * alpha;
                            // First-order imaginary correction.
                            let dx = (g.axes[0].x_at(k1) - g.axes[0].x_at(k0)).abs().max(1e-30);
                            let f_prime = (f1 - f0) / dx;
                            let fv_cplx = fv - Complex::new(0.0, disp.im) * f_prime;
                            val += fv_cplx * wi;
                        }
                        let flat = g.flat_idx(&[i0, i1, i2, i3, i4]);
                        dst.values[flat] = val * pi_inv_sqrt;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Apply `exp(sigma·X₂²)` with complex sigma to a complex grid function.
///
/// Uses the X₂ integral curve (polynomial coupling in 5 coordinates).
/// Source displacement `s = √(2σ)·ξₖ` is complex; the real part is used for
/// the grid lookup, with a first-order imaginary correction via `apply_im_correction`.
// Result kept for API symmetry with filiform5 variants and future error paths.
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn cplx_diffuse_x2(
    src: &CplxGridFn5,
    dst: &mut CplxGridFn5,
    sigma: Complex<f64>,
) -> Result<(), SemiflowError> {
    if sigma.re == 0.0 && sigma.im == 0.0 {
        dst.values.copy_from_slice(&src.values);
        return Ok(());
    }
    let g = &src.grid;
    let [n0, n1, n2, n3, n4] = [
        g.axes[0].n,
        g.axes[1].n,
        g.axes[2].n,
        g.axes[3].n,
        g.axes[4].n,
    ];
    let sqrt2s = (sigma + sigma).sqrt();
    let pi_inv_sqrt = 1.0 / libm::sqrt(core::f64::consts::PI);
    for i4 in 0..n4 {
        for i3 in 0..n3 {
            for i2 in 0..n2 {
                let x3c = g.axes[2].x_at(i2);
                let x4c = g.axes[3].x_at(i3);
                let x5c = g.axes[4].x_at(i4);
                for i1 in 0..n1 {
                    let x2n = g.axes[1].x_at(i1);
                    for i0 in 0..n0 {
                        let x1f = g.axes[0].x_at(i0);
                        let val = cplx_x2_gh_sum(src, sqrt2s, x1f, x2n, x3c, x4c, x5c, g);
                        let flat = g.flat_idx(&[i0, i1, i2, i3, i4]);
                        dst.values[flat] = val * pi_inv_sqrt;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Accumulate the 32-point GH quadrature sum for one (x₁,x₂,x₃,x₄,x₅) point
/// in the complex X₂ diffusion step.
///
/// Returns `Σ wₖ · f(Φ^{−s_re}(x)) · (imaginary correction)` before π^{-1/2}
/// scaling. Extracted from `cplx_diffuse_x2` to satisfy the 50-line function cap.
#[allow(clippy::too_many_arguments)]
fn cplx_x2_gh_sum(
    src: &CplxGridFn5,
    sqrt2s: Complex<f64>,
    x1f: f64,
    x2n: f64,
    x3c: f64,
    x4c: f64,
    x5c: f64,
    g: &GridND<f64, 5>,
) -> Complex<f64> {
    let mut val = Complex::new(0.0, 0.0);
    for (&xi, &wi) in GH32_NODES_ENGEL.iter().zip(GH32_WEIGHTS_ENGEL.iter()) {
        let s = sqrt2s * xi;
        let s_re = s.re;
        let x2_src = x2n - s_re;
        let x3_src = x3c - s_re * x1f;
        let x4_src = x4c - s_re * s_re * x1f * 0.5 - s_re * x3c;
        let x5_src = x5c - s_re * s_re * s_re * x1f / 6.0 - s_re * s_re * x3c * 0.5 - s_re * x4c;
        let fv = sample_5d_cplx(src, x1f, x2_src, x3_src, x4_src, x5_src);
        let fv_corr = apply_im_correction(src, fv, s.im, x1f, x2_src, x3_src, x4_src, x5_src, g);
        val += fv_corr * wi;
    }
    val
}

/// First-order imaginary-displacement correction for the X₂ diffusion step.
///
/// Corrects `f(x−s)` for the imaginary component of `s`, using the X₂ direction:
///   `f(x−s) ≈ f(x−Re(s)) − i·Im(s)·X₂·f(x−Re(s))`
/// where `X₂·f = ∂_{x₂}f + x₁·∂_{x₃}f` (leading coupling terms only).
///
/// The ∂x₂ and ∂x₃ finite-difference stencils are centred at the ACTUAL source
/// coordinates `(x4_src, x5_src)` — not at zero — to avoid mislocating the sample.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_im_correction(
    src: &CplxGridFn5,
    fv: Complex<f64>,
    s_im: f64,
    x1f: f64,
    x2_src: f64,
    x3_src: f64,
    x4_src: f64,
    x5_src: f64,
    g: &GridND<f64, 5>,
) -> Complex<f64> {
    if s_im.abs() < 1e-30 {
        return fv;
    }
    let eps2 = (g.axes[1].x_at(1) - g.axes[1].x_at(0)).abs().max(1e-30);
    let f_plus = sample_5d_cplx(src, x1f, x2_src + eps2, x3_src, x4_src, x5_src);
    let f_minus = sample_5d_cplx(src, x1f, x2_src - eps2, x3_src, x4_src, x5_src);
    let df_dx2 = (f_plus - f_minus) * (0.5 / eps2);
    let eps3 = (g.axes[2].x_at(1) - g.axes[2].x_at(0)).abs().max(1e-30);
    let f3p = sample_5d_cplx(src, x1f, x2_src, x3_src + eps3, x4_src, x5_src);
    let f3m = sample_5d_cplx(src, x1f, x2_src, x3_src - eps3, x4_src, x5_src);
    let df_dx3 = (f3p - f3m) * (0.5 / eps3);
    let x2_dir = df_dx2 + df_dx3 * x1f;
    fv - Complex::new(0.0, s_im) * x2_dir
}

/// Sample a 5D complex grid function via multilinear (32-corner) interpolation.
#[inline]
pub(crate) fn sample_5d_cplx(
    src: &CplxGridFn5,
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    x4: f64,
) -> Complex<f64> {
    let g = &src.grid;
    let (k0, k0h, a0) = lerp_idx_1d_f64(x0, &g.axes[0]);
    let (k1, k1h, a1) = lerp_idx_1d_f64(x1, &g.axes[1]);
    let (k2, k2h, a2) = lerp_idx_1d_f64(x2, &g.axes[2]);
    let (k3, k3h, a3) = lerp_idx_1d_f64(x3, &g.axes[3]);
    let (k4, k4h, a4) = lerp_idx_1d_f64(x4, &g.axes[4]);
    let b0 = 1.0 - a0;
    let b1 = 1.0 - a1;
    let b2 = 1.0 - a2;
    let b3 = 1.0 - a3;
    let b4 = 1.0 - a4;
    let v = |i0, i1, i2, i3, i4| src.values[g.flat_idx(&[i0, i1, i2, i3, i4])];
    let s4 = |jk4: usize| {
        b3 * (b2
            * (b1 * (b0 * v(k0, k1, k2, k3, jk4) + a0 * v(k0h, k1, k2, k3, jk4))
                + a1 * (b0 * v(k0, k1h, k2, k3, jk4) + a0 * v(k0h, k1h, k2, k3, jk4)))
            + a2 * (b1 * (b0 * v(k0, k1, k2h, k3, jk4) + a0 * v(k0h, k1, k2h, k3, jk4))
                + a1 * (b0 * v(k0, k1h, k2h, k3, jk4) + a0 * v(k0h, k1h, k2h, k3, jk4))))
            + a3 * (b2
                * (b1 * (b0 * v(k0, k1, k2, k3h, jk4) + a0 * v(k0h, k1, k2, k3h, jk4))
                    + a1 * (b0 * v(k0, k1h, k2, k3h, jk4) + a0 * v(k0h, k1h, k2, k3h, jk4)))
                + a2 * (b1 * (b0 * v(k0, k1, k2h, k3h, jk4) + a0 * v(k0h, k1, k2h, k3h, jk4))
                    + a1 * (b0 * v(k0, k1h, k2h, k3h, jk4) + a0 * v(k0h, k1h, k2h, k3h, jk4))))
    };
    b4 * s4(k4) + a4 * s4(k4h)
}
