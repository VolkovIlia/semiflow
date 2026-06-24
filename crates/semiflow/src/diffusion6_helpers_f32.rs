//! Private f32-specific helpers for `Diffusion6thChernoff<f32>` (ADR-0175, Phase 5b).
//!
//! Declared as `#[path = "diffusion6_helpers_f32.rs"] mod helpers_f32;` inside
//! `diffusion6.rs` — this file is a child of that module, so `super::` works.
//!
//! SIMD path: `fd9_simd_f32` uses `F32x8` (8-lane AVX2 / scalar fallback).
//! Catmull-Rom sampling: `gamma6_a_baseline_f32` uses `sample_f32` which
//! dispatches to `catmull_rom_f32` (4-lane NEON / scalar).
//!
//! Determinism contract (ADR-0175): SIMD output is byte-identical to scalar
//! via `FORCE_SCALAR` hook; verified by `SIMD_F32_BIT_EQUAL` gate.
//!
//! `cast_possible_truncation`: This module intentionally casts f64 constants
//! and coefficients to f32 — precision loss is the explicit design (ADR-0175).
#![allow(clippy::cast_possible_truncation)]

pub(super) use diffusion_zeta_common::{
    validate_a_x_generic as validate_a_x_f32, validate_tau_generic as validate_tau_f32,
};
use num_traits::Float;

use super::{Diffusion6thChernoff, C1_9, C2_9, C3_9, K7_P, K7_W0, K7_W1, K7_W2, K7_W3};
#[cfg(feature = "simd")]
use crate::simd::{F32x8, SimdF32x8};
use crate::{diffusion_zeta_common, error::SemiflowError, grid_fn::GridFn1D};

// ---------------------------------------------------------------------------
// γ⁶-A baseline (f32, uses sample_f32 with catmull_rom_f32 SIMD dispatcher).
// ---------------------------------------------------------------------------

/// γ⁶-A baseline: `D_γ⁶(τ) = S(τ/2) ∘ K7(τ;a) ∘ S(τ/2)` (f32, SIMD path).
#[inline]
pub(super) fn gamma6_a_baseline_f32(
    dc: &Diffusion6thChernoff<f32>,
    tau: f32,
    f: &GridFn1D<f32>,
    x: f32,
) -> Result<f32, SemiflowError> {
    let s_half = 0.5_f32 * tau;
    let w0 = K7_W0 as f32;
    let w1 = K7_W1 as f32;
    let w2 = K7_W2 as f32;
    let w3 = K7_W3 as f32;
    let kp = K7_P as f32;

    let x_pre = x + s_half * (dc.a_prime)(x);
    let a_pre = (dc.a)(x_pre);
    validate_a_x_f32(a_pre, x_pre)?;

    let h = 2.0_f32 * Float::sqrt(a_pre * tau);
    let h_3 = 2.0_f32 * Float::sqrt(3.0_f32 * a_pre * tau);
    let j_5 = 2.0_f32 * Float::sqrt(kp * a_pre * tau);

    let post = |xr: f32| xr + s_half * (dc.a_prime)(xr);

    let v_c = sample_f32(f, post(x_pre))?;
    let vnp = sample_f32(f, post(x_pre + h))?;
    let vnn = sample_f32(f, post(x_pre - h))?;
    let vfp = sample_f32(f, post(x_pre + h_3))?;
    let vfn = sample_f32(f, post(x_pre - h_3))?;
    let vep = sample_f32(f, post(x_pre + j_5))?;
    let ven = sample_f32(f, post(x_pre - j_5))?;

    Ok(w0 * v_c + w1 * (vnp + vnn) + w2 * (vfp + vfn) + w3 * (vep + ven))
}

// ---------------------------------------------------------------------------
// 9-point Fornberg FD stencil — f32 scalar reference path.
// ---------------------------------------------------------------------------

/// f32 scalar 9-pt FD stencil. Reduction order matches `fd9_simd_f32`.
///
/// Tree: `(((fa0*c0 + fa1*c1) + (fa2*c2 + fa3*c3)) + ((fb0*c5 + fb1*c6) + (fb2*c7 + fb3*c8)))`
/// then `+ tail`. Must be identical to `F32x8Scalar::horizontal_sum` applied on the same data.
#[allow(dead_code)]
#[inline]
pub(super) fn fd9_scalar_f32(
    f: &GridFn1D<f32>,
    x: f32,
    delta: f32,
    coeffs: &[f64; 9],
    deriv: i32,
) -> Result<f32, SemiflowError> {
    // Samples at offsets [-4,-3,-2,-1] and [1,2,3,4].
    let fa0 = sample_f32(f, x - 4.0_f32 * delta)?;
    let fa1 = sample_f32(f, x - 3.0_f32 * delta)?;
    let fa2 = sample_f32(f, x - 2.0_f32 * delta)?;
    let fa3 = sample_f32(f, x - 1.0_f32 * delta)?;
    let fb0 = sample_f32(f, x + 1.0_f32 * delta)?;
    let fb1 = sample_f32(f, x + 2.0_f32 * delta)?;
    let fb2 = sample_f32(f, x + 3.0_f32 * delta)?;
    let fb3 = sample_f32(f, x + 4.0_f32 * delta)?;

    // Cast f64 coefficients to f32.
    let (c0, c1, c2, c3) = (
        coeffs[0] as f32,
        coeffs[1] as f32,
        coeffs[2] as f32,
        coeffs[3] as f32,
    );
    let (c5, c6, c7, c8) = (
        coeffs[5] as f32,
        coeffs[6] as f32,
        coeffs[7] as f32,
        coeffs[8] as f32,
    );
    let c4 = coeffs[4] as f32;

    // Reduction tree matching F32x8Scalar::horizontal_sum:
    //   (((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7))) + tail
    let lo = (fa0 * c0 + fa1 * c1) + (fa2 * c2 + fa3 * c3);
    let hi = (fb0 * c5 + fb1 * c6) + (fb2 * c7 + fb3 * c8);
    let sum_ab = lo + hi;
    let tail = c4 * sample_f32(f, x)?;

    let denom = Float::powi(delta, deriv);
    Ok((sum_ab + tail) / denom)
}

// ---------------------------------------------------------------------------
// 9-point Fornberg FD stencil — f32 SIMD path (F32x8, 8+1 split).
// ---------------------------------------------------------------------------

/// SIMD 9-pt stencil for f32: 8+1 split using `F32x8` (ADR-0175, Phase 5b).
#[cfg(feature = "simd")]
#[allow(clippy::similar_names)]
#[inline]
pub(super) fn fd9_simd_f32(
    f: &GridFn1D<f32>,
    x: f32,
    delta: f32,
    coeffs: &[f64; 9],
    deriv: i32,
) -> Result<f32, SemiflowError> {
    // Samples at offsets [-4,-3,-2,-1, 1,2,3,4] — pack into 8 lanes.
    let fa0 = sample_f32(f, x - 4.0_f32 * delta)?;
    let fa1 = sample_f32(f, x - 3.0_f32 * delta)?;
    let fa2 = sample_f32(f, x - 2.0_f32 * delta)?;
    let fa3 = sample_f32(f, x - 1.0_f32 * delta)?;
    let fb0 = sample_f32(f, x + 1.0_f32 * delta)?;
    let fb1 = sample_f32(f, x + 2.0_f32 * delta)?;
    let fb2 = sample_f32(f, x + 3.0_f32 * delta)?;
    let fb3 = sample_f32(f, x + 4.0_f32 * delta)?;

    let vals: [f32; 8] = [fa0, fa1, fa2, fa3, fb0, fb1, fb2, fb3];
    let ws: [f32; 8] = [
        coeffs[0] as f32,
        coeffs[1] as f32,
        coeffs[2] as f32,
        coeffs[3] as f32,
        coeffs[5] as f32,
        coeffs[6] as f32,
        coeffs[7] as f32,
        coeffs[8] as f32,
    ];

    let vv = F32x8::load_unaligned(&vals);
    let vw = F32x8::load_unaligned(&ws);
    let sum_ab = vv.mul(vw).horizontal_sum();

    let tail = (coeffs[4] as f32) * sample_f32(f, x)?;
    let denom = Float::powi(delta, deriv);
    Ok((sum_ab + tail) / denom)
}

// ---------------------------------------------------------------------------
// fd9 dispatcher (f32).
// ---------------------------------------------------------------------------

/// Apply 9-point Fornberg FD stencil (f32, SIMD dispatch).
#[inline]
pub(super) fn fd9_f32(
    f: &GridFn1D<f32>,
    x: f32,
    delta: f32,
    coeffs: &[f64; 9],
    deriv: i32,
) -> Result<f32, SemiflowError> {
    #[cfg(feature = "simd")]
    {
        if cfg!(test) && crate::simd::FORCE_SCALAR.with(core::cell::Cell::get) {
            return fd9_scalar_f32(f, x, delta, coeffs, deriv);
        }
        fd9_simd_f32(f, x, delta, coeffs, deriv)
    }
    #[cfg(not(feature = "simd"))]
    fd9_scalar_f32(f, x, delta, coeffs, deriv)
}

// ---------------------------------------------------------------------------
// ζ⁶ τ²-correction (f32).
// ---------------------------------------------------------------------------

/// ζ⁶ τ²-correction with 9-point Fornberg FD (f32 path).
#[allow(clippy::similar_names)]
#[inline]
pub(super) fn zeta6_correction_f32(
    dc: &Diffusion6thChernoff<f32>,
    tau: f32,
    f: &GridFn1D<f32>,
    x: f32,
) -> Result<f32, SemiflowError> {
    let delta = Float::max(4.0_f32 * dc.grid.dx(), Float::sqrt(tau));

    let a_val = (dc.a)(x);
    let a_prime_val = (dc.a_prime)(x);
    let a_dbl_val = (dc.a_double_prime)(x);

    if a_prime_val == 0.0_f32 && a_dbl_val == 0.0_f32 {
        return Ok(0.0_f32);
    }

    let fd1 = fd9_f32(f, x, delta, &C1_9, 1)?;
    let fd2 = fd9_f32(f, x, delta, &C2_9, 2)?;
    let fd3 = fd9_f32(f, x, delta, &C3_9, 3)?;

    Ok(tau
        * tau
        * (a_val * a_prime_val * fd3
            + (a_val * a_dbl_val / 2.0_f32) * fd2
            + (a_prime_val * a_dbl_val / 4.0_f32) * fd1))
}

// ---------------------------------------------------------------------------
// Per-node apply (f32).
// ---------------------------------------------------------------------------

/// Apply ζ⁶ at a single grid node `i` (f32 SIMD path).
#[inline]
pub(super) fn apply_at_node_f32(
    dc: &Diffusion6thChernoff<f32>,
    tau: f32,
    f: &GridFn1D<f32>,
    i: usize,
) -> Result<f32, SemiflowError> {
    let x = dc.grid.x_at(i);
    Ok(gamma6_a_baseline_f32(dc, tau, f, x)? + zeta6_correction_f32(dc, tau, f, x)?)
}

// ---------------------------------------------------------------------------
// f32 sample helper (calls catmull_rom_f32 SIMD dispatcher).
// ---------------------------------------------------------------------------

/// Sample `f` at `x` using `catmull_rom_f32` (f32 SIMD dispatcher, ADR-0175).
///
/// Mirrors `GridFn1D<f64>::sample` → `Grid1D<f64>::interp` → `cubic_hermite_at`
/// → `catmull_rom` (SIMD dispatcher), but for f32 values.
// Result<> is kept for API parity with the f64 sample path — callers use `?`.
#[allow(clippy::unnecessary_wraps)]
#[inline]
fn sample_f32(f: &GridFn1D<f32>, x: f32) -> Result<f32, SemiflowError> {
    use num_traits::ToPrimitive;

    use crate::{boundary::bc_value_generic, grid_cubic::catmull_rom_f32};

    let grid = &f.grid;
    let values = &f.values;
    let dx = grid.dx();
    let t_frac = (x - grid.xmin) / dx;
    let t_floor = Float::floor(t_frac);
    #[allow(clippy::cast_possible_truncation)]
    let idx = t_floor.to_i64().unwrap_or(0);
    let s = t_frac - t_floor;

    let bnd = grid.boundary;
    let n = grid.n;
    let pm1 = bc_value_generic(bnd, values, n, idx - 1, dx);
    let p0 = bc_value_generic(bnd, values, n, idx, dx);
    let p1 = bc_value_generic(bnd, values, n, idx + 1, dx);
    let p2 = bc_value_generic(bnd, values, n, idx + 2, dx);

    Ok(catmull_rom_f32(pm1, p0, p1, p2, s))
}
