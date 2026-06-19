//! Catmull-Rom cubic-Hermite SIMD dispatch (Phase-3, ADR-0019).
//!
//! Extracted from `grid.rs` to keep that module ≤500 lines.
//! Provides `catmull_rom_dispatch` used by `Grid1D::interp` (`CubicHermite` arm).
//!
//! SIMD path reformulates as `0.5 * dot(coeffs(s), [pm1, p0, p1, p2])`.
//! Bit-equality to scalar verified by `SIMD_BIT_EQUAL` gate (`simd_bit_equal.rs`).

#[cfg(feature = "simd")]
use crate::simd::{F64x4, SimdF64x4};

/// Catmull-Rom kernel — scalar path.
///
/// `result = 0.5 * (2·p0 + (−pm1+p1)·s + (2·pm1−5·p0+4·p1−p2)·s²
///                        + (−pm1+3·p0−3·p1+p2)·s³)`.
#[allow(dead_code)] // used under #[cfg(not(feature = "simd"))] and force-scalar hook
#[inline]
pub(crate) fn catmull_rom_scalar(pm1: f64, p0: f64, p1: f64, p2: f64, s: f64) -> f64 {
    let s2 = s * s;
    let s3 = s2 * s;
    0.5 * ((2.0 * p0)
        + (-pm1 + p1) * s
        + (2.0 * pm1 - 5.0 * p0 + 4.0 * p1 - p2) * s2
        + (-pm1 + 3.0 * p0 - 3.0 * p1 + p2) * s3)
}

/// Catmull-Rom kernel — SIMD path.
///
/// Coefficients per control point (derived by expanding scalar formula):
/// `[-s+2s²-s³, 2-5s²+3s³, s+4s²-3s³, -s²+s³]`.
#[cfg(feature = "simd")]
#[inline]
fn catmull_rom_simd(pm1: f64, p0: f64, p1: f64, p2: f64, s: f64) -> f64 {
    let s2 = s * s;
    let s3 = s2 * s;
    let pts = [pm1, p0, p1, p2];
    let coeffs = [
        -s + 2.0 * s2 - s3,
        2.0 - 5.0 * s2 + 3.0 * s3,
        s + 4.0 * s2 - 3.0 * s3,
        -s2 + s3,
    ];
    0.5 * F64x4::load_unaligned(&pts)
        .mul(F64x4::load_unaligned(&coeffs))
        .horizontal_sum()
}

/// Catmull-Rom dispatcher — SIMD when `simd` feature active, scalar otherwise.
///
/// `cfg!(test) && FORCE_SCALAR` collapses to false in release builds (zero cost).
#[inline]
pub(crate) fn catmull_rom(pm1: f64, p0: f64, p1: f64, p2: f64, s: f64) -> f64 {
    #[cfg(feature = "simd")]
    {
        if cfg!(test) && crate::simd::FORCE_SCALAR.with(core::cell::Cell::get) {
            return catmull_rom_scalar(pm1, p0, p1, p2, s);
        }
        catmull_rom_simd(pm1, p0, p1, p2, s)
    }
    #[cfg(not(feature = "simd"))]
    catmull_rom_scalar(pm1, p0, p1, p2, s)
}
