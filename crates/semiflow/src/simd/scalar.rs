//! Scalar fallback implementations for all SIMD traits. NO `unsafe`.
//!
//! Always compiled — each platform needs at least one of these fallbacks:
//! - `x86_64+avx2`: provides `F32x4Scalar` (no native AVX2 f32x4 in our trait)
//! - aarch64+neon: provides `F64x4Scalar`, `F32x8Scalar` (no f64x4/f32x8 in NEON)
//! - other arches: provides all three scalar fallbacks
//! - cfg(test): force-scalar hook uses these on all arches (ADR-0019 §6.5A)
//!
//! Items that are unused on a given arch are suppressed by #[`allow(dead_code)`]
//! to avoid warnings without conditional compilation noise.

use super::{SimdF32x4, SimdF32x8, SimdF64x4};

/// 4-lane f64 SIMD backed by a plain `[f64; 4]` array. No unsafe.
#[allow(dead_code)] // primary alias on non-avx2/non-neon arches; fallback in tests
#[derive(Clone, Copy)]
pub(crate) struct F64x4Scalar([f64; 4]);

impl SimdF64x4 for F64x4Scalar {
    #[inline]
    fn splat(x: f64) -> Self {
        F64x4Scalar([x; 4])
    }

    #[inline]
    fn load_unaligned(src: &[f64; 4]) -> Self {
        F64x4Scalar(*src)
    }

    #[inline]
    fn store_unaligned(self, dst: &mut [f64; 4]) {
        *dst = self.0;
    }

    #[inline]
    fn add(self, r: Self) -> Self {
        F64x4Scalar([
            self.0[0] + r.0[0],
            self.0[1] + r.0[1],
            self.0[2] + r.0[2],
            self.0[3] + r.0[3],
        ])
    }

    #[inline]
    fn sub(self, r: Self) -> Self {
        F64x4Scalar([
            self.0[0] - r.0[0],
            self.0[1] - r.0[1],
            self.0[2] - r.0[2],
            self.0[3] - r.0[3],
        ])
    }

    #[inline]
    fn mul(self, r: Self) -> Self {
        F64x4Scalar([
            self.0[0] * r.0[0],
            self.0[1] * r.0[1],
            self.0[2] * r.0[2],
            self.0[3] * r.0[3],
        ])
    }

    /// Deterministic horizontal sum: `((l0 + l1) + l2) + l3`.
    #[inline]
    fn horizontal_sum(self) -> f64 {
        ((self.0[0] + self.0[1]) + self.0[2]) + self.0[3]
    }
}

// ---------------------------------------------------------------------------
// f32 scalar fallbacks (ADR-0175, Phase 5b).
// ---------------------------------------------------------------------------

/// 8-lane f32 SIMD backed by `[f32; 8]`. No unsafe. Used on non-AVX2 arches.
#[allow(dead_code)] // primary alias on non-avx2 arches; fallback in tests on avx2
#[derive(Clone, Copy)]
pub(crate) struct F32x8Scalar([f32; 8]);

impl SimdF32x8 for F32x8Scalar {
    #[inline]
    fn splat(x: f32) -> Self {
        F32x8Scalar([x; 8])
    }

    #[inline]
    fn load_unaligned(src: &[f32; 8]) -> Self {
        F32x8Scalar(*src)
    }

    #[inline]
    fn store_unaligned(self, dst: &mut [f32; 8]) {
        *dst = self.0;
    }

    #[inline]
    fn add(self, r: Self) -> Self {
        F32x8Scalar([
            self.0[0] + r.0[0],
            self.0[1] + r.0[1],
            self.0[2] + r.0[2],
            self.0[3] + r.0[3],
            self.0[4] + r.0[4],
            self.0[5] + r.0[5],
            self.0[6] + r.0[6],
            self.0[7] + r.0[7],
        ])
    }

    #[inline]
    fn sub(self, r: Self) -> Self {
        F32x8Scalar([
            self.0[0] - r.0[0],
            self.0[1] - r.0[1],
            self.0[2] - r.0[2],
            self.0[3] - r.0[3],
            self.0[4] - r.0[4],
            self.0[5] - r.0[5],
            self.0[6] - r.0[6],
            self.0[7] - r.0[7],
        ])
    }

    #[inline]
    fn mul(self, r: Self) -> Self {
        F32x8Scalar([
            self.0[0] * r.0[0],
            self.0[1] * r.0[1],
            self.0[2] * r.0[2],
            self.0[3] * r.0[3],
            self.0[4] * r.0[4],
            self.0[5] * r.0[5],
            self.0[6] * r.0[6],
            self.0[7] * r.0[7],
        ])
    }

    /// Fixed-tree horizontal sum: `(((l0+l1)+(l2+l3))+((l4+l5)+(l6+l7)))`.
    ///
    /// MUST match the AVX2 store-then-add-tree order exactly (ADR-0175).
    #[inline]
    fn horizontal_sum(self) -> f32 {
        let lo = (self.0[0] + self.0[1]) + (self.0[2] + self.0[3]);
        let hi = (self.0[4] + self.0[5]) + (self.0[6] + self.0[7]);
        lo + hi
    }
}

/// 4-lane f32 SIMD backed by `[f32; 4]`. No unsafe. Used on non-NEON arches,
/// including `x86_64+avx2` (no native f32x4 type in our AVX2 trait surface).
#[allow(dead_code)] // used on x86_64+avx2 and all non-neon arches
#[derive(Clone, Copy)]
pub(crate) struct F32x4Scalar([f32; 4]);

impl SimdF32x4 for F32x4Scalar {
    #[inline]
    fn splat(x: f32) -> Self {
        F32x4Scalar([x; 4])
    }

    #[inline]
    fn load_unaligned(src: &[f32; 4]) -> Self {
        F32x4Scalar(*src)
    }

    #[inline]
    fn store_unaligned(self, dst: &mut [f32; 4]) {
        *dst = self.0;
    }

    #[inline]
    fn add(self, r: Self) -> Self {
        F32x4Scalar([
            self.0[0] + r.0[0],
            self.0[1] + r.0[1],
            self.0[2] + r.0[2],
            self.0[3] + r.0[3],
        ])
    }

    #[inline]
    fn sub(self, r: Self) -> Self {
        F32x4Scalar([
            self.0[0] - r.0[0],
            self.0[1] - r.0[1],
            self.0[2] - r.0[2],
            self.0[3] - r.0[3],
        ])
    }

    #[inline]
    fn mul(self, r: Self) -> Self {
        F32x4Scalar([
            self.0[0] * r.0[0],
            self.0[1] * r.0[1],
            self.0[2] * r.0[2],
            self.0[3] * r.0[3],
        ])
    }

    /// Deterministic horizontal sum: `((l0 + l1) + l2) + l3`.
    #[inline]
    fn horizontal_sum(self) -> f32 {
        ((self.0[0] + self.0[1]) + self.0[2]) + self.0[3]
    }
}
