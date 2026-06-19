//! Scalar fallback for non-x86_64/aarch64 arches. NO `unsafe`.
//!
//! Also compiled under `#[cfg(test)]` on all arches to enable the
//! force-scalar test hook (ADR-0019, Section 6.5 option A).

#![cfg(any(
    test,
    not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))
))]

use super::SimdF64x4;

/// 4-lane f64 SIMD backed by a plain `[f64; 4]` array. No unsafe.
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
