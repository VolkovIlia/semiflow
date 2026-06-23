//! NEON (4-lane f64 = two `float64x2_t`) impl of [`super::SimdF64x4`] (ADR-0019).
//!
//! Compiled only on aarch64 targets; NEON is baseline since ARMv8-A.
//!
//! # Safety
//! All unsafe blocks in this file call NEON intrinsics that require:
//! - Target CPU is aarch64 (AArch64 baseline; NEON is always present).
//! - All pointer operands point to valid, non-null memory (guaranteed by
//!   `&[f64; 4]` / `&mut [f64; 4]` reference invariants).
//!
//! FORBIDDEN: `vfmaq_f64`, `vfmsq_f64`, `vfmaq_lane_f64`, `vaddvq_f64` —
//! these break the bit-equal contract (ADR-0019).

#![cfg(all(target_arch = "aarch64", target_feature = "neon"))]

use core::arch::aarch64::{
    float32x4_t, float64x2_t,
    vaddq_f32, vaddq_f64, vdupq_n_f32, vdupq_n_f64,
    vld1q_f32, vld1q_f64, vmulq_f32, vmulq_f64,
    vst1q_f32, vst1q_f64, vsubq_f32, vsubq_f64,
};

use super::{SimdF32x4, SimdF64x4};

/// NEON 4-lane f64 SIMD wrapper (two 2-lane registers).
#[derive(Clone, Copy)]
pub(crate) struct F64x4Neon {
    lo: float64x2_t,
    hi: float64x2_t,
}

impl SimdF64x4 for F64x4Neon {
    #[inline]
    fn splat(x: f64) -> Self {
        // Safety: vdupq_n_f64 has no preconditions on x.
        unsafe {
            F64x4Neon {
                lo: vdupq_n_f64(x),
                hi: vdupq_n_f64(x),
            }
        }
    }

    #[inline]
    fn load_unaligned(src: &[f64; 4]) -> Self {
        // Safety: src is a valid &[f64; 4] — non-null, 4*8 bytes valid.
        unsafe {
            F64x4Neon {
                lo: vld1q_f64(src.as_ptr()),
                hi: vld1q_f64(src.as_ptr().add(2)),
            }
        }
    }

    #[inline]
    fn store_unaligned(self, dst: &mut [f64; 4]) {
        // Safety: dst is a valid &mut [f64; 4] — non-null, 4*8 bytes writable.
        unsafe {
            vst1q_f64(dst.as_mut_ptr(), self.lo);
            vst1q_f64(dst.as_mut_ptr().add(2), self.hi);
        }
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // Safety: both operands are valid float64x2_t values from prior ops.
        unsafe {
            F64x4Neon {
                lo: vaddq_f64(self.lo, rhs.lo),
                hi: vaddq_f64(self.hi, rhs.hi),
            }
        }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        // Safety: both operands are valid float64x2_t values from prior ops.
        unsafe {
            F64x4Neon {
                lo: vsubq_f64(self.lo, rhs.lo),
                hi: vsubq_f64(self.hi, rhs.hi),
            }
        }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Safety: both operands are valid float64x2_t values from prior ops.
        unsafe {
            F64x4Neon {
                lo: vmulq_f64(self.lo, rhs.lo),
                hi: vmulq_f64(self.hi, rhs.hi),
            }
        }
    }

    /// Deterministic horizontal sum: `((l0 + l1) + l2) + l3`.
    ///
    /// Intentionally uses tmp[] extract rather than `vaddvq_f64` — that
    /// intrinsic performs pair-sum internally in a different order than scalar.
    #[inline]
    fn horizontal_sum(self) -> f64 {
        let mut tmp = [0.0_f64; 4];
        // Safety: tmp is local, properly sized; self.lo/hi are valid float64x2_t.
        unsafe {
            vst1q_f64(tmp.as_mut_ptr(), self.lo);
            vst1q_f64(tmp.as_mut_ptr().add(2), self.hi);
        }
        ((tmp[0] + tmp[1]) + tmp[2]) + tmp[3]
    }
}

// ---------------------------------------------------------------------------
// ADR-0175, Phase 5b: 4-lane f32 NEON impl (`float32x4_t`).
// FORBIDDEN: vfmaq_f32, vfmsq_f32, vaddvq_f32.
// ---------------------------------------------------------------------------

/// NEON 4-lane f32 SIMD wrapper (ADR-0175).
#[derive(Clone, Copy)]
pub(crate) struct F32x4Neon(float32x4_t);

impl SimdF32x4 for F32x4Neon {
    #[inline]
    fn splat(x: f32) -> Self {
        // Safety: vdupq_n_f32 has no preconditions on x.
        unsafe { F32x4Neon(vdupq_n_f32(x)) }
    }

    #[inline]
    fn load_unaligned(src: &[f32; 4]) -> Self {
        // Safety: src is &[f32; 4] — non-null, 4*4 bytes valid.
        unsafe { F32x4Neon(vld1q_f32(src.as_ptr())) }
    }

    #[inline]
    fn store_unaligned(self, dst: &mut [f32; 4]) {
        // Safety: dst is &mut [f32; 4] — non-null, 4*4 bytes writable.
        unsafe { vst1q_f32(dst.as_mut_ptr(), self.0) }
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // Safety: both operands are valid float32x4_t values from prior ops.
        unsafe { F32x4Neon(vaddq_f32(self.0, rhs.0)) }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        // Safety: both operands are valid float32x4_t values from prior ops.
        unsafe { F32x4Neon(vsubq_f32(self.0, rhs.0)) }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Safety: both operands are valid float32x4_t values from prior ops.
        unsafe { F32x4Neon(vmulq_f32(self.0, rhs.0)) }
    }

    /// Deterministic horizontal sum: `((l0 + l1) + l2) + l3`.
    ///
    /// Uses store-then-scalar to avoid `vaddvq_f32` reordering (ADR-0175).
    #[inline]
    fn horizontal_sum(self) -> f32 {
        let mut tmp = [0.0_f32; 4];
        // Safety: tmp is local, properly sized; self.0 is a valid float32x4_t.
        unsafe { vst1q_f32(tmp.as_mut_ptr(), self.0) };
        ((tmp[0] + tmp[1]) + tmp[2]) + tmp[3]
    }
}

// ---------------------------------------------------------------------------
// Wave B3: G⁴ 5-point stencil — 4-node-parallel NEON kernel.
// ---------------------------------------------------------------------------

/// Apply the G⁴ 5-point stencil to 4 consecutive **interior** nodes simultaneously.
///
/// NEON processes 2 lanes per register; two `float64x2_t` pairs cover the
/// 4-node batch.  Arithmetic order mirrors `apply_g4_stencil_avx2_4nodes`
/// and the scalar reference.  FMA FORBIDDEN (`vfmaq_f64` not used).
///
/// # Safety
/// Requires NEON (aarch64 baseline).  Caller MUST ensure `base >= 2` and
/// `base + 6 <= n` (interior invariant).  Slice bounds guaranteed by caller.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub(crate) fn apply_g4_stencil_neon_4nodes(
    base: usize,
    prev: &[f64],
    ar3h: &[f64],
    ar1h: &[f64],
    al1h: &[f64],
    al3h: &[f64],
    dx_sq_inv: f64,
    out: &mut [f64],
) {
    // Safety: slices have len >= base+5 (prev) or base+4 (caches), per caller.
    // NEON available as aarch64 baseline; module-level cfg enforces this.
    unsafe {
        // ld2: load 2-lane f64 register from slice at offset.
        macro_rules! ld2 {
            ($sl:expr, $off:expr) => {
                vld1q_f64($sl.as_ptr().add($off))
            };
        }
        // lo = lanes 0-1 (base, base+1); hi = lanes 2-3 (base+2, base+3).
        let (lm2_lo, lm2_hi) = (ld2!(prev, base - 2), ld2!(prev, base));
        let (lm1_lo, lm1_hi) = (ld2!(prev, base - 1), ld2!(prev, base + 1));
        let (ctr_lo, ctr_hi) = (ld2!(prev, base), ld2!(prev, base + 2));
        let (rp1_lo, rp1_hi) = (ld2!(prev, base + 1), ld2!(prev, base + 3));
        let (rp2_lo, rp2_hi) = (ld2!(prev, base + 2), ld2!(prev, base + 4));
        let (ar3h_lo, ar3h_hi) = (ld2!(ar3h, base), ld2!(ar3h, base + 2));
        let (ar1h_lo, ar1h_hi) = (ld2!(ar1h, base), ld2!(ar1h, base + 2));
        let (al1h_lo, al1h_hi) = (ld2!(al1h, base), ld2!(al1h, base + 2));
        let (al3h_lo, al3h_hi) = (ld2!(al3h, base), ld2!(al3h, base + 2));

        let res_lo = neon_g4_lane_pair(
            lm2_lo, lm1_lo, ctr_lo, rp1_lo, rp2_lo, ar3h_lo, ar1h_lo, al1h_lo, al3h_lo, dx_sq_inv,
        );
        let res_hi = neon_g4_lane_pair(
            lm2_hi, lm1_hi, ctr_hi, rp1_hi, rp2_hi, ar3h_hi, ar1h_hi, al1h_hi, al3h_hi, dx_sq_inv,
        );
        vst1q_f64(out.as_mut_ptr().add(base), res_lo);
        vst1q_f64(out.as_mut_ptr().add(base + 2), res_hi);
    }
}

/// NEON G⁴ flux kernel for 2 consecutive nodes (one `float64x2_t` lane pair).
///
/// Arithmetic order matches the scalar reference; FMA FORBIDDEN (ADR-0019).
///
/// # Safety
/// NEON is aarch64 baseline; all `float64x2_t` args must be valid lane vectors.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
#[inline]
unsafe fn neon_g4_lane_pair(
    lm2: float64x2_t,
    lm1: float64x2_t,
    ctr: float64x2_t,
    rp1: float64x2_t,
    rp2: float64x2_t,
    ar3h: float64x2_t,
    ar1h: float64x2_t,
    al1h: float64x2_t,
    al3h: float64x2_t,
    dx_sq_inv: f64,
) -> float64x2_t {
    let five = vdupq_n_f64(5.0);
    let neg_one = vdupq_n_f64(-1.0);
    let neg_five = vdupq_n_f64(-5.0);
    let inv4 = vdupq_n_f64(0.25);
    let inv12 = vdupq_n_f64(1.0 / 12.0);
    let dxsq = vdupq_n_f64(dx_sq_inv);

    // flux_right = 5 * ar1h * (rp1 − ctr) / 4
    let flux_right = vmulq_f64(vmulq_f64(vmulq_f64(five, ar1h), vsubq_f64(rp1, ctr)), inv4);
    // flux_right_far = −1 * ar3h * (rp2 − rp1) / 12
    let flux_right_far = vmulq_f64(
        vmulq_f64(vmulq_f64(neg_one, ar3h), vsubq_f64(rp2, rp1)),
        inv12,
    );
    // flux_left = −5 * al1h * (ctr − lm1) / 4
    let flux_left = vmulq_f64(
        vmulq_f64(vmulq_f64(neg_five, al1h), vsubq_f64(ctr, lm1)),
        inv4,
    );
    // flux_left_far = al3h * (lm1 − lm2) / 12
    let flux_left_far = vmulq_f64(vmulq_f64(al3h, vsubq_f64(lm1, lm2)), inv12);
    // sum / dx_sq  [order: ((rfar + r) + l) + lfar]
    let s = vaddq_f64(
        vaddq_f64(vaddq_f64(flux_right_far, flux_right), flux_left),
        flux_left_far,
    );
    vmulq_f64(s, dxsq)
}

// ---------------------------------------------------------------------------
// Internal golden-vector unit test (build-time sanity).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neon_golden_vector() {
        // splat(2.0).mul(load([1,2,3,4])).horizontal_sum() == 2*(1+2+3+4) == 20
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let v = F64x4Neon::load_unaligned(&data);
        let two = F64x4Neon::splat(2.0);
        let result = two.mul(v).horizontal_sum();
        assert_eq!(result, 20.0_f64, "NEON golden vector failed: got {result}");
    }

    #[test]
    fn neon_f32_golden_vector() {
        // splat(2.0).mul(load([1,2,3,4])).horizontal_sum() == 2*(1+2+3+4) == 20
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let v = F32x4Neon::load_unaligned(&data);
        let two = F32x4Neon::splat(2.0_f32);
        let result = two.mul(v).horizontal_sum();
        #[allow(clippy::float_cmp)]
        let exact_20 = 20.0_f32;
        assert!(
            result.to_bits() == exact_20.to_bits(),
            "NEON f32 golden vector failed: got {result}"
        );
    }
}
