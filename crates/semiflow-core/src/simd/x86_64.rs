//! AVX2 (4-lane f64 = `__m256d`) impl of [`super::SimdF64x4`] (ADR-0019).
//!
//! Compiled only on `x86_64` targets; requires AVX2 available at link time
//! (CI sets `RUSTFLAGS="-C target-feature=+avx2"` per Cargo.toml contract).
//!
//! # Safety
//! All unsafe blocks in this file call AVX2 intrinsics that require:
//! - Target CPU supports AVX2 (guaranteed by build-time `-C target-feature=+avx2`).
//! - All pointer operands point to valid, non-null memory (guaranteed by
//!   `&[f64; 4]` / `&mut [f64; 4]` reference invariants).
//!
//! FORBIDDEN: `_mm256_fmadd_pd`, `_mm256_fnmadd_pd`, `_mm256_dp_pd`,
//! `_mm256_hadd_pd` — break the bit-equal contract (ADR-0019).

#![cfg(all(target_arch = "x86_64", target_feature = "avx2"))]

use core::arch::x86_64::{
    __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    _mm256_sub_pd,
};

use super::SimdF64x4;

/// AVX2 4-lane f64 SIMD wrapper.
#[derive(Clone, Copy)]
pub(crate) struct F64x4Avx2(__m256d);

#[allow(dead_code)] // trait surface — all ops required by contract; not all used in v0.8.0 hot paths
impl SimdF64x4 for F64x4Avx2 {
    #[inline]
    fn splat(x: f64) -> Self {
        // Safety: _mm256_set1_pd has no preconditions.
        unsafe { F64x4Avx2(_mm256_set1_pd(x)) }
    }

    #[inline]
    fn load_unaligned(src: &[f64; 4]) -> Self {
        // Safety: src is a valid &[f64; 4] reference — non-null, 4*8 bytes valid.
        unsafe { F64x4Avx2(_mm256_loadu_pd(src.as_ptr())) }
    }

    #[inline]
    fn store_unaligned(self, dst: &mut [f64; 4]) {
        // Safety: dst is a valid &mut [f64; 4] — non-null, 4*8 bytes writable.
        unsafe { _mm256_storeu_pd(dst.as_mut_ptr(), self.0) }
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // Safety: both operands are valid __m256d values from prior ops.
        unsafe { F64x4Avx2(_mm256_add_pd(self.0, rhs.0)) }
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        // Safety: both operands are valid __m256d values from prior ops.
        unsafe { F64x4Avx2(_mm256_sub_pd(self.0, rhs.0)) }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Safety: both operands are valid __m256d values from prior ops.
        unsafe { F64x4Avx2(_mm256_mul_pd(self.0, rhs.0)) }
    }

    /// Deterministic horizontal sum: `((l0 + l1) + l2) + l3`.
    ///
    /// Intentionally uses tmp[] extract rather than `_mm256_hadd_pd`
    /// or `_mm256_dp_pd` — those change the addition order vs scalar.
    #[inline]
    fn horizontal_sum(self) -> f64 {
        let mut tmp = [0.0_f64; 4];
        // Safety: tmp is local, properly sized; self.0 is a valid __m256d.
        unsafe { _mm256_storeu_pd(tmp.as_mut_ptr(), self.0) };
        ((tmp[0] + tmp[1]) + tmp[2]) + tmp[3]
    }
}

// ---------------------------------------------------------------------------
// Wave B3: G⁴ 5-point stencil — 4-node-parallel AVX2 kernel.
// ---------------------------------------------------------------------------

/// Apply the G⁴ 5-point stencil to 4 consecutive **interior** nodes simultaneously.
///
/// Processes nodes `[base, base+1, base+2, base+3]` in a single AVX2 pass.
/// Caller MUST ensure `base >= 2` and `base + 6 <= n` (interior invariant).
///
/// # Arithmetic order
///
/// Each lane computes in the same left-to-right order as the scalar path
/// (`apply_g4_at_node_cached`).  FMA is FORBIDDEN (ADR-0019 §`determinism_contract`):
/// only `_mm256_mul_pd` + `_mm256_add_pd` / `_mm256_sub_pd` are used.
///
/// # Safety
/// Requires AVX2.  `prev`, `ar3h`, `ar1h`, `al1h`, `al3h` slices must have
/// `len >= base + 4` (checked by caller).  `out` must have `len >= base + 4`.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub(crate) fn apply_g4_stencil_avx2_4nodes(
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
    // AVX2 available by module-level cfg(target_feature = "avx2").
    unsafe {
        let lm2 = _mm256_loadu_pd(prev.as_ptr().add(base - 2));
        let lm1 = _mm256_loadu_pd(prev.as_ptr().add(base - 1));
        let ctr = _mm256_loadu_pd(prev.as_ptr().add(base));
        let rp1 = _mm256_loadu_pd(prev.as_ptr().add(base + 1));
        let rp2 = _mm256_loadu_pd(prev.as_ptr().add(base + 2));
        let v_ar3h = _mm256_loadu_pd(ar3h.as_ptr().add(base));
        let v_ar1h = _mm256_loadu_pd(ar1h.as_ptr().add(base));
        let v_al1h = _mm256_loadu_pd(al1h.as_ptr().add(base));
        let v_al3h = _mm256_loadu_pd(al3h.as_ptr().add(base));
        let result = avx2_g4_fluxes(
            lm2, lm1, ctr, rp1, rp2, v_ar3h, v_ar1h, v_al1h, v_al3h, dx_sq_inv,
        );
        _mm256_storeu_pd(out.as_mut_ptr().add(base), result);
    }
}

/// Compute 4-node G⁴ stencil result via AVX2 (no FMA — ADR-0019 determinism contract).
///
/// # Safety
/// AVX2 must be available (guaranteed by module-level cfg).  All `__m256d` args
/// must be valid (caller loads them from valid memory).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
#[inline]
unsafe fn avx2_g4_fluxes(
    lm2: __m256d,
    lm1: __m256d,
    ctr: __m256d,
    rp1: __m256d,
    rp2: __m256d,
    v_ar3h: __m256d,
    v_ar1h: __m256d,
    v_al1h: __m256d,
    v_al3h: __m256d,
    dx_sq_inv: f64,
) -> __m256d {
    let five = _mm256_set1_pd(5.0);
    let neg_one = _mm256_set1_pd(-1.0);
    let neg_five = _mm256_set1_pd(-5.0);
    let one_over_4 = _mm256_set1_pd(0.25);
    let one_over_12 = _mm256_set1_pd(1.0 / 12.0);
    let dx_sq_inv_v = _mm256_set1_pd(dx_sq_inv);

    // flux_right  = 5.0 * ar1h * (rp1 − ctr) * 0.25   [scalar order: no FMA]
    let flux_right = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_mul_pd(five, v_ar1h), _mm256_sub_pd(rp1, ctr)),
        one_over_4,
    );
    // flux_right_far = −1 * ar3h * (rp2 − rp1) * (1/12)
    let flux_right_far = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_mul_pd(neg_one, v_ar3h), _mm256_sub_pd(rp2, rp1)),
        one_over_12,
    );
    // flux_left   = −5 * al1h * (ctr − lm1) * 0.25
    let flux_left = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_mul_pd(neg_five, v_al1h), _mm256_sub_pd(ctr, lm1)),
        one_over_4,
    );
    // flux_left_far = al3h * (lm1 − lm2) * (1/12)
    let flux_left_far = _mm256_mul_pd(_mm256_mul_pd(v_al3h, _mm256_sub_pd(lm1, lm2)), one_over_12);
    // sum / dx_sq  [scalar order: ((rfar+r)+l)+lfar]
    let sum = _mm256_add_pd(
        _mm256_add_pd(_mm256_add_pd(flux_right_far, flux_right), flux_left),
        flux_left_far,
    );
    _mm256_mul_pd(sum, dx_sq_inv_v)
}

// ---------------------------------------------------------------------------
// Internal golden-vector unit test (build-time sanity).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_golden_vector() {
        // splat(2.0).mul(load([1,2,3,4])).horizontal_sum() == 2*(1+2+3+4) == 20
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let v = F64x4Avx2::load_unaligned(&data);
        let two = F64x4Avx2::splat(2.0);
        let result = two.mul(v).horizontal_sum();
        // These are exact integers in IEEE-754; bit-equality is guaranteed.
        #[allow(clippy::float_cmp)]
        let exact_20 = 20.0_f64;
        assert!(
            result.to_bits() == exact_20.to_bits(),
            "AVX2 golden vector failed: got {result}"
        );
    }
}
