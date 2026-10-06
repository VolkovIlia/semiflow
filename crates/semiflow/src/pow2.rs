//! Exact power-of-two prescaling helpers (ADR-0202, §62.2).

use crate::float::SemiflowFloat;

/// `2^k` as `f64` for `|k| ≤ 2044`, as a product of two exact normal powers of two.
fn pow2_f64(k: i32) -> f64 {
    let half = k / 2;
    let one = |j: i32| f64::from_bits(u64::try_from(1023 + j).unwrap_or(0) << 52);
    one(half) * one(k - half)
}

/// `(2^{-e}, 2^{e})` with `e = ⌊log₂ ‖b‖∞⌋`, so `‖b·2^{-e}‖∞ ∈ [1, 2)`; `None` if `b = 0`.
///
/// Scaling by a power of two is exact in IEEE arithmetic except that entries more than
/// 1022 binades below the maximum land in the subnormal range and lose bits. The solve
/// and the residual then run on an O(1) right-hand side and cannot underflow or
/// overflow in the squared norms. If the exact factors are not representable in `F`
/// (subnormal `b`), the exponent is clamped to the largest representable one.
pub(crate) fn pow2_scale<F: SemiflowFloat>(rhs: &[F]) -> Option<(F, F)> {
    let bmax = rhs.iter().fold(F::zero(), |m, &v| m.max(v.abs()));
    if bmax == F::zero() {
        return None;
    }
    let mut mag = bmax.to_f64().unwrap_or(1.0);
    let mut shift = 0;
    while mag < f64::MIN_POSITIVE {
        mag *= 18_014_398_509_481_984.0; // 2^54, exact
        shift += 54;
    }
    let exp = i32::try_from((mag.to_bits() >> 52) & 0x7ff).unwrap_or(1023) - 1023 - shift;
    // Largest |e| ≤ |exp| for which both factors are finite and nonzero in `F`. A
    // subnormal `b` (exp ≈ −1030 in f64) cannot use 2^1030; 2^1023 still lifts it to O(1)
    // enough that its squared norm cannot underflow.
    let mut e = exp;
    loop {
        if let Some(pair) = factors::<F>(e) {
            return Some(pair);
        }
        if e == 0 {
            return Some((F::one(), F::one()));
        }
        e -= e.signum();
    }
}

/// `(2^{-e}, 2^{e})` if both are finite and nonzero in `F`.
fn factors<F: SemiflowFloat>(e: i32) -> Option<(F, F)> {
    let down = F::from(pow2_f64(-e))?;
    let up = F::from(pow2_f64(e))?;
    (down.is_finite() && up.is_finite() && down != F::zero() && up != F::zero())
        .then_some((down, up))
}
