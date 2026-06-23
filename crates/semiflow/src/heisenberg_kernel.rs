//! Heisenberg group H₁ heat kernel oracle via Gaveau-Hulanicki integral form.
//!
//! Per math.md §28 AMENDMENT 2 (corrected from AMENDMENT 1, 4 transcription
//! bugs fixed; see ADR-0087 AMENDMENT 1). Implementation mirrors
//! `scripts/verify_hormander_heisenberg.py::_gh_integrand` line-for-line
//! per spec AC4 cross-validation requirement.
//!
//! Formula (math.md §28 AMENDMENT 2):
//!
//! ```text
//! p_h(x, y, t) = (1/(2π)²) · ∫_{-Λ}^{+Λ}
//!   (λ/sinh(λh/2)) · exp(−(λ/4)·coth(λh/2)·(x²+y²)) · cos(λt) dλ
//! ```
//!
//! where Λ = 16/h. Quadrature: 32-pt Gauss-Legendre on `[-1, 1]` mapped to
//! `[-Λ, Λ]` (canonical nodes/weights from Abramowitz & Stegun Table 25.4).
//!
//! λ=0 removable singularity by L'Hôpital: `(2/h) · exp(−r²/(2h))`.
//!
//! This file is `no_std + libm`-compatible: only `sinh`, `cosh`, `exp`,
//! `cos` consumed via `SemiflowFloat` trait methods (ADR-0025).
//!
//! # References
//!
//! - Beals-Greiner 1988 *Calculus on Heisenberg Manifolds* Theorem 5.18
//! - KTX 2005 arxiv math/0401243 §2
//! - Hulanicki 1976 *Studia Math.* 56:165-173
//! - math.md §28 AMENDMENT 2

use crate::float::{from_f64, SemiflowFloat};

// ─── 32-pt Gauss-Legendre nodes/weights on [-1, 1] ───────────────────────────
//
// Canonical table from Abramowitz & Stegun Table 25.4.
// These are Gauss-LEGENDRE on [-1,1], NOT the Gauss-Laguerre nodes in
// resolvent_quad.rs (which are for ∫₀^∞ e^{-s} h(s) ds — different form).

/// 32-pt Gauss-Legendre nodes on `[-1, 1]` (symmetric pairs, A&S Table 25.4).
pub(crate) const GL32_LEGENDRE_NODES: [f64; 32] = [
    -9.972_638_618_494_816e-01,
    -9.856_115_115_452_684e-01,
    -9.647_622_555_875_064e-01,
    -9.349_060_759_377_397e-01,
    -8.963_211_557_660_521e-01,
    -8.493_676_137_325_7e-1,
    -7.944_837_959_679_424e-01,
    -7.321_821_187_402_897e-01,
    -6.630_442_669_302_152e-01,
    -5.877_157_572_407_623e-01,
    -5.068_999_089_322_294e-01,
    -4.213_512_761_306_353e-01,
    -3.318_686_022_821_276e-01,
    -2.392_873_622_521_371e-01,
    -1.444_719_615_827_965e-01,
    -4.830_766_568_773_832e-02,
    4.830_766_568_773_832e-02,
    1.444_719_615_827_965e-01,
    2.392_873_622_521_371e-01,
    3.318_686_022_821_276e-01,
    4.213_512_761_306_353e-01,
    5.068_999_089_322_294e-01,
    5.877_157_572_407_623e-01,
    6.630_442_669_302_152e-01,
    7.321_821_187_402_897e-01,
    7.944_837_959_679_424e-01,
    8.493_676_137_325_7e-1,
    8.963_211_557_660_521e-01,
    9.349_060_759_377_397e-01,
    9.647_622_555_875_064e-01,
    9.856_115_115_452_684e-01,
    9.972_638_618_494_816e-01,
];

/// 32-pt Gauss-Legendre weights for `[-1, 1]` (A&S Table 25.4).
pub(crate) const GL32_LEGENDRE_WEIGHTS: [f64; 32] = [
    7.018_610_009_470_506e-03,
    1.627_439_473_090_574e-02,
    2.539_206_530_926_202e-02,
    3.427_386_291_302_177e-02,
    4.283_589_802_222_684e-02,
    5.099_805_926_237_61e-2,
    5.868_409_347_853_557e-02,
    6.582_222_277_636_168e-02,
    7.234_579_410_884_834e-02,
    7.819_389_578_707_023e-02,
    8.331_192_422_694_671e-02,
    8.765_209_300_440_379e-02,
    9.117_387_869_576_378e-02,
    9.384_439_908_080_451e-02,
    9.563_872_007_927_471e-02,
    9.654_008_851_472_766e-02,
    9.654_008_851_472_766e-02,
    9.563_872_007_927_471e-02,
    9.384_439_908_080_451e-02,
    9.117_387_869_576_378e-02,
    8.765_209_300_440_379e-02,
    8.331_192_422_694_671e-02,
    7.819_389_578_707_023e-02,
    7.234_579_410_884_834e-02,
    6.582_222_277_636_168e-02,
    5.868_409_347_853_557e-02,
    5.099_805_926_237_61e-2,
    4.283_589_802_222_684e-02,
    3.427_386_291_302_177e-02,
    2.539_206_530_926_202e-02,
    1.627_439_473_090_574e-02,
    7.018_610_009_470_506e-03,
];

// ─── Gaveau-Hulanicki integrand (mirrors _gh_integrand in the Python oracle) ──

/// Single integrand evaluation at `λ`, mirroring `_gh_integrand` in
/// `scripts/verify_hormander_heisenberg.py` byte-for-byte.
///
/// Formula: `(λ/sinh(λh/2)) · exp(−(λ/4)·coth(λh/2)·r²) · cos(λt)`.
/// λ=0 limit: `(2/h) · exp(−r²/(2h))`.
// cosh_lhh and coth_lhh are mathematically related hyperbolic functions; names are intentional.
#[allow(clippy::similar_names)]
#[inline]
fn heisenberg_integrand<F: SemiflowFloat>(lam: F, h: F, r2: F, tc: F) -> F {
    let tiny = from_f64::<F>(1e-10_f64);
    if lam.abs() < tiny {
        // L'Hôpital limit: (2/h) · exp(-r²/(2h))
        let two = from_f64::<F>(2.0_f64);
        let exp_arg = -(r2 / (two * h));
        return (two / h) * exp_arg.exp();
    }

    let half = from_f64::<F>(0.5_f64);
    let lam_h_half = lam * h * half; // λh/2
    let sinh_lhh = lam_h_half.sinh();

    // Guard against numerical underflow near sinh=0 (defensive; only triggers
    // at very large |λ| where the exponential decays the integrand to zero).
    let tiny2 = from_f64::<F>(1e-10_f64);
    if sinh_lhh.abs() < tiny2 {
        return F::zero();
    }

    let cosh_lhh = lam_h_half.cosh();
    let coth_lhh = cosh_lhh / sinh_lhh; // coth(λh/2)
    let four = from_f64::<F>(4.0_f64);
    let exp_arg = -(lam / four) * coth_lhh * r2;
    let cos_arg = lam * tc;

    (lam / sinh_lhh) * exp_arg.exp() * cos_arg.cos()
}

// ─── Public surface ───────────────────────────────────────────────────────────

/// Evaluate the Gaveau-Hulanicki heat kernel `p_h(x, y, t)` for the
/// Heisenberg sub-Laplacian `L = ½(X₁² + X₂²)` on `ℍ¹`.
///
/// Uses 32-pt Gauss-Legendre on `[-16/h, +16/h]` (production oracle for the
/// `G_HORM_HEISENBERG` slope gate). For higher-precision reference, use
/// `scripts/verify_hormander_heisenberg.py::_gh_kernel_mp` (mpmath, ~1000×
/// slower; used only in the sympy `T_HORM_HEISENBERG` sub-checks).
///
/// # Arguments
/// - `h`: evolution time (must be positive; returns `F::zero()` for `h ≤ 0`).
/// - `x`, `y`: horizontal Heisenberg coordinates.
/// - `t`: central coordinate (third dimension, direction of `[X₁, X₂]`).
///
/// # References
/// - Beals-Greiner 1988 Theorem 5.18; math.md §28 AMENDMENT 2.
pub fn heisenberg_heat_kernel<F: SemiflowFloat>(h: F, x: F, y: F, tc: F) -> F {
    if h <= F::zero() {
        return F::zero();
    }

    let r2 = x * x + y * y;
    let sixteen = from_f64::<F>(16.0_f64);
    let lam_max = sixteen / h;
    let four_pi_sq_inv = {
        let pi = from_f64::<F>(core::f64::consts::PI);
        let four = from_f64::<F>(4.0_f64);
        F::one() / (four * pi * pi)
    };

    let mut total = F::zero();
    for (&node, &weight) in GL32_LEGENDRE_NODES.iter().zip(GL32_LEGENDRE_WEIGHTS.iter()) {
        let node_f = from_f64::<F>(node);
        let weight_f = from_f64::<F>(weight);
        let lam = lam_max * node_f; // map [-1,1] → [-Λ, Λ]
        total += weight_f * heisenberg_integrand(lam, h, r2, tc);
    }
    // Jacobian: dλ = Λ dξ
    four_pi_sq_inv * (lam_max * total)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    /// Cross-validate 32-pt GL Rust impl against the INDEPENDENT Python
    /// `_gh_kernel` reference at 6 probe points.
    ///
    /// **Reference values are the Python oracle output** (independent
    /// implementation), NOT the Rust kernel's own output. They were generated
    /// via:
    ///   ```text
    ///   python3 -c "
    ///   from scripts.verify_hormander_heisenberg import _gh_kernel
    ///   cases = [(0.5,0.1,0.0,0.0),(0.5,0.0,0.1,0.0),(0.5,0.0,0.0,0.1),
    ///            (1.0,0.0,0.0,0.0),(0.5,0.2,0.2,0.0),(0.3,0.0,0.0,0.05)]
    ///   for h,x,y,tc in cases: print(f'{_gh_kernel(h,x,y,tc):.18e}')
    ///   "
    ///   ```
    ///
    /// Both implementations use the identical 32-pt GL formula on `[-16/h, +16/h]`
    /// but differ in summation/evaluation order (Rust: libm, Python: `math`),
    /// producing genuine floating-point noise of ≤1.124e-9 — the measured max
    /// diff across all 6 cases on x86-64 release build. The fact that Rust and
    /// Python values differ by ~1e-9 (not zero) confirms they are independent.
    ///
    /// **Tolerance: 2e-9 absolute** (~1.8× above the measured max cross-impl diff
    /// of 1.124e-9). This test answers: "does the Rust GL32 kernel agree with an
    /// independent Python GL32 implementation to eval-order noise?" — yes, ~1e-9.
    ///
    /// Note: `_gh_kernel_mp` (384-pt mpmath on `[-50/h, +50/h]`) differs by ~4e-3
    /// due to a wider integration range; that gap is expected quadrature truncation
    /// and is independent of the cross-implementation agreement measured here.
    #[test]
    fn heisenberg_kernel_matches_sympy_oracle() {
        // Columns: (h, x, y, tc, python_expected)
        // python_expected = _gh_kernel(h,x,y,tc) from scripts/verify_hormander_heisenberg.py
        // These are INDEPENDENT Python oracle values (not the Rust kernel's own output).
        // Rust vs Python diffs are ~1e-9 (eval-order noise, NOT regression baseline).
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            (0.5, 0.1, 0.0, 0.0, 1.956_071_483_627_801_7),
            (0.5, 0.0, 0.1, 0.0, 1.956_071_483_627_801_7),
            (0.5, 0.0, 0.0, 0.1, 1.383_716_318_640_359_6),
            (1.0, 0.0, 0.0, 0.0, 4.987_763_792_291_724_5e-1),
            (0.5, 0.2, 0.2, 0.0, 1.709_440_273_079_713_6),
            (0.3, 0.0, 0.0, 0.05, 4.285_703_873_427_739),
        ];
        let mut max_diff: f64 = 0.0;
        for &(h, x, y, tc, expected) in cases {
            let actual = heisenberg_heat_kernel(h, x, y, tc);
            let diff = (actual - expected).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 2e-9,
                "h={h} x={x} y={y} t={tc}: Rust={actual:.18e} python_oracle={expected:.18e} \
                 diff={diff:.3e} (gate 2e-9, cross-impl eval-order noise expected ~1e-9)"
            );
        }
        // Verify the overall max cross-implementation diff stays well below gate.
        assert!(
            max_diff < 2e-9,
            "max Rust-vs-Python diff {max_diff:.3e} exceeds gate 2e-9 — \
             cross-impl GL32 summation-order noise"
        );
    }

    /// Verify the on-diagonal analytic identity: `p_h(0,0,0)` = 1/(2h²).
    ///
    /// The 32-pt GL on `[-16/h, +16/h]` misses the tails ∫_{|λ|>16/h} by ~0.24%
    /// (confirmed against Python `_gh_kernel`). Gate 5e-3 accommodates this
    /// quadrature truncation; the slope test uses the oracle consistently so
    /// the truncation is a systematic bias and does not affect slope measurement.
    #[test]
    fn heisenberg_kernel_on_diagonal_identity() {
        for h in [0.25_f64, 0.5, 1.0, 2.0] {
            let expected = 1.0 / (2.0 * h * h);
            let actual = heisenberg_heat_kernel(h, 0.0_f64, 0.0_f64, 0.0_f64);
            let rel = (actual - expected).abs() / expected;
            assert!(
                rel < 5e-3,
                "h={h}: p_h(0,0,0)={actual:.6e} expected={expected:.6e} rel={rel:.3e}"
            );
        }
    }

    /// Verify kernel returns zero for non-positive h (defensive guard).
    #[test]
    fn heisenberg_kernel_nonpositive_h() {
        assert_eq!(heisenberg_heat_kernel(0.0_f64, 0.5, 0.5, 0.5), 0.0);
        assert_eq!(heisenberg_heat_kernel(-1.0_f64, 0.5, 0.5, 0.5), 0.0);
    }
}
