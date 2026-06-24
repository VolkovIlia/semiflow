//! Helper constants and interpolation utilities for the Engel step-3 Carnot
//! Chernoff implementation in `hormander_engel.rs` (ADR-0095, math.md §28.bis).
//!
//! Extracted to keep `hormander_engel.rs` ≤ 500 `LoC` (suckless default).
//!
//! Contents:
//! - 32-pt Gauss-Hermite nodes and weights (A&S Table 25.10, same as Heisenberg)
//! - Clamped multilinear 4D interpolation for the X₂ flow
//! - Clamped linear 1D interpolation for the X₁ flow

// Clamped idx_f → usize: x ∈ [lo,hi] ensures idx_f ∈ [0, n-1], so cast is safe.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use crate::{grid::Grid1D, grid_nd::GridFnND};

// ─── Gauss-Hermite 32-pt constants ───────────────────────────────────────────

/// 32-pt Gauss-Hermite nodes on `(-∞, +∞)` with weight `exp(-ξ²)`.
///
/// Source: Abramowitz & Stegun Table 25.10.
/// Same constants as used in `hormander_heisenberg.rs`.
pub(crate) const GH32_NODES_ENGEL: [f64; 32] = [
    -7.125_813_909_804_3,
    -6.409_498_149_087_0,
    -5.812_069_112_532_0,
    -5.275_747_992_688_2,
    -4.776_931_005_826_0,
    -4.304_685_659_042_5,
    -3.851_031_808_504_7,
    -3.410_489_170_120_0,
    -2.979_413_031_329_4,
    -2.555_372_589_851_4,
    -2.136_440_818_498_6,
    -1.721_269_808_059_8,
    -1.308_864_282_863_9,
    -0.898_547_906_488_8,
    -0.489_831_413_702_4,
    -0.082_295_724_565_3,
    0.082_295_724_565_3,
    0.489_831_413_702_4,
    0.898_547_906_488_8,
    1.308_864_282_863_9,
    1.721_269_808_059_8,
    2.136_440_818_498_6,
    2.555_372_589_851_4,
    2.979_413_031_329_4,
    3.410_489_170_120_0,
    3.851_031_808_504_7,
    4.304_685_659_042_5,
    4.776_931_005_826_0,
    5.275_747_992_688_2,
    5.812_069_112_532_0,
    6.409_498_149_087_0,
    7.125_813_909_804_3,
];

/// 32-pt Gauss-Hermite weights corresponding to `GH32_NODES_ENGEL`.
///
/// Source: Abramowitz & Stegun Table 25.10.
pub(crate) const GH32_WEIGHTS_ENGEL: [f64; 32] = [
    1.720_700_069_649_1e-23,
    3.802_932_352_530_9e-20,
    1.909_592_453_219_7e-17,
    4.148_048_493_747_0e-15,
    5.138_752_481_415_7e-13,
    3.965_938_919_406_9e-11,
    2.010_479_448_553_4e-09,
    6.897_296_393_888_1e-08,
    1.635_615_455_893_3e-06,
    2.731_827_892_040_9e-05,
    3.222_948_453_484_6e-04,
    2.690_407_218_800_3e-03,
    1.587_990_081_584_5e-02,
    6.624_469_254_241_8e-02,
    1.932_791_709_069_8e-01,
    3.946_862_759_041_5e-01,
    3.946_862_759_041_5e-01,
    1.932_791_709_069_8e-01,
    6.624_469_254_241_8e-02,
    1.587_990_081_584_5e-02,
    2.690_407_218_800_3e-03,
    3.222_948_453_484_6e-04,
    2.731_827_892_040_9e-05,
    1.635_615_455_893_3e-06,
    6.897_296_393_888_1e-08,
    2.010_479_448_553_4e-09,
    3.965_938_919_406_9e-11,
    5.138_752_481_415_7e-13,
    4.148_048_493_747_0e-15,
    1.909_592_453_219_7e-17,
    3.802_932_352_530_9e-20,
    1.720_700_069_649_1e-23,
];

// ─── Multilinear interpolation helpers ───────────────────────────────────────

/// Sample src at (`x0_src`, `i1_fixed`, `i2_fixed`, `i3_fixed`) via clamped linear
/// interp along axis 0 only (X₁ flow: axes 1,2,3 are index-exact).
#[inline]
pub(crate) fn sample_axis0(
    src: &GridFnND<f64, 4>,
    x0_src: f64,
    i1: usize,
    i2: usize,
    i3: usize,
) -> f64 {
    let (k0, k1, alpha) = lerp_idx_1d(x0_src, &src.grid.axes[0]);
    let f0 = src.values[src.grid.flat_idx(&[k0, i1, i2, i3])];
    let f1 = src.values[src.grid.flat_idx(&[k1, i1, i2, i3])];
    f0 * (1.0 - alpha) + f1 * alpha
}

/// Sample src at arbitrary 4D point via clamped multilinear interpolation.
///
/// Uses multilinear (16-corner tensor-product) interpolation. Out-of-range
/// coordinates are clamped to the boundary, which is appropriate for Gaussian
/// ICs that decay to near-zero at domain edges.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_4d(src: &GridFnND<f64, 4>, x0: f64, x1: f64, x2: f64, x3: f64) -> f64 {
    let (k0, k0h, a0) = lerp_idx_1d(x0, &src.grid.axes[0]);
    let (k1, k1h, a1) = lerp_idx_1d(x1, &src.grid.axes[1]);
    let (k2, k2h, a2) = lerp_idx_1d(x2, &src.grid.axes[2]);
    let (k3, k3h, a3) = lerp_idx_1d(x3, &src.grid.axes[3]);
    // 16-corner multilinear interpolation over 4D cell.
    let b0 = 1.0 - a0;
    let b1 = 1.0 - a1;
    let b2 = 1.0 - a2;
    let b3 = 1.0 - a3;
    let v = |i0, i1, i2, i3| src.values[src.grid.flat_idx(&[i0, i1, i2, i3])];
    b3 * (b2
        * (b1 * (b0 * v(k0, k1, k2, k3) + a0 * v(k0h, k1, k2, k3))
            + a1 * (b0 * v(k0, k1h, k2, k3) + a0 * v(k0h, k1h, k2, k3)))
        + a2 * (b1 * (b0 * v(k0, k1, k2h, k3) + a0 * v(k0h, k1, k2h, k3))
            + a1 * (b0 * v(k0, k1h, k2h, k3) + a0 * v(k0h, k1h, k2h, k3))))
        + a3 * (b2
            * (b1 * (b0 * v(k0, k1, k2, k3h) + a0 * v(k0h, k1, k2, k3h))
                + a1 * (b0 * v(k0, k1h, k2, k3h) + a0 * v(k0h, k1h, k2, k3h)))
            + a2 * (b1 * (b0 * v(k0, k1, k2h, k3h) + a0 * v(k0h, k1, k2h, k3h))
                + a1 * (b0 * v(k0, k1h, k2h, k3h) + a0 * v(k0h, k1h, k2h, k3h))))
}

/// Clamped linear interpolation index on a uniform 1D grid.
///
/// Returns `(k0, k1, alpha)` such that:
/// `f(x) ≈ (1-alpha)·f[k0] + alpha·f[k1]`, with `0 ≤ alpha ≤ 1`.
///
/// Clamps `x < lo` to `(0, 0, 0.0)` and `x > hi` to `(n-1, n-1, 0.0)`.
#[inline]
pub(crate) fn lerp_idx_1d(x: f64, grid: &Grid1D) -> (usize, usize, f64) {
    let n = grid.n;
    let lo = grid.x_at(0);
    let hi = grid.x_at(n - 1);
    if x <= lo {
        return (0, 0, 0.0);
    }
    if x >= hi {
        return (n - 1, n - 1, 0.0);
    }
    let dx = (hi - lo) / (n - 1) as f64;
    let idx_f = (x - lo) / dx;
    let k0 = (idx_f as usize).min(n - 2);
    let k1 = k0 + 1;
    let alpha = (idx_f - k0 as f64).clamp(0.0, 1.0);
    (k0, k1, alpha)
}
