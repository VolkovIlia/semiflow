//! Interpolation helpers for `carnot_stepk.rs` (filiform N=5 step-4 Carnot).
//!
//! Extracted to keep `carnot_stepk.rs` ≤ 500 `LoC` (suckless default).
//!
//! Contents:
//! - Clamped linear interp index helper (mirrors `hormander_engel_helpers.rs`)
//! - Clamped multilinear 5D interpolation for the X₂ flow
//!
//! The GH32 nodes/weights are reused from `hormander_engel_helpers`.

// Grid/index/count values (usize) cast to f64 for coordinates; clamped float idx→usize
// is provably non-negative by bounds check at lines 28-34 (x in [lo, hi] ensures idx_f >= 0).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use crate::{grid::Grid1D, grid_nd::GridFnND};

/// Clamped linear interpolation index on a uniform 1D f64 grid.
///
/// Returns `(k0, k1, alpha)` such that:
/// `f(x) ≈ (1-alpha)·f[k0] + alpha·f[k1]`, with `0 ≤ alpha ≤ 1`.
///
/// Clamps `x < lo` to `(0, 0, 0.0)` and `x > hi` to `(n-1, n-1, 0.0)`.
#[inline]
pub(crate) fn lerp_idx_1d_f64(x: f64, grid: &Grid1D) -> (usize, usize, f64) {
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

/// Sample src at arbitrary 5D point via clamped multilinear interpolation.
///
/// Uses multilinear (32-corner, tensor-product) interpolation. Out-of-range
/// coordinates are clamped to the boundary (appropriate for Gaussian ICs that
/// decay near zero at domain edges).
///
/// Argument order: (x₁, x₂, x₃, x₄, x₅) matching filiform N=5 coordinates.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_5d(
    src: &GridFnND<f64, 5>,
    x0: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    x4: f64,
) -> f64 {
    let (k0, k0h, a0) = lerp_idx_1d_f64(x0, &src.grid.axes[0]);
    let (k1, k1h, a1) = lerp_idx_1d_f64(x1, &src.grid.axes[1]);
    let (k2, k2h, a2) = lerp_idx_1d_f64(x2, &src.grid.axes[2]);
    let (k3, k3h, a3) = lerp_idx_1d_f64(x3, &src.grid.axes[3]);
    let (k4, k4h, a4) = lerp_idx_1d_f64(x4, &src.grid.axes[4]);
    let b0 = 1.0 - a0;
    let b1 = 1.0 - a1;
    let b2 = 1.0 - a2;
    let b3 = 1.0 - a3;
    let b4 = 1.0 - a4;
    let v = |i0, i1, i2, i3, i4| src.values[src.grid.flat_idx(&[i0, i1, i2, i3, i4])];
    // 32-corner tensor-product multilinear interpolation.
    // Factored as nested 2-point blends over each dimension.
    let s3 = |i0: usize, i1: usize, i2: usize, i3: usize| {
        b3 * (b2
            * (b1 * (b0 * v(i0, i1, i2, i3, k4) + a0 * v(k0h, i1, i2, i3, k4))
                + a1 * (b0 * v(i0, k1h, i2, i3, k4) + a0 * v(k0h, k1h, i2, i3, k4)))
            + a2 * (b1 * (b0 * v(i0, i1, k2h, i3, k4) + a0 * v(k0h, i1, k2h, i3, k4))
                + a1 * (b0 * v(i0, k1h, k2h, i3, k4) + a0 * v(k0h, k1h, k2h, i3, k4))))
            + a3 * (b2
                * (b1 * (b0 * v(i0, i1, i2, k3h, k4) + a0 * v(k0h, i1, i2, k3h, k4))
                    + a1 * (b0 * v(i0, k1h, i2, k3h, k4) + a0 * v(k0h, k1h, i2, k3h, k4)))
                + a2 * (b1 * (b0 * v(i0, i1, k2h, k3h, k4) + a0 * v(k0h, i1, k2h, k3h, k4))
                    + a1 * (b0 * v(i0, k1h, k2h, k3h, k4) + a0 * v(k0h, k1h, k2h, k3h, k4))))
    };
    // Blend over dimension 4 (x₅) between k4 slice and k4h slice.
    // Reuse s3 evaluated at k4h by shifting the k4 → k4h in the inner closure.
    let s3h = |i0: usize, i1: usize, i2: usize, i3: usize| {
        b3 * (b2
            * (b1 * (b0 * v(i0, i1, i2, i3, k4h) + a0 * v(k0h, i1, i2, i3, k4h))
                + a1 * (b0 * v(i0, k1h, i2, i3, k4h) + a0 * v(k0h, k1h, i2, i3, k4h)))
            + a2 * (b1 * (b0 * v(i0, i1, k2h, i3, k4h) + a0 * v(k0h, i1, k2h, i3, k4h))
                + a1 * (b0 * v(i0, k1h, k2h, i3, k4h) + a0 * v(k0h, k1h, k2h, i3, k4h))))
            + a3 * (b2
                * (b1 * (b0 * v(i0, i1, i2, k3h, k4h) + a0 * v(k0h, i1, i2, k3h, k4h))
                    + a1 * (b0 * v(i0, k1h, i2, k3h, k4h) + a0 * v(k0h, k1h, i2, k3h, k4h)))
                + a2 * (b1 * (b0 * v(i0, i1, k2h, k3h, k4h) + a0 * v(k0h, i1, k2h, k3h, k4h))
                    + a1 * (b0 * v(i0, k1h, k2h, k3h, k4h) + a0 * v(k0h, k1h, k2h, k3h, k4h))))
    };
    b4 * s3(k0, k1, k2, k3) + a4 * s3h(k0, k1, k2, k3)
}
