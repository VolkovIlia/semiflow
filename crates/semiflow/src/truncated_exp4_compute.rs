//! Private generic and slice-based compute helpers for `TruncatedExp4thDiffusionChernoff`.
//!
//! Declared as `#[path = "truncated_exp4_compute.rs"] mod compute;` inside
//! `truncated_exp4.rs` — this file is a child of that module, so `super::` works.

use super::{TruncatedExp4thDiffusionChernoff, FACTORIAL_INVERSE, TRUNC_ORDER_USIZE};
use crate::{
    error::SemiflowError,
    float::{from_f64, half, SemiflowFloat},
    grid::Grid1D,
    grid_fn::GridFn1D,
};

// Private computation helpers — generic

#[inline]
pub(super) fn compute_x_mid_generic<F: SemiflowFloat>(
    mc: &TruncatedExp4thDiffusionChernoff<F>,
    x_i: F,
    tau: F,
) -> F {
    match mc.b_for_conjugation {
        Some(b) => x_i - half::<F>() * tau * b(x_i),
        None => x_i,
    }
}

pub(super) fn precompute_g4_grids_generic<F: SemiflowFloat>(
    mc: &TruncatedExp4thDiffusionChernoff<F>,
    f: &GridFn1D<F>,
) -> Result<[GridFn1D<F>; TRUNC_ORDER_USIZE + 1], SemiflowError> {
    let g0 = f.clone();
    let g1 = apply_g4_stencil_generic(mc, &g0)?;
    let g2 = apply_g4_stencil_generic(mc, &g1)?;
    let g3 = apply_g4_stencil_generic(mc, &g2)?;
    let g4 = apply_g4_stencil_generic(mc, &g3)?;
    Ok([g0, g1, g2, g3, g4])
}

pub(super) fn apply_g4_stencil_generic<F: SemiflowFloat>(
    mc: &TruncatedExp4thDiffusionChernoff<F>,
    prev: &GridFn1D<F>,
) -> Result<GridFn1D<F>, SemiflowError> {
    let n = prev.values.len();
    let dx = mc.grid.dx();
    let dx_sq = dx * dx;
    let mut out = prev.zeroed_like();

    for i in 0..n {
        let x_i = mc.grid.x_at(i);
        out.values[i] = apply_g4_at_node_generic(mc, prev, i, n, x_i, dx, dx_sq)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub(super) fn apply_g4_at_node_generic<F: SemiflowFloat>(
    mc: &TruncatedExp4thDiffusionChernoff<F>,
    prev: &GridFn1D<F>,
    i: usize,
    n: usize,
    x_i: F,
    dx: F,
    dx_sq: F,
) -> Result<F, SemiflowError> {
    let two = from_f64::<F>(2.0);
    let half_v = half::<F>();
    let three_halves = from_f64::<F>(1.5);
    let five = from_f64::<F>(5.0);
    let four = from_f64::<F>(4.0);
    let twelve = from_f64::<F>(12.0);

    let rp2 = if i + 2 < n {
        prev.values[i + 2]
    } else {
        prev.sample_generic(x_i + two * dx)?
    };
    let rp1 = if i + 1 < n {
        prev.values[i + 1]
    } else {
        prev.sample_generic(x_i + dx)?
    };
    let ctr = prev.values[i];
    let lm1 = if i >= 1 {
        prev.values[i - 1]
    } else {
        prev.sample_generic(x_i - dx)?
    };
    let lm2 = if i >= 2 {
        prev.values[i - 2]
    } else {
        prev.sample_generic(x_i - two * dx)?
    };

    let ar3h = (mc.a)(x_i + three_halves * dx);
    let ar1h = (mc.a)(x_i + half_v * dx);
    let al1h = (mc.a)(x_i - half_v * dx);
    let al3h = (mc.a)(x_i - three_halves * dx);

    let flux_right = five * ar1h * (rp1 - ctr) / four;
    let flux_right_far = -(ar3h * (rp2 - rp1)) / twelve;
    let flux_left = -(five * al1h * (ctr - lm1)) / four;
    let flux_left_far = al3h * (lm1 - lm2) / twelve;

    Ok((flux_right_far + flux_right + flux_left + flux_left_far) / dx_sq)
}

pub(super) fn apply_power_series_generic<F: SemiflowFloat>(
    tau: F,
    g_grids: &[GridFn1D<F>; TRUNC_ORDER_USIZE + 1],
    x_mid: F,
) -> Result<F, SemiflowError> {
    let mut sum = F::zero();
    let mut tau_pow = F::one();
    for k in 0..=TRUNC_ORDER_USIZE {
        let gk_val = g_grids[k].sample_generic(x_mid)?;
        let fk = from_f64::<F>(FACTORIAL_INVERSE[k]);
        sum += fk * tau_pow * gk_val;
        tau_pow *= tau;
    }
    Ok(sum)
}

// Slice-based helpers for apply_into (no GridFn1D allocation)

/// G⁴ stencil once into slice; boundary via `grid.interp` (no `GridFn1D` alloc).
///
/// `grid` MUST have the same `n` and `dx` as `prev.len()` (pass `src.grid`,
/// not `mc.grid`, when the two may differ across grid-size sweeps).
#[allow(clippy::similar_names)]
pub(super) fn apply_g4_stencil_into_slice(
    mc: &TruncatedExp4thDiffusionChernoff<f64>,
    grid: Grid1D<f64>,
    prev: &[f64],
    out: &mut [f64],
    n: usize,
    dx: f64,
) -> Result<(), SemiflowError> {
    let dx_sq = dx * dx;
    for i in 0..n {
        let x_i = grid.x_at(i);
        let rp2 = if i + 2 < n {
            prev[i + 2]
        } else {
            grid.interp(prev, x_i + 2.0 * dx)?
        };
        let rp1 = if i + 1 < n {
            prev[i + 1]
        } else {
            grid.interp(prev, x_i + dx)?
        };
        let ctr = prev[i];
        let lm1 = if i >= 1 {
            prev[i - 1]
        } else {
            grid.interp(prev, x_i - dx)?
        };
        let lm2 = if i >= 2 {
            prev[i - 2]
        } else {
            grid.interp(prev, x_i - 2.0 * dx)?
        };
        let ar3h = (mc.a)(x_i + 1.5 * dx);
        let ar1h = (mc.a)(x_i + 0.5 * dx);
        let al1h = (mc.a)(x_i - 0.5 * dx);
        let al3h = (mc.a)(x_i - 1.5 * dx);
        let flux_right = 5.0 * ar1h * (rp1 - ctr) / 4.0;
        let flux_right_far = -ar3h * (rp2 - rp1) / 12.0;
        let flux_left = -5.0 * al1h * (ctr - lm1) / 4.0;
        let flux_left_far = al3h * (lm1 - lm2) / 12.0;
        out[i] = (flux_right_far + flux_right + flux_left + flux_left_far) / dx_sq;
    }
    Ok(())
}

/// Power-series from slices with interpolation (drift case).
pub(super) fn apply_power_series_slices(
    tau: f64,
    g_slices: &[&[f64]; TRUNC_ORDER_USIZE + 1],
    grid: Grid1D<f64>,
    x_mid: f64,
) -> Result<f64, SemiflowError> {
    let mut sum = 0.0;
    let mut tau_pow = 1.0;
    for k in 0..=TRUNC_ORDER_USIZE {
        let gk_val = grid.interp(g_slices[k], x_mid)?;
        sum += FACTORIAL_INVERSE[k] * tau_pow * gk_val;
        tau_pow *= tau;
    }
    Ok(sum)
}

/// Power-series from slices at node index (no-drift fast path; no interpolation).
#[inline]
pub(super) fn apply_power_series_slices_at_node(
    tau: f64,
    g_slices: &[&[f64]; TRUNC_ORDER_USIZE + 1],
    i: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut tau_pow = 1.0;
    for k in 0..=TRUNC_ORDER_USIZE {
        sum += FACTORIAL_INVERSE[k] * tau_pow * g_slices[k][i];
        tau_pow *= tau;
    }
    sum
}
