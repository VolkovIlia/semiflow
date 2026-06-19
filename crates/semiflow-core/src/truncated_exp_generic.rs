// Generic (non-f64) computation helpers for [`TruncatedExpDiffusionChernoff`].
// Extracted from `truncated_exp.rs` per suckless ≤500-line cap.
// Declared as `pub(super) mod generic_helpers;` inside `truncated_exp.rs`.

use super::{TruncatedExpDiffusionChernoff, FACTORIAL_INVERSE, TRUNC_ORDER_USIZE};
use crate::{
    error::SemiflowError,
    float::{from_f64, half, SemiflowFloat},
    grid_fn::GridFn1D,
};

#[inline]
pub(super) fn compute_x_mid_generic<F: SemiflowFloat>(
    mc: &TruncatedExpDiffusionChernoff<F>,
    x_i: F,
    tau: F,
) -> F {
    match mc.b_for_conjugation {
        Some(b) => x_i - half::<F>() * tau * b(x_i),
        None => x_i,
    }
}

pub(super) fn precompute_g_grids_generic<F: SemiflowFloat>(
    mc: &TruncatedExpDiffusionChernoff<F>,
    f: &GridFn1D<F>,
) -> Result<[GridFn1D<F>; TRUNC_ORDER_USIZE + 1], SemiflowError> {
    let g0 = f.clone();
    let g1 = apply_g_once_grid_generic(mc, &g0)?;
    let g2 = apply_g_once_grid_generic(mc, &g1)?;
    let g3 = apply_g_once_grid_generic(mc, &g2)?;
    let g4 = apply_g_once_grid_generic(mc, &g3)?;
    Ok([g0, g1, g2, g3, g4])
}

pub(super) fn apply_g_once_grid_generic<F: SemiflowFloat>(
    mc: &TruncatedExpDiffusionChernoff<F>,
    prev: &GridFn1D<F>,
) -> Result<GridFn1D<F>, SemiflowError> {
    let n = prev.values.len();
    let dx = mc.grid.dx();
    let dx_sq = dx * dx;
    let two = from_f64::<F>(2.0);
    let mut out = prev.zeroed_like();

    for i in 0..n {
        let x_i = mc.grid.x_at(i);

        let x_right = x_i + dx;
        let h_right = if i + 1 < n {
            prev.values[i + 1]
        } else {
            prev.sample_generic(x_right)?
        };

        let x_left = x_i - dx;
        let h_left = if i > 0 {
            prev.values[i - 1]
        } else {
            prev.sample_generic(x_left)?
        };

        let h_i = prev.values[i];

        let a_half_right = (mc.a)((x_i + x_right) / two);
        let a_half_left = (mc.a)((x_i + x_left) / two);

        out.values[i] = (a_half_right * (h_right - h_i) - a_half_left * (h_i - h_left)) / dx_sq;
    }
    Ok(out)
}

pub(super) fn apply_at_node_generic<F: SemiflowFloat>(
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
