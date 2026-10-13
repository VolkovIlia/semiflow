//! τ-keyed step plan of `ShiftChernoff1D<f64>` (ADR-0215): per node `τ·c(x)` and
//! the three planned samples at `x ± 2√(aτ)`, `x + 2bτ`, by the expressions of
//! `apply_at_node_f64` (bit-identical).

use super::{validate_a_f64, validate_tau_f64, ShiftChernoff1D};
use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    grid_fn::GridFn1D,
    sample_table::{plan_sample, planned_stride, PreparedGridFn, Sample1D},
    scratch::ScratchPool,
    step_plan::{header, header_matches, HEADER},
};

const PLAN_MAX: usize = 1 << 24;
const TAG: f64 = 3.0;
/// `[stride]` after the header.
const META: usize = HEADER + 1;

/// `(data, work)` for states shaped like `proto`.
pub(super) fn len(proto: &GridFn1D<f64>) -> Option<(usize, usize)> {
    let data = META + proto.values.len() * (1 + 3 * planned_stride(&proto.grid));
    (data <= PLAN_MAX).then(|| (data, PreparedGridFn::work_len(&proto.grid)))
}

/// Fill the plan for `(τ, proto)`.
///
/// # Errors
/// `tau` invalid or `a(x)` invalid at some node — as `apply_into`.
#[allow(clippy::cast_precision_loss)] // stride ≤ 12
pub(super) fn fill(
    s: &ShiftChernoff1D<f64>,
    tau: f64,
    proto: &GridFn1D<f64>,
    data: &mut [f64],
) -> Result<(), SemiflowError> {
    validate_tau_f64(tau)?;
    let grid = proto.grid;
    let stride = planned_stride(&grid);
    data[..HEADER].copy_from_slice(&header(s, TAG, tau, proto.values.len(), &grid));
    data[HEADER] = stride as f64;
    for (i, node) in data[META..].chunks_exact_mut(1 + 3 * stride).enumerate() {
        let x = s.grid.x_at(i);
        let (a_x, b_x, c_x) = (s.eval_a(x), s.eval_b(x), s.eval_c(x));
        validate_a_f64(a_x, x)?;
        let s_diff = 2.0 * libm::sqrt(a_x * tau);
        let s_drift = 2.0 * b_x * tau;
        node[0] = tau * c_x;
        for (pos, slot) in [x + s_diff, x - s_diff, x + s_drift]
            .into_iter()
            .zip(node[1..].chunks_exact_mut(stride))
        {
            plan_sample(&grid, pos, slot);
        }
    }
    Ok(())
}

/// One planned step (a mismatched plan falls back to `apply_into`).
///
/// # Errors
/// As `apply_into`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)] // stride ≤ 12; the hook's arguments
pub(super) fn apply(
    s: &ShiftChernoff1D<f64>,
    tau: f64,
    data: &[f64],
    work: &mut [f64],
    src: &GridFn1D<f64>,
    dst: &mut GridFn1D<f64>,
    scratch: &mut ScratchPool<f64>,
) -> Result<(), SemiflowError> {
    let n = src.values.len();
    let table_len = PreparedGridFn::work_len(&src.grid);
    if !header_matches(data, &header(s, TAG, tau, n, &src.grid)) || work.len() < table_len {
        return s.apply_into(tau, src, dst, scratch);
    }
    let stride = data[HEADER] as usize;
    let per_node = 1 + 3 * stride;
    dst.values.resize(n, 0.0);
    let table = &mut work[..table_len];
    PreparedGridFn::fill(&src.values, &src.grid, table);
    let f = PreparedGridFn::view(&src.values, src.grid, table);
    let nodes = &data[META..];
    crate::parallel1d::parallel_eval_into(&mut dst.values, |i| {
        let node = &nodes[i * per_node..(i + 1) * per_node];
        let smp = |k: usize| {
            let at = 1 + k * stride;
            f.sample_planned(&node[at..at + stride])
        };
        let term1 = 0.25 * smp(0)?;
        let term2 = 0.25 * smp(1)?;
        let term3 = 0.50 * smp(2)?;
        let term4 = node[0] * f.node(i);
        Ok(term1 + term2 + term3 + term4)
    })
}
