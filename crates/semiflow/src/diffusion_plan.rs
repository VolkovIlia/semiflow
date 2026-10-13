//! τ-keyed step plan of `DiffusionChernoff<f64>` (ADR-0215).
//!
//! Per node the plan stores the five (constant `a`) or ten (variable `a`: γ-A
//! baseline plus ζ-A correction) sample positions as planned samples — table
//! offset and interpolation weights — and, for variable `a`, the step-constant
//! scalars of the ζ correction. Every value is produced by the expression the
//! node kernels in `diffusion_helpers.rs` evaluate, and combined in the same
//! order, so a planned step is bit-identical to `apply_into` (`G_STEP_PLAN_BIT_EQUAL`).
//! The coefficient closures are called once per plan instead of once per step.

use super::{validate_a_x_f64, validate_tau_f64, DiffusionChernoff, W0, W1, W2};
use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    grid::Grid1D,
    grid_fn::GridFn1D,
    sample_table::{plan_sample, planned_stride, PreparedGridFn},
    scratch::ScratchPool,
    step_plan::{header, header_matches, HEADER},
};

/// Largest plan accepted (doubles): beyond, `plan_len` is `None` (today's path).
const PLAN_MAX: usize = 1 << 24;
/// `[stride, τ²]` after the header.
const META: usize = HEADER + 2;
/// Per-node scalars of the variable-`a` plan: `2δ, δ², 2δ³, a·a′, a·a″/2, a′·a″/4`.
const VAR_SCALARS: usize = 6;

/// `(tag, scalars per node, samples per node)`.
fn shape(dc: &DiffusionChernoff<f64>) -> (f64, usize, usize) {
    if dc.is_const_a() {
        (1.0, 0, 5)
    } else {
        (2.0, VAR_SCALARS, 10)
    }
}

/// `(data, work)` for states shaped like `proto`.
pub(super) fn len(dc: &DiffusionChernoff<f64>, proto: &GridFn1D<f64>) -> Option<(usize, usize)> {
    let (_, scalars, samples) = shape(dc);
    let per_node = scalars + samples * planned_stride(&proto.grid);
    let data = META + proto.values.len() * per_node;
    (data <= PLAN_MAX).then(|| (data, PreparedGridFn::work_len(&proto.grid)))
}

/// Fill the plan for `(τ, proto)`.
///
/// # Errors
/// `tau` invalid or `a(x_pre)` invalid at some node — as `apply_into`.
#[allow(clippy::cast_precision_loss)] // stride ≤ 12
pub(super) fn fill(
    dc: &DiffusionChernoff<f64>,
    tau: f64,
    proto: &GridFn1D<f64>,
    data: &mut [f64],
) -> Result<(), SemiflowError> {
    validate_tau_f64(tau)?;
    let (tag, scalars, samples) = shape(dc);
    let grid = proto.grid;
    let stride = planned_stride(&grid);
    data[..HEADER].copy_from_slice(&header(dc, tag, tau, proto.values.len(), &grid));
    data[HEADER] = stride as f64;
    data[HEADER + 1] = tau * tau;
    let per_node = scalars + samples * stride;
    for (i, node) in data[META..].chunks_exact_mut(per_node).enumerate() {
        let x = dc.grid.x_at(i);
        if dc.is_const_a() {
            fill_const(dc, tau, x, &grid, node)?;
        } else {
            fill_variable(dc, tau, x, &grid, node)?;
        }
    }
    Ok(())
}

/// The five positions of `gamma_a_const_f64`.
fn fill_const(
    dc: &DiffusionChernoff<f64>,
    tau: f64,
    x: f64,
    grid: &Grid1D,
    node: &mut [f64],
) -> Result<(), SemiflowError> {
    let a_val = dc.call_a(x);
    validate_a_x_f64(a_val, x)?;
    let h0 = 2.0 * libm::sqrt(a_val * tau);
    let h0_3 = 2.0 * libm::sqrt(3.0 * a_val * tau);
    let stride = planned_stride(grid);
    for (pos, slot) in [x, x + h0, x - h0, x + h0_3, x - h0_3]
        .into_iter()
        .zip(node.chunks_exact_mut(stride))
    {
        plan_sample(grid, pos, slot);
    }
    Ok(())
}

/// The five positions of `gamma_a_baseline_f64`, the five of
/// `zeta_correction_f64`, and the ζ scalars.
fn fill_variable(
    dc: &DiffusionChernoff<f64>,
    tau: f64,
    x: f64,
    grid: &Grid1D,
    node: &mut [f64],
) -> Result<(), SemiflowError> {
    let s_half = 0.5 * tau;
    let x_pre = x + s_half * dc.call_a_prime(x);
    let a_at_pre = dc.call_a(x_pre);
    validate_a_x_f64(a_at_pre, x_pre)?;
    let h0 = 2.0 * libm::sqrt(a_at_pre * tau);
    let h0_3 = 2.0 * libm::sqrt(3.0 * a_at_pre * tau);
    let shifted = |raw: f64| raw + s_half * dc.call_a_prime(raw);
    let center_pos = x_pre + s_half * dc.call_a_prime(x_pre);
    let (near_p, near_n) = (shifted(x_pre + h0), shifted(x_pre - h0));
    let (far_p, far_n) = (shifted(x_pre + h0_3), shifted(x_pre - h0_3));
    let delta = libm::sqrt(tau).max(2.0 * dc.grid.dx());
    let positions = [
        center_pos,
        near_p,
        near_n,
        far_p,
        far_n,
        x,
        x + delta,
        x - delta,
        x + 2.0 * delta,
        x - 2.0 * delta,
    ];
    let (a_x, a_prime_x, app_x) = (dc.call_a(x), dc.call_a_prime(x), dc.call_a_double_prime(x));
    let (scalars, rest) = node.split_at_mut(VAR_SCALARS);
    scalars.copy_from_slice(&[
        2.0 * delta,
        delta * delta,
        2.0 * delta * delta * delta,
        a_x * a_prime_x,
        a_x * app_x / 2.0,
        a_prime_x * app_x / 4.0,
    ]);
    for (pos, slot) in positions
        .into_iter()
        .zip(rest.chunks_exact_mut(planned_stride(grid)))
    {
        plan_sample(grid, pos, slot);
    }
    Ok(())
}

/// One planned step; a plan that does not match `(engine, τ, src)` falls back
/// to `apply_into`.
///
/// # Errors
/// As `apply_into`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments
)] // stride ≤ 12; the hook's arguments
pub(super) fn apply(
    dc: &DiffusionChernoff<f64>,
    tau: f64,
    data: &[f64],
    work: &mut [f64],
    src: &GridFn1D<f64>,
    dst: &mut GridFn1D<f64>,
    scratch: &mut ScratchPool<f64>,
) -> Result<(), SemiflowError> {
    let (tag, scalars, samples) = shape(dc);
    let n = src.values.len();
    let table_len = PreparedGridFn::work_len(&src.grid);
    let want = header(dc, tag, tau, n, &src.grid);
    if !header_matches(data, &want) || work.len() < table_len {
        return dc.apply_into(tau, src, dst, scratch);
    }
    let (stride, tt) = (data[HEADER] as usize, data[HEADER + 1]);
    let per_node = scalars + samples * stride;
    dst.values.resize(n, 0.0);
    let table = &mut work[..table_len];
    PreparedGridFn::fill(&src.values, &src.grid, table);
    let f = PreparedGridFn::view(&src.values, src.grid, table);
    let nodes = &data[META..];
    crate::parallel1d::parallel_eval_into(&mut dst.values, |i| {
        let node = &nodes[i * per_node..(i + 1) * per_node];
        if scalars == 0 {
            const_node(&f, node, stride)
        } else {
            variable_node(&f, node, stride, tt)
        }
    })
}

/// `gamma_a_const_f64` from planned samples.
fn const_node(f: &PreparedGridFn<'_>, node: &[f64], stride: usize) -> Result<f64, SemiflowError> {
    let s = |k: usize| f.sample_planned(&node[k * stride..(k + 1) * stride]);
    let center = W0 * s(0)?;
    let near = W1 * (s(1)? + s(2)?);
    let far = W2 * (s(3)? + s(4)?);
    Ok(center + near + far)
}

/// `gamma_a_baseline_f64 + zeta_correction_f64` from planned samples.
fn variable_node(
    f: &PreparedGridFn<'_>,
    node: &[f64],
    stride: usize,
    tt: f64,
) -> Result<f64, SemiflowError> {
    let (sc, smp) = node.split_at(VAR_SCALARS);
    let s = |k: usize| f.sample_planned(&smp[k * stride..(k + 1) * stride]);
    let center = W0 * s(0)?;
    let near = W1 * (s(1)? + s(2)?);
    let far = W2 * (s(3)? + s(4)?);
    let baseline = center + near + far;
    let (f0, f_pos1, f_neg1, f_pos2, f_neg2) = (s(5)?, s(6)?, s(7)?, s(8)?, s(9)?);
    let f1 = (f_pos1 - f_neg1) / sc[0];
    let f2 = (f_pos1 - 2.0 * f0 + f_neg1) / sc[1];
    let f3 = (f_pos2 - 2.0 * f_pos1 + 2.0 * f_neg1 - f_neg2) / sc[2];
    Ok(baseline + tt * (sc[3] * f3 + sc[4] * f2 + sc[5] * f1))
}
