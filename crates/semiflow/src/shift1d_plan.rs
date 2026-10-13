//! τ-keyed step plan of `ShiftChernoff1D<f64>` (ADR-0215): per node `τ·c(x)` and
//! the three planned samples at `x ± 2√(aτ)`, `x + 2bτ`, by the expressions of
//! `apply_at_node_f64` (bit-identical). Layout: header, `[stride]`, the engine's
//! grid and coefficient identity (public `a, b, c` and the closure storage), nodes.

use super::{validate_a_f64, validate_tau_f64, ShiftChernoff1D};
use crate::{
    chernoff::ChernoffFunction,
    diffusion_storage::Storage3,
    error::SemiflowError,
    grid::Grid1D,
    grid_fn::GridFn1D,
    parallel1d::parallel_nodes_into,
    sample_table::{plan_sample, planned_stride, sample_probe, PreparedGridFn, Sample1D},
    scratch::ScratchPool,
    step_plan::{grid_words, header, header_matches, ptr_bits, words_match, HEADER},
};

/// Largest plan accepted (doubles, 32 MiB).
const PLAN_MAX: usize = 1 << 22;
const TAG: f64 = 3.0;
/// Engine words after `[stride]`: grid (3), `a, b, c` (3), closure storage (4).
const ENGINE: usize = 10;
/// Start of the node data.
const META: usize = HEADER + 1 + ENGINE;

/// The engine state a plan depends on besides the header.
fn engine_words(s: &ShiftChernoff1D<f64>) -> [f64; ENGINE] {
    let [g0, g1, g2] = grid_words(&s.grid);
    let [c0, c1, c2, c3] = s.storage.as_ref().map_or([0.0; 4], Storage3::fingerprint);
    let fp = |f: fn(f64) -> f64| ptr_bits(f as *const ());
    [g0, g1, g2, fp(s.a), fp(s.b), fp(s.c), c0, c1, c2, c3]
}

/// Plan length for `n` nodes on `grid` (`None` on overflow or above `PLAN_MAX`).
fn data_len(n: usize, grid: &Grid1D) -> Option<usize> {
    let data = n
        .checked_mul(1 + 3 * planned_stride(grid))?
        .checked_add(META)?;
    (data <= PLAN_MAX).then_some(data)
}

/// `(data, work)` for states shaped like `proto`.
pub(super) fn len(proto: &GridFn1D<f64>) -> Option<(usize, usize)> {
    let data = data_len(proto.values.len(), &proto.grid)?;
    Some((data, PreparedGridFn::work_len(&proto.grid)))
}

/// Fill the plan for `(τ, proto)`.
///
/// # Errors
/// The first error `apply_into` raises in node order (`tau`, `a(x)`, an
/// unsupported interpolant), or a `data` length other than `plan_len`.
#[allow(clippy::cast_precision_loss)] // stride ≤ 12
pub(super) fn fill(
    s: &ShiftChernoff1D<f64>,
    tau: f64,
    proto: &GridFn1D<f64>,
    data: &mut [f64],
) -> Result<(), SemiflowError> {
    validate_tau_f64(tau)?;
    if data_len(proto.values.len(), &proto.grid) != Some(data.len()) {
        return Err(SemiflowError::DomainViolation {
            what: "ShiftChernoff1D::fill_plan: data length differs from plan_len",
            value: data.len() as f64,
        });
    }
    let grid = proto.grid;
    let stride = planned_stride(&grid);
    data[..HEADER].copy_from_slice(&header(s, TAG, tau, proto.values.len(), &grid));
    data[HEADER] = stride as f64;
    data[HEADER + 1..META].copy_from_slice(&engine_words(s));
    parallel_nodes_into(&mut data[META..], 1 + 3 * stride, |i, node| {
        let x = s.grid.x_at(i);
        let (a_x, b_x, c_x) = (s.eval_a(x), s.eval_b(x), s.eval_c(x));
        validate_a_f64(a_x, x)?;
        let s_diff = 2.0 * libm::sqrt(a_x * tau);
        let s_drift = 2.0 * b_x * tau;
        // apply_into samples right after node 0's coefficient check.
        if i == 0 {
            sample_probe(&proto.values, &grid, x + s_diff)?;
        }
        node[0] = tau * c_x;
        for (pos, slot) in [x + s_diff, x - s_diff, x + s_drift]
            .into_iter()
            .zip(node[1..].chunks_exact_mut(stride))
        {
            plan_sample(&grid, pos, slot);
        }
        Ok(())
    })
}

/// One planned step (a plan that is not this engine's current plan for
/// `(τ, src)` falls back to `apply_into`).
///
/// # Errors
/// As `apply_into`.
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)] // stride ≤ 12; the hook's arguments
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
    let (table_len, stride) = (
        PreparedGridFn::work_len(&src.grid),
        planned_stride(&src.grid),
    );
    let usable = header_matches(data, &header(s, TAG, tau, n, &src.grid))
        && data_len(n, &src.grid) == Some(data.len())
        && data[HEADER].to_bits() == (stride as f64).to_bits()
        && words_match(data, HEADER + 1, &engine_words(s))
        && work.len() >= table_len;
    if !usable {
        return s.apply_into(tau, src, dst, scratch);
    }
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
