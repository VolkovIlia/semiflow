//! Z-pass of the parallel 3-D Strang sweep (stride-`nx·ny` pencils; ADR-0045,
//! τ-keyed plan per ADR-0215).

use alloc::{sync::Arc, vec::Vec};
use std::sync::Mutex;

use super::{record_error, ErrorSlot};
use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    parallel_pool::ParallelPool3D,
    step_plan::{axis_plan, ThreadPencil},
    strang2d_parallel::chunk_count,
};

// ---------------------------------------------------------------------------
// Z-pass (3D) — stride-(nx*ny) pencils, gather/scatter over (i,j) pairs
// ---------------------------------------------------------------------------

/// Shared context for Z-pass threads.
#[derive(Clone, Copy)]
struct ZCtx3D<'p, F: SemiflowFloat> {
    nx: usize,
    ny: usize,
    gz: Grid1D<F>,
    plan: Option<&'p (Vec<F>, usize)>,
}

/// Apply one Z-pass to `state` (3D x-fastest) using `n_threads`.
///
/// Total Z-pencils = `nx * ny`. Two-phase gather/scatter via `temp_z`.
/// Generic over `F: SemiflowFloat + Send + Sync` (Wave 5, ADR-0045 §5.3).
#[allow(clippy::too_many_arguments)]
pub(crate) fn parallel_z_pass_3d<Z, F>(
    state: &mut [F],
    nx: usize,
    ny: usize,
    gz: Grid1D<F>,
    n_threads: usize,
    op: &Z,
    tau: F,
    temp_z: &mut Vec<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat + Send + Sync + ParallelPool3D,
    Z: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    let nz = gz.n;
    let total = nx * ny;
    let col_chunk = chunk_count(total, n_threads);
    let row_chunk = chunk_count(total, n_threads);
    let plan = axis_plan(op, tau, gz)?;
    let plan = plan.as_ref();
    let ctx = ZCtx3D { nx, ny, gz, plan };
    temp_z.resize(total * nz, F::zero());
    let err: ErrorSlot = Arc::new(Mutex::new(None));

    // Phase 1: gather + apply each (i,j) Z-pencil.
    let state_ref: &[F] = state;
    std::thread::scope(|s| {
        for (cidx, tc) in temp_z.chunks_mut(col_chunk * nz).enumerate() {
            let ps = cidx * col_chunk;
            let p_count = tc.len() / nz;
            let e2 = Arc::clone(&err);
            s.spawn(move || z_apply_chunk(state_ref, ctx, op, tau, ps, p_count, tc, &e2));
        }
    });
    super::first_error(&err)?;

    // Phase 2: scatter temp_z → state (parallel over (j,k) rows).
    let temp_ref: &[F] = temp_z;
    std::thread::scope(|s| {
        for (ridx, sr) in state.chunks_mut(row_chunk * nx).enumerate() {
            let rs = ridx * row_chunk;
            let rows = sr.len() / nx;
            s.spawn(move || z_scatter(sr, temp_ref, nx, ny, nz, col_chunk, rs, rows));
        }
    });
    Ok(())
}

/// Gather + apply Z-pencils `ps..ps+p_count`.
///
/// Pencil index `p = i + j*nx` → `(i = p%nx, j = p/nx)`.
#[allow(clippy::too_many_arguments)]
fn z_apply_chunk<Z, F>(
    state: &[F],
    ctx: ZCtx3D<'_, F>,
    op: &Z,
    tau: F,
    ps: usize,
    p_count: usize,
    tc: &mut [F],
    err: &ErrorSlot,
) where
    F: SemiflowFloat + Send + Sync + ParallelPool3D,
    Z: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let nz = ctx.gz.n;
    let nx = ctx.nx;
    let ny = ctx.ny;
    let total = nx * ny;
    let mut buf = F::with_pool_3d(|pool| pool.take_vec(nz));
    let mut tp = ThreadPencil::new(ctx.plan);
    for p in 0..p_count {
        let pi = ps + p;
        if pi >= total {
            break;
        }
        let i = pi % nx;
        let j = pi / nx;
        for k in 0..nz {
            buf[k] = state[k * nx * ny + j * nx + i];
        }
        let col_fn = GridFn1D {
            values: core::mem::take(&mut buf),
            grid: ctx.gz,
        };
        match tp.apply(op, tau, &col_fn) {
            Ok(ev) => {
                let ts = p * nz;
                tc[ts..ts + nz].copy_from_slice(ev);
                buf = col_fn.values;
            }
            Err(e) => {
                F::with_pool_3d(|pool| pool.return_vec(col_fn.values));
                record_error(err, e);
                return;
            }
        }
    }
    F::with_pool_3d(|pool| pool.return_vec(buf));
}

/// Scatter Z-pass results: for row `(j,k)` in `state`, read from temp at
/// `(pencil_idx / col_chunk)*col_chunk*nz + (pencil_idx % col_chunk)*nz + k`
/// where `pencil_idx = i + j*nx`.
#[allow(clippy::too_many_arguments)]
fn z_scatter<F: SemiflowFloat + Send>(
    sr: &mut [F],
    temp: &[F],
    nx: usize,
    ny: usize,
    nz: usize,
    col_chunk: usize,
    rs: usize,
    rows: usize,
) {
    for r in 0..rows {
        let gr = rs + r;
        if gr >= ny * nz {
            break;
        }
        let j = gr % ny;
        let k = gr / ny;
        if k >= nz {
            break;
        }
        for i in 0..nx {
            let pi = i + j * nx;
            let ti = (pi / col_chunk) * col_chunk * nz + (pi % col_chunk) * nz + k;
            sr[r * nx + i] = temp[ti];
        }
    }
}
