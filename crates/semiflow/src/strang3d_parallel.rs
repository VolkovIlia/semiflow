//! Parallel helpers for `Strang3D::apply` when the `parallel` feature is enabled.
//!
//! Mirrors `strang2d_parallel` for 3D: same scatter/gather approach, same
//! bit-equality contract (ADR-0018). Three passes:
//!
//! - **X-pass** — X-pencils are contiguous; plain `chunks_mut` suffices.
//! - **Y-pass** — Y-pencils have stride `nx`; gather/scatter via `temp` buffer.
//! - **Z-pass** — Z-pencils have stride `nx*ny`; same gather/scatter approach.
//!
//! Each pencil is fully local to one thread; output written to disjoint memory.
//! Ceiling-divide chunking is deterministic. No FP summation across threads.
//!
//! ## Wave 5 (ADR-0045) — generic-over-F lift
//!
//! All functions are now generic over `F: SemiflowFloat + Send + Sync`.
//! Thread-local pools are dispatched via the sealed `ParallelPool3D` trait
//! (see [`crate::parallel_pool`]). The f64 codegen path is **byte-identical** to Wave 2.
//!
//! ## Wave 2 (ADR-0042) thread-local pools
//!
//! Each worker thread owns a `thread_local!` `RefCell<ScratchPool<F>>`. The
//! per-pencil `buf` that was previously allocated with `vec![F::zero(); n]` is
//! now borrowed from the per-precision pool. Capacity is grow-only; the pool is
//! cleared only via the explicit test hook [`drain_thread_local_pools_3d`].

use alloc::{sync::Arc, vec::Vec};
use core::cell::Cell;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Re-export the f64 3D pool for backward compat (tests use PARALLEL_3D_POOL)
// ---------------------------------------------------------------------------
/// Per-thread grow-only scratch pool for 3D parallel pencil buffers (f64).
///
/// Not part of the stable public API; not frozen at v1.0.0.
#[doc(hidden)]
pub use crate::parallel_pool::PARALLEL_3D_POOL_F64 as PARALLEL_3D_POOL;
use crate::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    parallel_pool::{drain_thread_local_pools_3d_for, ParallelPool3D},
    strang2d_parallel::chunk_count,
};

/// Drain the calling thread's 3D parallel scratch pool (f64 + f32).
///
/// Clears all free buffers (capacity released). Used in
/// `tests/parallel_scratch_drain.rs`.
///
/// Not part of the stable public API; not frozen at v1.0.0.
#[doc(hidden)]
pub fn drain_thread_local_pools_3d() {
    drain_thread_local_pools_3d_for::<f64>();
    drain_thread_local_pools_3d_for::<f32>();
}

// ---------------------------------------------------------------------------
// Constants and test hook
// ---------------------------------------------------------------------------

/// Minimum pencils per thread; below this threshold the serial path is used.
pub(crate) const MIN_PENCILS_PER_THREAD: usize = 16;

thread_local! {
    /// Test hook: pin the 3D parallel thread count to a fixed value.
    ///
    /// `None` → `available_parallelism()`. `Some(k)` → exactly `k` threads.
    /// Separate from `strang2d_parallel::FORCE_THREADS` to avoid cross-
    /// contamination between 2D and 3D test runs in the same process.
    ///
    /// Not part of the stable public API; not frozen at v1.0.0.
    #[doc(hidden)]
    pub static FORCE_THREADS_3D: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Resolve thread count for `total_pencils`.
pub(crate) fn resolve_threads_3d(total_pencils: usize) -> usize {
    let raw = FORCE_THREADS_3D.with(Cell::get).unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    });
    raw.min(total_pencils / MIN_PENCILS_PER_THREAD).max(1)
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

type ErrorSlot = Arc<Mutex<Option<SemiflowError>>>;

fn record_error(slot: &ErrorSlot, e: SemiflowError) {
    let mut g = slot.lock().unwrap();
    if g.is_none() {
        *g = Some(e);
    }
}

fn extract_error_3d(slot: ErrorSlot) -> Result<(), SemiflowError> {
    match Arc::try_unwrap(slot) {
        Ok(mutex) => match mutex.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        },
        Err(_) => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// X-pass (3D) — contiguous pencils, no gather/scatter needed
// ---------------------------------------------------------------------------

/// Apply one X-pass to a 3D `state` (x-fastest) using `n_threads`.
///
/// Generic over `F: SemiflowFloat + Send + Sync` (Wave 5, ADR-0045 §5.3).
/// f64 codegen path is byte-identical to Wave 2.
pub(crate) fn parallel_x_pass_3d<X, F>(
    state: &mut [F],
    gx: Grid1D<F>,
    n_threads: usize,
    op: &X,
    tau: F,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat + Send + Sync + ParallelPool3D,
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    let nx = gx.n;
    let total_rows = state.len() / nx;
    let chunk_size = chunk_count(total_rows, n_threads);
    let error_slot: ErrorSlot = Arc::new(Mutex::new(None));

    std::thread::scope(|s| {
        for row_chunk in state.chunks_mut(chunk_size * nx) {
            let rows = row_chunk.len() / nx;
            let err_arc = Arc::clone(&error_slot);
            s.spawn(move || x_pass_chunk_3d(row_chunk, gx, rows, op, tau, &err_arc));
        }
    });

    extract_error_3d(error_slot)
}

fn x_pass_chunk_3d<X, F>(
    row_chunk: &mut [F],
    gx: Grid1D<F>,
    rows: usize,
    op: &X,
    tau: F,
    err: &ErrorSlot,
) where
    F: SemiflowFloat + Send + Sync + ParallelPool3D,
    X: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let nx = gx.n;
    // Borrow pencil buffer from thread-local pool (Wave 2, ADR-0042).
    let mut buf = F::with_pool_3d(|pool| pool.take_vec(nx));
    for r in 0..rows {
        let start = r * nx;
        buf.copy_from_slice(&row_chunk[start..start + nx]);
        let row_fn = GridFn1D {
            values: core::mem::take(&mut buf),
            grid: gx,
        };
        match op.apply_chernoff(tau, &row_fn) {
            Ok(ev) => {
                row_chunk[start..start + nx].copy_from_slice(&ev.values);
                buf = row_fn.values;
            }
            Err(e) => {
                F::with_pool_3d(|pool| pool.return_vec(row_fn.values));
                record_error(err, e);
                return;
            }
        }
    }
    F::with_pool_3d(|pool| pool.return_vec(buf));
}

// ---------------------------------------------------------------------------
// Y-pass (3D) — stride-nx pencils, gather/scatter over (i,k) pairs
// ---------------------------------------------------------------------------

/// Shared context for Y-pass threads.
#[derive(Clone, Copy)]
struct YCtx3D<F: SemiflowFloat> {
    nx: usize,
    nz: usize,
    gy: Grid1D<F>,
}

/// Apply one Y-pass to `state` (3D x-fastest) using `n_threads`.
///
/// Total Y-pencils = `nx * nz`. Two-phase gather/scatter via `temp`.
/// Generic over `F: SemiflowFloat + Send + Sync` (Wave 5, ADR-0045 §5.3).
/// Phase 2 of Y-pass: scatter `temp` back into `state` (parallel, row-major).
#[allow(clippy::too_many_arguments)]
fn scatter_y_temp<F: SemiflowFloat + Send + Sync>(
    state: &mut [F],
    temp: &[F],
    nx: usize,
    ny: usize,
    nz: usize,
    row_chunk: usize,
    col_chunk: usize,
) {
    std::thread::scope(|s| {
        for (ridx, sr) in state.chunks_mut(row_chunk * nx).enumerate() {
            let rs = ridx * row_chunk;
            let rows = sr.len() / nx;
            s.spawn(move || y_scatter(sr, temp, nx, ny, nz, col_chunk, rs, rows));
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parallel_y_pass_3d<Y, F>(
    state: &mut [F],
    nx: usize,
    gy: Grid1D<F>,
    nz: usize,
    n_threads: usize,
    op: &Y,
    tau: F,
    temp: &mut Vec<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat + Send + Sync + ParallelPool3D,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    let ny = gy.n;
    let total = nx * nz;
    let col_chunk = chunk_count(total, n_threads);
    let row_chunk = chunk_count(total, n_threads);
    let ctx = YCtx3D { nx, nz, gy };
    temp.resize(total * ny, F::zero());
    let err: ErrorSlot = Arc::new(Mutex::new(None));

    // Phase 1: gather + apply each (i,k) Y-pencil.
    let state_ref: &[F] = state;
    std::thread::scope(|s| {
        for (cidx, tc) in temp.chunks_mut(col_chunk * ny).enumerate() {
            let ps = cidx * col_chunk;
            let p_count = tc.len() / ny;
            let e2 = Arc::clone(&err);
            s.spawn(move || y_apply_chunk(state_ref, ctx, op, tau, ps, p_count, tc, &e2));
        }
    });
    {
        let slot = err.lock().unwrap();
        if let Some(ref e) = *slot {
            return Err(e.clone());
        }
    }

    // Phase 2: scatter temp → state.
    scatter_y_temp(state, temp, nx, ny, nz, row_chunk, col_chunk);
    Ok(())
}

/// Gather + apply Y-pencils `ps..ps+p_count`.
///
/// Pencil index `p = i + k*nx` → `(i = p%nx, k = p/nx)`.
#[allow(clippy::too_many_arguments)]
fn y_apply_chunk<Y, F>(
    state: &[F],
    ctx: YCtx3D<F>,
    op: &Y,
    tau: F,
    ps: usize,
    p_count: usize,
    tc: &mut [F],
    err: &ErrorSlot,
) where
    F: SemiflowFloat + Send + Sync + ParallelPool3D,
    Y: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let ny = ctx.gy.n;
    let nx = ctx.nx;
    let total = nx * ctx.nz;
    let mut buf = F::with_pool_3d(|pool| pool.take_vec(ny));
    for p in 0..p_count {
        let pi = ps + p;
        if pi >= total {
            break;
        }
        let i = pi % nx;
        let k = pi / nx;
        for j in 0..ny {
            buf[j] = state[k * nx * ny + j * nx + i];
        }
        let col_fn = GridFn1D {
            values: core::mem::take(&mut buf),
            grid: ctx.gy,
        };
        match op.apply_chernoff(tau, &col_fn) {
            Ok(ev) => {
                let ts = p * ny;
                tc[ts..ts + ny].copy_from_slice(&ev.values);
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

/// Scatter Y-pass results: for row `(j,k)` in `state`, read from temp at
/// `(pencil_idx / col_chunk)*col_chunk*ny + (pencil_idx % col_chunk)*ny + j`
/// where `pencil_idx = i + k*nx`.
#[allow(clippy::too_many_arguments)]
fn y_scatter<F: SemiflowFloat + Send>(
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
        for i in 0..nx {
            let pi = i + k * nx;
            let ti = (pi / col_chunk) * col_chunk * ny + (pi % col_chunk) * ny + j;
            sr[r * nx + i] = temp[ti];
        }
    }
}

// ---------------------------------------------------------------------------
// Z-pass (3D) — stride-(nx*ny) pencils, gather/scatter over (i,j) pairs
// ---------------------------------------------------------------------------

/// Shared context for Z-pass threads.
#[derive(Clone, Copy)]
struct ZCtx3D<F: SemiflowFloat> {
    nx: usize,
    ny: usize,
    gz: Grid1D<F>,
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
    let ctx = ZCtx3D { nx, ny, gz };
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
    {
        let slot = err.lock().unwrap();
        if let Some(ref e) = *slot {
            return Err(e.clone());
        }
    }

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
    ctx: ZCtx3D<F>,
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
        match op.apply_chernoff(tau, &col_fn) {
            Ok(ev) => {
                let ts = p * nz;
                tc[ts..ts + nz].copy_from_slice(&ev.values);
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
