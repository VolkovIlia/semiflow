//! Parallel helpers for `Strang2D::apply` when the `parallel` feature is enabled.
//!
//! Implements palindromic Strang `S(τ) = X(τ/2) ∘ Y(τ) ∘ X(τ/2)` via
//! `std::thread::scope` with disjoint row/column ownership. No allocation
//! inside the time loop after the initial state clone; per-thread scratch
//! buffers are freed at the end of each pass.
//!
//! ## Wave 5 (ADR-0045) — generic-over-F lift
//!
//! All functions are now generic over `F: SemiflowFloat + Send + Sync`.
//! Thread-local scratch pools are dispatched via the sealed `ParallelPool2D`
//! trait (see [`crate::parallel_pool`]). The f64 codegen path is **byte-identical**
//! to Wave 2: only the type signatures changed, not the bodies.
//!
//! ## Wave 2 (ADR-0042) thread-local pools
//!
//! Each worker thread owns a `thread_local!` `RefCell<ScratchPool<F>>`.  The
//! per-thread `row_buf` / `col_buf` that were previously allocated with
//! `vec![F::zero(); n]` inside every `*_chunk` helper are now borrowed from the
//! per-precision pool. Capacity is grow-only; the pool is cleared only via the
//! explicit test hook [`drain_thread_local_pools_2d`].
//!
//! All new symbols are either `pub(crate)` (internal helpers) or `pub` (test
//! hook `FORCE_THREADS`) — none are part of the stable public API.
//!
//! See `docs/adr/0018-parallel-strang2d.md` and
//! `contracts/semiflow-core.tensor.yaml` §3 `parallel_constants`.

use alloc::{sync::Arc, vec::Vec};
use core::cell::Cell;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Re-export the per-precision pools for backward compatibility
// (tests/parallel_scratch_drain.rs uses PARALLEL_2D_POOL directly)
// ---------------------------------------------------------------------------
/// Per-thread grow-only scratch pool for 2D parallel pencil buffers (f64).
///
/// Replaces the per-call `vec![0.0_f64; n]` allocations in `x_pass_chunk`
/// and `y_apply_cols`. Capacity is grow-only; cleared only via
/// [`drain_thread_local_pools_2d`].
///
/// Not part of the stable public API; not frozen at v1.0.0.
#[doc(hidden)]
pub use crate::parallel_pool::PARALLEL_2D_POOL_F64 as PARALLEL_2D_POOL;
use crate::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    parallel_pool::{drain_thread_local_pools_2d_for, ParallelPool2D},
};

/// Drain the calling thread's 2D parallel scratch pool (f64 + f32).
///
/// Clears all free buffers (capacity released). After this call the next
/// borrow will re-allocate. Used in `tests/parallel_scratch_drain.rs`.
///
/// Not part of the stable public API; not frozen at v1.0.0.
#[doc(hidden)]
pub fn drain_thread_local_pools_2d() {
    drain_thread_local_pools_2d_for::<f64>();
    drain_thread_local_pools_2d_for::<f32>();
}

// ---------------------------------------------------------------------------
// Parallel constants — ADR-0018 §`parallel_constants`
// ---------------------------------------------------------------------------

/// Minimum number of rows each thread owns in the X-pass.
///
/// Threads are not spawned if `ny < 2 * MIN_ROWS_PER_THREAD`, falling back
/// to the serial path (bit-equal by construction).
///
/// Tunable in a follow-up PR if perf data justifies it.
/// See `docs/adr/0018-parallel-strang2d.md` §`parallel_constants`.
pub(crate) const MIN_ROWS_PER_THREAD: usize = 16;

// ---------------------------------------------------------------------------
// Thread-count override — test hook
// ---------------------------------------------------------------------------

thread_local! {
    /// Test hook: pin the parallel thread count to a fixed value.
    ///
    /// `None` (default) → use `available_parallelism()`.
    /// `Some(k)` → use exactly `k` threads (bit-equal sweep in
    /// `tests/strang2d_parallel_bit_equal.rs`).
    ///
    /// Exposed as `pub` so integration tests in
    /// `tests/strang2d_parallel_bit_equal.rs` can set this directly.
    /// This is an intentional narrow surface gated on `feature = "parallel"`.
    ///
    /// Not part of the stable public API; not frozen at v1.0.0.
    #[doc(hidden)]
    pub static FORCE_THREADS: Cell<Option<usize>> = const { Cell::new(None) };
}

// ---------------------------------------------------------------------------
// Thread-count resolution
// ---------------------------------------------------------------------------

/// Resolve the number of threads for a given `ny`.
///
/// Reads `FORCE_THREADS` first (test hook), then `available_parallelism`.
/// Caps at `ny / MIN_ROWS_PER_THREAD` and floors at `1`.
pub(crate) fn resolve_threads(ny: usize) -> usize {
    let raw = FORCE_THREADS.with(Cell::get).unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    });
    raw.min(ny / MIN_ROWS_PER_THREAD).max(1)
}

// ---------------------------------------------------------------------------
// Chunking helper
// ---------------------------------------------------------------------------

/// Ceiling-divide `total` items into `n_threads` contiguous chunks.
///
/// Returns chunk size ≥ 1. The last chunk may be smaller.
pub(crate) fn chunk_count(total: usize, n_threads: usize) -> usize {
    total.div_ceil(n_threads)
}

// ---------------------------------------------------------------------------
// Shared error slot type
// ---------------------------------------------------------------------------

type ErrorSlot = Arc<Mutex<Option<SemiflowError>>>;

// ---------------------------------------------------------------------------
// Parallel X-pass
// ---------------------------------------------------------------------------

/// Apply one X-pass to `state` (row-major, `nx × ny`) using `n_threads`.
///
/// Each thread owns a contiguous range of rows. Row chunks are non-overlapping
/// (`chunks_mut`), so no synchronisation is needed on `state`.
///
/// Generic over `F: SemiflowFloat + Send + Sync` (Wave 5, ADR-0045 §5.3).
/// f64 codegen path is byte-identical to Wave 2.
///
/// # Errors
/// Any `SemiflowError` from `op.apply` is captured and returned after all
/// threads finish.
pub(crate) fn parallel_x_pass<X, F>(
    state: &mut [F],
    gx: Grid1D<F>,
    n_threads: usize,
    op: &X,
    tau: F,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat + Send + Sync + ParallelPool2D,
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    let nx = gx.n;
    let ny = state.len() / nx;
    let error_slot: ErrorSlot = Arc::new(Mutex::new(None));
    let chunk_size = chunk_count(ny, n_threads);

    std::thread::scope(|s| {
        for row_chunk in state.chunks_mut(chunk_size * nx) {
            let rows = row_chunk.len() / nx;
            let err_arc = Arc::clone(&error_slot);
            s.spawn(move || x_pass_chunk(row_chunk, gx, rows, op, tau, &err_arc));
        }
    });

    extract_error(error_slot)
}

/// Process one chunk of rows in the X-pass (per-thread work unit).
///
/// Allocates one `row_buf` of length `nx` before the row loop (Block A scratch
/// reuse). The buffer is moved into `GridFn1D` for each `op.apply` call and
/// reclaimed immediately after, so there is exactly one `Vec` allocation per
/// thread per X-pass rather than one per row.
fn x_pass_chunk<X, F>(
    row_chunk: &mut [F],
    gx: Grid1D<F>,
    rows: usize,
    op: &X,
    tau: F,
    err: &ErrorSlot,
) where
    F: SemiflowFloat + Send + Sync + ParallelPool2D,
    X: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let nx = gx.n;
    // Borrow row buffer from thread-local pool (Wave 2, ADR-0042).
    // take_vec + return_vec avoids per-row allocation while pool is shared.
    let mut row_buf = F::with_pool_2d(|pool| pool.take_vec(nx));
    for r in 0..rows {
        let row_start = r * nx;
        row_buf.copy_from_slice(&row_chunk[row_start..row_start + nx]);
        let row_fn = GridFn1D {
            values: core::mem::take(&mut row_buf),
            grid: gx,
        };
        match op.apply_chernoff(tau, &row_fn) {
            Ok(evolved) => {
                row_chunk[row_start..row_start + nx].copy_from_slice(&evolved.values);
                row_buf = row_fn.values;
            }
            Err(e) => {
                // Return buffer before early exit.
                F::with_pool_2d(|pool| pool.return_vec(row_fn.values));
                let mut slot = err.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(e);
                }
                return;
            }
        }
    }
    F::with_pool_2d(|pool| pool.return_vec(row_buf));
}

// ---------------------------------------------------------------------------
// Parallel Y-pass — context struct
// ---------------------------------------------------------------------------

/// Bundled grid dimensions for the Y-pass column work.
///
/// Collects the parameters that are shared across all column-worker threads,
/// reducing per-function argument counts (clippy `too_many_arguments`).
#[derive(Clone, Copy)]
struct YColCtx<F: SemiflowFloat> {
    /// Width (number of columns) of the full 2-D state.
    nx: usize,
    /// Y-axis grid (gives `ny = gy.n`).
    gy: Grid1D<F>,
    /// Number of columns per thread chunk.
    col_chunk: usize,
}

// ---------------------------------------------------------------------------
// Parallel Y-pass
// ---------------------------------------------------------------------------

/// Apply one Y-pass to `state` (row-major, `nx × ny`) using `n_threads`.
///
/// Two-phase gather/scatter (column-gather, reference kernel option b):
///
/// 1. Phase 1 — gather + apply (read-only from `state`, write column-major
///    `temp`): each thread owns a contiguous range of *column* indices.
/// 2. Phase 2 — scatter `temp` (column-major) back into `state` (row-major):
///    each thread owns a contiguous range of *rows*.
///
/// `temp` is a caller-owned scratch buffer of length `nx × ny` (Block A reuse).
/// The caller allocates once and passes the same buffer for every Y-pass call
/// in one Strang step, eliminating the per-call allocation.
///
/// Generic over `F: SemiflowFloat + Send + Sync` (Wave 5, ADR-0045 §5.3).
///
/// # Errors
/// Any `SemiflowError` from `op.apply` is captured and returned after Phase 1.
pub(crate) fn parallel_y_pass<Y, F>(
    state: &mut [F],
    gy: Grid1D<F>,
    n_threads: usize,
    op: &Y,
    tau: F,
    temp: &mut Vec<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat + Send + Sync + ParallelPool2D,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    let ny = gy.n;
    let nx = state.len() / ny;
    let col_chunk = chunk_count(nx, n_threads);
    let row_chunk = chunk_count(ny, n_threads);
    let ctx = YColCtx { nx, gy, col_chunk };
    // Reuse caller-provided scratch; resize only if capacity is insufficient.
    temp.resize(nx * ny, F::zero());
    let error_slot: ErrorSlot = Arc::new(Mutex::new(None));

    // Phase 1: gather + apply columns (parallel).
    y_phase1_apply(state, temp, ctx, op, tau, &error_slot);
    {
        let slot = error_slot.lock().unwrap();
        if let Some(ref e) = *slot {
            return Err(e.clone());
        }
    }

    // Phase 2: scatter column-major temp → row-major state (parallel).
    y_phase2_scatter(state, temp, nx, ny, row_chunk, col_chunk);

    Ok(())
}

/// Y-pass Phase 1: apply `op` to each column, write to column-major `temp`.
fn y_phase1_apply<Y, F>(
    state: &[F],
    temp: &mut [F],
    ctx: YColCtx<F>,
    op: &Y,
    tau: F,
    err: &ErrorSlot,
) where
    F: SemiflowFloat + Send + Sync + ParallelPool2D,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    let ny = ctx.gy.n;
    std::thread::scope(|s| {
        for (cidx, temp_chunk) in temp.chunks_mut(ctx.col_chunk * ny).enumerate() {
            let col_start = cidx * ctx.col_chunk;
            let cols = temp_chunk.len() / ny;
            let err_arc = Arc::clone(err);
            s.spawn(move || {
                y_apply_cols(state, ctx, op, tau, col_start, cols, temp_chunk, &err_arc);
            });
        }
    });
}

/// Apply `op` to columns `col_start..col_start+cols`, writing to `temp_chunk`.
///
/// Borrows one `col_buf` of length `ny` from the thread-local pool before the
/// column loop (Wave 2, ADR-0042 — replaces `vec![0.0; ny]`). The buffer is
/// moved into `GridFn1D` for each `op.apply` call and reclaimed immediately
/// after, so there is exactly one pool-borrow per thread per Y-pass.
#[allow(clippy::too_many_arguments)]
fn y_apply_cols<Y, F>(
    state: &[F],
    ctx: YColCtx<F>,
    op: &Y,
    tau: F,
    col_start: usize,
    cols: usize,
    temp_chunk: &mut [F],
    err: &ErrorSlot,
) where
    F: SemiflowFloat + Send + Sync + ParallelPool2D,
    Y: ChernoffFunction<F, S = GridFn1D<F>>,
{
    let ny = ctx.gy.n;
    let nx = ctx.nx;
    let mut col_buf = F::with_pool_2d(|pool| pool.take_vec(ny));
    for c in 0..cols {
        let col_idx = col_start + c;
        if col_idx >= nx {
            break;
        }
        for j in 0..ny {
            col_buf[j] = state[j * nx + col_idx];
        }
        let col_fn = GridFn1D {
            values: core::mem::take(&mut col_buf),
            grid: ctx.gy,
        };
        match op.apply_chernoff(tau, &col_fn) {
            Ok(evolved) => {
                let ts = c * ny;
                temp_chunk[ts..ts + ny].copy_from_slice(&evolved.values);
                col_buf = col_fn.values;
            }
            Err(e) => {
                F::with_pool_2d(|pool| pool.return_vec(col_fn.values));
                let mut slot = err.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(e);
                }
                return;
            }
        }
    }
    F::with_pool_2d(|pool| pool.return_vec(col_buf));
}

// ---------------------------------------------------------------------------
// Scatter context struct
// ---------------------------------------------------------------------------

/// Bundled parameters for the scatter phase.
#[derive(Clone, Copy)]
struct ScatterCtx {
    /// Width (columns) of the 2-D state.
    nx: usize,
    /// Height (rows) of the 2-D state.
    ny: usize,
    /// Number of columns per column-chunk (from Phase 1 chunking).
    col_chunk: usize,
}

/// Y-pass Phase 2: scatter column-major `temp` → row-major `state`.
fn y_phase2_scatter<F: SemiflowFloat + Send + Sync>(
    state: &mut [F],
    temp: &[F],
    nx: usize,
    ny: usize,
    row_chunk: usize,
    col_chunk: usize,
) {
    let sctx = ScatterCtx { nx, ny, col_chunk };
    std::thread::scope(|s| {
        for (ridx, state_chunk) in state.chunks_mut(row_chunk * nx).enumerate() {
            let row_start = ridx * row_chunk;
            let rows = state_chunk.len() / nx;
            s.spawn(move || scatter_rows(state_chunk, temp, sctx, row_start, rows));
        }
    });
}

/// Write column-major `temp` values into one row-chunk of `state`.
fn scatter_rows<F: SemiflowFloat>(
    state_chunk: &mut [F],
    temp: &[F],
    ctx: ScatterCtx,
    row_start: usize,
    rows: usize,
) {
    let ScatterCtx { nx, ny, col_chunk } = ctx;
    for r in 0..rows {
        let j = row_start + r;
        for col_idx in 0..nx {
            let temp_idx = (col_idx / col_chunk) * col_chunk * ny + (col_idx % col_chunk) * ny + j;
            state_chunk[r * nx + col_idx] = temp[temp_idx];
        }
    }
}

// ---------------------------------------------------------------------------
// Error extraction
// ---------------------------------------------------------------------------

/// Consume an [`ErrorSlot`] after `thread::scope` exits and return its error.
fn extract_error(slot: ErrorSlot) -> Result<(), SemiflowError> {
    match Arc::try_unwrap(slot) {
        Ok(mutex) => match mutex.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        },
        Err(_) => Ok(()), // unreachable after scope closes
    }
}
