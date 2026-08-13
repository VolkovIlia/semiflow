//! Batched multi-channel evolve for 1-D grid kernels (ADR-0193, Issue #19).
//!
//! ADR-0184 gave the **graph** kernels a batched path; the **grid** 1-D family
//! (`ShiftChernoff1D`, `DiffusionChernoff`, `Diffusion4thChernoff`,
//! `DriftReactionChernoff`, …) still evolved one state per object. Three
//! quant-finance workloads degrade to Python loops without it: pricing a strike
//! strip is 11–50 independent solves under the *same* generator with only `u0`
//! differing; bump Greeks are ±bump re-solves that could amortise the
//! coefficient setup; and a Fokker–Planck backtest evolves ~250 density anchors
//! under one local-vol operator, paying object construction and a GIL round-trip
//! for each.
//!
//! ## Memory layout (normative)
//!
//! `[C, N]` **channel-major**: channel `c` occupies `cols[c*N .. (c+1)*N]`,
//! contiguous. Identical to ADR-0184 D1, and for the same reason — contiguity is
//! what lets `chunks_mut(n)` hand disjoint output slices to workers with no
//! synchronisation. Python callers use `[N, C]`; the transpose is dissolved into
//! the mandatory GIL-boundary copy.
//!
//! ## 0-ULP
//!
//! Every channel runs the same single-channel `apply_into` sequence at the same
//! `tau`, and there is no cross-channel reduction, so the batched result is
//! bit-identical to C sequential solves (ADR-0184 D5 carried over). Batching is
//! a throughput device, not an accuracy one.

extern crate alloc;

use alloc::vec;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

/// Validate the `[C, N]` layout and return the channel count.
///
/// # Errors
/// `DomainViolation` if `n == 0`, the buffers differ in length, or the length is
/// not divisible by `n`.
fn validate_layout<F: SemiflowFloat>(
    src: &[F],
    dst: &[F],
    n: usize,
) -> Result<usize, SemiflowError> {
    if n == 0 {
        return Err(SemiflowError::DomainViolation {
            what: "evolve_batched_1d: grid.n == 0",
            value: 0.0,
        });
    }
    if src.len() != dst.len() {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "evolve_batched_1d: src_cols.len() != dst_cols.len()",
            value: src.len() as f64,
        });
    }
    if src.len() % n != 0 {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "evolve_batched_1d: src_cols.len() not divisible by grid.n",
            value: src.len() as f64,
        });
    }
    Ok(src.len() / n)
}

/// Ping-pong one channel through `n_steps` applications, writing into `dst`.
#[allow(clippy::too_many_arguments)]
fn evolve_channel<C, F>(
    func: &C,
    tau: F,
    n_steps: usize,
    src: &[F],
    dst: &mut [F],
    buf_a: &mut GridFn1D<F>,
    buf_b: &mut GridFn1D<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
{
    buf_a.values.copy_from_slice(src);
    let mut src_is_a = true;
    for _ in 0..n_steps {
        if src_is_a {
            func.apply_into(tau, buf_a, buf_b, scratch)?;
        } else {
            func.apply_into(tau, buf_b, buf_a, scratch)?;
        }
        src_is_a = !src_is_a;
    }
    let result: &GridFn1D<F> = if src_is_a { buf_a } else { buf_b };
    dst.copy_from_slice(&result.values);
    Ok(())
}

/// Evolve `C` channels of a 1-D grid kernel in one call (serial build).
///
/// `src_cols` / `dst_cols` are `[C, N]` channel-major (see the module note).
/// `n_steps == 0` copies `src` to `dst` unchanged.
///
/// # Errors
/// `DomainViolation` on a layout inconsistency; propagates kernel errors.
#[cfg(not(feature = "parallel"))]
pub fn evolve_batched_1d<C, F>(
    func: &C,
    grid: Grid1D<F>,
    t_final: F,
    n_steps: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
{
    serial_evolve(func, grid, t_final, n_steps, src_cols, dst_cols)
}

/// Evolve `C` channels of a 1-D grid kernel in one call (parallel build).
///
/// Above [`MIN_CHANNELS_PARALLEL`] channels the work is split across threads,
/// one worker per channel, each with its own `ScratchPool` and ping-pong pair.
/// Bit-identical to the serial path (no cross-channel reduction).
///
/// # Errors
/// `DomainViolation` on a layout inconsistency; propagates kernel errors.
#[cfg(feature = "parallel")]
pub fn evolve_batched_1d<C, F>(
    func: &C,
    grid: Grid1D<F>,
    t_final: F,
    n_steps: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>> + Sync,
    F: SemiflowFloat,
{
    let n_cols = validate_layout::<F>(src_cols, dst_cols, grid.n)?;
    if n_cols == 0 {
        return Ok(());
    }
    if n_steps == 0 {
        dst_cols.copy_from_slice(src_cols);
        return Ok(());
    }
    if n_cols >= MIN_CHANNELS_PARALLEL && !node_parallelism_engages(grid.n) {
        #[allow(clippy::cast_precision_loss)]
        let tau = t_final / from_f64::<F>(n_steps as f64);
        return par_evolve(func, grid, tau, n_steps, src_cols, dst_cols);
    }
    serial_evolve(func, grid, t_final, n_steps, src_cols, dst_cols)
}

/// Shared serial driver (also the fallback for the parallel build).
fn serial_evolve<C, F>(
    func: &C,
    grid: Grid1D<F>,
    t_final: F,
    n_steps: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
{
    let n = grid.n;
    let n_cols = validate_layout::<F>(src_cols, dst_cols, n)?;
    if n_cols == 0 {
        return Ok(());
    }
    if n_steps == 0 {
        dst_cols.copy_from_slice(src_cols);
        return Ok(());
    }
    #[allow(clippy::cast_precision_loss)]
    let tau = t_final / from_f64::<F>(n_steps as f64);
    let mut scratch = ScratchPool::<F>::new();
    let mut buf_a = GridFn1D::<F>::new_generic(grid, vec![F::zero(); n])?;
    let mut buf_b = GridFn1D::<F>::new_generic(grid, vec![F::zero(); n])?;
    for c in 0..n_cols {
        let src_c = &src_cols[c * n..(c + 1) * n];
        let dst_c = &mut dst_cols[c * n..(c + 1) * n];
        evolve_channel(
            func, tau, n_steps, src_c, dst_c, &mut buf_a, &mut buf_b, &mut scratch,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Parallel path
// ---------------------------------------------------------------------------

/// Minimum channel count before channel-parallelism engages.
#[cfg(feature = "parallel")]
pub const MIN_CHANNELS_PARALLEL: usize = 2;

/// Whether `parallel1d` would already be threading over nodes at this grid size.
///
/// Channel-parallel and node-parallel are made **mutually exclusive** rather
/// than nested. `parallel1d::resolve_threads_1d` returns 1 below
/// `2 × min_points_per_thread()`, so the two regimes are exactly complementary:
/// no `(C, N)` engages both, and none is left with neither axis available.
/// Large-N is already saturated node-wise; small-N is precisely where node
/// parallelism declines and channels are the only axis left — which is also the
/// regime the motivating workloads live in (`n = 513…801`).
#[cfg(feature = "parallel")]
fn node_parallelism_engages(n: usize) -> bool {
    n >= 2 * crate::parallel1d::min_points_per_thread()
}

/// One `std::thread::scope` worker per channel.
#[cfg(feature = "parallel")]
fn par_evolve<C, F>(
    func: &C,
    grid: Grid1D<F>,
    tau: F,
    n_steps: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>> + Sync,
    F: SemiflowFloat,
{
    use alloc::sync::Arc;
    use std::sync::Mutex;

    let n = grid.n;
    let err: Arc<Mutex<Option<SemiflowError>>> = Arc::new(Mutex::new(None));
    std::thread::scope(|s| {
        for (c, dst_c) in dst_cols.chunks_mut(n).enumerate() {
            let src_c = &src_cols[c * n..(c + 1) * n];
            let err_arc = Arc::clone(&err);
            s.spawn(move || {
                if let Err(e) = run_worker(func, grid, tau, n_steps, src_c, dst_c) {
                    store_err(&err_arc, e);
                }
            });
        }
    });
    take_err(&err)
}

/// One worker's whole job: own buffers, own pool, one channel.
#[cfg(feature = "parallel")]
fn run_worker<C, F>(
    func: &C,
    grid: Grid1D<F>,
    tau: F,
    n_steps: usize,
    src_c: &[F],
    dst_c: &mut [F],
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GridFn1D<F>>,
    F: SemiflowFloat,
{
    // Belt-and-braces against oversubscription: even if the node-parallel
    // threshold is later retuned, this worker will not fan out again.
    crate::parallel1d::pin_single_thread_1d();
    let n = grid.n;
    let mut scratch = ScratchPool::<F>::new();
    let mut buf_a = GridFn1D::<F>::new_generic(grid, vec![F::zero(); n])?;
    let mut buf_b = GridFn1D::<F>::new_generic(grid, vec![F::zero(); n])?;
    evolve_channel(
        func, tau, n_steps, src_c, dst_c, &mut buf_a, &mut buf_b, &mut scratch,
    )
}

/// Record the first error reported by any worker.
#[cfg(feature = "parallel")]
fn store_err(slot: &alloc::sync::Arc<std::sync::Mutex<Option<SemiflowError>>>, e: SemiflowError) {
    if let Ok(mut guard) = slot.lock() {
        if guard.is_none() {
            *guard = Some(e);
        }
    }
}

/// Drain the error slot after the scope joins.
#[cfg(feature = "parallel")]
fn take_err(
    slot: &alloc::sync::Arc<std::sync::Mutex<Option<SemiflowError>>>,
) -> Result<(), SemiflowError> {
    match slot.lock() {
        Ok(mut guard) => guard.take().map_or(Ok(()), Err),
        Err(_) => Err(SemiflowError::DomainViolation {
            what: "evolve_batched_1d: worker error slot poisoned",
            value: 0.0,
        }),
    }
}
