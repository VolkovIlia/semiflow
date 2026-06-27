//! Channel-parallel helpers for `graph_batched` (ADR-0184 D3/D4/D5).
//!
//! All symbols are `pub(super)`. Gated by `#[cfg(feature = "parallel")]` at
//! call-sites in `mod.rs`.
//!
//! ## Bit-equality guarantee (ADR-0184 D5)
//!
//! Forward pass — each worker calls the SAME single-channel kernel as the
//! serial path. No cross-channel reduction → 0-ULP identical.
//!
//! Gradient path — per-channel partials are computed independently, then
//! summed into `grad_theta` in ascending channel index on the calling thread.
//! This is numerically identical to the serial loop that does the same adds
//! in the same order (ADR-0184 D4).

use alloc::{sync::Arc, vec, vec::Vec};
use std::sync::Mutex;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::SemiflowFloat,
    graph::{Graph, Laplacian},
    graph_sensitivity::GeneratorSensitivity,
    graph_signal::GraphSignal,
    magnus_graph::MagnusGraphHeatChernoff,
    scratch::ScratchPool,
};

use super::helpers::{
    accumulate_grad_channel, evolve_channel, evolve_magnus6_channel, evolve_magnus_channel,
    evolve_vc_channel,
};

// ---------------------------------------------------------------------------
// Threshold
// ---------------------------------------------------------------------------

/// Minimum channels to trigger parallel execution (ADR-0184 D3).
pub(super) const MIN_CHANNELS_PARALLEL: usize = 2;

// ---------------------------------------------------------------------------
// Error-slot helpers
// ---------------------------------------------------------------------------

type ErrSlot = Arc<Mutex<Option<SemiflowError>>>;

/// Consume an `ErrSlot` after `thread::scope` exits and return its error.
fn take_err(slot: ErrSlot) -> Result<(), SemiflowError> {
    match Arc::try_unwrap(slot) {
        Ok(m) => match m.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(()),
        },
        Err(_) => Ok(()), // unreachable after scope closes
    }
}

/// Store an error in the slot (first error wins).
fn store_err(slot: &ErrSlot, e: SemiflowError) {
    let mut guard = slot.lock().unwrap();
    if guard.is_none() {
        *guard = Some(e);
    }
}

// ---------------------------------------------------------------------------
// par_evolve_batched — generic ChernoffFunction
// ---------------------------------------------------------------------------

/// Channel-parallel generic [`ChernoffFunction`] evolve.
///
/// `C: Sync` is required so `&C` can be shared across `thread::scope` workers.
/// All graph Chernoff types satisfy `Sync` — they contain `Arc<_>` and
/// `Box<dyn Fn … + Send + Sync>`.
#[allow(clippy::too_many_arguments)]
pub(super) fn par_evolve_batched<C, F>(
    func: &C,
    graph: &Arc<Graph<F>>,
    tau: F,
    n_steps: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
    n: usize,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GraphSignal<F>> + Sync,
    F: SemiflowFloat,
{
    let err: ErrSlot = Arc::new(Mutex::new(None));
    std::thread::scope(|s| {
        for (c, dst_c) in dst_cols.chunks_mut(n).enumerate() {
            let src_c = &src_cols[c * n..(c + 1) * n];
            let g = Arc::clone(graph);
            let err_arc = Arc::clone(&err);
            s.spawn(move || {
                let mut scr = ScratchPool::<F>::new();
                let mut a = GraphSignal::zeros(Arc::clone(&g));
                let mut b = GraphSignal::zeros(g);
                if let Err(e) = evolve_channel(func, tau, n_steps, src_c, dst_c, &mut a, &mut b, &mut scr) {
                    store_err(&err_arc, e);
                }
            });
        }
    });
    take_err(err)
}

// ---------------------------------------------------------------------------
// par_evolve_magnus — Magnus K=4 (pre-hoisted Laplacians)
// ---------------------------------------------------------------------------

/// Channel-parallel Magnus K=4. `l1`, `l2` are pre-hoisted and shared read-only.
///
/// `Laplacian<F>: Send + Sync` (contains only `Vec<_>` and `Copy` scalars).
#[allow(clippy::too_many_arguments)]
pub(super) fn par_evolve_magnus<F: SemiflowFloat>(
    l1: &Laplacian<F>,
    l2: &Laplacian<F>,
    tau: F,
    n_steps: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
    n: usize,
    g: &Arc<Graph<F>>,
) {
    std::thread::scope(|s| {
        for (c, dst_c) in dst_cols.chunks_mut(n).enumerate() {
            let src_c = &src_cols[c * n..(c + 1) * n];
            let g = Arc::clone(g);
            s.spawn(move || {
                let mut scr = ScratchPool::<F>::new();
                let mut a = GraphSignal::zeros(Arc::clone(&g));
                let mut b = GraphSignal::zeros(g);
                evolve_magnus_channel(l1, l2, tau, n_steps, src_c, dst_c, &mut a, &mut b, &mut scr);
            });
        }
    });
}

// ---------------------------------------------------------------------------
// par_evolve_magnus6 — Magnus K=6 (f64 only, pre-hoisted)
// ---------------------------------------------------------------------------

/// Channel-parallel Magnus K=6 (f64 only, ADR-0056).
#[allow(clippy::too_many_arguments)]
pub(super) fn par_evolve_magnus6(
    l1: &Laplacian<f64>,
    l2: &Laplacian<f64>,
    l3: &Laplacian<f64>,
    tau: f64,
    n_steps: usize,
    src_cols: &[f64],
    dst_cols: &mut [f64],
    n: usize,
    g: &Arc<Graph<f64>>,
) {
    std::thread::scope(|s| {
        for (c, dst_c) in dst_cols.chunks_mut(n).enumerate() {
            let src_c = &src_cols[c * n..(c + 1) * n];
            let g = Arc::clone(g);
            s.spawn(move || {
                let mut scr = ScratchPool::<f64>::new();
                let mut a = GraphSignal::zeros(Arc::clone(&g));
                let mut b = GraphSignal::zeros(g);
                evolve_magnus6_channel(l1, l2, l3, tau, n_steps, src_c, dst_c, &mut a, &mut b, &mut scr);
            });
        }
    });
}

// ---------------------------------------------------------------------------
// par_evolve_vc — VarCoef Magnus K=4 (pre-hoisted samples)
// ---------------------------------------------------------------------------

/// Channel-parallel variable-coefficient Magnus K=4.
#[allow(clippy::too_many_arguments)]
pub(super) fn par_evolve_vc<F: SemiflowFloat>(
    l1: &Laplacian<F>,
    sqrt_a1: &[F],
    l2: &Laplacian<F>,
    sqrt_a2: &[F],
    tau: F,
    n_steps: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
    n: usize,
    g: &Arc<Graph<F>>,
) {
    std::thread::scope(|s| {
        for (c, dst_c) in dst_cols.chunks_mut(n).enumerate() {
            let src_c = &src_cols[c * n..(c + 1) * n];
            let g = Arc::clone(g);
            s.spawn(move || {
                let mut scr = ScratchPool::<F>::new();
                let mut a = GraphSignal::zeros(Arc::clone(&g));
                let mut b = GraphSignal::zeros(g);
                evolve_vc_channel(l1, sqrt_a1, l2, sqrt_a2, tau, n_steps, src_c, dst_c, &mut a, &mut b, &mut scr);
            });
        }
    });
}

// ---------------------------------------------------------------------------
// par_grad_batched — adjoint gradient, order-pinned reduction (ADR-0184 D4)
// ---------------------------------------------------------------------------

/// Channel-parallel adjoint gradient with order-pinned reduction.
///
/// Each worker accumulates its channel into an isolated `Vec<F>` starting at
/// zero (so the first add = set). The calling thread then sums `tmp_grads[c]`
/// into `grad_theta` in ascending channel order — yielding the same f64 bit
/// pattern as the serial loop (ADR-0184 D4).
///
/// `P: Sync` is required because workers share `&param_deriv`.
/// `MagnusGraphHeatChernoff<F>` is `Sync` by construction (its `lap_at_t`
/// field is `Box<dyn Fn + Send + Sync>`), so no explicit bound is needed.
#[allow(clippy::too_many_arguments)]
pub(super) fn par_grad_batched<F, P>(
    mc: &MagnusGraphHeatChernoff<F>,
    u0_cols: &[F],
    dj_cols: &[F],
    n_steps: usize,
    tau: F,
    param_deriv: &P,
    grad_theta: &mut [F],
    n_cols: usize,
    n: usize,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F> + Sync,
{
    let n_params = param_deriv.n_params();
    let err: ErrSlot = Arc::new(Mutex::new(None));
    let mut tmp_grads: Vec<Vec<F>> = (0..n_cols).map(|_| vec![F::zero(); n_params]).collect();
    let g_arc = Arc::clone(&mc.graph);
    std::thread::scope(|s| {
        for (c, tmp_c) in tmp_grads.iter_mut().enumerate() {
            let u0_c = &u0_cols[c * n..(c + 1) * n];
            let dj_c = &dj_cols[c * n..(c + 1) * n];
            let g = Arc::clone(&g_arc);
            let err_arc = Arc::clone(&err);
            s.spawn(move || {
                let mut scr = ScratchPool::<F>::new();
                // accumulate_grad_channel zeros its internal tmp then ADDS into tmp_c.
                // tmp_c starts at zero → effectively sets it to per-channel gradient.
                if let Err(e) = accumulate_grad_channel(
                    mc, u0_c, dj_c, &g, n_steps, tau, param_deriv, n_params, tmp_c, &mut scr,
                ) {
                    store_err(&err_arc, e);
                }
            });
        }
    });
    take_err(err)?;
    // Sum in ascending channel order — 0-ULP identical to serial (D4).
    for tmp_c in &tmp_grads {
        for (k, &v) in tmp_c.iter().enumerate() {
            grad_theta[k] += v;
        }
    }
    Ok(())
}
