//! Batched multi-channel evolve for graph heat kernels (ADR-0184, Issue #10).
//!
//! **Memory layout**: `[C, N]` channel-major flat buffer.
//! Channel `c` occupies `cols[c*N .. c*N+N]`.  Python callers use `[N, C]`
//! (torch-native); the transpose is dissolved into the mandatory GIL-boundary
//! copy (ADR-0184 D1).
//!
//! ## Forward API
//! | Function | Covers |
//! |---|---|
//! | [`evolve_batched`] | all plain `ChernoffFunction<F, S = GraphSignal>` kernels |
//! | [`evolve_batched_magnus`] | `MagnusGraphHeatChernoff` (K=4, hoisted GL₄) |
//! | [`evolve_batched_magnus6`] | `MagnusGraphHeat6thChernoff` (K=6, hoisted GL₆, f64 only) |
//! | [`evolve_batched_varcoef_magnus`] | `VarCoefMagnusGraphHeatChernoff` (hoisted GL₄ + a) |
//!
//! ## Adjoint API (impls on existing types)
//! `PreSampledMagnusAdj::evolve_state_adjoint_batched_into`
//! `PreSampledVarCoefAdj::evolve_state_adjoint_batched_into`
//! [`adjoint_state_gradient_batched`]
//!
//! ## 0-ULP correctness
//! Every batched path calls the SAME single-channel kernel C times (D5).
//! All typed helpers use `t_start = 0`, matching `ChernoffFunction::apply_into`,
//! so hoisting Laplacian samples (identical every step) preserves bit patterns.

mod helpers;
use helpers::{
    accumulate_grad_channel, evolve_channel, evolve_magnus6_channel, evolve_magnus_channel,
    evolve_vc_channel, hoist_vc_gl4_samples, run_presampled_adj_buf,
    run_presampled_varcoef_adj_buf, validate_adj_grad_layout, validate_batched_layout,
};

use alloc::sync::Arc;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph::Graph,
    graph_adjoint_presampled::{PreSampledMagnusAdj, PreSampledVarCoefAdj},
    graph_sensitivity::GeneratorSensitivity,
    graph_signal::GraphSignal,
    magnus6_graph::{MagnusGraphHeat6thChernoff, GL6_C1, GL6_C2, GL6_C3},
    magnus_graph::{MagnusGraphHeatChernoff, GL4_C1_F64, GL4_C2_F64},
    scratch::ScratchPool,
    varcoef_magnus_graph::VarCoefMagnusGraphHeatChernoff,
};

// ---------------------------------------------------------------------------
// evolve_batched — generic (covers GraphHeat 1/2, GraphHeat4th, GraphHeat6)
// ---------------------------------------------------------------------------

/// Evolve `n_cols` channels in one call using a plain [`ChernoffFunction`].
///
/// `graph` is used only for ping-pong buffer allocation; its topology must
/// match `src_cols` (i.e. `graph.n_nodes() == src_cols.len() / n_cols`).
/// `n_steps` == 0 → copy src to dst unchanged.
///
/// # Errors
/// `DomainViolation` if layout is inconsistent.
#[allow(clippy::cast_precision_loss)]
pub fn evolve_batched<C, F>(
    func: &C,
    graph: &Arc<Graph<F>>,
    t_final: F,
    n_steps: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GraphSignal<F>>,
    F: SemiflowFloat,
{
    let n = graph.n_nodes();
    let n_cols = validate_batched_layout::<F>(src_cols, dst_cols, n)?;
    if n_cols == 0 {
        return Ok(());
    }
    if n_steps == 0 {
        dst_cols.copy_from_slice(src_cols);
        return Ok(());
    }
    let tau = t_final / from_f64::<F>(n_steps as f64);
    let mut scratch = ScratchPool::<F>::new();
    let mut buf_a = GraphSignal::zeros(Arc::clone(graph));
    let mut buf_b = GraphSignal::zeros(Arc::clone(graph));
    for c in 0..n_cols {
        let src_c = &src_cols[c * n..(c + 1) * n];
        let dst_c = &mut dst_cols[c * n..(c + 1) * n];
        evolve_channel(
            func,
            tau,
            n_steps,
            src_c,
            dst_c,
            &mut buf_a,
            &mut buf_b,
            &mut scratch,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// evolve_batched_magnus — MagnusGraphHeatChernoff K=4 (hoisted GL₄)
// ---------------------------------------------------------------------------

/// Evolve `n_cols` channels with Magnus K=4, hoisting GL₄ Laplacian samples.
///
/// `t_start = 0` on every step (mirrors `apply_into`) → l1/l2 are identical
/// each step, so they are sampled ONCE and shared across all channels and steps.
/// `n_steps` == 0 → copy src to dst unchanged.
///
/// # Errors
/// Layout violations or convergence-radius exceeded.
#[allow(clippy::cast_precision_loss)]
pub fn evolve_batched_magnus<F: SemiflowFloat>(
    mc: &MagnusGraphHeatChernoff<F>,
    t_final: F,
    n_steps: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
) -> Result<(), SemiflowError> {
    let n = mc.graph.n_nodes();
    let n_cols = validate_batched_layout::<F>(src_cols, dst_cols, n)?;
    if n_cols == 0 {
        return Ok(());
    }
    if n_steps == 0 {
        dst_cols.copy_from_slice(src_cols);
        return Ok(());
    }
    let tau = t_final / from_f64::<F>(n_steps as f64);
    crate::magnus_graph_helpers::validate_tau(tau)?;
    if mc.convergence_radius_check {
        crate::magnus_graph_helpers::validate_magnus_radius(mc.rho_bar_max, tau)?;
    }
    // Hoist: t_start = 0 every step → same l1, l2 every step.
    let c1 = from_f64::<F>(GL4_C1_F64);
    let c2 = from_f64::<F>(GL4_C2_F64);
    let l1 = mc.laplacian_at(c1 * tau);
    let l2 = mc.laplacian_at(c2 * tau);
    let mut scratch = ScratchPool::<F>::new();
    let mut buf_a = GraphSignal::zeros(Arc::clone(&mc.graph));
    let mut buf_b = GraphSignal::zeros(Arc::clone(&mc.graph));
    for c in 0..n_cols {
        let src_c = &src_cols[c * n..(c + 1) * n];
        let dst_c = &mut dst_cols[c * n..(c + 1) * n];
        evolve_magnus_channel(
            &l1,
            &l2,
            tau,
            n_steps,
            src_c,
            dst_c,
            &mut buf_a,
            &mut buf_b,
            &mut scratch,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// evolve_batched_magnus6 — MagnusGraphHeat6thChernoff K=6 (f64 only, hoisted)
// ---------------------------------------------------------------------------

/// Evolve `n_cols` channels with Magnus K=6, hoisting GL₆ Laplacian samples.
///
/// `f64` only (`ADR-0056`). `t_start` = 0 on every step → l1/l2/l3 sampled ONCE.
/// `n_steps` == 0 → copy src to dst unchanged.
///
/// # Errors
/// Layout violations or convergence-radius exceeded.
#[allow(clippy::cast_precision_loss)]
pub fn evolve_batched_magnus6(
    mc: &MagnusGraphHeat6thChernoff<f64>,
    t_final: f64,
    n_steps: usize,
    src_cols: &[f64],
    dst_cols: &mut [f64],
) -> Result<(), SemiflowError> {
    let n = mc.graph().n_nodes();
    let n_cols = validate_batched_layout::<f64>(src_cols, dst_cols, n)?;
    if n_cols == 0 {
        return Ok(());
    }
    if n_steps == 0 {
        dst_cols.copy_from_slice(src_cols);
        return Ok(());
    }
    let tau = t_final / n_steps as f64;
    crate::magnus_graph_helpers::validate_tau(tau)?;
    if mc.convergence_radius_check {
        crate::magnus_graph_helpers::validate_magnus_radius(mc.rho_bar_max, tau)?;
    }
    // Hoist GL₆ samples.
    let l1 = mc.laplacian_at(GL6_C1 * tau);
    let l2 = mc.laplacian_at(GL6_C2 * tau);
    let l3 = mc.laplacian_at(GL6_C3 * tau);
    let graph_arc = Arc::new(Graph::<f64>::path(n));
    let mut scratch = ScratchPool::<f64>::new();
    let mut buf_a = GraphSignal::zeros(Arc::clone(&graph_arc));
    let mut buf_b = GraphSignal::zeros(Arc::clone(&graph_arc));
    for c in 0..n_cols {
        let src_c = &src_cols[c * n..(c + 1) * n];
        let dst_c = &mut dst_cols[c * n..(c + 1) * n];
        evolve_magnus6_channel(
            &l1,
            &l2,
            &l3,
            tau,
            n_steps,
            src_c,
            dst_c,
            &mut buf_a,
            &mut buf_b,
            &mut scratch,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// evolve_batched_varcoef_magnus — VarCoef Magnus K=4 (hoisted GL₄ + a)
// ---------------------------------------------------------------------------

/// Evolve `n_cols` channels with variable-coefficient Magnus K=4.
///
/// Hoists GL₄ Laplacian + diffusion-coefficient samples (same `t_start = 0`
/// every step). `n_steps` == 0 → copy src to dst unchanged.
///
/// # Errors
/// Layout violations, bad a-weights, or convergence-radius exceeded.
#[allow(clippy::cast_precision_loss)]
pub fn evolve_batched_varcoef_magnus<F: SemiflowFloat>(
    mc: &VarCoefMagnusGraphHeatChernoff<F>,
    t_final: F,
    n_steps: usize,
    src_cols: &[F],
    dst_cols: &mut [F],
) -> Result<(), SemiflowError> {
    let n = mc.n_nodes();
    let n_cols = validate_batched_layout::<F>(src_cols, dst_cols, n)?;
    if n_cols == 0 {
        return Ok(());
    }
    if n_steps == 0 {
        dst_cols.copy_from_slice(src_cols);
        return Ok(());
    }
    let tau = t_final / from_f64::<F>(n_steps as f64);
    crate::varcoef_magnus_graph::validate_tau(tau)?;
    if mc.convergence_radius_check {
        crate::varcoef_magnus_graph::validate_magnus_radius(mc.rho_bar_max, mc.a_sup_max, tau)?;
    }
    // Hoist GL₄ + a samples once (t_start = 0 every step → identical each step).
    let samples = hoist_vc_gl4_samples(mc, tau, n)?;
    // Build ping-pong buffers using dummy path graph (n_nodes only matters).
    let dummy_g = Arc::new(Graph::<F>::path(n));
    let mut buf_a = GraphSignal::zeros(Arc::clone(&dummy_g));
    let mut buf_b = GraphSignal::zeros(Arc::clone(&dummy_g));
    let mut scratch = ScratchPool::<F>::new();
    for c in 0..n_cols {
        let src_c = &src_cols[c * n..(c + 1) * n];
        let dst_c = &mut dst_cols[c * n..(c + 1) * n];
        evolve_vc_channel(
            &samples.l1,
            &samples.sqrt_a1,
            &samples.l2,
            &samples.sqrt_a2,
            tau,
            n_steps,
            src_c,
            dst_c,
            &mut buf_a,
            &mut buf_b,
            &mut scratch,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PreSampledMagnusAdj — batched state adjoint
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> PreSampledMagnusAdj<F> {
    /// Backward costate sweep for `n_cols` channels simultaneously.
    ///
    /// `src_cols` and `dst_cols` are `[C, N]` flat (channel-major).
    /// `n_steps` == 0 → copy src to dst unchanged.
    ///
    /// 0-ULP identical to calling `evolve_state_adjoint_into` C times in
    /// ascending channel order (same kernel, same presampled sequence).
    ///
    /// # Errors
    /// `DomainViolation` if `n_steps != self.n_steps()`, layout mismatch, or
    /// convergence check fails.
    pub fn evolve_state_adjoint_batched_into(
        &self,
        tau: F,
        n_steps: usize,
        src_cols: &[F],
        dst_cols: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let n = self.n_nodes;
        let n_cols = validate_batched_layout::<F>(src_cols, dst_cols, n)?;
        if n_cols == 0 {
            return Ok(());
        }
        // Delegate to single-channel method per channel (0-ULP by construction).
        // Build a dummy graph once for GraphSignal construction.
        let dummy_g: Arc<Graph<F>> = Arc::new(Graph::path(n));
        for c in 0..n_cols {
            let src_c = &src_cols[c * n..(c + 1) * n];
            let dst_c = &mut dst_cols[c * n..(c + 1) * n];
            run_presampled_adj_buf(tau, n_steps, &self.seq, &dummy_g, src_c, dst_c, scratch)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PreSampledVarCoefAdj — batched state adjoint
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> PreSampledVarCoefAdj<F> {
    /// Backward costate sweep for `n_cols` channels simultaneously (`VarCoef`).
    ///
    /// 0-ULP identical to calling `evolve_state_adjoint_into` C times.
    ///
    /// # Errors
    /// `DomainViolation` if layout mismatch or `n_steps != self.n_steps()`.
    pub fn evolve_state_adjoint_batched_into(
        &self,
        tau: F,
        n_steps: usize,
        src_cols: &[F],
        dst_cols: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let n = self.n_nodes;
        let n_cols = validate_batched_layout::<F>(src_cols, dst_cols, n)?;
        if n_cols == 0 {
            return Ok(());
        }
        let dummy_g: Arc<Graph<F>> = Arc::new(Graph::path(n));
        for c in 0..n_cols {
            let src_c = &src_cols[c * n..(c + 1) * n];
            let dst_c = &mut dst_cols[c * n..(c + 1) * n];
            run_presampled_varcoef_adj_buf(
                tau,
                n_steps,
                &self.seq,
                &self.a_seq,
                n,
                &dummy_g,
                src_c,
                dst_c,
                scratch,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// adjoint_state_gradient_batched — summed edge-weight gradient
// ---------------------------------------------------------------------------

/// Summed adjoint-state gradient `∂J/∂θ = Σ_c ∂J_c/∂θ` over `n_cols` channels.
///
/// `u0_cols` and `dj_cols` are `[C, N]` flat.  `grad_theta` (length `n_params`)
/// is **zeroed once** then accumulated in ascending channel index (ADR-0184 D4).
///
/// 0-ULP identical to calling [`adjoint_state_gradient`] C times in ascending
/// channel order: each channel's gradient is computed via `accumulate_grad_channel`
/// into a zeroed temp (`adjoint_state_gradient` zeros its output buffer on entry),
/// then added into `grad_theta`. Fixed accumulation order `c = 0..n_cols` preserves
/// bit patterns.
///
/// # Errors
/// `DomainViolation` if layout inconsistent or `grad_theta.len() != n_params`.
#[allow(clippy::too_many_arguments)]
pub fn adjoint_state_gradient_batched<F, P>(
    mc: &MagnusGraphHeatChernoff<F>,
    u0_cols: &[F],
    dj_cols: &[F],
    n_steps: usize,
    tau: F,
    param_deriv: &P,
    grad_theta: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    let n = mc.graph.n_nodes();
    let n_cols = validate_adj_grad_layout(
        u0_cols,
        dj_cols,
        n,
        grad_theta.len(),
        param_deriv.n_params(),
    )?;
    // Zero ONCE — accumulated in ascending channel index for 0-ULP identity.
    grad_theta.fill(F::zero());
    if n_steps == 0 || n_cols == 0 {
        return Ok(());
    }
    let graph_arc = Arc::clone(&mc.graph);
    let n_params = param_deriv.n_params();
    for c in 0..n_cols {
        let u0_c = &u0_cols[c * n..(c + 1) * n];
        let dj_c = &dj_cols[c * n..(c + 1) * n];
        accumulate_grad_channel(
            mc,
            u0_c,
            dj_c,
            &graph_arc,
            n_steps,
            tau,
            param_deriv,
            n_params,
            grad_theta,
            scratch,
        )?;
    }
    Ok(())
}
