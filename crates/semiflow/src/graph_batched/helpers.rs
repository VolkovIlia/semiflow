//! Private per-channel ping-pong helpers for each batched kernel flavour.

use alloc::sync::Arc;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph::{Graph, Laplacian},
    graph_adjoint_presampled::PreSampledLaplacianSeq,
    graph_sensitivity::{adjoint_state_gradient, GeneratorSensitivity},
    graph_signal::GraphSignal,
    magnus6_graph::apply_exp_omega6_kernel,
    magnus_graph::{MagnusGraphHeatChernoff, GL4_C1_F64, GL4_C2_F64},
    magnus_graph_helpers::apply_exp_omega4_kernel,
    scratch::ScratchPool,
    state::State,
    varcoef_magnus_graph::{
        apply_exp_omega4_la_kernel, validate_a_weights, VarCoefMagnusGraphHeatChernoff,
    },
};

/// Validate `[C, N]` flat layout; return `n_cols = src.len() / n_nodes`.
///
/// # Errors
/// `DomainViolation` if `n_nodes == 0`, lengths differ, or not divisible.
pub(super) fn validate_batched_layout<F: SemiflowFloat>(
    src: &[F],
    dst: &[F],
    n_nodes: usize,
) -> Result<usize, SemiflowError> {
    if n_nodes == 0 {
        return Err(SemiflowError::DomainViolation {
            what: "evolve_batched: n_nodes == 0",
            value: 0.0,
        });
    }
    if src.len() != dst.len() {
        return Err(SemiflowError::DomainViolation {
            what: "evolve_batched: src_cols.len() != dst_cols.len()",
            #[allow(clippy::cast_precision_loss)]
            value: src.len() as f64,
        });
    }
    if src.len() % n_nodes != 0 {
        return Err(SemiflowError::DomainViolation {
            what: "evolve_batched: src_cols.len() not divisible by n_nodes",
            #[allow(clippy::cast_precision_loss)]
            value: src.len() as f64,
        });
    }
    Ok(src.len() / n_nodes)
}

/// Evolve a single channel in `buf_a`/`buf_b` via ping-pong, write result to `dst`.
#[allow(clippy::too_many_arguments)]
pub(in crate::graph_batched) fn evolve_channel<C, F>(
    func: &C,
    tau: F,
    n_steps: usize,
    src: &[F],
    dst: &mut [F],
    buf_a: &mut GraphSignal<F>,
    buf_b: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    C: ChernoffFunction<F, S = GraphSignal<F>>,
    F: SemiflowFloat,
{
    // Load src into buf_a (buf_b may contain previous channel's state — irrelevant
    // because apply_into fully overwrites dst on every step).
    buf_a.zero_into();
    buf_a.axpy_into_slice(F::one(), src);
    let mut src_is_a = true;
    for _ in 0..n_steps {
        if src_is_a {
            func.apply_into(tau, buf_a, buf_b, scratch)?;
        } else {
            func.apply_into(tau, buf_b, buf_a, scratch)?;
        }
        src_is_a = !src_is_a;
    }
    let result: &GraphSignal<F> = if src_is_a { buf_a } else { buf_b };
    dst.copy_from_slice(result.values());
    Ok(())
}

/// Ping-pong one channel with hoisted Magnus K=4 Laplacians.
#[allow(clippy::too_many_arguments)]
pub(in crate::graph_batched) fn evolve_magnus_channel<F: SemiflowFloat>(
    l1: &Laplacian<F>,
    l2: &Laplacian<F>,
    tau: F,
    n_steps: usize,
    src: &[F],
    dst: &mut [F],
    buf_a: &mut GraphSignal<F>,
    buf_b: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) {
    buf_a.zero_into();
    buf_a.axpy_into_slice(F::one(), src);
    let mut src_is_a = true;
    for _ in 0..n_steps {
        if src_is_a {
            apply_exp_omega4_kernel(l1, l2, tau, buf_a, buf_b, scratch);
        } else {
            apply_exp_omega4_kernel(l1, l2, tau, buf_b, buf_a, scratch);
        }
        src_is_a = !src_is_a;
    }
    let result: &GraphSignal<F> = if src_is_a { buf_a } else { buf_b };
    dst.copy_from_slice(result.values());
}

/// Ping-pong one channel with hoisted Magnus K=6 Laplacians.
#[allow(clippy::too_many_arguments)]
pub(in crate::graph_batched) fn evolve_magnus6_channel(
    l1: &Laplacian<f64>,
    l2: &Laplacian<f64>,
    l3: &Laplacian<f64>,
    tau: f64,
    n_steps: usize,
    src: &[f64],
    dst: &mut [f64],
    buf_a: &mut GraphSignal<f64>,
    buf_b: &mut GraphSignal<f64>,
    scratch: &mut ScratchPool<f64>,
) {
    buf_a.zero_into();
    buf_a.axpy_into_slice(1.0_f64, src);
    let mut src_is_a = true;
    for _ in 0..n_steps {
        if src_is_a {
            apply_exp_omega6_kernel(l1, l2, l3, tau, buf_a, buf_b, scratch);
        } else {
            apply_exp_omega6_kernel(l1, l2, l3, tau, buf_b, buf_a, scratch);
        }
        src_is_a = !src_is_a;
    }
    let result: &GraphSignal<f64> = if src_is_a { buf_a } else { buf_b };
    dst.copy_from_slice(result.values());
}

/// Ping-pong one channel with hoisted `VarCoef` Magnus K=4 samples.
#[allow(clippy::too_many_arguments)]
pub(in crate::graph_batched) fn evolve_vc_channel<F: SemiflowFloat>(
    l1: &Laplacian<F>,
    sqrt_a1: &[F],
    l2: &Laplacian<F>,
    sqrt_a2: &[F],
    tau: F,
    n_steps: usize,
    src: &[F],
    dst: &mut [F],
    buf_a: &mut GraphSignal<F>,
    buf_b: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) {
    buf_a.zero_into();
    buf_a.axpy_into_slice(F::one(), src);
    let mut src_is_a = true;
    for _ in 0..n_steps {
        if src_is_a {
            apply_exp_omega4_la_kernel(l1, sqrt_a1, l2, sqrt_a2, tau, buf_a, buf_b, scratch);
        } else {
            apply_exp_omega4_la_kernel(l1, sqrt_a1, l2, sqrt_a2, tau, buf_b, buf_a, scratch);
        }
        src_is_a = !src_is_a;
    }
    let result: &GraphSignal<F> = if src_is_a { buf_a } else { buf_b };
    dst.copy_from_slice(result.values());
}

/// Core adjoint sweep for one channel using pre-sampled Magnus K=4 sequence.
///
/// Equivalent to `run_presampled_adj` from `graph_adjoint_presampled` but
/// operates on flat slices to avoid storing `Arc<Graph<F>>` in the struct.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_presampled_adj_buf<F: SemiflowFloat>(
    tau: F,
    n_steps: usize,
    seq: &PreSampledLaplacianSeq<F>,
    dummy_g: &Arc<Graph<F>>,
    src: &[F],
    dst: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if n_steps == 0 {
        dst.copy_from_slice(src);
        return Ok(());
    }
    let mut lam = GraphSignal::from_fn(Arc::clone(dummy_g), |i| src[i as usize]);
    let mut lam_next = GraphSignal::zeros(Arc::clone(dummy_g));
    for k in 0..n_steps {
        let lap1 = seq.lap_for_step(k, 0)?;
        let lap2 = seq.lap_for_step(k, 1)?;
        crate::magnus_graph_adjoint::apply_exp_omega4_adj_kernel(
            &lap1,
            &lap2,
            tau,
            &lam,
            &mut lam_next,
            scratch,
        );
        core::mem::swap(&mut lam, &mut lam_next);
    }
    dst.copy_from_slice(lam.values());
    Ok(())
}

/// Core adjoint sweep for one channel using pre-sampled `VarCoef` sequence.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_presampled_varcoef_adj_buf<F: SemiflowFloat>(
    tau: F,
    n_steps: usize,
    seq: &PreSampledLaplacianSeq<F>,
    a_seq: &[F],
    n_nodes: usize,
    dummy_g: &Arc<Graph<F>>,
    src: &[F],
    dst: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if n_steps == 0 {
        dst.copy_from_slice(src);
        return Ok(());
    }
    let mut lam = GraphSignal::from_fn(Arc::clone(dummy_g), |i| src[i as usize]);
    let mut lam_next = GraphSignal::zeros(Arc::clone(dummy_g));
    for k in 0..n_steps {
        let lap1 = seq.lap_for_step(k, 0)?;
        let lap2 = seq.lap_for_step(k, 1)?;
        let a1 = &a_seq[2 * k * n_nodes..(2 * k + 1) * n_nodes];
        let a2 = &a_seq[(2 * k + 1) * n_nodes..(2 * k + 2) * n_nodes];
        let sqrt_a1: alloc::vec::Vec<F> = a1.iter().map(|&x| x.sqrt()).collect();
        let sqrt_a2: alloc::vec::Vec<F> = a2.iter().map(|&x| x.sqrt()).collect();
        crate::magnus_graph_adjoint::validate_varcoef_sqrt_a(&sqrt_a1, &sqrt_a2)?;
        let mut sa1 = scratch.take_vec(n_nodes);
        let mut sa2 = scratch.take_vec(n_nodes);
        sa1.copy_from_slice(&sqrt_a1);
        sa2.copy_from_slice(&sqrt_a2);
        crate::magnus_graph_adjoint::apply_exp_omega4_la_adj_kernel(
            &lap1,
            &sa1,
            &lap2,
            &sa2,
            tau,
            &lam,
            &mut lam_next,
            scratch,
        );
        scratch.return_vec(sa2);
        scratch.return_vec(sa1);
        core::mem::swap(&mut lam, &mut lam_next);
    }
    dst.copy_from_slice(lam.values());
    Ok(())
}

// ---------------------------------------------------------------------------
// VarCoef GL₄ hoisting helper
// ---------------------------------------------------------------------------

/// Hoisted GL₄ samples for `VarCoefMagnusGraphHeatChernoff`.
pub(super) struct VcGl4Samples<F: SemiflowFloat> {
    pub l1: alloc::sync::Arc<Laplacian<F>>,
    pub sqrt_a1: alloc::vec::Vec<F>,
    pub l2: alloc::sync::Arc<Laplacian<F>>,
    pub sqrt_a2: alloc::vec::Vec<F>,
}

/// Hoist GL₄ Laplacian and diffusivity samples for `VarCoefMagnusGraphHeatChernoff`.
///
/// `t_start = 0` on every step → same l1, l2, a1, a2 every step.
///
/// # Errors
/// `DomainViolation` if a-weights are invalid.
pub(super) fn hoist_vc_gl4_samples<F: SemiflowFloat>(
    mc: &VarCoefMagnusGraphHeatChernoff<F>,
    tau: F,
    n: usize,
) -> Result<VcGl4Samples<F>, SemiflowError> {
    let c1 = from_f64::<F>(GL4_C1_F64);
    let c2 = from_f64::<F>(GL4_C2_F64);
    let l1 = (mc.lap_at_t)(c1 * tau);
    let l2 = (mc.lap_at_t)(c2 * tau);
    let a1 = (mc.a_at_t)(c1 * tau);
    let a2 = (mc.a_at_t)(c2 * tau);
    validate_a_weights(n, &a1, &a2)?;
    let sqrt_a1: alloc::vec::Vec<F> = a1.iter().map(|&x| x.sqrt()).collect();
    let sqrt_a2: alloc::vec::Vec<F> = a2.iter().map(|&x| x.sqrt()).collect();
    Ok(VcGl4Samples {
        l1,
        sqrt_a1,
        l2,
        sqrt_a2,
    })
}

// ---------------------------------------------------------------------------
// Gradient-batched layout validator
// ---------------------------------------------------------------------------

/// Validate layout for `adjoint_state_gradient_batched`; return `n_cols`.
///
/// # Errors
/// `DomainViolation` if `u0_cols.len() != dj_cols.len()`,
/// not divisible by `n`, or `grad_len != n_params`.
pub(super) fn validate_adj_grad_layout<F: SemiflowFloat>(
    u0_cols: &[F],
    dj_cols: &[F],
    n: usize,
    grad_len: usize,
    n_params: usize,
) -> Result<usize, SemiflowError> {
    if u0_cols.len() != dj_cols.len() {
        return Err(SemiflowError::DomainViolation {
            what: "adjoint_state_gradient_batched: u0_cols.len() != dj_cols.len()",
            #[allow(clippy::cast_precision_loss)]
            value: u0_cols.len() as f64,
        });
    }
    if u0_cols.len() % n != 0 {
        return Err(SemiflowError::DomainViolation {
            what: "adjoint_state_gradient_batched: u0_cols.len() not divisible by n_nodes",
            #[allow(clippy::cast_precision_loss)]
            value: u0_cols.len() as f64,
        });
    }
    if grad_len != n_params {
        return Err(SemiflowError::DomainViolation {
            what: "adjoint_state_gradient_batched: grad_theta.len() != n_params()",
            #[allow(clippy::cast_precision_loss)]
            value: grad_len as f64,
        });
    }
    Ok(u0_cols.len() / n)
}

// ---------------------------------------------------------------------------
// Per-channel gradient accumulation helper
// ---------------------------------------------------------------------------

/// Compute per-channel gradient and accumulate into `grad_theta`.
///
/// `adjoint_state_gradient` zeroes its output buffer on entry; this helper
/// routes through a temporary then adds, preserving the 0-ULP identity vs
/// calling the single-channel function C times in ascending order.
#[allow(clippy::too_many_arguments)]
pub(in crate::graph_batched) fn accumulate_grad_channel<F, P>(
    mc: &MagnusGraphHeatChernoff<F>,
    u0_c: &[F],
    dj_c: &[F],
    graph_arc: &Arc<Graph<F>>,
    n_steps: usize,
    tau: F,
    param_deriv: &P,
    n_params: usize,
    grad_theta: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    let u0_sig = GraphSignal::from_fn(Arc::clone(graph_arc), |i| u0_c[i as usize]);
    let dj_sig = GraphSignal::from_fn(Arc::clone(graph_arc), |i| dj_c[i as usize]);
    let mut tmp_grad = alloc::vec![F::zero(); n_params];
    adjoint_state_gradient(
        mc,
        &u0_sig,
        n_steps,
        tau,
        &dj_sig,
        param_deriv,
        &mut tmp_grad,
        scratch,
    )?;
    for k in 0..n_params {
        grad_theta[k] += tmp_grad[k];
    }
    Ok(())
}
