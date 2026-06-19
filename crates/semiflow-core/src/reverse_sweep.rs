//! Genuine cotangent backward sweep for `ReverseChernoff` (§51.9, ADR-0156 Amdt 2).
//!
//! Split rationale: additive sibling to keep `reverse_ad.rs` ≤500 lines (constitution,
//! no carve-out).  This module owns the genuine backward-sweep internals;
//! `reverse_ad` owns the public structs and the checkpoint forward pass.
//!
//! ## Algorithm (§51.9 — genuine reverse mode)
//!
//! 1. Forward pass with `⌈√n⌉` checkpoints; seed cotangent `λ_n = 2(u_n − target)`.
//! 2. For `k = n … 1` (STRICTLY DECREASING — reverse direction witness):
//!    a. Replay `u_{k-1}` via `recompute_segment` (bit-exact from nearest checkpoint).
//!    b. For each `p ∈ 0..K`: `∇J[p] += ⟨λ_k, b_k^{(p)}⟩`
//!       where `b_k^{(p)} = step_jacobian_col(θ_p-seeded dual kernel, Dual::constant(τ), u_{k-1})`.
//!    c. Propagate cotangent: `λ_{k-1} = apply_transpose_step(τ, λ_k)`.
//!       For constant-a `DiffusionChernoff`, `F^⊤ = F` (symmetric), so this
//!       delegates to `apply_f`.  The transport terms live ONLY in `b_k`, NOT in `Jᵀ`
//!       (§51.9 normative clarification: `F_θ` is linear in state ⇒ no transport in `∂F/∂u`).
//!
//! `apply_transpose_step` and `step_jacobian_col` are BOTH called on the public path
//! (NORMATIVE load-bearing requirement — §51.6 `G_REVERSE_AD_STRUCTURE`).

use alloc::vec::Vec;

use crate::{
    diffusion::DiffusionChernoff,
    dual::Dual,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid_fn::GridFn1D,
    reverse_ad::{recompute_segment, step_jacobian_col, CheckpointSchedule, TransposeApply},
};

// ---------------------------------------------------------------------------
// backward_sweep — genuine K-vector cotangent backward sweep (§51.9, Amdt 2)
// ---------------------------------------------------------------------------

/// Execute one backward step: accumulate gradient and propagate cotangent.
///
/// Step 2(b)+(c) of Algorithm §51.9 at position `k`:
/// - Accumulates `∇J[p] += ⟨λ_k, b_k^{(p)}⟩` for each `p ∈ 0..k_params`.
/// - Propagates `λ_{k-1} = F^⊤ λ_k` (LOAD-BEARING §51.6 oracle — transpose path).
#[allow(clippy::too_many_arguments)]
fn backward_step<F: SemiflowFloat>(
    kernel: &DiffusionChernoff<F>,
    kernel_dual: &DiffusionChernoff<Dual<F>>,
    tau: F,
    tau_dual: Dual<F>,
    u_prev: &GridFn1D<F>,
    lambda: &mut Vec<F>,
    grad: &mut [F],
    k_params: usize,
) -> Result<(), SemiflowError> {
    // (b) Accumulate gradient.
    for g in &mut grad[..k_params] {
        let b_k = step_jacobian_col(kernel_dual, tau_dual, u_prev)?;
        let dot: F = lambda
            .iter()
            .zip(b_k.iter())
            .fold(F::zero(), |acc, (&l, &b)| acc + l * b);
        *g += dot;
    }
    // (c) Propagate cotangent (LOAD-BEARING — §51.6 sub-check a).
    let lambda_fn = GridFn1D {
        values: lambda.clone(),
        grid: kernel.grid,
    };
    let lambda_next = kernel.apply_transpose_step(tau, &lambda_fn)?;
    *lambda = lambda_next.values;
    Ok(())
}

/// Compute `∂J/∂θ` for all `K = theta.len()` parameters via the genuine cotangent
/// backward sweep (§51.9, ADR-0156 Amendment 2).
///
/// Algorithm (backward cotangent, `k = n … 1`):
/// 1. Seed: `λ_n = 2(u_n − target)`.
/// 2. For `k = n…1` (strictly decreasing loop — reverse-mode direction witness):
///    a. Replay `u_{k-1}` from nearest checkpoint (bit-exact).
///    b. Accumulate: `∇J[p] += ⟨λ_k, b_k^{(p)}⟩` for all `p ∈ 0..K`.
///    c. Propagate: `λ_{k-1} = apply_transpose_step(τ, λ_k)`.
///
/// # Normative notes (§51.9)
///
/// `apply_transpose_step` IS called in step 2c — it is LOAD-BEARING (§51.6
/// `G_REVERSE_AD_STRUCTURE` sub-check a proves this by mutation oracle).
/// `step_jacobian_col` IS called in step 2b — also load-bearing.
/// Both are called on EVERY step of the backward loop, not just at the boundary.
///
/// Transport terms (`h₀(θ) = 2√(θτ)` sample-position dependency) live
/// ONLY in `b_k^{(p)}` (captured by `step_jacobian_col`), NOT in `Jᵀλ`.
///
/// # Errors
/// Propagates `SemiflowError` from kernel or recompute applications.
// 8 args: kernel pair, τ, u_n, target, checkpoints, schedule, k_params — all required.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backward_sweep<F: SemiflowFloat>(
    kernel: &DiffusionChernoff<F>,
    kernel_dual: &DiffusionChernoff<Dual<F>>,
    tau: F,
    u_n: &GridFn1D<F>,
    target: &GridFn1D<F>,
    checkpoints: &[GridFn1D<F>],
    schedule: &CheckpointSchedule,
    k_params: usize,
) -> Result<Vec<F>, SemiflowError> {
    let n = schedule.n_steps;
    let stride = schedule.stride;
    let n_vals = u_n.values.len();

    // ── 1. Seed cotangent λ_n = 2(u_n − target) ───────────────────────────
    let two = from_f64::<F>(2.0);
    let mut lambda: Vec<F> = (0..n_vals)
        .map(|i| two * (u_n.values[i] - target.values[i]))
        .collect();
    let mut grad = vec![F::zero(); k_params];

    let fwd = |s: F, u: &GridFn1D<F>| kernel.apply_f(s, u);
    let tau_dual = Dual::constant(tau);

    // ── 2. Backward loop k = n … 1 (STRICTLY DECREASING) ─────────────────
    for k in (1..=n).rev() {
        // (a) Replay u_{k-1} from nearest checkpoint.
        let base = ((k - 1) / stride) * stride;
        let ck_idx = base / stride;
        let seg = recompute_segment(&fwd, tau, &checkpoints[ck_idx], base, k - 1)?;
        let u_prev = seg.last().expect("recompute_segment is non-empty");
        backward_step(
            kernel,
            kernel_dual,
            tau,
            tau_dual,
            u_prev,
            &mut lambda,
            &mut grad,
            k_params,
        )?;
    }

    Ok(grad)
}
