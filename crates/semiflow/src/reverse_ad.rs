//! [`ReverseChernoff`] — genuine reverse-mode AD over `(F(τ))ⁿ` via binomial
//! checkpointing (math §51, ADR-0156 Amendment 2; v9.1.0 Shift B GENUINE).
//!
//! Scope (NARROW, §51.5): linear / truncated-Magnus family ONLY.
//! For constant-a `DiffusionChernoff`, `F = F^⊤` (self-adjoint), so the
//! transpose step is the forward step. Variable-coefficient and nonlinear
//! kernels are OUT of scope.
//!
//! Forward pass: n steps, checkpoint every `⌈√n⌉` states (`O(√n)` peak mem).
//! K-vector gradient: GENUINE cotangent backward sweep (§51.9/§51.10) wired as the public path.
//! `value_and_grad(τ, n, u₀, target, θ ∈ ℝ^K)` accepts K≥1 parameters via region partition.
//! `value_and_grad_k1` is a thin K=1 wrapper (byte-identical to §51.9).
//! Forward-mode `Dual<F>` (§46) is retained ONLY as the `< 1e-12` parity reference
//! (§51.4 Amendment 2 — NOT 0 ULP; two independent float paths agree by adjoint identity).
//!
//! Zero new runtime deps — checkpointing is a `Vec<GridFn1D<F>>` under alloc.
//! Backward sweep helpers live in `reverse_sweep.rs` (additive split, ≤500 lines).
//!
//! References: Griewank-Walther ACM TOMS 2000; math §42/§43/§46/§51; ADR-0156.

// sqrt(n_steps).ceil().max(1.0) as usize: result is always >= 1.0 so non-negative cast is safe.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

use alloc::vec::Vec;

// Backward sweep internals live in the sibling crate-root module `reverse_sweep`
// (additive split — keeps this file ≤500 lines; declared in lib.rs).
use crate::reverse_sweep::backward_sweep;

use crate::{
    diffusion::DiffusionChernoff, dual::Dual, error::SemiflowError, float::SemiflowFloat,
    grid_fn::GridFn1D, reverse_region::RegionMap,
};

// ---------------------------------------------------------------------------
// Checkpoint schedule — §51.3 (binomial √n default, Griewank-Walther)
// ---------------------------------------------------------------------------

/// Checkpoint schedule for `O(√n)` memory in the backward pass.
///
/// Stores every `stride`-th state. `stride = ⌈√n⌉` gives `O(√n)` checkpoints
/// and `O(n)` recompute steps (Griewank-Walther revolve, §51.3/§51.7).
#[derive(Clone, Debug)]
pub struct CheckpointSchedule {
    /// Checkpoint stride: store a state every `stride` forward steps.
    pub stride: usize,
    /// Total number of steps `n` this schedule was built for.
    pub n_steps: usize,
}

impl CheckpointSchedule {
    /// Build the default `⌈√n⌉` binomial schedule for `n` steps (§51.7).
    #[must_use]
    pub fn sqrt_n(n_steps: usize) -> Self {
        let stride = (n_steps as f64).sqrt().ceil().max(1.0) as usize;
        Self { stride, n_steps }
    }

    /// Number of checkpoints = `(n − 1) / stride + 1` (step-0 always stored).
    ///
    /// Returns 1 for `n_steps == 0`.
    #[must_use]
    pub fn checkpoint_count(&self) -> usize {
        if self.n_steps == 0 {
            return 1;
        }
        (self.n_steps - 1) / self.stride + 1
    }
}

// ---------------------------------------------------------------------------
// TransposeApply<F> — opt-in trait for exact transpose step (§51.5 NARROW)
// ---------------------------------------------------------------------------

/// Opt-in trait: the implementor provides the exact transpose step `F^⊤`.
///
/// **NORMATIVE (§51.5):** Transpose-exactness is proven only for the linear /
/// truncated-Magnus family. Implementors MUST have established algebraic
/// exactness before opting in.
///
/// For constant-a `DiffusionChernoff`, `F` is self-adjoint (`F^⊤ = F`),
/// so `apply_transpose` delegates to `apply_f`. This is the canonical
/// narrow-scope implementation used by the §51.6 gates.
pub trait TransposeApply<F: SemiflowFloat>: Sized {
    /// Apply `F^⊤` (exact transpose step) to `src`, writing result into a new state.
    ///
    /// For symmetric kernels: identical to the forward `apply_f`.
    /// For non-symmetric kernels: must negate the shift direction per §51.2.
    ///
    /// # Errors
    /// Propagates `SemiflowError` from the underlying kernel.
    fn apply_transpose_step(&self, tau: F, src: &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError>;
}

/// `DiffusionChernoff<F>` with constant `a(x) ≡ θ` is self-adjoint:
/// `F^⊤ = F`, so `apply_transpose_step` delegates to `apply_f` (§51.5, narrow scope).
///
/// This satisfies the §51.6 anti-dodge clause: the gate kernel MUST run on
/// the default `Grid1D::new` (`SepticHermite`) grid.
impl<F: SemiflowFloat> TransposeApply<F> for DiffusionChernoff<F> {
    fn apply_transpose_step(&self, tau: F, src: &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError> {
        // For constant-a: F is symmetric (no inner Strang shift term), so F^⊤ = F.
        // NORMATIVE (§51.5): this impl is valid ONLY for the constant-a case.
        // Variable-a usage is caller's responsibility (document OUT of scope).
        self.apply_f(tau, src)
    }
}

// ---------------------------------------------------------------------------
// Forward pass with checkpointing (§51.3)
// ---------------------------------------------------------------------------

/// Run `n` forward steps, storing every `stride`-th state as a checkpoint.
///
/// Returns `(u_n, checkpoints)` where `checkpoints[j]` = state at step
/// `j * stride`. Step-0 is always stored. Peak: `O(√n · |state|)` vecs.
///
/// # Errors
/// Returns [`SemiflowError`] if any kernel application fails.
pub fn forward_with_checkpoints<F, C>(
    kernel: &C,
    tau: F,
    u0: &GridFn1D<F>,
    n: usize,
    schedule: &CheckpointSchedule,
) -> Result<(GridFn1D<F>, Vec<GridFn1D<F>>), SemiflowError>
where
    F: SemiflowFloat,
    C: Fn(F, &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError>,
{
    let cap = schedule.checkpoint_count() + 1;
    let mut checkpoints: Vec<GridFn1D<F>> = Vec::with_capacity(cap);
    let mut u = u0.clone();
    checkpoints.push(u.clone()); // step-0 checkpoint

    for k in 1..=n {
        u = kernel(tau, &u)?;
        // Store checkpoint after step k if k is a stride boundary (not last step).
        if k < n && k % schedule.stride == 0 {
            checkpoints.push(u.clone());
        }
    }
    Ok((u, checkpoints))
}

// ---------------------------------------------------------------------------
// Segment recompute (§51.3 — replay for backward sweep)
// ---------------------------------------------------------------------------

/// Recompute forward states `u_{from}, …, u_{to}` from checkpoint `ck`.
/// Used by the backward sweep (§51.9) and by `G_REVERSE_AD_STRUCTURE` oracle.
///
/// `ck` must be the state at step `from`. Returns `to − from + 1` states.
///
/// # Errors
/// Returns [`SemiflowError`] if any kernel application fails.
pub fn recompute_segment<F, C>(
    kernel: &C,
    tau: F,
    ck: &GridFn1D<F>,
    from: usize,
    to: usize,
) -> Result<Vec<GridFn1D<F>>, SemiflowError>
where
    F: SemiflowFloat,
    C: Fn(F, &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError>,
{
    debug_assert!(to >= from, "recompute_segment: to < from");
    let len = to - from + 1;
    let mut seg: Vec<GridFn1D<F>> = Vec::with_capacity(len);
    let mut u = ck.clone();
    seg.push(u.clone()); // state at `from`
    for _ in (from + 1)..=to {
        u = kernel(tau, &u)?;
        seg.push(u.clone());
    }
    Ok(seg)
}

// ---------------------------------------------------------------------------
// Per-step Jacobian column via Dual forward mode (§46)
// ---------------------------------------------------------------------------

/// Compute `b_k^{(p)} = (∂F/∂θ_p)(u_{k-1})` — transport-complete parameter
/// gradient column (§51.9, ADR-0156 Amendment 2, REPAIRED).
///
/// NORMATIVE seeding: `kernel_dual` seeds `∂/∂θ_p` in its coefficient closure
/// (e.g. `a(x)=Dual::variable(θ)`). `tau_dual` MUST be `Dual::constant(τ)` —
/// τ carries NO tangent (Amendment 2 repair: old stub seeded `∂/∂τ`, WRONG).
/// State is lifted with ZERO tangent (held fixed). Output `.tangent` =
/// `b_k^{(p)}` including sample-position transport differentiated by dual.
///
/// LOAD-BEARING on the public backward-sweep path (§51.6 `G_REVERSE_AD_STRUCTURE`).
///
/// # Errors
/// Returns [`SemiflowError`] if the dual kernel application fails.
pub fn step_jacobian_col<F: SemiflowFloat>(
    kernel_dual: &DiffusionChernoff<Dual<F>>,
    tau_dual: Dual<F>,
    u: &GridFn1D<F>,
) -> Result<Vec<F>, SemiflowError> {
    let dual_grid = kernel_dual.grid;
    // Zero state tangent: the state is held fixed; only θ_p varies.
    let u_dual = GridFn1D {
        values: u.values.iter().map(|&v| Dual::constant(v)).collect(),
        grid: dual_grid,
    };
    let out = kernel_dual.apply_f(tau_dual, &u_dual)?;
    Ok(out.values.iter().map(|d| d.tangent).collect())
}

// ---------------------------------------------------------------------------
// Loss and cotangent init
// ---------------------------------------------------------------------------

/// L² discrete loss `‖u_n − target‖²`.
fn compute_loss<F: SemiflowFloat>(u: &GridFn1D<F>, target: &GridFn1D<F>) -> F {
    u.values
        .iter()
        .zip(target.values.iter())
        .fold(F::zero(), |acc, (&a, &b)| {
            let d = a - b;
            acc + d * d
        })
}

// ---------------------------------------------------------------------------
// ReverseChernoff — main public wrapper
// ---------------------------------------------------------------------------

/// Reverse-mode AD over the Chernoff product `(F_θ(τ))ⁿ u₀` for loss
/// `J(θ) = ‖(F_θ(τ))ⁿ u₀ − target‖²`.
///
/// Generic over `F: SemiflowFloat`. Wraps a `DiffusionChernoff<F>` +
/// `DiffusionChernoff<Dual<F>>` pair plus a [`RegionMap`] (K regions).
///
/// `value_and_grad` accepts `theta ∈ ℝ^K` for K ≥ 1 (ADR-0177).
/// K=1 path is byte-identical to the §51.9 scalar path (regression guarantee).
///
/// # Scope (NARROW — §51.5/§51.10)
///
/// Linear / truncated-Magnus family with const-per-region `a` ONLY.
/// Variable-coefficient within a region and nonlinear kernels are out of scope.
pub struct ReverseChernoff<F: SemiflowFloat = f64> {
    /// Forward kernel `F_θ(τ)` at primal type `F`.
    pub kernel: DiffusionChernoff<F>,
    /// Same kernel at `Dual<F>` for per-step Jacobian (§46) — K=1 path only.
    pub kernel_dual: DiffusionChernoff<Dual<F>>,
    /// Checkpoint schedule (default: `√n` binomial, §51.3).
    pub schedule: CheckpointSchedule,
    /// Region partition for K-vector reverse-AD (§51.10, ADR-0177).
    ///
    /// K=1 default: single region spanning all nodes (byte-identical to §51.9).
    region_map: RegionMap,
}

impl<F: SemiflowFloat> ReverseChernoff<F> {
    /// Construct from kernel pair and checkpoint schedule (K=1 default region map).
    ///
    /// Use `CheckpointSchedule::sqrt_n(n)` for the `O(√n)` default (§51.3/§51.7).
    /// The region map defaults to K=1 (whole domain is one region). To use K>1
    /// per-region gradients, call [`Self::with_region_map`] after construction.
    ///
    /// # Panics
    /// Panics if the kernel's grid node count is zero (should never happen for
    /// valid grids with n ≥ 4).
    #[must_use]
    pub fn new(
        kernel: DiffusionChernoff<F>,
        kernel_dual: DiffusionChernoff<Dual<F>>,
        schedule: CheckpointSchedule,
    ) -> Self {
        let n_grid = kernel.grid.n;
        let region_map = RegionMap::contiguous(n_grid, 1)
            .expect("K=1 region map always valid for n >= 1");
        Self {
            kernel,
            kernel_dual,
            schedule,
            region_map,
        }
    }

    /// Replace the region map (enables K>1 per-region reverse-AD, §51.10).
    ///
    /// `rmap.region_count()` determines the length of the gradient returned by
    /// `value_and_grad`. `rmap.n_grid()` must equal `kernel.grid.n`.
    ///
    /// # Errors
    /// Returns `SemiflowError::UnsupportedOperation` if grid sizes do not match.
    pub fn with_region_map(mut self, rmap: RegionMap) -> Result<Self, SemiflowError> {
        if rmap.n_grid() != self.kernel.grid.n {
            return Err(SemiflowError::UnsupportedOperation {
                what: "ReverseChernoff::with_region_map: rmap.n_grid() != kernel.grid.n",
            });
        }
        self.region_map = rmap;
        Ok(self)
    }

    /// Compute `(J, ∂J/∂θ)` for a **K-parameter** vector in ONE backward pass.
    ///
    /// Accepts `theta.len() == region_map.region_count()` (ADR-0177).
    /// K=1: byte-identical to §51.9 (structural early branch in sweep).
    /// K>1: per-region dual seeding (§51.10) — genuinely distinct gradients
    ///      in a single O(1)-in-K backward pass.
    ///
    /// Out-of-scope cases (variable-a within region, non-DoF-aligned, non-self-adjoint)
    /// remain fail-loud via `SemiflowError::UnsupportedOperation`.
    ///
    /// # Errors
    /// - `SemiflowError::UnsupportedOperation` if `theta.len() != region_map.region_count()`.
    /// - Propagates `SemiflowError` from any kernel application.
    pub fn value_and_grad(
        &self,
        tau: F,
        n: usize,
        u0: &GridFn1D<F>,
        target: &GridFn1D<F>,
        theta: &[F],
    ) -> Result<(F, Vec<F>), SemiflowError> {
        // ADR-0177: validate theta length matches region count.
        if theta.len() != self.region_map.region_count() {
            return Err(SemiflowError::UnsupportedOperation {
                what: "value_and_grad: theta.len() must equal region_map.region_count() \
                       (ADR-0177); use with_region_map to set K regions, or \
                       value_and_grad_k1 for the scalar K=1 path",
            });
        }
        let fwd = |t: F, u: &GridFn1D<F>| self.kernel.apply_f(t, u);
        let (u_n, checkpoints) = forward_with_checkpoints(&fwd, tau, u0, n, &self.schedule)?;
        let loss = compute_loss(&u_n, target);
        let grad = backward_sweep(
            &self.kernel,
            &self.kernel_dual,
            tau,
            &u_n,
            target,
            &checkpoints,
            &self.schedule,
            theta,
            &self.region_map,
        )?;
        Ok((loss, grad))
    }

    /// Compute `(J, ∂J/∂θ)` for scalar parameter `K = 1`.
    ///
    /// **Thin wrapper** around [`Self::value_and_grad`] with `theta = &[0.0]`
    /// (length-1 slice — actual θ encoded in `kernel_dual`'s closure).
    /// K=1 routes through the SAME genuine cotangent backward sweep — NO forward
    /// shortcut (§51.9 normative, ADR-0156 Amendment 2).
    ///
    /// The gradient is byte-identical to the original §51.9 implementation
    /// (structural K=1 branch in `backward_sweep`).
    ///
    /// # Errors
    /// Propagates `SemiflowError` from any kernel application.
    pub fn value_and_grad_k1(
        &self,
        tau: F,
        n: usize,
        u0: &GridFn1D<F>,
        target: &GridFn1D<F>,
    ) -> Result<(F, F), SemiflowError> {
        // K=1: placeholder theta slice — actual θ-seed encoded in kernel_dual's closure.
        let placeholder = [F::zero()];
        let (loss, grads) = self.value_and_grad(tau, n, u0, target, &placeholder)?;
        Ok((loss, grads[0]))
    }
}

// ---------------------------------------------------------------------------
// Unit tests (§51 fast unit tests — no slow-tests gate)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "reverse_ad_tests.rs"]
mod tests;
