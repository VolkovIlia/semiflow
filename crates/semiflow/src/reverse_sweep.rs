//! Genuine cotangent backward sweep for `ReverseChernoff` (§51.9, ADR-0156 Amdt 2).
//! K>1 extension via per-region dual seeding (§51.10, ADR-0177, issue #1).
//!
//! Split rationale: additive sibling to keep `reverse_ad.rs` ≤500 lines (constitution,
//! no carve-out).  This module owns the genuine backward-sweep internals;
//! `reverse_ad` owns the public structs and the checkpoint forward pass.
//!
//! ## Algorithm (§51.9 — genuine reverse mode, generalised to K regions)
//!
//! 1. Forward pass with `⌈√n⌉` checkpoints; seed cotangent `λ_n = 2(u_n − target)`.
//! 2. For `k = n … 1` (STRICTLY DECREASING — reverse direction witness):
//!    a. Replay `u_{k-1}` via `recompute_segment` (bit-exact from nearest checkpoint).
//!    b. For each `r ∈ 0..K`: `grad[r] += ⟨λ_k, b_k^{(r)}⟩`
//!       (K=1: byte-identical §51.9 path via single-seed `kernel_dual`;
//!        K>1: per-region `step_jacobian_col_region`.)
//!    c. Propagate cotangent: `λ_{k-1} = F^⊤ λ_k`.
//!       K=1: `apply_transpose_step` (F = F^⊤ for constant-a, §51.5).
//!       K>1: matrix-explicit F^⊤ via unit-vector probing (§51.10 ADR-0177 Amdt 1).
//!            F is time-homogeneous; F^⊤ is built ONCE per `backward_sweep` call.

use alloc::{sync::Arc, vec::Vec};

use crate::{
    diffusion::DiffusionChernoff,
    dual::Dual,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid::Grid1D,
    grid_fn::GridFn1D,
    reverse_ad::{recompute_segment, step_jacobian_col, CheckpointSchedule, TransposeApply},
    reverse_region::RegionMap,
};

// ---------------------------------------------------------------------------
// Dual grid builder (mirrors pattern in g_reverse_ad.rs)
// ---------------------------------------------------------------------------

/// Build a `Grid1D<Dual<F>>` matching the geometry of a `Grid1D<F>`.
fn dual_grid<F: SemiflowFloat>(g: &Grid1D<F>) -> Grid1D<Dual<F>> {
    Grid1D::<Dual<F>>::new_generic(Dual::constant(g.xmin), Dual::constant(g.xmax), g.n)
        .expect("dual grid from valid primal grid")
        .with_interp(g.interp)
}

// ---------------------------------------------------------------------------
// Per-region Jacobian column (§51.10 — per-region dual seeding)
// ---------------------------------------------------------------------------

/// Compute `b_k^{(r)} = (∂F/∂θ_r)(u_{k-1})` for region `r`.
///
/// Per §51.10: builds a fresh `DiffusionChernoff<Dual<F>>` whose coefficient
/// closure returns `Dual::variable(θ_r)` on nodes `i ∈ Ω_r` (x near node i)
/// and `Dual::constant(θ_{ρ(i)})` elsewhere.
///
/// Const-per-region means `a'(x) = 0`, so sample positions `x_pre = x_i`
/// (no inner shift). The closure maps any queried x to the nearest grid node
/// to recover the region assignment.
///
/// # Errors
/// Returns `SemiflowError` if the dual kernel application fails.
/// Build per-node lookup tables for region dual seeding.
///
/// Returns `(is_in_r, theta_by_region)` where:
/// - `is_in_r[i]` = true if node i belongs to `region`
/// - `theta_by_region[i]` = θ_{ρ(i)} (primal value at node i)
fn region_lookup_tables<F: SemiflowFloat>(
    theta: &[F],
    rmap: &RegionMap,
    region: usize,
) -> (Arc<Vec<bool>>, Arc<Vec<F>>) {
    let n = rmap.n_grid();
    let is_in_r = Arc::new((0..n).map(|i| rmap.region_of(i) == region).collect());
    let tbr = Arc::new((0..n).map(|i| theta[rmap.region_of(i)]).collect());
    (is_in_r, tbr)
}

fn step_jacobian_col_region<F: SemiflowFloat>(
    primal: &DiffusionChernoff<F>,
    tau: F,
    u: &GridFn1D<F>,
    theta: &[F],
    rmap: &RegionMap,
    region: usize,
) -> Result<Vec<F>, SemiflowError> {
    let grid = primal.grid;
    let n_nodes = grid.n;
    let theta_r: F = theta[region];
    let (is_in_r_a, tbr_a) = region_lookup_tables(theta, rmap, region);

    // Grid geometry for node recovery inside the closure.
    let xmin_f64 = grid.xmin.to_f64().unwrap_or(f64::NEG_INFINITY);
    let xmax_f64 = grid.xmax.to_f64().unwrap_or(f64::INFINITY);
    let n_minus1 = (n_nodes - 1).max(1);
    // cast_precision_loss: n_minus1 < 2^52 in practice (grid size ≤ billions).
    #[allow(clippy::cast_precision_loss)]
    let dx_f64 = (xmax_f64 - xmin_f64) / n_minus1 as f64;

    // Build per-region dual kernel. a'=a''=0 for const-per-region (§51.10 scope).
    let kernel_dual = DiffusionChernoff::<Dual<F>>::with_closure(
        move |x: Dual<F>| {
            let xv = x.value.to_f64().unwrap_or(xmin_f64);
            // cast_possible_truncation: `.round()` bounds the f64; clamp below guards sign.
            // cast_sign_loss: clamp(0, …) ensures non-negative before usize cast.
            // cast_possible_wrap: n_minus1 < isize::MAX (grid size limited in practice).
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                clippy::cast_possible_wrap
            )]
            let raw = ((xv - xmin_f64) / dx_f64).round() as isize;
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
            let idx = raw.clamp(0, n_minus1 as isize) as usize;
            if is_in_r_a[idx] {
                Dual::variable(theta_r)
            } else {
                Dual::constant(tbr_a[idx])
            }
        },
        |_: Dual<F>| Dual::constant(F::zero()),
        |_: Dual<F>| Dual::constant(F::zero()),
        primal.a_norm_bound,
        dual_grid(&primal.grid),
    );
    step_jacobian_col(&kernel_dual, Dual::constant(tau), u)
}

// ---------------------------------------------------------------------------
// backward_step_k1 — original K=1 path (byte-identical to §51.9)
// ---------------------------------------------------------------------------

/// One backward step for K=1: accumulate gradient + propagate cotangent.
///
/// Byte-identical to the §51.9 path — uses the same `kernel_dual` and
/// `step_jacobian_col` as the original implementation. LOAD-BEARING (§51.6).
#[allow(clippy::too_many_arguments)]
fn backward_step_k1<F: SemiflowFloat>(
    kernel: &DiffusionChernoff<F>,
    kernel_dual: &DiffusionChernoff<Dual<F>>,
    tau: F,
    tau_dual: Dual<F>,
    u_prev: &GridFn1D<F>,
    lambda: &mut Vec<F>,
    grad: &mut [F],
) -> Result<(), SemiflowError> {
    // (b) Accumulate K=1 gradient: ∇J[0] += ⟨λ_k, b_k^{(0)}⟩.
    let b_k = step_jacobian_col(kernel_dual, tau_dual, u_prev)?;
    let dot: F = lambda
        .iter()
        .zip(b_k.iter())
        .fold(F::zero(), |acc, (&l, &b)| acc + l * b);
    grad[0] += dot;
    // (c) Propagate cotangent (LOAD-BEARING — §51.6 sub-check a).
    let lambda_fn = GridFn1D {
        values: lambda.clone(),
        grid: kernel.grid,
    };
    let lambda_next = kernel.apply_transpose_step(tau, &lambda_fn)?;
    *lambda = lambda_next.values;
    Ok(())
}

// ---------------------------------------------------------------------------
// build_f_transpose — exact F^⊤ for piecewise-constant a (§51.10 Amdt 1)
// ---------------------------------------------------------------------------

/// Build the N×N transpose matrix of the GH step operator F via unit-vector probing.
///
/// For piecewise-constant a(x) with K>1 regions, F is NOT self-adjoint (non-Toeplitz).
/// This builds `F^T[j][i] = F[i][j]` by applying `F` to each unit basis vector `e_j`.
///
/// Result: `ft[j * n + i] = F[i][j]` stored row-major, where n = grid.n.
/// Cotangent propagation: `(F^⊤ λ)[j] = Σ_i ft[j * n + i] · λ_i`.
///
/// Cost: n `apply_f` calls. Built once per `backward_sweep` for K>1.
///
/// # Errors
/// Returns `SemiflowError` if any unit-vector kernel application fails.
fn build_f_transpose<F: SemiflowFloat>(
    kernel: &DiffusionChernoff<F>,
    tau: F,
) -> Result<Vec<F>, SemiflowError> {
    let n = kernel.grid.n;
    let mut ft = vec![F::zero(); n * n]; // ft[j * n + i] = F[i][j]
    let mut e_j = GridFn1D {
        values: vec![F::zero(); n],
        grid: kernel.grid,
    };
    for j in 0..n {
        // Set unit vector e_j.
        if j > 0 {
            e_j.values[j - 1] = F::zero();
        }
        e_j.values[j] = F::one();
        // F · e_j gives column j of F (= row j of F^T).
        let col_j = kernel.apply_f(tau, &e_j)?;
        for i in 0..n {
            // ft[j * n + i] = F[i][j] = col_j[i].
            ft[j * n + i] = col_j.values[i];
        }
    }
    // Reset last unit vector.
    e_j.values[n - 1] = F::zero();
    Ok(ft)
}

/// Apply precomputed F^⊤ (from `build_f_transpose`) to `lambda`, returning `F^⊤ λ`.
///
/// `ft[j * n + i] = F[i][j]`; `(F^⊤ λ)[j] = Σ_i ft[j * n + i] * λ[i]`.
fn apply_f_transpose<F: SemiflowFloat>(ft: &[F], lambda: &[F], n: usize) -> Vec<F> {
    (0..n)
        .map(|j| {
            let row = &ft[j * n..(j + 1) * n];
            row.iter()
                .zip(lambda.iter())
                .fold(F::zero(), |acc, (&ftji, &li)| acc + ftji * li)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// backward_step_kvec — K>1 per-region accumulation
// ---------------------------------------------------------------------------

/// One backward step for K>1: per-region accumulation + exact F^⊤ cotangent propagate.
///
/// For each `r ∈ 0..K`: `grad[r] += ⟨λ_k, b_k^{(r)}⟩` via per-region seeding.
/// Then propagates `λ_{k-1} = F^⊤ λ_k` using precomputed transpose matrix. (§51.10)
#[allow(clippy::too_many_arguments)]
fn backward_step_kvec<F: SemiflowFloat>(
    kernel: &DiffusionChernoff<F>,
    tau: F,
    u_prev: &GridFn1D<F>,
    lambda: &mut Vec<F>,
    grad: &mut [F],
    theta: &[F],
    rmap: &RegionMap,
    ft: &[F],
) -> Result<(), SemiflowError> {
    let k = rmap.region_count();
    let n = kernel.grid.n;
    // (b) Per-region gradient accumulation.
    // `r` is passed as region index to step_jacobian_col_region, so iter() is not applicable.
    #[allow(clippy::needless_range_loop)]
    for r in 0..k {
        let b_kr = step_jacobian_col_region(kernel, tau, u_prev, theta, rmap, r)?;
        let dot: F = lambda
            .iter()
            .zip(b_kr.iter())
            .fold(F::zero(), |acc, (&l, &b)| acc + l * b);
        grad[r] += dot;
    }
    // (c) Propagate cotangent via exact F^⊤ (LOAD-BEARING for K>1).
    *lambda = apply_f_transpose(ft, lambda, n);
    Ok(())
}

// ---------------------------------------------------------------------------
// backward_sweep — generalised K-vector cotangent backward sweep
// ---------------------------------------------------------------------------

/// Compute `∂J/∂θ` (length K) via the genuine cotangent backward sweep.
///
/// K=1: routes through the **byte-identical** §51.9 path (structural early branch).
/// K>1: routes through §51.10 per-region seeding.
///
/// # Errors
/// Propagates `SemiflowError` from kernel or recompute applications.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backward_sweep<F: SemiflowFloat>(
    kernel: &DiffusionChernoff<F>,
    kernel_dual: &DiffusionChernoff<Dual<F>>,
    tau: F,
    u_n: &GridFn1D<F>,
    target: &GridFn1D<F>,
    checkpoints: &[GridFn1D<F>],
    schedule: &CheckpointSchedule,
    theta: &[F],
    rmap: &RegionMap,
) -> Result<Vec<F>, SemiflowError> {
    let (mut lambda, mut grad, ft_kvec) = init_sweep_state(kernel, tau, u_n, target, rmap)?;
    let fwd = |s: F, u: &GridFn1D<F>| kernel.apply_f(s, u);
    let tau_dual = Dual::constant(tau);
    let n = schedule.n_steps;
    let stride = schedule.stride;
    let k = rmap.region_count();

    // ── 2. Backward loop k = n … 1 (STRICTLY DECREASING) ─────────────────
    for step in (1..=n).rev() {
        let base = ((step - 1) / stride) * stride;
        let ck_idx = base / stride;
        let seg = recompute_segment(&fwd, tau, &checkpoints[ck_idx], base, step - 1)?;
        let u_prev = seg.last().expect("recompute_segment is non-empty");

        if k == 1 {
            // K=1 BYTE-IDENTICAL PATH (§51.9 — structural early branch).
            backward_step_k1(
                kernel,
                kernel_dual,
                tau,
                tau_dual,
                u_prev,
                &mut lambda,
                &mut grad,
            )?;
        } else {
            // K>1 PER-REGION PATH (§51.10) with exact F^⊤.
            let ft = ft_kvec.as_deref().expect("ft built for K>1");
            backward_step_kvec(kernel, tau, u_prev, &mut lambda, &mut grad, theta, rmap, ft)?;
        }
    }

    Ok(grad)
}

/// Initialise cotangent vector, gradient accumulator, and optional F^⊤ matrix.
///
/// Returns `(lambda, grad, ft_kvec)`. `ft_kvec` is `Some` only when K>1.
///
/// # Errors
/// Returns `SemiflowError` if F^⊤ construction fails.
// (Vec<F>, Vec<F>, Option<Vec<F>>) = (lambda, grad, ft_kvec) — 3-tuple is simpler than a struct.
#[allow(clippy::type_complexity)]
fn init_sweep_state<F: SemiflowFloat>(
    kernel: &DiffusionChernoff<F>,
    tau: F,
    u_n: &GridFn1D<F>,
    target: &GridFn1D<F>,
    rmap: &RegionMap,
) -> Result<(Vec<F>, Vec<F>, Option<Vec<F>>), SemiflowError> {
    let two = from_f64::<F>(2.0);
    let n_vals = u_n.values.len();
    let k = rmap.region_count();
    // Seed cotangent λ_n = 2(u_n − target).
    let lambda: Vec<F> = (0..n_vals)
        .map(|i| two * (u_n.values[i] - target.values[i]))
        .collect();
    let grad = vec![F::zero(); k];
    // For K>1: build exact F^⊤ via unit-vector probing (once per sweep).
    // F is time-homogeneous so it is reused across all n steps.
    let ft_kvec = if k > 1 {
        Some(build_f_transpose(kernel, tau)?)
    } else {
        None
    };
    Ok((lambda, grad, ft_kvec))
}
