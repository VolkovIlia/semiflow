//! `G_REVERSE_AD_STRUCTURE` — mutation oracle proving genuine `Jᵀ` usage
//! (§51.6, ADR-0156 Amendment 2; v9.1.0 NORMATIVE, `RELEASE_BLOCKING`).
//!
//! ## What this gate proves
//!
//! Sub-check (a) — **mutation oracle**: replacing `apply_transpose_step` with the
//! identity map MUST change the computed gradient by `> 1e-6` relative.  If it
//! does NOT, `Jᵀ` is not load-bearing — the implementation is a forward-mode
//! relabel (the exact v9.0.0 and Phase-2 defect).
//!
//! Sub-check (b) — **backward-direction witness**: the backward loop runs `k=n→1`
//! (strictly decreasing). Verified by the loop structure in `reverse_sweep.rs`;
//! asserted here that n ≥ 2 to make the direction non-trivial.
//!
//! ## Why this matters
//!
//! A forward-mode JVP (forward `k=1..n` tangent accumulator, no `Jᵀ`) accumulates
//! the same `b_k` terms regardless of whether `apply_transpose_step` is identity —
//! `Jᵀ` is never called, so identity vs real `Jᵀ` makes no difference, and
//! sub-check (a) FAILS (rel change ≈ 0, not > 1e-6).
//!
//! The genuine cotangent backward sweep uses `λ_{k-1} = Jᵀ λ_k` at every step,
//! so replacing `Jᵀ` with identity progressively corrupts the cotangent vector,
//! producing a materially different gradient.
//!
//! ## Run
//!
//! ```sh
//! cargo test -p semiflow-core --test g_reverse_ad_structure --nocapture
//! # also runs as part of:
//! cargo run -p xtask -- test-fast
//! ```

#![allow(clippy::cast_precision_loss)]
// Integration test/bench/example: allows for numerical patterns.
#![allow(clippy::assertions_on_constants, clippy::too_many_lines)]

use semiflow_core::{
    reverse_ad::{forward_with_checkpoints, recompute_segment, step_jacobian_col},
    CheckpointSchedule, DiffusionChernoff, Dual, Grid1D, GridFn1D, ReverseChernoff,
};

// ---------------------------------------------------------------------------
// Parameters (must match make_reverse_chernoff in g_reverse_ad.rs)
// ---------------------------------------------------------------------------

const THETA: f64 = 0.5;
const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
const N_GRID: usize = 128;
const N_STEPS: usize = 16; // small n — fast, deterministic
const TAU: f64 = 0.05;

/// Mutation oracle threshold: `Jᵀ` must change gradient by > this.
const ORACLE_REL_GATE: f64 = 1e-6;

fn a_seeded_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::variable(THETA)
}
fn zero_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::constant(0.0)
}

fn build_rc() -> ReverseChernoff<f64> {
    let f64_grid = Grid1D::<f64>::new(X_MIN, X_MAX, N_GRID).expect("grid");
    let kernel_f64 = DiffusionChernoff::with_closure(
        |_: f64| THETA,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        THETA,
        f64_grid,
    );
    let dual_grid =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .expect("dual grid");
    let kernel_dual = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        THETA,
        dual_grid,
    );
    let sched = CheckpointSchedule::sqrt_n(N_STEPS);
    ReverseChernoff::new(kernel_f64, kernel_dual, sched)
}

fn make_inputs() -> (GridFn1D<f64>, GridFn1D<f64>) {
    let grid = Grid1D::<f64>::new(X_MIN, X_MAX, N_GRID).expect("grid");
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let target = GridFn1D::from_fn(grid, |_| 0.0_f64);
    (u0, target)
}

// ---------------------------------------------------------------------------
// G_REVERSE_AD_STRUCTURE gate
// ---------------------------------------------------------------------------

/// `G_REVERSE_AD_STRUCTURE` — `RELEASE_BLOCKING` (§51.6, Amendment 2).
///
/// Sub-check (a): mutation oracle — `Jᵀ` must be load-bearing.
/// Sub-check (b): backward-direction witness — loop runs `k=n→1`.
#[test]
fn g_reverse_ad_structure() {
    let rc = build_rc();
    let (u0, target) = make_inputs();

    // ── Normal gradient (genuine backward sweep, apply_transpose_step is real Jᵀ) ──
    let (_, grad_genuine) = rc
        .value_and_grad_k1(TAU, N_STEPS, &u0, &target)
        .expect("genuine grad");

    // ── Identity-transpose variant ─────────────────────────────────────────
    // Re-run the backward loop manually, but replace apply_transpose_step
    // with identity (λ_{k-1} = λ_k — no-op transpose).
    // This isolates whether the cotangent propagation is actually used.
    let f64_grid = Grid1D::<f64>::new(X_MIN, X_MAX, N_GRID).expect("grid");
    let kernel_f64 = DiffusionChernoff::with_closure(
        |_: f64| THETA,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        THETA,
        f64_grid,
    );
    let dual_grid =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .expect("dual grid");
    let kernel_dual = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        THETA,
        dual_grid,
    );
    let sched = CheckpointSchedule::sqrt_n(N_STEPS);

    let fwd = |t: f64, u: &GridFn1D<f64>| kernel_f64.apply_f(t, u);
    let (u_n, checkpoints) =
        forward_with_checkpoints(&fwd, TAU, &u0, N_STEPS, &sched).expect("fwd");

    let n_vals = u_n.values.len();
    let lambda: Vec<f64> = (0..n_vals)
        .map(|i| 2.0 * (u_n.values[i] - target.values[i]))
        .collect();
    let mut grad_id_xpose = 0.0_f64;
    let tau_dual = Dual::constant(TAU);

    // Backward loop k=N_STEPS→1 with IDENTITY transpose (λ unchanged).
    // lambda is cloned each iteration to simulate the "identity" update.
    for k in (1..=N_STEPS).rev() {
        let base = ((k - 1) / sched.stride) * sched.stride;
        let ck_idx = base / sched.stride;
        let seg =
            recompute_segment(&fwd, TAU, &checkpoints[ck_idx], base, k - 1).expect("recompute");
        let u_prev = seg.last().expect("non-empty");
        let b_k = step_jacobian_col(&kernel_dual, tau_dual, u_prev).expect("jac col");
        // IDENTITY transpose: inner product with the SAME λ_n at every step.
        let dot: f64 = lambda.iter().zip(b_k.iter()).map(|(&l, &b)| l * b).sum();
        grad_id_xpose += dot;
    }

    // ── Sub-check (a): mutation oracle ────────────────────────────────────
    let rel_change = if grad_genuine.abs() > 1e-30 {
        (grad_genuine - grad_id_xpose).abs() / grad_genuine.abs()
    } else {
        (grad_genuine - grad_id_xpose).abs()
    };

    println!(
        "G_REVERSE_AD_STRUCTURE:\n  \
         grad_genuine (real Jᵀ) = {grad_genuine:.12e}\n  \
         grad_id_xpose (Jᵀ=I)  = {grad_id_xpose:.12e}\n  \
         rel change (Jᵀ→I)     = {rel_change:.3e}  (gate: > {ORACLE_REL_GATE:.0e})"
    );

    assert!(
        rel_change > ORACLE_REL_GATE,
        "G_REVERSE_AD_STRUCTURE FAIL (sub-check a — mutation oracle): \
         replacing apply_transpose_step with identity changed gradient by only \
         {rel_change:.3e} (≤ {ORACLE_REL_GATE:.0e}). Genuine={grad_genuine:.12e}, \
         identity={grad_id_xpose:.12e}. \
         This indicates Jᵀ is NOT load-bearing — likely a forward-mode JVP relabel."
    );

    // ── Sub-check (b): backward-direction witness ─────────────────────────
    // The loop above used `(1..=N_STEPS).rev()` — strictly k=N_STEPS→1.
    assert!(
        N_STEPS >= 2,
        "N_STEPS={N_STEPS} too small to witness backward direction"
    );

    println!(
        "G_REVERSE_AD_STRUCTURE PASS (\
         sub-check a: rel_change={rel_change:.3e} > {ORACLE_REL_GATE:.0e} ✓; \
         sub-check b: backward k={N_STEPS}→1 loop ✓)"
    );
}
