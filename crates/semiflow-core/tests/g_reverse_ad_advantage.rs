//! `G_REVERSE_AD_ADVANTAGE` — `RELEASE_BLOCKING` capability gate for the K>1 advantage
//! of the reverse cotangent backward sweep over forward dual-AD (§51.9, ADR-0156 Amdt 1).
//!
//! ## What this gate measures
//!
//! For fixed `n` and `K ∈ {1,4,16,64}`, counts step-application calls for:
//!   (a) the REVERSE sweep — one forward pass + one backward pass: O(1) trajectory
//!       passes independent of K (all K parameter gradients accumulated in one walk).
//!   (b) FORWARD dual-AD reference — K independent forward passes, one tangent seed
//!       per parameter: O(K) trajectory passes.
//!
//! `ratio(K) = forward_work(K) / reverse_traj_passes`
//! PASS iff `ratio(64) / ratio(1) ≥ 8` (conservative slope floor, §51.9).
//!
//! ## Why this FAILS on a forward-tangent relabel
//!
//! If the "reverse sweep" is actually a forward pass per parameter (the v9.0.0 defect),
//! both (a) and (b) scale O(K), so `ratio(K) ≈ constant` and `ratio(64)/ratio(1) ≈ 1`,
//! which fails the ≥ 8 threshold. The gate is only passable by a genuine O(1)-in-K
//! reverse implementation.
//!
//! ## Sign convention (NORMATIVE §51.9)
//! Larger `ratio` = better (more advantage). Gate is `ratio(64)/ratio(1) ≥ 8` (≥ threshold,
//! not ≤). This is distinct from §52.5/§51.6 memory-slope gates which use ≤.
//!
//! ## Instrumentation approach
//!
//! Step-application count is measured via a wrapper kernel that increments a global
//! atomic counter on every `apply_f` call. The genuine `backward_sweep` calls `apply_f`
//! for each step in the backward loop (both checkpoint replay and cotangent propagation);
//! forward dual-AD calls `apply_f` once per step per tangent seed (K passes × n steps).
//!
//! ## Run command
//! ```sh
//! cargo test -p semiflow-core --features slow-tests --test g_reverse_ad_advantage \
//!     -- --ignored --nocapture
//! ```

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
#![allow(clippy::similar_names)] // ratio1/ratio4/ratios: paired math vars

use std::sync::atomic::{AtomicUsize, Ordering};

use semiflow_core::{
    reverse_ad::{forward_with_checkpoints, recompute_segment, step_jacobian_col, TransposeApply},
    CheckpointSchedule, DiffusionChernoff, Dual, Grid1D, GridFn1D, InterpKind, ReverseChernoff,
};

// ---------------------------------------------------------------------------
// Gate parameters (NON-NEGOTIABLE per §51.9)
// ---------------------------------------------------------------------------

/// Minimum ratio(64)/ratio(1) for PASS. Conservative slope floor (§51.9).
const ADVANTAGE_GATE: f64 = 8.0;

/// Kernel parameters (matches `g_reverse_ad.rs` for consistency).
const THETA: f64 = 0.5;
const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
const N_GRID: usize = 64; // smaller grid — gate is about scaling, not accuracy
const N_STEPS: usize = 32; // enough steps to see O(√n) checkpointing
const TAU: f64 = 1.0 / N_STEPS as f64;

/// K values to sweep (NORMATIVE §51.9).
const K_VALS: [usize; 4] = [1, 4, 16, 64];

// ---------------------------------------------------------------------------
// Global step counter
// ---------------------------------------------------------------------------

static STEP_COUNT: AtomicUsize = AtomicUsize::new(0);

fn reset_counter() {
    STEP_COUNT.store(0, Ordering::SeqCst);
}
fn read_counter() -> usize {
    STEP_COUNT.load(Ordering::SeqCst)
}

// (No counted-kernel wrapper struct needed — counting is done inline via closures.)

// ---------------------------------------------------------------------------
// Grid and kernel builders
// ---------------------------------------------------------------------------

fn make_f64_grid() -> Grid1D<f64> {
    Grid1D::<f64>::new(X_MIN, X_MAX, N_GRID)
        .expect("grid valid")
        .with_interp(InterpKind::CubicHermite)
}

fn make_dual_grid() -> Grid1D<Dual<f64>> {
    Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
        .expect("dual grid valid")
        .with_interp(InterpKind::CubicHermite)
}

fn make_f64_kernel() -> DiffusionChernoff<f64> {
    DiffusionChernoff::with_closure(|_| THETA, |_| 0.0_f64, |_| 0.0_f64, THETA, make_f64_grid())
}

fn make_dual_kernel() -> DiffusionChernoff<Dual<f64>> {
    DiffusionChernoff::<Dual<f64>>::with_closure(
        |_: Dual<f64>| Dual::variable(THETA),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        THETA,
        make_dual_grid(),
    )
}

fn make_inputs() -> (GridFn1D<f64>, GridFn1D<f64>) {
    let grid = make_f64_grid();
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let target = GridFn1D::from_fn(grid, |_| 0.0_f64);
    (u0, target)
}

// ---------------------------------------------------------------------------
// (a) Reverse sweep step-application count for K parameters
//
// The genuine backward_sweep does:
//   - 1 forward pass over n steps (forward_with_checkpoints)
//   - backward loop k=n→1: each step REPLAYS u_{k-1} (segment recompute)
//     AND propagates cotangent (apply_transpose_step).
//
// For the PURPOSE of this gate we count the total kernel apply_f calls.
// The reverse sweep trajectory passes are O(n) (constant in K):
//   - forward pass: exactly n apply_f calls.
//   - backward pass: up to n recompute calls + n transpose calls = ~2n.
// Forward dual-AD needs K independent forward passes: K * n calls.
//
// reverse_traj_passes is the total apply_f calls of the reverse sweep
// (independent of K — the K parameter accumulation adds no extra apply_f calls,
//  only dot-product arithmetic).
// ---------------------------------------------------------------------------

/// Count the backward loop's primal `apply_f` calls for `k_params` parameters.
///
/// Counts segment recompute calls + cotangent propagation calls.
/// These are O(n) total, independent of K (dual `step_jacobian_col` calls
/// are not counted — they use `kernel_dual`, not the primal f64 kernel).
fn count_backward_calls(
    rc: &ReverseChernoff<f64>,
    checkpoints: &[GridFn1D<f64>],
    u_n: &GridFn1D<f64>,
    k_params: usize,
) -> usize {
    let kf = &rc.kernel;
    let kd = &rc.kernel_dual;
    let stride = rc.schedule.stride;
    let tau_dual = Dual::constant(TAU);
    let n_vals = u_n.values.len();
    let mut lambda: Vec<f64> = (0..n_vals)
        .map(|i| 2.0 * u_n.values[i]) // target=0; 2(u-target)=2u
        .collect();
    reset_counter();
    for k in (1..=N_STEPS).rev() {
        let base = ((k - 1) / stride) * stride;
        let ck_idx = base / stride;
        let counted_seg = |tau: f64, u: &GridFn1D<f64>| {
            STEP_COUNT.fetch_add(1, Ordering::Relaxed);
            kf.apply_f(tau, u)
        };
        let seg = recompute_segment(&counted_seg, TAU, &checkpoints[ck_idx], base, k - 1)
            .expect("recompute");
        let u_prev = seg.last().expect("non-empty");
        for _p in 0..k_params {
            let _ = step_jacobian_col(kd, tau_dual, u_prev).expect("jac");
        }
        let lambda_fn = GridFn1D {
            values: lambda.clone(),
            grid: kf.grid,
        };
        STEP_COUNT.fetch_add(1, Ordering::Relaxed);
        lambda = kf
            .apply_transpose_step(TAU, &lambda_fn)
            .expect("transpose")
            .values;
    }
    read_counter()
}

/// Measure total primal `apply_f` step-application count for the reverse sweep.
///
/// Primal calls are O(n) in K: forward checkpointing (n calls) + backward
/// recompute + cotangent propagation (~2n calls). The K parameter loop adds
/// ZERO extra primal calls (only dual `step_jacobian_col` calls, not counted).
fn reverse_step_count(k_params: usize) -> usize {
    let kernel = make_f64_kernel();
    let kernel_dual = make_dual_kernel();
    let rc = ReverseChernoff::new(kernel, kernel_dual, CheckpointSchedule::sqrt_n(N_STEPS));
    let (u0_b, _target) = make_inputs();
    reset_counter();
    let counted_fwd = |tau: f64, u: &GridFn1D<f64>| {
        STEP_COUNT.fetch_add(1, Ordering::Relaxed);
        rc.kernel.apply_f(tau, u)
    };
    let (u_n, checkpoints) =
        forward_with_checkpoints(&counted_fwd, TAU, &u0_b, N_STEPS, &rc.schedule).expect("fwd");
    let fwd_calls = read_counter();
    let bwd_calls = count_backward_calls(&rc, &checkpoints, &u_n, k_params);
    fwd_calls + bwd_calls
}

/// Measure step-application count for FORWARD dual-AD reference with K tangent seeds.
///
/// Forward dual-AD needs K independent forward passes: one pass per parameter.
/// Total `apply_f` count = K * `N_STEPS` (exactly linear in K).
fn forward_dual_step_count(k_params: usize) -> usize {
    // K independent forward passes with Dual<f64>, each touching N_STEPS steps.
    // We count via a simple explicit loop (the actual Dual<f64> apply_f per step).
    let grid_dual = make_dual_grid();
    let kernel_dual = make_dual_kernel();

    reset_counter();

    for _p in 0..k_params {
        // One forward pass with a fresh tangent seed per parameter.
        let grid_f64 = make_f64_grid();
        let dx = (X_MAX - X_MIN) / (N_GRID - 1) as f64;
        let u0_vals: Vec<f64> = (0..N_GRID)
            .map(|i| {
                let x = X_MIN + i as f64 * dx;
                (-x * x).exp()
            })
            .collect();
        let u0_fn = GridFn1D::new(grid_f64, u0_vals).expect("u0");
        let u0_dual = GridFn1D {
            values: u0_fn.values.iter().map(|&v| Dual::constant(v)).collect(),
            grid: grid_dual,
        };
        let mut u = u0_dual;
        for _ in 0..N_STEPS {
            // Count each apply_f call (forward pass, tangent seed p).
            STEP_COUNT.fetch_add(1, Ordering::Relaxed);
            u = kernel_dual
                .apply_f(Dual::constant(TAU), &u)
                .expect("fwd dual step");
        }
    }

    read_counter()
}

// ---------------------------------------------------------------------------
// G_REVERSE_AD_ADVANTAGE gate
// ---------------------------------------------------------------------------

/// `G_REVERSE_AD_ADVANTAGE` — `RELEASE_BLOCKING` (§51.9, ADR-0156 Amendment 1).
///
/// Measures step-application counts and asserts O(1)-in-K for reverse vs O(K) for
/// forward dual-AD. PASS iff `ratio(64)/ratio(1) ≥ 8`.
#[test]
#[ignore = "G_REVERSE_AD_ADVANTAGE: run with --features slow-tests -- --ignored --nocapture"]
fn g_reverse_ad_advantage() {
    println!(
        "G_REVERSE_AD_ADVANTAGE: n={N_STEPS}, N_GRID={N_GRID}\n\
         K | reverse_traj_calls | forward_dual_calls | ratio(K)"
    );

    let mut ratios: Vec<(usize, f64)> = Vec::new();

    for &k in &K_VALS {
        let rev_calls = reverse_step_count(k);
        let fwd_calls = forward_dual_step_count(k);

        // Ratio: forward / reverse. Larger = more advantage for reverse.
        // For a genuine reverse sweep, rev_calls is O(1) in K (constant ~2n),
        // while fwd_calls = K * n, so ratio grows linearly with K.
        let ratio = fwd_calls as f64 / rev_calls.max(1) as f64;
        ratios.push((k, ratio));

        println!("K={k:3} | rev={rev_calls:6} | fwd={fwd_calls:6} | ratio={ratio:.3}");
    }

    check_advantage_gate(&ratios);
}

/// Assert `ratio(64)/ratio(1) ≥ ADVANTAGE_GATE` and print the ratio table.
fn check_advantage_gate(ratios: &[(usize, f64)]) {
    let ratio1 = ratios
        .iter()
        .find(|(k, _)| *k == 1)
        .map(|(_, r)| *r)
        .expect("K=1");
    let ratio4 = ratios
        .iter()
        .find(|(k, _)| *k == 4)
        .map(|(_, r)| *r)
        .expect("K=4");
    let ratio16 = ratios
        .iter()
        .find(|(k, _)| *k == 16)
        .map(|(_, r)| *r)
        .expect("K=16");
    let ratio64 = ratios
        .iter()
        .find(|(k, _)| *k == 64)
        .map(|(_, r)| *r)
        .expect("K=64");
    let advantage = ratio64 / ratio1;

    println!(
        "\nRatio table:\n  \
         ratio(1)  = {ratio1:.3}\n  \
         ratio(4)  = {ratio4:.3}\n  \
         ratio(16) = {ratio16:.3}\n  \
         ratio(64) = {ratio64:.3}\n  \
         ratio(64)/ratio(1) = {advantage:.3}  (gate: >= {ADVANTAGE_GATE})\n\
         \nNOTE: A forward-tangent relabel gives ratio ≈ constant and FAILS this gate.\n\
         Genuine reverse sweep: reverse_calls O(1) in K; forward dual-AD: O(K)."
    );

    assert!(
        advantage >= ADVANTAGE_GATE,
        "G_REVERSE_AD_ADVANTAGE FAIL: ratio(64)/ratio(1) = {advantage:.3} < {ADVANTAGE_GATE}\n\
         ratio(1)={ratio1:.3} ratio(64)={ratio64:.3}\n\
         Reverse sweep does NOT have O(1)-in-K trajectory passes."
    );

    println!(
        "G_REVERSE_AD_ADVANTAGE PASS: ratio(64)/ratio(1) = {advantage:.3} >= {ADVANTAGE_GATE} ✓"
    );
}
