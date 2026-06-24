//! `G_REVERSE_AD_GRADIENT (K-vector)` — `RELEASE_BLOCKING` gate for K>1 per-region
//! gradients (§51.10, ADR-0177, issue #1).
//!
//! Isolated in its own binary so the peak-tracking global allocator in
//! `g_reverse_ad.rs` (which measures O(√n) memory scaling) is not disturbed
//! by the `build_f_transpose` N×N matrix allocation that K>1 paths require.
//! Every file in `tests/` compiles into a separate binary; isolation here is
//! structural (compiler-enforced), not `#[serial]` or `--test-threads=1`.
//!
//! ## Gate
//!
//! For each region r ∈ 0..K:
//!   `|grad[r] − fd_r| / |fd_r| < 1e-9`
//!
//! Uses `N_GRID=128`, K=4, contiguous regions, distinct `θ_r`.
//! Initial condition: Lorentzian `u₀(x) = 1 / (1 + (x/5)²)`.
//!
//! ## Run
//!
//! ```sh
//! cargo test -p semiflow-core --features slow-tests --test g_reverse_ad_kvector \
//!     -- --ignored --nocapture
//! ```

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]

use semiflow::{
    CheckpointSchedule, DiffusionChernoff, Dual, Grid1D, GridFn1D, RegionMap, ReverseChernoff,
};

// ---------------------------------------------------------------------------
// Gate constants (mirrors g_reverse_ad.rs)
// ---------------------------------------------------------------------------

const GRAD_REL_GATE: f64 = 1e-9;
const T_FINAL: f64 = 1.0;
const N_STEPS_GRAD: usize = 32;
const N_GRID: usize = 128;
const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
const FD_H: f64 = 1e-3;

// ---------------------------------------------------------------------------
// Dual coefficient functions
// ---------------------------------------------------------------------------

fn a_seeded_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::variable(0.5)
}
fn zero_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::constant(0.0)
}

// ---------------------------------------------------------------------------
// Grid builders
// ---------------------------------------------------------------------------

fn f64_default_grid() -> Grid1D<f64> {
    Grid1D::new(X_MIN, X_MAX, N_GRID).expect("grid valid")
}
fn dual_default_grid() -> Grid1D<Dual<f64>> {
    Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
        .expect("grid valid")
}

// ---------------------------------------------------------------------------
// K=1 gradient (needed for cross-check test)
// ---------------------------------------------------------------------------

fn reverse_mode_grad_k1(tau: f64, n: usize) -> f64 {
    let f64_grid = f64_default_grid();
    let kernel_f64 =
        DiffusionChernoff::with_closure(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5_f64, f64_grid);
    let dual_grid = dual_default_grid();
    let kernel_dual = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        0.5_f64,
        dual_grid,
    );
    let sched = CheckpointSchedule::sqrt_n(n);
    let rc = ReverseChernoff::new(kernel_f64, kernel_dual, sched);
    let grid = f64_default_grid();
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let target = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let (_, grad) = rc
        .value_and_grad_k1(tau, n, &u0, &target)
        .expect("K=1 grad");
    grad
}

// ---------------------------------------------------------------------------
// Map x-coordinate to nearest grid node index (uniform grid)
// ---------------------------------------------------------------------------

fn node_idx(x: f64, n: usize, xmin: f64, xmax: f64) -> usize {
    let dx = (xmax - xmin) / (n - 1) as f64;
    let raw = ((x - xmin) / dx).round() as isize;
    raw.clamp(0, (n - 1) as isize) as usize
}

// ---------------------------------------------------------------------------
// Loss evaluation helpers
// ---------------------------------------------------------------------------

/// Eval ‖(F_θ(τ))^n u₀‖² with Lorentzian IC u₀ = 1/(1+(x/5)²).
fn eval_loss_region_lorentz(thetas: &[f64], tau: f64, n: usize) -> f64 {
    let k = thetas.len();
    let grid = f64_default_grid();
    let rmap = RegionMap::contiguous(N_GRID, k).expect("valid region map");
    let thetas_v = thetas.to_vec();
    let rmap2 = rmap.clone();
    let kernel = DiffusionChernoff::with_closure(
        move |x: f64| thetas_v[rmap2.region_of(node_idx(x, N_GRID, X_MIN, X_MAX))],
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        thetas.iter().cloned().fold(0.0_f64, f64::max),
        grid,
    );
    let u0 = GridFn1D::from_fn(grid, |x| 1.0 / (1.0 + (x / 5.0) * (x / 5.0)));
    let mut u = u0;
    for _ in 0..n {
        u = kernel.apply_f(tau, &u).expect("step");
    }
    u.values.iter().map(|v| v * v).sum()
}

/// Eval ‖(F_θ(τ))^n u₀‖² with Gaussian IC u₀ = exp(−x²).
fn eval_loss_region(thetas: &[f64], tau: f64, n: usize) -> f64 {
    let k = thetas.len();
    let grid = f64_default_grid();
    let rmap = RegionMap::contiguous(N_GRID, k).expect("valid region map");
    let thetas_v = thetas.to_vec();
    let rmap2 = rmap.clone();
    let kernel = DiffusionChernoff::with_closure(
        move |x: f64| thetas_v[rmap2.region_of(node_idx(x, N_GRID, X_MIN, X_MAX))],
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        thetas.iter().cloned().fold(0.0_f64, f64::max),
        grid,
    );
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let mut u = u0;
    for _ in 0..n {
        u = kernel.apply_f(tau, &u).expect("step");
    }
    u.values.iter().map(|v| v * v).sum()
}

// ---------------------------------------------------------------------------
// Finite-difference gradient helpers
// ---------------------------------------------------------------------------

/// Richardson 4-point ∂J/∂θ_r for region r, Lorentzian IC (O(h⁴)).
fn fd_grad_region_lorentz(thetas: &[f64], perturb_r: usize, tau: f64, n: usize) -> f64 {
    let h = FD_H;
    let mut t_p2 = thetas.to_vec();
    let mut t_p1 = thetas.to_vec();
    let mut t_m1 = thetas.to_vec();
    let mut t_m2 = thetas.to_vec();
    t_p2[perturb_r] += 2.0 * h;
    t_p1[perturb_r] += h;
    t_m1[perturb_r] -= h;
    t_m2[perturb_r] -= 2.0 * h;
    let fp2 = eval_loss_region_lorentz(&t_p2, tau, n);
    let fp1 = eval_loss_region_lorentz(&t_p1, tau, n);
    let fm1 = eval_loss_region_lorentz(&t_m1, tau, n);
    let fm2 = eval_loss_region_lorentz(&t_m2, tau, n);
    (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h)
}

/// 2-point central FD, Gaussian IC (for n=1 diagnostic).
fn fd_grad_region(thetas: &[f64], perturb_r: usize, tau: f64, n: usize) -> f64 {
    let h = FD_H;
    let mut tp = thetas.to_vec();
    let mut tm = thetas.to_vec();
    tp[perturb_r] += h;
    tm[perturb_r] -= h;
    let loss_p = eval_loss_region(&tp, tau, n);
    let loss_m = eval_loss_region(&tm, tau, n);
    (loss_p - loss_m) / (2.0 * h)
}

// ---------------------------------------------------------------------------
// G_REVERSE_AD_GRADIENT (K-vector) — RELEASE_BLOCKING
// ---------------------------------------------------------------------------

/// `G_REVERSE_AD_GRADIENT (K-vector)` — RELEASE_BLOCKING (§51.10, ADR-0177).
///
/// For each region r: `|grad[r] − fd_r| / |fd_r| < 1e-9`.
/// Uses N_GRID=128, K=4, contiguous regions, distinct θ_r.
///
/// Initial condition: `u₀(x) = 1 / (1 + (x/5)²)` (Lorentzian with half-width 5).
/// Non-negligible across all 4 regions (avoids near-zero gradient for outer regions).
#[test]
#[ignore = "G_REVERSE_AD_GRADIENT (K-vector): run with --features slow-tests -- --ignored --nocapture"]
fn g_reverse_ad_gradient_kvector() {
    let k = 4_usize;
    let thetas = [0.3_f64, 0.5, 0.7, 0.4]; // distinct per region
    let tau = T_FINAL / N_STEPS_GRAD as f64;
    let grid = f64_default_grid();
    let rmap = RegionMap::contiguous(N_GRID, k).expect("region map");
    let rmap2 = rmap.clone();
    let thetas_arc: std::sync::Arc<[f64; 4]> = std::sync::Arc::new(thetas);
    let rmap3 = rmap.clone();
    let kernel = DiffusionChernoff::with_closure(
        move |x: f64| thetas_arc[rmap3.region_of(node_idx(x, N_GRID, X_MIN, X_MAX))],
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        0.7_f64, // max of thetas
        grid,
    );
    let dual_grid = dual_default_grid();
    let kernel_dual = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        0.7_f64,
        dual_grid,
    );
    let rc = ReverseChernoff::new(
        kernel,
        kernel_dual,
        CheckpointSchedule::sqrt_n(N_STEPS_GRAD),
    )
    .with_region_map(rmap2)
    .expect("region map size matches grid");
    let u0 = GridFn1D::from_fn(grid, |x| 1.0 / (1.0 + (x / 5.0) * (x / 5.0)));
    let target = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let theta_slice: &[f64] = &thetas;
    let (_, grads) = rc
        .value_and_grad(tau, N_STEPS_GRAD, &u0, &target, theta_slice)
        .expect("K-vector reverse AD");
    for r in 0..k {
        let fd = fd_grad_region_lorentz(theta_slice, r, tau, N_STEPS_GRAD);
        let rel_err = if fd.abs() > 1e-30 {
            (grads[r] - fd).abs() / fd.abs()
        } else {
            (grads[r] - fd).abs()
        };
        println!(
            "G_REVERSE_AD_GRADIENT (K-vector) r={r}: grad={:.10e} fd={:.10e} rel_err={rel_err:.3e}",
            grads[r], fd
        );
        assert!(
            rel_err < GRAD_REL_GATE,
            "G_REVERSE_AD_GRADIENT (K-vector) FAIL r={r}: rel_err={rel_err:.3e} >= {GRAD_REL_GATE:.0e}"
        );
    }
    println!("G_REVERSE_AD_GRADIENT (K-vector) PASS (K={k}, all rel_err < {GRAD_REL_GATE:.0e})");
}

// ---------------------------------------------------------------------------
// DIAGNOSTIC: K-vector gradient with n=1 step
// ---------------------------------------------------------------------------

/// DIAGNOSTIC: K-vector gradient with n=1 step (no cotangent propagation).
/// If this PASSES, issue is in cotangent propagation (apply_transpose_step).
/// If this FAILS, issue is in step_jacobian_col_region.
#[test]
#[ignore = "DIAGNOSTIC: run with --features slow-tests -- --ignored --nocapture"]
fn g_reverse_ad_gradient_kvector_n1() {
    let k = 4_usize;
    let thetas = [0.3_f64, 0.5, 0.7, 0.4];
    let tau = T_FINAL / N_STEPS_GRAD as f64;
    let grid = f64_default_grid();
    let rmap = RegionMap::contiguous(N_GRID, k).expect("region map");
    let rmap2 = rmap.clone();
    let thetas_arc: std::sync::Arc<[f64; 4]> = std::sync::Arc::new(thetas);
    let rmap3 = rmap.clone();
    let kernel = DiffusionChernoff::with_closure(
        move |x: f64| thetas_arc[rmap3.region_of(node_idx(x, N_GRID, X_MIN, X_MAX))],
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        0.7_f64,
        grid,
    );
    let kernel_dual = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        0.7_f64,
        dual_default_grid(),
    );
    // n=1: backward loop runs ONCE. No cotangent propagation used.
    let rc = ReverseChernoff::new(kernel, kernel_dual, CheckpointSchedule::sqrt_n(1))
        .with_region_map(rmap2)
        .expect("region map");
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let target = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let theta_slice: &[f64] = &thetas;
    let (_, grads) = rc
        .value_and_grad(tau, 1, &u0, &target, theta_slice)
        .expect("n=1 K-vector");
    for r in 0..k {
        let fd = fd_grad_region(theta_slice, r, tau, 1);
        let rel_err = if fd.abs() > 1e-30 {
            (grads[r] - fd).abs() / fd.abs()
        } else {
            (grads[r] - fd).abs()
        };
        println!(
            "n=1 K-vector r={r}: grad={:.10e} fd={:.10e} rel_err={rel_err:.3e}",
            grads[r], fd
        );
    }
}

// ---------------------------------------------------------------------------
// DIAGNOSTIC: K=2 uniform cross-check
// ---------------------------------------------------------------------------

/// DIAGNOSTIC: K=2 with θ_1=θ_2=θ (should match K=1 scalar reverse gradient).
/// If this PASSES (cross-mode parity), K>1 math is correct but FD is limited.
/// If this FAILS, there is a genuine error in K>1 reverse sweep.
#[test]
#[ignore = "DIAGNOSTIC: run with --features slow-tests -- --ignored --nocapture"]
fn g_reverse_ad_gradient_k2_uniform_cross_check() {
    // Both regions with same θ: K>1 gradient should equal 2 × K=1 gradient.
    // (sum of ∂J/∂θ_r for r=0..2 = total ∂J/∂θ for uniform θ).
    let theta_uniform = 0.5_f64;
    let thetas_k2 = [theta_uniform, theta_uniform];
    let tau = T_FINAL / N_STEPS_GRAD as f64;
    let grid = f64_default_grid();
    let rmap = RegionMap::contiguous(N_GRID, 2).expect("region map");
    let rmap2 = rmap.clone();
    let thetas_arc = std::sync::Arc::new(thetas_k2);
    let rmap3 = rmap.clone();
    let kernel = DiffusionChernoff::with_closure(
        move |x: f64| thetas_arc[rmap3.region_of(node_idx(x, N_GRID, X_MIN, X_MAX))],
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        theta_uniform,
        grid,
    );
    let kernel_dual = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        theta_uniform,
        dual_default_grid(),
    );
    let rc = ReverseChernoff::new(
        kernel,
        kernel_dual,
        CheckpointSchedule::sqrt_n(N_STEPS_GRAD),
    )
    .with_region_map(rmap2)
    .expect("region map");
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let target = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let theta_slice: &[f64] = &thetas_k2;
    let (_, grads_k2) = rc
        .value_and_grad(tau, N_STEPS_GRAD, &u0, &target, theta_slice)
        .expect("K=2 reverse AD");
    let grads_k1 = reverse_mode_grad_k1(tau, N_STEPS_GRAD);
    let sum_k2 = grads_k2[0] + grads_k2[1];
    let rel_err = (sum_k2 - grads_k1).abs() / grads_k1.abs();
    println!(
        "K=2 uniform cross-check: grad_k2[0]={:.10e} grad_k2[1]={:.10e} sum={:.10e} \
         k1={:.10e} rel={:.3e}",
        grads_k2[0], grads_k2[1], sum_k2, grads_k1, rel_err
    );
}
