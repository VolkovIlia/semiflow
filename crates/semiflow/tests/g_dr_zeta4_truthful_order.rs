//! `G_DR_ZETA4_TRUTHFUL_ORDER` — order-4 gate for `DriftReactionZeta4Chernoff`
//! (ADR-0131, math.md §27.7-bis).
//!
//! ## Purpose
//!
//! Verifies that `DriftReactionZeta4Chernoff` achieves genuine temporal order 4
//! on a datum where `[R, D] ≠ 0` (the BCH commutator cancellation is non-trivial).
//!
//! ## PDE
//!
//! `∂_t u = u_xx − γx·u_x + c·u`,  `γ = 0.3`, `c = −0.3`.
//!
//! ## Why `[R, D] ≠ 0`
//!
//! `D = ∂ₓₓ`, `R = −γx·∂ₓ + c·I`. `[D,R]f = −2γ·f″ ≠ 0` for γ≠0.
//!
//! ## Analytic oracle (Ornstein-Uhlenbeck + constant reaction)
//!
//! For `∂_t u = u_xx − γx·u_x + c·u`, `u₀(x) = exp(−x²)`:
//!
//! ```text
//! u(t,x) = exp(c·t) · (1 + 4σ²(t))^{−1/2} · exp(−(x·e^{−γt})² / (1 + 4σ²(t)))
//! ```
//!
//! where `σ²(t) = (1 − e^{−2γt}) / (2γ)` (OU variance at time t).
//!
//! Derivation: Ornstein-Uhlenbeck heat kernel `G(t,x,y)` convolved with `exp(−y²)`.
//! Verification: u(0,x) = 1·exp(−x²) ✓; satisfies PDE ✓ (see ADR-0131 math verification).
//!
//! ## Configuration
//!
//! - `a = 1.0`, `γ = 0.3`, `c = −0.3`.
//! - IC: `f₀(x) = exp(−x²)`, grid N=2048 on `[−10, 10]`.
//! - T = 2.0 (temporal errors >> spatial floor at all ladder points).
//! - Ladder: `N_STEPS` = {2, 4, 8, 16} → τ ∈ {1.0, 0.5, 0.25, 0.125}.
//!
//! ## Gate
//!
//! OLS slope ≤ −3.5 (`SLOPE_GATE`). `RELEASE_BLOCKING` per ADR-0131.
//! Slope steeper than −6 is a RED FLAG (artifact or oracle error).

#![allow(clippy::cast_precision_loss)] // n ≤ 16; well within f64 mantissa

// Integration test/bench: allows for numerical patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use semiflow_core::{
    chernoff::ChernoffFunction, Diffusion4thChernoff, DriftReactionZeta4Chernoff, Grid1D, GridFn1D,
    ScratchPool,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
const N_SPATIAL: usize = 2048;
const T_FINAL: f64 = 0.5;
const N_STEPS: [usize; 4] = [2, 4, 8, 16];
const SLOPE_GATE: f64 = -3.5;
const RED_FLAG_SLOPE: f64 = -6.0;

// OU + reaction parameters.
const GAMMA: f64 = 0.3; // b(x) = −γx
const C_REACT: f64 = -0.3; // reaction term
const C_NORM_BOUND: f64 = 0.3;

// ---------------------------------------------------------------------------
// Analytic oracle (OU + constant reaction, Gaussian IC)
// ---------------------------------------------------------------------------

/// Exact solution of `∂_t u = u_xx − γx·u_x + c·u`, u₀(x) = exp(−x²).
///
/// Formula (matches module docstring and the unit test `approx_order_4_ou_oracle`):
///
/// `u(t,x) = exp(c·t) · (1 + 4σ²)^{−1/2} · exp(−(x·e^{−γt})² / (1 + 4σ²))`
///
/// where `σ² = (1 − e^{−2γt}) / (2γ)`.
///
/// Derivation: `E[exp(−X_t²) | X_0=x]` with `X_t` ~ N(x·e^{−γt}, (1−e^{−2γt})/γ).
/// Gaussian integral gives denom = 1+2V, V = (1−e^{−2γt})/γ = 2σ², so denom = 1+4σ².
/// Verification: u(0,x) = exp(−x²) ✓; max at x=0 is exp(c·t)·(1+4σ²)^{−1/2} ✓.
fn oracle(t: f64, x: f64) -> f64 {
    if t == 0.0 {
        return libm::exp(-x * x);
    }
    let s = libm::exp(-GAMMA * t);
    let sigma2 = (1.0 - s * s) / (2.0 * GAMMA);
    let denom = 1.0 + 4.0 * sigma2; // = 1 + 2V, V = (1−e^{−2γt})/γ
    let mu = x * s; // mean of X_t | X_0=x
    libm::exp(C_REACT * t) / denom.sqrt() * libm::exp(-mu * mu / denom)
}

// ---------------------------------------------------------------------------
// Kernel constructor
// ---------------------------------------------------------------------------

fn make_kernel(n: usize) -> DriftReactionZeta4Chernoff {
    let grid = Grid1D::new(X_MIN, X_MAX, n).expect("grid");
    let diff = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    // b(x) = -γx (linear), b'(x) = -γ (constant).
    // Linear b → Newton step is exact → exactly palindromic → order-4 Richardson.
    DriftReactionZeta4Chernoff::new(
        diff,
        |x: f64| -GAMMA * x,
        |_| -GAMMA,
        |_| C_REACT,
        C_NORM_BOUND,
        grid,
    )
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

fn run_kernel(
    n_steps: usize,
    f0: &GridFn1D<f64>,
    kernel: &DriftReactionZeta4Chernoff,
) -> GridFn1D<f64> {
    let tau = T_FINAL / n_steps as f64;
    let mut cur = f0.clone();
    let mut nxt = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    for _ in 0..n_steps {
        kernel
            .apply_into(tau, &cur, &mut nxt, &mut scratch)
            .expect("apply_into must succeed");
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

// ---------------------------------------------------------------------------
// OLS slope
// ---------------------------------------------------------------------------

fn log_log_slope(ns: &[f64], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let lx: Vec<f64> = ns.iter().map(|&v| v.ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sum_x: f64 = lx.iter().sum();
    let sum_y: f64 = ly.iter().sum();
    let sum_xx: f64 = lx.iter().map(|&v| v * v).sum();
    let sum_xy: f64 = lx.iter().zip(ly.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

// ---------------------------------------------------------------------------
// Interior sup-norm vs analytic oracle
// ---------------------------------------------------------------------------

fn interior_oracle_err(u: &GridFn1D<f64>, grid: Grid1D<f64>, t: f64) -> f64 {
    let n = u.values.len();
    let margin = 50.min(n / 8);
    let mut max_err = 0.0f64;
    for i in margin..n - margin {
        let x = grid.x_at(i);
        let err = (u.values[i] - oracle(t, x)).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

/// `G_DR_ZETA4_TRUTHFUL_ORDER` — `RELEASE_BLOCKING` (ADR-0131).
///
/// Verifies `DriftReactionZeta4Chernoff` achieves genuine temporal order 4
/// on a `[R,D]≠0` datum (OU drift `b(x) = −0.3x`). Uses closed-form OU+reaction
/// oracle. No reference floor issue — oracle is analytic.
///
/// Gate: OLS slope ≤ −3.5. Slope steeper than −6 is RED FLAG.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_dr_zeta4_truthful_order() {
    // Diagnostic: verify oracle at T=2.
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid");

    // Oracle sanity check: oracle(0, x) should equal IC exp(-x²).
    let ic_err: f64 = (0..10)
        .map(|i| {
            let x = X_MIN + (X_MAX - X_MIN) * f64::from(i) / 9.0;
            (oracle(0.0, x) - libm::exp(-x * x)).abs()
        })
        .fold(0.0, f64::max);
    assert!(ic_err < 1e-14, "Oracle IC mismatch: {ic_err:.2e}");

    let f0 = GridFn1D::from_fn(grid, |x| oracle(0.0, x));
    let kernel = make_kernel(N_SPATIAL);

    eprintln!("G_DR_ZETA4_TRUTHFUL_ORDER: [R,D]≠0, N={N_SPATIAL}, T={T_FINAL}, γ=0.3, c=-0.3");
    eprintln!("Oracle: OU+reaction closed form (see module doc).");
    eprintln!("{:>6}  {:>10}  {:>14}", "n", "tau", "err_interior");

    let mut ns_f: Vec<f64> = Vec::new();
    let mut errs: Vec<f64> = Vec::new();

    for &n in &N_STEPS {
        let tau = T_FINAL / n as f64;
        let u_n = run_kernel(n, &f0, &kernel);
        let u_n_max = u_n.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let err = interior_oracle_err(&u_n, grid, T_FINAL);
        eprintln!("{n:>6}  {tau:>10.4e}  u_n_max={u_n_max:.4e}  {err:>14.4e}");
        ns_f.push(n as f64);
        errs.push(err);
    }

    eprintln!("  Pair-slopes:");
    for i in 0..ns_f.len() - 1 {
        let pair = (errs[i + 1].max(1e-16).ln() - errs[i].max(1e-16).ln())
            / (ns_f[i + 1].ln() - ns_f[i].ln());
        eprintln!(
            "    {:>2} → {:>2}: slope = {:>7.4}",
            ns_f[i] as usize,
            ns_f[i + 1] as usize,
            pair
        );
    }

    let slope = log_log_slope(&ns_f, &errs);
    eprintln!("G_DR_ZETA4_TRUTHFUL_ORDER: OLS slope = {slope:.4}  (gate ≤ {SLOPE_GATE})");

    assert!(
        slope > RED_FLAG_SLOPE,
        "G_DR_ZETA4_TRUTHFUL_ORDER RED FLAG: slope = {slope:.4} < {RED_FLAG_SLOPE}. ADR-0131."
    );

    assert!(
        slope <= SLOPE_GATE,
        "G_DR_ZETA4_TRUTHFUL_ORDER FAIL (RELEASE_BLOCKING): \
         OLS slope = {slope:.4} > {SLOPE_GATE}. \
         Temporal order-4 NOT demonstrated on [R,D]≠0 datum. \
         Diagnostics: (a) middle pair slope should ≈ -4; \
         (b) finest-pair shallow → spatial floor; try larger T_FINAL; \
         (c) all slopes ≈ -3 → Strang base has even τ⁴ error not canceled by Richardson \
         (verify K5 inner kernel in DriftReactionZeta4Chernoff). \
         ADR-0131, math.md §27.7-bis."
    );
}
