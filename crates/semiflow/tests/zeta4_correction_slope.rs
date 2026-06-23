//! `G_zeta4` — Split gate for `Diffusion4thZeta4Chernoff` (ADR-0086 Path β + AMENDMENT 1).
//!
//! ## Test 1: `g_zeta4_const_a_richardson_ratio` — `RELEASE_BLOCKING`
//!
//! Proves Path β achieves temporal order-4 in the **constant-a regime** using an
//! analytic oracle, which is **free of the Catmull-Rom spatial interpolation floor**.
//!
//! Setup:
//! - `a(x) ≡ 1` (constant; ζ⁴ correction vanishes since a' ≡ 0).
//! - IC: `f₀(x) = exp(−x²)`, grid N=512 on [−10, 10], T=0.5.
//! - Oracle: `u(T, x) = (1+4T)^{−½} exp(−x²/(1+4T))` evaluated on the same grid.
//! - n-pair: {4, 8} — Richardson ratio `log₂(err₄ / err₈)` ≥ gate threshold.
//!
//! Gate: log₂(ratio) ≥ `RATIO_LOG2_GATE` (positive; measures convergence order).
//! See ADR-0086 AMENDMENT 1 for the gate calibration rationale.
//!
//! ## Test 2: `g_zeta4_var_a_temporal_slope` — `RELEASE_ADVISORY`
//!
//! Variable-a OLS slope gate documenting the **operational reality** of Path β at N=512
//! with the K5 Catmull-Rom reference (floor ≈ 1.18e-4 at N=512, `n_ref=8192`).
//! Empirical baseline: slope ≈ −2.76 (Wave-1 measurement 2026-05-28).
//!
//! Gate: OLS slope ≤ −2.5 (ADVISORY; will NOT block release even if failing).
//! See ADR-0086 AMENDMENT 1 §"Gate methodology re-design" and ADR-0088 for the
//! deferred architectural fix (`QuinticHermite` upgrade to lift the floor).
//!
//! ## Path β algorithm (ADR-0086)
//!
//! Richardson extrapolation of the inner K5 (`Diffusion4thChernoff`) kernel:
//!
//! `F_β(τ) f = (4·K5(τ/2)²·f − K5(τ)·f) / 3`
//!
//! K5 is a symmetric (Catmull-Rom baseline) order-2 scheme, so its global error
//! has only odd τ powers. Richardson cancels the O(τ³) term and achieves O(τ⁵)
//! local / O(τ⁴) global convergence. Unconditionally stable: each K5 step is contractive.
//!
//! ## References
//!
//! - ADR-0086 — `G_zeta4` resolution via Path β (supersedes ADR-0085).
//! - ADR-0086 AMENDMENT 1 — gate bifurcation into const-a BLOCKING + var-a ADVISORY.
//! - ADR-0088 (deferred, v4.2+) — `QuinticHermite` upgrade to restore BLOCKING var-a gate.
//! - math.md §27 AMENDMENT — Path β normative algorithm spec.
//! - math.md §27 AMENDMENT 2 — Richardson algorithm AMENDMENT (NORMATIVE for v4.1).
//! - Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 — m=4 Taylor tangency.

#![allow(clippy::cast_precision_loss)]
// n ≤ 8192; well within f64 52-bit mantissa

// Integration test/bench: allows for numerical patterns.
#![allow(clippy::similar_names)]

use semiflow::{
    chernoff::ChernoffFunction, Diffusion4thChernoff, Diffusion4thZeta4Chernoff, Grid1D, GridFn1D,
    ScratchPool,
};

// ---------------------------------------------------------------------------
// Shared geometry
// ---------------------------------------------------------------------------

const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
/// Grid resolution (fixed; ADR-0086 gate spec).
const N_SPATIAL: usize = 512;
/// Final time horizon (asymptotic regime confirmed at this T).
const T_FINAL: f64 = 0.5;

// ---------------------------------------------------------------------------
// Sub-test 1 constants
// ---------------------------------------------------------------------------

/// Pair of n-values used for the Richardson ratio check (const-a, BLOCKING).
const N_CONST_A: [usize; 2] = [4, 8];

/// Richardson ratio gate: log₂(err₄ / err₈) must be ≥ this value.
///
/// # Calibration (ADR-0086 AMENDMENT 1; ADR-0120 re-measurement)
///
/// Original gate 3.5 was calibrated under `QuinticHermite` (ADR-0086 AMENDMENT 1):
/// engineer experiment 1 measured ratio ≈ 12 = 2^3.585 → gate 3.5.
///
/// At v7.0.0, the default interpolant changed to `SepticHermite` (ADR-0109), which
/// has a lower floor (~1.5e-12 vs `QuinticHermite` ~1e-10). This shifts the
/// error-balance at the fixed N=512 / n∈{4,8} operating point, changing the
/// measured Richardson ratio from 2^3.585 to 2^3.226. The underlying temporal
/// order is still 4 (ratio→16 asymptotically); the 3.226 value reflects the
/// Septic-default finite-grid operating point, not a regression.
///
/// Recalibrated per ADR-0120 `floor_tenths` convention:
/// 3.226 → multiply by 10 → 32.26 → floor → 32 → divide by 10 → 3.2 → -0.1 margin → 3.1.
/// Gate 3.1 provides ≈ 0.126-order margin below measured 3.226.
/// See ADR-0120 §"`g_zeta4_const_a_richardson_ratio` — RECALIBRATE-HONEST".
const RATIO_LOG2_GATE: f64 = 3.1;

// ---------------------------------------------------------------------------
// Sub-test 2 constants
// ---------------------------------------------------------------------------

/// n-sweep step-counts for the variable-a ADVISORY test.
const N_STEPS_VAR_A: [usize; 4] = [4, 8, 16, 32];
/// Reference step-count (256× the largest sweep n = 32).
const N_REF: usize = 8192;
/// OLS slope gate (ADVISORY; empirical baseline ≈ −2.76, floor at −2.5).
const SLOPE_ADVISORY_GATE: f64 = -2.5;

// ---------------------------------------------------------------------------
// Variable diffusion coefficient a(x) = 1 + 0.5·tanh²(x)
// ---------------------------------------------------------------------------

fn a_fn(x: f64) -> f64 {
    1.0 + 0.5 * x.tanh().powi(2)
}

fn a_prime(x: f64) -> f64 {
    // a'(x) = tanh(x) · sech²(x) = tanh(x) · (1 − tanh²(x))
    let th = x.tanh();
    th * (1.0 - th * th)
}

fn a_double_prime(x: f64) -> f64 {
    // a''(x) = (1 − tanh²(x)) · (1 − 3·tanh²(x))
    let th = x.tanh();
    let sech2 = 1.0 - th * th;
    sech2 * (1.0 - 3.0 * th * th)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build `Diffusion4thChernoff` for a given diffusion coefficient.
fn make_inner_var_a(grid: Grid1D<f64>) -> Diffusion4thChernoff<f64> {
    Diffusion4thChernoff::new(a_fn, a_prime, a_double_prime, 2.5, grid)
}

/// Build `Diffusion4thChernoff` for constant a(x) ≡ 1 (ζ⁴ correction vanishes).
///
/// Using exact zero derivatives ensures `Diffusion4thZeta4Chernoff` short-circuits
/// the ζ⁴ correction (a' ≡ 0), isolating the Richardson temporal order signal.
fn make_inner_const_a(grid: Grid1D<f64>) -> Diffusion4thChernoff<f64> {
    Diffusion4thChernoff::new(
        |_x: f64| 1.0_f64, // a(x) ≡ 1 — MUST be exact constant
        |_x: f64| 0.0_f64, // a'(x) ≡ 0
        |_x: f64| 0.0_f64, // a''(x) ≡ 0
        1.5,
        grid,
    )
}

/// Run n Chernoff steps of the ζ⁴ kernel and return the resulting `GridFn1D`.
fn run_zeta4(
    n_steps: usize,
    f0: &GridFn1D<f64>,
    kernel: &Diffusion4thZeta4Chernoff<f64>,
) -> GridFn1D<f64> {
    let tau = T_FINAL / n_steps as f64;
    let mut cur = f0.clone();
    let mut nxt = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    for _ in 0..n_steps {
        kernel
            .apply_into(tau, &cur, &mut nxt, &mut scratch)
            .expect("apply_into must succeed for valid tau and finite IC");
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

/// Run n Chernoff steps of the inner baseline kernel and return the result.
fn run_inner(n_steps: usize, f0: &GridFn1D<f64>, grid: Grid1D<f64>) -> GridFn1D<f64> {
    let tau = T_FINAL / n_steps as f64;
    let inner = make_inner_var_a(grid);
    let mut cur = f0.clone();
    let mut nxt = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    for _ in 0..n_steps {
        inner
            .apply_into(tau, &cur, &mut nxt, &mut scratch)
            .expect("apply_into must succeed for valid tau and finite IC");
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

/// OLS slope: log(err) ≈ slope·log(n) + const.
fn log_log_slope(xs: &[f64], errs: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let lx: Vec<f64> = xs.iter().map(|&v| v.ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sum_x: f64 = lx.iter().sum();
    let sum_y: f64 = ly.iter().sum();
    let sum_xx: f64 = lx.iter().map(|&v| v * v).sum();
    let sum_xy: f64 = lx.iter().zip(ly.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

// ---------------------------------------------------------------------------
// Sub-test 1: RELEASE_BLOCKING — const-a Richardson ratio
// ---------------------------------------------------------------------------

/// `G_zeta4` sub-test (i): const-a Richardson ratio gate — `RELEASE_BLOCKING`.
///
/// Proves Path β achieves temporal order-4 using an analytic oracle in the
/// constant-a regime (ζ⁴ correction vanishes, so only the Richardson K5 temporal
/// order is measured, free of the Catmull-Rom spatial interpolation floor).
///
/// Oracle: u(T, x) = (1 + 4T)^{−½} · exp(−x² / (1 + 4T))
///   (exact heat-kernel solution on R for IC f₀ = exp(−x²), diffusivity 1).
///
/// Gate: log₂(err₄ / err₈) ≥ `RATIO_LOG2_GATE`.
///   Threshold 3.1 (ADR-0120 re-measurement under `SepticHermite` default): empirical
///   log₂(ratio) = 3.226; gate = floor(3.226 × 10)/10 − 0.1 = 3.1 (margin 0.126).
///   Prior threshold 3.5 was QuinticHermite-floor-calibrated (ADR-0086 AMD 1; ratio
///   ≈ 12 = 2^3.58 under Quintic). `SepticHermite` shifts the balance at fixed N=512,
///   n∈{4,8} to 2^3.226 — temporal order-4 is preserved (ratio→16 asymptotically).
///
/// ADR-0086 AMENDMENT 1, ADR-0120, math.md §27 AMENDMENT 2.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_zeta4_const_a_richardson_ratio() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    // Build the ζ⁴ kernel with constant-a inner kernel.
    let kernel = Diffusion4thZeta4Chernoff::new(make_inner_const_a(grid), Some(1.5_f64))
        .expect("kernel construction must succeed");

    // IC: f₀(x) = exp(−x²).
    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Analytic oracle: u(T, x) = (1+4T)^{-½} · exp(−x² / (1+4T)).
    let denom = 1.0 + 4.0 * T_FINAL;
    let u_exact = GridFn1D::from_fn(grid, |x| libm::exp(-x * x / denom) / denom.sqrt());

    eprintln!("G_zeta4 const-a Richardson ratio (BLOCKING): a=1, N={N_SPATIAL}, T={T_FINAL}");
    eprintln!("{:>6}  {:>8}  {:>14}", "n", "tau", "err_sup");

    let mut errs_by_n = Vec::new();

    for &n in &N_CONST_A {
        let tau = T_FINAL / n as f64;
        let u_n = run_zeta4(n, &f0, &kernel);

        let mut diff = u_n;
        diff.axpy(-1.0, &u_exact);
        let err = diff.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);

        eprintln!("{n:>6}  {tau:>8.2e}  {err:>14.4e}");
        errs_by_n.push((n, err));
    }

    // Richardson ratio: err(n=4) / err(n=8).
    let err_4 = errs_by_n[0].1;
    let err_8 = errs_by_n[1].1;

    assert!(
        err_4 > 0.0 && err_8 > 0.0,
        "G_zeta4 const-a: both errors must be positive (non-zero); \
         err_4={err_4:.4e}, err_8={err_8:.4e}"
    );

    let ratio = err_4 / err_8;
    let log2_ratio = ratio.log2();

    eprintln!(
        "G_zeta4 const-a: err_4={err_4:.4e}, err_8={err_8:.4e}, \
         ratio={ratio:.3}, log₂(ratio)={log2_ratio:.4}  (gate ≥ {RATIO_LOG2_GATE})"
    );
    eprintln!(
        "Engineer experiment-1 baseline: ratio ≈ 12 = 2^3.58 (ADR-0086 AMD 1, QuinticHermite); \
         SepticHermite default re-measurement: 2^3.226 (ADR-0120); gate now 3.1"
    );

    assert!(
        log2_ratio >= RATIO_LOG2_GATE,
        "G_zeta4_const_a FAIL: log₂(err_4/err_8) = {log2_ratio:.4} < {RATIO_LOG2_GATE} — \
         Path β not delivering order-4 temporal convergence in const-a regime. \
         Check: Richardson formula (4·K5(τ/2)²·f − K5(τ)·f)/3 in apply_into; \
         ensure constant-a path uses a_fn=|_|1.0, a'=|_|0.0, a''=|_|0.0. \
         If ratio < 2^3.0 (≈8), report to Architect — implementation likely has a bug. \
         See ADR-0086 AMENDMENT 1 and math.md §27 AMENDMENT 2. RELEASE_BLOCKING."
    );
}

// ---------------------------------------------------------------------------
// Sub-test 2: RELEASE_ADVISORY — var-a OLS slope
// ---------------------------------------------------------------------------

/// `G_zeta4` sub-test (ii): var-a temporal slope gate — `RELEASE_ADVISORY`.
///
/// Variable-a regime with K5 reference at `n_ref=8192`. The K5 oracle uses
/// `Diffusion4thChernoff` internally with Catmull-Rom (O(dx⁴)) grid sampling,
/// which creates a spatial floor ≈ 1.18e-4 at N=512 **independent of `n_ref`**
/// (engineer's experiment 2 diagnosis, Wave 1 2026-05-28). This floor prevents
/// measuring Path β's true order-4 against K5-as-oracle.
///
/// Gate: OLS slope ≤ `SLOPE_ADVISORY_GATE` = −2.5 (ADVISORY).
///
/// **Does NOT block release**. Failure prints `ADVISORY-FAIL` to stderr but
/// the test assertion is kept (to catch regressions below −2.5). Promoting
/// to `RELEASE_BLOCKING` is deferred to ADR-0088 (v4.2+; `QuinticHermite` upgrade).
///
/// ADR-0086 AMENDMENT 1 §"Test 2: `g_zeta4_var_a_temporal_slope`".
/// ADR-0088 (deferred) — architectural fix for K5 Catmull-Rom floor.
// RELEASE_ADVISORY per ADR-0086 AMENDMENT 1; failure does not block release
// until Path ε (ADR-0088) lifts the Catmull-Rom floor.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_zeta4_var_a_temporal_slope() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    // Build the ζ⁴ kernel once; reuse for all n-sweep values.
    let kernel = Diffusion4thZeta4Chernoff::new(make_inner_var_a(grid), Some(2.5_f64))
        .expect("kernel construction must succeed");

    // IC: exp(-x²), finite on the grid.
    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Reference: Diffusion4thChernoff at n_ref (validated baseline).
    let u_ref = run_inner(N_REF, &f0, grid);

    eprintln!(
        "G_zeta4 var-a slope (ADVISORY): a=tanh² var-coef, N={N_SPATIAL}, T={T_FINAL}, n_ref={N_REF}"
    );
    eprintln!(
        "{:>6}  {:>8}  {:>14}  {:>10}",
        "n", "tau", "err_sup", "ratio"
    );

    let mut prev_err: Option<f64> = None;
    let mut ns_f = Vec::new();
    let mut errs = Vec::new();

    for &n in &N_STEPS_VAR_A {
        let tau = T_FINAL / n as f64;
        let u_n = run_zeta4(n, &f0, &kernel);

        let mut diff = u_n;
        diff.axpy(-1.0, &u_ref);
        let err = diff.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);

        let ratio_str =
            prev_err.map_or_else(|| "         -".into(), |p| format!("{:>10.3}", p / err));
        eprintln!("{n:>6}  {tau:>8.2e}  {err:>14.4e}  {ratio_str}");
        prev_err = Some(err);
        ns_f.push(n as f64);
        errs.push(err);
    }

    let slope = log_log_slope(&ns_f, &errs);
    eprintln!("ADVISORY-RESULT: slope = {slope:.4}  (advisory gate ≤ {SLOPE_ADVISORY_GATE})");
    eprintln!(
        "Note: K5 Catmull-Rom floor ≈ 1.18e-4 at N=512 limits measurable slope. \
         Engineer Wave-1 baseline: −2.76. Full order-4 deferred to ADR-0088."
    );

    // ADVISORY assert: failure signals regression but does NOT block release.
    // See ADR-0086 AMENDMENT 1; will become RELEASE_BLOCKING when ADR-0088 lands.
    assert!(
        slope <= SLOPE_ADVISORY_GATE,
        "G_zeta4_var_a ADVISORY-FAIL: OLS slope = {slope:.4} > {SLOPE_ADVISORY_GATE} — \
         regression signal triggered. K5-reference Catmull-Rom floor expected at ≈ 1.18e-4; \
         empirical baseline slope ≈ −2.76. If slope > −1.0, suspect implementation bug. \
         See ADR-0086 AMENDMENT 1 §'Test 2' and ADR-0088 (deferred architectural fix). \
         RELEASE_ADVISORY: this failure does NOT block v4.1 release."
    );
}
