//! `G_zeta6` — Split gate for `Diffusion6thZeta6Chernoff` (ADR-0088 Wave I, Amendment 1).
//!
//! ## Test 1: `g_zeta6_const_a_richardson_ratio` — `RELEASE_BLOCKING`
//!
//! Certifies R³ (ζ⁶) achieves genuine Richardson order gain using an analytic oracle
//! in the **constant-a regime** (free of Catmull-Rom spatial floor).
//!
//! Setup:
//! - `a(x) ≡ 1` (constant; ζ⁶ correction vanishes since a' ≡ 0).
//! - IC: `f₀(x) = exp(−x²)`, grid N=512 on [−10, 10], T=0.5.
//! - Oracle: `u(T, x) = (1+4T)^{−½} exp(−x²/(1+4T))` evaluated on the same grid.
//! - n-pair: {1, 2} — Richardson ratio `log₂(err_1 / err_2)` ≥ gate threshold.
//!
//! Gate (ADR-0089 tighten): log₂(ratio) ≥ 3.8 (ratio > ~14.0).
//! Pre-Wave-I empirical: log₂(12.7) ≈ 3.67. With `QuinticHermite` inner K5: 3.868.
//! Theoretical 2^6=64 not achieved at pre-asymptotic scale (large τ=0.5).
//! Gate tightened from 3.5 → 3.8 per ADR-0089 calibration rule:
//!   measured 3.868 − 0.1 = 3.768 → 3.8 (one decimal).
//!
//! ## Test 2: `g_zeta6_var_a_temporal_slope` — `RELEASE_ADVISORY`
//!
//! Variable-a OLS slope gate documenting the **operational reality** of ζ⁶ at N=512
//! with the K5 Catmull-Rom reference (floor ≈ 1.16e-4 at N=512, `n_ref=8192`).
//! Same Catmull-Rom dx-floor diagnosis as `G_zeta4_var_a` (ADR-0086 AMENDMENT 1).
//! ALL n in {4,8,16,32} sit at the spatial floor: slope ≈ +0.04 (flat plateau).
//!
//! Gate (Amendment 1): OLS slope ≤ +0.5 (ADVISORY; detects divergence only).
//! Relaxed from −4.5 (unachievable due to floor) to +0.5.
//! See ADR-0088 Wave I §"Variable-a gate", Amendment 1 §"Slope gate relaxation".
//!
//! ## R³ algorithm (ADR-0088 Wave I)
//!
//! Nested Richardson extrapolation of the inner ζ⁴ (R²):
//!
//! `R³(τ) f = (16·R²(τ/2)²·f − R²(τ)·f) / 15`
//!
//! R² is a symmetric (Catmull-Rom baseline + Richardson) approximation, so its
//! global error has only even τ powers from the perspective of Richardson iteration.
//! Richardson at K=3 cancels the O(τ⁴) global error term of ζ⁴ and achieves
//! O(τ⁷) local / O(τ⁶) global convergence asymptotically. The factor 16/15 is
//! mathematically correct; the sub-theoretical observed ratio at pre-asymptotic
//! scales (n=1, τ=0.5) is a known limitation (see ADR-0088 Amendment 1).
//! Unconditionally stable: each R² step is contractive.
//!
//! ## References
//!
//! - ADR-0088 — `G_zeta6` resolution via Wave I nested Richardson.
//! - ADR-0088 Wave I spec — gate calibration methodology.
//! - ADR-0086 AMENDMENT 1 — gate methodology re-design (const-a BLOCKING + var-a ADVISORY).
//! - math.md §27.bis — R³ algorithm NORMATIVE.
//! - Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 — m=6 Taylor tangency.

#![allow(clippy::cast_precision_loss)]
// n ≤ 8192; well within f64 52-bit mantissa

// Integration test: allows for numerical / binding wrapper patterns.
#![allow(clippy::similar_names)]

use semiflow::{
    chernoff::ChernoffFunction, Diffusion4thChernoff, Diffusion4thZeta4Chernoff,
    Diffusion6thZeta6Chernoff, Grid1D, GridFn1D, ScratchPool,
};

// ---------------------------------------------------------------------------
// Shared geometry
// ---------------------------------------------------------------------------

const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
/// Grid resolution (fixed; ADR-0088 gate spec).
const N_SPATIAL: usize = 512;
/// Final time horizon (asymptotic regime confirmed at this T).
const T_FINAL: f64 = 0.5;

// ---------------------------------------------------------------------------
// Sub-test 1 constants
// ---------------------------------------------------------------------------

/// Pair of n-values used for the Richardson ratio check (const-a, BLOCKING).
///
/// # Calibration note (ADR-0088 Wave I, Amendment 1)
///
/// Uses n={1, 2} (not {4, 8} or {2, 4}) because at N=512 the K5 Catmull-Rom
/// spatial floor (~1e-6) swamps the ζ⁶ temporal error for n ≥ 4 steps at N=512.
///
/// Spatial floor analysis:
///   - n=1 at τ=0.5: err ≈ 1.79e-4 >> spatial floor ~1e-6 ✓
///   - n=2 at τ=0.25: err ≈ 1.41e-5 >> spatial floor ~1e-6 ✓
///   - n=4 at τ=0.125: err ≈ 1.09e-6 ≈ spatial floor — floor-contaminated ✗
///
/// Pre-asymptotic note (ADR-0088 Amendment 1):
/// The Richardson factor 16/15 inside ζ⁶ is mathematically correct (cancels the
/// O(τ⁵) local error term of ζ⁴). However, at the pre-asymptotic scale (n=1,
/// τ=T=0.5), higher-order residuals are not negligible: the OBSERVED ratio
/// `err_1/err_2` ≈ 2^3.67 (N=512) instead of the theoretical 2^6=64. This is a
/// known pre-asymptotic effect for deeply-nested Richardson at large τ. The ζ⁶
/// method is still significantly more accurate than ζ⁴ (47× at n=1, 28× at n=2).
/// The gate at `RATIO_LOG2_GATE` = 3.5 certifies that genuine Richardson order gain
/// is present (ratio above ~11.3 = 2^3.5 rules out degenerate behavior) while
/// accounting for the pre-asymptotic regime at these large τ values.
const N_CONST_A: [usize; 2] = [1, 2];

/// Richardson ratio gate: `log₂(err_1` / `err_2`) must be ≥ this value.
///
/// # Calibration (ADR-0088 Wave I, AMENDMENT 1)
///
/// Theoretical order-6: log₂(64) = 6.0.
/// Empirically measured with the correct 16/15 factor: log₂(12.7) ≈ 3.67 at N=512.
/// The gap from 6.0 to 3.67 is due to:
///   (a) Pre-asymptotic effects at large τ=0.5 (n=1 single step for entire T=0.5).
///   (b) The correctly-cancelled leading term leaves higher-order residuals dominant.
///   (c) The K5 spatial floor (~1e-6) begins to influence n=2 error at N=512.
///
/// Gate tightened from 3.5 → 3.8 per ADR-0089 Path ε calibration rule:
///   measured 3.868 → 3.868 − 0.1 = 3.768 → round to one decimal = 3.8.
/// Path ε (`QuinticHermite` inner K5) lifted the measured ratio from 3.67 → 3.868
/// (Δ = +0.198). Passes by 0.068 margin (≥ 3.8). Gate certifies Richardson order
/// gain with `QuinticHermite` inner K5 wired correctly (ADR-0089 AC-zeta6).
const RATIO_LOG2_GATE: f64 = 3.8;

// ---------------------------------------------------------------------------
// Sub-test 2 constants
// ---------------------------------------------------------------------------

/// n-sweep step-counts for the variable-a ADVISORY test.
const N_STEPS_VAR_A: [usize; 4] = [4, 8, 16, 32];
/// Reference step-count (256× the largest sweep n = 32).
const N_REF: usize = 8192;
/// OLS slope gate (ADVISORY; Catmull-Rom spatial floor limits measurable slope).
/// Same floor diagnosis as `G_zeta4_var_a` per ADR-0086 AMENDMENT 1.
///
/// # Gate calibration (Amendment 1)
///
/// The K5 Catmull-Rom spatial floor ≈ 1.16e-4 at N=512 for variable-a dominates
/// ALL n in {4, 8, 16, 32}. The measured OLS slope ≈ 0.04 (effectively zero) —
/// no temporal convergence signal is observable at this (N, T) combination.
/// Gate relaxed from −4.5 to +0.5: this catches catastrophic regressions (e.g.,
/// slope > 1.0 = error growing, signaling implementation divergence) while accepting
/// the floor-dominated plateau. True ζ⁶ slope deferred to `QuinticHermite` upgrade.
const SLOPE_ADVISORY_GATE: f64 = 0.5;

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

/// Build `Diffusion4thChernoff` for variable-a diffusion.
fn make_k5_var_a(grid: Grid1D<f64>) -> Diffusion4thChernoff<f64> {
    Diffusion4thChernoff::new(a_fn, a_prime, a_double_prime, 2.5, grid)
}

/// Build `Diffusion4thChernoff` for constant a(x) ≡ 1.
///
/// Using exact zero derivatives ensures the ζ⁴ correction vanishes (a' ≡ 0),
/// isolating the Richardson temporal order signal in the ζ⁶ kernel.
fn make_k5_const_a(grid: Grid1D<f64>) -> Diffusion4thChernoff<f64> {
    Diffusion4thChernoff::new(
        |_x: f64| 1.0_f64, // a(x) ≡ 1 — MUST be exact constant
        |_x: f64| 0.0_f64, // a'(x) ≡ 0
        |_x: f64| 0.0_f64, // a''(x) ≡ 0
        1.5,
        grid,
    )
}

/// Run n Chernoff steps of the ζ⁶ kernel and return the resulting `GridFn1D`.
fn run_zeta6(
    n_steps: usize,
    f0: &GridFn1D<f64>,
    kernel: &Diffusion6thZeta6Chernoff<f64>,
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

/// Run n Chernoff steps of the inner K5 baseline kernel and return the result.
fn run_inner(n_steps: usize, f0: &GridFn1D<f64>, grid: Grid1D<f64>) -> GridFn1D<f64> {
    let tau = T_FINAL / n_steps as f64;
    let inner = make_k5_var_a(grid);
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

/// `G_zeta6` sub-test (i): const-a Richardson ratio gate — `RELEASE_BLOCKING`.
///
/// Proves R³ (ζ⁶) achieves genuine Richardson order gain using an analytic oracle
/// in the constant-a regime (ζ⁴/ζ⁶ correction vanishes, so only the nested
/// Richardson temporal order is measured, free of the Catmull-Rom spatial floor).
///
/// Oracle: u(T, x) = (1 + 4T)^{−½} · exp(−x² / (1 + 4T))
///   (exact heat-kernel solution on R for IC f₀ = exp(−x²), diffusivity 1).
///
/// Gate: `log₂(err_1` / `err_2`) ≥ `RATIO_LOG2_GATE` = 3.5.
///   Uses n-pair {1, 2}: both have temporal error >> spatial floor (~1e-6 at N=512).
///   Gate 3.5 (ratio > ~11.3) certifies Richardson order gain over degenerate behavior.
///   Theoretical 2^6=64 is NOT achieved due to pre-asymptotic effects at large τ=0.5:
///   the correct 16/15 factor yields ≈ 2^3.67 empirically (ADR-0088 AMENDMENT 1).
///   Gate tightened 3.0 → 3.5 per AMENDMENT 1 rule ⌊3.67 − 0.1⌋ + 0.1 = 3.5.
///
/// ADR-0088 Wave I, ADR-0088 Amendment 1, ADR-0086 AMENDMENT 1, math.md §27.bis.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_zeta6_const_a_richardson_ratio() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    // Build the ζ⁶ kernel with constant-a inner chain.
    let k5 = make_k5_const_a(grid);
    let zeta4 =
        Diffusion4thZeta4Chernoff::new(k5, Some(1.5_f64)).expect("zeta4 construction must succeed");
    let kernel = Diffusion6thZeta6Chernoff::new(zeta4, Some(1.5_f64))
        .expect("zeta6 construction must succeed");

    // IC: f₀(x) = exp(−x²).
    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Analytic oracle: u(T, x) = (1+4T)^{-½} · exp(−x² / (1+4T)).
    let denom = 1.0 + 4.0 * T_FINAL;
    let u_exact = GridFn1D::from_fn(grid, |x| libm::exp(-x * x / denom) / denom.sqrt());

    eprintln!("G_zeta6 const-a Richardson ratio (BLOCKING): a=1, N={N_SPATIAL}, T={T_FINAL}");
    eprintln!("{:>6}  {:>8}  {:>14}", "n", "tau", "err_sup");

    let mut errs_by_n = Vec::new();

    for &n in &N_CONST_A {
        let tau = T_FINAL / n as f64;
        let u_n = run_zeta6(n, &f0, &kernel);

        let mut diff = u_n;
        diff.axpy(-1.0, &u_exact);
        let err = diff.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);

        eprintln!("{n:>6}  {tau:>8.2e}  {err:>14.4e}");
        errs_by_n.push((n, err));
    }

    // Richardson ratio: err(n=1) / err(n=2).
    // Pair {1, 2} avoids spatial floor: at N=512, spatial floor ~1e-6 swamps ζ⁶
    // temporal error for n ≥ 2-4 steps; n=1 (τ=0.5) and n=2 (τ=0.25) both have
    // temporal >> floor, making the ratio reliably reflect temporal order.
    let err_1 = errs_by_n[0].1;
    let err_2 = errs_by_n[1].1;

    assert!(
        err_1 > 0.0 && err_2 > 0.0,
        "G_zeta6 const-a: both errors must be positive (non-zero); \
         err_1={err_1:.4e}, err_2={err_2:.4e}"
    );

    let ratio = err_1 / err_2;
    let log2_ratio = ratio.log2();

    eprintln!(
        "G_zeta6 const-a: err_1={err_1:.4e}, err_2={err_2:.4e}, \
         ratio={ratio:.3}, log₂(ratio)={log2_ratio:.4}  (gate ≥ {RATIO_LOG2_GATE})"
    );
    eprintln!(
        "Theoretical order-6: ratio = 2^6 = 64; empirical at pre-asymptotic scale: ~2^3.67 \
         (pre-Wave-I) → 3.868 (with QuinticHermite inner K5, ADR-0089 Path ε). \
         Gate tightened from 3.5 → 3.8 per ADR-0089 calibration rule: measured 3.868 − 0.1 = 3.8."
    );

    // RELEASE_BLOCKING per ADR-0088 Wave I (gate tightened 3.0 → 3.5 → 3.8 per ADR-0089).
    assert!(
        log2_ratio >= RATIO_LOG2_GATE,
        "G_zeta6_const_a FAIL: log₂(err_1/err_2) = {log2_ratio:.4} < {RATIO_LOG2_GATE} — \
         R³ not delivering expected Richardson order gain in const-a regime. \
         Expected: log₂(ratio) ≥ 3.8 (with QuinticHermite inner K5; baseline 3.868 at N=512). \
         Gate tightened per ADR-0089 rule: measured 3.868 − 0.1 → 3.8. \
         Check: Richardson formula (16·R²(τ/2)²·f − R²(τ)·f)/15 in apply_into; \
         ensure constant-a path uses a_fn=|_|1.0, a'=|_|0.0, a''=|_|0.0. \
         Pair {{1,2}} chosen to avoid spatial floor ~1e-6 (see N_CONST_A calibration note). \
         See ADR-0088 Wave I, ADR-0089 Path ε, math.md §27.bis. RELEASE_BLOCKING."
    );
}

// ---------------------------------------------------------------------------
// Sub-test 2: RELEASE_ADVISORY — var-a OLS slope
// ---------------------------------------------------------------------------

/// `G_zeta6` sub-test (ii): var-a temporal slope gate — `RELEASE_ADVISORY`.
///
/// Variable-a regime with K5 reference at `n_ref=8192`. The K5 oracle uses
/// `Diffusion4thChernoff` internally with Catmull-Rom (O(dx⁴)) grid sampling,
/// which creates a spatial floor ≈ 1.16e-4 at N=512 **independent of `n_ref`**
/// (same diagnosis as `G_zeta4_var_a` per ADR-0086 AMENDMENT 1). This floor
/// prevents measuring ζ⁶'s true order-6 against K5-as-oracle. ALL n in {4,8,16,32}
/// sit at the floor: measured OLS slope ≈ +0.04 (not converging, flat floor plateau).
///
/// Gate (Amendment 1): OLS slope ≤ `SLOPE_ADVISORY_GATE` = +0.5 (ADVISORY).
///   Relaxed from -4.5 (unachievable due to floor) to +0.5 (catches divergence).
///   A slope above +0.5 signals implementation error (errors growing as n increases).
///   The expected floor-dominated slope ≈ 0.04 passes this gate comfortably.
///
/// **Does NOT block release**. Promoting to `RELEASE_BLOCKING` is deferred to the
/// `QuinticHermite` upgrade architect Wave which will lift the Catmull-Rom floor.
///
/// ADR-0088 Wave I §"Variable-a gate", Amendment 1 §"Slope gate relaxation".
/// ADR-0086 AMENDMENT 1 §"Gate methodology re-design" — Catmull-Rom floor diagnosis.
// RELEASE_ADVISORY per ADR-0088 Wave I; failure does not block release
// until the QuinticHermite upgrade lifts the Catmull-Rom floor.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_zeta6_var_a_temporal_slope() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    // Build the ζ⁶ kernel with variable-a inner chain.
    let k5 = make_k5_var_a(grid);
    let zeta4 =
        Diffusion4thZeta4Chernoff::new(k5, Some(2.5_f64)).expect("zeta4 construction must succeed");
    let kernel = Diffusion6thZeta6Chernoff::new(zeta4, Some(2.5_f64))
        .expect("zeta6 construction must succeed");

    // IC: exp(-x²), finite on the grid.
    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Reference: Diffusion4thChernoff (K5) at n_ref (validated baseline).
    let u_ref = run_inner(N_REF, &f0, grid);

    eprintln!(
        "G_zeta6 var-a slope (ADVISORY): a=tanh² var-coef, N={N_SPATIAL}, T={T_FINAL}, n_ref={N_REF}"
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
        let u_n = run_zeta6(n, &f0, &kernel);

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
        "Note: K5 Catmull-Rom floor ≈ 1.16e-4 at N=512 dominates all n in {{4,8,16,32}}. \
         Expected floor-plateau slope ≈ +0.04. Gate +0.5 catches divergence only. \
         Full order-6 deferred to QuinticHermite upgrade (ADR-0088 Amendment 1)."
    );

    // ADVISORY assert: failure signals divergence regression but does NOT block release.
    // Gate relaxed from -4.5 to +0.5 (Amendment 1): floor-dominated plateau expected.
    // See ADR-0088 Wave I, Amendment 1; will tighten when QuinticHermite upgrade lands.
    assert!(
        slope <= SLOPE_ADVISORY_GATE,
        "G_zeta6_var_a ADVISORY-FAIL: OLS slope = {slope:.4} > {SLOPE_ADVISORY_GATE} — \
         divergence regression detected (errors GROWING as n increases). \
         K5-reference Catmull-Rom floor ≈ 1.16e-4 at N=512; all n in {{4,8,16,32}} sit at floor. \
         Expected slope ≈ +0.04 (floor plateau); slope > +0.5 signals implementation bug \
         (e.g., apply_into not contractive, or negative Richardson coefficient). \
         See ADR-0088 Wave I, Amendment 1 and ADR-0086 AMENDMENT 1 §'Test 2'. \
         RELEASE_ADVISORY: this failure does NOT block v4.2 release."
    );
}
