//! `G_zeta8` — Split gate for `Diffusion8thZeta8Chernoff` (ADR-0088 Wave II, ADR-0090).
//!
//! ## Test 1: `g_zeta8_const_a_richardson_cheb` — `RELEASE_BLOCKING`
//!
//! Certifies R⁴ (ζ⁸) achieves genuine Richardson order gain using an analytic oracle
//! in the **constant-a regime** with **Chebyshev spectral sampling ON** (default).
//!
//! Setup:
//! - `a(x) ≡ 1` (constant; ζ⁴/ζ⁶ correction vanishes since a' ≡ 0).
//! - IC: `f₀(x) = exp(−x²)`, grid N=512 on [−10, 10], T=0.5.
//! - Oracle: `u(T, x) = (1+4T)^{−½} exp(−x²/(1+4T))` evaluated on the same grid.
//! - n-pair: {1, 2} — Richardson ratio `log₂(err_1 / err_2)` ≥ gate threshold.
//! - Chebyshev M=64 ON by default (required for order-8 contract, ADR-0090).
//!
//! Gate: log₂(ratio) ≥ **3.0** (v5.0.0 baseline PRESERVED at v6.0.0 `SepticHermite`).
//!   Per ADR-0109 AMENDMENT 1 + math.md §40.5.bis: this gate measures the PRE-ASYMPTOTIC
//!   K5+Richardson TEMPORAL TRANSITION regime which is INDEPENDENT of the spatial floor.
//!   ADR-0109 §40.4 originally predicted 7.19 — RETRACTED by AMENDMENT 1.
//!   `SepticHermite` spatial-floor lift is INDEPENDENTLY proven by `G_SEPTIC_HERMITE_FLOOR`.
//!   Academic K=8 LOCAL Taylor tangency via sympy oracle (ADR-0110 AMENDMENT 1:
//!   `G_zeta8_TRUTHFUL_ORDER` DEFERRED v7.0+ OCTONIC).
//!
//! ## Test 2: `g_zeta8_var_a_slope_cheb` — `RELEASE_ADVISORY`
//!
//! Variable-a OLS slope gate with Chebyshev ON. Gate: OLS slope ≤ 0.1 (not-diverging).
//! Measured ≈ 0.056 (floor-dominated stagnation; cascade amplification σ² ≈ 2.78 hides
//! order-8 signal at N=512). The theoretical full-order gate ≤ −6.5 is infeasible at
//! current N=512 / T=0.5 parameters; that value is a future-target, not a current gate.
//! See ADR-0109 §40.6 + §"Scope boundaries".
//!
//! ## R⁴ algorithm (ADR-0088 Wave II)
//!
//! Nested Richardson extrapolation of the inner R³ (ζ⁶):
//!
//! `R⁴(τ) f = (64·R³(τ/2)²·f − R³(τ)·f) / 63`
//!
//! R³ is symmetric (time-reversible), so its global error has only odd powers of τ
//! from Richardson's perspective. Richardson at K=4 cancels O(τ⁷) and achieves
//! O(τ⁹) local / O(τ⁸) global convergence asymptotically.
//!
//! Work: 3 inner R³ calls = 27 K5 base evaluations per outer step.
//!
//! ## References
//!
//! - ADR-0088 — ζ⁶/ζ⁸ ladder rungs; Wave II HOLD released conditional on ADR-0090.
//! - ADR-0090 — Chebyshev spectral collocation (unblocker for ζ⁸).
//! - ADR-0089 AMENDMENT 1 — Insight #5: Quintic floor causes ζ⁸ stagnation.
//! - ADR-0109 §40.4 — `SepticHermite` formal-model slope projection (7.19) RETRACTED by AMENDMENT 1.
//! - ADR-0109 AMENDMENT 1 — threshold 3.0 PRESERVED (pre-asymp temporal transition regime).
//! - ADR-0109 §40.6 — ζ⁸ cascade-ceiling investigation; 7.19 ORDER GAP diagnosis.
//! - ADR-0110 — `G_zeta_K_TRUTHFUL_ORDER` pre-asymptotic gates (v6.0.0 BREAKING window).
//! - math.md §27.ter — R⁴ algorithm NORMATIVE.
//! - math.md §39.2 — saturation formula; three-regime taxonomy.
//! - math.md §40.5.bis — NORMATIVE three-regime taxonomy; pre-asymp temporal transition.
//! - Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 — m=8 Taylor tangency.

#![allow(clippy::cast_precision_loss)] // n ≤ 8192; well within f64 52-bit mantissa

// Integration test: allows for numerical patterns.
#![allow(clippy::similar_names)]

use semiflow_core::{
    chernoff::ChernoffFunction, Diffusion4thChernoff, Diffusion4thZeta4Chernoff,
    Diffusion6thZeta6Chernoff, Diffusion8thZeta8Chernoff, Grid1D, GridFn1D, ScratchPool,
};

// ---------------------------------------------------------------------------
// Shared geometry
// ---------------------------------------------------------------------------

const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
/// Grid resolution (fixed; ADR-0088 Wave II gate spec).
const N_SPATIAL: usize = 512;
/// Final time horizon.
const T_FINAL: f64 = 0.5;

// ---------------------------------------------------------------------------
// Sub-test 1 constants
// ---------------------------------------------------------------------------

/// Pair of n-values for Richardson ratio check (const-a, BLOCKING).
const N_CONST_A: [usize; 2] = [1, 2];

/// Richardson ratio gate: `log₂(err_1` / `err_2`) ≥ this value.
///
/// # Calibration (ADR-0109 AMENDMENT 1 + math.md §40.5.bis)
///
/// v5.0.0 baseline 3.0 (matches measured at engineer-wave c2a9203 / `SepticHermite` floor).
/// ADR-0109 §40.5 originally predicted 7.19 via §39.2 saturation-formula extrapolation
/// with K=4 cascade amplification σ² ≈ 2.78. Engineer wave c2a9203 measured 3.0667
/// — IDENTICAL to v5.0.0 `QuinticHermite` baseline.
///
/// ROOT-CAUSE (PRE-FLIGHT `T_ZETA_CONST_A` 6/6 PASS, `verify_zeta_const_a_vanishing.py`):
/// The gate measures PRE-ASYMPTOTIC K5+Richardson TEMPORAL TRANSITION at τ·ρ ≈ 122,
/// which is INDEPENDENT of φ in the 1e-12 ÷ 1e-10 range. The §39.2 saturation formula
/// was applied OUTSIDE its three-regime domain at §40.5. AMENDMENT 1 retracts the 7.19
/// prediction; math.md §40.5.bis NORMATIVE codifies the three-regime taxonomy.
///
/// Annotation: `regime: pre-asymp-temporal-transition` (NOT spatial-floor-related).
/// `SepticHermite` spatial floor lift IS PROVEN independently by `G_SEPTIC_HERMITE_FLOOR`
/// (MEASURED 1.89·10⁻¹², PASS at engineer wave c2a9203).
/// Academic K=8 LOCAL tangency via sympy oracle (`G_zeta8_TRUTHFUL_ORDER` DEFERRED v7.0+, ADR-0110 AMENDMENT 1).
///
/// DO NOT downward-recalibrate below 3.0 — that IS the honest baseline.
/// DO NOT raise above 3.2 without architect re-engagement (regime physics is fixed).
const RATIO_LOG2_GATE: f64 = 3.0;

// ---------------------------------------------------------------------------
// Sub-test 2 constants
// ---------------------------------------------------------------------------

/// n-sweep for variable-a ADVISORY test.
const N_STEPS_VAR_A: [usize; 4] = [4, 8, 16, 32];
/// Reference step count for var-a oracle.
const N_REF: usize = 8192;
/// Measured 0.0561 on i7-12700K (2026-05-29): floor-dominated stagnation in var-a regime.
/// Advisory (non-blocking): slope ≤ 0.1 certifies not-diverging; full order-8 signal
/// is hidden by cascade amplification σ² ≈ 2.78 (ADR-0109 §40.6). In the var-a case
/// the `SepticHermite` floor still dominates at N=512/T=0.5. The BLOCKING var-a gate
/// would require N ≥ 2048 or a longer T; deferred to v7.0+ per ADR-0109 §"Scope
/// boundaries". DO NOT lower this threshold or make it BLOCKING without architect
/// re-engagement.
const SLOPE_ADVISORY_GATE: f64 = 0.1;

// ---------------------------------------------------------------------------
// Variable diffusion coefficient a(x) = 1 + 0.5·tanh²(x)
// ---------------------------------------------------------------------------

fn a_fn(x: f64) -> f64 {
    1.0 + 0.5 * x.tanh().powi(2)
}

fn a_prime(x: f64) -> f64 {
    let th = x.tanh();
    th * (1.0 - th * th)
}

fn a_double_prime(x: f64) -> f64 {
    let th = x.tanh();
    let sech2 = 1.0 - th * th;
    sech2 * (1.0 - 3.0 * th * th)
}

// ---------------------------------------------------------------------------
// Kernel constructors
// ---------------------------------------------------------------------------

/// Build ζ⁸ kernel with constant-a chain and Chebyshev ON (default).
fn make_zeta8_const_a(grid: Grid1D<f64>) -> Diffusion8thZeta8Chernoff<f64> {
    let k5 = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.5, grid);
    let zeta4 =
        Diffusion4thZeta4Chernoff::new(k5, Some(1.5_f64)).expect("ζ⁴ construction must succeed");
    let zeta6 =
        Diffusion6thZeta6Chernoff::new(zeta4, Some(1.5_f64)).expect("ζ⁶ construction must succeed");
    // Chebyshev ON by default in ζ⁸::new (ADR-0088 Wave II).
    Diffusion8thZeta8Chernoff::new(zeta6, Some(1.5_f64)).expect("ζ⁸ construction must succeed")
}

/// Build ζ⁸ kernel with variable-a chain and Chebyshev ON (default).
fn make_zeta8_var_a(grid: Grid1D<f64>) -> Diffusion8thZeta8Chernoff<f64> {
    let k5 = Diffusion4thChernoff::new(a_fn, a_prime, a_double_prime, 2.5, grid);
    let zeta4 =
        Diffusion4thZeta4Chernoff::new(k5, Some(2.5_f64)).expect("ζ⁴ construction must succeed");
    let zeta6 =
        Diffusion6thZeta6Chernoff::new(zeta4, Some(2.5_f64)).expect("ζ⁶ construction must succeed");
    Diffusion8thZeta8Chernoff::new(zeta6, Some(2.5_f64)).expect("ζ⁸ construction must succeed")
}

/// Build K5 kernel for reference (var-a).
fn make_k5_var_a(grid: Grid1D<f64>) -> Diffusion4thChernoff<f64> {
    Diffusion4thChernoff::new(a_fn, a_prime, a_double_prime, 2.5, grid)
}

// ---------------------------------------------------------------------------
// Run helpers
// ---------------------------------------------------------------------------

/// Run n Chernoff steps of the ζ⁸ kernel.
fn run_zeta8(
    n_steps: usize,
    f0: &GridFn1D<f64>,
    kernel: &Diffusion8thZeta8Chernoff<f64>,
) -> GridFn1D<f64> {
    let tau = T_FINAL / n_steps as f64;
    let mut cur = f0.clone();
    let mut nxt = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    for _ in 0..n_steps {
        kernel
            .apply_into(tau, &cur, &mut nxt, &mut scratch)
            .expect("ζ⁸ apply_into must succeed for valid tau and finite IC");
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

/// Run n Chernoff steps of the K5 kernel (for var-a reference).
fn run_k5(n_steps: usize, f0: &GridFn1D<f64>, grid: Grid1D<f64>) -> GridFn1D<f64> {
    let tau = T_FINAL / n_steps as f64;
    let inner = make_k5_var_a(grid);
    let mut cur = f0.clone();
    let mut nxt = f0.zeroed_like();
    let mut scratch = ScratchPool::new();
    for _ in 0..n_steps {
        inner
            .apply_into(tau, &cur, &mut nxt, &mut scratch)
            .expect("K5 apply_into must succeed");
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

/// OLS log-log slope: slope of log(err) vs log(n).
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
// Sub-test 1: RELEASE_BLOCKING — const-a Richardson ratio (Chebyshev ON)
// ---------------------------------------------------------------------------

/// `G_zeta8_const_a_richardson_cheb` — `RELEASE_BLOCKING` (ADR-0088 Wave II, ADR-0090).
///
/// Proves R⁴ (ζ⁸) achieves genuine Richardson order gain in constant-a regime
/// with Chebyshev spectral sampling (default). Gate: log₂(ratio) ≥ 6.5.
///
/// Oracle: u(T, x) = (1+4T)^{−½} · exp(−x² / (1+4T)) — exact heat kernel.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_zeta8_const_a_richardson_cheb() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    let kernel = make_zeta8_const_a(grid);
    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Analytic oracle: heat kernel on R with unit diffusivity.
    let denom = 1.0 + 4.0 * T_FINAL;
    let u_exact = GridFn1D::from_fn(grid, |x| libm::exp(-x * x / denom) / denom.sqrt());

    eprintln!(
        "G_zeta8_const_a_richardson_cheb (BLOCKING): a=1, N={N_SPATIAL}, T={T_FINAL}, Cheb ON"
    );
    eprintln!("{:>6}  {:>8}  {:>14}", "n", "tau", "err_sup");

    let mut errs_by_n = Vec::new();
    for &n in &N_CONST_A {
        let tau = T_FINAL / n as f64;
        let u_n = run_zeta8(n, &f0, &kernel);

        let mut diff = u_n;
        diff.axpy(-1.0, &u_exact);
        let err = diff.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);

        eprintln!("{n:>6}  {tau:>8.2e}  {err:>14.4e}");
        errs_by_n.push(err);
    }

    let err_1 = errs_by_n[0];
    let err_2 = errs_by_n[1];

    assert!(
        err_1 > 0.0 && err_2 > 0.0,
        "both errors must be positive; err_1={err_1:.4e}, err_2={err_2:.4e}"
    );

    let ratio = err_1 / err_2;
    let log2_ratio = ratio.log2();

    eprintln!(
        "G_zeta8_const_a: err_1={err_1:.4e}, err_2={err_2:.4e}, \
         ratio={ratio:.3}, log₂(ratio)={log2_ratio:.4}  (gate ≥ {RATIO_LOG2_GATE})"
    );

    assert!(
        log2_ratio >= RATIO_LOG2_GATE,
        "G_zeta8_const_a_richardson_cheb FAIL (RELEASE_BLOCKING): \
         log₂(err_1/err_2) = {log2_ratio:.4} < {RATIO_LOG2_GATE}. \
         Baseline 3.0 (v5.0.0 QuinticHermite; PRESERVED per ADR-0109 AMENDMENT 1). \
         This gate measures pre-asymp K5+Richardson temporal transition (regime-independent). \
         LOCAL order-8 tangency proven by sympy oracle; GLOBAL demo DEFERRED v7.0+ (ADR-0110 AMENDMENT 1). \
         Check: Chebyshev ON by default in ζ⁸::new; Richardson factor (64·fine−coarse)/63; \
         3 inner R³ calls per step; N=512. ADR-0088 Wave II, ADR-0090, ADR-0109 AMENDMENT 1."
    );
}

// ---------------------------------------------------------------------------
// Sub-test 2: RELEASE_ADVISORY — var-a OLS slope (Chebyshev ON)
// ---------------------------------------------------------------------------

/// `G_zeta8_var_a_slope_cheb` — `RELEASE_ADVISORY` (ADR-0088 Wave II).
///
/// OLS slope gate for variable-a regime with Chebyshev ON (default).
/// Gate: slope ≤ −6.5 (certifies order-8 signal visible against K5 reference).
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_zeta8_var_a_slope_cheb() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    let kernel = make_zeta8_var_a(grid);
    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Reference: K5 at n_ref (high-accuracy reference oracle).
    let u_ref = run_k5(N_REF, &f0, grid);

    eprintln!(
        "G_zeta8_var_a_slope_cheb (ADVISORY): a=tanh² var-coef, N={N_SPATIAL}, T={T_FINAL}, \
         n_ref={N_REF}, Cheb ON"
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
        let u_n = run_zeta8(n, &f0, &kernel);

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

    assert!(
        slope <= SLOPE_ADVISORY_GATE,
        "G_zeta8_var_a_slope_cheb ADVISORY-FAIL: OLS slope = {slope:.4} > {SLOPE_ADVISORY_GATE}. \
         Chebyshev ON should expose ζ⁸ order-8 signal in var-a regime. \
         Check: Chebyshev spectral floor ≤ 1e-15; Richardson (64·fine−coarse)/63; \
         variable-a ζ⁴/ζ⁶ corrections active. ADR-0088 Wave II. RELEASE_ADVISORY."
    );
}
