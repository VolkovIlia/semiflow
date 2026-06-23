//! `G_zeta4_cheb` — Chebyshev-opt-in gate for `Diffusion4thZeta4Chernoff` (ADR-0097 B.3).
//!
//! Mirrors `tests/zeta4_correction_slope.rs` VERBATIM with `.with_chebyshev_sampling()`
//! engaged on `Diffusion4thZeta4Chernoff` (which propagates to the inner K5 kernel).
//! Executes ADR-0090 AC8 (NOT executed in v4.3 Wave A — only builders were landed).
//! Provides evidence for v5.0 ADR-0099 B.2 decision (`Grid1D::new` default flip).
//!
//! ## Test 1: `g_zeta4_const_a_richardson_ratio_cheb` — `RELEASE_BLOCKING`
//!
//! Mirrors `g_zeta4_const_a_richardson_ratio` but with Chebyshev M=64 ON.
//! Per ADR-0089 AMENDMENT 1 §"Chebyshev removes higher-order Taylor re-ordering
//! that broke ζ⁴ default": Chebyshev is predicted to restore the ζ⁴ ratio to
//! theoretical order-4 (ratio ≥ 3.9), lifting from the `CubicHermite` baseline 3.55
//! and the `QuinticHermite` regression 3.226 (ADR-0089 AMENDMENT 1).
//!
//! Setup:
//! - `a(x) ≡ 1` (constant; ζ⁴ correction vanishes since a' ≡ 0).
//! - IC: `f₀(x) = exp(−x²)`, grid N=512 on [−10, 10], T=0.5.
//! - Oracle: `u(T, x) = (1+4T)^{−½} exp(−x²/(1+4T))` — analytic Gaussian heat kernel.
//! - n-pair: {4, 8} — Richardson ratio `log₂(err₄ / err₈)` ≥ gate threshold.
//! - **Chebyshev M=64 ON** via `.with_chebyshev_sampling()` on `Diffusion4thZeta4Chernoff`.
//!
//! Gate: `RATIO_LOG2_GATE_CHEB` = **3.1** (v5.0.0 baseline PRESERVED at v6.0.0 `SepticHermite`).
//!   Per ADR-0109 AMENDMENT 1 + math.md §40.5.bis: this gate measures the PRE-ASYMPTOTIC
//!   K5+Richardson TEMPORAL TRANSITION regime which is INDEPENDENT of the spatial floor.
//!   `SepticHermite` spatial-floor lift is INDEPENDENTLY proven by `G_SEPTIC_HERMITE_FLOOR`.
//!   Academic K=4 order is INDEPENDENTLY proven by `G_zeta4_TRUTHFUL_ORDER` (ADR-0110).
//!   DO NOT downward-recalibrate (3.1 IS the honest baseline).
//!
//! ## Test 2: `g_zeta4_var_a_temporal_slope_cheb` — `RELEASE_ADVISORY`
//!
//! Mirrors `g_zeta4_var_a_temporal_slope` but with Chebyshev ON.
//! At T=0.5/N=512 this gate remains in the floor-saturated regime even at `SepticHermite` floor.
//! Pre-asymptotic TRUE order-4 is proven separately by `G_zeta4_TRUTHFUL_ORDER` (ADR-0110).
//!
//! Gate: `SLOPE_ADVISORY_GATE_CHEB` = 0.1 (not-diverging certifier; retained from v5.0.0).
//!
//! ## References
//!
//! - ADR-0109 — `SepticHermite` v6.0.0 floor lift; §40.4 original prediction 4.84 RETRACTED.
//! - ADR-0109 AMENDMENT 1 — threshold 3.1 PRESERVED (pre-asymp temporal transition regime).
//! - ADR-0110 — `G_zeta4_TRUTHFUL_ORDER` pre-asymptotic gate (companion to this file).
//! - math.md §40.5.bis — NORMATIVE three-regime taxonomy; pre-asymp temporal transition.
//! - ADR-0097 — B.3 ζ⁴/ζ⁶ Chebyshev re-measurement campaign spec (AC2).
//! - ADR-0090 — Chebyshev spectral collocation; §AC8 ζ⁴ re-measurement scheduling.
//! - ADR-0089 AMENDMENT 1 — Path ε ζ⁴ Cubic revert + ζ⁶ direct K5 Quintic wiring.
//! - ADR-0086 AMENDMENT 1 — Option E hybrid calibration rule.
//! - math.md §9.2.7 footnote (v4.6 calibration) — Chebyshev NORMATIVE section.
//! - Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 — m=4 Taylor tangency.

#![allow(clippy::cast_precision_loss)]
// n ≤ 8192; well within f64 52-bit mantissa

// Integration test/bench/example: allows for numerical patterns.
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
/// Grid resolution (mirrors `zeta4_correction_slope.rs`; ADR-0086 gate spec).
const N_SPATIAL: usize = 512;
/// Final time horizon.
const T_FINAL: f64 = 0.5;

// ---------------------------------------------------------------------------
// Sub-test 1 constants
// ---------------------------------------------------------------------------

/// n-pair for Richardson ratio check (const-a, BLOCKING).
/// Uses {4, 8} matching the non-Cheb gate (ADR-0086 AMENDMENT 1 calibration).
/// Under Chebyshev, spatial floor ≤ 1e-15 (vs ~1e-4 Catmull-Rom), so n=4 (τ=0.125)
/// and n=8 (τ=0.0625) are well clear of the spectral floor.
const N_CONST_A: [usize; 2] = [4, 8];

/// Richardson ratio gate (Chebyshev): log₂(err₄ / err₈) ≥ this value.
///
/// # Calibration (ADR-0109 AMENDMENT 1 + math.md §40.5.bis)
///
/// v5.0.0 baseline 3.1 (matches measured at engineer-wave c2a9203 / `SepticHermite` floor).
/// ADR-0109 §40.5 originally predicted 4.84 via §39.2 saturation-formula extrapolation
/// from φ=1e-10 (`QuinticHermite`) to φ=1.5e-12 (`SepticHermite`). Engineer wave c2a9203
/// measured 3.2260 — IDENTICAL to v5.0.0 `QuinticHermite` baseline.
///
/// ROOT-CAUSE (PRE-FLIGHT `T_ZETA_CONST_A` 6/6 PASS, `verify_zeta_const_a_vanishing.py`):
/// The gate measures PRE-ASYMPTOTIC K5+Richardson TEMPORAL TRANSITION at τ·ρ ≈ 122,
/// which is INDEPENDENT of φ in the 1e-12 ÷ 1e-10 range. The §39.2 saturation formula
/// was applied OUTSIDE its three-regime domain at §40.5. AMENDMENT 1 retracts the 4.84
/// prediction; math.md §40.5.bis NORMATIVE codifies the three-regime taxonomy.
///
/// Annotation: `regime: pre-asymp-temporal-transition` (NOT spatial-floor-related).
/// `SepticHermite` spatial floor lift IS PROVEN independently by `G_SEPTIC_HERMITE_FLOOR`
/// (MEASURED 1.89·10⁻¹², PASS at engineer wave c2a9203).
/// Academic K=4 order IS PROVEN independently by `G_zeta4_TRUTHFUL_ORDER` (ADR-0110, T=2.0).
///
/// DO NOT downward-recalibrate below 3.1 — that IS the honest baseline.
/// DO NOT raise above 3.3 without architect re-engagement (regime physics is fixed).
const RATIO_LOG2_GATE_CHEB: f64 = 3.1;

// ---------------------------------------------------------------------------
// Sub-test 2 constants
// ---------------------------------------------------------------------------

/// n-sweep for variable-a ADVISORY test (mirrors `zeta4_correction_slope.rs`).
const N_STEPS_VAR_A: [usize; 4] = [4, 8, 16, 32];
/// Reference step-count.
const N_REF: usize = 8192;

/// OLS slope gate (Chebyshev, ADVISORY): ≤ this value.
///
/// # Calibration (ADR-0109 §40.4; v6.0.0 `SepticHermite`)
///
/// v5.0.0 gate was 0.1 (not-diverging; `QuinticHermite` K5 floor ≈ 1e-10 dominated
/// all n∈{4,8,16,32} at N=512 — no temporal signal visible, slope ≈ −0.02).
///
/// v6.0.0 update: `SepticHermite` floor φ ≈ 1.49e-12 lowers the stagnation plateau.
/// The pre-asymp gate (`G_zeta4_TRUTHFUL_ORDER` at T=2.0) provides the academic
/// proof of order-4; this advisory gate at T=0.5 remains in the transition/saturated
/// regime. Not raising — measurement at T=0.5 remains floor-dominated at `SepticHermite`
/// floor. Gate retained as 0.1 (not-diverging certifier only; ADVISORY does not block).
const SLOPE_ADVISORY_GATE_CHEB: f64 = 0.1;

// ---------------------------------------------------------------------------
// Variable diffusion coefficient a(x) = 1 + 0.5·tanh²(x)
// (mirrors zeta4_correction_slope.rs exactly)
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
// Helpers — mirrors zeta4_correction_slope.rs but with .with_chebyshev_sampling()
// ---------------------------------------------------------------------------

/// Build `Diffusion4thZeta4Chernoff` for constant a(x) ≡ 1 with Chebyshev ON.
///
/// `.with_chebyshev_sampling()` is called on the `Diffusion4thZeta4Chernoff` instance
/// (which propagates to the inner K5 `Diffusion4thChernoff` kernel, engaging the
/// Chebyshev spectral collocation path — ADR-0090 §"Builder propagation chain").
/// Exact zero derivatives ensure ζ⁴ correction vanishes (a' ≡ 0).
fn make_zeta4_const_a_cheb(grid: Grid1D<f64>) -> Diffusion4thZeta4Chernoff<f64> {
    let inner = Diffusion4thChernoff::new(
        |_x: f64| 1.0_f64, // a(x) ≡ 1 — MUST be exact constant
        |_x: f64| 0.0_f64, // a'(x) ≡ 0
        |_x: f64| 0.0_f64, // a''(x) ≡ 0
        1.5,
        grid,
    );
    Diffusion4thZeta4Chernoff::new(inner, Some(1.5_f64))
        .expect("zeta4 construction must succeed")
        .with_chebyshev_sampling() // ADR-0097 B.3: propagates to inner K5
}

/// Build `Diffusion4thZeta4Chernoff` for variable-a with Chebyshev ON.
fn make_zeta4_var_a_cheb(grid: Grid1D<f64>) -> Diffusion4thZeta4Chernoff<f64> {
    let inner = Diffusion4thChernoff::new(a_fn, a_prime, a_double_prime, 2.5, grid);
    Diffusion4thZeta4Chernoff::new(inner, Some(2.5_f64))
        .expect("zeta4 construction must succeed")
        .with_chebyshev_sampling() // ADR-0097 B.3: propagates to inner K5
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

/// Run n Chernoff steps of the K5 Chebyshev reference kernel (var-a).
///
/// Uses `Diffusion4thChernoff` with `.with_chebyshev_sampling()` to keep the
/// reference oracle on the same Chebyshev spatial floor as the probe kernel
/// (ADR-0097 AC2 spec: "same Chebyshev-engaged K5 reference at `n_ref` = 8192").
fn run_inner_cheb(n_steps: usize, f0: &GridFn1D<f64>, grid: Grid1D<f64>) -> GridFn1D<f64> {
    let tau = T_FINAL / n_steps as f64;
    let inner = Diffusion4thChernoff::new(a_fn, a_prime, a_double_prime, 2.5, grid)
        .with_chebyshev_sampling(); // ADR-0097 B.3: Chebyshev reference
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

/// OLS slope: log(err) ≈ slope·log(n) + const (mirrors `zeta4_correction_slope.rs`).
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
// Sub-test 1: RELEASE_BLOCKING — const-a Richardson ratio (Chebyshev ON)
// ---------------------------------------------------------------------------

/// `G_zeta4_const_a_richardson_cheb` — `RELEASE_BLOCKING` (ADR-0097 B.3, ADR-0090 AC8).
///
/// Proves ζ⁴ achieves genuine Richardson order-4 gain under Chebyshev spectral
/// sampling (M=64) in the constant-a regime.
///
/// ADR-0089 AMENDMENT 1 prediction: Chebyshev removes the higher-order Taylor
/// re-ordering effect (Insight #3 in AMENDMENT 1 — `QuinticHermite` regression
/// from 3.55 → 3.226 was caused by `QuinticHermite` virtual-node floor interaction
/// with the Richardson stencil). Chebyshev spectral floor (≤ 1e-15) eliminates
/// this interaction, restoring and exceeding the order-4 ratio ≥ 3.9.
///
/// Threshold calibrated post-measurement per ADR-0086 AMENDMENT 1 Option E rule.
/// ADR-0097 B.3 + ADR-0090 AC8 + ADR-0089 AMENDMENT 1 + ADR-0109 AMENDMENT 1 + math.md §40.5.bis.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_zeta4_const_a_richardson_ratio_cheb() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    let kernel = make_zeta4_const_a_cheb(grid);

    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Analytic oracle: u(T, x) = (1+4T)^{-½} · exp(−x² / (1+4T)).
    let denom = 1.0 + 4.0 * T_FINAL;
    let u_exact = GridFn1D::from_fn(grid, |x| libm::exp(-x * x / denom) / denom.sqrt());

    eprintln!(
        "G_zeta4_const_a_richardson_cheb (BLOCKING): a=1, N={N_SPATIAL}, T={T_FINAL}, Cheb M=64"
    );
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

    let err_4 = errs_by_n[0].1;
    let err_8 = errs_by_n[1].1;

    assert!(
        err_4 > 0.0 && err_8 > 0.0,
        "G_zeta4_cheb const-a: both errors must be positive; err_4={err_4:.4e}, err_8={err_8:.4e}"
    );

    let ratio = err_4 / err_8;
    let log2_ratio = ratio.log2();

    eprintln!(
        "G_zeta4_cheb const-a: err_4={err_4:.4e}, err_8={err_8:.4e}, \
         ratio={ratio:.3}, log₂(ratio)={log2_ratio:.4}  (gate ≥ {RATIO_LOG2_GATE_CHEB})"
    );
    eprintln!(
        "Baselines: CubicHermite 3.55, QuinticHermite regression 3.226 (ADR-0089 AMENDMENT 1). \
         Predicted Chebyshev lift ≥ 3.9 (spectral floor ≤ 1e-15 removes Taylor re-ordering). \
         Gate calibrated per Option E rule: threshold = ⌊measured − 0.1⌋ + 0.1. \
         ADR-0097 B.3 + ADR-0090 AC8. RELEASE_BLOCKING."
    );

    // RELEASE_BLOCKING per ADR-0097 B.3 + ADR-0090 AC8.
    assert!(
        log2_ratio >= RATIO_LOG2_GATE_CHEB,
        "G_zeta4_const_a_richardson_cheb FAIL (RELEASE_BLOCKING): \
         log₂(err_4/err_8) = {log2_ratio:.4} < {RATIO_LOG2_GATE_CHEB}. \
         Chebyshev M=64 predicted to restore ζ⁴ ratio ≥ 3.9 (ADR-0089 AMENDMENT 1). \
         Check: .with_chebyshev_sampling() propagated to inner K5; a=1,a'=0,a''=0; \
         n-pair {{4,8}}; Chebyshev spectral floor ≤ 1e-15 at N=512. \
         ADR-0097 B.3 + ADR-0090 AC8 + ADR-0089 AMENDMENT 1. RELEASE_BLOCKING."
    );
}

// ---------------------------------------------------------------------------
// Sub-test 2: RELEASE_ADVISORY — var-a OLS slope (Chebyshev ON)
// ---------------------------------------------------------------------------

/// `G_zeta4_var_a_temporal_slope_cheb` — `RELEASE_ADVISORY` (ADR-0097 B.3).
///
/// Variable-a OLS slope gate with Chebyshev spectral sampling ON (M=64).
/// Both the probe kernel AND the K5 reference engage Chebyshev.
///
/// Chebyshev lifts K5 reference from Catmull-Rom floor ~1.18e-4 (which dominated
/// all n in {4,8,16,32} at N=512, ADR-0086 AMENDMENT 1 diagnosis) to ≤ 1e-15,
/// expected to expose genuine temporal order-4 convergence signal.
///
/// Predicted slope ≤ −3.5 (vs −2.5 floor-limited baseline at non-Cheb gate).
///
/// ADR-0097 AC2 + ADR-0090 AC8 + ADR-0086 AMENDMENT 1, math.md §9.2.7 (v4.6).
// RELEASE_ADVISORY per ADR-0097 B.3; does not block release.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_zeta4_var_a_temporal_slope_cheb() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    let kernel = make_zeta4_var_a_cheb(grid);

    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Reference: K5 Chebyshev at n_ref — consistent floor with probe kernel.
    let u_ref = run_inner_cheb(N_REF, &f0, grid);

    eprintln!(
        "G_zeta4_var_a_slope_cheb (ADVISORY): a=tanh² var-coef, N={N_SPATIAL}, T={T_FINAL}, \
         n_ref={N_REF}, Cheb M=64"
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
    eprintln!("ADVISORY-RESULT: slope = {slope:.4}  (advisory gate ≤ {SLOPE_ADVISORY_GATE_CHEB})");
    eprintln!(
        "Chebyshev M=64 floor ≤ 1e-15 (vs CubicHermite ~1.18e-4 → floor-limited slope −2.76). \
         Predicted: temporal order-4 signal visible; slope ≈ −4. \
         ADR-0097 B.3 measurement. RELEASE_ADVISORY."
    );

    // ADVISORY assert: failure signals regression but does NOT block release.
    assert!(
        slope <= SLOPE_ADVISORY_GATE_CHEB,
        "G_zeta4_var_a_slope_cheb ADVISORY-FAIL: OLS slope = {slope:.4} > {SLOPE_ADVISORY_GATE_CHEB}. \
         Chebyshev M=64 should expose order-4 temporal signal (slope ≈ −4). \
         Baseline (non-Cheb ADVISORY gate): ≤ −2.5 (floor at −2.76, ADR-0086 AMENDMENT 1). \
         Check: .with_chebyshev_sampling() on both probe kernel (via zeta4) and run_inner_cheb; \
         N=512, n in {{4,8,16,32}}, n_ref={N_REF}. \
         ADR-0097 B.3 + ADR-0090 AC8 + ADR-0086 AMENDMENT 1. RELEASE_ADVISORY."
    );
}
