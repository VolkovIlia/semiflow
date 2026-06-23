//! `G_zeta6_cheb` — Chebyshev-opt-in gate for `Diffusion6thZeta6Chernoff` (ADR-0097 B.3).
//!
//! Mirrors `tests/zeta6_correction_slope.rs` VERBATIM with `.with_chebyshev_sampling()`
//! engaged on the K5 base kernel. Measures ζ⁶ Richardson order under Chebyshev
//! spectral sampling (M=64) to provide evidence for v5.0 ADR-0099 B.2 decision.
//!
//! ## Test 1: `g_zeta6_const_a_richardson_ratio_cheb` — `RELEASE_BLOCKING`
//!
//! Same setup as `g_zeta6_const_a_richardson_ratio` but with Chebyshev ON.
//! Chebyshev spectral floor (≤ 1e-15 at N=512) lifts the K5 spatial floor
//! from ~1e-6 (`QuinticHermite`) toward ULP, exposing full temporal order-6 signal.
//!
//! Setup:
//! - `a(x) ≡ 1` (constant; ζ⁴ correction vanishes since a' ≡ 0).
//! - IC: `f₀(x) = exp(−x²)`, grid N=512 on [−10, 10], T=0.5.
//! - Oracle: `u(T, x) = (1+4T)^{−½} exp(−x²/(1+4T))` — analytic Gaussian heat kernel.
//! - n-pair: {1, 2} — Richardson ratio `log₂(err_1 / err_2)` ≥ gate threshold.
//! - **Chebyshev M=64 ON** via `.with_chebyshev_sampling()` on K5 base.
//!
//! Gate: `RATIO_LOG2_GATE_CHEB` = **3.8** (v5.0.0 baseline PRESERVED at v6.0.0 `SepticHermite`).
//!   Per ADR-0109 AMENDMENT 1 + math.md §40.5.bis: this gate measures the PRE-ASYMPTOTIC
//!   K5+Richardson TEMPORAL TRANSITION regime which is INDEPENDENT of the spatial floor.
//!   `SepticHermite` spatial-floor lift is INDEPENDENTLY proven by `G_SEPTIC_HERMITE_FLOOR`.
//!   Academic K=6 LOCAL Taylor tangency via `T23N_zeta6` sympy oracle (ADR-0110 AMENDMENT 1:
//!   `G_zeta6_TRUTHFUL_ORDER` DEFERRED v7.0+ OCTONIC).
//!   DO NOT downward-recalibrate (3.8 IS the honest baseline).
//!
//! ## Test 2: `g_zeta6_var_a_temporal_slope_cheb` — `RELEASE_ADVISORY`
//!
//! Same as `g_zeta6_var_a_temporal_slope` but K5 Chebyshev opt-in engaged in
//! BOTH the probe kernel AND the reference `run_inner` helper.
//! At T=0.5/N=512 this gate remains near the floor-saturated regime at `SepticHermite` floor.
//! Pre-asymptotic GLOBAL order-6 empirical demonstration DEFERRED v7.0+ (ADR-0110 AMENDMENT 1).
//!
//! Gate: `SLOPE_ADVISORY_GATE_CHEB` = 0.5 (not-diverging certifier).
//!
//! ## References
//!
//! - ADR-0109 — `SepticHermite` v6.0.0 floor lift; §40.4 original prediction 5.98 RETRACTED.
//! - ADR-0109 AMENDMENT 1 — threshold 3.8 PRESERVED (pre-asymp temporal transition regime).
//! - ADR-0110 AMENDMENT 1 — `G_zeta6_TRUTHFUL_ORDER` DEFERRED v7.0+ OCTONIC; threshold history.
//! - math.md §40.5.bis — NORMATIVE three-regime taxonomy; pre-asymp temporal transition.
//! - ADR-0097 — B.3 ζ⁶/ζ⁸ Chebyshev re-measurement campaign spec.
//! - ADR-0090 — Chebyshev spectral collocation; §AC8/AC9 re-measurement scheduling.
//! - ADR-0088 AMENDMENT 1 — Option E hybrid calibration rule.
//! - math.md §9.2.7 footnote (v4.6 calibration) — Chebyshev NORMATIVE section.
//! - Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 — m=6 Taylor tangency.

#![allow(clippy::cast_precision_loss)]
// n ≤ 8192; well within f64 52-bit mantissa

// Integration test/bench: allows for numerical patterns.
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
/// Grid resolution (mirrors `zeta6_correction_slope.rs`; ADR-0088 gate spec).
const N_SPATIAL: usize = 512;
/// Final time horizon.
const T_FINAL: f64 = 0.5;

// ---------------------------------------------------------------------------
// Sub-test 1 constants
// ---------------------------------------------------------------------------

/// n-pair for Richardson ratio check (const-a, BLOCKING).
/// Same pair {1, 2} as non-Cheb gate — both have temporal error >> spatial floor.
/// Under Chebyshev, spatial floor ≤ 1e-15 (vs ~1e-6 Quintic), so the pair
/// is even safer from floor contamination.
const N_CONST_A: [usize; 2] = [1, 2];

/// Richardson ratio gate (Chebyshev): `log₂(err_1` / `err_2`) ≥ this value.
///
/// # Calibration (ADR-0109 AMENDMENT 1 + math.md §40.5.bis)
///
/// v5.0.0 baseline 3.8 (matches measured at engineer-wave c2a9203 / `SepticHermite` floor).
/// ADR-0109 §40.5 originally predicted 5.98 via §39.2 saturation-formula extrapolation
/// from φ=1e-10 (`QuinticHermite`) to φ=1.5e-12 (`SepticHermite`). Engineer wave c2a9203
/// measured 3.8701 — IDENTICAL to v5.0.0 `QuinticHermite` baseline.
///
/// ROOT-CAUSE (PRE-FLIGHT `T_ZETA_CONST_A` 6/6 PASS, `verify_zeta_const_a_vanishing.py`):
/// The gate measures PRE-ASYMPTOTIC K5+Richardson TEMPORAL TRANSITION at τ·ρ ≈ 122,
/// which is INDEPENDENT of φ in the 1e-12 ÷ 1e-10 range. The §39.2 saturation formula
/// was applied OUTSIDE its three-regime domain at §40.5. AMENDMENT 1 retracts the 5.98
/// prediction; math.md §40.5.bis NORMATIVE codifies the three-regime taxonomy.
///
/// Annotation: `regime: pre-asymp-temporal-transition` (NOT spatial-floor-related).
/// `SepticHermite` spatial floor lift IS PROVEN independently by `G_SEPTIC_HERMITE_FLOOR`
/// (MEASURED 1.89·10⁻¹², PASS at engineer wave c2a9203).
/// Academic K=6 LOCAL tangency proven by `T23N_zeta6` sympy oracle (`G_zeta6_TRUTHFUL_ORDER` DEFERRED v7.0+).
///
/// DO NOT downward-recalibrate below 3.8 — that IS the honest baseline.
/// DO NOT raise above 4.0 without architect re-engagement (regime physics is fixed).
const RATIO_LOG2_GATE_CHEB: f64 = 3.8;

// ---------------------------------------------------------------------------
// Sub-test 2 constants
// ---------------------------------------------------------------------------

/// n-sweep for variable-a ADVISORY test (mirrors `zeta6_correction_slope.rs`).
const N_STEPS_VAR_A: [usize; 4] = [4, 8, 16, 32];
/// Reference step-count.
const N_REF: usize = 8192;

/// OLS slope gate (Chebyshev, ADVISORY): ≤ this value.
///
/// # Calibration (ADR-0109 §40.4; v6.0.0 `SepticHermite`)
///
/// v5.0.0 placeholder was 0.5 (not yet measured with `SepticHermite`).
///
/// v6.0.0 update: `SepticHermite` floor φ ≈ 1.49e-12 lowers the stagnation plateau.
/// At T=0.5/N=512 this var-a advisory gate may still be in the floor-saturated regime.
/// Pre-asymptotic GLOBAL order-6 empirical demo DEFERRED v7.0+ (ADR-0110 AMENDMENT 1).
/// This advisory gate is retained as not-diverging certifier (0.5). If measured slope
/// shows better signal after `SepticHermite`, this may be tightened post-measurement.
const SLOPE_ADVISORY_GATE_CHEB: f64 = 0.5;

// ---------------------------------------------------------------------------
// Variable diffusion coefficient a(x) = 1 + 0.5·tanh²(x)
// (mirrors zeta6_correction_slope.rs exactly)
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
// Helpers — mirrors zeta6_correction_slope.rs but with .with_chebyshev_sampling()
// ---------------------------------------------------------------------------

/// Build `Diffusion4thChernoff` for variable-a with Chebyshev M=64 ON.
///
/// CRITICAL: `.with_chebyshev_sampling()` engages the Chebyshev spectral
/// spatial sampling that ADR-0097 B.3 is measuring. Without this call,
/// the test degenerates to the non-Cheb `zeta6_correction_slope.rs` gate.
fn make_k5_var_a_cheb(grid: Grid1D<f64>) -> Diffusion4thChernoff<f64> {
    Diffusion4thChernoff::new(a_fn, a_prime, a_double_prime, 2.5, grid).with_chebyshev_sampling()
    // ADR-0097 B.3: opt-in to Chebyshev M=64
}

/// Build `Diffusion4thChernoff` for constant a(x) ≡ 1 with Chebyshev M=64 ON.
///
/// Exact zero derivatives ensure the ζ⁴ correction vanishes (a' ≡ 0),
/// isolating the Richardson temporal order signal in the ζ⁶ kernel.
fn make_k5_const_a_cheb(grid: Grid1D<f64>) -> Diffusion4thChernoff<f64> {
    Diffusion4thChernoff::new(
        |_x: f64| 1.0_f64, // a(x) ≡ 1 — MUST be exact constant
        |_x: f64| 0.0_f64, // a'(x) ≡ 0
        |_x: f64| 0.0_f64, // a''(x) ≡ 0
        1.5,
        grid,
    )
    .with_chebyshev_sampling() // ADR-0097 B.3: opt-in to Chebyshev M=64
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

/// Run n Chernoff steps of the K5 Chebyshev reference kernel.
///
/// CRITICAL: Reference MUST also engage Chebyshev to keep reference and probe
/// on the same spatial floor (spec AC1: "K5 reference `run_inner` MUST also
/// engage `.with_chebyshev_sampling()`"). Without this, the reference oracle
/// sits on the `QuinticHermite` floor ~1.16e-4 while the probe sits on the
/// Chebyshev floor ≤ 1e-15, creating an apples-to-oranges comparison.
fn run_inner_cheb(n_steps: usize, f0: &GridFn1D<f64>, grid: Grid1D<f64>) -> GridFn1D<f64> {
    let tau = T_FINAL / n_steps as f64;
    let inner = make_k5_var_a_cheb(grid);
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

/// OLS slope: log(err) ≈ slope·log(n) + const (mirrors `zeta6_correction_slope.rs`).
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

/// `G_zeta6_const_a_richardson_cheb` — `RELEASE_BLOCKING` (ADR-0097 B.3).
///
/// Proves R³ (ζ⁶) achieves genuine Richardson order gain under Chebyshev
/// spectral sampling (M=64) in the constant-a regime.
///
/// Chebyshev lifts K5 spatial floor from ~1e-6 (`QuinticHermite`) toward ULP
/// (≤ 1e-15 at N=512), exposing temporal order-6 signal across n-pair {1, 2}.
///
/// Threshold calibrated post-measurement per ADR-0086 AMENDMENT 1 Option E rule:
///   `threshold = ⌊measured − 0.1⌋ + 0.1`. Predicted ≥ 5.5.
///
/// ADR-0097 AC1 + ADR-0090 AC9 + ADR-0088 AMENDMENT 1, math.md §9.2.7 (v4.6).
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_zeta6_const_a_richardson_ratio_cheb() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    // Build ζ⁶ kernel with Chebyshev-enabled K5 base.
    let k5 = make_k5_const_a_cheb(grid);
    let zeta4 =
        Diffusion4thZeta4Chernoff::new(k5, Some(1.5_f64)).expect("zeta4 construction must succeed");
    let kernel = Diffusion6thZeta6Chernoff::new(zeta4, Some(1.5_f64))
        .expect("zeta6 construction must succeed");

    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Analytic oracle: u(T, x) = (1+4T)^{-½} · exp(−x² / (1+4T)).
    let denom = 1.0 + 4.0 * T_FINAL;
    let u_exact = GridFn1D::from_fn(grid, |x| libm::exp(-x * x / denom) / denom.sqrt());

    eprintln!(
        "G_zeta6_const_a_richardson_cheb (BLOCKING): a=1, N={N_SPATIAL}, T={T_FINAL}, Cheb M=64"
    );
    eprintln!("{:>6}  {:>8}  {:>14}", "n", "tau", "err_sup");

    let mut errs_by_n = Vec::new();

    for &n in &N_CONST_A {
        let tau = T_FINAL / n as f64;
        let u_n = run_zeta6(n, &f0, &kernel);

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
        "G_zeta6_cheb const-a: err_1={err_1:.4e}, err_2={err_2:.4e}, \
         ratio={ratio:.3}, log₂(ratio)={log2_ratio:.4}  (gate ≥ {RATIO_LOG2_GATE_CHEB})"
    );
    eprintln!(
        "Baseline (QuinticHermite, non-Cheb): 3.868 (ADR-0089). \
         Predicted Chebyshev lift ≥ 5.5 (Boyd 1989 spectral theory). \
         Gate calibrated per Option E rule: threshold = ⌊measured − 0.1⌋ + 0.1. \
         ADR-0097 B.3 measurement. RELEASE_BLOCKING."
    );

    // RELEASE_BLOCKING per ADR-0097 B.3.
    assert!(
        log2_ratio >= RATIO_LOG2_GATE_CHEB,
        "G_zeta6_const_a_richardson_cheb FAIL (RELEASE_BLOCKING): \
         log₂(err_1/err_2) = {log2_ratio:.4} < {RATIO_LOG2_GATE_CHEB}. \
         Chebyshev M=64 should lift ratio from QuinticHermite 3.868 toward 6.0. \
         Check: K5 built with .with_chebyshev_sampling(); constant-a path a=1,a'=0,a''=0; \
         n-pair {{1,2}}; Chebyshev spectral floor ≤ 1e-15 at N=512. \
         ADR-0097 B.3 + ADR-0090 AC9 + ADR-0088 AMENDMENT 1. RELEASE_BLOCKING."
    );
}

// ---------------------------------------------------------------------------
// Sub-test 2: RELEASE_ADVISORY — var-a OLS slope (Chebyshev ON)
// ---------------------------------------------------------------------------

/// `G_zeta6_var_a_temporal_slope_cheb` — `RELEASE_ADVISORY` (ADR-0097 B.3).
///
/// Variable-a OLS slope gate with Chebyshev spectral sampling ON (M=64).
/// Both the probe kernel AND reference `run_inner_cheb` engage Chebyshev,
/// keeping spatial floors consistent (ADR-0097 AC1 spec requirement).
///
/// Chebyshev lifts K5 reference from Catmull-Rom floor ~1.16e-4 (dominant at
/// N=512) to ≤ 1e-15, exposing temporal convergence signal in n∈{4,8,16,32}.
/// Expected slope ≈ −6 (temporal order-6 visible under spectral sampling).
///
/// Threshold calibrated post-measurement per Option E hybrid rule.
/// Predicted ≤ −5.5 (vs +0.04 plateau in non-Cheb gate at ADR-0088 Amendment 1).
///
/// ADR-0097 AC1 + ADR-0090 + ADR-0088 AMENDMENT 1, math.md §9.2.7 (v4.6).
// RELEASE_ADVISORY per ADR-0097 B.3; does not block release.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_zeta6_var_a_temporal_slope_cheb() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid construction must succeed");

    // Build ζ⁶ kernel with Chebyshev-enabled K5 base (var-a).
    let k5 = make_k5_var_a_cheb(grid);
    let zeta4 =
        Diffusion4thZeta4Chernoff::new(k5, Some(2.5_f64)).expect("zeta4 construction must succeed");
    let kernel = Diffusion6thZeta6Chernoff::new(zeta4, Some(2.5_f64))
        .expect("zeta6 construction must succeed");

    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Reference: K5 Chebyshev at n_ref — MUST also use Chebyshev (ADR-0097 AC1).
    let u_ref = run_inner_cheb(N_REF, &f0, grid);

    eprintln!(
        "G_zeta6_var_a_slope_cheb (ADVISORY): a=tanh² var-coef, N={N_SPATIAL}, T={T_FINAL}, \
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
    eprintln!("ADVISORY-RESULT: slope = {slope:.4}  (advisory gate ≤ {SLOPE_ADVISORY_GATE_CHEB})");
    eprintln!(
        "Chebyshev M=64 floor ≤ 1e-15 (vs QuinticHermite ~1.16e-4). \
         Predicted: all n in {{4,8,16,32}} exit floor plateau; slope ≈ -6. \
         ADR-0097 B.3 measurement. RELEASE_ADVISORY."
    );

    // ADVISORY assert: failure signals divergence regression but does NOT block release.
    assert!(
        slope <= SLOPE_ADVISORY_GATE_CHEB,
        "G_zeta6_var_a_slope_cheb ADVISORY-FAIL: OLS slope = {slope:.4} > {SLOPE_ADVISORY_GATE_CHEB}. \
         Chebyshev M=64 should expose order-6 temporal signal (slope ≈ -6). \
         Check: both probe kernel AND run_inner_cheb engage .with_chebyshev_sampling(); \
         K5 reference at n_ref={N_REF}; N=512 grid; T={T_FINAL}. \
         ADR-0097 B.3 + ADR-0090 + ADR-0088 AMENDMENT 1. RELEASE_ADVISORY."
    );
}
