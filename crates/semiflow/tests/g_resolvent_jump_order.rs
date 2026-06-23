//! `G_RESOLVENT_JUMP_ORDER` — large-step resolvent time-jump self-convergence gate.
//!
//! Gate spec (properties.yaml v8.0.0, `RELEASE_BLOCKING`, ADR-0134, §47.5):
//!
//!   OLS slope d `log‖jump_M` − ref‖_∞ / d log(1/M) ≥ +1.95
//!   M ∈ {6, 8, 10, 12, 14}, t = 100, N = 64, Gaussian IC, Neumann BC.
//!
//!   G24 convention: positive slope = convergence as M grows. The contour is
//!   geometrically convergent (TWS 2006 Theorem 47.1); the pre-flight oracle
//!   measured slope ≈ +9.86, far above the floor ≥ +1.95.
//!
//!   Also checks t-independence: error at M = 16 across t ∈ {1, 20, 100}
//!   spreads ≤ 10× (cost decoupling via M/t contour scaling, §47.2).
//!
//! ## Spectral oracle and reference design
//!
//! Operator: A = ∂²ₓ on [−5, 5], N = 64, Neumann BC (3-pt stencil).
//! `resolve_lhp` in the kernel assembles and solves (λI − A) for the same discrete
//! Neumann stencil. So `jump()` approximates e^{t · `A_discrete`} · g.
//!
//! The reference MUST target the same discrete operator.
//! `DiffusionChernoff::apply_into` applies a continuous Gaussian convolution
//! (approximates e^{τ · ∂`²ₓ_continuous`}), accumulating O(t · dx²) ≈ 2.5 offset
//! vs e^{t · `A_discrete`} at t = 100, N = 64 — this saturates the error at ~2.8e-3
//! regardless of the Chernoff node count (confirmed by diagnostic run).
//!
//! Self-convergence reference: `ResolventJumpChernoff` at `M_ref` = 40 targets the
//! same discrete A. At t = 100, `M_ref` = 40 has error ~1.9e-8 vs scipy expm, well
//! below the M = 14 probe error ~1.6e-7 (confirmed by Python oracle).
//!
//! ## Node count and round-off floor
//!
//! At M = 14, t = 100: pre-flight gives err ≈ 1.6e-7 (well above f64 floor ~1e-14).
//! At M = 6 the error is O(7e-4). The spec M ∈ {6..14} keeps the gate in the
//! geometric-convergence regime, not the round-off plateau.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize→f64 in OLS/sweep; values ≤ 40 ≤ 2^52

use semiflow::{DiffusionChernoff, Grid1D, GridFn1D, ResolventJumpChernoff};

// ---------------------------------------------------------------------------
// Gate constants — do NOT relax without ADR + properties.yaml bump.
// ---------------------------------------------------------------------------

/// OLS slope gate: d log(err) / d log(1/M) ≥ +1.95 (G24 convention).
const SLOPE_GATE: f64 = 1.95;

/// Spread gate for t-independence sub-check: `max_err` / `min_err` ≤ 10× at M=16.
const SPREAD_GATE: f64 = 10.0;

/// Spatial grid size (matches Python oracle canonical setup).
const N: usize = 64;

/// Domain half-width.
const L: f64 = 5.0;

/// Large horizon for the slope sub-test (§47.5, ADR-0134).
const T_SLOPE: f64 = 100.0;

/// Contour-node sweep for slope test (§47.5; keeps errors above f64 floor).
const M_SWEEP: [usize; 5] = [6, 8, 10, 12, 14];

/// Self-convergence reference: `M_ref=40` has error ~1.9e-8 at t=100 (vs exact),
/// well below the M=14 probe error ~1.6e-7 — accurate reference for slope measurement.
const M_REF: usize = 40;

/// Node count for the t-independence sub-check.
const M_T_INDEP: usize = 16;

/// Time values for t-independence sub-check (§47.2, t-independence claim).
const T_INDEP: [f64; 3] = [1.0, 20.0, 100.0];

// ---------------------------------------------------------------------------
// Coefficient functions (fn-pointers, required by DiffusionChernoff::new)
// ---------------------------------------------------------------------------

fn a_one(_: f64) -> f64 {
    1.0
}
fn a_zero(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Helper: sup-norm of pointwise difference
// ---------------------------------------------------------------------------

fn pointwise_sup_err(a: &GridFn1D<f64>, b: &GridFn1D<f64>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Helper: OLS slope of log(y) vs log(x) — G24 convention
// ---------------------------------------------------------------------------

fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let lx: Vec<f64> = xs.iter().map(|&v| v.ln()).collect();
    let ly: Vec<f64> = ys.iter().map(|&v| v.ln()).collect();
    let mx = lx.iter().sum::<f64>() / m;
    let my = ly.iter().sum::<f64>() / m;
    let num: f64 = lx
        .iter()
        .zip(ly.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let den: f64 = lx.iter().map(|x| (x - mx).powi(2)).sum();
    num / den
}

// ---------------------------------------------------------------------------
// Helper: run jump at node count `m`, return result
// ---------------------------------------------------------------------------

fn run_jump(grid: Grid1D<f64>, m: usize, t: f64, g: &GridFn1D<f64>) -> GridFn1D<f64> {
    let kernel = DiffusionChernoff::new(a_one, a_zero, a_zero, 1.0, grid);
    let rj = ResolventJumpChernoff::new(kernel, m, grid).unwrap();
    rj.jump(t, g).unwrap()
}

// ---------------------------------------------------------------------------
// G_RESOLVENT_JUMP_ORDER gate
// ---------------------------------------------------------------------------

/// `G_RESOLVENT_JUMP_ORDER` — slope ≥ +1.95 (`RELEASE_BLOCKING`, ADR-0134).
///
/// Sub-test A: OLS slope of log(err) vs log(1/M) at t=100, M ∈ {6,8,10,12,14}.
///   Reference: `ResolventJumpChernoff` at `M_ref=40` (self-convergence; same discrete A).
/// Sub-test B: t-independence — spread ≤ 10× at M=16, t ∈ {1, 20, 100}.
#[test]
#[ignore = "RELEASE_BLOCKING slow gate; run with: cargo run -p xtask -- test-flagship"]
fn g_resolvent_jump_order() {
    let grid = Grid1D::new(-L, L, N).unwrap();
    let g = GridFn1D::from_fn(grid, |x: f64| (-x * x).exp());

    // ---- Sub-test A: self-convergence slope at t=100, M ∈ {6,8,10,12,14} -

    // Self-convergence reference at M_ref=40 (same discrete A, ~1.9e-8 vs exact).
    let ref_result = run_jump(grid, M_REF, T_SLOPE, &g);

    println!("G_RESOLVENT_JUMP_ORDER sub-test A (slope at t={T_SLOPE}, M_ref={M_REF}):");
    let mut errs_a: Vec<f64> = Vec::with_capacity(M_SWEEP.len());
    for &m in &M_SWEEP {
        let out = run_jump(grid, m, T_SLOPE, &g);
        let err = pointwise_sup_err(&out, &ref_result);
        println!("  M={m:2}  err_inf = {err:.4e}");
        assert!(
            err.is_finite(),
            "G_RESOLVENT_JUMP_ORDER: non-finite error at M={m}"
        );
        errs_a.push(err);
    }

    // G24 convention: xs = 1/M (improvement as M grows), ys = errors.
    let inv_m: Vec<f64> = M_SWEEP.iter().map(|&m| 1.0 / m as f64).collect();
    let slope = ols_slope(&inv_m, &errs_a);
    println!("  OLS slope d log(err)/d log(1/M) = {slope:+.4}  (gate ≥ +{SLOPE_GATE})");
    assert!(
        slope >= SLOPE_GATE,
        "G_RESOLVENT_JUMP_ORDER sub-test A FAIL: slope {slope:+.4} < +{SLOPE_GATE}. \
         Errors by M: {errs_a:?}. \
         Expected geometric convergence (pre-flight slope ≈ +9.86). \
         Reference: M_ref={M_REF} (self-convergence, same discrete A). \
         Check TWS contour constants / resolve_lhp Thomas solve (resolvent_jump.rs).",
    );

    // ---- Sub-test B: t-independence, M=16, t ∈ {1, 20, 100} ---------------

    println!("G_RESOLVENT_JUMP_ORDER sub-test B (t-independence at M={M_T_INDEP}):");
    let mut errs_b: Vec<f64> = Vec::with_capacity(T_INDEP.len());
    for &t in &T_INDEP {
        let ref_t = run_jump(grid, M_REF, t, &g);
        let probe = run_jump(grid, M_T_INDEP, t, &g);
        let err = pointwise_sup_err(&probe, &ref_t);
        println!("  t={t:6.1}  err(M={M_T_INDEP}) = {err:.4e}");
        errs_b.push(err);
    }
    let spread = errs_b.iter().copied().fold(0.0_f64, f64::max)
        / errs_b.iter().copied().fold(f64::INFINITY, f64::min);
    println!("  spread max/min = {spread:.2}×  (gate ≤ {SPREAD_GATE}×)");
    assert!(
        spread <= SPREAD_GATE,
        "G_RESOLVENT_JUMP_ORDER sub-test B FAIL: t-independence spread {spread:.2}× > {SPREAD_GATE}×. \
         Errors by t: {errs_b:?}. \
         The M/t contour scaling must decouple node count from t (§47.2).",
    );

    println!("G_RESOLVENT_JUMP_ORDER PASS");
}
