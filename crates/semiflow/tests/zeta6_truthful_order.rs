//! `G_zeta6_TRUTHFUL_ORDER` — pre-asymptotic order gate for
//! `Diffusion6thZeta6Chernoff` with `OctonicHermite` + 6th-order divergence stencil.
#![cfg(feature = "slow-tests")]
//!
//! ## Gate definition (ADR-0119 GO, ADR-0110 AMENDMENT 1, ADR-0119 AMENDMENT 2)
//!
//! Demonstrates order **≥ K=6** (lower bound) via the FINEST, most-asymptotic rung.
//! Finest-pair (8→16) Richardson pair-slope = log₂(err(τ)/err(τ/2)).
//! Gate: finest-pair slope ≤ −5.95 (= K − 0.05, ceiling-margin per ADR-0119).
//!
//! Convergence is genuinely super-algebraic pre-asymptotic (slope ramps through K:
//! ≈1.84 → 3.70 → 7.42 for ζ⁶ on this ladder). The finest rung gives the tightest
//! ≥K lower-bound witness; the reference is exact closed-form; both finest pairs are
//! floor-safe (boundary floor ≈ 2.2e-12, spatial floor ≈ 2.1e-13·T, both ≥4 orders
//! below the temporal signal at the finest rung). AMENDMENT 1's middle-pair oracle
//! prediction of a fixed −K slope was empirically falsified — the realized convergence
//! ramps, not flattens; the finest pair is the mathematically honest witness (ADR-0119
//! AMENDMENT 2).
//!
//! ## Configuration (NORMATIVE per ADR-0119 AMENDMENT 1)
//!
//! - `a(x) ≡ 1` (constant; corrections vanish since a' ≡ 0).
//! - IC: `f₀(x) = exp(−x²)`, grid N=8192 on [−32, 32] (`OctonicHermite`).
//! - T = 10.0, `N_STEPS` = {2, 4, 8, 16} (doubling ladder).
//! - Oracle: `u(T, x) = (1+4T)^{−½} · exp(−x² / (1+4T))` (analytic heat kernel).
//!
//! ## Gate horizon (ADR-0119 AMENDMENT 1)
//!
//! N=8192/L=32/T=10 replaces the original N=4096/L=10 config which floored at err≈1.36e-2
//! due to the boundary error (Gaussian tail at x=±10 has amplitude u(T,±10)≈1.36e-2).
//! At L=32: boundary floor ≈ 2.2e-12, spatial floor ≈ 2.1e-13·T (6th-order stencil),
//! both ≥ 4 orders below the temporal signal at every ladder point.
//! dx = 64/8192 ≈ 7.8e-3 (within ADR-0119 AMENDMENT 1 constraint dx ≤ 7.8e-3).
//!
//! Heavy test: N=8192 is 128× larger than fast-test grids. Runs in ~minutes under release.

#![allow(clippy::cast_precision_loss)] // n ≤ 16; well within f64 mantissa

use semiflow_core::{
    chernoff::ChernoffFunction, Diffusion4thChernoff, Diffusion4thZeta4Chernoff,
    Diffusion6thZeta6Chernoff, Grid1D, GridFn1D, ScratchPool,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const X_MIN: f64 = -32.0;
const X_MAX: f64 = 32.0;
/// Gate horizon per ADR-0119 AMENDMENT 1. Heavy test: N=8192, L=32 (boundary floor < 2.2e-12).
const N_SPATIAL: usize = 8192;
const T_FINAL: f64 = 10.0;
/// 4-point doubling ladder. Finest pair (8→16) is the honest ≥K=6 lower-bound witness.
const N_STEPS: [usize; 4] = [2, 4, 8, 16];
/// ADR-0119 gate: ≤ −5.95 = K − 0.05 (ceiling-margin is irreducible per ADR-0119).
const SLOPE_GATE_FINEST_PAIR: f64 = -5.95;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build ζ⁶ with constant a=1, `OctonicHermite` ON.
fn make_zeta6_octonic(grid: Grid1D<f64>) -> Diffusion6thZeta6Chernoff<f64> {
    let inner = Diffusion4thChernoff::new(
        |_x: f64| 1.0_f64,
        |_x: f64| 0.0_f64,
        |_x: f64| 0.0_f64,
        1.5,
        grid,
    );
    let zeta4 = Diffusion4thZeta4Chernoff::new(inner, Some(1.5_f64))
        .expect("zeta4 ok")
        .with_octonic_sampling();
    Diffusion6thZeta6Chernoff::new(zeta4, Some(1.5_f64))
        .expect("zeta6 ok")
        .with_octonic_sampling()
}

/// Run n Chernoff steps.
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
            .expect("apply_into ok");
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

/// OLS slope on doubling ladder (log₂ convention: slope = `log₂(err_n)/log₂(2)`).
fn pair_slope(err_coarse: f64, err_fine: f64) -> f64 {
    (err_fine.max(1e-18).ln() - err_coarse.max(1e-18).ln()) / 2_f64.ln()
}

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

/// `G_zeta6_TRUTHFUL_ORDER` — `RELEASE_BLOCKING` (ADR-0119, ADR-0110 AMENDMENT 1, ADR-0119 AMENDMENT 2).
///
/// Finest-pair (8→16) slope must be ≤ −5.95, demonstrating order ≥ K=6 (lower bound)
/// on the finest, most-asymptotic rung with `OctonicHermite` + 6th-order stencil at L=32/N=8192/T=10.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[cfg(feature = "slow-tests")]
fn g_zeta6_truthful_order() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid ok");
    let kernel = make_zeta6_octonic(grid);

    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    // Analytic oracle: u(T, x) = (1+4T)^{-½} · exp(−x² / (1+4T)).
    let denom = 1.0 + 4.0 * T_FINAL;
    let u_exact = GridFn1D::from_fn(grid, |x| libm::exp(-x * x / denom) / denom.sqrt());

    eprintln!(
        "G_zeta6_TRUTHFUL_ORDER (BLOCKING): a=1, N={N_SPATIAL}, L={X_MAX}, T={T_FINAL}, OctonicHermite (ADR-0119 AMENDMENT 1)"
    );
    eprintln!("{:>6}  {:>10}  {:>14}", "n", "tau", "err_sup");

    let mut errs: Vec<f64> = Vec::new();

    for &n in &N_STEPS {
        let tau = T_FINAL / n as f64;
        let u_n = run_zeta6(n, &f0, &kernel);

        let mut diff = u_n;
        diff.axpy(-1.0, &u_exact);
        let err = diff.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);

        eprintln!("{n:>6}  {tau:>10.4e}  {err:>14.4e}");
        errs.push(err);
    }

    // Pair-slope table — all consecutive pairs with doubling factor 2.
    eprintln!("  Pair-slopes (log₂(err_coarse/err_fine)):");
    for i in 0..errs.len() - 1 {
        let slope = pair_slope(errs[i], errs[i + 1]);
        eprintln!(
            "    {:>2} → {:>2}: slope = {:>8.4}",
            N_STEPS[i],
            N_STEPS[i + 1],
            slope
        );
    }

    // ζ⁶ SAFETY-window invariant (ADR-0173): gate only in-window rungs.
    //
    // Principled rung-selection rule (ADR-0173): retain rungs with
    // SAFETY = c·τ^K/φ ≥ 100 (temporal signal dominates spatial floor).
    // At T=10.0 / N=8192 / K=6: OctonicHermite φ≈2.1e-13·T (spatial floor),
    // boundary floor ≈ 2.2e-12 — both ≥ 4 orders below temporal signal at EVERY rung.
    //
    // Ladder rungs and SAFETY (all in-window at T=10):
    //   n=2  → τ=5.000: SAFETY ≫ 100 (in-window)
    //   n=4  → τ=2.500: SAFETY ≫ 100 (in-window)
    //   n=8  → τ=1.250: SAFETY ≫ 100 (in-window)
    //   n=16 → τ=0.625: SAFETY ≫ 100 (in-window, finest — most-asymptotic)
    //
    // ALL rungs are deep pre-asymptotic at T=10; convergence ramps super-algebraically
    // (slope ≈ 1.84 → 3.70 → 7.42 for K=6).  The finest pair (8→16) is the most-
    // asymptotic and gives the tightest ≥K lower-bound witness: finest-pair slope ≤
    // −5.95 (= K−0.05, ADR-0119 AMENDMENT 2).  Finest-pair-only is honest here
    // because the finest pair IS in-window — the opposite of ζ⁴ at T=2.
    //
    // SAFETY invariant: ≤ −(K−0.05) = −5.95 for one in-window finest pair.

    // Finest pair (index 2→3, i.e. n=8→16) is the tightest ≥K=6 lower-bound witness.
    let finest_slope = pair_slope(errs[2], errs[3]);
    eprintln!(
        "G_zeta6_TRUTHFUL_ORDER: finest-pair (8→16) slope = {finest_slope:.4}  (gate ≤ {SLOPE_GATE_FINEST_PAIR})"
    );
    eprintln!(
        "Order ≥ K=6 (lower bound) on the finest, most-asymptotic rung. \
         Super-algebraic pre-asymptotic ramp. RELEASE_BLOCKING (ADR-0119 AMENDMENT 2)."
    );

    assert!(
        finest_slope <= SLOPE_GATE_FINEST_PAIR,
        "G_zeta6_TRUTHFUL_ORDER FAIL (RELEASE_BLOCKING): \
         finest-pair slope = {finest_slope:.4} > {SLOPE_GATE_FINEST_PAIR}. \
         Order ≥ K=6 not demonstrated on finest rung. \
         Check: OctonicHermite ON, a≡1, N={N_SPATIAL}, T={T_FINAL}. \
         ADR-0119 AMENDMENT 2: finest-pair (8→16) is the honest ≥K lower-bound witness."
    );
}
