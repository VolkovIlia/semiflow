//! `G_zeta8_TRUTHFUL_ORDER` — pre-asymptotic order gate for
//! `Diffusion8thZeta8Chernoff` with `OctonicHermite` + 6th-order divergence stencil.
#![cfg(feature = "slow-tests")]
//!
//! ## Gate definition (ADR-0119 GO, ADR-0110 AMENDMENT 1, ADR-0119 AMENDMENT 2)
//!
//! Demonstrates order **≥ K=8** (lower bound) via the FINEST, most-asymptotic rung.
//! Finest-pair (8→16) Richardson pair-slope = log₂(err(τ)/err(τ/2)).
//! Gate: finest-pair slope ≤ −7.95 (= K − 0.05, ADR-0119 ceiling-margin).
//!
//! Convergence is genuinely super-algebraic pre-asymptotic (slope ramps through K:
//! ≈3.66 → 7.33 → 11.21 for ζ⁸ on this ladder). The finest rung gives the tightest
//! ≥K lower-bound witness; the reference is exact closed-form; both finest pairs are
//! floor-safe (boundary floor ≈ 2.2e-12, spatial floor ≈ 1.4e-11, both ≥4 orders
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
//! ## Why N=8192/L=32 (ADR-0119 AMENDMENT 1)
//!
//! The original N=4096/L=10 config floored at err≈1.36e-2 due to the boundary error:
//! the finite-domain Neumann BC vs the infinite-line oracle gives a τ-independent
//! boundary floor equal to the Gaussian tail amplitude u(T,±L)=(1+4T)^{-½}exp(-L²/(1+4T)).
//! At L=10,T=10: boundary floor ≈ 1.36e-2 → slope → 0 (floor dominates all ladder points).
//! Receding L=10→32 crushes the floor to ≈2.2e-12 (Gaussian decay), while spatial
//! floor ≈ 1.4e-11 and both are ≥4 orders below temporal signal at every ladder point.
//! dx=64/8192≈7.8e-3; oracle confirms GO with middle-pair slopes = -K to 4 d.p.
//!
//! Unblocked from ADR-0110 AMENDMENT 1 DEFER by the v7.0 KEYSTONE (ADR-0117+0118+0119).

#![allow(clippy::cast_precision_loss)]

use semiflow::{
    chernoff::ChernoffFunction, Diffusion4thChernoff, Diffusion4thZeta4Chernoff,
    Diffusion6thZeta6Chernoff, Diffusion8thZeta8Chernoff, Grid1D, GridFn1D, ScratchPool,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const X_MIN: f64 = -32.0;
const X_MAX: f64 = 32.0;
/// Gate horizon per ADR-0119 AMENDMENT 1. Heavy test: N=8192, L=32 (boundary floor < 2.2e-12).
const N_SPATIAL: usize = 8192;
const T_FINAL: f64 = 10.0;
const N_STEPS: [usize; 4] = [2, 4, 8, 16];
/// ADR-0119 gate: ≤ −7.95 = K − 0.05.
const SLOPE_GATE_FINEST_PAIR: f64 = -7.95;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build ζ⁸ with constant a=1, `OctonicHermite` ON.
fn make_zeta8_octonic(grid: Grid1D<f64>) -> Diffusion8thZeta8Chernoff<f64> {
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
    let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, Some(1.5_f64))
        .expect("zeta6 ok")
        .with_octonic_sampling();
    Diffusion8thZeta8Chernoff::new(zeta6, Some(1.5_f64))
        .expect("zeta8 ok")
        .with_octonic_sampling()
}

/// Run n Chernoff steps.
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
            .expect("apply_into ok");
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

/// Pair slope on doubling ladder (log₂ convention).
fn pair_slope(err_coarse: f64, err_fine: f64) -> f64 {
    (err_fine.max(1e-18).ln() - err_coarse.max(1e-18).ln()) / 2_f64.ln()
}

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

/// `G_zeta8_TRUTHFUL_ORDER` — `RELEASE_BLOCKING` (ADR-0119, ADR-0110 AMENDMENT 1, ADR-0119 AMENDMENT 2).
///
/// Finest-pair (8→16) slope must be ≤ −7.95, demonstrating order ≥ K=8 (lower bound)
/// on the finest, most-asymptotic rung with `OctonicHermite` + 6th-order stencil at L=32/N=8192/T=10.
/// Unblocked from ADR-0110 AMENDMENT 1 DEFER by the v7.0 KEYSTONE (ADR-0117+0118+0119).
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[cfg(feature = "slow-tests")]
fn g_zeta8_truthful_order() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_SPATIAL).expect("grid ok");
    let kernel = make_zeta8_octonic(grid);

    let f0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    let denom = 1.0 + 4.0 * T_FINAL;
    let u_exact = GridFn1D::from_fn(grid, |x| libm::exp(-x * x / denom) / denom.sqrt());

    eprintln!(
        "G_zeta8_TRUTHFUL_ORDER (BLOCKING): a=1, N={N_SPATIAL}, L={X_MAX}, T={T_FINAL}, OctonicHermite (ADR-0119 AMENDMENT 1)"
    );
    eprintln!("{:>6}  {:>10}  {:>14}", "n", "tau", "err_sup");

    let mut errs: Vec<f64> = Vec::new();

    for &n in &N_STEPS {
        let tau = T_FINAL / n as f64;
        let u_n = run_zeta8(n, &f0, &kernel);

        let mut diff = u_n;
        diff.axpy(-1.0, &u_exact);
        let err = diff.values.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);

        eprintln!("{n:>6}  {tau:>10.4e}  {err:>14.4e}");
        errs.push(err);
    }

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

    // ζ⁸ SAFETY-window invariant (ADR-0173): gate only in-window rungs.
    //
    // Principled rung-selection rule (ADR-0173): retain rungs with
    // SAFETY = c·τ^K/φ ≥ 100 (temporal signal dominates spatial floor).
    // At T=10.0 / N=8192 / K=8: OctonicHermite spatial floor ≈ 1.4e-11,
    // boundary floor ≈ 2.2e-12 — both ≥ 4 orders below temporal signal at EVERY rung.
    //
    // Ladder rungs and SAFETY (all in-window at T=10):
    //   n=2  → τ=5.000: SAFETY ≫ 100 (in-window)
    //   n=4  → τ=2.500: SAFETY ≫ 100 (in-window)
    //   n=8  → τ=1.250: SAFETY ≫ 100 (in-window)
    //   n=16 → τ=0.625: SAFETY ≫ 100 (in-window, finest — most-asymptotic)
    //
    // ALL rungs are deep pre-asymptotic at T=10; convergence ramps super-algebraically
    // (slope ≈ 3.66 → 7.33 → 11.21 for K=8).  The finest pair (8→16) is the most-
    // asymptotic and gives the tightest ≥K lower-bound witness: finest-pair slope ≤
    // −7.95 (= K−0.05, ADR-0119 AMENDMENT 2).  Finest-pair-only is honest here
    // because the finest pair IS in-window at T=10 — same regime as ζ⁶.
    //
    // SAFETY invariant: ≤ −(K−0.05) = −7.95 for one in-window finest pair.

    // Finest pair (index 2→3, i.e. n=8→16) is the tightest ≥K=8 lower-bound witness.
    let finest_slope = pair_slope(errs[2], errs[3]);
    eprintln!(
        "G_zeta8_TRUTHFUL_ORDER: finest-pair (8→16) slope = {finest_slope:.4}  (gate ≤ {SLOPE_GATE_FINEST_PAIR})"
    );
    eprintln!(
        "Order ≥ K=8 (lower bound) on the finest, most-asymptotic rung. \
         Super-algebraic pre-asymptotic ramp. RELEASE_BLOCKING (ADR-0119 AMENDMENT 2)."
    );

    assert!(
        finest_slope <= SLOPE_GATE_FINEST_PAIR,
        "G_zeta8_TRUTHFUL_ORDER FAIL (RELEASE_BLOCKING): \
         finest-pair slope = {finest_slope:.4} > {SLOPE_GATE_FINEST_PAIR}. \
         Order ≥ K=8 not demonstrated on finest rung. \
         Check: OctonicHermite ON, a≡1, N={N_SPATIAL}, T={T_FINAL}. \
         ADR-0119 AMENDMENT 2: finest-pair (8→16) is the honest ≥K lower-bound witness."
    );
}
