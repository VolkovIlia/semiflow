//! G3-strang — empirical convergence-rate test (slope ≤ −1.95, order ≥ 1.95).
//!
//! Runs the advection-diffusion oracle with
//! `Φ = StrangSplit<DiffusionChernoff(α=0.5), DriftReactionChernoff(β=0.5, c≡0)>`
//! for `n ∈ {32, 64, 128, 256, 512, 1024}`, records `err(n) = ‖u_Φ − u_oracle‖_∞`,
//! fits a log-log least-squares slope on `(log n, log err)`, and asserts
//! `slope ≤ −1.95` (empirical order ≥ 1.95).
//!
//! # Gate (NON-NEGOTIABLE)
//!
//! `slope ≤ −1.95` from `acceptance-criteria.md §G3-strang` (v0.2.0, ADR-0006 v2).
//! The threshold leaves `0.05` margin below the theoretical `−2.0` to absorb
//! grid-discretization and pre-asymptotic effects at small `n`. Do NOT relax it.
//!
//! # Oracle
//!
//! PDE: `∂_t u = (1/2)∂_xx u + (1/2)∂_x u`, `u(0,x) = exp(-x²)`.
//! Closed form (math.md §9.5): `u(t,x) = (1+2t)^{-1/2} exp(-(x+t/2)²/(1+2t))`.
//! At `t=1`: `u(1,x) = 3^{-1/2} exp(-(x+0.5)²/3)`.
//!
//! Reference: `contracts/semiflow-core.math.md §6.2, §9.4, §9.5`; ADR-0006 v2.

use semiflow_core::{
    ChernoffSemigroup, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, StrangSplit,
};

// ---------------------------------------------------------------------------
// Gate constants (NON-NEGOTIABLE)
// ---------------------------------------------------------------------------

/// G3-strang gate: log-log slope must be ≤ this value (i.e., order ≥ 1.95).
const SLOPE_GATE: f64 = -1.95;

/// n-values over which convergence is measured. Must match acceptance-criteria.md §G3-strang.
const N_VALUES: [usize; 6] = [32, 64, 128, 256, 512, 1024];

/// Advection-diffusion parameters (§6.2, §9.5).
const ALPHA: f64 = 0.5;
const BETA: f64 = 0.5;
const T_FINAL: f64 = 1.0;
const N_NODES: usize = 100_000; // amended 2026-04-29: was 8000 (Amendment 3), now 100000 (Amendment 4)

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// Closed-form advection-diffusion oracle (math.md §6.2 box, §9.5).
///
/// Normative: `u(t,x) = (1 + 2t)^{-1/2} · exp(-(x + t/2)² / (1 + 2t))`.
/// At `t=1`: `u(1,x) = 3^{-1/2} exp(-(x+0.5)²/3)`.
/// The denominator `(1+2t)` is a concrete result for `a = 1/2`; not `(1+2αt)`.
#[inline]
fn oracle(t: f64, x: f64) -> f64 {
    // Normative per math.md §6.2: (1+2t)^{-1/2} exp(-(x+t/2)^2/(1+2t))
    let denom = 1.0 + 2.0 * t;
    denom.sqrt().recip() * (-(x + BETA * t).powi(2) / denom).exp()
}

// ---------------------------------------------------------------------------
// Strang error at fixed n
// ---------------------------------------------------------------------------

/// Sup-norm error of `(Φ(T/n))^n u0` vs. oracle at `t = T_FINAL`.
///
/// Grid: `N_NODES = 100_000` nodes on `[-10, 10]`, Reflect BC, `CubicHermite` interp.
fn error_at_n(n_steps: usize) -> f64 {
    let grid = Grid1D::new(-10.0, 10.0, N_NODES).expect("grid params valid");
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    // v0.3.0 (ADR-0008 Amendment 1, ζ-A): a_prime = a_double_prime = |_| 0.0 for constant α
    // (a' ≡ a'' ≡ 0 ⇒ S(s) = id AND τ²-correction = 0 ⇒ D_ζ = D_γ = K = v0.2.2 bit-equal).
    let diff = DiffusionChernoff::new(|_| ALPHA, |_| 0.0_f64, |_| 0.0_f64, ALPHA, grid);
    let drift = DriftReactionChernoff::new(|_| BETA, |_| 0.0_f64, 0.0, grid);
    let strang = StrangSplit::new(diff, drift);
    let semi = ChernoffSemigroup::new(strang, n_steps).expect("n >= 1");
    let u_n = semi.evolve(T_FINAL, &u0).expect("evolve succeeds");

    let mut max_err: f64 = 0.0;
    for i in 0..N_NODES {
        let x = grid.x_at(i);
        let err = (u_n.values[i] - oracle(T_FINAL, x)).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Log-log slope (least squares)
// ---------------------------------------------------------------------------

/// Compute the OLS slope of `(log n_i, log err_i)` pairs.
///
/// Returns `Σ(x_i − x̄)(y_i − ȳ) / Σ(x_i − x̄)²` where `x_i = ln(n_i)`,
/// `y_i = ln(err_i)`.  Identical formula as in `convergence_rate.rs`.
#[allow(clippy::similar_names, clippy::cast_precision_loss)]
// sum_x/sum_y/sum_xy: OLS variable names by convention; n ≤ 1024; well within f64 mantissa
fn log_log_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xx: f64 = xs.iter().map(|&x| x * x).sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

// ---------------------------------------------------------------------------
// G3-strang
// ---------------------------------------------------------------------------

/// G3-strang: empirical log-log slope ≤ −1.95 (order ≥ 1.95).
///
/// Gate from `acceptance-criteria.md §G3-strang` (v0.2.0, ADR-0006 v2).
/// Non-negotiable: if this fails, report the slope and escalate.
#[test]
fn g3_strang_convergence_slope() {
    let errs: Vec<f64> = N_VALUES.iter().map(|&n| error_at_n(n)).collect();

    for (&n, &e) in N_VALUES.iter().zip(errs.iter()) {
        println!("n={n:5}, err={e:.6e}");
    }

    let slope = log_log_slope(&N_VALUES, &errs);
    println!("G3-strang: log-log slope = {slope:.4}  (gate: <= {SLOPE_GATE})");

    assert!(
        slope <= SLOPE_GATE,
        "G3-strang FAIL: slope {slope:.4} > {SLOPE_GATE} (order < 1.95) — Gate FAILED, escalate to architect"
    );
}
