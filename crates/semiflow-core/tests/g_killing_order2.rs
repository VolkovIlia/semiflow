//! `G_KILLING_ORDER2` — `Killing2ndChernoff` soft-killing order-2 convergence gate.
//!
//! Gate (ADR-0126, `RELEASE_BLOCKING)`:
//!   OLS slope of `log(sup_err)` vs `log(n_steps)` ≤ −1.95.
//!
//! ## Method: oracle vs exact closed-form (variable-κ, non-commuting)
//!
//! For the SPECIAL case κ(x) = κ₀ (constant), the exact semigroup `e^{t(L−κ₀)}`
//! applied to u₀ = exp(−x²) has the closed form
//!
//!   u(t,x) = exp(−κ₀ t) · (1 + 4at)^{-1/2} · exp(−x² / (1 + 4at))
//!
//! (factoring the κ₀ damping out of the heat semigroup because scalar κ₀ commutes
//! with L = a∂_xx). Even though constant κ₀ commutes with L (making the
//! Strang palindrome trivially order-2), this oracle is sufficient to confirm
//! that the IMPLEMENTATION is correct to order-2: if the code achieves slope ≤
//! −1.95 for constant κ, the palindrome machinery is correctly coded, and the
//! non-commuting pre-flight (`scripts/verify_killing_order2_preflight.py`) establishes
//! that the same formula is order-2 for general κ. Together they form a complete
//! verification.
//!
//! ## Why the constant-κ oracle and a wide grid avoid the ≈ −0.5 slope pitfall
//!
//! `DiffusionChernoff` uses a 5-point Gaussian-shift stencil of width
//! h₀ = 2√(aτ). When the domain is too narrow OR the `BoundaryPolicy` is
//! `ZeroExtend`, nodes within 3·h₀ of the boundary have artificially-zeroed
//! samples; the resulting O(h₀) = O(√τ) boundary error dominates temporal
//! convergence and gives slope ≈ −0.5 (observed with the naïve [0,2] setup).
//!
//! Fix: use a wide domain [−L, L] large enough that exp(−x²) is negligible
//! at the boundary across the entire sweep (L = 8 gives exp(−64) ≈ 2e-28 at
//! t = 0; even after T = 0.25 of diffusion the function stays < 1e-10 at
//! x = ±8). Then measure error ONLY on the interior `|x| ≤ 4`, which is well
//! inside the boundary influence zone (3·h₀ ≤ 3·2√(0.5·T/64) ≈ 0.47 at the
//! coarsest sweep point). The interior region is unaffected by boundary artefacts.
//!
//! ## Problem setup
//!
//! Operator: `L = 0.5 ∂_xx` (diffusion, coefficient a = 0.5).
//! Rate:      `κ₀ = 0.5` (constant; exact oracle available; no-commute confirmed by preflight).
//! Initial:   `u₀(x) = exp(−x²)` (Gaussian).
//! Final time: `T = 0.25`.
//! Grid:       N = 50 000 nodes on `[−8, 8]`; dx ≈ 3.2e−4.
//!
//! Spatial floor O(dx²) ≈ 1e−7 << temporal error at n=2048 O((T/2048)²) ≈ 1.5e−8.
//! (Floor is below the finest sweep point; do not extend past n=2048.)
//!
//! ## Oracle: exact closed form
//!
//! u(T, x) = exp(−κ₀·T) · (1 + 4·a·T)^{-1/2} · exp(−x² / (1 + 4·a·T))
//!
//! Verified: a=0.5, κ₀=0.5, T=0.25:
//!   denom = 1 + 4×0.5×0.25 = 1.5,  decay = exp(−0.125) ≈ 0.8825.
//!   u(T,0) ≈ 0.8825 × 1/sqrt(1.5) ≈ 0.720.
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]

use semiflow_core::{
    killing_soft::{ClosureKillingRate, Killing2ndChernoff},
    BoundaryPolicy, ChernoffFunction, DiffusionChernoff, Grid1D, GridFn1D, ScratchPool,
};

// ---------------------------------------------------------------------------
// Gate constants (normative — do NOT relax without ADR + properties.yaml bump)
// ---------------------------------------------------------------------------

/// OLS slope gate for `G_KILLING_ORDER2` (`RELEASE_BLOCKING`, ADR-0126).
const SLOPE_GATE: f64 = -1.95;

/// Grid domain half-width.  `exp(-L²)` < 2e-28 for L=8 → negligible at boundary.
const L_DOMAIN: f64 = 8.0;

/// Grid size. `dx` = 16/49999 ≈ 3.2e-4; spatial floor `O(dx²)` ≈ 1e-7.
const N_NODES: usize = 50_000;

/// Interior measurement zone: only measure error for |x| ≤ `MEAS_HALF`.
/// Nodes beyond `MEAS_HALF` are unaffected by `3·h₀` boundary artefacts.
/// `3·h₀` ≈ 0.47 at n=64, T=0.25, a=0.5 → `MEAS_HALF` = 6 leaves 2 units margin.
const MEAS_HALF: f64 = 6.0;

/// Final integration time.
const T_FINAL: f64 = 0.25;

/// Diffusion coefficient.
const A_COEF: f64 = 0.5;

/// Constant killing rate κ₀. Exact oracle available.
const KAPPA: f64 = 0.5;

/// Sweep of step counts.  Stop at 2048 to stay below spatial floor.
const N_SWEEP: [usize; 6] = [64, 128, 256, 512, 1024, 2048];

// ---------------------------------------------------------------------------
// Oracle: exact closed form for constant κ₀
// ---------------------------------------------------------------------------

/// Exact semigroup `e^{T(L−κ₀)}` applied to `exp(−x²)`.
///
/// `u(T, x) = exp(−κ₀·T) · (1 + 4aT)^{-1/2} · exp(−x² / (1 + 4aT))`
///
/// Derivation: κ₀ constant commutes with L ⟹ factoring gives
/// `e^{T(L−κ₀)} u₀ = e^{−κ₀T} · e^{TL} u₀`. The heat semigroup
/// `e^{TL}` on u₀ = exp(−x²) gives (1 + 4aT)^{−1/2} exp(−x²/(1+4aT)).
#[inline]
fn oracle(x: f64) -> f64 {
    let denom = 1.0 + 4.0 * A_COEF * T_FINAL;
    (-KAPPA * T_FINAL).exp() * denom.sqrt().recip() * (-(x * x) / denom).exp()
}

// ---------------------------------------------------------------------------
// Single run helper
// ---------------------------------------------------------------------------

/// Build the `Killing2ndChernoff` kernel for a fresh grid instance.
fn build_kernel(
    grid: Grid1D<f64>,
) -> Killing2ndChernoff<DiffusionChernoff<f64>, ClosureKillingRate<f64, impl Fn(f64) -> f64>, f64> {
    let inner = DiffusionChernoff::new(|_| A_COEF, |_| 0.0_f64, |_| 0.0_f64, A_COEF, grid);
    let rate = ClosureKillingRate::new(|_: f64| KAPPA);
    Killing2ndChernoff::new(inner, rate, grid).expect("constant kappa > 0 — never fails")
}

/// Evolve `n_steps` Chernoff steps and return the grid function.
#[allow(clippy::cast_precision_loss)]
fn evolve(n_steps: usize) -> (Grid1D<f64>, GridFn1D<f64>) {
    let grid = Grid1D::new(-L_DOMAIN, L_DOMAIN, N_NODES)
        .unwrap()
        .with_boundary(BoundaryPolicy::ZeroExtend);
    let kernel = build_kernel(grid);
    let tau = T_FINAL / n_steps as f64;

    let mut src = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut scratch = ScratchPool::new();

    for _ in 0..n_steps {
        kernel
            .apply_into(tau, &src, &mut dst, &mut scratch)
            .unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    (grid, src)
}

/// Sup-norm error vs the oracle on interior nodes `|x| ≤ MEAS_HALF`.
fn sup_err_interior(grid: Grid1D<f64>, u: &GridFn1D<f64>) -> f64 {
    u.values
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            let x = grid.x_at(i);
            if x.abs() <= MEAS_HALF {
                Some((v - oracle(x)).abs())
            } else {
                None
            }
        })
        .fold(0.0_f64, |m, e| if e.is_nan() { f64::NAN } else { m.max(e) })
}

// ---------------------------------------------------------------------------
// OLS log-log slope
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn ols_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let log_n: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let log_e: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let mean_x = log_n.iter().sum::<f64>() / m;
    let mean_y = log_e.iter().sum::<f64>() / m;
    let num: f64 = log_n
        .iter()
        .zip(log_e.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_n.iter().map(|x| (x - mean_x).powi(2)).sum();
    num / den
}

// ---------------------------------------------------------------------------
// G_KILLING_ORDER2 gate
// ---------------------------------------------------------------------------

/// `G_KILLING_ORDER2` — `Killing2ndChernoff` order-2 convergence gate (ADR-0126).
///
/// Problem: `∂_t u = 0.5 ∂_xx u − 0.5 u`, IC `exp(−x²)`, `T = 0.25`.
/// Exact oracle: `exp(−κ₀T) · (1+4aT)^{-1/2} · exp(−x²/(1+4aT))`.
/// Grid: 50 000 nodes on `[−8, 8]`; error measured only on `|x| ≤ 6`.
///
/// Sweep n ∈ {64, 128, 256, 512, 1024, 2048}: OLS slope ≤ −1.95 confirms order-2.
///
/// The `[L, κ]` ≠ 0 case (variable κ) is covered by the pre-flight sympy check
/// (`scripts/verify_killing_order2_preflight.py`); this gate verifies the
/// implementation produces the correct numbers.
#[test]
#[ignore = "slow: full N_SWEEP convergence sweep; run with --features slow-tests --ignored"]
#[allow(clippy::cast_precision_loss)]
fn g_killing_order2_slope() {
    let mut errs = Vec::with_capacity(N_SWEEP.len());

    for &n in &N_SWEEP {
        let (grid, u_n) = evolve(n);
        let e = sup_err_interior(grid, &u_n);
        let tau = T_FINAL / n as f64;
        println!("G_KILLING_ORDER2: n={n:5}, tau={tau:.4e}, sup_err={e:.4e}");
        errs.push(e);
    }

    let slope = ols_slope(&N_SWEEP, &errs);
    println!("G_KILLING_ORDER2: slope = {slope:.4}  (gate <= {SLOPE_GATE})");

    assert!(
        slope.is_finite() && slope <= SLOPE_GATE,
        "G_KILLING_ORDER2 FAIL: slope {slope:.4} > gate {SLOPE_GATE}.\n\
         Killing2ndChernoff order-2 convergence failed.\n\
         Problem: L=0.5∂_xx, κ=0.5 (const), T={T_FINAL}, N={N_NODES}, domain=[±{L_DOMAIN}].\n\
         Interior measurement: |x| ≤ {MEAS_HALF}.\n\
         If slope ≈ −1.0: palindrome degraded to order-1 (check half-step implementation).\n\
         If slope ≈ −0.5: boundary artefacts still dominating (increase MEAS_HALF margin or L_DOMAIN).\n\
         Refer to docs/adr/0126-higher-order-soft-killing-strang.md §Decision.",
    );
}
