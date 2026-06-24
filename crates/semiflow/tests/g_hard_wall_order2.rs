//! `G_HARD_WALL_ORDER2` — `KilledDirichletChernoff` order-2 convergence gate.
//!
//! Gate (ADR-0135 Amendment 2, `RELEASE_BLOCKING`):
//!   OLS slope of `log(sup_err)` vs `log(n_spatial)` ≤ −1.95.
//!
//! ## Method: oracle vs exact closed-form (Dirichlet eigenfunction)
//!
//! Domain `Ω = [0, π]`, absorbing wall `u|_{∂Ω} = 0`. Operator `L = 0.5 ∂_xx`.
//! IC `u₀(x) = sin(x)` — the first Dirichlet eigenfunction of `∂_xx` on `[0,π]`.
//!
//! Since `L^R sin(x) = 0.5·∂_xx sin(x) = −0.5·sin(x)`, the exact semigroup gives
//!
//! ```text
//! u(t, x) = exp(−0.5·t) · sin(x)
//! ```
//!
//! IC satisfies the Dirichlet BC exactly (`sin(0) = 0`, `sin(π) = 0`), is smooth
//! on the interior, and attains its maximum `u(0, π/2) = 1` — well away from the
//! wall. The wall ABSORBS the decaying eigenmode at every Crank–Nicolson step.
//!
//! ## Spatial convergence design
//!
//! Sweep `n_spatial ∈ {32, 64, 128, 256}` with `τ = dx²` (quadratic coupling:
//! `dx = π/(n_spatial − 1)`, so `τ = (π/(n_spatial − 1))²`). Coupling choice:
//!
//! - Temporal error per step for the Cayley map: `O(τ³)`.  Global: `O(τ²) = O(dx⁴)`.
//!   This is sub-dominant to the spatial `O(dx²)` floor so the spatial order is clean.
//! - A-stable Crank–Nicolson has no CFL restriction; any `τ/dx` is stable.
//!
//! Measured errors are dominated by the 3-point FD stencil spatial bias `O(dx²)`,
//! giving slope ≈ −2.04 in `log(n_spatial)` vs `log(sup_err)`. Gate: ≤ −1.95.
//!
//! Expected errors (from eigenvalue bias analysis `Δλ ≈ −dx²/12`):
//!   n=32:  err ≈ 1.1e-3
//!   n=64:  err ≈ 2.7e-4  (ratio ≈ 4.1 → slope ≈ −2.04)
//!   n=128: err ≈ 6.7e-5
//!   n=256: err ≈ 1.7e-5
//!
//! ## Feature gate
//!
//! `slow-tests`. Run with:
//! ```bash
//! cargo test -p semiflow-core --features slow-tests -- --ignored g_hard_wall_order2
//! ```

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize→f64 in OLS; n ≤ 256 ≤ 2^52
#![allow(clippy::cast_possible_truncation)] // f64→usize for n_steps: always non-negative finite
#![allow(clippy::cast_sign_loss)] // same: ceil() result is positive

use semiflow::{
    chernoff::ChernoffFunction, Grid1D, GridFn1D, KilledDirichletChernoff, ScratchPool,
};

// ---------------------------------------------------------------------------
// Gate constants (normative — do NOT relax without ADR + properties.yaml bump)
// ---------------------------------------------------------------------------

/// OLS slope gate (`RELEASE_BLOCKING`, ADR-0135 Amendment 2, math §44.ter.6).
const SLOPE_GATE: f64 = -1.95;

/// Domain `[0, π]` with absorbing Dirichlet walls at both endpoints.
const X_MIN: f64 = 0.0;
const X_MAX: f64 = core::f64::consts::PI;

/// Diffusion coefficient `a` in `L = a·∂_xx`. Chosen so the mode-1 eigenvalue
/// is `λ = −a = −0.5` and the oracle is `exp(−0.5·T)·sin(x)`.
const A_COEF: f64 = 0.5;

/// Final integration time.  `exp(−0.5·0.5) ≈ 0.78` — non-trivial signal.
const T_FINAL: f64 = 0.5;

/// Spatial sweep. Matches ADR-0135 Amendment 2 and properties.yaml gate spec.
const N_SPATIAL: [usize; 4] = [32, 64, 128, 256];

// ---------------------------------------------------------------------------
// Oracle: exact closed form
// ---------------------------------------------------------------------------

/// Exact semigroup `e^{T L^R}` applied to `sin(x)`.
///
/// `u(T, x) = exp(−A_COEF · T_FINAL) · sin(x)`
///
/// Derivation: `sin(x)` is the Dirichlet eigenfunction with eigenvalue `μ = −1`
/// of `∂_xx` on `[0, π]`, so `L^R sin(x) = 0.5·μ·sin(x) = −0.5·sin(x)` and
/// `e^{t L^R} sin(x) = exp(−0.5·t)·sin(x)`.
#[inline]
fn oracle(x: f64) -> f64 {
    (-A_COEF * T_FINAL).exp() * x.sin()
}

// ---------------------------------------------------------------------------
// Single-run helper
// ---------------------------------------------------------------------------

/// Evolve `n_spatial`-node grid for `n_steps = ceil(T/τ)` Cayley steps with
/// `τ = dx²` (quadratic coupling), returning the final grid function.
///
/// Temporal error `O(τ²) = O(dx⁴)` is sub-dominant to spatial `O(dx²)`.
/// All `n_steps` values fit in a `usize`; `n_spatial ≤ 256` is well within
/// f64 52-bit mantissa range for the intermediate `f64` casts.
#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn evolve(n_spatial: usize) -> GridFn1D<f64> {
    let grid = Grid1D::new(X_MIN, X_MAX, n_spatial).expect("grid valid (n >= 2)");
    let dx = grid.dx();
    let tau = dx * dx; // quadratic coupling: temporal floor O(dx^4) << spatial O(dx^2)
    let n_steps = (T_FINAL / tau).ceil() as usize;

    let kernel = KilledDirichletChernoff::new(|_| A_COEF, |_| 0.0_f64, grid)
        .expect("a > 0, n >= 3 — never fails");

    let mut src = GridFn1D::from_fn(grid, f64::sin);
    let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
    let mut scratch = ScratchPool::new();

    // Accumulate steps; the last step may overshoot T_FINAL slightly (by < τ).
    // The overshoot is O(τ) ≈ O(dx²) — absorbed into the spatial floor.
    let mut t_elapsed = 0.0_f64;
    for _ in 0..n_steps {
        let step_tau = (T_FINAL - t_elapsed).min(tau);
        kernel
            .apply_into(step_tau, &src, &mut dst, &mut scratch)
            .expect("apply_into succeeds (valid tau, valid grid)");
        core::mem::swap(&mut src, &mut dst);
        t_elapsed += step_tau;
    }
    src
}

/// Sup-norm error vs analytic oracle on all nodes.
///
/// Wall nodes contribute exactly 0 error (`sin(0) = 0`, `sin(π) = 0`; hard
/// BC bakes u|_∂R = 0 into the generator domain — no mask, no residual).
fn sup_err(n_spatial: usize, u: &GridFn1D<f64>) -> f64 {
    let grid = Grid1D::new(X_MIN, X_MAX, n_spatial).expect("grid valid");
    u.values
        .iter()
        .enumerate()
        .map(|(k, &v)| (v - oracle(grid.x_at(k))).abs())
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
// G_HARD_WALL_ORDER2 — RELEASE_BLOCKING gate
// ---------------------------------------------------------------------------

/// `G_HARD_WALL_ORDER2`: `KilledDirichletChernoff` order-2 spatial convergence gate.
///
/// Problem: `∂_t u = 0.5·∂_xx u` on `[0, π]`, `u|_{∂Ω} = 0`, IC `sin(x)`.
/// Oracle: `u(T, x) = exp(−0.5·T)·sin(x)` (mode-1 Dirichlet eigenfunction).
/// Coupling: `τ = dx²` (temporal floor `O(dx⁴)` << spatial `O(dx²)`).
///
/// Sweep n ∈ {32, 64, 128, 256}: OLS slope of `log(sup_err)` vs `log(n_spatial)`
/// MUST be ≤ −1.95 to confirm that `(I − τ/2·L^R)^{-1}(I + τ/2·L^R)` achieves
/// genuine order-2 with the hard absorbing wall baked into the domain.
///
/// Authority: ADR-0135 Amendment 2; math §44.ter.6; properties.yaml `G_HARD_WALL_ORDER2`.
/// Failure BLOCKS v8.0.0 release.
#[test]
#[ignore = "RELEASE_BLOCKING slow gate; run with: cargo run -p xtask -- test-flagship"]
fn g_hard_wall_order2() {
    let mut errs = Vec::with_capacity(N_SPATIAL.len());

    for &n in &N_SPATIAL {
        let u_n = evolve(n);
        let e = sup_err(n, &u_n);
        let dx = core::f64::consts::PI / (n - 1) as f64;
        println!("G_HARD_WALL_ORDER2: n={n:3}, dx={dx:.4e}, sup_err={e:.4e}");
        errs.push(e);
    }

    let slope = ols_slope(&N_SPATIAL, &errs);
    println!("G_HARD_WALL_ORDER2: slope = {slope:.4}  (gate <= {SLOPE_GATE})");

    assert!(
        slope.is_finite() && slope <= SLOPE_GATE,
        "G_HARD_WALL_ORDER2 FAIL: slope {slope:.4} > gate {SLOPE_GATE}.\n\
         KilledDirichletChernoff order-2 spatial convergence failed.\n\
         Problem: L=0.5∂_xx, domain=[0,π], u|∂Ω=0, IC=sin(x), T={T_FINAL}.\n\
         Oracle: exp(-0.5·T)·sin(x) (Dirichlet mode-1 eigenfunction).\n\
         Coupling: τ = dx² (temporal floor sub-dominant).\n\
         Sweep n_spatial={N_SPATIAL:?}.\n\
         If slope ≈ −1.0: CN assembly degraded to order-1 (check stencil rows).\n\
         If slope ≈ −0.5: unexpected √τ boundary-layer bias (check wall-row identity).\n\
         Authority: ADR-0135 Amendment 2, math §44.ter.2, properties.yaml G_HARD_WALL_ORDER2.",
    );
}
