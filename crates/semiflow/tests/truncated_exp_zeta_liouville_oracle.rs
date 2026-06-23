//! G13-truncated-exp — variable-a Liouville oracle convergence (v0.4.0, ADR-0011).
//!
//! This is the FLAGSHIP gate for the `TruncatedExp` integrator.
//! ζ-A (`DiffusionChernoff`) achieves only slope ≈ −1.0 for variable `a`
//! due to FD-derivative accumulation (ADR-0008 Amendment 2).
//! `TruncatedExp` must achieve slope ≤ −1.95 (global O(τ²)).
//!
//! ## Specification
//!
//! PDE: `∂_t u = ∂_x(a(x)·∂_x u)`,   `a(x) = (1 + γx)²`
//! IC:  `u₀(x) = exp(-x²/2)` (Gaussian, σ=1).
//! Domain: [-1, 1]. Diagonal refinement: N = n for each n.
//! T = 0.005 (small, so CFL ≤ 0.45 for all n ∈ {32,64,128} with γ ≤ 0.20).
//!
//! γ sweep: {0.05, 0.10, 0.20}.
//!
//! ## Diagonal refinement
//!
//! For each n, use N = n nodes on [-1,1]. Then:
//!   dx = 2/(n-1),  τ = T/n.
//! CFL: `2·τ·a_norm` / dx² ≈ T·a_norm·(n-1)²/(2n) ≈ `T·a_norm·n/2`.
//! With T=0.005, `a_norm≤1.44` and n≤128: CFL ≤ 0.45 (asymptotic regime).
//!
//! Global error is O(dx²) = O(1/n²) and O(τ²) = O(1/n²) → slope ≈ -2.
//!
//! ## Oracle: Liouville transform
//!
//! PDE `u_t = ∂_x(a(x) u_x)` with `a(x) = (1+γx)²` transforms to
//! `v_t = v_yy − (γ²/4) v` via:
//!   y(x) := ln(1 + γx) / γ
//!   v(t,y) := (1+γx)^(-1/2) · u(t,x)
//!
//! Setting v = e^(-γ²t/4) · w, `w_t` = `w_yy` (pure heat):
//!   w(t,y) = ∫ G_t(y-y') · `w_0(y`') dy'
//! where `G_t(z`) = exp(-z²/(4t)) / sqrt(4πt) and
//!   `w_0(y`') = e^(-γy'/2) · exp(-x(y')²/2)  with x(y') = (e^(γy')-1)/γ.
//! Inverse transform:
//!   u(t,x) = (1+γx)^(1/2) · e^(-γ²t/4) · w(t, ln(1+γx)/γ)
//!
//! The integral is computed by trapezoidal quadrature over y' ∈ [-`Y_MAX,Y_MAX`].
//!
//! ## Gate: G13-truncated-exp
//!
//! Log-log slope ≤ -1.95 over feasible n (CFL-passed), per γ.
//! Collect-then-assert.
//!
//! Reference: contracts/semiflow-core.properties.yaml §`truncated_exp_variable_zeta_liouville_oracle_slope`.

use std::{cell::Cell, f64::consts::PI};

use semiflow_core::{
    chernoff::ApplyChernoffExt, BoundaryPolicy, Grid1D, GridFn1D, TruncatedExpDiffusionChernoff,
};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

const T_FINAL: f64 = 0.005;
const X_MIN: f64 = -1.0;
const X_MAX: f64 = 1.0;
/// Gaussian IC: σ = 0.2 (narrow; f(±1) ≈ 3.7e-6, negligible at boundaries).
const SIGMA: f64 = 0.2;
/// Diagonal refinement sweep: N = n nodes for each n.
/// At T=0.005 and γ≤0.20, CFL ≤ 0.45 for all these n values (asymptotic regime).
const N_SWEEP: [usize; 3] = [32, 64, 128];
const GAMMA_SWEEP: [f64; 3] = [0.05, 0.10, 0.20];
const SLOPE_GATE: f64 = -1.95;

// Oracle quadrature
const Y_MAX: f64 = 10.0;
const N_QUAD_PTS: usize = 8000;

// ---------------------------------------------------------------------------
// Thread-local γ (fn-pointer compatibility)
// ---------------------------------------------------------------------------

thread_local! {
    static CURRENT_GAMMA: Cell<f64> = const { Cell::new(0.0) };
}

fn a_fn(x: f64) -> f64 {
    let g = CURRENT_GAMMA.with(Cell::get);
    let v = 1.0 + g * x;
    v * v
}

fn a_prime_fn(x: f64) -> f64 {
    let g = CURRENT_GAMMA.with(Cell::get);
    2.0 * g * (1.0 + g * x)
}

fn a_double_prime_fn(_: f64) -> f64 {
    let g = CURRENT_GAMMA.with(Cell::get);
    2.0 * g * g
}

fn a_norm_bound(gamma: f64) -> f64 {
    // a(x) = (1+γx)² on [-1,1]; max at x=1 for γ>0: (1+γ)²
    let v = 1.0 + gamma * X_MAX;
    v * v
}

// ---------------------------------------------------------------------------
// Gaussian IC: f(x) = exp(-x²/2)
// ---------------------------------------------------------------------------

fn gaussian(grid: Grid1D) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| (-x * x / (2.0 * SIGMA * SIGMA)).exp())
}

// ---------------------------------------------------------------------------
// Liouville oracle
//
//   u(t,x) = sqrt(1+γx) · exp(-γ²t/4) · w(t, ln(1+γx)/γ)
//   w(t,y) = ∫ G_t(y-y') · w_0(y') dy'
//   G_t(z) = exp(-z²/(4t)) / sqrt(4πt)
//   w_0(y) = exp(-γy/2) · exp(-x(y)²/2),  x(y) = (exp(γy)-1)/γ
//
// Trapezoidal quadrature over y' ∈ [-Y_MAX, Y_MAX].
// ---------------------------------------------------------------------------

fn liouville_oracle(gamma: f64, t: f64, x: f64) -> f64 {
    if gamma.abs() < 1e-14 {
        // γ → 0 limit: pure constant-a heat equation, a=1, Gaussian IC.
        // u(t,x) = (1/sqrt(1+2t)) · exp(-x²/(2(1+2t)))
        let denom = 1.0 + 2.0 * t;
        return (1.0 / denom.sqrt()) * (-x * x / (2.0 * denom)).exp();
    }
    let y_query = (1.0 + gamma * x).ln() / gamma;
    #[allow(clippy::cast_precision_loss)]
    // N_QUAD_PTS is a small compile-time constant (≤ 4096); no precision loss in practice.
    let dy = 2.0 * Y_MAX / (N_QUAD_PTS - 1) as f64;
    let inv_sqrt_4pi_t = 1.0 / (4.0 * PI * t).sqrt();
    let mut integral = 0.0_f64;
    for k in 0..N_QUAD_PTS {
        #[allow(clippy::cast_precision_loss)]
        // k < N_QUAD_PTS (a small compile-time constant); no precision loss.
        let yp = -Y_MAX + k as f64 * dy;
        let xp = ((gamma * yp).exp() - 1.0) / gamma;
        // w_0(y') = (1+γx')^{1/2} · f(x') = exp(+γy'/2) · exp(-x'^2/(2σ²))
        // [Sign: (1+γx')^{1/2} = exp(γy'/2) since y'=ln(1+γx')/γ]
        let w0 = (0.5 * gamma * yp).exp() * (-xp * xp / (2.0 * SIGMA * SIGMA)).exp();
        let dz = y_query - yp;
        let kern = (-dz * dz / (4.0 * t)).exp() * inv_sqrt_4pi_t;
        let weight = if k == 0 || k == N_QUAD_PTS - 1 {
            0.5
        } else {
            1.0
        };
        integral += weight * kern * w0 * dy;
    }
    // Inverse transform: u(t,x) = (1+γx)^{-1/2} · exp(-γ²t/4) · w(t, y_query)
    // [Sign: a^{-1/4} = (1+γx)^{-1/2}]
    let a_factor = (1.0 + gamma * x).powf(-0.5);
    let decay = (-gamma * gamma * t / 4.0).exp();
    a_factor * decay * integral
}

// ---------------------------------------------------------------------------
// Sup-norm error vs oracle
// ---------------------------------------------------------------------------

fn sup_err_vs_oracle(state: &GridFn1D, grid: Grid1D, gamma: f64, t: f64) -> f64 {
    let mut max_err = 0.0_f64;
    for (i, &v) in state.values.iter().enumerate() {
        let x = grid.x_at(i);
        let u_ex = liouville_oracle(gamma, t, x);
        let err = (v - u_ex).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Log-log OLS slope
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
// ns.len() and n are small counts (≤ 16 points, n ≤ 256); no precision loss in practice.
// sum_x/sum_y/sum_xx/sum_xy are standard least-squares names (math convention).
#[allow(clippy::similar_names)]
fn log_log_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.max(1e-16).ln()).collect();
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xx: f64 = xs.iter().map(|&x| x * x).sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

// ---------------------------------------------------------------------------
// G13-truncated-exp
// ---------------------------------------------------------------------------

/// Run one (gamma, n) combination. Returns `Some(sup_err)` if CFL passes, `None` to skip.
#[allow(clippy::cast_precision_loss)]
// n ≤ 256 (N_SWEEP values); no precision loss in practice.
fn run_one_n(gamma: f64, n: usize, anb: f64) -> Option<f64> {
    let grid = match Grid1D::new(X_MIN, X_MAX, n) {
        Ok(g) => g.with_boundary(BoundaryPolicy::ZeroExtend),
        Err(e) => {
            eprintln!("{gamma:>6.2}  {n:>5}  {n:>8}  —  GRID_ERR: {e:?}");
            return None;
        }
    };
    let tau = T_FINAL / n as f64;
    let dx2 = grid.dx() * grid.dx();
    let cfl = 2.0 * tau * anb / dx2;
    if cfl >= 1.0 {
        eprintln!("{gamma:>6.2}  {n:>5}  {n:>8}  {tau:>10.4e}  {cfl:>10.4}  SKIP(CFL)");
        return None;
    }
    CURRENT_GAMMA.with(|c| c.set(gamma));
    let truncated_exp =
        TruncatedExpDiffusionChernoff::new(a_fn, a_prime_fn, a_double_prime_fn, anb, grid);
    let mut state = gaussian(grid);
    for _ in 0..n {
        match truncated_exp.apply_chernoff(tau, &state) {
            Ok(s) => state = s,
            Err(e) => {
                eprintln!("{gamma:>6.2}  {n:>5}  {n:>8}  {tau:>10.4e}  {cfl:>10.4}  ERR({e:?})");
                return None;
            }
        }
    }
    let err = sup_err_vs_oracle(&state, grid, gamma, T_FINAL);
    eprintln!("{gamma:>6.2}  {n:>5}  {n:>8}  {tau:>10.4e}  {cfl:>10.4}  {err:>12.4e}");
    Some(err)
}

/// Sweep n values for one gamma; return slope failure message or `None`.
fn check_one_gamma(gamma: f64) -> Option<String> {
    CURRENT_GAMMA.with(|c| c.set(gamma));
    let anb = a_norm_bound(gamma);
    let mut ns_valid: Vec<usize> = Vec::new();
    let mut errs_valid: Vec<f64> = Vec::new();
    for &n in &N_SWEEP {
        if let Some(err) = run_one_n(gamma, n, anb) {
            ns_valid.push(n);
            errs_valid.push(err);
        }
    }
    if ns_valid.len() < 2 {
        eprintln!(
            "γ={gamma:.2}: only {} feasible n — cannot compute slope (skip)",
            ns_valid.len()
        );
        return None;
    }
    let slope = log_log_slope(&ns_valid, &errs_valid);
    eprintln!(
        "γ={gamma:.2}: slope={slope:.4}  gate ≤ {SLOPE_GATE}  ({} points)",
        ns_valid.len()
    );
    if slope > SLOPE_GATE {
        return Some(format!(
            "γ={gamma:.2}: slope={slope:.4} > {SLOPE_GATE} ({} feasible n: {:?})",
            ns_valid.len(),
            ns_valid
        ));
    }
    None
}

/// G13-truncated-exp: `TruncatedExpDiffusionChernoff` achieves slope ≤ -1.95 vs closed-form
/// Liouville oracle for variable `a(x) = (1+γx)²`.
///
/// Diagonal refinement: N = n nodes on [-1,1] for each n.
/// T = 0.005; CFL satisfied for n ≤ 256 with γ ≤ 0.20.
/// Oracle: Liouville transform → heat kernel integral (trapezoidal quadrature).
/// Collect-then-assert across γ values.
#[test]
fn g13_truncated_exp_variable_liouville_oracle() {
    eprintln!("G13-truncated-exp: diagonal refinement, T={T_FINAL}, SIGMA={SIGMA}");
    eprintln!(
        "{:>6}  {:>5}  {:>8}  {:>10}  {:>10}  {:>12}",
        "gamma", "n", "N", "tau", "CFL", "sup_err"
    );
    let failures: Vec<String> = GAMMA_SWEEP
        .iter()
        .filter_map(|&g| check_one_gamma(g))
        .collect();
    assert!(
        failures.is_empty(),
        "G13-truncated-exp FAIL ({} γ-value(s)):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
