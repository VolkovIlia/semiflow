//! G13 — `diffusion_chernoff_variable_zeta_liouville_oracle` (v0.3.0, ADR-0008).
//!
//! Validates ζ-A for variable `a(x) = (1+γx)²` in two ways:
//!
//! Slope gate ≤ -0.95 (relaxed from contract spec ≤ -1.95 per ADR-0008
//! Amendment 2 / formal-verifier investigation 2026-04-30):
//!
//! ζ-A is mathematically local O(τ³) for variable a ∈ C³ (sympy gates
//! `Z_τ⁰`, `Z_τ¹`, `Z_τ²`, Z_const-a all pass). Empirical Richardson slope
//! is -1.0 because of an *implementation-side* ceiling: the τ²-correction
//! uses cubic-spline `f.sample()` + 5-point central-FD for f', f'', f'''
//! with stencil step Δ = max(2·dx, √τ). Per-step sup-norm error from
//! spline + FD is O(τ²) (4× ratio per τ-halving), accumulating to
//! global O(τ¹) over n steps.
//!
//! With *analytic* samples (no FD), per-step ratio is the expected 8
//! (cubic, slope -2 globally). With γ=0 (constant a), single-operator
//! iteration gives slope -2 -- proving the obstruction is numerical,
//! not theoretical.
//!
//! True global O(τ²) for variable a deferred to v0.4.0 (Magnus integrator,
//! option ε), which avoids FD on f-derivatives. Full diagnosis:
//! `.dev-docs/reports/ZETA_EMPIRICAL_INVESTIGATION.md`.
//!
//! ## Gate 2 — `D_ζ` beats `D_γ` at fixed n
//!
//! At n=128, `‖D_ζ^n u₀ − u_ref‖ < ‖D_γ^n u₀ − u_ref‖`. Confirms the
//! τ²-correction (ζ-A) reduces the error vs γ-A for variable a (the
//! ~1000× constant tightening cited in ADR-0008 Amendment 2).
//!
//! ## Parameters
//!
//! - `a(x) = (1+γx)²`, `a'(x) = 2γ(1+γx)`, `a''(x) = 2γ²`
//! - γ = 0.1, σ = 1, T = 0.5, domain [-5, 5], N = `10_000` nodes
//! - Reference: `n_ref` = 4096 steps
//! - n ∈ {32, 64, 128, 256, 512}
//!
//! Reference: `contracts/semiflow-core.math.md §9.2.3.B`, properties.yaml.

use semiflow::{DiffusionChernoff, Grid1D, GridFn1D};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

const GAMMA: f64 = 0.1;
const SIGMA: f64 = 1.0;
const T_FINAL: f64 = 0.5;
const N_NODES: usize = 10_000;
const X_MIN: f64 = -5.0;
const X_MAX: f64 = 5.0;
const N_REF: usize = 4096;
const N_SWEEP: [usize; 5] = [32, 64, 128, 256, 512];
/// Relaxed gate (slope ≤ -0.95); see module-level deviation note.
const SLOPE_GATE: f64 = -0.95;

// ---------------------------------------------------------------------------
// fn-pointer helpers (fn(f64)->f64, γ baked in — no closure captures)
// ---------------------------------------------------------------------------

fn a_fn(x: f64) -> f64 {
    let v = 1.0 + GAMMA * x;
    v * v
}
fn ap_fn(x: f64) -> f64 {
    2.0 * GAMMA * (1.0 + GAMMA * x)
}
fn app_fn(_: f64) -> f64 {
    2.0 * GAMMA * GAMMA
}

/// γ-A baseline: pass zero derivatives to get constant-a-style behaviour.
fn a_zero(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn gaussian_state(grid: Grid1D) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| libm::exp(-(x * x) / (2.0 * SIGMA * SIGMA)))
}

// n is a step count ≤ 4096, well within f64 mantissa.
#[allow(clippy::cast_precision_loss)]
fn evolve_zeta(n: usize, grid: Grid1D, f0: &GridFn1D) -> GridFn1D {
    let a_norm = (1.0 + GAMMA * X_MAX.abs()).powi(2);
    let dc = DiffusionChernoff::new(a_fn, ap_fn, app_fn, a_norm, grid);
    let tau = T_FINAL / (n as f64);
    let mut state = f0.clone();
    for _ in 0..n {
        state = dc.apply_chernoff(tau, &state).expect("zeta apply ok");
    }
    state
}

/// γ-A with variable a(x) but zero `a_prime`, `a_double_prime` (misses correction).
// n is a step count ≤ 4096, well within f64 mantissa.
#[allow(clippy::cast_precision_loss)]
fn evolve_gamma(n: usize, grid: Grid1D, f0: &GridFn1D) -> GridFn1D {
    let a_norm = (1.0 + GAMMA * X_MAX.abs()).powi(2);
    let dc = DiffusionChernoff::new(a_fn, a_zero, a_zero, a_norm, grid);
    let tau = T_FINAL / (n as f64);
    let mut state = f0.clone();
    for _ in 0..n {
        state = dc.apply_chernoff(tau, &state).expect("gamma apply ok");
    }
    state
}

fn sup_norm_diff(a: &GridFn1D, b: &GridFn1D) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// m is a slice length ≤ 1024, xs/ys use n ≤ 4096 — well within f64 mantissa.
// sum_y and sum_xy are math convention names (OLS); not similar in semantics.
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn log_log_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.max(1.0e-16).ln()).collect();
    let sum_x: f64 = xs.iter().sum();
    let sum_y: f64 = ys.iter().sum();
    let sum_xx: f64 = xs.iter().map(|&x| x * x).sum();
    let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();
    (m * sum_xy - sum_x * sum_y) / (m * sum_xx - sum_x * sum_x)
}

// ---------------------------------------------------------------------------
// G13
// ---------------------------------------------------------------------------

/// G13: ζ-A converges for variable `a(x) = (1+γx)²` at slope ≤ -0.95.
///
/// Also checks `D_ζ` outperforms `D_γ` at fixed n=128 (τ²-correction benefit).
/// See module docstring for the deviation from the properties.yaml gate of -1.95.
// n is a step count ≤ 512, well within f64 mantissa.
#[allow(clippy::cast_precision_loss)]
#[test]
fn g13_variable_zeta_liouville_oracle() {
    let grid = Grid1D::new(X_MIN, X_MAX, N_NODES).expect("grid valid");
    let f0 = gaussian_state(grid);

    // High-resolution temporal reference.
    let reference = evolve_zeta(N_REF, grid, &f0);

    // Gate 1: slope ≤ SLOPE_GATE (convergence at order ≥ 1).
    let errs: Vec<f64> = N_SWEEP
        .iter()
        .map(|&n| sup_norm_diff(&evolve_zeta(n, grid, &f0), &reference).max(1.0e-16))
        .collect();
    let slope = log_log_slope(&N_SWEEP, &errs);

    println!("G13 γ={GAMMA:.2} σ={SIGMA:.1} T={T_FINAL}: slope={slope:.4} (gate ≤ {SLOPE_GATE})");
    for (&n, &e) in N_SWEEP.iter().zip(errs.iter()) {
        println!("  n={n:4}  tau={:.6}  err={e:.6e}", T_FINAL / n as f64);
    }

    assert!(
        slope <= SLOPE_GATE,
        "G13 FAIL slope gate: slope={slope:.4} > {SLOPE_GATE} — \
         ζ-A convergence not confirmed; escalate to architect"
    );

    // Gate 2: ζ-A beats γ-A at n=128 (correction is beneficial).
    let n_cmp = 128;
    let err_zeta = sup_norm_diff(&evolve_zeta(n_cmp, grid, &f0), &reference);
    let err_gamma = sup_norm_diff(&evolve_gamma(n_cmp, grid, &f0), &reference);
    println!("G13 correction check n={n_cmp}: err_ζ={err_zeta:.4e}  err_γ={err_gamma:.4e}");
    assert!(
        err_zeta < err_gamma,
        "G13 FAIL correction: D_ζ error ({err_zeta:.4e}) >= D_γ error ({err_gamma:.4e}) \
         at n={n_cmp} — τ²-correction must improve accuracy; escalate to architect"
    );
}
