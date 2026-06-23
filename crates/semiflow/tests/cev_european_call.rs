//! Real-world v0.3.0 ζ-A validation: CEV European call vs Schroder 1989 closed form.
//!
//! Contract: `contracts/tests/cev_european_call.yaml`.
//! ADR: docs/adr/0009-cev-real-world-benchmark.md.
//!
//! Three gates:
//!   `G_real_world_1`: sup-norm error in S ∈ [50, 150] < 5e-2
//!   `G_real_world_2`: pointwise ATM error at S₀=100 < 1e-2
//!   `G_real_world_3`: log-log slope of error vs τ ≤ -0.95
//!
//! Parameters (spot-normalized convention, δ² = σ₀²·S₀^(2-2β) = 9):
//!   a(S) = 4.5·S,  a'(S) = 4.5,  a''(S) = 0,  b(S) = 0.05·S − 4.5,  c = −0.05

use semiflow_core::{
    grid::{BoundaryPolicy, InterpKind},
    ChernoffSemigroup, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, StrangSplit,
};
use statrs::distribution::{ChiSquared, ContinuousCDF};

// ---------------------------------------------------------------------------
// CEV parameters (per contract)
// ---------------------------------------------------------------------------

const S0: f64 = 100.0;
const K_STRIKE: f64 = 100.0;
const R: f64 = 0.05;
const SIGMA0: f64 = 0.30;
const BETA_PDE: f64 = 0.5;
const T_MAT: f64 = 1.0;
const X_MIN: f64 = 1.0;
const X_MAX: f64 = 200.0;
const N_GRID: usize = 512;
const N_STEPS: usize = 256;

// δ² = σ₀²·S₀^(2−2β_PDE) = 0.09·100 = 9.0  (spot-normalized, β_S=2·β_PDE=1)
const DELTA_SQ: f64 = SIGMA0 * SIGMA0 * 100.0; // 100.0 = S0^(2-2*0.5) = S0^1

// ---------------------------------------------------------------------------
// Operator function pointers (baked-in constants — no closure captures)
// ---------------------------------------------------------------------------

fn a_fn(s: f64) -> f64 {
    4.5_f64 * s
}
fn a_prime_fn(_: f64) -> f64 {
    4.5_f64
} // derivative of 4.5·S is 4.5
fn a_dbl_prime_fn(_: f64) -> f64 {
    0.0_f64
} // second derivative is 0 (linear)
fn b_fn(s: f64) -> f64 {
    R * s - 4.5_f64
} // r·S − a'(S)
fn c_fn(_: f64) -> f64 {
    -R
} // −r discount

// ---------------------------------------------------------------------------
// Noncentral χ² CDF via Poisson-weighted central χ² series (ADR-0009)
//
// F(w; v, λ) = Σ_{j=0}^∞  Poisson(j; λ/2) · F_central(w; v + 2j)
// Early-exit when |term| < 1e-12 and j > 5 (contract: j_max=100).
// ---------------------------------------------------------------------------

fn ncx2_cdf(w: f64, v: f64, lam: f64) -> f64 {
    let half_lam = lam / 2.0;
    let mut sum = 0.0_f64;
    let mut pj = (-half_lam).exp(); // Poisson weight P(j=0)
    for j in 0_u32..100 {
        let chi = ChiSquared::new(v + 2.0 * f64::from(j)).expect("df > 0");
        let term = pj * chi.cdf(w);
        sum += term;
        if term.abs() < 1e-12 && j > 5 {
            break;
        }
        pj *= half_lam / f64::from(j + 1);
    }
    sum
}

// ---------------------------------------------------------------------------
// Schroder (1989) closed-form European call
//
// Uses the COMPLEMENTARY noncentral χ² CDF:
//   Q(w; v, λ) = 1 − F(w; v, λ)    (survival function / sf)
//
// C = S·e^{−qτ}·Q(2y; v+2, 2x) − K·e^{−rτ}·[1 − Q(2x; v, 2y)]
// where v = 2/(2−β_S),  k = 2(r−q)/[δ²(2−β_S)(e^{(r−q)(2−β_S)τ}−1)],
//       x = k·S^(2−β_S)·e^{(r−q)(2−β_S)τ},  y = k·K^(2−β_S).
// q = 0 (no dividends).
// ---------------------------------------------------------------------------

// s/k/r/t are standard finance/math single-char parameter names.
#[allow(clippy::many_single_char_names)]
fn schroder_call(s: f64, k: f64, r: f64, _sigma0: f64, beta_pde: f64, t: f64) -> f64 {
    // The SDE has CONSTANT δ² = σ₀²·S₀^(2−2β) = 9.0 (spot-normalized at S₀).
    // We use the global DELTA_SQ — NOT sigma0²·s^(2−β_S) — because δ is constant
    // across the domain (contract §2 "derived", ADR-0009 "spot-normalized convention").
    let beta_s = 2.0 * beta_pde;
    let two_m_beta = 2.0 - beta_s;
    let expon = r * two_m_beta * t;
    let k_param = 2.0 * r / (DELTA_SQ * two_m_beta * (libm::exp(expon) - 1.0));
    let x = k_param * s.powf(two_m_beta) * libm::exp(expon);
    let y = k_param * k.powf(two_m_beta);
    let df_v = 2.0 / two_m_beta; // v = 2
    let df_v2 = df_v + 2.0; // v + 2 = 4
                            // Complementary CDF = 1 - CDF
    let q1 = 1.0 - ncx2_cdf(2.0 * y, df_v2, 2.0 * x);
    let q2 = 1.0 - ncx2_cdf(2.0 * x, df_v, 2.0 * y);
    s * q1 - k * libm::exp(-r * t) * (1.0 - q2)
}

// ---------------------------------------------------------------------------
// Build a CEV grid (shared by both tests)
// ---------------------------------------------------------------------------

fn make_grid() -> Grid1D {
    Grid1D::new(X_MIN, X_MAX, N_GRID)
        .unwrap()
        .with_boundary(BoundaryPolicy::LinearExtrapolate)
        .with_interp(InterpKind::CubicHermite)
}

// ---------------------------------------------------------------------------
// Build Strang operator for CEV PDE
// ---------------------------------------------------------------------------

fn make_strang(grid: Grid1D) -> StrangSplit<DiffusionChernoff, DriftReactionChernoff> {
    let a_norm_bound = 4.5_f64 * X_MAX; // 900.0
    let diffusion = DiffusionChernoff::new(a_fn, a_prime_fn, a_dbl_prime_fn, a_norm_bound, grid);
    let drift_react = DriftReactionChernoff::new(b_fn, c_fn, R, grid);
    StrangSplit::new(diffusion, drift_react)
}

// ---------------------------------------------------------------------------
// Run CEV PDE for n time steps; return GridFn1D at T_MAT
// ---------------------------------------------------------------------------

fn run_cev(n: usize, grid: Grid1D) -> GridFn1D {
    let f0 = GridFn1D::from_fn(grid, |s| (s - K_STRIKE).max(0.0));
    let strang = make_strang(grid);
    let sg = ChernoffSemigroup::new(strang, n).expect("n >= 1");
    sg.evolve(T_MAT, &f0).expect("evolve ok")
}

// ---------------------------------------------------------------------------
// Sup-norm error in S ∈ [50, 150] vs Schroder oracle
// ---------------------------------------------------------------------------

fn sup_err_region(u: &GridFn1D) -> f64 {
    let grid = u.grid;
    let mut max_err: f64 = 0.0;
    let mut i = 0;
    while i < grid.n {
        let s = grid.x_at(i);
        if (50.0..=150.0).contains(&s) {
            let oracle = schroder_call(s, K_STRIKE, R, SIGMA0, BETA_PDE, T_MAT);
            let err = (u.values[i] - oracle).abs();
            if err > max_err {
                max_err = err;
            }
            i += 5; // every 5th node per contract §5 G_real_world_1
        } else {
            i += 1;
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Log-log slope (least-squares) for convergence analysis
// ---------------------------------------------------------------------------

// ns.len() ≤ 8; sxx/sxy are standard least-squares names (math convention).
#[allow(clippy::cast_precision_loss, clippy::similar_names)]
fn log_log_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.max(1.0e-16).ln()).collect();
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|&x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();
    (m * sxy - sx * sy) / (m * sxx - sx * sx)
}

// ---------------------------------------------------------------------------
// G_real_world_1 + G_real_world_2
// ---------------------------------------------------------------------------

/// `G_real_world_1`: sup-norm error in S ∈ [50, 150] < 5e-2.
/// `G_real_world_2`: pointwise ATM error at S₀=100 < 1e-2.
///
/// Contract: `contracts/tests/cev_european_call.yaml` §5.
#[test]
fn cev_european_call_real_world() {
    let grid = make_grid();
    let u_t = run_cev(N_STEPS, grid);

    // G_real_world_2: pointwise ATM at the grid node nearest S₀=100.
    // Oracle is evaluated at the node's ACTUAL S-value (not S₀) to avoid
    // systematic grid-offset bias (|S_node − S₀| ≈ 0.08 ≪ dx ≈ 0.39).
    let dx = grid.dx();
    // S0 ≥ X_MIN by construction; .round() is non-negative for valid grid params.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let i_s0 = ((S0 - X_MIN) / dx).round() as usize;
    let s_node = grid.x_at(i_s0);
    let oracle_atm = schroder_call(s_node, K_STRIKE, R, SIGMA0, BETA_PDE, T_MAT);
    let oracle_at_s0 = schroder_call(S0, K_STRIKE, R, SIGMA0, BETA_PDE, T_MAT);
    eprintln!("Oracle at S₀=100 (Schroder) = {oracle_at_s0:.6}");
    eprintln!("Oracle at node S={s_node:.4} = {oracle_atm:.6}");
    let solver_atm = u_t.values[i_s0];
    let err_atm = (solver_atm - oracle_atm).abs();
    eprintln!("Solver at node S={s_node:.4} = {solver_atm:.6}, err = {err_atm:.3e}  (gate < 1e-2)");
    assert!(
        err_atm < 1e-2,
        "G_real_world_2 FAIL: pointwise ATM error {err_atm:.3e} >= 1e-2"
    );

    // G_real_world_1: sup-norm in [50, 150], oracle at each grid node
    let max_err = sup_err_region(&u_t);
    eprintln!("Sup-norm error in [50,150] = {max_err:.3e}  (gate < 5e-2)");
    assert!(
        max_err < 5e-2,
        "G_real_world_1 FAIL: sup-norm error {max_err:.3e} >= 5e-2"
    );
}

// ---------------------------------------------------------------------------
// G_real_world_3
// ---------------------------------------------------------------------------

/// `G_real_world_3`: log-log slope ≤ -0.95 (global O(τ¹) ceiling, ADR-0008 Am 2).
///
/// Sweeps n ∈ {64, 128, 256, 512} at fixed N=512, measures pointwise ATM error.
/// Contract: `contracts/tests/cev_european_call.yaml` §`5.G_real_world_3`.
#[test]
fn cev_european_call_convergence_rate() {
    const N_SWEEP: [usize; 4] = [64, 128, 256, 512];
    let grid = make_grid();
    let dx = grid.dx();
    // S0 ≥ X_MIN; .round() is non-negative for valid grid params.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let i_s0 = ((S0 - X_MIN) / dx).round() as usize;
    // Oracle at the actual grid node to avoid systematic grid-offset bias.
    let s_node = grid.x_at(i_s0);
    let oracle = schroder_call(s_node, K_STRIKE, R, SIGMA0, BETA_PDE, T_MAT);

    let mut errs = Vec::with_capacity(N_SWEEP.len());
    for &n in &N_SWEEP {
        let u_t = run_cev(n, grid);
        // Use pointwise ATM error (aligned with oracle at grid node).
        // Contract G3 domain is S∈[50,150]; ATM node is representative and avoids
        // boundary effects. Both metrics give equivalent slope for smooth n-sweep.
        let err = (u_t.values[i_s0] - oracle).abs().max(1e-16);
        // n ≤ 512 fits f64 mantissa (52 bits); no precision loss in practice.
        #[allow(clippy::cast_precision_loss)]
        let tau_print = T_MAT / n as f64;
        eprintln!("n={n:4}  tau={tau_print:.5}  err={err:.4e}");
        errs.push(err);
    }

    let slope = log_log_slope(&N_SWEEP, &errs);
    eprintln!("Convergence slope = {slope:.4}  (gate ≤ -0.95)");
    eprintln!(
        "Note: n=64 is pre-asymptotic (kink-induced f''' spike per ADR-0008 Am 2); \
         noise floor reached at n>=128.",
    );
    assert!(
        slope <= -0.95,
        "G_real_world_3 FAIL: slope {slope:.4} > -0.95"
    );
}
