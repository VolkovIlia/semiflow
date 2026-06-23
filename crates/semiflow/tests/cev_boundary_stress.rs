//! v0.3.1 boundary stress: CEV European call vs varying `S_max` truncation.
//!
//! Contract: `contracts/tests/cev_boundary_stress.yaml`.
//! ADR: docs/adr/0010-v0_3_1-cev-hardening.md.
//!
//! Fixed: S₀=100, K=100, σ₀=0.30, β=0.5, T=1, r=0.05.
//! Swept: `S_max` ∈ {150, 175, 200, 250, 300, 400} with N scaled to dx ≈ 0.39.
//!
//! `sup_err` window: S ∈ [50, min(150, `S_max−25`)].
//! Guard band of 25 units keeps the `LinearExtrapolate` boundary residual out of
//! the measurement at `S_max=150`, where the right window edge would otherwise be
//! the boundary itself (risk `R_S_max_150` in contract §3).
//! For `S_max≥175` the full [50,150] window applies (boundary ≥25 away).
//!
//! Three backstop gates (primary deliverable is the printed saturation table):
//!   boundary-A: err(300) ≤ err(150)  AND  err(400) ≤ err(200)   (saturation)
//!   boundary-B: max `sup_err` for `S_max` ≥ 250  <  1e-2             (saturated regime)
//!   boundary-C: `sup_err` at `S_max=150`  <  2.5e-1                  (no blow-up; guard-band window)
//!     Threshold 2.5e-1 = empirical 1.16e-1 × ~2 rounded to a clean value.
//!     The residual at `S_max=150` (window [50,125]) is a pure corner artefact:
//!     by `S_max=175` error drops to 1.59e-2 (full [50,150] window).

use semiflow_core::{
    grid::{BoundaryPolicy, InterpKind},
    ChernoffSemigroup, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, StrangSplit,
};
use statrs::distribution::{ChiSquared, ContinuousCDF};

// ---------------------------------------------------------------------------
// Fixed parameters (canonical instance, σ₀=0.30 per contract §1)
// ---------------------------------------------------------------------------

const S0: f64 = 100.0;
const K: f64 = 100.0;
const R: f64 = 0.05;
const BETA_PDE: f64 = 0.5;
const T_MAT: f64 = 1.0;
const S_MIN: f64 = 1.0;
const N_STEPS: usize = 256;

// δ² = σ₀²·S₀^(2-2β) = 0.09·100^1 = 9.0  (β=0.5)
const DELTA_SQ: f64 = 9.0;

// ---------------------------------------------------------------------------
// Noncentral χ² CDF (j_max=100 is sufficient for β=0.5 fixed parameters).
// Peak-pass guard applied for symmetry with cev_european_call_sweep.rs.
// ---------------------------------------------------------------------------

fn ncx2_cdf(w: f64, v: f64, lam: f64) -> f64 {
    let half_lam = lam / 2.0;
    let mut sum = 0.0_f64;
    let mut pj = (-half_lam).exp();
    for j in 0_u32..100 {
        let chi = ChiSquared::new(v + 2.0 * f64::from(j)).expect("df > 0");
        let term = pj * chi.cdf(w);
        sum += term;
        // Poisson mass peaks at j ≈ half_lam; tail-test only after passing the peak.
        let past_peak = f64::from(j) > half_lam.max(5.0);
        if term.abs() < 1e-12 && past_peak {
            break;
        }
        pj *= half_lam / f64::from(j + 1);
    }
    sum
}

// ---------------------------------------------------------------------------
// Schroder (1989) closed-form European call (β=0.5 specialised).
// Uses constant DELTA_SQ (spot-normalised convention).
// ---------------------------------------------------------------------------

fn schroder_call(s: f64) -> f64 {
    let beta_s = 2.0 * BETA_PDE; // 1.0
    let two_m_beta = 2.0 - beta_s; // 1.0
    let expon = R * two_m_beta * T_MAT;
    let k_param = 2.0 * R / (DELTA_SQ * two_m_beta * (libm::exp(expon) - 1.0));
    let x = k_param * s.powf(two_m_beta) * libm::exp(expon);
    let y = k_param * K.powf(two_m_beta);
    let df_v = 2.0 / two_m_beta; // 2.0
    let df_v2 = df_v + 2.0; // 4.0
    let q1 = 1.0 - ncx2_cdf(2.0 * y, df_v2, 2.0 * x);
    let q2 = 1.0 - ncx2_cdf(2.0 * x, df_v, 2.0 * y);
    s * q1 - K * libm::exp(-R * T_MAT) * (1.0 - q2)
}

// ---------------------------------------------------------------------------
// Run the CEV PDE on a grid [S_min, s_max] with n_grid nodes.
// Returns (sup_err over windowed [50, win_hi], atm_err at i_atm, win_hi).
// win_hi = min(150, s_max − 25): 25-unit guard band keeps the boundary
// residual out of the measurement when s_max is close to 150.
// ---------------------------------------------------------------------------

fn run_cev_at_smax(s_max: f64, n_grid: usize) -> (f64, f64, f64) {
    let grid = Grid1D::new(S_MIN, s_max, n_grid)
        .unwrap()
        .with_boundary(BoundaryPolicy::LinearExtrapolate)
        .with_interp(InterpKind::CubicHermite);
    let a_norm = 4.5_f64 * s_max; // ½·DELTA_SQ·S_max = 4.5·S_max
    let diffusion =
        DiffusionChernoff::new(|s: f64| 4.5 * s, |_: f64| 4.5, |_: f64| 0.0, a_norm, grid);
    let drift = DriftReactionChernoff::new(|s: f64| R * s - 4.5, |_: f64| -R, R, grid);
    let strang: StrangSplit<DiffusionChernoff, DriftReactionChernoff> =
        StrangSplit::new(diffusion, drift);
    let f0 = GridFn1D::from_fn(grid, |s| (s - K).max(0.0));
    let sg = ChernoffSemigroup::new(strang, N_STEPS).expect("n >= 1");
    let u = sg.evolve(T_MAT, &f0).expect("evolve ok");
    let dx = grid.dx();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // Safety: (S0 - S_MIN) / dx is positive by construction (S0 > S_MIN, dx > 0).
    let i_atm = ((S0 - S_MIN) / dx).round() as usize;
    let s_atm = grid.x_at(i_atm);
    let atm_oracle = schroder_call(s_atm);
    let atm_err = (u.values[i_atm] - atm_oracle).abs();
    let win_hi = 150.0_f64.min(s_max - 25.0);
    let sup_err = compute_sup_err_windowed(&u, grid, win_hi);
    (sup_err, atm_err, win_hi)
}

// Sup-norm error in S ∈ [50, win_hi]; guard band controlled by caller.
fn compute_sup_err_windowed(u: &GridFn1D, grid: Grid1D, win_hi: f64) -> f64 {
    let mut max_err = 0.0_f64;
    for i in 0..grid.n {
        let s = grid.x_at(i);
        if s >= 50.0 && s <= win_hi {
            let oracle = schroder_call(s);
            let err = (u.values[i] - oracle).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

// Print markdown saturation table (primary deliverable per contract §4).
// Columns include the measurement window upper bound for self-documenting output.
fn print_table(rows: &[(f64, usize, f64, f64, f64)]) {
    eprintln!("| S_max | N    | window | sup_err  | atm_err  |");
    eprintln!("|-------|------|--------|----------|----------|");
    for &(s_max, n, sup_err, atm_err, win_hi) in rows {
        eprintln!("| {s_max:5.1} | {n:4} | {win_hi:6.1} | {sup_err:.2e} | {atm_err:.2e} |");
    }
}

// ---------------------------------------------------------------------------
// G_cev_boundary_A/B/C — gate helper and stress test
// ---------------------------------------------------------------------------

/// Assert the three backstop gates from `contracts/tests/cev_boundary_stress.yaml` §4.
///
/// rows: [(`s_max`, `n_grid`, `sup_err`, `atm_err`, `win_hi`)] — same order as configs.
/// Index layout: 0→150, 1→175, 2→200, 3→250, 4→300, 5→400.
fn assert_gates(rows: &[(f64, usize, f64, f64, f64)]) {
    // G_cev_boundary_A: saturation (not strict monotone).
    let err_150 = rows[0].2;
    let err_200 = rows[2].2;
    let err_300 = rows[4].2;
    let err_400 = rows[5].2;
    assert!(
        err_300 <= err_150,
        "G_cev_boundary_A: err(300)={err_300:.3e} > err(150)={err_150:.3e}",
    );
    assert!(
        err_400 <= err_200,
        "G_cev_boundary_A: err(400)={err_400:.3e} > err(200)={err_200:.3e}",
    );

    // G_cev_boundary_B: saturated regime sup_err < 1e-2.
    let sat_max = rows
        .iter()
        .filter(|r| r.0 >= 250.0)
        .map(|r| r.2)
        .fold(0.0_f64, f64::max);
    assert!(
        sat_max < 1e-2,
        "G_cev_boundary_B: saturated sup_err={sat_max:.3e} >= 1e-2",
    );

    // G_cev_boundary_C: no-blow-up at most aggressive truncation.
    // Guard-band window [50,125] excludes the boundary residual. Threshold 2.5e-1
    // = empirical 1.16e-1 × ~2. By S_max=175 error drops to ~1.6e-2 (full window).
    assert!(
        err_150 < 2.5e-1,
        "G_cev_boundary_C: err(150)={err_150:.3e} >= 2.5e-1 — solver blow-up",
    );
}

/// Boundary stress: `S_max` ∈ {150, 175, 200, 250, 300, 400}, dx ≈ 0.39.
///
/// Primary output is the saturation table (stderr). Gates are loose backstops.
/// `sup_err` uses a guard-band window [50, min(150, `S_max−25`)] so the
/// `LinearExtrapolate` boundary residual is excluded from the measurement.
/// Contract: `contracts/tests/cev_boundary_stress.yaml` §4.
#[test]
fn cev_boundary_stress() {
    let configs = [
        (150.0_f64, 384_usize),
        (175.0, 448),
        (200.0, 512),
        (250.0, 640),
        (300.0, 768),
        (400.0, 1024),
    ];
    // rows: (s_max, n_grid, sup_err, atm_err, win_hi)
    let mut rows = Vec::with_capacity(configs.len());
    for &(s_max, n_grid) in &configs {
        let (sup_err, atm_err, win_hi) = run_cev_at_smax(s_max, n_grid);
        rows.push((s_max, n_grid, sup_err, atm_err, win_hi));
    }
    // Primary deliverable: table before any assert (contract §4 reporting).
    print_table(&rows);
    assert_gates(&rows);
}
