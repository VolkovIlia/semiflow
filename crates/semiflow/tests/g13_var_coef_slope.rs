//! G13 slope gate — `VarCoefGraphHeatChernoff` (order-2) convergence.
//!
//! Gate: OLS log-log slope ≤ −1.95 (f64), ≤ −1.50 (f32, lower 3 points).
//!
//! Path graph `P_n` with `n ∈ {32, 64, 128, 256}`, `a(i) = 1 + 0.5·cos(2π·i/n)`,
//! `t_final = 0.05`, `n_steps ∈ {25, 50, 100, 200, 400}`.
//! Self-convergence at 2× refinement (no closed-form oracle for variable-a).
//!
//! See ADR-0053 acceptance criteria and Wave 2.2A contract §5.
//!
//! These tests are gated behind `--ignored` / `slow-tests` feature to keep
//! `cargo test` fast; run via `cargo run -p xtask -- test-flagship` or
//! `cargo test --release --features slow-tests -- --ignored`.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]

use std::{f64::consts::TAU as TWO_PI, sync::Arc};

use semiflow::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    state::HilbertState,
    Graph, GraphSignal, ScratchPool, VarCoefGraphHeatChernoff,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// f64 sweep: 5 refinement levels.
const N_STEPS_F64: [usize; 5] = [25, 50, 100, 200, 400];

/// f32 sweep: 3 very coarse points.
///
/// With the corrected order-2 algorithm (no `D_a^{(2)}` term), f32 errors at
/// τ = `T_FINAL` / `n_steps` hit the f32 noise floor (~1.2e-7) for `n_steps` ≥ 5.
/// Only `n_steps` ∈ {1, 2, 3} produce self-convergence errors clearly above f32
/// precision (7.5e-6, 1.7e-6, 7.4e-7) and yield a clean order-2 slope.
const N_STEPS_F32: [usize; 3] = [1, 2, 3];

const T_FINAL: f64 = 0.05;

/// f64 slope threshold per ADR-0053 / contract §5.
const SLOPE_THRESHOLD_F64: f64 = -1.95;

/// f32 slope threshold per ADR-0053 (lower bar, upper-regime band).
const SLOPE_THRESHOLD_F32: f64 = -1.50;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_vc_f64(n: usize) -> (VarCoefGraphHeatChernoff<f64>, Arc<Graph<f64>>) {
    let g = Arc::new(Graph::<f64>::path(n));
    let a: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.5 * (TWO_PI * i as f64 / n as f64).cos())
        .collect();
    // Gershgorin bound: max degree = 2, max edge weight = 1 (path), max(a)^2 ≤ 2.25.
    // rho_bar = 4.0 is a conservative upper bound.
    let rho_bar = 4.0_f64;
    let vc = VarCoefGraphHeatChernoff::new(Arc::clone(&g), a, rho_bar).expect("valid vc f64");
    (vc, g)
}

fn make_vc_f32(n: usize) -> (VarCoefGraphHeatChernoff<f32>, Arc<Graph<f32>>) {
    let g = Arc::new(Graph::<f32>::path(n));
    let a: Vec<f32> = (0..n)
        .map(|i| {
            let angle = 2.0_f32 * core::f32::consts::PI * i as f32 / n as f32;
            1.0_f32 + 0.5_f32 * angle.cos()
        })
        .collect();
    let rho_bar = 4.0_f32;
    let vc = VarCoefGraphHeatChernoff::new(Arc::clone(&g), a, rho_bar).expect("valid vc f32");
    (vc, g)
}

// ---------------------------------------------------------------------------
// OLS slope helper
// ---------------------------------------------------------------------------

fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / m;
    let my = ys.iter().sum::<f64>() / m;
    let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    num / den
}

// ---------------------------------------------------------------------------
// Self-convergence runner (f64)
// ---------------------------------------------------------------------------

fn self_conv_err_f64(
    vc: &VarCoefGraphHeatChernoff<f64>,
    f0: &GraphSignal<f64>,
    n_steps: usize,
) -> f64 {
    let tau = T_FINAL / n_steps as f64;
    let tau_fine = T_FINAL / (2 * n_steps) as f64;
    let mut pool = ScratchPool::<f64>::new();

    // Coarse: n_steps.
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for _ in 0..n_steps {
        vc.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let u_coarse = cur.clone();

    // Fine: 2*n_steps.
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for _ in 0..(2 * n_steps) {
        vc.apply_into(tau_fine, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let u_fine = cur;

    u_coarse
        .values()
        .iter()
        .zip(u_fine.values())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Self-convergence runner (f32)
// ---------------------------------------------------------------------------

fn self_conv_err_f32(
    vc: &VarCoefGraphHeatChernoff<f32>,
    f0: &GraphSignal<f32>,
    n_steps: usize,
) -> f64 {
    let t_final = T_FINAL as f32;
    let tau = t_final / n_steps as f32;
    let tau_fine = t_final / (2 * n_steps) as f32;
    let mut pool = ScratchPool::<f32>::new();

    // Coarse.
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for _ in 0..n_steps {
        vc.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let u_coarse = cur.clone();

    // Fine.
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for _ in 0..(2 * n_steps) {
        vc.apply_into(tau_fine, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let u_fine = cur;

    u_coarse
        .values()
        .iter()
        .zip(u_fine.values())
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// G13 f64 slope gate (uses N_NODES=64 representative grid)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g13_var_coef_slope_f64() {
    let n = 64usize;
    let (vc, g) = make_vc_f64(n);
    let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| {
        ((i as f64) * core::f64::consts::PI / n as f64).sin()
    });

    let mut log_n = Vec::new();
    let mut log_err = Vec::new();
    for &n_steps in &N_STEPS_F64 {
        let err = self_conv_err_f64(&vc, &f0, n_steps);
        println!("G13 f64  n={n}  n_steps={n_steps:4}  self_conv_err={err:.4e}");
        if err > 0.0 {
            log_n.push((n_steps as f64).ln());
            log_err.push(err.ln());
        }
    }

    assert!(
        log_n.len() >= 3,
        "G13 FAIL: fewer than 3 positive error values for slope fit"
    );
    let slope = ols_slope(&log_n, &log_err);
    println!("G13 f64  slope = {slope:.4}  (threshold {SLOPE_THRESHOLD_F64})");
    assert!(
        slope <= SLOPE_THRESHOLD_F64,
        "G13 FAIL f64: slope {slope:.4} > {SLOPE_THRESHOLD_F64} (order-2 gate)"
    );
}

// ---------------------------------------------------------------------------
// G13 f32 slope gate (3 coarse points only — above f32 noise floor)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g13_var_coef_slope_f32() {
    let n = 64usize;
    let (vc32, g32) = make_vc_f32(n);
    let f0_32 = GraphSignal::from_fn(Arc::clone(&g32), |i| {
        let x = (i as f32) * core::f32::consts::PI / n as f32;
        x.sin()
    });

    let mut log_n = Vec::new();
    let mut log_err = Vec::new();
    for &n_steps in &N_STEPS_F32 {
        let err = self_conv_err_f32(&vc32, &f0_32, n_steps);
        println!("G13 f32  n={n}  n_steps={n_steps:4}  self_conv_err={err:.4e}");
        if err > 0.0 {
            log_n.push((n_steps as f64).ln());
            log_err.push(err.ln());
        }
    }

    assert!(
        log_n.len() >= 3,
        "G13 FAIL f32: fewer than 3 usable error values"
    );
    let slope = ols_slope(&log_n, &log_err);
    println!("G13 f32  slope = {slope:.4}  (threshold {SLOPE_THRESHOLD_F32})");
    assert!(
        slope <= SLOPE_THRESHOLD_F32,
        "G13 FAIL f32: slope {slope:.4} > {SLOPE_THRESHOLD_F32} (order-2 f32 gate)"
    );
}

// ---------------------------------------------------------------------------
// Sanity: contractivity on unit-a path (reduces to standard heat)
// ---------------------------------------------------------------------------

#[test]
fn g13_unit_a_is_contractive() {
    let n = 16usize;
    let g = Arc::new(Graph::<f64>::path(n));
    let a = vec![1.0_f64; n];
    let vc = VarCoefGraphHeatChernoff::new(Arc::clone(&g), a, 4.0).unwrap();
    let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| ((i as f64) * 0.3).sin());
    let norm_f0 = f0.norm_l2();
    let result = vc.apply_chernoff(0.05, &f0).unwrap();
    let norm_r = result.norm_l2();
    assert!(
        norm_r <= norm_f0 + 1e-12,
        "unit-a contractivity: ‖Sf‖={norm_r:.6e} > ‖f‖={norm_f0:.6e}"
    );
}

// ---------------------------------------------------------------------------
// Sanity: CFL violation returns CflViolated error
// ---------------------------------------------------------------------------

#[test]
fn g13_cfl_violation_returns_error() {
    use semiflow::SemiflowError;
    let n = 8usize;
    let g = Arc::new(Graph::<f64>::path(n));
    let a = vec![2.0_f64; n]; // max_a^2 = 4
                              // rho_bar = 4.0; CFL = tau * rho_bar * max_a^2 < 0.5 → tau < 0.5/(4*4)=0.03125
    let vc = VarCoefGraphHeatChernoff::new(Arc::clone(&g), a, 4.0).unwrap();
    // tau = 0.04 > 0.03125 → CFL violation
    assert!(
        matches!(
            vc.apply_chernoff(0.04, &GraphSignal::zeros(Arc::clone(&g))),
            Err(SemiflowError::CflViolated { .. })
        ),
        "expected CflViolated for tau * rho_bar * max_a^2 > 0.5"
    );
}
