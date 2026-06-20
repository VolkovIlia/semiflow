//! `G_HORM_ENGEL` — Engel step-3 Carnot palindromic Strang-Hörmander self-convergence gate.
//!
//! Properties.yaml v4.5+ ADR-0095 (`RELEASE_BLOCKING)`:
//!
//! - **`G_HORM_ENGEL`**: Self-convergence slope ‖`u_n` − u_{2n}‖_∞ ∝ τ² (OLS ≤ -1.95).
//!   Sweep n ∈ {16, 32, 64, 128} on `N_GRID=32` per axis 4D Gaussian IC; palindromic
//!   Strang-Hörmander composition `exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)`.
//!
//! No closed-form oracle exists for Engel (Bonfiglioli 2007 §18.3 — Folland-Kaplan
//! restricted to H-type groups; Engel is NOT H-type). Validation uses probe-vs-2N
//! self-convergence (mirror v2.2 `G_NS2D_aniso` pattern).
//!
//! Memory: 32⁴ × 8 B = 8 MB per state buffer × 4 active = 32 MB peak.
//! Compute: ~4-5 minutes on i7-12700K (acceptable under slow-tests budget).
//!
//! Feature gate: `slow-tests`.
//!
//! References: ADR-0095, math.md §28.bis, Bonfiglioli-Lanconelli-Uguzzoni 2007 §4.3.6.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize→f64 in OLS; n ≤ 128 ≤ 2^52

use semiflow_core::{
    grid_nd::{GridFnND, GridND},
    hormander::HypoellipticChernoff,
    ChernoffFunction, Grid1D, ScratchPool,
};

// ─── Gate constants ───────────────────────────────────────────────────────────

/// Slope gate: OLS ≤ `SLOPE_GATE`. Gate -1.95 gives 2.5% margin vs theory -2.0.
const SLOPE_GATE: f64 = -1.95;

/// Total evolution time.
const T_FINAL: f64 = 0.5;

/// Spatial domain: each axis ∈ [-L, L] (4D box [-L, L]⁴).
const DOMAIN_HALF: f64 = 2.5;

/// Grid resolution per axis. 32⁴ ≈ 1M points × 8 B = 8 MB per state.
const N_GRID: usize = 32;

/// Chernoff step sweep for self-convergence (probe-vs-2N mirror `G_NS2D_aniso`).
const N_SWEEP: [usize; 4] = [16, 32, 64, 128];

// ─── Helper: evolve n Chernoff steps ─────────────────────────────────────────

fn evolve(
    chernoff: &HypoellipticChernoff<f64, 4, 2>,
    u0: &GridFnND<f64, 4>,
    n: usize,
    tau: f64,
    scratch: &mut ScratchPool<f64>,
) -> GridFnND<f64, 4> {
    let grid = u0.grid.clone();
    let len = u0.values.len();
    let mut src = u0.clone();
    let mut dst = GridFnND {
        values: vec![0.0_f64; len],
        grid: grid.clone(),
    };
    for _ in 0..n {
        chernoff.apply_into(tau, &src, &mut dst, scratch).unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    src
}

// ─── Helper: OLS slope of log(y) vs log(x) ───────────────────────────────────

fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    if den.abs() < 1e-30 {
        0.0
    } else {
        num / den
    }
}

// ─── Helper: sup-norm difference ─────────────────────────────────────────────

fn sup_diff(a: &GridFnND<f64, 4>, b: &GridFnND<f64, 4>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ─── Main test ───────────────────────────────────────────────────────────────

#[test]
#[ignore = "slow flagship gate; run with: cargo run -p xtask -- test-flagship"]
fn g_horm_engel_slope() {
    // 4D grid: (x₁, x₂, x₃, x₄) ∈ [-L, L]⁴ with N_GRID nodes on each axis.
    let ax = Grid1D::new(-DOMAIN_HALF, DOMAIN_HALF, N_GRID).unwrap();
    let grid = GridND::<f64, 4>::new([ax, ax, ax, ax]).unwrap();

    // Engel step-3 Chernoff kernel: D=4, M=2.
    let chernoff = HypoellipticChernoff::<f64, 4, 2>::new_engel()
        .expect("Engel fields satisfy Hörmander step-3 condition");

    // IC: 4D Gaussian centered at origin (smooth, well-resolved on 32-pt grid).
    // f₀(x₁, x₂, x₃, x₄) = exp(-½(x₁² + x₂² + x₃² + x₄²))
    let u0 = GridFnND::from_fn(grid.clone(), |x: &[f64; 4]| {
        (-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) * 0.5).exp()
    });

    println!("Computing G_HORM_ENGEL self-convergence sweep...");
    let grid_len = grid.len();
    println!("Grid: {N_GRID}⁴ = {grid_len} points, domain [-{DOMAIN_HALF}, {DOMAIN_HALF}]⁴");
    println!("Sweep: n ∈ {N_SWEEP:?}, T_FINAL={T_FINAL}");

    let mut self_errs: Vec<f64> = Vec::new();
    let mut scratch = ScratchPool::new();

    for &n in &N_SWEEP {
        let tau = T_FINAL / n as f64;
        let tau_fine = T_FINAL / (2 * n) as f64;

        // Coarse: n steps with step τ = T/n.
        let u_coarse = evolve(&chernoff, &u0, n, tau, &mut scratch);

        // Fine: 2n steps with step τ/2 = T/(2n).
        let u_fine = evolve(&chernoff, &u0, 2 * n, tau_fine, &mut scratch);

        // Self-convergence error ‖u_n − u_{2n}‖_∞ (probe-vs-2N).
        let self_err = sup_diff(&u_coarse, &u_fine);
        self_errs.push(self_err);
        println!("G_HORM_ENGEL n={n:3}: ‖u_n−u_{{2n}}‖_∞={self_err:.4e}  τ={tau:.4e}");
    }

    // OLS slope of log(‖u_n − u_{2n}‖) vs log(n).
    // Palindromic Strang-Hörmander order-2 → slope ≈ −2.
    let xs: Vec<f64> = N_SWEEP.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = self_errs.iter().map(|&e| e.ln()).collect();
    let slope = ols_slope(&xs, &ys);

    println!();
    println!("G_HORM_ENGEL OLS slope: {slope:.4}  (gate ≤ {SLOPE_GATE:.2})");
    println!("Per-n summary:");
    for (&n, &err) in N_SWEEP.iter().zip(self_errs.iter()) {
        println!("  n={n:3}: self_err={err:.4e}");
    }

    // ADR-0095 failure mode interpretation:
    // slope ≤ -1.95: PASS — Galkin-Remizov K=2 framework empirically extends to step-3 Engel.
    // slope ∈ (-1.95, -1.0): partial — ship as experimental.
    // slope > -1.0: refuted — Outcome B fallback.
    assert!(
        slope <= SLOPE_GATE,
        "G_HORM_ENGEL FAIL: OLS slope {slope:.4} > {SLOPE_GATE} \
         (Engel step-3 Carnot palindromic-Strang order-2 gate; ADR-0095)"
    );

    println!("G_HORM_ENGEL PASS: slope {slope:.4} ≤ {SLOPE_GATE} ✓");
}
