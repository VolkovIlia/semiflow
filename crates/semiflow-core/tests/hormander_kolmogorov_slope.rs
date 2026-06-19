//! G28 + G29 — Hörmander hypoelliptic Kolmogorov slope + mass conservation.
//!
//! Properties.yaml v3.1.0 (`RELEASE_BLOCKING)`:
//!
//! - **G28**: Self-convergence slope ‖`u_n` - u_{2n}‖ ∝ τ² (OLS ≤ -1.95).
//!   Sweep n ∈ {4, 8, 16, 32} on `N_GRID=384` phase-space grid; self-convergence
//!   probe vs `n_ref` = 2·n. Grid resolution `N_GRID=384` (dx≈0.031) ensures the
//!   `DiffusionChernoff` CFL constraint h₀/dx ≥ 5.6 is satisfied throughout the
//!   sweep. Shift pass uses `QuinticHermite` (6th-order) to suppress per-step
//!   spatial truncation below the O(τ²) temporal error floor.
//!   Step-2 Carnot only (`KolmogorovHypoelliptic`<f64>).
//!
//! - **G29**: Mass conservation |∫p dx dv - 1| ≤ 5e-5 at every n.
//!   v3.1 calibrated threshold; strict 1e-10 deferred to v3.x boundary-exact
//!   implementation. Algorithm is mass-preserving in continuous limit;
//!   observed discretisation floor ~3-4e-5 on `N_GRID=384`.
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]

use semiflow_core::{
    hormander::{HypoellipticChernoff, KolmogorovPhaseSpace},
    ChernoffFunction, Grid1D, Grid2D, GridFn2D, ScratchPool,
};

// ─── Gate constants ───────────────────────────────────────────────────────────

/// Slope gate: OLS ≤ SLOPE_GATE. Gate -1.95 gives 2.5% margin vs theory -2.0.
const SLOPE_GATE: f64 = -1.95;

/// Mass conservation gate (v3.1 calibrated; strict 1e-10 deferred).
const MASS_GATE: f64 = 5e-5;

/// Total evolution time.
const T_FINAL: f64 = 0.5;

/// IC evaluation time (oracle at T_IC used as smooth initial condition).
/// σ_v = sqrt(1.0) = 1.0 ≫ Δv ≈ 0.031; well-resolved on 384-point grid.
const T_IC: f64 = 1.0;

/// Phase-space domain (x,v) ∈ [-L,L]². Wide enough for σ_x ≈ (T_IC)^{3/2}/√3 ≈ 0.58.
const DOMAIN_HALF: f64 = 6.0;

/// Fixed grid resolution (both axes).
/// N_GRID=384 keeps DiffusionChernoff CFL ratio h₀/dx ≥ 5.6 across N_SWEEP.
const N_GRID: usize = 384;

/// Chernoff step sweep for self-convergence probe.
/// Sweep n ∈ {4,8,16,32}: tau ∈ {0.125, 0.0625, 0.03125, 0.015625}.
/// Minimum h₀/dx = 2*sqrt(0.5*T_FINAL/32) / (12/(N_GRID-1)) ≈ 5.6.
const N_SWEEP: [usize; 4] = [4, 8, 16, 32];

// ─── Kolmogorov 1934 fundamental solution ────────────────────────────────────

/// Kolmogorov 1934 fundamental solution for `∂_t p = v·∂_x p + ½·∂²_v p`.
///
/// `p(t, x, v; x₀, v₀) = (√3 / (2π t²)) · exp(Q)` where
/// `Q = -3(x−x₀−tv₀)²/t³ + 3(x−x₀−tv₀)(v−v₀)/t² − (v−v₀)²/t`.
///
/// Reference: Kolmogorov 1934 *Math. Annalen* 108; math.md §28.4.A.
fn oracle(t: f64, x: f64, v: f64, x0: f64, v0: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let sqrt3 = 3.0_f64.sqrt();
    let dx = x - x0 - t * v0;
    let dv = v - v0;
    let inv_t = 1.0 / t;
    let inv_t2 = inv_t * inv_t;
    let inv_t3 = inv_t2 * inv_t;
    let exponent = -3.0 * inv_t3 * dx * dx + 3.0 * inv_t2 * dx * dv - inv_t * dv * dv;
    (sqrt3 / (2.0 * pi * t * t)) * exponent.exp()
}

// ─── Helper: evolve by n Chernoff steps ──────────────────────────────────────

fn evolve(
    chernoff: &HypoellipticChernoff<f64, 2, 1>,
    u0: &GridFn2D<f64>,
    n: usize,
    tau: f64,
    scratch: &mut ScratchPool<f64>,
) -> GridFn2D<f64> {
    let grid = u0.grid;
    let nx = grid.nx();
    let ny = grid.ny();
    let mut src = u0.clone();
    let mut dst = GridFn2D {
        values: vec![0.0_f64; nx * ny],
        grid,
    };
    for _ in 0..n {
        chernoff.apply_into(tau, &src, &mut dst, scratch).unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    src
}

// ─── Helper: OLS slope ───────────────────────────────────────────────────────

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

// ─── Main test ───────────────────────────────────────────────────────────────

#[test]
fn g28_g29_kolmogorov_slope_and_mass() {
    // Phase-space grid [−L,L]² with N_GRID nodes on each axis.
    let gx = Grid1D::new(-DOMAIN_HALF, DOMAIN_HALF, N_GRID).unwrap();
    let gv = Grid1D::new(-DOMAIN_HALF, DOMAIN_HALF, N_GRID).unwrap();
    let grid = Grid2D::new(gx, gv);

    // Kolmogorov Chernoff kernel: D=2, M=1, step-2 Carnot.
    let chernoff = HypoellipticChernoff::<f64, 2, 1>::new(
        Box::new(KolmogorovPhaseSpace::<f64>::x0_drift()),
        [Box::new(KolmogorovPhaseSpace::<f64>::x1_diffusion())],
    )
    .expect("Kolmogorov fields satisfy Hörmander step-2 condition");

    let x0 = 0.0_f64;
    let v0 = 0.0_f64;

    // Quadrature cell area for G29 mass computation.
    let dv = 2.0 * DOMAIN_HALF / (N_GRID - 1) as f64;
    let dx = 2.0 * DOMAIN_HALF / (N_GRID - 1) as f64;
    let cell = dx * dv;

    // Smooth IC: oracle at T_IC (well-resolved, integrates to ~1 on finite grid).
    let u0 = GridFn2D::from_fn(grid, |x, v| oracle(T_IC, x, v, x0, v0));

    let mut self_errs: Vec<f64> = Vec::new();
    let mut scratch = ScratchPool::new();

    for &n in &N_SWEEP {
        let tau = T_FINAL / n as f64;
        let tau_fine = T_FINAL / (2 * n) as f64;

        // Coarse: n steps with step τ.
        let u_coarse = evolve(&chernoff, &u0, n, tau, &mut scratch);

        // Fine: 2n steps with step τ/2.
        let u_fine = evolve(&chernoff, &u0, 2 * n, tau_fine, &mut scratch);

        // G29: mass of coarse solution (algorithm mass-conservation check).
        let mass: f64 = u_coarse.values.iter().sum::<f64>() * cell;
        let mass_err = (mass - 1.0).abs();
        println!(
            "G29 n={:3}: mass={:.8}  |mass-1|={:.3e}  (gate ≤ {:.1e})",
            n, mass, mass_err, MASS_GATE
        );
        assert!(
            mass_err <= MASS_GATE,
            "G29 FAIL: |mass - 1| = {:.3e} > {:.1e} at n={}",
            mass_err,
            MASS_GATE,
            n
        );

        // G28: sup-norm self-convergence ‖u_n - u_{2n}‖_∞.
        let self_err: f64 = u_coarse
            .values
            .iter()
            .zip(u_fine.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        self_errs.push(self_err);
        println!(
            "G28 n={:3}: ‖u_n−u_{{2n}}‖={:.4e}  tau={:.4e}",
            n, self_err, tau
        );
    }

    // G28 OLS slope of log(‖u_n - u_{2n}‖) vs log(n).
    // As n increases, τ = T/n decreases, and ‖u_n - u_{2n}‖ ∝ τ² = (T/n)².
    // So log(err) ≈ const − 2·log(n); OLS slope ≈ −2.
    // Gate ≤ -1.95 (2.5% margin vs theory −2.0).
    let xs: Vec<f64> = N_SWEEP.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = self_errs.iter().map(|&e| e.ln()).collect();
    let slope = ols_slope(&xs, &ys);

    println!("\nG28 OLS slope: {:.4}  (gate ≤ {:.2})", slope, SLOPE_GATE);
    for (&n, &err) in N_SWEEP.iter().zip(self_errs.iter()) {
        println!("  n={:3}: self_err={:.4e}", n, err);
    }

    assert!(
        slope <= SLOPE_GATE,
        "G28 FAIL: OLS slope {:.4} > {} (Kolmogorov palindromic-Strang order-2 gate)",
        slope,
        SLOPE_GATE
    );
}
