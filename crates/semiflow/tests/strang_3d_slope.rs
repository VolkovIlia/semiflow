//! `G5_3D` — empirical convergence-rate test (slope ≤ -1.95).
//!
//! math.md §10.8.7; ADR-0024; gate `G5_3D`.
//!
//! Gated: `#[cfg(feature = "slow-tests")]`
//!
//! Validates order-2 (Chernoff-step / spatial) convergence of
//! `Strang3D<DiffusionChernoff, DiffusionChernoff, DiffusionChernoff>` via
//! a closed-form 3D Gaussian oracle (eq. 10.8.7) over `N ∈ {32, 64, 128, 256}` per-axis
//! (mirror `G4_NS2D_aniso` convention; see `docs/adr/0024-tensor-3d.md` Amendment 2026-05-09
//! for diagnosis and v0.9.0 `G4_NS2D_aniso` precedent).
//! N=16 was dropped: dx=0.667 under-resolves the σ=0.707 Gaussian IC, producing
//! a D1 calibration bias that contaminates the coarse-end slope estimate
//! (ADR-0024 Amendment 2026-05-09).
//!
//! # Design
//! - Domain `[-5, 5]³`, `T = 0.2`, `n = 200` Chernoff steps (fixed temporal).
//! - `N ∈ {32, 64, 128, 256}` — probe spatial convergence.
//! - Diffusion: `a = b = c = 0.1` (constant, per axis).
//! - Initial datum: `u_0(x,y,z) = exp(-(x²+y²+z²))`.
//! - Oracle: `D^{-3/2}·exp(-(x²+y²+z²)/D)`, `D = 1+4·a·T`.
//! - Slope gate: `≤ −1.95` (NON-NEGOTIABLE, math.md §10.8.7, ADR-0024).

#![cfg(feature = "slow-tests")]

use semiflow_core::{ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid3D, GridFn3D, Strang3D};

// ---------------------------------------------------------------------------
// Gate constants
// ---------------------------------------------------------------------------

const SLOPE_GATE: f64 = -1.95;
/// Per-axis node counts for spatial convergence.
const N_SPATIAL: [usize; 4] = [32, 64, 128, 256];
/// Fixed Chernoff step count (large → temporal error negligible vs. spatial).
const N_STEPS: usize = 200;
const T_FINAL: f64 = 0.2;
const DIFFUSION_A: f64 = 0.1;
const X_MIN: f64 = -5.0;
const X_MAX: f64 = 5.0;

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// 3D heat oracle: `D^{-3/2}·exp(-(x²+y²+z²)/D)`, `D = 1+4·a·t`.
#[allow(clippy::many_single_char_names)]
#[inline]
fn oracle(t: f64, x: f64, y: f64, z: f64) -> f64 {
    let d = 1.0 + 4.0 * DIFFUSION_A * t;
    d.powf(-1.5) * (-(x * x + y * y + z * z) / d).exp()
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

// n_nodes ≤ 64; n_steps ≤ 200 — within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
fn heat_3d_error(n_nodes: usize) -> f64 {
    let gx = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid x valid");
    let gy = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid y valid");
    let gz = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid z valid");
    let g3 = Grid3D::new(gx, gy, gz).expect("Grid3D valid");

    let f0 = GridFn3D::from_fn(g3, |x, y, z| (-(x * x + y * y + z * z)).exp());
    let cx = DiffusionChernoff::new(|_| DIFFUSION_A, |_| 0.0_f64, |_| 0.0_f64, DIFFUSION_A, gx);
    let cy = DiffusionChernoff::new(|_| DIFFUSION_A, |_| 0.0_f64, |_| 0.0_f64, DIFFUSION_A, gy);
    let cz = DiffusionChernoff::new(|_| DIFFUSION_A, |_| 0.0_f64, |_| 0.0_f64, DIFFUSION_A, gz);

    let phi3d = Strang3D::new(cx, cy, cz);
    let semi = ChernoffSemigroup::new(phi3d, N_STEPS).expect("n >= 1");
    let u_n = semi.evolve(T_FINAL, &f0).expect("evolve succeeds");

    let nx = g3.nx();
    let ny = g3.ny();
    let nz = g3.nz();
    let mut max_err: f64 = 0.0;
    for k in 0..nz {
        let zk = gz.x_at(k);
        for j in 0..ny {
            let yj = gy.x_at(j);
            for i in 0..nx {
                let xi = gx.x_at(i);
                let exact = oracle(T_FINAL, xi, yj, zk);
                let err = (u_n.values[k * nx * ny + j * nx + i] - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// OLS log-log slope
// ---------------------------------------------------------------------------

// n ≤ 64 — well within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
fn loglog_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let log_n: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let log_e: Vec<f64> = errs.iter().map(|e| e.ln()).collect();
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
// G5_3D — slope test over N ∈ {16, 32, 64}
// ---------------------------------------------------------------------------

/// `G5_3D`: spatial slope ≤ -1.95 for `Strang3D<DiffusionChernoff×3>` on the
/// 3D Gaussian heat oracle. Fixed n=200 temporal steps; spatial sweep `N ∈ {16, 32, 64}`.
///
/// Gate from `contracts/semiflow-core.properties.yaml` `G5_3D` (v0.9.0, ADR-0024,
/// math.md §10.8.7). Failure BLOCKS v0.9.0 release.
#[test]
fn g5_3d_slope() {
    let mut errs = Vec::with_capacity(N_SPATIAL.len());
    for &n in &N_SPATIAL {
        let e = heat_3d_error(n);
        #[allow(clippy::cast_precision_loss)]
        let dx = (X_MAX - X_MIN) / (n - 1) as f64;
        println!("G5_3D: N={n:3}, dx={dx:.4}, err={e:.4e}");
        errs.push(e);
    }
    let slope = loglog_slope(&N_SPATIAL, &errs);
    println!("G5_3D: slope = {slope:.4}  (gate <= {SLOPE_GATE})");
    assert!(
        slope <= SLOPE_GATE,
        "G5_3D slope {slope:.4} > gate {SLOPE_GATE} — escalate to Architect"
    );
}
