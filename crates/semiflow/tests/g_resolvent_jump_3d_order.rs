//! `G_RESOLVENT_JUMP_3D_ORDER` — 3D parabolic resolvent time-jump convergence gate.
//!
//! Gate spec (`RELEASE_BLOCKING`, slow-tests, ADR-0148, §47.8):
//!
//!   OLS slope d `log‖jump_M` − ref‖_∞ / d log(1/M) ≥ +1.95
//!   M ∈ {6, 8, 10, 12, 14}, t = 100, N = 8³, Gaussian IC, Neumann BC.
//!
//!   G24 convention (SAME AS `G_RESOLVENT_JUMP_ORDER` — copy verbatim, §47.5):
//!   positive slope = convergence as M grows; ≥ +1.95 PASSES.
//!   Pre-flight (`scripts/resolvent_jump_2d3d_kit.py` A5): slope +9.79.
//!
//! Small grid N=8³ per the pre-flight oracle (banded solve bw=nx*ny=64 is O(512·64²)
//! = ~2M ops per node; 14 × 40 = 560 nodes total — feasible in slow-tests).
//!
//! Self-convergence reference: `ResolventJumpChernoff3D` at `M_ref` = 40. Both use
//! the banded LHP solve, ensuring no continuous/discrete mismatch.
//!
//! ## Operator
//!
//! A = ∂²ₓ + ∂²ᵧ + ∂`²_z` on [−5,5]³, N=8 per axis, Neumann BC.
//! Row-major x-fastest: `idx(i,j,k) = k*nx*ny + j*nx + i` (grid3d.rs I-T1-3D).
//!
//! ## Sign convention (NORMATIVE — do NOT alter)
//!
//! Same G24 convention as `G_RESOLVENT_JUMP_ORDER` (§47.5) and `G_RESOLVENT_JUMP_2D_ORDER`.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize→f64 in OLS/sweep; values ≤ 40 ≤ 2^52

use semiflow::{Grid1D, Grid3D, GridFn3D, ResolventJumpChernoff3D};

// ---------------------------------------------------------------------------
// Gate constants — do NOT relax without ADR + properties.yaml bump.
// ---------------------------------------------------------------------------

/// OLS slope gate: d log(err) / d log(1/M) ≥ +1.95 (G24 convention).
const SLOPE_GATE: f64 = 1.95;

/// Grid size per axis (small for feasible slow-test runtime).
const N: usize = 8;

/// Domain half-width per axis.
const L: f64 = 5.0;

/// Large horizon (§47.5 convention).
const T_SLOPE: f64 = 100.0;

/// Contour-node sweep (mirrors `G_RESOLVENT_JUMP_ORDER` `M_SWEEP`).
const M_SWEEP: [usize; 5] = [6, 8, 10, 12, 14];

/// High-M reference for self-convergence anchor.
const M_REF: usize = 40;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_grid() -> Grid3D<f64> {
    let g = Grid1D::new(-L, L, N).unwrap();
    Grid3D::new(g, g, g).unwrap()
}

fn run_jump(grid: Grid3D<f64>, m: usize, t: f64, g: &GridFn3D<f64>) -> GridFn3D<f64> {
    let rj = ResolventJumpChernoff3D::new(grid, m).unwrap();
    rj.jump(t, g).unwrap()
}

fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let lx: Vec<f64> = xs.iter().map(|&v| v.ln()).collect();
    let ly: Vec<f64> = ys.iter().map(|&v| v.ln()).collect();
    let mx = lx.iter().sum::<f64>() / m;
    let my = ly.iter().sum::<f64>() / m;
    let num: f64 = lx
        .iter()
        .zip(ly.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let den: f64 = lx.iter().map(|x| (x - mx).powi(2)).sum();
    num / den
}

fn sup_err(a: &GridFn3D<f64>, b: &GridFn3D<f64>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// G_RESOLVENT_JUMP_3D_ORDER gate
// ---------------------------------------------------------------------------

/// `G_RESOLVENT_JUMP_3D_ORDER` — slope ≥ +1.95 (`RELEASE_BLOCKING`, ADR-0148).
///
/// Self-convergence: reference at `M_ref=40`, probe at M ∈ {6,8,10,12,14}.
/// Mirrors `G_RESOLVENT_JUMP_ORDER` design (§47.5) but for the 3D banded LHP solve.
#[test]
#[ignore = "RELEASE_BLOCKING slow gate; run with: cargo run -p xtask -- test-flagship"]
fn g_resolvent_jump_3d_order() {
    let grid = make_grid();
    let g = GridFn3D::from_fn(grid, |x: f64, y: f64, z: f64| {
        (-x * x - y * y - z * z).exp()
    });

    // Self-convergence reference at M_ref=40.
    let ref_result = run_jump(grid, M_REF, T_SLOPE, &g);

    println!("G_RESOLVENT_JUMP_3D_ORDER (slope at t={T_SLOPE}, M_ref={M_REF}, N={N}³):");
    let mut errs: Vec<f64> = Vec::with_capacity(M_SWEEP.len());
    for &m in &M_SWEEP {
        let out = run_jump(grid, m, T_SLOPE, &g);
        let err = sup_err(&out, &ref_result);
        println!("  M={m:2}  err_inf = {err:.4e}");
        assert!(
            err.is_finite(),
            "G_RESOLVENT_JUMP_3D_ORDER: non-finite error at M={m}"
        );
        errs.push(err);
    }

    // G24 convention: xs = 1/M (shrinks as M grows), ys = errors.
    let inv_m: Vec<f64> = M_SWEEP.iter().map(|&m| 1.0 / m as f64).collect();
    let slope = ols_slope(&inv_m, &errs);
    println!("  OLS slope d log(err)/d log(1/M) = {slope:+.4}  (gate ≥ +{SLOPE_GATE})");
    assert!(
        slope >= SLOPE_GATE,
        "G_RESOLVENT_JUMP_3D_ORDER FAIL: slope {slope:+.4} < +{SLOPE_GATE}. \
         Errors by M: {errs:?}. \
         Pre-flight slope +9.79 (scripts/resolvent_jump_2d3d_kit.py A5). \
         Check banded LHP solve (resolvent_jump_nd.rs) + TWS contour constants.",
    );

    println!("G_RESOLVENT_JUMP_3D_ORDER PASS");
}
