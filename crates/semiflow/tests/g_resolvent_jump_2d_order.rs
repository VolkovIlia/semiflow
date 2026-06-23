//! `G_RESOLVENT_JUMP_2D_ORDER` — 2D parabolic resolvent time-jump convergence gate.
//!
//! Gate spec (`RELEASE_BLOCKING`, slow-tests, ADR-0148, §47.8):
//!
//!   OLS slope d `log‖jump_M` − ref‖_∞ / d log(1/M) ≥ +1.95
//!   M ∈ {6, 8, 10, 12, 14}, t = 100, Nx = Ny = 16, Gaussian IC, Neumann BC.
//!
//!   G24 convention (SAME AS `G_RESOLVENT_JUMP_ORDER` — copy verbatim, §47.5):
//!   positive slope = convergence as M grows; ≥ +1.95 PASSES.
//!   Pre-flight (`scripts/resolvent_jump_2d3d_kit.py` A3): slope +9.97.
//!
//! Self-convergence reference: `ResolventJumpChernoff2D` at `M_ref` = 40 targets the
//! same discrete 2D Neumann Laplacian. Both use the banded LHP solve, ensuring no
//! continuous/discrete mismatch (mirrors the 1D self-convergence design, §47.5).
//!
//! ## Operator
//!
//! A = ∂²ₓ + ∂²ᵧ on [−5,5]², Nx=Ny=16, Neumann BC (3-pt stencil per axis).
//! Row-major: `idx(i,j) = j*nx + i`. Kronecker sum (§47.8, grid2d.rs I-T1).
//!
//! ## Sign convention (NORMATIVE — do NOT alter)
//!
//! Slope is computed as `d log(err) / d log(1/M)` with `x_i = 1/M_i`:
//! as M grows, `1/M` shrinks, errors shrink → slope positive → ≥ +1.95 PASSES.
//! This is the **G24 convention** from ADR-0134 §47.5 — identical here.

#![cfg(feature = "slow-tests")]

use semiflow_core::{Grid1D, Grid2D, GridFn2D, ResolventJumpChernoff2D};

// ---------------------------------------------------------------------------
// Gate constants — do NOT relax without ADR + properties.yaml bump.
// ---------------------------------------------------------------------------

/// OLS slope gate: d log(err) / d log(1/M) ≥ +1.95 (G24 convention).
const SLOPE_GATE: f64 = 1.95;

/// Spatial grid sizes per axis.
const NX: usize = 16;
const NY: usize = 16;

/// Domain half-width per axis.
const L: f64 = 5.0;

/// Large horizon (§47.5 convention).
const T_SLOPE: f64 = 100.0;

/// Contour-node sweep (mirrors G_RESOLVENT_JUMP_ORDER M_SWEEP).
const M_SWEEP: [usize; 5] = [6, 8, 10, 12, 14];

/// High-M reference for self-convergence anchor.
const M_REF: usize = 40;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_grid() -> Grid2D<f64> {
    let gx = Grid1D::new(-L, L, NX).unwrap();
    let gy = Grid1D::new(-L, L, NY).unwrap();
    Grid2D::new(gx, gy)
}

fn run_jump(grid: Grid2D<f64>, m: usize, t: f64, g: &GridFn2D<f64>) -> GridFn2D<f64> {
    let rj = ResolventJumpChernoff2D::new(grid, m).unwrap();
    rj.jump(t, g).unwrap()
}

#[allow(clippy::cast_precision_loss)]
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

fn sup_err(a: &GridFn2D<f64>, b: &GridFn2D<f64>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// G_RESOLVENT_JUMP_2D_ORDER gate
// ---------------------------------------------------------------------------

/// G_RESOLVENT_JUMP_2D_ORDER — slope ≥ +1.95 (RELEASE_BLOCKING, ADR-0148).
///
/// Self-convergence: reference at M_ref=40, probe at M ∈ {6,8,10,12,14}.
/// Mirrors G_RESOLVENT_JUMP_ORDER design (§47.5) but for the 2D banded LHP solve.
#[test]
#[ignore]
fn g_resolvent_jump_2d_order() {
    let grid = make_grid();
    let g = GridFn2D::from_fn(grid, |x: f64, y: f64| (-x * x - y * y).exp());

    // Self-convergence reference at M_ref=40.
    let ref_result = run_jump(grid, M_REF, T_SLOPE, &g);

    println!("G_RESOLVENT_JUMP_2D_ORDER (slope at t={T_SLOPE}, M_ref={M_REF}, Nx={NX}×Ny={NY}):");
    let mut errs: Vec<f64> = Vec::with_capacity(M_SWEEP.len());
    for &m in &M_SWEEP {
        let out = run_jump(grid, m, T_SLOPE, &g);
        let err = sup_err(&out, &ref_result);
        println!("  M={m:2}  err_inf = {err:.4e}");
        assert!(
            err.is_finite(),
            "G_RESOLVENT_JUMP_2D_ORDER: non-finite error at M={m}"
        );
        errs.push(err);
    }

    // G24 convention: xs = 1/M (shrinks as M grows), ys = errors.
    let inv_m: Vec<f64> = M_SWEEP.iter().map(|&m| 1.0 / m as f64).collect();
    let slope = ols_slope(&inv_m, &errs);
    println!("  OLS slope d log(err)/d log(1/M) = {slope:+.4}  (gate ≥ +{SLOPE_GATE})");
    assert!(
        slope >= SLOPE_GATE,
        "G_RESOLVENT_JUMP_2D_ORDER FAIL: slope {slope:+.4} < +{SLOPE_GATE}. \
         Errors by M: {errs:?}. \
         Pre-flight slope +9.97 (scripts/resolvent_jump_2d3d_kit.py A3). \
         Check banded LHP solve (resolvent_jump_nd.rs) + TWS contour constants.",
    );

    println!("G_RESOLVENT_JUMP_2D_ORDER PASS");
}
