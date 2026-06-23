//! `G_MATRIX_2D` + `G_MATRIX_3D` — self-convergence slope gate for
//! `MatrixDiffusionChernoff2D/3D` (ADR-0124).
//!
//! Gate: slope ≤ −0.80 for both 2D and 3D (documented lower slope — matrix
//! reaction half-step per ADR-0124 §Consequences).
//!
//! Test design (self-convergence, no oracle needed):
//!   For each probe `n_steps`, compute ‖u(n) − u(2n)‖_∞.  If the method is
//!   order-p, these differences converge as τ^p, so the OLS slope of
//!   (ln n, ln err) → −p.  We require slope ≤ −0.80.
//!
//! Non-commuting reaction datum: `Cx` is skew, `Cy` is symmetric,
//! `[Cx, Cy] ≠ 0` — this exercises the BCH-cancellation verified in the
//! PRE-FLIGHT (`scripts/verify_matrix_2d3d_preflight.py`).

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_lossless)]       // u32→f64 widening: always exact for u32
#![allow(clippy::cast_precision_loss)] // usize→f64 in OLS; len ≤ 4 ≤ 2^52
#![allow(clippy::similar_names)]       // u_n/u_2n are standard PDE convergence names

use semiflow_core::{
    ChernoffFunction, Grid1D, Grid2D, Grid3D, MatrixDiffusionChernoff, MatrixDiffusionChernoff2D,
    MatrixDiffusionChernoff3D, MatrixGridFn2D, MatrixGridFn3D, ScratchPool,
};

const M: usize = 2;

// ---------------------------------------------------------------------------
// Coefficient matrices — non-commuting Cx/Cy to exercise BCH cancellation.
// ---------------------------------------------------------------------------
//
// Cx: skew-symmetric (as in the PRE-FLIGHT)
// Cy: symmetric
// [Cx, Cy] = Cx*Cy - Cy*Cx ≠ 0  (verified in preflight script, C1b).

const A_X: [[f64; M]; M] = [[0.4, 0.0], [0.0, 0.3]];
const A_Y: [[f64; M]; M] = [[0.35, 0.0], [0.0, 0.45]];
const A_Z: [[f64; M]; M] = [[0.3, 0.0], [0.0, 0.4]];

const CX: [[f64; M]; M] = [[0.0, 0.2], [-0.2, 0.0]]; // skew
const CY: [[f64; M]; M] = [[0.1, 0.15], [0.15, 0.1]]; // symmetric; [CX,CY]≠0

fn make_kx(n: usize) -> MatrixDiffusionChernoff<f64, M> {
    let g = Grid1D::new(-4.0, 4.0, n).unwrap();
    MatrixDiffusionChernoff::<f64, M>::new(|_, a| *a = A_X, |_, _b| {}, |_, c| *c = CX, g).unwrap()
}

fn make_ky(n: usize) -> MatrixDiffusionChernoff<f64, M> {
    let g = Grid1D::new(-4.0, 4.0, n).unwrap();
    MatrixDiffusionChernoff::<f64, M>::new(|_, a| *a = A_Y, |_, _b| {}, |_, c| *c = CY, g).unwrap()
}

fn make_kz(n: usize) -> MatrixDiffusionChernoff<f64, M> {
    let g = Grid1D::new(-4.0, 4.0, n).unwrap();
    MatrixDiffusionChernoff::<f64, M>::new(|_, a| *a = A_Z, |_, _b| {}, |_, c| *c = CX, g).unwrap()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ols_slope(ns: &[u32], errs: &[f64]) -> f64 {
    let x: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let y: Vec<f64> = errs.iter().map(|&e| e.max(1e-300).ln()).collect();
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sxx: f64 = x.iter().map(|xi| xi * xi).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

// ---------------------------------------------------------------------------
// G_MATRIX_2D
// ---------------------------------------------------------------------------

fn run_2d(n_steps: u32) -> MatrixGridFn2D<f64, M> {
    const NX: usize = 20;
    const NY: usize = 16;
    const T: f64 = 0.3;
    let gx = Grid1D::new(-4.0, 4.0, NX).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, NY).unwrap();
    let grid2 = Grid2D::new(gx, gy);
    let op = MatrixDiffusionChernoff2D::new(make_kx(NX), make_ky(NY));
    let tau = T / n_steps as f64;
    let f0 = MatrixGridFn2D::<f64, M>::from_fn(grid2, |x, y| {
        [(-x * x - 0.5 * y * y).exp(), (-0.5 * x * x - y * y).exp()]
    });
    let mut pool = ScratchPool::<f64>::new();
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for _ in 0..n_steps {
        op.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

fn sup_diff_2d(a: &MatrixGridFn2D<f64, M>, b: &MatrixGridFn2D<f64, M>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
#[ignore = "slow flagship gate; run with: cargo run -p xtask -- test-flagship"]
fn g_matrix_2d() {
    let ns = [32_u32, 64, 128, 256];
    let mut errs = Vec::with_capacity(ns.len());
    for &n in &ns {
        let u_n = run_2d(n);
        let u_2n = run_2d(n * 2);
        let err = sup_diff_2d(&u_n, &u_2n);
        println!("G_MATRIX_2D: n={n} self-diff={err:.4e}");
        errs.push(err);
    }
    let slope = ols_slope(&ns, &errs);
    println!("G_MATRIX_2D: OLS slope = {slope:.4}");
    assert!(
        slope <= -0.80,
        "G_MATRIX_2D: slope {slope:.4} > -0.80 (gate FAILED)"
    );
}

// ---------------------------------------------------------------------------
// G_MATRIX_3D
// ---------------------------------------------------------------------------

fn run_3d(n_steps: u32) -> MatrixGridFn3D<f64, M> {
    const NX: usize = 8;
    const NY: usize = 8;
    const NZ: usize = 8;
    const T: f64 = 0.2;
    let gx = Grid1D::new(-4.0, 4.0, NX).unwrap();
    let gy = Grid1D::new(-4.0, 4.0, NY).unwrap();
    let gz = Grid1D::new(-4.0, 4.0, NZ).unwrap();
    let grid3 = Grid3D::new(gx, gy, gz).unwrap();
    let op = MatrixDiffusionChernoff3D::new(make_kx(NX), make_ky(NY), make_kz(NZ));
    let tau = T / n_steps as f64;
    let f0 = MatrixGridFn3D::<f64, M>::from_fn(grid3, |x, y, z| {
        [
            (-(x * x + y * y + z * z)).exp(),
            (-0.5 * (x * x + y * y + z * z)).exp(),
        ]
    });
    let mut pool = ScratchPool::<f64>::new();
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for _ in 0..n_steps {
        op.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

fn sup_diff_3d(a: &MatrixGridFn3D<f64, M>, b: &MatrixGridFn3D<f64, M>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
#[ignore = "slow flagship gate; run with: cargo run -p xtask -- test-flagship"]
fn g_matrix_3d() {
    let ns = [16_u32, 32, 64, 128];
    let mut errs = Vec::with_capacity(ns.len());
    for &n in &ns {
        let u_n = run_3d(n);
        let u_2n = run_3d(n * 2);
        let err = sup_diff_3d(&u_n, &u_2n);
        println!("G_MATRIX_3D: n={n} self-diff={err:.4e}");
        errs.push(err);
    }
    let slope = ols_slope(&ns, &errs);
    println!("G_MATRIX_3D: OLS slope = {slope:.4}");
    assert!(
        slope <= -0.80,
        "G_MATRIX_3D: slope {slope:.4} > -0.80 (gate FAILED)"
    );
}
