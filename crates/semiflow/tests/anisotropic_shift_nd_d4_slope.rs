//! `G_DDIM` D=4 — d-D anisotropic shift self-convergence slope (`RELEASE_BLOCKING`).
//!
//! Gate: slope ≤ -0.95 (order-1, ADR-0112 §Decision 2+3).
//!
//! Method: temporal self-convergence test calling the REAL `AnisotropicShiftChernoffND::apply_into`.
//! Fixed spatial grid `N_AXIS=8` per axis (8⁴=4096 nodes); reference at `n_ref=512` steps.
//! Sweep n ∈ {16,32,64,128}: iterate `apply_into` n times with tau=T/n.
//! Error = sup-norm vs reference on the SAME grid (spatial error cancels common-mode).
//! OLS slope of log(err) vs log(n); gate `assert!(slope.is_finite()` && slope <= -0.95).
//!
//! Sub-tests:
//!   1. F(0)=I smoke: ‖F(τ)·1 − 1‖_∞ < 1e-12 at τ ∈ {0, T/16, T/128}.
//!   2. Self-convergence slope ≤ -0.95.
//!
//! Feature: slow-tests.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize→f64 in OLS; values ≤ 512 ≤ 2^52
#![allow(clippy::cast_lossless)] // u32→f64 for n_steps: infallible, project idiom

use semiflow::{
    grid_nd::{GridFnND, GridND},
    AnisotropicShiftChernoffND, ChernoffFunction, Grid1D, ScratchPool, SquareMatrix,
};

const T: f64 = 0.5;
const N_AXIS: usize = 8;
const N_REF: u32 = 512;
const N_SWEEP: [u32; 4] = [16, 32, 64, 128];
const SLOPE_GATE: f64 = -0.95;

fn make_grid_d4(n: usize) -> GridND<f64, 4> {
    let ax = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    GridND::new([ax; 4]).unwrap()
}

fn make_kernel_d4(n: usize) -> AnisotropicShiftChernoffND<f64, 4> {
    let grid = make_grid_d4(n);
    AnisotropicShiftChernoffND::new(
        |x: &[f64; 4], a: &mut SquareMatrix<f64, 4>| {
            for i in 0..4 {
                a.set(i, i, 1.0);
            }
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let off = 0.25 * (x[i] + x[j]).tanh();
                    a.set(i, j, off);
                    a.set(j, i, off);
                }
            }
        },
        |_x: &[f64; 4], b: &mut [f64; 4]| {
            for v in b.iter_mut() {
                *v = 0.0;
            }
        },
        |_x: &[f64; 4]| 0.0_f64,
        grid,
    )
    .unwrap()
}

fn initial_fn(x: &[f64; 4]) -> f64 {
    (-x.iter().map(|xi| xi * xi).sum::<f64>()).exp()
}

/// Iterate `kernel.apply_into` `n_steps` times with step `tau=T/n_steps`.
fn run_steps(kernel: &AnisotropicShiftChernoffND<f64, 4>, n_steps: u32) -> GridFnND<f64, 4> {
    let tau = T / n_steps as f64;
    let f0 = GridFnND::from_fn(kernel.grid().clone(), initial_fn);
    let mut src = f0;
    let mut dst = GridFnND::from_fn(kernel.grid().clone(), |_| 0.0_f64);
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut pool).unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    src
}

/// NaN-propagating sup-norm of (a - b).
fn sup_diff(a: &GridFnND<f64, 4>, b: &GridFnND<f64, 4>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(&ai, &bi)| (ai - bi).abs())
        .fold(0.0_f64, |m, e| if e.is_nan() { f64::NAN } else { m.max(e) })
}

fn ols_slope(ns: &[u32], errs: &[f64]) -> f64 {
    let x: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let y: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sxx: f64 = x.iter().map(|xi| xi * xi).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

/// `G_DDIM` D=4 — anisotropic shift Chernoff self-convergence (calls real `apply_into`).
#[test]
fn g_ddim_d4_slope() {
    // --- F(0)=I smoke check (ADR-0112 §Decision 5) ---
    {
        let kernel_smoke = make_kernel_d4(N_AXIS);
        let one_fn = GridFnND::from_fn(kernel_smoke.grid().clone(), |_| 1.0_f64);
        let mut pool = ScratchPool::<f64>::new();
        let mut out = one_fn.clone();
        for &tau in &[0.0_f64, T / 16.0, T / 128.0] {
            kernel_smoke
                .apply_into(tau, &one_fn, &mut out, &mut pool)
                .unwrap();
            let sup_err = out
                .values
                .iter()
                .map(|&v| (v - 1.0_f64).abs())
                .fold(0.0_f64, |m, e| if e.is_nan() { f64::NAN } else { m.max(e) });
            assert!(
                sup_err < 1e-12,
                "G_DDIM D=4 F(0)=I smoke: tau={tau} ‖out−1‖_∞={sup_err:.3e} ≥ 1e-12"
            );
        }
    }

    // --- Self-convergence slope (calls real apply_into) ---
    // Reference run at n_ref=512; sweep n ∈ {16,32,64,128}.
    // Spatial grid is shared (N_AXIS=8): spatial error cancels common-mode.
    let kernel = make_kernel_d4(N_AXIS);
    let u_ref = run_steps(&kernel, N_REF);

    let errs: Vec<f64> = N_SWEEP
        .iter()
        .map(|&n| {
            let u_n = run_steps(&kernel, n);
            sup_diff(&u_n, &u_ref)
        })
        .collect();

    for (&n, &e) in N_SWEEP.iter().zip(errs.iter()) {
        println!(
            "G_DDIM D=4: n={n} tau={:.5} sup‖u_n−u_ref‖={e:.4e}",
            T / n as f64
        );
    }

    let slope = ols_slope(&N_SWEEP, &errs);
    println!("G_DDIM D=4: OLS slope = {slope:.4}  (gate: <= {SLOPE_GATE})");
    assert!(
        slope.is_finite() && slope <= SLOPE_GATE,
        "G_DDIM D=4: slope {slope:.4} not finite-and-≤{SLOPE_GATE}"
    );
}
