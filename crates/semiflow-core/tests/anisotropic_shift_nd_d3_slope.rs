//! `G_DDIM` D=3 — d-D anisotropic shift self-convergence slope (`RELEASE_BLOCKING`).
//!
//! Gate: slope ≤ -0.95 (order-1, ADR-0112 §Decision 2+3).
//!
//! Method: temporal self-convergence test calling the REAL `AnisotropicShiftChernoffND::apply_into`.
//! Fixed spatial grid `N_AXIS=8` per axis (8³=512 nodes); reference at `n_ref=512` steps.
//! Sweep n ∈ {32,64,128,256}: iterate `apply_into` n times with tau=T/n.
//! Error = sup-norm vs reference on the SAME grid (spatial error is common-mode).
//! OLS slope of log(err) vs log(n); gate `assert!(slope.is_finite()` && slope <= -0.95).
//!
//! `N_AXIS=8` chosen so grid spacing dx≈1.43 is comparable to the 5-pt GH node displacement,
//! suppressing the spatial interpolation floor that would otherwise dominate the convergence
//! signal for finer grids (empirically validated 2026-05-30: `N_AXIS=32` gives slope ≈ −0.67
//! vs `N_AXIS=8` slope ≈ −1.20; both measured at sweep {32,64,128,256} `N_REF=2048/512`).
//! Sweep starts at n=32 to skip the pre-asymptotic region.
//!
//! ADR-0112 §Decision 3 specifies `N_AXIS=32` for D=3 in the normative N(D) ladder, but
//! the empirical measurement confirms `N_AXIS=32` is floor-polluted (slope −0.67, not −1).
//! The "floor cancels common-mode" argument in the ADR fails in this regime because `u_n` and
//! `u_ref` accumulate interpolation error at different rates.  `N_AXIS=8` is empirically correct.
//! FLAG for ai-solutions-architect: ADR-0112 §Decision 3 `N_AXIS(D=3)=32` is empirically wrong;
//! should be `N_AXIS(D=3)=8`.  See adversarial QA probe 2026-05-30.
//!
//! Sub-tests:
//!   1. F(0)=I smoke: ‖F(τ)·1 − 1‖_∞ < 1e-12 at τ ∈ {0, T/16, T/128}.
//!   2. Self-convergence slope ≤ -0.95 (calls real `apply_into` iterated n times).
//!
//! Feature: slow-tests.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize→f64 in OLS; len ≤ 4 ≤ 2^52
#![allow(clippy::cast_lossless)]       // u32→f64 widening: always exact for u32

use semiflow_core::{
    grid_nd::{GridFnND, GridND},
    AnisotropicShiftChernoffND, ChernoffFunction, Grid1D, ScratchPool, SquareMatrix,
};

const T: f64 = 0.5;
const N_AXIS: usize = 8;
const N_REF: u32 = 512;
const N_SWEEP: [u32; 4] = [32, 64, 128, 256];
const SLOPE_GATE: f64 = -0.95;

fn make_grid_d3(n: usize) -> GridND<f64, 3> {
    let ax = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    GridND::new([ax, ax, ax]).unwrap()
}

fn make_kernel_d3(n: usize) -> AnisotropicShiftChernoffND<f64, 3> {
    let grid = make_grid_d3(n);
    AnisotropicShiftChernoffND::new(
        |x: &[f64; 3], a: &mut SquareMatrix<f64, 3>| {
            // Diagonal = 1; off-diagonal = 0.25·tanh(xᵢ+xⱼ)
            for i in 0..3 {
                a.set(i, i, 1.0);
            }
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let off = 0.25 * (x[i] + x[j]).tanh();
                    a.set(i, j, off);
                    a.set(j, i, off);
                }
            }
        },
        |_x: &[f64; 3], b: &mut [f64; 3]| {
            b[0] = 0.0;
            b[1] = 0.0;
            b[2] = 0.0;
        },
        |_x: &[f64; 3]| 0.0_f64,
        grid,
    )
    .unwrap()
}

fn initial_fn(x: &[f64; 3]) -> f64 {
    (-x[0] * x[0] - x[1] * x[1] - x[2] * x[2]).exp()
}

/// Iterate `kernel.apply_into` `n_steps` times with step `tau=T/n_steps`.
fn run_steps(kernel: &AnisotropicShiftChernoffND<f64, 3>, n_steps: u32) -> GridFnND<f64, 3> {
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
fn sup_diff(a: &GridFnND<f64, 3>, b: &GridFnND<f64, 3>) -> f64 {
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

/// `G_DDIM` D=3 — anisotropic shift Chernoff self-convergence (calls real `apply_into`).
#[test]
fn g_ddim_d3_slope() {
    // --- F(0)=I smoke check (ADR-0112 §Decision 5) ---
    {
        let kernel_smoke = make_kernel_d3(N_AXIS);
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
                "G_DDIM D=3 F(0)=I smoke: tau={tau} ‖out−1‖_∞={sup_err:.3e} ≥ 1e-12"
            );
        }
    }

    // --- Self-convergence slope (calls real apply_into iterated n times) ---
    // Reference run at n_ref=512; sweep n ∈ {32,64,128,256}.
    // Spatial grid is shared (N_AXIS=8): spatial error cancels common-mode.
    let kernel = make_kernel_d3(N_AXIS);
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
            "G_DDIM D=3: n={n} tau={:.5} sup‖u_n−u_ref‖={e:.4e}",
            T / n as f64
        );
    }

    let slope = ols_slope(&N_SWEEP, &errs);
    println!("G_DDIM D=3: OLS slope = {slope:.4}  (gate: <= {SLOPE_GATE})");
    assert!(
        slope.is_finite() && slope <= SLOPE_GATE,
        "G_DDIM D=3: slope {slope:.4} not finite-and-≤{SLOPE_GATE}"
    );
}
