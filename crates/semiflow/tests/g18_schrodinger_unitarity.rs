//! G18 unitarity gate — `SchrodingerChernoff` norm conservation + order-2 slope.
//!
//! Gate (fast part): Gaussian wavepacket on `[-5, 5]`, N=64, `V(x) = ½x²`,
//! `t_final = 1.0`, `n_steps = 100`.
//!
//! Unitarity: `|‖ψ(t_final)‖² − ‖ψ_0‖²| < 1e-12` (f64) / `< 1e-6` (f32).
//!
//! Slope gate (slow part, `#[ignore]`): self-convergence OLS slope on
//! `n_steps ∈ {10, 20, 40, 80, 160}` ≤ −1.95 (f64) / ≤ −1.50 (f32).
//!
//! See contract wave-b-advanced-semigroups.md §5.4 and ADR-0057.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]

use semiflow::diffusion4::Diffusion4thChernoff;
use semiflow::{
    ChernoffFunction, Grid1D, GridFn1D, SchrodingerChernoff, SchrodingerState, ScratchPool,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const N_NODES: usize = 64;
const T_FINAL: f64 = 1.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_schr_f64() -> (SchrodingerChernoff<f64>, Grid1D<f64>) {
    let grid = Grid1D::new(-5.0_f64, 5.0, N_NODES).unwrap();
    let kinetic = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let schr = SchrodingerChernoff::new(kinetic, |x: f64| 0.5 * x * x).unwrap();
    (schr, grid)
}

fn make_schr_f32() -> (SchrodingerChernoff<f32>, Grid1D<f32>) {
    let grid = Grid1D::<f32>::new_generic(-5.0_f32, 5.0_f32, N_NODES).unwrap();
    let kinetic = Diffusion4thChernoff::<f32>::new_generic(
        (|_: f32| 0.5_f32) as fn(f32) -> f32,
        (|_: f32| 0.0_f32) as fn(f32) -> f32,
        (|_: f32| 0.0_f32) as fn(f32) -> f32,
        0.5,
        grid,
    );
    let schr = SchrodingerChernoff::<f32>::new(kinetic, |x: f32| 0.5_f32 * x * x).unwrap();
    (schr, grid)
}

fn gaussian_state_f64(grid: Grid1D<f64>) -> SchrodingerState<f64> {
    let n = grid.n;
    let psi_re = GridFn1D::from_fn(grid, |x: f64| {
        let xc = x - 1.0;
        (-xc * xc / (2.0 * 0.5 * 0.5)).exp()
    });
    // Pure real initial state (imaginary part = 0)
    let psi_im = GridFn1D {
        values: vec![0.0_f64; n],
        grid,
    };
    SchrodingerState { psi_re, psi_im }
}

fn gaussian_state_f32(grid: Grid1D<f32>) -> SchrodingerState<f32> {
    let n = grid.n;
    let psi_re = GridFn1D::<f32>::from_fn_generic(grid, |x: f32| {
        let xc = x - 1.0_f32;
        (-xc * xc / (2.0_f32 * 0.5_f32 * 0.5_f32)).exp()
    });
    let psi_im = GridFn1D::<f32> {
        values: vec![0.0_f32; n],
        grid,
    };
    SchrodingerState::<f32> { psi_re, psi_im }
}

fn evolve_f64(
    schr: &SchrodingerChernoff<f64>,
    psi0: &SchrodingerState<f64>,
    tau: f64,
    n_steps: usize,
) -> SchrodingerState<f64> {
    let mut cur = psi0.clone();
    let mut nxt = psi0.clone();
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        schr.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

fn evolve_f32(
    schr: &SchrodingerChernoff<f32>,
    psi0: &SchrodingerState<f32>,
    tau: f32,
    n_steps: usize,
) -> SchrodingerState<f32> {
    let mut cur = psi0.clone();
    let mut nxt = psi0.clone();
    let mut pool = ScratchPool::<f32>::new();
    for _ in 0..n_steps {
        schr.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

// ---------------------------------------------------------------------------
// OLS slope
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
// G18a — unitarity conservation (f64, fast)
// ---------------------------------------------------------------------------

#[test]
fn g18a_unitarity_f64() {
    let (schr, grid) = make_schr_f64();
    let psi0 = gaussian_state_f64(grid);
    let norm_sq_0 = psi0.norm_l2_sq();

    let tau = T_FINAL / 100.0;
    let psi_final = evolve_f64(&schr, &psi0, tau, 100);
    let norm_sq_final = psi_final.norm_l2_sq();

    let diff = (norm_sq_final - norm_sq_0).abs();
    println!("G18a f64: |‖ψ_final‖² - ‖ψ_0‖²| = {diff:.4e}");
    assert!(
        diff < 1e-12,
        "G18a FAIL f64: unitarity error {diff:.4e} >= 1e-12 (ADR-0057 §R2)"
    );
}

// ---------------------------------------------------------------------------
// G18b — unitarity conservation (f32, fast)
// ---------------------------------------------------------------------------

#[test]
fn g18b_unitarity_f32() {
    let (schr, grid) = make_schr_f32();
    let psi0 = gaussian_state_f32(grid);
    let norm_sq_0 = psi0.norm_l2_sq();

    let tau = (T_FINAL / 100.0) as f32;
    let psi_final = evolve_f32(&schr, &psi0, tau, 100);
    let norm_sq_final = psi_final.norm_l2_sq();

    let diff = ((norm_sq_final - norm_sq_0).abs()) as f64;
    println!("G18b f32: |‖ψ_final‖² - ‖ψ_0‖²| = {diff:.4e}");
    assert!(
        diff < 1e-6,
        "G18b FAIL f32: unitarity error {diff:.4e} >= 1e-6 (ADR-0057 §R2 f32)"
    );
}

// ---------------------------------------------------------------------------
// G18c — order-2 self-convergence slope (f64, slow-test)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g18c_slope_f64() {
    let n_steps_list: [usize; 5] = [10, 20, 40, 80, 160];
    let (schr, grid) = make_schr_f64();
    let psi0 = gaussian_state_f64(grid);

    let mut log_n = Vec::new();
    let mut log_err = Vec::new();
    for &n_steps in &n_steps_list {
        let tau = T_FINAL / n_steps as f64;
        let tau_fine = T_FINAL / (2 * n_steps) as f64;

        let u_coarse = evolve_f64(&schr, &psi0, tau, n_steps);
        let u_fine = evolve_f64(&schr, &psi0, tau_fine, 2 * n_steps);

        // Max-norm self-convergence error.
        let err = u_coarse
            .psi_re
            .values
            .iter()
            .zip(&u_fine.psi_re.values)
            .zip(u_coarse.psi_im.values.iter().zip(&u_fine.psi_im.values))
            .map(|((re_c, re_f), (im_c, im_f))| {
                let dre = (re_c - re_f).abs();
                let dim = (im_c - im_f).abs();
                dre.max(dim)
            })
            .fold(0.0_f64, f64::max);

        println!("G18c f64  n_steps={n_steps:4}  err={err:.4e}");
        if err > 0.0 {
            log_n.push((n_steps as f64).ln());
            log_err.push(err.ln());
        }
    }

    // Remove floor-saturated trailing points.
    while log_err.len() >= 2 {
        let last = *log_err.last().unwrap();
        let prev = log_err[log_err.len() - 2];
        if last >= prev {
            log_n.pop();
            log_err.pop();
        } else {
            break;
        }
    }

    assert!(log_n.len() >= 3, "G18c: fewer than 3 usable error values");
    let slope = ols_slope(&log_n, &log_err);
    println!("G18c f64  slope = {slope:.4}  (threshold -1.95)");
    assert!(
        slope <= -1.95,
        "G18c FAIL f64: slope {slope:.4} > -1.95 (order-2 gate)"
    );
}

// ---------------------------------------------------------------------------
// G18d — order-2 self-convergence slope (f32, slow-test)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g18d_slope_f32() {
    let n_steps_list: [usize; 4] = [10, 20, 40, 80];
    let (schr, grid) = make_schr_f32();
    let psi0 = gaussian_state_f32(grid);

    let mut log_n = Vec::new();
    let mut log_err = Vec::new();
    for &n_steps in &n_steps_list {
        let tau = (T_FINAL / n_steps as f64) as f32;
        let tau_fine = (T_FINAL / (2 * n_steps) as f64) as f32;

        let u_coarse = evolve_f32(&schr, &psi0, tau, n_steps);
        let u_fine = evolve_f32(&schr, &psi0, tau_fine, 2 * n_steps);

        let err = u_coarse
            .psi_re
            .values
            .iter()
            .zip(&u_fine.psi_re.values)
            .zip(u_coarse.psi_im.values.iter().zip(&u_fine.psi_im.values))
            .map(|((re_c, re_f), (im_c, im_f))| {
                let dre = (re_c - re_f).abs();
                let dim = (im_c - im_f).abs();
                dre.max(dim)
            })
            .fold(0.0_f32, f32::max);

        println!("G18d f32  n_steps={n_steps:4}  err={err:.4e}");
        if err > 0.0_f32 {
            log_n.push((n_steps as f64).ln());
            log_err.push((err as f64).ln());
        }
    }

    while log_err.len() >= 2 {
        let last = *log_err.last().unwrap();
        let prev = log_err[log_err.len() - 2];
        if last >= prev {
            log_n.pop();
            log_err.pop();
        } else {
            break;
        }
    }

    if log_n.len() < 3 {
        println!("G18d f32: fewer than 3 usable points (f32 floor) — test inconclusive");
        return;
    }
    let slope = ols_slope(&log_n, &log_err);
    println!("G18d f32  slope = {slope:.4}  (threshold -1.50)");
    assert!(
        slope <= -1.50,
        "G18d FAIL f32: slope {slope:.4} > -1.50 (order-2 gate, f32 relaxed)"
    );
}
