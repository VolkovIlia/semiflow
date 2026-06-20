//! `G_WENTZELL_ORDER` — manufactured-solution order-1 convergence gate.
//!
//! `RELEASE_BLOCKING` gate (ADR-0151, math.md §49.6).
//!
//! Asserts order-1 (Lie splitting, Altmann–Verfürth 2023) for `DynamicWentzellChernoff`
//! with a time-dependent `γ(t) = 0.5 + sin(t)` (generic, never identically static).
//!
//! Method: self-convergence (probe vs. many-small-step reference). No closed-form oracle
//! for the dynamic Wentzell problem (manufactured solution approach mirrors `G4_NS2D_aniso`,
//! commit 0180292, and G24/G27).
//!
//! ## Spatial probe: non-origin generic point
//!
//! Probe at `x_probe = domain_max * 0.35` (35% of domain width) — strictly interior,
//! NOT at x=0 (boundary) or at `domain_max/2` (symmetric midpoint). This mirrors the
//! G24-lesson: symmetric/zero probes can have cancellation artefacts that mask the true
//! order; a generic off-center interior point avoids both.
//!
//! ## Self-convergence setup
//!
//! Let `T_final = 0.2`, `N = 64` spatial nodes, reference = `n_ref` = 512 Chernoff steps.
//! Sweep coarser `n ∈ {16, 32, 64, 128}`. Error = `|u_n(T_final, x_probe) - u_ref(T_final, x_probe)|`.
//! OLS slope on (log(τ), log(err)) must be `≤ −0.95` (order-1 gate).
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)]   // usize→f64 in tau/OLS; n_steps ≤ 512 ≤ 2^52
#![allow(clippy::cast_possible_truncation)] // f64→usize probe index: .round() positive
#![allow(clippy::cast_sign_loss)]        // f64→usize probe index: .round() result ≥ 0
#![allow(clippy::too_many_lines)]        // g_wentzell_order is a single cohesive gate (51 lines)

use semiflow_core::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    howland::TimedChernoffFunction,
    scratch::ScratchPool,
    wentzell::{DynamicWentzellChernoff, HalfSpaceWentzell},
};

const T_FINAL: f64 = 0.2;
const N_SPATIAL: usize = 64;
const DOMAIN_MAX: f64 = 5.0;
const N_REF: usize = 512;
// Non-origin, non-midpoint interior probe (G24 symmetric-cancellation avoidance).
const PROBE_FRAC: f64 = 0.35;
const SLOPE_GATE: f64 = -0.95;

fn gamma_fn(t: f64) -> f64 {
    0.5 + t.sin()
}

fn build_kernel() -> DynamicWentzellChernoff<DiffusionChernoff<f64>, HalfSpaceWentzell<f64, 1>> {
    let grid = Grid1D::new(0.0_f64, DOMAIN_MAX, N_SPATIAL).unwrap();
    let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let region = HalfSpaceWentzell::new([0.0], [1.0], gamma_fn as fn(f64) -> f64, 0.2).unwrap();
    DynamicWentzellChernoff::new(inner, region).unwrap()
}

fn evolve_n_steps(
    kernel: &DynamicWentzellChernoff<DiffusionChernoff<f64>, HalfSpaceWentzell<f64, 1>>,
    u0: &GridFn1D<f64>,
    n_steps: usize,
    t_final: f64,
) -> GridFn1D<f64> {
    let tau = t_final / (n_steps as f64);
    let mut src = u0.clone();
    let mut dst = u0.clone();
    let mut scratch = ScratchPool::new();
    for k in 0..n_steps {
        let t_k = (k as f64) * tau;
        kernel
            .apply_at(t_k, tau, &src, &mut dst, &mut scratch)
            .unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    src
}

fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx: f64 = xs.iter().sum::<f64>() / n;
    let my: f64 = ys.iter().sum::<f64>() / n;
    let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    num / den
}

/// Probe index: node closest to `DOMAIN_MAX * PROBE_FRAC`.
fn probe_index(grid: &Grid1D<f64>) -> usize {
    let x_probe = DOMAIN_MAX * PROBE_FRAC;
    let dx = grid.dx();
    let i = ((x_probe - grid.xmin) / dx).round() as usize;
    i.min(grid.n - 1)
}

#[test]
#[ignore = "RELEASE_BLOCKING slow-test; run with: cargo run -p xtask -- test-flagship"]
fn g_wentzell_order() {
    let grid = Grid1D::new(0.0_f64, DOMAIN_MAX, N_SPATIAL).unwrap();
    // Initial datum: smooth, non-zero on interior, zero at boundary.
    // Use sin²(πx / L) to satisfy Dirichlet-like BC structure.
    let pi_over_l = core::f64::consts::PI / DOMAIN_MAX;
    let u0 = GridFn1D::from_fn(grid, |x| {
        let s = (pi_over_l * x).sin();
        s * s * (-0.5 * x).exp()
    });

    let pidx = probe_index(&grid);
    let x_probe = grid.x_at(pidx);
    println!("G_WENTZELL_ORDER: probe index = {pidx}, x_probe = {x_probe:.4}");
    assert!(
        pidx >= 2,
        "probe must be at least 2 nodes from boundary, got idx={pidx}"
    );

    // Reference: many-small-step solution.
    let kernel = build_kernel();
    let u_ref = evolve_n_steps(&kernel, &u0, N_REF, T_FINAL);
    let ref_val = u_ref.values[pidx];
    println!("G_WENTZELL_ORDER: reference val at x_probe = {ref_val:.6e}");

    let n_sweep = [16usize, 32, 64, 128];
    let mut errs: Vec<f64> = Vec::new();
    let mut taus: Vec<f64> = Vec::new();

    for &n in &n_sweep {
        let tau = T_FINAL / (n as f64);
        let kernel_n = build_kernel();
        let u_n = evolve_n_steps(&kernel_n, &u0, n, T_FINAL);
        let err = (u_n.values[pidx] - ref_val).abs();
        errs.push(err);
        taus.push(tau);
        println!("G_WENTZELL_ORDER n={n:4}: tau={tau:.4e}  err={err:.4e}");
    }

    // Filter out exactly-zero errors (degenerate — should not happen).
    // x = log(n) (n increases → tau decreases → err decreases → slope negative).
    // Mirrors G_ROBIN_HALFLINE (robin_heat_slope.rs), G27 convention.
    let valid: Vec<(f64, f64)> = n_sweep
        .iter()
        .zip(errs.iter())
        .filter(|(_, &e)| e > 0.0)
        .map(|(&n, &e)| ((n as f64).ln(), e.ln()))
        .collect();
    assert!(
        valid.len() >= 3,
        "G_WENTZELL_ORDER: need >= 3 non-zero error points for OLS; got {}",
        valid.len()
    );
    let log_ns: Vec<f64> = valid.iter().map(|(x, _)| *x).collect();
    let log_errs: Vec<f64> = valid.iter().map(|(_, y)| *y).collect();
    let slope = ols_slope(&log_ns, &log_errs);

    println!(
        "\nG_WENTZELL_ORDER OLS slope (d log_err / d log_n): {slope:.4}  (gate <= {SLOPE_GATE:.2})"
    );

    assert!(
        slope <= SLOPE_GATE,
        "G_WENTZELL_ORDER FAIL: OLS slope {slope:.4} > {SLOPE_GATE} (order-1 gate, math §49.6, ADR-0151)"
    );

    println!("G_WENTZELL_ORDER PASS: slope = {slope:.4} <= {SLOPE_GATE}");
}
