//! `G_ROBIN_HALFLINE` + `G_ROBIN_SELF` — ADR-0098 Robin BC acceptance gates.
//!
//! `G_ROBIN_HALFLINE`: 1D half-line `[0, 10]` slope vs Carslaw-Jaeger 1959 §14.2 eq 5
//!   closed-form oracle (full 3-term kernel with erfc-correction; factor (α/β),
//!   NOT 2·(α/β) — see ADR-0098 Amendment 1).
//! `G_ROBIN_SELF`: 2D self-convergence on box `[0, 1]²` (no closed-form oracle for
//!   general convex 2D Robin BC — mirror v2.2 `G_NS2D_aniso` pattern).
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize→f64 in OLS and sweep; n ≤ 1024 ≤ 2^52

use semiflow::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    robin::{HalfSpaceRobin, RobinHeatChernoff},
    scratch::ScratchPool,
    ChernoffFunction,
};

const T_FINAL: f64 = 0.1;
const ALPHA: f64 = 1.0;
const BETA: f64 = 1.0;
const SLOPE_GATE: f64 = -0.95; // Order-1 gate (math §3.5.tris.5)

/// Carslaw-Jaeger 1959 §14.2 eq 5 — exact 3-term Robin heat kernel.
///
/// K^Robin(x, y; t) = K(x,y,t) + K(x,-y,t)
///                   - (α/β)·exp((α/β)(x+y) + (α/β)²t)·erfc((x+y)/(2√t) + (α/β)√t)
///
/// NOTE: the correction factor is (α/β), not 2·(α/β) — ADR-0098 Amendment 1.
fn cj_robin_kernel(x: f64, y: f64, t: f64, alpha: f64, beta: f64) -> f64 {
    let four_pi_t_inv_sqrt = 1.0 / libm::sqrt(4.0 * core::f64::consts::PI * t);
    let k_direct = four_pi_t_inv_sqrt * libm::exp(-(x - y).powi(2) / (4.0 * t));
    let k_image = four_pi_t_inv_sqrt * libm::exp(-(x + y).powi(2) / (4.0 * t));
    let ratio = alpha / beta;
    let arg = (x + y) / (2.0 * libm::sqrt(t)) + ratio * libm::sqrt(t);
    // Correct factor: ratio (not 2*ratio)
    let k_corr = ratio * libm::exp(ratio * (x + y) + ratio * ratio * t) * libm::erfc(arg);
    k_direct + k_image - k_corr
}

/// Oracle: u(T, x) = ∫_0^∞ K^Robin(x, y; T) · g(y) dy via 1024-pt composite Simpson.
fn oracle(x: f64, t_final: f64, alpha: f64, beta: f64) -> f64 {
    let n = 1024usize;
    let y_max = 10.0;
    let dy = y_max / (n as f64);
    let mut sum = 0.0_f64;
    for i in 0..=n {
        let y = (i as f64) * dy;
        let g_y = libm::exp(-y * y);
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * cj_robin_kernel(x, y, t_final, alpha, beta) * g_y;
    }
    sum * dy / 3.0
}

fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx: f64 = xs.iter().sum::<f64>() / n;
    let my: f64 = ys.iter().sum::<f64>() / n;
    let num: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    num / den
}

#[test]
#[ignore = "RELEASE_BLOCKING slow Robin convergence gate; run with: cargo run -p xtask -- test-flagship"]
fn g_robin_halfline_slope() {
    let domain_max = 10.0;
    // n_grid=512 is mandatory: the skew image BC introduces an O(dx) spatial error
    // (~α/β correction per ghost node) that is INDEPENDENT of τ and sets a floor
    // on the sup-norm convergence. With n_grid=64 the spatial floor (~6e-3) swamps
    // the temporal Chernoff error for the n ∈ {16,32,64,128} sweep, giving OLS
    // slope ≈ -0.26 instead of the true order-1 slope ≈ -1. With n_grid=512 the
    // spatial floor (~5e-4) stays well below the coarsest-τ temporal error (~1e-2),
    // so the OLS slope cleanly recovers the expected order-1 rate. The α=0
    // (Neumann) case has no such floor (even-reflection ghost = exact) and therefore
    // does not require a fine grid (G27 uses n_grid=64).
    let n_grid = 512usize;
    let grid = Grid1D::new(0.0_f64, domain_max, n_grid).unwrap();
    let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let region = HalfSpaceRobin::<f64, 1>::new([0.0], [1.0], ALPHA, BETA).unwrap();
    let wrapper = RobinHeatChernoff::new(inner, region).unwrap();
    let u0 = GridFn1D::from_fn(grid, |x| libm::exp(-x * x));

    let n_sweep = [16usize, 32, 64, 128];
    let mut errs: Vec<f64> = Vec::new();
    let mut scratch = ScratchPool::new();
    for &n in &n_sweep {
        let tau = T_FINAL / (n as f64);
        let mut src = u0.clone();
        let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
        for _ in 0..n {
            wrapper
                .apply_into(tau, &src, &mut dst, &mut scratch)
                .unwrap();
            core::mem::swap(&mut src, &mut dst);
        }
        let mut err_max = 0.0_f64;
        for i in 0..n_grid {
            let x_i = grid.x_at(i);
            let oracle_i = oracle(x_i, T_FINAL, ALPHA, BETA);
            let err_i = (src.values[i] - oracle_i).abs();
            if err_i > err_max {
                err_max = err_i;
            }
        }
        errs.push(err_max);
        println!("G_ROBIN_HALFLINE n={n:3}: err={err_max:.4e} tau={tau:.4e}");
    }
    let xs: Vec<f64> = n_sweep.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let slope = ols_slope(&xs, &ys);
    println!("\nG_ROBIN_HALFLINE OLS slope: {slope:.4}  (gate ≤ {SLOPE_GATE:.2})");
    assert!(
        slope <= SLOPE_GATE,
        "G_ROBIN_HALFLINE FAIL: OLS slope {slope:.4} > {SLOPE_GATE} (order-1 gate, math §3.5.tris.5)"
    );
}

#[test]
#[ignore = "v6.2.3: 2D Strang per-axis Robin composition deferred to v6.3.0 (ADR-0098 Am.2); not a panic-stub"]
fn g_robin_self_2d_slope() {
    // 2D self-convergence on `[0, 1]²` with Robin BC on all 4 walls.
    // Mirror v2.2 G_NS2D_aniso probe-vs-2N-1 pattern (no closed-form oracle).
    // Plan: Strang per-axis composition of 1D RobinHeatChernoff on `[0,1]²`
    // with `(alpha, beta) = (1.0, 1.0)` on all four walls; assert slope ≤ -0.95.
    // Deferred: requires per-axis BoundaryPolicy override in Strang2D (v6.3.0).
}
