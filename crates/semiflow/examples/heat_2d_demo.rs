//! 2D heat equation quickstart demo.
//!
//! Solves `∂_t u = ½(∂_xx + ∂_yy)u` from `u(0,x,y) = exp(-(x²+y²))`
//! on `[-10, 10]²` over `t ∈ [0, 1]` using
//! `Strang2D<DiffusionChernoff, DiffusionChernoff>`.
//!
//! Compares the Chernoff result against the closed-form 2D heat-kernel oracle
//!     `u(t,x,y) = (1+2t)^{-1} · exp(-(x²+y²)/(1+2t))`
//! at three step counts to show empirical second-order convergence.
//!
//! # Spatial grid
//! Uses `N = 1000` nodes per axis (1M cells total). This resolves the
//! `DiffusionChernoff` Chernoff shifts (±2√(a·τ)) well at all n values shown
//! and keeps accumulated spatial error well below the temporal floor.
//!
//! # Smoke gate
//! Exits 0 if sup-norm error at `n=50` is `< 5e-4`.
//!
//! # Note on convergence saturation
//! At fixed spatial grid, the Chernoff error eventually saturates at the
//! spatial-discretisation floor (∼dx⁴). This demo uses N=1000 so the floor
//! (~1.6e-7 per step, ~8e-6 total at n=50) is below 5e-4. For a systematic
//! slope test see `tests/strang_advdiff_2d.rs` (slow-tests, N=1000, n up to 256).
//!
//! Run with:  `cargo run --release --example heat_2d_demo -p semiflow-core`

use semiflow::{ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid2D, GridFn2D, Strang2D};

/// Spatial nodes per axis. 1000 × 1000 = 1M cells.
const N_NODES: usize = 1000;
const T_FINAL: f64 = 1.0;
const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;

/// 2D heat-kernel oracle: `(1+2t)^{-1} · exp(-(x²+y²)/(1+2t))`.
///
/// Normative formula from `contracts/semiflow-core.math.md §10.5(a)` (eq. 10.7).
/// Initial datum: `u_0(x,y) = exp(-(x²+y²))`.
fn oracle(t: f64, x: f64, y: f64) -> f64 {
    let denom = 1.0 + 2.0 * t;
    (1.0 / denom) * (-(x * x + y * y) / denom).exp()
}

/// Run `n` Chernoff steps and return sup-norm error vs. oracle at `T_FINAL`.
fn run(n: usize) -> f64 {
    let gx = Grid1D::new(X_MIN, X_MAX, N_NODES).expect("grid x OK");
    let gy = Grid1D::new(X_MIN, X_MAX, N_NODES).expect("grid y OK");
    let grid2d = Grid2D::new(gx, gy);

    // Initial datum: u_0(x, y) = exp(-(x² + y²)).
    let f0 = GridFn2D::from_fn(grid2d, |x, y| (-(x * x + y * y)).exp());

    // Per-axis heat: L_z = ½∂²_z, constant a=0.5, a'=0, a''=0 (ζ-A fast path).
    let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);

    // Palindromic Strang2D: Sx(τ/2) ∘ Sy(τ) ∘ Sx(τ/2).
    let phi2d = Strang2D::new(cx, cy);
    let semi = ChernoffSemigroup::new(phi2d, n).expect("n >= 1");
    let u_n = semi.evolve(T_FINAL, &f0).expect("evolve OK");

    let nx = grid2d.nx();
    let ny = grid2d.ny();
    let mut max_err: f64 = 0.0;
    for j in 0..ny {
        let yj = gy.x_at(j);
        for i in 0..nx {
            let xi = gx.x_at(i);
            let err = (u_n.values[j * nx + i] - oracle(T_FINAL, xi, yj)).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

fn print_header() {
    println!("=== semiflow-core v0.5.0 — 2D heat kernel demo ===");
    println!();
    println!("PDE:    ∂_t u = ½(∂_xx + ∂_yy)u");
    println!("IC:     u(0,x,y) = exp(-(x² + y²))");
    println!("BC:     reflective on Ω = [-10, 10]²");
    println!("T:      1.0");
    println!("Grid:   N = {N_NODES} × {N_NODES}");
    println!("Oracle: u(t,x,y) = (1+2t)^{{-1}} · exp(-(x²+y²)/(1+2t))");
    println!("        at t=1 → (1/3) · exp(-(x²+y²)/3)");
}

fn print_convergence_table() {
    println!();
    println!("--- Strang2D (palindromic, order-2) — N={N_NODES}×{N_NODES} ---");
    let mut prev: Option<f64> = None;
    for n in [10_usize, 20, 50] {
        let err = run(n);
        let ratio = prev.map_or(0.0, |p| p / err);
        println!(
            "  n={n:4}  sup-norm err = {err:.4e} \
             (ratio prev/this: {ratio:.2}× — expect ~{:.0}× per {:.0}× n)",
            if prev.is_some() { 4.0 } else { 0.0 },
            if prev.is_some() { 2.5_f64 } else { 0.0 },
        );
        prev = Some(err);
    }
}

fn main() {
    print_header();
    print_convergence_table();

    // Smoke gate: n=50, N=1000 must achieve < 5e-4.
    println!();
    let final_err = run(50);
    let gate = 5.0e-4_f64;
    println!(
        "Smoke gate (n=50, N={N_NODES}): err = {final_err:.4e}  (gate < {gate:.0e})  {}",
        if final_err < gate { "PASS" } else { "FAIL" }
    );
    if final_err >= gate {
        std::process::exit(1);
    }
}
