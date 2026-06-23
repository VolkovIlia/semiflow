//! Adaptive 2D heat — sanity gate for `AdaptivePI<Strang2D<Diffusion4thChernoff, …>>`.
//!
//! Demonstrates that `AdaptivePI` works at 2D scale: it wraps a `Strang2D` composed
//! of `Diffusion4thChernoff` per axis and integrates the 2D heat equation to `t=1`
//! with mixed tolerance `(abs=0, rel=1e-4)`.
//!
//! # PDE and oracle
//! `∂_t u = ½(∂_xx + ∂_yy)u`, `u_0(x,y) = exp(-(x²+y²))`.
//! Oracle: `u(1,x,y) = ⅓ · exp(-(x²+y²)/3)` (math.md §10.5(a) eq. 10.7).
//!
//! # Grid
//! N=200 per axis (40 000 cells) — small enough that the test completes in < 30 s
//! in release mode. The coarser grid means the spatial error is O(dx⁴) ≈ (0.1)⁴
//! = 1e-4, so the final error bound `5e-3` includes both temporal and spatial
//! contributions.
//!
//! # Gates
//! 1. `outcome.steps_accepted ≥ 1` — basic sanity.
//! 2. `‖final − exact‖_∞ ≤ 5e-3` — accuracy within tolerance × Lady Windermere fan.
//!    The spatial floor at N=200 (dx=0.1) is ~(dx)⁴ ≈ 1e-4; with tol_rel=1e-4
//!    the temporal error is similar. Budget 5× for accumulated error → 5e-3.
//!
//! # Notes
//! - `AdaptivePI<C>` is NOT a `ChernoffFunction` — do NOT wrap in `ChernoffSemigroup`.
//!   Each `evolve_adaptive` call is one complete integration to `t`.
//! - `with_tolerance(0.0, 1e-4)` sets `tol_abs=0`, `tol_rel=1e-4` (pure relative).
//!
//! Reference: `docs/adr/0014-adaptive-pi-controller.md`,
//! `contracts/semiflow-core.math.md §11`, `contracts/semiflow-core.properties.yaml`.

#![cfg(feature = "slow-tests")]

use semiflow::{AdaptivePI, Diffusion4thChernoff, Grid1D, Grid2D, GridFn2D, Strang2D};

// ---------------------------------------------------------------------------
// Grid parameters
// ---------------------------------------------------------------------------

/// `N_NODES` per axis for the adaptive 2D sanity test.
/// Small (200) to keep runtime ≤ 30 s in release mode.
const N_NODES: usize = 200;
const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
const T_FINAL: f64 = 1.0;

/// Accuracy gate: `‖final − exact‖_∞ ≤ TOL_ACCURACY`.
///
/// Spatial floor at N=200 (dx=0.1): O(dx⁴) ≈ 1e-4.
/// With tol_rel=1e-4, Lady Windermere fan budget ×5 → 5e-3.
const TOL_ACCURACY: f64 = 5.0e-3;

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// 2D heat-kernel oracle: `(1+2t)^{-1} · exp(-(x²+y²)/(1+2t))`.
#[inline]
fn oracle_heat_2d(t: f64, x: f64, y: f64) -> f64 {
    let denom = 1.0 + 2.0 * t;
    (1.0 / denom) * (-(x * x + y * y) / denom).exp()
}

// ---------------------------------------------------------------------------
// Adaptive 2D heat sanity test
// ---------------------------------------------------------------------------

/// Sanity gate: `AdaptivePI<Strang2D<Diffusion4thChernoff, Diffusion4thChernoff>>` at 2D scale.
///
/// Demonstrates that the generic `AdaptivePI<C>` works when `C = Strang2D<…>`:
/// `C::S = GridFn2D` (implements `State + Clone`), and Richardson half-stepping
/// operates on 2D grid functions without any 2D specialisation.
///
/// Requires `--features slow-tests`: N=200² at ~200 accepted steps ≈ 10 s release mode.
#[test]
fn adaptive_2d_heat_sanity() {
    let gx = Grid1D::new(X_MIN, X_MAX, N_NODES).expect("grid x valid");
    let gy = Grid1D::new(X_MIN, X_MAX, N_NODES).expect("grid y valid");
    let grid2d = Grid2D::new(gx, gy);

    // Initial datum: u_0(x, y) = exp(-(x² + y²)).
    let f0 = GridFn2D::from_fn(grid2d, |x, y| (-(x * x + y * y)).exp());

    // Per-axis 4th-order heat kernel.
    let cx = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = Diffusion4thChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);
    let phi2d = Strang2D::new(cx, cy);

    // AdaptivePI with pure relative tolerance tol_rel=1e-4 (tol_abs=0).
    let mut controller = AdaptivePI::new(phi2d).with_tolerance(0.0, 1e-4);

    let outcome = controller
        .evolve_adaptive(T_FINAL, &f0)
        .expect("adaptive integration should succeed for smooth heat PDE");

    println!(
        "adaptive_2d_heat: steps_accepted={}, steps_rejected={}, last_tau={:.3e}",
        outcome.steps_accepted, outcome.steps_rejected, outcome.last_tau
    );

    // Gate 1: at least one step accepted (basic sanity).
    assert!(
        outcome.steps_accepted >= 1,
        "adaptive_2d_heat: no steps accepted — controller logic broken"
    );

    // Compute sup-norm error vs. oracle.
    let nx = grid2d.nx();
    let ny = grid2d.ny();
    let mut max_err: f64 = 0.0;
    for j in 0..ny {
        let yj = gy.x_at(j);
        for i in 0..nx {
            let xi = gx.x_at(i);
            let exact = oracle_heat_2d(T_FINAL, xi, yj);
            let err = (outcome.final_state.values[j * nx + i] - exact).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    println!("adaptive_2d_heat: sup-norm err = {max_err:.3e}  (gate: < {TOL_ACCURACY:.0e})");

    // Gate 2: accuracy within spatial floor × Lady Windermere budget.
    assert!(
        max_err < TOL_ACCURACY,
        "adaptive_2d_heat: err {max_err:.3e} >= gate {TOL_ACCURACY:.0e} — \
         spatial floor at N=200 is ~1e-4; tol_rel=1e-4 budget exhausted. \
         Try N=500 or tol_rel=1e-5."
    );
}
