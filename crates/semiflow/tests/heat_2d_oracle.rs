//! G1-2D and G2-2D — 2D Gaussian heat-kernel accuracy tests.
//!
//! PDE: `∂_t u = ½(∂_xx + ∂_yy)u`, `u_0(x, y) = exp(-(x² + y²))`.
//!
//! Closed-form oracle (2D heat kernel, math.md §10.5(a), eq. 10.7):
//! ```text
//! u(t, x, y) = (1 + 2t)^{-1} · exp(-(x² + y²) / (1 + 2t))
//! ```
//!
//! At `t = 1`: `u(1, x, y) = (1/3) · exp(-(x² + y²) / 3)`.
//!
//! G1-2D (n=100):  sup-norm error < 5e-4 (grid 200×200)
//! G2-2D (n=1000): sup-norm error < 5e-5 (grid 500×500) [slow-tests]
//!
//! # Spatial grid selection
//! G1-2D uses N=200 per axis. For G2-2D at n=1000 (τ=0.001), the
//! `DiffusionChernoff` Chernoff shifts are h₀=2√(0.5·0.001)=0.044. With N=200
//! (dx=0.1), h₀ < dx causes degraded interpolation — N=200 is too coarse for
//! n=1000. G2-2D uses N=500 (dx=0.04), ensuring h₀=0.044≈dx (shift ≥ dx
//! gives stable cubic Hermite interpolation). N=500² = 250K cells at n=1000
//! steps takes ~60 s release mode (slow-tests gate is appropriate).
//!
//! Operator: `Strang2D<DiffusionChernoff, DiffusionChernoff>` with
//! constant `a = 0.5` per axis (heat: `½∂²_x + ½∂²_y`).
//!
//! Reference: `contracts/semiflow-core.tensor.yaml`, `contracts/semiflow-core.math.md`
//! §10.5(a), `docs/adr/0012-tensor-product-2d.md`.

use semiflow::{ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid2D, GridFn2D, Strang2D};

// ---------------------------------------------------------------------------
// Gate constants (NON-NEGOTIABLE — do NOT relax)
// ---------------------------------------------------------------------------

/// G1-2D gate: sup-norm error at n=100 must be strictly below this value.
const TOL_G1_2D: f64 = 5.0e-4;
/// G2-2D gate: sup-norm error at n=1000 must be strictly below this value.
#[cfg(feature = "slow-tests")]
const TOL_G2_2D: f64 = 5.0e-5;

const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
const T_FINAL: f64 = 1.0;

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// 2D heat-kernel oracle at time `t`: `(1+2t)^{-1} · exp(-(x²+y²)/(1+2t))`.
///
/// Normative formula from `contracts/semiflow-core.math.md §10.5(a)` eq. (10.7).
/// Initial datum: `u_0(x,y) = exp(-(x²+y²))`.
/// PDE: `∂_t u = ½(∂_xx + ∂_yy)u` (heat with diffusion coefficient 0.5 per axis).
#[inline]
fn oracle_heat_2d(t: f64, x: f64, y: f64) -> f64 {
    let denom = 1.0 + 2.0 * t;
    (1.0 / denom) * (-(x * x + y * y) / denom).exp()
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Run `n_steps` `Strang2D` Chernoff iterations from `t=0` to `T_FINAL=1` and
/// return the sup-norm error vs. the 2D heat-kernel oracle.
///
/// Grid: `n_nodes × n_nodes` on `[X_MIN, X_MAX]²`, Reflect BC (default).
/// Operator: `Strang2D<DiffusionChernoff(0.5), DiffusionChernoff(0.5)>`.
fn heat_2d_error(n_steps: usize, n_nodes: usize) -> f64 {
    let gx = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid x valid");
    let gy = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid y valid");
    let grid2d = Grid2D::new(gx, gy);

    // Initial datum: u_0(x, y) = exp(-(x² + y²)).
    let f0 = GridFn2D::from_fn(grid2d, |x, y| (-(x * x + y * y)).exp());

    // Per-axis heat operator: L_axis = 0.5 · ∂² (constant a, a'=0, a''=0).
    let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);

    // Palindromic Strang2D: Sx(τ/2) ∘ Sy(τ) ∘ Sx(τ/2).
    let phi2d = Strang2D::new(cx, cy);
    let semi = ChernoffSemigroup::new(phi2d, n_steps).expect("n >= 1");
    let u_n = semi
        .evolve(T_FINAL, &f0)
        .expect("evolve succeeds for valid inputs");

    // Sup-norm error vs. oracle at t = T_FINAL.
    let nx = grid2d.nx();
    let ny = grid2d.ny();
    let mut max_err: f64 = 0.0;
    for j in 0..ny {
        let yj = gy.x_at(j);
        for i in 0..nx {
            let xi = gx.x_at(i);
            let exact = oracle_heat_2d(T_FINAL, xi, yj);
            // row-major index: j*nx + i (I-T1)
            let err = (u_n.values[j * nx + i] - exact).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// G1-2D — n = 100, N = 200
// ---------------------------------------------------------------------------

/// G1-2D: sup-norm error at n=100 must satisfy `‖err‖_∞ < 5e-4`.
///
/// Grid: 200 × 200 on `[-10, 10]²`. Chernoff shift h₀=0.1≈dx so
/// interpolation is stable.
///
/// Gate from `contracts/semiflow-core.tensor.yaml` and `acceptance-criteria.md`
/// (v0.5.0, ADR-0012). Non-negotiable: if this fails, report the empirical
/// number and escalate to the Architect.
#[test]
fn g1_heat_2d_n100() {
    let err = heat_2d_error(100, 200);
    println!("G1-2D: sup-norm error at n=100, N=200 = {err:.3e}  (gate: < {TOL_G1_2D:.0e})");
    assert!(
        err < TOL_G1_2D,
        "G1-2D FAIL: max error {err:.3e} >= gate {TOL_G1_2D:.0e} — escalate to Architect"
    );
}

// ---------------------------------------------------------------------------
// G2-2D — n = 1000, N = 500  (slow-tests gated)
// ---------------------------------------------------------------------------

/// G2-2D: sup-norm error at n=1000 must satisfy `‖err‖_∞ < 5e-5`.
///
/// Grid: 500 × 500 on `[-10, 10]²`. N=500 is required at n=1000 because the
/// Chernoff shift h₀=2√(0.5/1000)=0.044≈dx=0.04 — aligns well with grid
/// nodes. N=200 (dx=0.1) is too coarse for n=1000 (h₀ < dx/2), causing
/// elevated interpolation error. See file-level comment for derivation.
///
/// Gate from `contracts/semiflow-core.tensor.yaml` (v0.5.0, ADR-0012).
/// Gated behind `slow-tests`: N=500×500 at n=1000 takes ~60 s release mode.
#[test]
#[cfg(feature = "slow-tests")]
fn g2_heat_2d_n1000() {
    let err = heat_2d_error(1000, 500);
    println!("G2-2D: sup-norm error at n=1000, N=500 = {err:.3e}  (gate: < {TOL_G2_2D:.0e})");
    assert!(
        err < TOL_G2_2D,
        "G2-2D FAIL: max error {err:.3e} >= gate {TOL_G2_2D:.0e} — escalate to Architect"
    );
}

// ---------------------------------------------------------------------------
// Sanity: oracle self-consistency (separability check)
// ---------------------------------------------------------------------------

/// Verify the oracle is separable: `u(t, x, y) = f(x, t) · f(y, t)` where
/// `f(z, t) = (1+2t)^{-1/2} · exp(-z²/(1+2t))` (1D heat kernel).
///
/// This confirms the closed-form formula is consistent with the 1D oracle.
#[test]
fn oracle_separability() {
    let t = 1.0;
    let denom = 1.0 + 2.0 * t;
    for &x in &[-1.0_f64, 0.0, 1.0, 2.0] {
        for &y in &[-0.5_f64, 0.5, 1.5] {
            let two_d = oracle_heat_2d(t, x, y);
            let one_d_x = denom.sqrt().recip() * (-x * x / denom).exp();
            let one_d_y = denom.sqrt().recip() * (-y * y / denom).exp();
            // u_2D(t,x,y) = (1+2t)^{-1} exp(-(x²+y²)/(1+2t))
            //             = [(1+2t)^{-1/2} exp(-x²/(1+2t))]
            //               · [(1+2t)^{-1/2} exp(-y²/(1+2t))]
            //             = one_d_x · one_d_y.
            let sep = one_d_x * one_d_y;
            assert!(
                (two_d - sep).abs() < 1e-14 * (1.0 + two_d.abs()),
                "oracle separability violated at (x={x}, y={y}): 2D={two_d:.15e}, \
                 sep={sep:.15e}"
            );
        }
    }
}
