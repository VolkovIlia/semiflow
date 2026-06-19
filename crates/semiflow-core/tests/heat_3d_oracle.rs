//! 3D Gaussian heat-kernel accuracy tests.
//!
//! PDE: `∂_t u = a(∂_xx + ∂_yy + ∂_zz)u`, `a = 0.1`.
//! Initial datum: `u_0(x, y, z) = exp(-(x² + y² + z²))`.
//!
//! Closed-form oracle (separable 3D Gaussian, math.md §10.8.7, eq. 10.8.7):
//! ```text
//! u(t, x, y, z) = (1 + 4·a·t)^{-3/2} · exp(-(x² + y² + z²) / (1 + 4·a·t))
//! ```
//! With `a = 0.1` at `T = 0.2`:
//!   denominator `D = 1 + 4·0.1·0.2 = 1.08`.
//!   `u(T, x,y,z) = D^{-3/2} · exp(-(x²+y²+z²)/D)`.
//!
//! Operator: `Strang3D<DiffusionChernoff(0.1), DiffusionChernoff(0.1), DiffusionChernoff(0.1)>`.
//!
//! # Accuracy note
//! At `N=64`, `n=200`: tau=0.001, Chernoff shift h0=2*sqrt(a*tau)~0.02, dx~0.159.
//! Since h0 << dx (coarse-grid regime), dominant error is spatial interpolation.
//! Gate 5e-3 is set accordingly — the tight convergence gate is the slope test
//! (`tests/strang_3d_slope.rs`, `G5_3D`, `slow-tests`).
//!
//! Reference: `contracts/semiflow-core.tensor.yaml`, `docs/adr/0024-tensor-3d.md`.

use semiflow_core::{ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid3D, GridFn3D, Strang3D};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Diffusion coefficient (heat with `a·∂²` per axis).
const DIFFUSION_A: f64 = 0.1;
const X_MIN: f64 = -5.0;
const X_MAX: f64 = 5.0;
/// Final time.
const T_FINAL: f64 = 0.2;
/// Oracle smoke-test gate: sup-norm error at n=200, N=16 must be below this value.
/// Note: tight slope gate over `N ∈ {16, 32, 64}` is in `tests/strang_3d_slope.rs`.
const TOL_SMOKE: f64 = 8.0e-2;

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

/// 3D heat oracle at time `t`: `D^{-3/2} · exp(-(x²+y²+z²)/D)`, `D=1+4·a·t`.
///
/// Normative formula from `contracts/semiflow-core.math.md §10.8.7` eq. (10.8.7).
/// Derived from separability: `u(t,x,y,z) = f(x,t)·f(y,t)·f(z,t)` where
/// `f(z,t) = (1+4·a·t)^{-1/2} · exp(-z²/(1+4·a·t))` (1D heat kernel with coeff `a`).
#[allow(clippy::many_single_char_names)]
#[inline]
fn oracle_heat_3d(t: f64, x: f64, y: f64, z: f64) -> f64 {
    let d = 1.0 + 4.0 * DIFFUSION_A * t;
    let norm_sq = x * x + y * y + z * z;
    d.powf(-1.5) * (-norm_sq / d).exp()
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// Run `n_steps` `Strang3D` Chernoff iterations and return the sup-norm error.
///
/// Grid: `n_nodes³` on `[X_MIN, X_MAX]³`, default Reflect BC.
/// Operator: `Strang3D<DiffusionChernoff(a), DiffusionChernoff(a), DiffusionChernoff(a)>`.
// n_nodes ≤ 64; n_steps ≤ 200 — within f64 52-bit mantissa range.
#[allow(clippy::cast_precision_loss)]
fn heat_3d_error(n_steps: usize, n_nodes: usize) -> f64 {
    let gx = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid x valid");
    let gy = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid y valid");
    let gz = Grid1D::new(X_MIN, X_MAX, n_nodes).expect("grid z valid");
    let g3 = Grid3D::new(gx, gy, gz).expect("Grid3D valid");

    let f0 = GridFn3D::from_fn(g3, |x, y, z| (-(x * x + y * y + z * z)).exp());

    let cx = DiffusionChernoff::new(|_| DIFFUSION_A, |_| 0.0_f64, |_| 0.0_f64, DIFFUSION_A, gx);
    let cy = DiffusionChernoff::new(|_| DIFFUSION_A, |_| 0.0_f64, |_| 0.0_f64, DIFFUSION_A, gy);
    let cz = DiffusionChernoff::new(|_| DIFFUSION_A, |_| 0.0_f64, |_| 0.0_f64, DIFFUSION_A, gz);

    let phi3d = Strang3D::new(cx, cy, cz);
    let semi = ChernoffSemigroup::new(phi3d, n_steps).expect("n >= 1");
    let u_n = semi.evolve(T_FINAL, &f0).expect("evolve succeeds");

    let nx = g3.nx();
    let ny = g3.ny();
    let nz = g3.nz();
    let mut max_err: f64 = 0.0;
    for k in 0..nz {
        let zk = gz.x_at(k);
        for j in 0..ny {
            let yj = gy.x_at(j);
            for i in 0..nx {
                let xi = gx.x_at(i);
                let exact = oracle_heat_3d(T_FINAL, xi, yj, zk);
                // x-fastest index (I-T1-3D)
                let err = (u_n.values[k * nx * ny + j * nx + i] - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Smoke test — n=200, N=16
// ---------------------------------------------------------------------------

/// `g5_heat_3d_oracle_smoke`: smoke test that `Strang3D` produces finite, bounded output.
///
/// Uses N=16 per axis to keep runtime under 2 s in debug mode.
/// The tight slope gate over `N ∈ {16, 32, 64}` lives in `tests/strang_3d_slope.rs`
/// (gated by `slow-tests`).
///
/// Gate: sup-norm error must be below 0.08 — coarse grid, but confirms the 3D
/// implementation runs without panicking and produces an approximate solution.
#[test]
fn g5_heat_3d_oracle_smoke() {
    let n_steps = 200;
    let n_nodes = 16;
    let err = heat_3d_error(n_steps, n_nodes);
    println!(
        "G5_3D smoke: n={n_steps}, N={n_nodes} => sup-norm error = {err:.3e}  (gate: < {TOL_SMOKE:.0e})"
    );
    assert!(
        err < TOL_SMOKE,
        "G5_3D smoke FAIL: max error {err:.3e} >= gate {TOL_SMOKE:.0e} — escalate to Architect"
    );
}

// ---------------------------------------------------------------------------
// Oracle self-consistency — separability check
// ---------------------------------------------------------------------------

/// Verify the 3D oracle is separable: `u(t,x,y,z) = f(x,t)·f(y,t)·f(z,t)`
/// where `f(z,t) = (1+4·a·t)^{-1/2} · exp(-z²/(1+4·a·t))`.
#[test]
fn oracle_3d_separability() {
    let t = T_FINAL;
    let d = 1.0 + 4.0 * DIFFUSION_A * t;
    for &x in &[-2.0_f64, 0.0, 1.5] {
        for &y in &[-1.0_f64, 0.5] {
            for &z in &[0.0_f64, 1.0, -1.5] {
                let three_d = oracle_heat_3d(t, x, y, z);
                let fx = d.sqrt().recip() * (-x * x / d).exp();
                let fy = d.sqrt().recip() * (-y * y / d).exp();
                let fz = d.sqrt().recip() * (-z * z / d).exp();
                let sep = fx * fy * fz;
                assert!(
                    (three_d - sep).abs() < 1e-14 * (1.0 + three_d.abs()),
                    "oracle_3d_separability violated at ({x},{y},{z}): \
                     3D={three_d:.15e}, sep={sep:.15e}"
                );
            }
        }
    }
}
