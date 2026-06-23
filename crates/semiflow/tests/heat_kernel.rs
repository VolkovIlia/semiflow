//! G1 and G2 — Gaussian heat-kernel accuracy tests.
//!
//! Evolve `u_0(x) = exp(-x²)` under `∂_t u = ½ ∂_xx u` from t=0 to t=1
//! using the Chernoff approximation `(S(t/n))^n` and compare to the exact
//! heat-kernel oracle `u(1,x) = (3)^{-1/2} exp(-x²/3)`.
//!
//! G1 (n=100):  sup-norm error < 5e-4
//! G2 (n=1000): sup-norm error < 5e-5

use semiflow::{ChernoffSemigroup, Grid1D, GridFn1D, ShiftChernoff1D};

// Revised per acceptance-criteria.md amendment 2026-04-29.
const TOL_N100: f64 = 5.0e-4;
// Revised per acceptance-criteria.md amendment 2026-04-29.
const TOL_N1000: f64 = 5.0e-5;

/// Build the standard heat-kernel test configuration, run `n_steps` Chernoff
/// iterations to t=1, and return the sup-norm error against the oracle.
fn heat_kernel_error(n_steps: usize) -> f64 {
    // N=1000 uniform nodes on [-10, 10]. Defaults: Reflect + CubicHermite.
    let grid = Grid1D::new(-10.0, 10.0, 1000).unwrap();
    let f0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

    // Operator: L = 0.5 ∂_xx  (a=0.5, b=0, c=0)
    let chernoff = ShiftChernoff1D::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.0, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps).unwrap();
    let u_t = semigroup.evolve(1.0, &f0).unwrap();

    // Oracle: u(1,x) = (1+2·1)^{-1/2} exp(-x²/(1+2·1)) = 3^{-1/2} exp(-x²/3)
    let inv_sqrt3 = (3.0_f64).sqrt().recip();
    let mut max_err: f64 = 0.0;
    for i in 0..u_t.values.len() {
        let x = grid.x_at(i);
        let oracle = inv_sqrt3 * (-(x * x) / 3.0).exp();
        let err = (u_t.values[i] - oracle).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

#[test]
fn g1_heat_kernel_n100() {
    let err = heat_kernel_error(100);
    assert!(
        err < TOL_N100,
        "G1 FAIL: max error {err:.3e} >= tol {TOL_N100:.3e}"
    );
}

#[test]
fn g2_heat_kernel_n1000() {
    let err = heat_kernel_error(1000);
    assert!(
        err < TOL_N1000,
        "G2 FAIL: max error {err:.3e} >= tol {TOL_N1000:.3e}"
    );
}
