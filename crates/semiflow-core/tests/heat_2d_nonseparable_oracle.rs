//! Oracle test: non-separable 2D heat equation with rotated Gaussian.
//!
//! PDE: ∂_t u = `axx`·∂_xx u + `ayy`·∂_yy u + `c`·∂_xy u
//!
//! Exact solution via Gaussian propagation:
//!   `u_0(x,y)` = exp(-x² - y²)  (`Σ_0` = (1/2)·I)
//!
//! `Σ_t` = `Σ_0` + 2t·D, where D = [[axx, c/2], [c/2, ayy]]
//!
//! u(t,x,y) = sqrt(det `Σ_0` / det `Σ_t`) · exp(-½ · [x,y]·`Σ_t`⁻¹·[x,y])
//!
//! Gate: sup-error < 1e-3 at T=0.05 with n=100 grid, N=200 steps.
//! See math.md §10.7-bis, ADR-0016.

use semiflow_core::{
    chernoff::ApplyChernoffExt, BoundaryPolicy, DiffusionChernoff, Grid1D, Grid2D, GridFn2D,
    NonSeparable2DChernoff,
};

// ---------------------------------------------------------------------------
// Exact solution helpers
// ---------------------------------------------------------------------------

/// PDE parameters for non-separable 2D heat equation.
struct Params {
    axx: f64,
    ayy: f64,
    c: f64,
}

/// Compute `Σ_t` = `Σ_0` + 2t·D, return [[s00,s01],[s01,s11]].
fn sigma_t(p: &Params, t: f64) -> [f64; 3] {
    let s00 = 0.5 + 2.0 * t * p.axx;
    let s11 = 0.5 + 2.0 * t * p.ayy;
    let s01 = t * p.c;
    [s00, s01, s11]
}

/// Determinant of 2×2 symmetric matrix [[s00,s01],[s01,s11]].
fn det2(s: &[f64; 3]) -> f64 {
    s[0] * s[2] - s[1] * s[1]
}

/// Exact u(t, x, y) via 2×2 Gaussian propagation.
// x, y, t, p, q are standard math symbols in this PDE context.
#[allow(clippy::many_single_char_names)]
fn exact(p: &Params, t: f64, x: f64, y: f64) -> f64 {
    if t == 0.0 {
        return (-x * x - y * y).exp();
    }
    let sig = sigma_t(p, t);
    let det_t = det2(&sig);
    let det_0 = 0.25_f64; // det(Σ_0) = 0.5 * 0.5 = 0.25
                          // Σ_t⁻¹ = (1/det) * [[s11, -s01],[-s01, s00]]
    let norm = (det_0 / det_t).sqrt();
    let q = sig[2] * x * x - 2.0 * sig[1] * x * y + sig[0] * y * y;
    let exponent = -0.5 * q / det_t;
    norm * exponent.exp()
}

// ---------------------------------------------------------------------------
// Test-specific function pointers (constant-coefficient PDE)
// ---------------------------------------------------------------------------

fn axx_fn(_: f64) -> f64 {
    0.1
}
fn ayy_fn(_: f64) -> f64 {
    0.1
}
fn c_fn(_: f64, _: f64) -> f64 {
    0.05
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

/// Oracle accuracy gate: sup-error < 1e-3.
// n_steps is grid size ≤ 2^16, within f64 mantissa precision.
#[allow(clippy::cast_precision_loss)]
#[test]
fn heat_2d_nonseparable_oracle_accuracy() {
    let p = Params {
        axx: 0.1,
        ayy: 0.1,
        c: 0.05,
    };
    let t_end = 0.05_f64;
    let n = 100_usize;
    let n_steps = 200_usize;
    let tau = t_end / n_steps as f64;

    // Grid: periodic BC on [-4,4]×[-4,4] — wider domain reduces Gaussian
    // tail aliasing from periodic wrap-around.
    let gx = Grid1D::new(-4.0, 4.0, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::Periodic);
    let gy = gx;
    let grid = Grid2D::new(gx, gy);

    // Inner diffusion operators.
    let ix = DiffusionChernoff::new(axx_fn, |_| 0.0, |_| 0.0, p.axx, gx);
    let iy = DiffusionChernoff::new(ayy_fn, |_| 0.0, |_| 0.0, p.ayy, gy);

    // Mixed coupling (constant c = 0.05, c_norm_bound = 0.05).
    let op = NonSeparable2DChernoff::new(ix, iy, c_fn, p.c.abs(), grid).unwrap();

    // Initial condition.
    let mut u = GridFn2D::from_fn(grid, |x, y| (-x * x - y * y).exp());

    // Time march.
    for _ in 0..n_steps {
        u = op.apply_chernoff(tau, &u).unwrap();
    }

    // Compute sup-error over interior (avoid boundary row/col for safety).
    let margin = 2_usize;
    let mut sup_err = 0.0_f64;
    for j in margin..(n - margin) {
        for i in margin..(n - margin) {
            let xi = gx.x_at(i);
            let yj = gy.x_at(j);
            let k = j * n + i;
            let err = (u.values[k] - exact(&p, t_end, xi, yj)).abs();
            if err > sup_err {
                sup_err = err;
            }
        }
    }

    assert!(
        sup_err < 1e-3,
        "Oracle test failed: sup_err = {sup_err:.2e} >= 1e-3",
    );
}
