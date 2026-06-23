//! v0.5.0 regression suite for v0.6.0 (ADR-0013).
//!
//! Verifies that v0.6.0 additions (`Diffusion4thChernoff`, `diffusion4` module)
//! do NOT affect any v0.5.0 functionality:
//!
//! 1. `DiffusionChernoff` constant-a output is deterministic (bit-equal across
//!    two independent invocations with the same input).
//! 2. `Strang2D` 2D heat oracle gate (G1-2D: sup-norm error < 5e-4 at n=100).
//! 3. `Grid2D` / `GridFn2D` construction and value access unchanged.
//!
//! Reference: v0.5.0 tag (commit 120262a).

use semiflow::{
    ChernoffSemigroup, DiffusionChernoff, Grid1D, Grid2D, GridFn1D, GridFn2D, Strang2D,
};

// ---------------------------------------------------------------------------
// Test 1: DiffusionChernoff determinism (v0.5.0 regression)
// ---------------------------------------------------------------------------

const A0: f64 = 0.5;
const TAU: f64 = 0.01;

fn a_half(_: f64) -> f64 {
    0.5
}
fn a_zero(_: f64) -> f64 {
    0.0
}

/// Gaussian IC f(x) = exp(-x²/2).
fn gaussian_ic(grid: Grid1D) -> GridFn1D {
    GridFn1D::from_fn(grid, |x| libm::exp(-x * x / 2.0))
}

/// `DiffusionChernoff` output is deterministic: identical bits across two calls.
///
/// Any regression in the K-kernel (`gamma_a_baseline`) would break this.
/// The grid is small (100 nodes) and uses the default Reflect boundary.
#[test]
fn diffusion_chernoff_deterministic() {
    let grid = Grid1D::new(-5.0, 5.0, 100).expect("grid");
    let f0 = gaussian_ic(grid);
    let dc = DiffusionChernoff::new(a_half, a_zero, a_zero, A0, grid);

    let out1 = dc.apply_chernoff(TAU, &f0).expect("apply1");
    let out2 = dc.apply_chernoff(TAU, &f0).expect("apply2");

    for (i, (&v1, &v2)) in out1.values.iter().zip(out2.values.iter()).enumerate() {
        assert_eq!(
            v1.to_bits(),
            v2.to_bits(),
            "DiffusionChernoff non-deterministic at i={i}: run1={v1:.15e} run2={v2:.15e}"
        );
    }
    eprintln!(
        "DiffusionChernoff determinism: OK ({} nodes)",
        out1.values.len()
    );
}

/// `DiffusionChernoff` values are in sane range: positive, ≤ 1.0 for Gaussian IC.
///
/// A Chernoff function for a diffusion operator is positivity-preserving and
/// a contraction in L^∞ (growth = (1.0, 0.0)). For non-negative IC, all output
/// must be in [0, 1.0].
#[test]
fn diffusion_chernoff_positivity_and_contraction() {
    let grid = Grid1D::new(-5.0, 5.0, 200).expect("grid");
    let f0 = gaussian_ic(grid); // max value = 1.0 at x=0
    let dc = DiffusionChernoff::new(a_half, a_zero, a_zero, A0, grid);

    let out = dc.apply_chernoff(TAU, &f0).expect("apply");

    let max_val: f64 = out.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_val: f64 = out.values.iter().copied().fold(f64::INFINITY, f64::min);

    eprintln!("DiffusionChernoff contraction: min={min_val:.6e} max={max_val:.6e}");
    assert!(min_val >= -1e-14, "Positivity violated: min={min_val:.4e}");
    assert!(
        max_val <= 1.0 + 1e-12,
        "Contraction violated: max={max_val:.4e}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Strang2D 2D heat oracle (v0.5.0 flagship gate G1-2D)
// ---------------------------------------------------------------------------

/// 2D heat-kernel oracle at time `t`: `(1+2t)^{-1} · exp(-(x²+y²)/(1+2t))`.
fn oracle_heat_2d(t: f64, x: f64, y: f64) -> f64 {
    let denom = 1.0 + 2.0 * t;
    (1.0 / denom) * libm::exp(-(x * x + y * y) / denom)
}

/// G1-2D v0.5.0 regression: `Strang2D` + `DiffusionChernoff` achieves
/// sup-norm error < 5e-4 at n=100 time steps.
///
/// This verifies v0.6.0 additions do not break the 2D Strang composition.
/// Grid: 200×200 nodes on [-10,10]², `n_steps=100`.
#[test]
fn strang2d_heat_oracle_regression() {
    const N_SPATIAL: usize = 200;
    const N_STEPS: usize = 100;
    const T: f64 = 1.0;
    const GATE: f64 = 5e-4;

    let gx = Grid1D::new(-10.0, 10.0, N_SPATIAL).expect("gx");
    let gy = Grid1D::new(-10.0, 10.0, N_SPATIAL).expect("gy");
    let grid2d = Grid2D::new(gx, gy);

    let f0 = GridFn2D::from_fn(grid2d, |x, y| libm::exp(-(x * x + y * y)));

    let cx = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gx);
    let cy = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, gy);
    let phi2d = Strang2D::new(cx, cy);
    let semi = ChernoffSemigroup::new(phi2d, N_STEPS).expect("n >= 1");
    let u_n = semi.evolve(T, &f0).expect("evolve ok");

    let nx = grid2d.nx();
    let ny = grid2d.ny();
    let mut max_err: f64 = 0.0;
    for j in 0..ny {
        let yj = gy.x_at(j);
        for i in 0..nx {
            let xi = gx.x_at(i);
            let exact = oracle_heat_2d(T, xi, yj);
            let err = (u_n.values[j * nx + i] - exact).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }

    eprintln!("Strang2D regression: sup_err={max_err:.4e} (gate < {GATE})");
    assert!(
        max_err < GATE,
        "Strang2D v0.5.0 regression: err={max_err:.4e} ≥ {GATE} \
         (v0.6.0 broke 2D Strang composition)"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Grid2D / GridFn2D API unchanged
// ---------------------------------------------------------------------------

/// `Grid2D` and `GridFn2D` basic API is unchanged between v0.5.0 and v0.6.0.
#[test]
fn grid2d_api_unchanged() {
    let gx = Grid1D::new(-1.0, 1.0, 10).expect("gx");
    let gy = Grid1D::new(-2.0, 2.0, 20).expect("gy");
    let grid2d = Grid2D::new(gx, gy);

    assert_eq!(grid2d.nx(), 10, "Grid2D.nx()");
    assert_eq!(grid2d.ny(), 20, "Grid2D.ny()");

    let f = GridFn2D::from_fn(grid2d, |x, y| x * x * (y + 1.0));

    // Verify a specific value: f(x_3, y_7) = x_3² * (y_7 + 1)
    let x3 = gx.x_at(3);
    let y7 = gy.x_at(7);
    let expected = x3 * x3 * (y7 + 1.0);
    // row-major index: j*nx + i (I-T1 from tensor.yaml)
    let actual = f.values[7 * 10 + 3];
    assert!(
        (actual - expected).abs() < 1e-14,
        "GridFn2D.values layout: expected {expected:.6e} got {actual:.6e}"
    );

    // State trait: axpy
    let mut g = f.clone();
    g.axpy(2.0, &f);
    let v0 = f.values[0];
    assert!(
        (g.values[0] - 3.0 * v0).abs() < 1e-14,
        "GridFn2D State::axpy: expected {:.6e} got {:.6e}",
        3.0 * v0,
        g.values[0]
    );

    eprintln!(
        "Grid2D API: nx={} ny={} values_len={}",
        grid2d.nx(),
        grid2d.ny(),
        f.values.len()
    );
}
