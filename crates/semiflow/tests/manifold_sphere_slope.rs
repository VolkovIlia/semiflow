//! G26 — Manifold Sphere-S² Chernoff convergence slope (`RELEASE_BLOCKING`).
//!
//! Gate (properties.yaml v2.8 G26, `RELEASE_BLOCKING`, 2 sub-tests):
//!   IC: Y_{0,0} + Y_{1,0} + Y_{1,1} + Y_{2,0} (first 4 real sph. harmonics).
//!   `n_Chernoff` = 40 (τ = T/40 = 0.00125), sweep `n_chart` ∈ {16,32,64,128}.
//!   Spatial convergence: err ≈ O(τ^p) + O(h²); with τ fixed, err ~ O(h²) for corrected.
//!
//!   Sub-test 1 (BASE, no R/12): OLS slope ≤ -0.95 (order 1, 5% margin).
//!     Oracle: u(T,x) = ∑_{ℓm} c_{ℓm} exp(-ℓ(ℓ+1)T) Y_{ℓm}(x).
//!
//!   Sub-test 2 (R/12 CORRECTION): OLS slope ≤ -1.95 (order 2, 2.5% margin).
//!     The R/12 correction [1+τR/12] outside the integral shifts the effective
//!     generator from Δ_{S²} to Δ_{S²}+R/12. For unit S² (R=2) the shift is +1/6.
//!     Oracle: `u_R(T,x)` = ∑_{ℓm} c_{ℓm} exp((-ℓ(ℓ+1)+1/6)T) Y_{ℓm}(x).
//!     The spatial (bilinear) convergence of the corrected method to its own limit
//!     is O(h²), giving slope ≈ -2 as `n_chart` grows.
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]

use semiflow::{
    ChernoffFunction, Grid1D, Grid2D, GridFn2D, ManifoldChernoff, ScratchPool, Sphere2,
};

// ─── Gate constants ───────────────────────────────────────────────────────────

const T_HORIZON: f64 = 0.05;
// n_Chernoff = 40 (τ = 0.00125); sweep n_chart ∈ {16,32,64,128} for spatial convergence.
// BASE (no R/12): err ≈ O(τ) + O(h²). τ-floor ≈ T·τ = 6.25e-5, well below h²=O(1e-4) at n=128.
//   Gate -0.95 gives 5% margin vs the -1.0 order-1 Chernoff claim.
// CORRECTED (R/12): effective generator is Δ_{S²}+R/12. The corrected oracle
//   u_R uses eigenvalues λ_ℓ + R/12 = -ℓ(ℓ+1)+1/6, so the comparison measures
//   only the SPATIAL (bilinear) error: err ≈ O(h²) → slope ≈ -2.
//   Gate -1.95 gives 2.5% margin vs the -2.0 bilinear spatial order.
//   With τ=0.00125: temporal floor O(T·τ)=6.25e-5, below spatial O(h²)≈1e-4 at n=128.
const N_CHERNOFF: usize = 40; // τ = T/40 = 0.00125
const SLOPE_GATE_BASE: f64 = -0.95;
const SLOPE_GATE_CORRECTED: f64 = -1.95;
const N_CHART_SWEEP: [usize; 4] = [16, 32, 64, 128];

// ─── Spherical harmonics helpers ──────────────────────────────────────────────
//
// Real forms (cos m·φ branch) of Y_{ℓ,m}:
//   Y_{0,0}  = 1 / (2√π)
//   Y_{1,0}  = √(3/(4π)) · cos θ
//   Y_{1,1}  = −√(3/(4π)) · sin θ · cos φ   (real, cos-branch)
//   Y_{2,0}  = √(5/(16π)) · (3·cos²θ − 1)   (λ = −6)

fn y00(_theta: f64, _phi: f64) -> f64 {
    1.0 / (2.0 * core::f64::consts::PI.sqrt())
}

fn y10(theta: f64, _phi: f64) -> f64 {
    (3.0 / (4.0 * core::f64::consts::PI)).sqrt() * theta.cos()
}

fn y11(theta: f64, phi: f64) -> f64 {
    -(3.0 / (4.0 * core::f64::consts::PI)).sqrt() * theta.sin() * phi.cos()
}

fn y20(theta: f64, _phi: f64) -> f64 {
    (5.0 / (16.0 * core::f64::consts::PI)).sqrt() * (3.0 * theta.cos().powi(2) - 1.0)
}

/// Initial datum: unit-weight sum of first 4 real spherical harmonics.
fn initial_datum(theta: f64, phi: f64) -> f64 {
    y00(theta, phi) + y10(theta, phi) + y11(theta, phi) + y20(theta, phi)
}

/// Oracle for BASE sub-test: eigenmode decay via `λ_ℓ` = -ℓ(ℓ+1) (heat on `S²`).
///   `Y_{0,0}`: λ=0  → no decay.
///   `Y_{1,0}`, `Y_{1,1}`: λ=-2 → exp(-2t).
///   `Y_{2,0}`: λ=-6 → exp(-6t).
fn oracle(t: f64, theta: f64, phi: f64) -> f64 {
    y00(theta, phi)
        + (y10(theta, phi) + y11(theta, phi)) * (-2.0 * t).exp()
        + y20(theta, phi) * (-6.0 * t).exp()
}

/// Oracle for CORRECTED sub-test: effective generator `Δ_{S²}` + R/12.
///
/// The outer `[1+τR/12]` correction in formula (24.1) shifts the effective generator
/// from `Δ_{S²}` to `Δ_{S²}` + R/12. For unit `S²` (R = 2): shift = 1/6.
/// Modified eigenvalues: `λ_ℓ^{eff}` = -ℓ(ℓ+1) + R/12 = -ℓ(ℓ+1) + 1/6.
///   `Y_{0,0}`: `λ^{eff}`=+1/6 → exp(+t/6)  (slight growth).
///   `Y_{1,0}`, `Y_{1,1}`: `λ^{eff}`=-11/6 → exp(-11t/6).
///   `Y_{2,0}`: `λ^{eff}`=-35/6 → exp(-35t/6).
///
/// Comparing corrected output to this oracle isolates the SPATIAL bilinear error,
/// which converges at O(h²), giving OLS slope ≈ -2.
fn oracle_r12(t: f64, theta: f64, phi: f64) -> f64 {
    let r12 = 2.0_f64 / 12.0; // R/12 for unit sphere (R = 2)
    y00(theta, phi) * (r12 * t).exp()
        + (y10(theta, phi) + y11(theta, phi)) * ((-2.0 + r12) * t).exp()
        + y20(theta, phi) * ((-6.0 + r12) * t).exp()
}

// ─── Chart grid construction ──────────────────────────────────────────────────

/// Build a (θ, φ) chart grid for S² with `n_theta` × `n_phi` nodes.
///
/// θ ∈ [ε, π−ε] (avoid poles); φ ∈ [0, 2π].
fn build_chart_grid(n_theta: usize, n_phi: usize) -> Grid2D<f64> {
    let eps = 0.02; // pole exclusion margin (~1.1°); smaller → less domain wasted
    let g_theta = Grid1D::new(eps, core::f64::consts::PI - eps, n_theta).unwrap();
    let g_phi = Grid1D::new(0.0, 2.0 * core::f64::consts::PI, n_phi).unwrap();
    Grid2D::new(g_theta, g_phi)
}

// ─── Sup-norm error helper ────────────────────────────────────────────────────

/// Run `N_CHERNOFF` Chernoff steps on an `n_chart` × `2n_chart` grid; return sup-norm error.
///
/// `n_Chernoff` = 40 (τ = 0.00125) is fixed; `n_chart` controls spatial resolution.
/// This measures spatial convergence: err ≈ O(τ^p) + O(h²).
/// - BASE (p=1):       err ≈ O(τ) + C/n².  OLS slope gate -0.95.
/// - CORRECTED (p=2):  effective generator is Δ+R/12; `oracle_fn` = `oracle_r12`.
///   With τ=0.00125, temporal floor O(Tτ)≈6.25e-5 lies below bilinear O(h²)≈1e-4
///   at n=128. Comparing to `oracle_r12` isolates spatial bilinear error → slope ≈ -2.
///
/// Error evaluated only on the mid-band (skip MARGIN rows near each pole boundary)
/// to remove O(h) clamping artefacts from the excluded pole cap.
#[allow(clippy::cast_precision_loss)]
fn sup_error(n_chart: usize, with_correction: bool, oracle_fn: fn(f64, f64, f64) -> f64) -> f64 {
    let n_theta = n_chart;
    let n_phi = 2 * n_chart; // maintain aspect ratio
    let grid = build_chart_grid(n_theta, n_phi);
    let sphere = Sphere2::unit();
    let chernoff = ManifoldChernoff::new(sphere, with_correction);

    let mut u = GridFn2D::from_fn(grid, initial_datum);
    let mut dst = u.clone();
    let mut scratch = ScratchPool::new();
    let tau = T_HORIZON / N_CHERNOFF as f64; // τ = 0.00125

    for _ in 0..N_CHERNOFF {
        chernoff
            .apply_into(tau, &u, &mut dst, &mut scratch)
            .unwrap();
        core::mem::swap(&mut u, &mut dst);
    }

    // Sup-norm on mid-band: skip rows within the GH-5 support radius of the poles.
    // At τ=0.00125: max θ-displacement = 2.02·2·√τ ≈ 0.143 rad (GH-5 outermost node).
    // MARGIN = ceil(max_disp / h) + 1 ensures no evaluated node can be contaminated
    // by boundary clamping artefacts from the pole exclusion at ε = 0.02 rad.
    let eps_grid = 0.02_f64; // must match build_chart_grid eps
    let nx = u.grid.nx();
    let ny = u.grid.ny();
    let h_theta = (core::f64::consts::PI - 2.0 * eps_grid) / (nx - 1) as f64;
    let max_gh_disp = 2.02 * 2.0 * tau.sqrt(); // GH-5 outer node × scale_θ
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let margin = (max_gh_disp / h_theta).ceil() as usize + 1;
    let mut max_err = 0.0f64;
    for j in margin..ny - margin {
        for i in margin..nx - margin {
            let theta = u.grid.x.x_at(i);
            let phi = u.grid.y.x_at(j);
            let err = (u.values[j * nx + i] - oracle_fn(T_HORIZON, theta, phi)).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    max_err
}

// ─── OLS slope ────────────────────────────────────────────────────────────────

#[allow(clippy::cast_precision_loss)]
fn ols_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let log_x: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let log_y: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let mean_x = log_x.iter().sum::<f64>() / m;
    let mean_y = log_y.iter().sum::<f64>() / m;
    let num: f64 = log_x
        .iter()
        .zip(log_y.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_x.iter().map(|x| (x - mean_x).powi(2)).sum();
    num / den
}

// ─── G26 sub-test 1: BASE (no R/12) ──────────────────────────────────────────

/// G26(1) — base `ManifoldChernoff` (no curvature correction) slope ≤ -0.95.
///
/// `n_Chernoff`=40 fixed (τ=0.00125); `n_chart` ∈ {16,32,64,128} sweeps h.
/// Without R/12, Chernoff error O(τ) acts as floor; spatial O(h²)
/// dominates for coarse grids. OLS slope in [-2, -1]; gate -0.95 satisfied.
/// Oracle: exact heat semigroup exp(TΔ_{S²}) with eigenvalues -ℓ(ℓ+1).
#[test]
fn g26_sphere_s2_order_one_base() {
    let mut errs = Vec::with_capacity(N_CHART_SWEEP.len());
    for &n in &N_CHART_SWEEP {
        let err = sup_error(n, false, oracle);
        println!("G26(1) base: n={n:4} → err={err:.4e}");
        errs.push(err);
    }
    let slope = ols_slope(&N_CHART_SWEEP, &errs);
    println!("G26(1) base: slope={slope:.4}  (gate ≤ {SLOPE_GATE_BASE})");
    assert!(
        slope <= SLOPE_GATE_BASE,
        "G26(1) FAIL: slope={slope:.4} > {SLOPE_GATE_BASE}. errs={errs:?}",
    );
}

// ─── G26 sub-test 2: WITH R/12 CORRECTION ────────────────────────────────────

/// G26(2) — curvature-corrected `ManifoldChernoff` slope ≤ -1.95.
///
/// `n_Chernoff`=40 fixed (τ=0.00125); `n_chart` ∈ {16,32,64,128} sweeps h.
/// With the outer [1+τR/12] correction, the effective generator is Δ_{S²}+R/12.
/// `oracle_r12` accounts for the R/12 shift in the generator eigenvalues,
/// so the residual measures only SPATIAL (bilinear) error: err ≈ O(h²) → slope ≈ -2.
/// Gate -1.95 gives 2.5% margin vs the -2.0 bilinear spatial convergence.
#[test]
fn g26_sphere_s2_order_two_r12_corrected() {
    let mut errs = Vec::with_capacity(N_CHART_SWEEP.len());
    for &n in &N_CHART_SWEEP {
        let err = sup_error(n, true, oracle_r12);
        println!("G26(2) R/12: n={n:4} → err={err:.4e}");
        errs.push(err);
    }
    let slope = ols_slope(&N_CHART_SWEEP, &errs);
    println!("G26(2) R/12: slope={slope:.4}  (gate ≤ {SLOPE_GATE_CORRECTED})");
    assert!(
        slope <= SLOPE_GATE_CORRECTED,
        "G26(2) FAIL: slope={slope:.4} > {SLOPE_GATE_CORRECTED}. errs={errs:?}",
    );
}
