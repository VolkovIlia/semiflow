//! G23 — `KillingChernoff` Dirichlet convergence slope.
//!
//! Gate (properties.yaml, `RELEASE_BLOCKING`):
//!   Sweep N ∈ {8, 16, 32, 64} at T = 0.05, `n_steps` = 100 · N.
//!   OLS slope of `log(sup_err)` vs `log(N)` ≤ −0.95.
//!
//! ## Interpretation of "`n_steps` = 100"
//!
//! The spec phrase "`n_steps` = 100" is interpreted as 100 temporal sub-steps
//! per spatial mesh node, so `n_steps_total` = 100 · N. This keeps the
//! Gaussian-shift width `h0` = 2√(a · τ) = 2√(a · T / (100N)) sub-grid
//! (`h0` / `dx` < 1 for N ≥ 8 at T = 0.05, a = 0.5), entering the spatial
//! asymptotic regime where the killing commutator error is O(`dx`) = O(1/N).
//!
//! When `h0` > `dx` (coarser temporal resolution), the near-boundary error
//! saturates at O(`h0`) = O(√τ) (constant with N), masking spatial
//! convergence and giving positive OLS slope. The 100 × N schedule keeps
//! `h0`/`dx` ≤ 0.25 for all N in the sweep.
//!
//! ## Setup
//!
//! Operator: `KillingChernoff<DiffusionChernoff(a=0.5), BoxRegion([0.0], [1.0])>`.
//! Oracle:   `u(t, x)` = Σ_{k=1..8} `a_k` · sin(kπx) · exp(−(kπ)²·t/2).
//! IC:       `u(0, x)` = oracle at t=0 = Σ `a_k` sin(kπx).
//!
//! ## Mathematical basis
//!
//! Butko 2018 §3.2 — the killing commutator `[L, 𝟙_R]` contributes an
//! irreducible O(τ) residual per step (O(`dx`) globally when τ ∝ 1/N).
//! Inner `DiffusionChernoff` has order 2; post-multiply masking caps global
//! convergence at order 1 (slope ≈ −1.0; gate at −0.95 gives 5% margin).
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]

use core::f64::consts::PI;

use semiflow::{
    killing::{BoxRegion, KillingChernoff},
    BoundaryPolicy, ChernoffFunction, DiffusionChernoff, Grid1D, GridFn1D, InterpKind,
    ScratchPool,
};

// ---------------------------------------------------------------------------
// Gate constants (normative — do NOT relax without ADR + properties.yaml bump)
// ---------------------------------------------------------------------------

const SLOPE_GATE: f64 = -0.95;

/// Spatial grid sizes to sweep (N ∈ {8, 16, 32, 64}).
/// These values satisfy `h0`/`dx` < 1 at `n_steps` = 100·N (sub-grid regime).
const N_SPATIAL: [usize; 4] = [8, 16, 32, 64];
/// Temporal sub-steps per spatial node: `n_steps_total` = `N_STEPS_PER_NODE` · N.
/// τ = T / (`N_STEPS_PER_NODE` · N) → decreases with N.
const N_STEPS_PER_NODE: usize = 100;
/// Fixed final time.
const T_FINAL: f64 = 0.05;
/// Diffusion coefficient a(x) ≡ 0.5 (heat equation ∂_t u = ½ ∂_xx u).
const A_COEF: f64 = 0.5;

// Deterministic amplitudes for the 8-mode eigenmode initial datum.
// Chosen to give a smooth, non-trivial IC with well-decayed higher modes.
// Amplitudes from a fixed pseudo-random sequence (not randomised — test is deterministic).
const AMPS: [f64; 8] = [0.5, -0.3, 0.2, 0.15, -0.1, 0.08, -0.05, 0.04];

// ---------------------------------------------------------------------------
// Oracle: eigenmode expansion u(t, x) = Σ_{k=1..8} a_k sin(kπx) exp(-(kπ)²t/2)
// ---------------------------------------------------------------------------

/// Exact solution for the absorbing-boundary heat equation on (0,1).
///
/// `u(0, x)` = Σ `a_k` sin(kπx); `u(t, 0)` = `u(t, 1)` = 0 for all t.
// k ∈ 1..=8 (loop bound 8) — within i32 and f64 mantissa; casts are safe.
#[allow(clippy::cast_precision_loss)]
fn oracle(t: f64, x: f64) -> f64 {
    AMPS.iter()
        .enumerate()
        .map(|(i, &a)| {
            let k = (i + 1) as f64;
            let eigenvalue = -(k * PI).powi(2) * A_COEF * t;
            a * (k * PI * x).sin() * eigenvalue.exp()
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Operator setup and single-run error
// ---------------------------------------------------------------------------

/// Evolve the killing operator for `n_steps_per_node * n` temporal steps
/// and return sup-norm error vs oracle.
///
/// τ = T / `n_steps_total` = T / (`N_STEPS_PER_NODE` · n), which ensures
/// h0 = 2√(a · τ) < dx = 1/(n-1) (sub-grid regime) for n ≥ 8.
// n ≤ 64, n_steps ≤ 6400 — well within f64 mantissa range; casts are safe.
#[allow(clippy::cast_precision_loss)]
fn sup_error_at(n: usize) -> f64 {
    let n_steps = N_STEPS_PER_NODE * n;
    let tau = T_FINAL / n_steps as f64;

    // Grid [0, 1] with ZeroExtend (Dirichlet-like stencil BC; killing handles
    // the operator-level absorbing BC via post-multiply masking).
    //
    // CubicHermite is pinned explicitly (not relying on the default). The
    // default changed CubicHermite → SepticHermite in v6.0 (ADR-0109). With
    // ZeroExtend BC and coarse N ∈ {8, 16, 32, 64}, SepticHermite's ±4-node
    // FD stencils access ghost nodes (ZeroExtend returns 0) at every
    // boundary-adjacent node, producing a non-convergent interpolation error
    // floor that masks the O(dx) killing commutator slope under test. CubicHermite
    // (Catmull-Rom) needs only ±1 node beyond the sample cell; for node i=1 the
    // leftward neighbour i=0 is a valid domain node, so no ghost contamination
    // occurs at the coarse sweep sizes used here.
    let grid = Grid1D::new(0.0_f64, 1.0, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::ZeroExtend)
        .with_interp(InterpKind::CubicHermite);

    // Inner: DiffusionChernoff(a=0.5, a'=0, a''=0)
    let inner = DiffusionChernoff::new(|_| A_COEF, |_| 0.0_f64, |_| 0.0_f64, A_COEF, grid);

    // Region: BoxRegion [0.0, 1.0) — open-R convention; x=1.0 is excluded (zeroed).
    // Together with eigenmode IC (which has u(0)=0 at x=0 by sin(kπ·0)=0 for all k),
    // this implements absorbing BC: u(t, x) = 0 for x ≤ 0 or x ≥ 1.
    let region = BoxRegion::<f64, 1>::new([0.0_f64], [1.0_f64]).unwrap();
    let killing = KillingChernoff::new(inner, region).unwrap();

    // Initial datum from the oracle at t=0.
    let mut u = GridFn1D::from_fn(grid, |x| oracle(0.0, x));
    let mut scratch = ScratchPool::new();

    // Evolve n_steps steps.
    for _ in 0..n_steps {
        let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
        killing.apply_into(tau, &u, &mut dst, &mut scratch).unwrap();
        u = dst;
    }

    // Compute sup-norm error vs exact oracle on interior nodes.
    (1..n - 1)
        .map(|i| {
            let x = grid.x_at(i);
            (u.values[i] - oracle(T_FINAL, x)).abs()
        })
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// OLS log-log slope: log(err) vs log(N)
// ---------------------------------------------------------------------------

// N ≤ 64 — well within f64 52-bit mantissa; cast is exact.
#[allow(clippy::cast_precision_loss)]
fn ols_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let log_n: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let log_e: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let mean_x = log_n.iter().sum::<f64>() / m;
    let mean_y = log_e.iter().sum::<f64>() / m;
    let num: f64 = log_n
        .iter()
        .zip(log_e.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_n.iter().map(|x| (x - mean_x).powi(2)).sum();
    num / den
}

// ---------------------------------------------------------------------------
// G23 gate
// ---------------------------------------------------------------------------

/// G23 — `KillingChernoff` Dirichlet convergence slope ≤ −0.95.
///
/// Verifies that `KillingChernoff<DiffusionChernoff, BoxRegion>` approximates the
/// absorbing-boundary heat semigroup with first-order convergence (slope ≈ −1).
///
/// N is swept as the spatial grid size; `n_steps` = 100 · N ensures the Chernoff
/// step width `h0` = 2√(a · τ) stays sub-grid (`h0` < `dx`). In this regime the
/// killing commutator error is O(`dx`) = O(1/N), giving slope ≈ −1.0.
/// The −0.95 gate provides 5% margin against the Butko 2018 §3.2 asymptote −1.0.
#[test]
#[allow(clippy::cast_precision_loss)]
fn g23_dirichlet_killing_slope() {
    let mut errs = Vec::with_capacity(N_SPATIAL.len());
    for &n in &N_SPATIAL {
        let e = sup_error_at(n);
        let tau = T_FINAL / (N_STEPS_PER_NODE * n) as f64;
        let h0 = 2.0 * (A_COEF * tau).sqrt();
        let dx = 1.0 / (n as f64 - 1.0);
        println!(
            "G23: N={n:4}, n_steps={:6}, tau={tau:.3e}, h0/dx={:.3}, sup_err={e:.4e}",
            N_STEPS_PER_NODE * n,
            h0 / dx
        );
        errs.push(e);
    }

    let slope = ols_slope(&N_SPATIAL, &errs);
    println!("G23: slope = {slope:.4}  (gate <= {SLOPE_GATE})");

    assert!(
        slope <= SLOPE_GATE,
        "G23 FAIL: slope {slope:.4} > gate {SLOPE_GATE}. \
         KillingChernoff Dirichlet convergence is sub-order-1. \
         Check DiffusionChernoff a=0.5, BoxRegion [0,1), T={T_FINAL}, \
         N_STEPS_PER_NODE={N_STEPS_PER_NODE}, N sweep={N_SPATIAL:?}.",
    );
}
