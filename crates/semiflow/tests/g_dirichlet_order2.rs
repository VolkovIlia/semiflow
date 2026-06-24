//! `G_DIRICHLET_ORDER2` — `DirichletHeat2ndChernoff` order-2 convergence gate.
//!
//! Gate (ADR-0176, `RELEASE_BLOCKING`):
//!   OLS slope of `log(sup_err)` vs `log(n)` ≤ −1.95.
//!
//! ## Method: oracle vs Dirichlet eigenmode expansion on (0, 1)
//!
//! Operator: `DirichletHeat2ndChernoff` wrapping `DiffusionChernoff(a=0.5)`.
//! Equation: ∂_t u = ½ ∂_xx u, with hard absorbing wall u|_{x∈{0,1}} = 0.
//!
//! Oracle: u(t, x) = Σ_{k=1}^{8} `a_k` sin(kπx) exp(−(kπ)² t/2)
//!   Each mode is an exact Dirichlet eigenfunction of ½∂_xx on (0,1):
//!   ½ ∂_xx sin(kπx) = −½(kπ)² sin(kπx),  sin(kπ·0) = sin(kπ·1) = 0.
//!
//! IC: `u_0(x)` = Σ_{k=1}^{8} `a_k` sin(kπx)   (oracle at t=0).
//!
//! ## Convergence design
//!
//! Sweep n ∈ {16, 32, 64, 128} with τ = T/n (one Chernoff step per n).
//! `DiffusionChernoff` is order-2; `DirichletHeat2ndChernoff` inherits it
//! (Proposition 21.9.1 — the odd-image commutator vanishes). Global error
//! scales as O(τ²) = O(1/n²) → log-log slope ≈ −2.0. Gate at −1.95 gives
//! 2.5 % margin against the asymptote.
//!
//! ## Grid + stencil note
//!
//! The wrapper internally sets `BoundaryPolicy::OddReflect` on the input grid,
//! which gives the stencil antisymmetric ghosts at both x=0 and x=1. This
//! enforces u=0 at both walls via the odd-extension mechanism (21.9.3).
//! Grid boundary is set to `ZeroExtend` before passing into the wrapper so
//! that the IC sampling is well-defined at boundary nodes (eigenmode IC has
//! u₀(0)=u₀(1)=0 by construction, so `ZeroExtend` produces the same IC values
//! as the analytical formula).
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // n, n_steps ≤ 128, well within 2^52
#![allow(clippy::doc_markdown)] // math notation in doc comments

use core::f64::consts::PI;

use semiflow::{
    killing_order2::DirichletHeat2ndChernoff, reflection::HalfSpaceRegion, BoundaryPolicy,
    ChernoffFunction, DiffusionChernoff, Grid1D, GridFn1D, InterpKind, ScratchPool,
};

// ---------------------------------------------------------------------------
// Gate constants (normative — do NOT relax without ADR + properties.yaml bump)
// ---------------------------------------------------------------------------

/// OLS slope gate — order-2 (gate at −1.95 gives 2.5% margin vs asymptote −2.0).
const SLOPE_GATE: f64 = -1.95;

/// Temporal-steps sweep (n Chernoff steps, τ = T/n).
const N_SWEEP: [usize; 4] = [16, 32, 64, 128];

/// Fixed final time.
const T_FINAL: f64 = 0.05;

/// Diffusion coefficient a(x) ≡ 0.5 (heat equation ∂_t u = ½ ∂_xx u).
const A_COEF: f64 = 0.5;

/// Spatial resolution multiplier: N_SPATIAL = SPATIAL_MULT * n.
/// Large enough that spatial error (O(dx^p) with high-order stencil) is
/// sub-dominant to the temporal O(τ²) error being measured.
const SPATIAL_MULT: usize = 8;

// Deterministic amplitudes for the 8-mode eigenmode initial datum.
// Same as G23 for comparability; a_k = AMPS[k-1].
const AMPS: [f64; 8] = [0.5, -0.3, 0.2, 0.15, -0.1, 0.08, -0.05, 0.04];

// ---------------------------------------------------------------------------
// Oracle: u(t, x) = Σ_{k=1..8} a_k sin(kπx) exp(−(kπ)² t/2)
// ---------------------------------------------------------------------------

/// Exact Dirichlet absorbing-boundary solution on (0, 1), operator ½∂_xx.
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

/// Evolve the DirichletHeat2ndChernoff wrapper for `n_steps` steps on a spatial
/// grid of `n_spatial` nodes and return sup-norm error vs the oracle.
fn sup_error_at(n_steps: usize, n_spatial: usize) -> f64 {
    let tau = T_FINAL / n_steps as f64;

    // Grid [0, 1] with CubicHermite + ZeroExtend (IC is zero at boundaries
    // by eigenmode construction; wrapper overrides boundary with OddReflect).
    // CubicHermite pinned explicitly (same rationale as G23: ±1-node stencil,
    // no ghost contamination at coarse N used in the sweep).
    let grid = Grid1D::new(0.0_f64, 1.0, n_spatial)
        .unwrap()
        .with_boundary(BoundaryPolicy::ZeroExtend)
        .with_interp(InterpKind::CubicHermite);

    // Inner: DiffusionChernoff(a=0.5, a'=0, a''=0).
    let inner = DiffusionChernoff::new(|_| A_COEF, |_| 0.0_f64, |_| 0.0_f64, A_COEF, grid);

    // Half-space region: origin=[0.0], normal=[1.0] (left wall at x=0).
    // The wrapper sets OddReflect globally on both ends of [0, 1], enforcing
    // u=0 at x=0 (via antisymmetric ghost) and u=0 at x=1 (same mechanism).
    let region = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0]).unwrap();
    let wrapper = DirichletHeat2ndChernoff::new(inner, region).unwrap();

    // Initial datum from the oracle at t=0.
    let mut u = GridFn1D::from_fn(grid, |x| oracle(0.0, x));
    let mut scratch = ScratchPool::new();

    // Evolve n_steps steps.
    for _ in 0..n_steps {
        let mut dst = GridFn1D::from_fn(grid, |_| 0.0_f64);
        wrapper.apply_into(tau, &u, &mut dst, &mut scratch).unwrap();
        u = dst;
    }

    // Compute sup-norm error vs exact oracle on interior nodes only.
    (1..n_spatial - 1)
        .map(|i| {
            let x = grid.x_at(i);
            (u.values[i] - oracle(T_FINAL, x)).abs()
        })
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// OLS log-log slope: log(err) vs log(n)
// ---------------------------------------------------------------------------

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
// G_DIRICHLET_ORDER2 gate
// ---------------------------------------------------------------------------

/// G_DIRICHLET_ORDER2 — `DirichletHeat2ndChernoff` order-2 Dirichlet convergence slope.
///
/// Verifies Proposition 21.9.1 (ADR-0176): the odd-image wrapper inherits the
/// inner Chernoff order. With `DiffusionChernoff` (order 2), the global slope must
/// be ≤ −1.95, giving 2.5% margin vs the theoretical −2.0.
///
/// The SLOPE_GATE = −1.95 (order-2 assurance) contrasts with G23's −0.95 (order-1
/// for `KillingChernoff`). If this test passes, the odd-image construction
/// genuinely overcomes the order-1 cap of the killing formulation (§21.2).
#[test]
#[ignore]
fn g_dirichlet_order2_slope() {
    let mut errs = Vec::with_capacity(N_SWEEP.len());

    for &n in &N_SWEEP {
        let n_spatial = SPATIAL_MULT * n;
        let tau = T_FINAL / n as f64;
        let e = sup_error_at(n, n_spatial);
        println!(
            "G_DIRICHLET_ORDER2: n={n:4}, n_spatial={n_spatial:4}, tau={tau:.3e}, sup_err={e:.4e}"
        );
        errs.push(e);
    }

    let slope = ols_slope(&N_SWEEP, &errs);
    println!("G_DIRICHLET_ORDER2: slope = {slope:.4}  (gate <= {SLOPE_GATE})");

    assert!(
        slope <= SLOPE_GATE,
        "G_DIRICHLET_ORDER2 FAIL: slope {slope:.4} > gate {SLOPE_GATE}. \
         DirichletHeat2ndChernoff is NOT achieving order-2 convergence. \
         Check odd-image BoundaryPolicy::OddReflect implementation. \
         T={T_FINAL}, a={A_COEF}, n_sweep={N_SWEEP:?}.",
    );

    println!("G_DIRICHLET_ORDER2 PASS");
}
