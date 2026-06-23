//! G25 — Howland Nonautonomous Lift slope (math.md §23.6, ADR-0070).
//!
//! Gate (properties.yaml v0.9.0 G25, `RELEASE_BLOCKING)`:
//!   Time-dependent heat ∂_t u = a(t) ∂_xx u, a(t) = 1 + 0.5·t.
//!   Closed-form oracle: u(t, x) = (1 + 4·A(t))^{-1/2} · exp(-x² / (1 + 4·A(t)))
//!   where A(t) = t + 0.25·t² = ∫₀^t (1 + 0.5·s) ds.
//!   (Factor 4 verified by `scripts/verify_howland_lift.py` `oracle_pde` sub-check.)
//!   Sweep `n_t` ∈ {32, 64, 128, 256}, T = 0.5, grid [-8, 8] N=256.
//!   OLS slope of log(err) vs `log(n_t)` ≤ -0.95 (order-1, 5% margin).
//!
//! Note on oracle constant: properties.yaml §23.6 says (1 + 2·A); sympy
//! script uses (1 + 4·A) which satisfies the full PDE ∂_t u = a(t) ∂_xx u.
//! We use (1 + 4·A) per sympy verification (`scripts/verify_howland_lift.py`).
//!
//! Note on iteration: `HowlandLift::apply_into` is ONE Chernoff step.
//! Applying it (`n_t` - 1) times from initial state [g, g, ..., g] cascades
//! the composition C(t_{n_t-2}) ∘ … ∘ `C(t_0)` into slot dst[n_t-1].
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]

use semiflow::{
    chernoff::Growth,
    howland::{HowlandLift, HowlandState, TimedChernoffFunction},
    BoundaryPolicy, ChernoffFunction, DiffusionChernoff, Grid1D, GridFn1D, SemiflowError,
    ScratchPool,
};

// ---------------------------------------------------------------------------
// Gate constants — do NOT relax without ADR + properties.yaml bump.
// ---------------------------------------------------------------------------

const SLOPE_GATE: f64 = -0.95;
const T_HORIZON: f64 = 0.5;
const N_T_SWEEP: [usize; 4] = [32, 64, 128, 256];

// ---------------------------------------------------------------------------
// TimedDiffusionChernoff — test-local adapter for time-dependent a(t)
//
// Wraps DiffusionChernoff for a(x) = 1 + 0.5·t (constant in x, linear in t).
// apply_at(t, …) re-constructs DiffusionChernoff with the sampled coefficient.
// ---------------------------------------------------------------------------

/// Test helper: DiffusionChernoff with a(x, t) = 1 + 0.5·t (autonomous bridge IC).
struct TimedDiffusionChernoff {
    grid: Grid1D<f64>,
}

impl ChernoffFunction<f64> for TimedDiffusionChernoff {
    type S = GridFn1D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        // Autonomous bridge: use a(x) = 1.0 (t=0 coefficient).
        let inner = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.5, self.grid);
        inner.apply_into(tau, src, dst, scratch)
    }

    fn order(&self) -> u32 {
        2
    }

    fn growth(&self) -> Growth<f64> {
        // a(x,t) ≤ 1 + 0.5·T_HORIZON = 1.25 in the sweep range.
        Growth::new(1.0, 1.5)
    }
}

impl TimedChernoffFunction<f64> for TimedDiffusionChernoff {
    fn apply_at(
        &self,
        t: f64,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError>
    where
        Self::S: Clone,
    {
        // a(t) = 1 + 0.5·t (constant in x at fixed t)
        let a_t = 1.0 + 0.5 * t;
        // Norm bound: a(t) ≤ 1 + 0.5·T_HORIZON ≤ 1.25 for T=0.5 sweep.
        let inner =
            DiffusionChernoff::with_closure(move |_| a_t, |_| 0.0_f64, |_| 0.0_f64, a_t, self.grid);
        inner.apply_into(tau, src, dst, scratch)
    }
}

// ---------------------------------------------------------------------------
// Analytical oracle
// ---------------------------------------------------------------------------

/// u(t, x) = (1 + 4·A(t))^{-1/2} · exp(-x² / (1 + 4·A(t)))
/// where A(t) = t + 0.25·t² = ∫₀^t (1 + 0.5·s) ds.
///
/// Satisfies ∂_t u = a(t) ∂_xx u with u(0, x) = exp(-x²).
/// Factor 4 verified by verify_howland_lift.py oracle_pde sub-check.
fn oracle(t: f64, x: f64) -> f64 {
    let a_t = t + 0.25 * t * t; // A(t) = ∫₀ᵗ (1 + 0.5s) ds
    let denom = 1.0 + 4.0 * a_t;
    (-x * x / denom).exp() / denom.sqrt()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let log_x: Vec<f64> = xs.iter().map(|&v| v.ln()).collect();
    let log_y: Vec<f64> = ys.iter().map(|&v| v.ln()).collect();
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

/// Run the Howland lift for a given n_t; return sup-norm error vs oracle at T.
#[allow(clippy::cast_precision_loss)]
fn sup_error_at(n_t: usize, grid: Grid1D<f64>) -> f64 {
    let inner = TimedDiffusionChernoff { grid };
    let lift = HowlandLift::new(inner, T_HORIZON, n_t).unwrap();
    let delta_s = lift.delta_s();

    // Initial state: all n_t slices hold u(0, x) = exp(-x²).
    let ic = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let samples: Vec<GridFn1D<f64>> = (0..n_t).map(|_| ic.clone()).collect();
    let mut src = HowlandState::new(samples).unwrap();
    let mut dst = src.clone();
    let mut scratch = ScratchPool::new();

    // Apply (n_t - 1) Howland steps; after k steps dst[k] = C(t_{k-1})∘…∘C(t_0) g.
    // After (n_t - 1) steps, dst[n_t - 1] holds the composition at t = T_HORIZON.
    for _ in 0..n_t - 1 {
        lift.apply_into(delta_s, &src, &mut dst, &mut scratch)
            .unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    // After n_t-1 swaps, src holds the last result.
    let last = &src.samples[n_t - 1];

    // Sup-norm error on interior nodes vs oracle at T_HORIZON.
    let n_grid = grid.n;
    (1..n_grid - 1)
        .map(|i| {
            let x = grid.x_at(i);
            (last.values[i] - oracle(T_HORIZON, x)).abs()
        })
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// G25 gate
// ---------------------------------------------------------------------------

/// G25 — Howland nonautonomous lift slope ≤ -0.95.
///
/// Time-dependent heat ∂_t u = (1 + 0.5t) ∂_xx u with Gaussian IC.
/// Iterates (n_t - 1) Howland steps and compares slot [n_t-1] to oracle.
/// OLS slope of log(err) vs log(n_t) must be ≤ -0.95 (order-1, 5% margin).
#[test]
fn g25_howland_lift_slope() {
    // Grid [-8, 8] with N=256, Reflect BC (wide domain: boundary error < 1e-6 at T=0.5).
    let grid = Grid1D::new(-8.0_f64, 8.0, 256)
        .unwrap()
        .with_boundary(BoundaryPolicy::Reflect);

    let mut errs: Vec<f64> = Vec::with_capacity(N_T_SWEEP.len());

    for &n_t in &N_T_SWEEP {
        let err = sup_error_at(n_t, grid);
        println!("G25: n_t={n_t:4}: err = {err:.4e}");
        errs.push(err);
    }

    let ns_f64: Vec<f64> = N_T_SWEEP.iter().map(|&n| n as f64).collect();
    let slope = ols_slope(&ns_f64, &errs);

    println!("G25: slope = {slope:.4}  (gate ≤ {SLOPE_GATE})");
    assert!(
        slope <= SLOPE_GATE,
        "G25 FAIL: slope {slope:.4} > {SLOPE_GATE}. \
         Check oracle factor (4 vs 2), TimedDiffusionChernoff::apply_at, \
         and iteration count (should be n_t - 1 steps). \
         Errors: {errs:?}",
    );
}
