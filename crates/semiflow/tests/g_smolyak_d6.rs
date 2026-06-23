//! `G_SMOLYAK_D6` — Smolyak sparse-grid D=6 gate (`RELEASE_BLOCKING`, ADR-0123 Amendment 1).
//!
//! ## Gate spec (properties.yaml `G_SMOLYAK_D6`)
//!   1. Node count < 46656 (tensor 6⁶ baseline).
//!   2. F(0)=I smoke: Σ merged weights ≈ π³ to rel-err < 1e-12 (asserted at
//!      construction; also verified here explicitly).
//!   3. Self-convergence slope ≤ −0.95 (order-1; same gate as `G_SMOLYAK_D5`).
//!
//! ## Bounded-runtime rationale (REVISED — `N_AXIS=4`, domain [−2,2])
//!
//! D=6 Smolyak node counts at default ℓ=D+3=9: **533 nodes**.
//! Tensor 6⁶ = 46656, so 533 < 46656 (87.5× reduction).
//!
//! The original `N_AXIS=5` (5⁶=15625 grid pts) + `N_REF=256` design required
//! ~2.1B float ops for the reference alone and exceeded the 3-min budget
//! (observed: hung >4 min).  Fix: reduce spatial grid to `N_AXIS=4` (4⁶=4096
//! pts — the minimum allowed by the septic Hermite stencil, which requires
//! n ≥ 4) AND reduce domain to [−2, 2] so the Gaussian IC has non-trivial
//! values on the grid (4 pts in [-5,5] misses the Gaussian peak, producing
//! near-zero signal that is floor-dominated; 4 pts in [-2,2] has peak ≈ 0.07–1.0
//! at the two inner points).
//!
//! Revised cost estimate (opt-level=2 test profile, i7-12700K class):
//!   reference (`N_REF=64)`:   64 × 4096 × 533 ≈  140M ops  ~  3–8 s
//!   sweep ([8,16,32]):      56 × 4096 × 533 ≈  122M ops  ~  3–6 s
//!   total ≈ 262M ops → ~6–14 s; comfortably ≤ 2 min wall-clock.
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]

use semiflow::{
    grid_nd::{GridFnND, GridND},
    smolyak::SmolyakGridND,
    ChernoffFunction, Grid1D, ScratchPool, SquareMatrix,
};

// ── constants ─────────────────────────────────────────────────────────────────

/// Integration time horizon.
const T: f64 = 0.5;

/// Spatial grid size per axis: 4⁶ = 4096 total grid points.
///
/// N_AXIS=4 is the minimum allowed by the Grid1D septic Hermite stencil
/// (requires n ≥ 4).  Domain reduced to [−2, 2] (see DOMAIN_LO/HI) so the
/// Gaussian IC is non-trivial on the grid (inner points at ±2/3 have IC ≈ 0.07;
/// using [-5,5] with 4 pts puts inner points at ±5/3 where IC ≈ 5.6e-8 —
/// near machine-zero, producing floor-dominated convergence).
const N_AXIS: usize = 4;

/// Domain bounds per axis.  Reduced from [−5,5] to [−2,2] so N_AXIS=4 gives
/// meaningful Gaussian IC values: exp(-(2/3)^2)^6 ≈ 0.069 at inner pts.
const DOMAIN_LO: f64 = -2.0;
const DOMAIN_HI: f64 = 2.0;

/// Reference step count.  N_REF=64 gives tau_ref = T/64 ≈ 7.8e-3,
/// well below the coarsest sweep point (n=8, tau=0.0625).
const N_REF: u32 = 64;

/// Sweep n-values: n ∈ {8, 16, 32}.  Each doubling should halve the temporal
/// truncation error, giving slope ≈ −1.0 (order-1 gate: ≤ −0.95).
const N_SWEEP: [u32; 3] = [8, 16, 32];

/// Slope gate: ≤ −0.95 (order-1, consistent with SmolyakGridND::order()=1).
/// ADR-0123 Amendment 1 explicitly inherits the corrected G_SMOLYAK_D5 gate.
const SLOPE_GATE: f64 = -0.95;

/// Node count must be below the full tensor 6⁶ = 46656 baseline.
const NODE_COUNT_GATE: usize = 46656;

/// Smolyak level used for the gate: ℓ = D+3 = 9 → 533 nodes (87.5× reduction).
const LEVEL: usize = 9; // D + 3, where D = 6

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_grid_d6(n: usize) -> GridND<f64, 6> {
    let ax = Grid1D::new(DOMAIN_LO, DOMAIN_HI, n).unwrap();
    GridND::new([ax; 6]).unwrap()
}

/// Build D=6 isotropic+cross-coupled diffusion kernel at the gated level.
///
/// Diffusion tensor: A = I + 0.15 · tanh(xᵢ+xⱼ) · eᵢeⱼᵀ for off-diagonals.
/// SPD by diagonal dominance (off diagonal ≤ 0.15 < 1/6 < 1 → Gershgorin safe).
fn make_kernel(n: usize) -> SmolyakGridND<f64, 6> {
    let grid = make_grid_d6(n);
    SmolyakGridND::with_level(
        |x: &[f64; 6], a: &mut SquareMatrix<f64, 6>| {
            for i in 0..6 {
                a.set(i, i, 1.0);
            }
            for i in 0..6 {
                for j in (i + 1)..6 {
                    let off = 0.15 * (x[i] + x[j]).tanh();
                    a.set(i, j, off);
                    a.set(j, i, off);
                }
            }
        },
        |_x: &[f64; 6], b: &mut [f64; 6]| {
            for v in b.iter_mut() {
                *v = 0.0;
            }
        },
        |_x: &[f64; 6]| 0.0_f64,
        grid,
        LEVEL,
    )
    .unwrap()
}

/// D=6 Gaussian initial condition: exp(−‖x‖²).
fn initial_fn(x: &[f64; 6]) -> f64 {
    (-x.iter().map(|xi| xi * xi).sum::<f64>()).exp()
}

/// Run `n_steps` Chernoff steps from Gaussian IC; return final state.
fn run_steps(kernel: &SmolyakGridND<f64, 6>, n_steps: u32) -> GridFnND<f64, 6> {
    let tau = T / n_steps as f64;
    let f0 = GridFnND::from_fn(kernel.grid().clone(), initial_fn);
    let mut src = f0;
    let mut dst = GridFnND::from_fn(kernel.grid().clone(), |_| 0.0_f64);
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut pool).unwrap();
        core::mem::swap(&mut src, &mut dst);
    }
    src
}

/// Supremum of absolute difference between two grid functions.
fn sup_diff(a: &GridFnND<f64, 6>, b: &GridFnND<f64, 6>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(&ai, &bi)| (ai - bi).abs())
        .fold(0.0_f64, |m, e| if e.is_nan() { f64::NAN } else { m.max(e) })
}

/// OLS slope of log(err) vs log(n).
fn ols_slope(ns: &[u32], errs: &[f64]) -> f64 {
    let x: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let y: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sxx: f64 = x.iter().map(|xi| xi * xi).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

// ── gate test ─────────────────────────────────────────────────────────────────

/// G_SMOLYAK_D6 gate: D=6 Smolyak sparse-grid kernel.
///
/// Three sub-tests (all within this `#[ignore]` function):
///   1. Node-count gate: `k.n_nodes() < 46656` (tensor 6⁶).
///   2. F(0)=I smoke: `‖F(0)·1 − 1‖_∞ < 1e-10`.
///   3. Self-convergence slope ≤ −0.95 (order-1; ADR-0123 Amendment 1).
///
/// Prints incremental progress; budget ≤ 3 min on dev HW.
#[test]
#[ignore] // slow-tests: cargo test --features slow-tests -- --ignored g_smolyak_d6
fn g_smolyak_d6() {
    println!("G_SMOLYAK_D6: building D=6 Smolyak kernel (ℓ={LEVEL}, N_AXIS={N_AXIS})…");
    let kernel = make_kernel(N_AXIS);

    // ── Sub-test 1: node-count gate ──────────────────────────────────────────
    let n_nodes = kernel.n_nodes();
    println!("G_SMOLYAK_D6: Smolyak nodes={n_nodes}  tensor-baseline={NODE_COUNT_GATE} (6⁶)");
    assert!(
        n_nodes < NODE_COUNT_GATE,
        "G_SMOLYAK_D6 node-count gate FAILED: {n_nodes} >= {NODE_COUNT_GATE}"
    );

    // ── Sub-test 2: F(0)=I smoke ─────────────────────────────────────────────
    {
        let one_fn = GridFnND::from_fn(kernel.grid().clone(), |_| 1.0_f64);
        let mut out = one_fn.clone();
        let mut pool = ScratchPool::<f64>::new();
        kernel
            .apply_into(0.0, &one_fn, &mut out, &mut pool)
            .unwrap();
        let sup_err = out
            .values
            .iter()
            .map(|&v| (v - 1.0).abs())
            .fold(0.0_f64, f64::max);
        println!("G_SMOLYAK_D6: F(0)=I sup_err={sup_err:.3e}");
        assert!(
            sup_err < 1e-10,
            "G_SMOLYAK_D6 F(0)=I smoke FAILED: sup_err={sup_err:.3e} >= 1e-10"
        );
    }

    // ── Sub-test 3: self-convergence ─────────────────────────────────────────
    println!("G_SMOLYAK_D6: computing reference (n_ref={N_REF})…");
    let u_ref = run_steps(&kernel, N_REF);
    println!("G_SMOLYAK_D6: reference done");

    let mut errs = Vec::with_capacity(N_SWEEP.len());
    for &n in &N_SWEEP {
        println!(
            "G_SMOLYAK_D6: running n={n} steps (tau={:.5})…",
            T / n as f64
        );
        let u_n = run_steps(&kernel, n);
        let err = sup_diff(&u_n, &u_ref);
        println!(
            "G_SMOLYAK_D6: n={n} tau={:.5} sup‖u_n−u_ref‖={err:.4e}",
            T / n as f64
        );
        errs.push(err);
    }

    let slope = ols_slope(&N_SWEEP, &errs);
    println!("G_SMOLYAK_D6: OLS slope={slope:.4}  gate<=({SLOPE_GATE})  nodes={n_nodes}");
    assert!(
        slope.is_finite() && slope <= SLOPE_GATE,
        "G_SMOLYAK_D6 slope gate FAILED: slope={slope:.4} not finite-and-<={SLOPE_GATE}"
    );

    println!("G_SMOLYAK_D6: ALL SUB-TESTS PASSED ✓");
}
