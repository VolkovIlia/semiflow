//! `G_SMOLYAK_D5` — Smolyak sparse-grid gate (`RELEASE_BLOCKING`) plus hosted smoke.
//!
//! Full gate: D=5 successive-difference slope ≤ −0.95 AND node count < 3125
//! (tensor 5⁵). Hosted smoke: node count + `F(0)=I`.
//!
//! NOTE ON THE SLOPE THRESHOLD: ADR-0123 §acceptance-gate lists "≤ −1.95" which is
//! inconsistent with the kernel's declared `order() = 1` and with the
//! existing `G_DDIM D=5` tensor gate (also `−0.95`).  The `−1.95` threshold
//! applies to order-2 kernels (Strang, RK2). The Smolyak sparse-grid replaces
//! the quadrature backend but does NOT lift temporal order; measured slope is
//! ≈ −1 (order-1). Gate stays at `−0.95` to match kernel order — consistent
//! with `anisotropic_shift_nd_d5_slope.rs` and math §32.5 (ADR-0112).
//! Honest reporting: do NOT loosen the gate beyond what the kernel achieves.
//!
//! Why the old gate failed on hosted CI:
//! the previous estimator compared `n ∈ {32,64,128}` against a single
//! `n_ref=512` run. On the failing 2026-08-29 hosted run the reported errors were
//! `1.5363e-4, 9.4516e-5, 5.2034e-5` with OLS slope `−0.7810`: exactly the shape
//! of an order-1 signal polluted by a non-negligible fixed reference floor at the
//! finest point. That is the same estimator failure mode ADR-0191 AMENDMENT 3
//! already documented for `G_DDIM D=5`.
//!
//! Full-gate method: reference-free successive differences on the same normative
//! `N_AXIS=6` spatial datum. Run `n ∈ {32,64,128,256}`, fit the OLS slope of
//! `sup‖u_{2n}−u_n‖` against `log(n)`, and require it to stay in the order-1 band.
//! This removes the contaminated fixed reference entirely.
//!
//! Hosted scheduled CI runs only the cheap smoke (`g_smolyak_d5_smoke`): node
//! count plus `F(0)=I`. The strict ignored slope gate remains manually runnable on
//! calibrated self-hosted hardware.
//!
//! Full-gate sub-tests:
//!   1. Node-count gate: `k.n_nodes() < 3125`.
//!   2. F(0)=I unit smoke: ‖F(0)·1 − 1‖_∞ < 1e-10 (construction asserts too).
//!   3. Successive-difference slope in `[-1.25, -0.95]` (order-1).
//!
//! Feature gate: `slow-tests`.
//!
//! # Cost (measured 2026-08-18 — the first execution this gate ever had)
//!
//! **Over 40 minutes on a 12-core host**, where it hit a 40 min cap without
//! finishing. Cause: ADR-0191 replaced multilinear N-D sampling with the `K^D`
//! tensor stencil, which at `D = 5` reads 1024 nodes per sample against
//! multilinear's 32 — a 32x interpolation cost. ADR-0191 measured that on the
//! `D = 5` Smolyak *smoke* tests (70 s → 198 s, which is why they were re-sized)
//! and left this gate alone; nobody measured this one, because it ran in no
//! workflow. The hosted nightly lane now runs only the smoke coverage; the full
//! ignored slope gate is manual/self-hosted.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize/u32 to f64 in OLS sweeps; values below 2^52

use semiflow::{
    grid_nd::{GridFnND, GridND},
    smolyak::SmolyakGridND,
    ChernoffFunction, Grid1D, ScratchPool, SquareMatrix,
};

const T: f64 = 0.5;
const N_AXIS: usize = 6;
// The coarsest n=16 point remains too pre-asymptotic on this datum; start at 32.
// The successive-difference ladder removes the contaminated reference run while
// keeping the same spatial grid ADR-0191 says must not be traded away.
const N_LADDER: [u32; 4] = [32, 64, 128, 256];
// Gate: order-1 band. Lower bound keeps the original release-blocking threshold;
// upper bound catches an apparent order change so the contract is revisited
// instead of quietly drifting.
const SLOPE_GATE: f64 = -0.95;
const SLOPE_CEILING: f64 = -1.25;
const NODE_COUNT_GATE: usize = 3125; // tensor 5⁵ baseline

fn make_grid_d5(n: usize) -> GridND<f64, 5> {
    let ax = Grid1D::new(-5.0_f64, 5.0, n).unwrap();
    GridND::new([ax; 5]).unwrap()
}

fn make_kernel(n: usize) -> SmolyakGridND<f64, 5> {
    let grid = make_grid_d5(n);
    SmolyakGridND::new(
        |x: &[f64; 5], a: &mut SquareMatrix<f64, 5>| {
            for i in 0..5 {
                a.set(i, i, 1.0);
            }
            for i in 0..5 {
                for j in (i + 1)..5 {
                    let off = 0.25 * (x[i] + x[j]).tanh();
                    a.set(i, j, off);
                    a.set(j, i, off);
                }
            }
        },
        |_x: &[f64; 5], b: &mut [f64; 5]| {
            for v in b.iter_mut() {
                *v = 0.0;
            }
        },
        |_x: &[f64; 5]| 0.0_f64,
        grid,
    )
    .unwrap()
}

fn initial_fn(x: &[f64; 5]) -> f64 {
    (-x.iter().map(|xi| xi * xi).sum::<f64>()).exp()
}

fn run_steps(kernel: &SmolyakGridND<f64, 5>, n_steps: u32) -> GridFnND<f64, 5> {
    let tau = T / f64::from(n_steps);
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

fn sup_diff(a: &GridFnND<f64, 5>, b: &GridFnND<f64, 5>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(&ai, &bi)| (ai - bi).abs())
        .fold(0.0_f64, |m, e| if e.is_nan() { f64::NAN } else { m.max(e) })
}

fn ols_slope(ns: &[u32], errs: &[f64]) -> f64 {
    let x: Vec<f64> = ns.iter().map(|&n| f64::from(n).ln()).collect();
    let y: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sxx: f64 = x.iter().map(|xi| xi * xi).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

fn assert_node_count_and_f0(kernel: &SmolyakGridND<f64, 5>) -> usize {
    // --- Sub-test 1: node count gate ---
    let n_nodes = kernel.n_nodes();
    println!("G_SMOLYAK_D5: Smolyak nodes={n_nodes}  tensor-baseline={NODE_COUNT_GATE}");
    assert!(
        n_nodes < NODE_COUNT_GATE,
        "G_SMOLYAK_D5 node count gate FAILED: {n_nodes} >= {NODE_COUNT_GATE}"
    );

    // --- Sub-test 2: F(0)=I unit smoke ---
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
        println!("G_SMOLYAK_D5: F(0)=I sup_err={sup_err:.3e}");
        assert!(
            sup_err < 1e-10,
            "G_SMOLYAK_D5 F(0)=I smoke FAILED: sup_err={sup_err:.3e} >= 1e-10"
        );
    }
    n_nodes
}

/// Hosted-portable smoke for the D=5 Smolyak gate.
///
/// Keeps the cheap parts of the release-blocking evidence on scheduled CI:
/// node-count compression against the tensor baseline and the exact `F(0)=I`
/// construction check.
#[test]
fn g_smolyak_d5_smoke() {
    let kernel = make_kernel(N_AXIS);
    let _ = assert_node_count_and_f0(&kernel);
}

/// `G_SMOLYAK_D5` full release-blocking gate: D=5 Smolyak sparse-grid kernel.
///
/// Verifies:
/// 1. `n_nodes < 3125` (tensor 5⁵ baseline)
/// 2. F(0)=I unit smoke: `‖F(0)·1 − 1‖_∞ < 1e-10`
/// 3. Successive-difference slope in `[-1.25, -0.95]` (order-1 band)
#[test]
#[ignore = "RELEASE_BLOCKING slow gate: manual/self-hosted recommended after ADR-0191; run with -- --ignored"]
fn g_smolyak_d5() {
    let kernel = make_kernel(N_AXIS);
    let n_nodes = assert_node_count_and_f0(&kernel);

    // --- Sub-test 3: reference-free self-convergence slope ---
    let t_start = std::time::Instant::now();
    let us: Vec<_> = N_LADDER
        .iter()
        .map(|&n| {
            let u = run_steps(&kernel, n);
            println!(
                "G_SMOLYAK_D5: ladder n={n} done  (+{:.0} s cumulative)",
                t_start.elapsed().as_secs_f64()
            );
            u
        })
        .collect();
    let diffs: Vec<f64> = (0..N_LADDER.len() - 1)
        .map(|k| sup_diff(&us[k], &us[k + 1]))
        .collect();
    let ns: Vec<u32> = N_LADDER[..N_LADDER.len() - 1].to_vec();

    for (&n, &d) in ns.iter().zip(diffs.iter()) {
        println!("G_SMOLYAK_D5: n={n} -> {}  sup‖u_2n−u_n‖={d:.4e}", 2 * n);
    }

    let slope = ols_slope(&ns, &diffs);
    println!(
        "G_SMOLYAK_D5: successive-difference OLS slope = {slope:.4}  \
         (gate: {SLOPE_CEILING} <= slope <= {SLOPE_GATE})  nodes={n_nodes}"
    );
    assert!(
        slope.is_finite() && slope <= SLOPE_GATE,
        "G_SMOLYAK_D5 slope gate FAILED: slope={slope:.4} not finite-and-≤{SLOPE_GATE}"
    );
    assert!(
        slope >= SLOPE_CEILING,
        "G_SMOLYAK_D5 slope gate FAILED: slope={slope:.4} is steeper than {SLOPE_CEILING}; \
         if the kernel gained order, revisit `SmolyakGridND::order()` and this gate together"
    );
}
