//! G12 slope gate — `MagnusGraphHeatChernoff::evolve_with_traj` (order-4).
//!
//! Gate: OLS log-log slope ≤ −3.95 (f64).
//!
//! Path graph `P_64` with 3 segments, each with a different constant Laplacian
//! (piecewise-smooth weight). `n_steps_per_segment ∈ {10, 20, 40, 80, 160}`.
//! Self-convergence at 2× refinement.
//!
//! See ADR-0052 acceptance criterion and Wave 2.2A contract §5.1.
//!
//! These tests are gated behind `--ignored` / `slow-tests` feature.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use semiflow::{
    graph_traj::{GraphTraj, SegmentWeightFn},
    magnus_graph::{LaplacianAtTime, MagnusGraphHeatChernoff},
    state::HilbertState,
    ChernoffFunction, Graph, GraphSignal, Laplacian, ScratchPool,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const N_NODES: usize = 64;

/// Segment breakpoints: [0, 1/3, 2/3, 1].
/// Using `T_final` = 1.0 total horizon with 3 equal segments.
const T_FINAL: f64 = 0.3;

/// f64 sweep: 5 refinement levels.
const N_STEPS_PER_SEG_F64: [usize; 5] = [10, 20, 40, 80, 160];

/// f64 slope threshold per ADR-0052 / contract §5.1.
const SLOPE_THRESHOLD_F64: f64 = -3.95;

// ---------------------------------------------------------------------------
// Segment weight functions (3 segments, piecewise-constant Laplacians)
// ---------------------------------------------------------------------------

/// Build Laplacian for `P_N` with uniform edge weight `w`.
fn make_lap_uniform(n: usize, w: f64) -> Arc<Laplacian<f64>> {
    let edges = (0..n as u32 - 1).map(|i| (i, i + 1, w));
    let g = Graph::from_edges(n, edges).expect("valid path edges");
    Arc::new(Laplacian::assemble_combinatorial(&g))
}

/// Build the test trajectory: 3 segments on `P_N` with weights 1.0, 0.7, 1.2.
fn make_test_traj(n: usize) -> (GraphTraj<f64>, Arc<Graph<f64>>) {
    let g_ref = Arc::new(Graph::<f64>::path(n));
    let dt = T_FINAL / 3.0;
    let breakpoints = vec![0.0_f64, dt, 2.0 * dt, T_FINAL];

    let weights = [1.0_f64, 0.7, 1.2];
    let snapshots: Vec<Arc<Graph<f64>>> = weights.iter().map(|_| Arc::clone(&g_ref)).collect();

    let weight_fns: Vec<SegmentWeightFn<f64>> = weights
        .iter()
        .copied()
        .map(|w| {
            let lap = make_lap_uniform(n, w);
            let wfn: SegmentWeightFn<f64> = Box::new(move |_t: f64| Arc::clone(&lap));
            wfn
        })
        .collect();

    let traj = GraphTraj::new(breakpoints, snapshots, weight_fns).expect("valid 3-segment traj");
    (traj, g_ref)
}

/// Build `MagnusGraphHeatChernoff` with a Laplacian that matches segment weights
/// in the range [0.7, 1.2]. `rho_bar` bounds the spectral radius: max degree = 2,
/// max weight = 1.2 → `rho_bar` = 2.4 + margin.
fn make_magnus(n: usize) -> MagnusGraphHeatChernoff<f64> {
    let topology = Arc::new(Graph::<f64>::path(n));
    let rho_bar = 3.0_f64;
    // Closure: constant weight 1.0 (used only for apply_into, not evolve_with_traj).
    let lap0 = make_lap_uniform(n, 1.0);
    let lap_at: LaplacianAtTime<f64> = Box::new(move |_t: f64| Arc::clone(&lap0));
    MagnusGraphHeatChernoff::new(Arc::clone(&topology), lap_at, rho_bar, true)
        .expect("valid magnus")
}

// ---------------------------------------------------------------------------
// OLS slope helper
// ---------------------------------------------------------------------------

fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / m;
    let my = ys.iter().sum::<f64>() / m;
    let num: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    num / den
}

// ---------------------------------------------------------------------------
// Self-convergence runner
// ---------------------------------------------------------------------------

fn self_conv_err_f64(
    mc: &MagnusGraphHeatChernoff<f64>,
    f0: &GraphSignal<f64>,
    n_steps_per_seg: usize,
) -> f64 {
    let n = N_NODES;
    let mut pool = ScratchPool::<f64>::new();

    // Coarse: n_steps_per_seg.
    let (traj_c, _) = make_test_traj(n);
    let u_coarse = mc
        .evolve_with_traj(&traj_c, n_steps_per_seg, f0, &mut pool)
        .expect("evolve_with_traj coarse ok");

    // Fine: 2*n_steps_per_seg.
    let (traj_f, _) = make_test_traj(n);
    let u_fine = mc
        .evolve_with_traj(&traj_f, 2 * n_steps_per_seg, f0, &mut pool)
        .expect("evolve_with_traj fine ok");

    u_coarse
        .values()
        .iter()
        .zip(u_fine.values())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// G12 f64 slope gate
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g12_graph_traj_slope_f64() {
    let mc = make_magnus(N_NODES);
    let (_, g_ref) = make_test_traj(N_NODES);
    let f0 = GraphSignal::from_fn(Arc::clone(&g_ref), |i| {
        ((i as f64) * core::f64::consts::PI / N_NODES as f64).sin()
    });

    let mut log_n = Vec::new();
    let mut log_err = Vec::new();
    for &n_steps in &N_STEPS_PER_SEG_F64 {
        let err = self_conv_err_f64(&mc, &f0, n_steps);
        println!("G12 f64  n_steps_per_seg={n_steps:4}  self_conv_err={err:.4e}");
        if err > 0.0 {
            log_n.push((n_steps as f64).ln());
            log_err.push(err.ln());
        }
    }

    // Filter out trailing points that hit the f64 epsilon floor: once the error
    // stops decreasing (non-monotone), remove all trailing non-decreasing points
    // to avoid contaminating the OLS slope with rounding-floor noise.
    while log_err.len() >= 2 {
        let last = *log_err.last().unwrap();
        let prev = log_err[log_err.len() - 2];
        if last >= prev {
            log_n.pop();
            log_err.pop();
        } else {
            break;
        }
    }

    assert!(
        log_n.len() >= 3,
        "G12 FAIL: fewer than 3 usable error values after floor filter"
    );
    let slope = ols_slope(&log_n, &log_err);
    println!("G12 f64  slope = {slope:.4}  (threshold {SLOPE_THRESHOLD_F64})");
    assert!(
        slope <= SLOPE_THRESHOLD_F64,
        "G12 FAIL f64: slope {slope:.4} > {SLOPE_THRESHOLD_F64} (order-4 gate)"
    );
}

// ---------------------------------------------------------------------------
// Sanity: single-segment traj = same as direct apply_into
// ---------------------------------------------------------------------------

#[test]
fn g12_single_segment_matches_apply_into() {
    let n = 8usize;
    let g = Arc::new(Graph::<f64>::path(n));
    let w = 0.8_f64;
    let lap_arc = make_lap_uniform(n, w);
    let lap_arc2 = Arc::clone(&lap_arc);

    // Magnus operator with fixed Laplacian.
    let topology = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> = Box::new(move |_t| Arc::clone(&lap_arc2));
    let mc = MagnusGraphHeatChernoff::new(topology, lap_at, 3.0, true).unwrap();

    let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
    let tau_total = 0.02_f64;
    let n_steps = 5usize;
    let tau = tau_total / n_steps as f64;

    // Direct apply_into loop.
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        mc.apply_into(tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let u_direct = cur;

    // evolve_with_traj single-segment with same total time.
    let lap_snap = Arc::clone(&lap_arc);
    let wfn: SegmentWeightFn<f64> = Box::new(move |_t| Arc::clone(&lap_snap));
    let traj = GraphTraj::new(vec![0.0, tau_total], vec![Arc::clone(&g)], vec![wfn]).unwrap();
    let mut pool2 = ScratchPool::<f64>::new();
    let u_traj = mc
        .evolve_with_traj(&traj, n_steps, &f0, &mut pool2)
        .unwrap();

    let diff: f64 = u_direct
        .values()
        .iter()
        .zip(u_traj.values())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        diff < 1e-13,
        "single-segment traj diverges from direct apply_into: diff = {diff:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Sanity: contractivity across 3 segments
// ---------------------------------------------------------------------------

#[test]
fn g12_traj_is_contractive() {
    let n = 16usize;
    let mc = make_magnus(n);
    let (traj, g_ref) = make_test_traj(n);
    let f0 = GraphSignal::from_fn(Arc::clone(&g_ref), |i| ((i as f64) * 0.5).sin());
    let norm_f0 = f0.norm_l2();
    let mut pool = ScratchPool::<f64>::new();
    let result = mc.evolve_with_traj(&traj, 5, &f0, &mut pool).unwrap();
    let norm_r = result.norm_l2();
    assert!(
        norm_r <= norm_f0 + 1e-10,
        "traj contractivity: ‖Sf‖={norm_r:.6e} > ‖f‖={norm_f0:.6e}"
    );
}
