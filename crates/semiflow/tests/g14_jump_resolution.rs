//! G14 jump-resolution gate — `evolve_with_traj` vs naive fixed-topology.
//!
//! Gate: `slope_with_traj ≤ 0.5 × slope_without_traj`
//! (with-traj convergence slope is ≥ 2× steeper than naive fixed-Laplacian).
//!
//! Setup: Star graph `S_4`. Edge `(0, 1)` weight switches `1 → 0.5 → 2 → 0.5`
//! at breakpoints `[0.1, 0.2, 0.3]`. `t_final = 0.4`.
//!
//! `evolve_with_traj` respects topology changes; naive `apply_into_at` uses
//! a single fixed Laplacian (ignoring the weight jumps), limiting convergence.
//!
//! See ADR-0054 acceptance criterion and Wave 2.2A contract §5.3.
//!
//! These tests are gated behind `--ignored` / `slow-tests` feature.

#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]

use std::sync::Arc;

use semiflow::{
    graph_traj::{GraphTraj, SegmentWeightFn},
    magnus_graph::{LaplacianAtTime, MagnusGraphHeatChernoff},
    state::HilbertState,
    Graph, GraphSignal, Laplacian, ScratchPool,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const T_FINAL: f64 = 0.4;

/// Self-convergence refinement levels: n_steps ∈ {4, 8, 16, 32, 64}.
/// These are TOTAL steps across the full time interval (not per-segment).
const N_STEPS_F64: [usize; 5] = [4, 8, 16, 32, 64];

/// Gate: slope_with_traj ≤ JUMP_RATIO_THRESHOLD × slope_without_traj
/// (both slopes are negative; 0.5 means with-traj is ≥ 2× steeper).
const JUMP_RATIO_THRESHOLD: f64 = 0.5;

// ---------------------------------------------------------------------------
// Star S_4 topology helpers
// ---------------------------------------------------------------------------

/// Build star graph S_4 (hub=0, leaves=1,2,3).
///
/// Edge (0,k) for k in 1..=3 with specified per-edge weights.
/// `w01`, `w02`, `w03` are weights for edges (0,1), (0,2), (0,3) respectively.
fn make_star4(w01: f64, w02: f64, w03: f64) -> Graph<f64> {
    let edges = [(0u32, 1u32, w01), (0, 2, w02), (0, 3, w03)];
    Graph::from_edges(4, edges.iter().copied()).expect("valid star edges")
}

/// Build star Laplacian with specified edge (0,1) weight; others = 1.0.
fn make_star_lap(w01: f64) -> Arc<Laplacian<f64>> {
    let g = make_star4(w01, 1.0, 1.0);
    Arc::new(Laplacian::assemble_combinatorial(&g))
}

/// Build the 4-segment trajectory for the jump test.
///
/// Segments: [0.0, 0.1), [0.1, 0.2), [0.2, 0.3), [0.3, 0.4]
/// Edge (0,1) weights: 1.0, 0.5, 2.0, 0.5
/// Edges (0,2), (0,3): always 1.0
fn make_jump_traj() -> GraphTraj<f64> {
    let breakpoints = vec![0.0_f64, 0.1, 0.2, 0.3, T_FINAL];
    let edge_weights = [1.0_f64, 0.5, 2.0, 0.5];

    let snaps: Vec<Arc<Graph<f64>>> = edge_weights
        .iter()
        .map(|&w| Arc::new(make_star4(w, 1.0, 1.0)))
        .collect();

    let weight_fns: Vec<SegmentWeightFn<f64>> = edge_weights
        .iter()
        .map(|&w| {
            let lap = make_star_lap(w);
            let wfn: SegmentWeightFn<f64> = Box::new(move |_t: f64| Arc::clone(&lap));
            wfn
        })
        .collect();

    GraphTraj::new(breakpoints, snaps, weight_fns).expect("valid jump traj")
}

/// Build Magnus operator for S_4 with initial weight 1.0 on edge (0,1).
/// rho_bar: max eigenvalue of L_{S_4} ≤ max_degree + max_weight.
/// Max degree = 3 (hub), max weight = 2.0 → rho_bar = 5.0 (conservative).
fn make_magnus_star() -> MagnusGraphHeatChernoff<f64> {
    let topology = Arc::new(make_star4(1.0, 1.0, 1.0));
    // For naive baseline: fixed Laplacian ignores weight changes.
    let lap_fixed = make_star_lap(1.0);
    let lap_at: LaplacianAtTime<f64> = Box::new(move |_t: f64| Arc::clone(&lap_fixed));
    MagnusGraphHeatChernoff::new(Arc::clone(&topology), lap_at, 5.0, true)
        .expect("valid magnus star")
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
// Self-convergence: evolve_with_traj (with-traj)
// ---------------------------------------------------------------------------

fn self_conv_with_traj(
    mc: &MagnusGraphHeatChernoff<f64>,
    f0: &GraphSignal<f64>,
    n_steps: usize,
) -> f64 {
    // n_steps_per_segment = n_steps / 4 (4 segments of equal length).
    // Minimum 1 step per segment.
    let steps_per_seg = (n_steps / 4).max(1);

    let mut pool = ScratchPool::<f64>::new();

    // Coarse.
    let traj_c = make_jump_traj();
    let u_c = mc
        .evolve_with_traj(&traj_c, steps_per_seg, f0, &mut pool)
        .expect("with-traj coarse ok");

    // Fine (2× steps).
    let traj_f = make_jump_traj();
    let u_f = mc
        .evolve_with_traj(&traj_f, 2 * steps_per_seg, f0, &mut pool)
        .expect("with-traj fine ok");

    u_c.values()
        .iter()
        .zip(u_f.values())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Self-convergence: naive apply_into_at (without-traj, ignores jumps)
// ---------------------------------------------------------------------------

fn self_conv_without_traj(
    mc: &MagnusGraphHeatChernoff<f64>,
    f0: &GraphSignal<f64>,
    n_steps: usize,
) -> f64 {
    let tau = T_FINAL / n_steps as f64;
    let tau_fine = T_FINAL / (2 * n_steps) as f64;
    let mut pool = ScratchPool::<f64>::new();

    // Coarse: n_steps uniform steps, fixed Laplacian (no jump awareness).
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for k in 0..n_steps {
        let t_start = k as f64 * tau;
        mc.apply_into_at(t_start, tau, &cur, &mut nxt, &mut pool)
            .unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let u_c = cur.clone();

    // Fine: 2*n_steps steps.
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    for k in 0..(2 * n_steps) {
        let t_start = k as f64 * tau_fine;
        mc.apply_into_at(t_start, tau_fine, &cur, &mut nxt, &mut pool)
            .unwrap();
        core::mem::swap(&mut cur, &mut nxt);
    }
    let u_f = cur;

    u_c.values()
        .iter()
        .zip(u_f.values())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// G14 jump-resolution gate
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g14_jump_resolution() {
    let mc = make_magnus_star();
    let g_star = Arc::new(make_star4(1.0, 1.0, 1.0));
    let f0 = GraphSignal::from_fn(Arc::clone(&g_star), |i| {
        [1.0_f64, -0.5, 0.3, 0.2][i as usize]
    });

    // Collect self-convergence errors for both methods.
    let mut log_n_with = Vec::new();
    let mut log_err_with = Vec::new();
    let mut log_n_without = Vec::new();
    let mut log_err_without = Vec::new();

    for &n_steps in &N_STEPS_F64 {
        let err_with = self_conv_with_traj(&mc, &f0, n_steps);
        let err_without = self_conv_without_traj(&mc, &f0, n_steps);
        println!(
            "G14  n_steps={n_steps:4}  err_with={err_with:.4e}  err_without={err_without:.4e}"
        );

        if err_with > 0.0 {
            log_n_with.push((n_steps as f64).ln());
            log_err_with.push(err_with.ln());
        }
        if err_without > 0.0 {
            log_n_without.push((n_steps as f64).ln());
            log_err_without.push(err_without.ln());
        }
    }

    assert!(
        log_n_with.len() >= 3 && log_n_without.len() >= 3,
        "G14 FAIL: not enough positive error values for slope fit"
    );

    let slope_with = ols_slope(&log_n_with, &log_err_with);
    let slope_without = ols_slope(&log_n_without, &log_err_without);

    println!("G14  slope_with_traj = {slope_with:.4}");
    println!("G14  slope_without_traj = {slope_without:.4}");
    println!(
        "G14  ratio = {:.4}  (threshold {JUMP_RATIO_THRESHOLD})",
        slope_with / slope_without
    );

    // Gate: slope_with_traj ≤ JUMP_RATIO_THRESHOLD × slope_without_traj.
    // Both slopes are negative. 0.5 × (negative) = less negative threshold.
    // Condition passes if with-traj slope is ≥ 2× steeper.
    assert!(
        slope_with <= JUMP_RATIO_THRESHOLD * slope_without,
        "G14 FAIL: slope_with_traj {slope_with:.4} > 0.5 × slope_without_traj {:.4} \
         (with-traj must be ≥ 2× steeper than naive)",
        JUMP_RATIO_THRESHOLD * slope_without
    );
}

// ---------------------------------------------------------------------------
// Sanity: jump trajectory preserves norm (dissipative)
// ---------------------------------------------------------------------------

#[test]
fn g14_jump_traj_is_dissipative() {
    let mc = make_magnus_star();
    let g_star = Arc::new(make_star4(1.0, 1.0, 1.0));
    let f0 = GraphSignal::from_fn(Arc::clone(&g_star), |i| {
        [1.0_f64, -0.5, 0.3, 0.2][i as usize]
    });
    let norm_f0 = f0.norm_l2();
    let traj = make_jump_traj();
    let mut pool = ScratchPool::<f64>::new();
    let result = mc.evolve_with_traj(&traj, 5, &f0, &mut pool).unwrap();
    let norm_r = result.norm_l2();
    assert!(
        norm_r <= norm_f0 + 1e-10,
        "jump traj must be dissipative: ‖Sf‖={norm_r:.6e} > ‖f‖={norm_f0:.6e}"
    );
}

// ---------------------------------------------------------------------------
// Sanity: 4 segments constructed correctly
// ---------------------------------------------------------------------------

#[test]
fn g14_jump_traj_has_4_segments() {
    let traj = make_jump_traj();
    assert_eq!(traj.n_segments(), 4);
    assert_eq!(traj.breakpoints().len(), 5);
    assert!((traj.t_horizon() - T_FINAL).abs() < 1e-14);
}
