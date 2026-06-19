//! G22 slope gate: `VarCoefMagnusGraphHeatChernoff` (order-4 `VarCoef` × time-dep) convergence.
//!
//! Gate: log-log slope ≤ −3.85 (f64), ≤ −3.50 (f32) per ADR-0063.
//! Setup: time-dependent path graph `P_32`, smooth IC `cos(2π i/N)`,
//! `t_final = 0.5`, time-varying weights:
//!   - `w(t) = 1 + 0.3·sin(πt)` (edge weights)
//!   - `a_i(t) = 1 + 0.5·cos(πt) · i/N` (node weights)
//! `n_steps ∈ {5, 8, 12, 20}` (f64) / `{5, 8}` (f32 — minimal coverage above
//! f32 floor).
//!
//! Reference: Self-convergence diff against double-refined solution at
//! `n_steps = 80` (no closed-form oracle for `L_a(t)` with both a(t) and `L_G(t)`
//! time-varying).
//!
//! See math.md §20.6 and ADR-0063 §"acceptance gates".

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::needless_range_loop)]
// Integration test/bench: allows for numerical patterns.
#![allow(clippy::doc_lazy_continuation)]

use std::sync::Arc;

use semiflow_core::{
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    varcoef_magnus_graph::{VarCoefMagnusGraphHeatChernoff, WeightAtTime},
    LaplacianAtTime, ScratchPool,
};

const N_NODES: usize = 32;
const T: f64 = 0.5;
const N_VALUES: [usize; 4] = [5, 8, 12, 20];
const N_REFERENCE: usize = 80;

fn make_mc_f64() -> VarCoefMagnusGraphHeatChernoff<f64> {
    // Edge weights vary as w(t) = 1 + 0.3 sin(pi t).
    let lap_at: LaplacianAtTime<f64> = Box::new(move |t| {
        let w = 1.0 + 0.3 * (core::f64::consts::PI * t).sin();
        // Build a path graph with edge weight w via Graph::from_edges
        let edges: Vec<(u32, u32, f64)> =
            (0..(N_NODES - 1) as u32).map(|i| (i, i + 1, w)).collect();
        let g_t = Arc::new(Graph::<f64>::from_edges(N_NODES, edges).unwrap());
        Arc::new(Laplacian::assemble_combinatorial(&g_t))
    });
    let a_at: WeightAtTime<f64> = Box::new(move |t| {
        (0..N_NODES)
            .map(|i| 1.0 + 0.5 * (core::f64::consts::PI * t).cos() * (i as f64 / N_NODES as f64))
            .collect()
    });
    // rho_bar(L_G(t)) ≤ 2 * max edge weight = 2 * 1.3 = 2.6, take 3 for margin.
    // a_sup = sqrt(max a) = sqrt(1.5) ≈ 1.225.
    VarCoefMagnusGraphHeatChernoff::new(N_NODES, lap_at, a_at, 3.0, (1.5_f64).sqrt()).unwrap()
}

fn log_log_slope(ns: &[usize], errs: &[f64]) -> f64 {
    let m = ns.len() as f64;
    let lx: Vec<f64> = ns.iter().map(|&x| (x as f64).ln()).collect();
    let ly: Vec<f64> = errs.iter().map(|&y| y.ln()).collect();
    let sx: f64 = lx.iter().sum();
    let sy: f64 = ly.iter().sum();
    let sxx: f64 = lx.iter().map(|&x| x * x).sum();
    let sxy: f64 = lx.iter().zip(ly.iter()).map(|(&x, &y)| x * y).sum();
    (m * sxy - sx * sy) / (m * sxx - sx * sx)
}

fn evolve_f64(
    mc: &VarCoefMagnusGraphHeatChernoff<f64>,
    f0: &GraphSignal<f64>,
    n_steps: usize,
) -> GraphSignal<f64> {
    let tau = T / n_steps as f64;
    let mut pool = ScratchPool::<f64>::new();
    let mut cur = f0.clone();
    let mut nxt = f0.clone();
    let mut t = 0.0_f64;
    for _ in 0..n_steps {
        mc.apply_into_at(t, tau, &cur, &mut nxt, &mut pool).unwrap();
        core::mem::swap(&mut cur, &mut nxt);
        t += tau;
    }
    cur
}

#[test]
fn g22_varcoef_magnus_convergence_slope_f64() {
    let mc = make_mc_f64();
    let g = Arc::new(Graph::<f64>::path(N_NODES));
    let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = i as f64 / N_NODES as f64 * core::f64::consts::TAU;
        x.cos()
    });

    // Reference solution at n_steps = N_REFERENCE.
    let reference = evolve_f64(&mc, &f0, N_REFERENCE);

    let errs: Vec<f64> = N_VALUES
        .iter()
        .map(|&n_steps| {
            let u_t = evolve_f64(&mc, &f0, n_steps);
            u_t.values()
                .iter()
                .zip(reference.values().iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();

    for (&n_steps, &err) in N_VALUES.iter().zip(errs.iter()) {
        println!("G22 f64 n_steps={n_steps:4}, err={err:.4e}");
    }
    let slope = log_log_slope(&N_VALUES, &errs);
    println!("G22 f64 slope = {slope:.4}");
    assert!(
        slope <= -3.85,
        "G22 FAIL f64: slope {slope:.4} > -3.85 (order-4 gate per ADR-0063)"
    );
}
