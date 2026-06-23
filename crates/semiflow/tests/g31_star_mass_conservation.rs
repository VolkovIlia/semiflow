//! G31 — Star-graph (degree ≥ 3) mass conservation (`RELEASE_BLOCKING`).
//!
//! Neumann–Kirchhoff heat semigroup on a metric star graph MUST conserve
//! arc-length-weighted mass ∫u(t,·)ds (no sources, insulated boundary).
//!
//! Path graphs (all internal vertices degree-2) already pass via G30.
//! This gate specifically covers degree-≥3 hubs, which have a separate
//! code path (combined-domain Phase-1 kernel vs per-edge).
//!
//! Gate criterion: |mass(T) - mass(0)| / mass(0) ≤ 3% for star-3 and star-4
//! (comparable to the path-4 discretization level of ~1.3%).
//!
//! Math: §29, ADR-0078. Conserved quantity = ∫`₀^L_total` u(t,s) ds where
//! s is arc-length parameterisation. Discrete approximation: sum of values
//! times uniform weight h = `L_total` / (`n_total` − 1).
//!
//! ALSO verifies that a path graph at the same resolution conserves mass
//! within 3%, confirming the per-edge fallback path is healthy.

// Integration test/example: allows for numerical patterns.
#![allow(clippy::cast_precision_loss, clippy::needless_pass_by_value)]

use semiflow_core::{Evolver, QuantumGraph, QuantumGraphHeatChernoff, QuantumGraphSignal};

const MASS_TOL: f64 = 0.03; // 3% drift tolerance
const T_FINAL: f64 = 0.8; // long enough to exercise diffusion through hub
const N_CHERNOFF: usize = 80; // Chernoff steps
const N_GRID: usize = 40; // grid points per edge

/// Compute arc-length-weighted mass for a signal.
/// Weight h = `L_total` / (`n_total` − 1); all edges uniform.
fn arc_mass(graph: &QuantumGraph<f64>, sig: &QuantumGraphSignal<f64>) -> f64 {
    let n = sig.per_edge[0].values.len();
    let h = graph.edge_lengths[0] / (n - 1) as f64;
    let mut total = 0.0_f64;
    for edge in &sig.per_edge {
        // Trapezoidal rule per edge: sum all interior points fully, add 0.5 for each endpoint.
        let vals = &edge.values;
        let edge_sum: f64 = vals[0] * 0.5
            + vals[1..vals.len() - 1].iter().sum::<f64>()
            + vals[vals.len() - 1] * 0.5;
        total += edge_sum * h;
    }
    total
}

/// Run mass-conservation test for any graph topology.
fn check_mass_conservation(label: &str, graph: QuantumGraph<f64>) {
    let kernel =
        QuantumGraphHeatChernoff::<f64>::new(graph.clone()).expect("kernel construction failed");
    let evolver = Evolver::new(kernel, N_CHERNOFF).expect("Evolver construction failed");

    // Localized initial condition: value 1 on all grid points of the first edge, 0 elsewhere.
    // This exercises heat flow from one arm into the hub and out through other arms.
    let ic = QuantumGraphSignal::from_fn(&graph, |e, _x| if e == 0 { 1.0_f64 } else { 0.0_f64 });

    let mass0 = arc_mass(&graph, &ic);
    assert!(mass0 > 0.0, "initial mass must be positive");

    let u_final = evolver.evolve(T_FINAL, &ic).expect("evolve failed");
    let mass_final = arc_mass(&graph, &u_final);

    let drift = (mass_final - mass0).abs() / mass0;
    let drift_pct = drift * 100.0;
    let tol_pct = MASS_TOL * 100.0;
    println!(
        "G31 {label}: mass {mass0:.4} -> {mass_final:.4}  drift {drift_pct:.1}%  (gate <= {tol_pct:.0}%)"
    );

    assert!(
        drift <= MASS_TOL,
        "G31 FAIL {label}: mass drift {drift_pct:.1}% > {tol_pct:.0}% (mass conservation violated). \
         Check QuantumGraphHeatChernoff combined-domain Phase-1 for non-path topologies."
    );
}

/// G31a — Path-4 (degree-2 only): baseline that the per-edge path conserves mass.
#[test]
fn g31a_path4_mass_conservation() {
    let graph = QuantumGraph::<f64>::path(4, 1.0, N_GRID).expect("path graph construction failed");
    check_mass_conservation("path-4", graph);
}

/// G31b — Star-3 (hub degree 3): mass MUST be conserved.
#[test]
fn g31b_star3_mass_conservation() {
    let graph = QuantumGraph::<f64>::star(3, 1.0, N_GRID).expect("star graph construction failed");
    check_mass_conservation("star-3", graph);
}

/// G31c — Star-4 (hub degree 4): mass MUST be conserved.
#[test]
fn g31c_star4_mass_conservation() {
    let graph = QuantumGraph::<f64>::star(4, 1.0, N_GRID).expect("star graph construction failed");
    check_mass_conservation("star-4", graph);
}
