//! G30 — Quantum graph first 4 Friedlander eigenmodes match (`RELEASE_BLOCKING`).
//!
//! Path graph `P_3`: 3 vertices, 2 equal-length unit edges.
//! Total arc length L = 2. Arc-length parameterisation s ∈ [0, 2]:
//!   - Edge 0 covers s ∈ [0, 1]
//!   - Edge 1 covers s ∈ [1, 2]
//!
//! Eigenmodes (math §29.4, Friedlander 2005 *Ann. Inst. Fourier* 55:1):
//!   `φ_k(s)` = cos(k·π·s/2)       k = 0, 1, 2, 3
//! Eigenvalues of L = -(1/2)∂²:
//!   `λ_k` = k²·π² / 8             k = 0, 1, 2, 3
//!
//! Heat semigroup reference solution at T = 0.1:
//!   `u_ref_k(T, s)` = `exp(-λ_k · T)` · `φ_k(s)`
//!
//! Gate: `||u_k - u_ref_k||_∞` ≤ 1e-3 for all k ∈ {0, 1, 2, 3}.
//!
//! All 4 sub-checks MUST pass. Failure BLOCKS v3.1 release.
//!
//! Test uses `Evolver::new(kernel, 64).evolve(T, &ic)` per the G30 spec
//! in `contracts/semiflow-core.properties.yaml`.
//!
//! ADR-0078. Feature gate: slow-tests.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // k ≤ 3 in loop, never exceeds 2^52

use core::f64::consts::PI;

use semiflow::{Evolver, QuantumGraph, QuantumGraphHeatChernoff, QuantumGraphSignal, State};

// ---------------------------------------------------------------------------
// Gate constant (NON-NEGOTIABLE per ADR-0078 G30 spec)
// ---------------------------------------------------------------------------

const TOL_G30: f64 = 1.0e-3;
const T_FINAL: f64 = 0.1;
const N_CHERNOFF: usize = 64;

// ---------------------------------------------------------------------------
// G30 test
// ---------------------------------------------------------------------------

/// G30 — First 4 Friedlander eigenmodes on path P_3 reconstructed at ≤ 1e-3.
///
/// Setup per `contracts/semiflow-core.properties.yaml` G30:
/// ```text
///   edge_endpoints = [(0,1), (1,2)]
///   edge_lengths   = [1.0, 1.0]
///   graph  = QuantumGraph::<f64>::new(edge_endpoints, edge_lengths, 64)
///   kernel = QuantumGraphHeatChernoff::<f64>::new(graph.clone())
///   evolver = Evolver::new(kernel, 64)
/// ```
///
/// For each k ∈ {0,1,2,3}:
/// ```text
///   f_0_k   = QuantumGraphSignal::from_eigenmode(&graph, k)
///   u_k     = evolver.evolve(T, &f_0_k)
///   u_ref_k = QuantumGraphSignal::from_eigenmode_scaled(&graph, k, exp(-λ_k · T))
///   err_k   = ||u_k - u_ref_k||_∞
///   ASSERT err_k ≤ 1e-3
/// ```
#[test]
fn g30_path_graph_first_4_eigenmodes() {
    // Build path graph P_3: 3 vertices, 2 unit edges, 64 grid points per edge.
    let edge_endpoints = vec![(0usize, 1usize), (1usize, 2usize)];
    let edge_lengths = vec![1.0_f64, 1.0_f64];
    let graph = QuantumGraph::<f64>::new(edge_endpoints, edge_lengths, N_CHERNOFF)
        .expect("path graph construction failed");

    let kernel =
        QuantumGraphHeatChernoff::<f64>::new(graph.clone()).expect("kernel construction failed");

    let evolver = Evolver::new(kernel, N_CHERNOFF).expect("Evolver construction failed");

    let mut max_err_overall = 0.0_f64;

    for k in 0..4 {
        // Eigenvalue λ_k = k²·π² / 8.
        let lam_k = (k as f64 * PI).powi(2) / 8.0;
        let decay = (-lam_k * T_FINAL).exp();

        // Initial condition: φ_k sampled on the path graph.
        let ic = QuantumGraphSignal::from_eigenmode(&graph, k);

        // Evolve via Chernoff product formula.
        let mut u_k = evolver.evolve(T_FINAL, &ic).expect("evolve failed");

        // Reference solution: decay · φ_k.
        let u_ref = QuantumGraphSignal::from_eigenmode_scaled(&graph, k, decay);

        // Compute ||u_k - u_ref||_∞ via axpy_into.
        State::<f64>::axpy_into(&mut u_k, -1.0, &u_ref);
        let err = State::<f64>::norm_sup(&u_k);

        println!(
            "G30 k={k}: λ_k={lam_k:.6}, decay={decay:.6}, err={err:.4e}  (gate ≤ {TOL_G30:.0e})"
        );

        assert!(
            err <= TOL_G30,
            "G30 FAIL k={k}: ||u_k - u_ref||_∞ = {err:.4e} > {TOL_G30:.0e}. \
             Check QuantumGraphHeatChernoff Kirchhoff projection (ADR-0078)."
        );
        max_err_overall = max_err_overall.max(err);
    }

    println!("G30 PASS: max_err over k=0..3 = {max_err_overall:.4e} ≤ {TOL_G30:.0e}");
}
