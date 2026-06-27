//! `G_GRAPH_FRECHET_FD` (`RELEASE_BLOCKING`): `graph_expmv_frechet` vs central FD.
//!
//! Two sub-tests:
//! 1. N=2 single-edge (commutativity: [L,E]=0, rectangle rule accidentally exact).
//! 2. N=3 triangle (non-commuting: [L,E]≠0, `RELEASE_BLOCKING` gate with teeth).
//!
//! Relative error against central FD oracle must be ≤ 1e-7 for EVERY edge.
//!
//! Oracle: recompute J(w±ε) using an independent Krylov solve on the
//! perturbed graph.  No closed-form expressions used.

use std::sync::Arc;

use semiflow::{
    graph::{Graph, Laplacian},
    graph_frechet::graph_expmv_frechet,
    graph_krylov::{GraphKrylovChernoff, KrylovPath},
    graph_sensitivity::EdgeWeightSensitivity,
    graph_signal::GraphSignal,
    scratch::ScratchPool,
    chernoff::ChernoffFunction,
};

/// Compute J = ⟨dj, e^{−t L(w)} u0⟩ for a 2-node graph with edge weight `w`.
fn j_oracle(w: f64, t: f64, u0: &[f64; 2], dj: &[f64; 2]) -> f64 {
    let edges: Vec<(u32, u32, f64)> = vec![(0, 1, w)];
    let g = Arc::new(Graph::<f64>::from_edges(2, edges).expect("graph"));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let krylov = GraphKrylovChernoff::new(Arc::clone(&lap), KrylovPath::Chebyshev, 1e-12)
        .expect("krylov");
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| u0[i as usize]);
    let mut dst = GraphSignal::zeros(Arc::clone(&g));
    let mut scratch = ScratchPool::new();
    krylov.apply_into(t, &src, &mut dst, &mut scratch).expect("apply_into");
    dst.values().iter().zip(dj.iter()).map(|(&v, &d)| v * d).sum()
}

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_graph_frechet_fd() {
    // Fixed problem parameters.
    let w: f64 = 0.7;
    let t: f64 = 0.5;
    let u0: [f64; 2] = [1.0, -0.5];
    let dj: [f64; 2] = [0.3, -0.8];

    // Build the graph with the nominal edge weight.
    let edges: Vec<(u32, u32, f64)> = vec![(0, 1, w)];
    let g = Arc::new(Graph::<f64>::from_edges(2, edges).expect("graph"));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let krylov = GraphKrylovChernoff::new(Arc::clone(&lap), KrylovPath::Chebyshev, 1e-12)
        .expect("krylov");

    // A2 gradient via graph_expmv_frechet.
    let param_deriv = EdgeWeightSensitivity { params: vec![(0, 1)], n_nodes: 2 };
    let mut grad_w = vec![0.0_f64];
    let mut scratch = ScratchPool::new();
    graph_expmv_frechet(
        &krylov,
        &u0,
        &dj,
        1,
        t,
        &param_deriv,
        &mut grad_w,
        &mut scratch,
    )
    .expect("graph_expmv_frechet");
    let a2 = grad_w[0];

    // Central FD oracle: J(w+ε) and J(w-ε) via independent Krylov solves.
    let eps = 1e-6_f64;
    let j_plus = j_oracle(w + eps, t, &u0, &dj);
    let j_minus = j_oracle(w - eps, t, &u0, &dj);
    let fd = (j_plus - j_minus) / (2.0 * eps);

    let rel_err = (a2 - fd).abs() / (fd.abs() + 1e-30);
    eprintln!(
        "G_GRAPH_FRECHET_FD  a2={a2:.10e}  fd={fd:.10e}  rel_err={rel_err:.3e}"
    );
    assert!(
        rel_err <= 1e-7,
        "G_GRAPH_FRECHET_FD: rel_err={rel_err:.3e} > 1e-7 (a2={a2:.10e}, fd={fd:.10e})"
    );
}

// ---------------------------------------------------------------------------
// N=3 triangle — NON-COMMUTING gate (RELEASE_BLOCKING)
// ---------------------------------------------------------------------------
// For edges (0,1,w01), (1,2,w12), (0,2,w02) with distinct weights,
// [L, ∂L/∂w_k] ≠ 0 for every edge k.  The right-endpoint rectangle rule
// is WRONG here; only the full Duhamel integral (§54.5 augmented Fréchet)
// achieves rel-err ≤ 1e-7.

/// Compute J = ⟨dj, e^{−t L(w01,w12,w02)} u0⟩ for a 3-node triangle graph.
fn j_oracle_tri(w01: f64, w12: f64, w02: f64, t: f64, u0: &[f64; 3], dj: &[f64; 3]) -> f64 {
    let edges: Vec<(u32, u32, f64)> = vec![(0, 1, w01), (1, 2, w12), (0, 2, w02)];
    let g = Arc::new(Graph::<f64>::from_edges(3, edges).expect("graph"));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let krylov = GraphKrylovChernoff::new(Arc::clone(&lap), KrylovPath::Chebyshev, 1e-12)
        .expect("krylov");
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| u0[i as usize]);
    let mut dst = GraphSignal::zeros(Arc::clone(&g));
    let mut scratch = ScratchPool::new();
    krylov.apply_into(t, &src, &mut dst, &mut scratch).expect("apply_into");
    dst.values().iter().zip(dj.iter()).map(|(&v, &d)| v * d).sum()
}

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_graph_frechet_fd_triangle() {
    // Distinct weights ensure [L, ∂L/∂w_k] ≠ 0 for each edge k.
    let w = [0.7_f64, 1.3, 0.4]; // weights for edges (0,1),(1,2),(0,2)
    let t = 0.5_f64;
    let u0 = [1.0_f64, -0.5, 0.3];
    let dj = [0.3_f64, -0.8, 0.2];
    let edge_pairs = [(0usize, 1usize), (1, 2), (0, 2)];

    // Build nominal graph.
    let edges: Vec<(u32, u32, f64)> = vec![(0, 1, w[0]), (1, 2, w[1]), (0, 2, w[2])];
    let g = Arc::new(Graph::<f64>::from_edges(3, edges).expect("graph"));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let krylov =
        GraphKrylovChernoff::new(Arc::clone(&lap), KrylovPath::Chebyshev, 1e-12).expect("krylov");

    // A2 gradient for all 3 edges simultaneously.
    let param_deriv =
        EdgeWeightSensitivity { params: edge_pairs.to_vec(), n_nodes: 3 };
    let mut grad_w = vec![0.0_f64; 3];
    let mut scratch = ScratchPool::new();
    graph_expmv_frechet(&krylov, &u0, &dj, 1, t, &param_deriv, &mut grad_w, &mut scratch)
        .expect("graph_expmv_frechet");

    // Central FD oracle: perturb each edge weight independently.
    let eps = 1e-6_f64;
    for k in 0..3 {
        let (pi, pj) = edge_pairs[k];
        let mut wp = w;
        wp[k] += eps;
        let mut wm = w;
        wm[k] -= eps;
        let fd = (j_oracle_tri(wp[0], wp[1], wp[2], t, &u0, &dj)
            - j_oracle_tri(wm[0], wm[1], wm[2], t, &u0, &dj))
            / (2.0 * eps);
        let a2 = grad_w[k];
        let rel_err = (a2 - fd).abs() / (fd.abs() + 1e-30);
        eprintln!(
            "G_GRAPH_FRECHET_FD_TRIANGLE edge({pi},{pj})  a2={a2:.10e}  fd={fd:.10e}  rel_err={rel_err:.3e}"
        );
        assert!(
            rel_err <= 1e-7,
            "G_GRAPH_FRECHET_FD_TRIANGLE: edge({pi},{pj}) rel_err={rel_err:.3e} > 1e-7 \
             (a2={a2:.10e}, fd={fd:.10e})"
        );
    }
}
