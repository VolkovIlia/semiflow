//! `G_GRAPH_EXPMV_DENSE` (`RELEASE_BLOCKING`): `graph_expmv` vs dense `mat_exp_pade13`
//! on a small path graph.  `sup_error` ≤ 1e-10.
//!
//! Type: `BACKWARD_ERROR`; tolerance-driven (order = `u32::MAX`).
//! This mirrors `G_MATRIX_PADE_M5` — reuses the dense oracle, no sympy.

use std::sync::Arc;

use semiflow::{
    chernoff::ChernoffFunction,
    dense_graph_expmv_ref,
    graph::{Graph, Laplacian},
    graph_krylov::{GraphKrylovChernoff, KrylovPath},
    graph_signal::GraphSignal,
    scratch::ScratchPool,
};

/// N=10 path graph, τ=1.0.  Chebyshev at tol=1e-12 vs dense `mat_exp_pade13` reference.
///
/// Expected: `sup_error` ≤ 1e-10 (gate threshold).
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_graph_expmv_dense() {
    let n = 10usize;
    let tau = 1.0_f64;
    let tol = 1e-12_f64;  // tigher than gate threshold for extra margin

    let g = Arc::new(Graph::<f64>::path(n));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));

    // Build Chebyshev solver.
    let krylov = GraphKrylovChernoff::new(
        Arc::clone(&lap),
        KrylovPath::Chebyshev,
        tol,
    )
    .unwrap();

    // Gaussian-like signal centred on node 5.
    // i is u32 (from_fn index type) — f64::from(i) is exact and infallible.
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
        let x = f64::from(i) - 5.0;
        (-0.5 * x * x).exp()
    });
    let mut dst_krylov = GraphSignal::zeros(Arc::clone(&g));
    let mut scratch = ScratchPool::new();
    krylov
        .apply_into(tau, &src, &mut dst_krylov, &mut scratch)
        .expect("Chebyshev action failed");

    // Dense reference via mat_exp_pade13.
    let mut dst_dense = vec![0.0_f64; n];
    dense_graph_expmv_ref(&lap, tau, src.values(), &mut dst_dense)
        .expect("dense reference failed");

    // Supremum error.
    let sup_error = dst_krylov
        .values()
        .iter()
        .zip(dst_dense.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    eprintln!("G_GRAPH_EXPMV_DENSE  n={n}  tau={tau}  sup_error={sup_error:.3e}");
    assert!(
        sup_error <= 1e-10,
        "G_GRAPH_EXPMV_DENSE: sup_error={sup_error:.3e} > 1e-10"
    );
}
