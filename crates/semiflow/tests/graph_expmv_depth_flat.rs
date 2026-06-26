//! `G_GRAPH_EXPMV_DEPTH_FLAT` (`RELEASE_BLOCKING`): matvec-count is flat in `t`.
//!
//! For a small graph with small `λ_max` (edge weight 0.01), the Chebyshev degree
//! barely grows as t increases from 1 to 64 (because z = `t·λ_max/2` stays small).
//! Gate: `max_count` / `min_count` ≤ 4.
//!
//! Type: STRUCTURAL/PERF gate; no oracle.  Depth-independence verification for A1.

use std::sync::Arc;

use semiflow::{
    graph::{Graph, Laplacian},
    graph_krylov::{graph_expmv_matvec_count, KrylovPath},
};

/// Path-8 graph with edge weight 0.01.
/// `λ_max_Gershgorin` = 4·0.01 = 0.04 (interior node row-sum = 2·0.02 + 0.02 = 0.04).
/// t ∈ {1, 4, 16, 64}; z = t·0.02 ∈ {0.02, 0.08, 0.32, 1.28}.
///
/// Expected Chebyshev degrees at tol=1e-10: approximately {4, 5, 7, 10}.
/// Band ratio = max/min ≈ 10/4 = 2.5 ≤ 4 (gate threshold).
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_graph_expmv_depth_flat() {
    let n = 8usize;
    let edge_weight = 0.01_f64;
    let tol = 1e-10_f64;

    // Build a path graph with custom edge weight.
    // n=8 fits u32 easily; truncation is impossible here.
    #[allow(clippy::cast_possible_truncation)]
    let edges: Vec<(u32, u32, f64)> = (0..(n - 1) as u32)
        .map(|i| (i, i + 1, edge_weight))
        .collect();
    let g = Arc::new(Graph::from_edges(n, edges).expect("graph construction failed"));
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let lambda_max = lap.spectral_radius_bound();

    eprintln!("λ_max_Gershgorin = {lambda_max:.6}");

    let t_values = [1.0_f64, 4.0, 16.0, 64.0];
    let path = KrylovPath::Chebyshev;
    let mut counts: Vec<u32> = Vec::new();

    for &t in &t_values {
        let (_s, m) = graph_expmv_matvec_count(lambda_max, t, tol, &path);
        eprintln!("  t={t:5.0}  z={:.4}  chebyshev_degree={m}", t * lambda_max / 2.0);
        counts.push(m);
    }

    let max_count = *counts.iter().max().unwrap();
    let min_count = *counts.iter().min().unwrap();
    // max_count and min_count are u32 — f64::from is exact and infallible.
    let ratio = f64::from(max_count) / f64::from(min_count);

    eprintln!("G_GRAPH_EXPMV_DEPTH_FLAT  min={min_count}  max={max_count}  ratio={ratio:.2}");
    assert!(
        ratio <= 4.0,
        "G_GRAPH_EXPMV_DEPTH_FLAT: band ratio = {ratio:.2} > 4.0 (depth not flat)"
    );
}
