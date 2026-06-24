//! `G_GRAPH_ADJOINT_SAMPLED_PARITY` — `RELEASE_BLOCKING` gate (ADR-0180).
//!
//! Pre-sampled GL₄-aware path == closure path to 0 ULP (bit-exact), for both
//! Magnus K=4 and `VarCoef` kernels. Run:
//!
//! ```text
//! cargo test -p semiflow-core --features slow-tests graph_adjoint_sampled_parity
//! ```
//!
//! The test is `#[ignore]` unless `slow-tests` feature is active, so it does
//! not run in the standard fast suite.

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use semiflow::{
    graph_adjoint_presampled::{fill_abscissa_times, PreSampledLaplacianSeq},
    Graph, GraphSignal, Laplacian, LaplacianAtTime, LaplacianKind, MagnusGraphHeatChernoff,
    ScratchPool, VarCoefMagnusGraphHeatChernoff, WeightAtTime,
};

// GL4 abscissae (normative values from magnus_graph.rs; pub(crate) so inlined here)
// Used only in the unused-variable check annotation for build_vals_seq.
#[allow(dead_code)]
const _GL4_C1: f64 = 0.211_324_865_405_187_13;
#[allow(dead_code)]
const _GL4_C2: f64 = 0.788_675_134_594_812_9;

// ---------------------------------------------------------------------------
// Test parameters (mirrors oracle in scripts/verify_graphadjoint_sampled.py)
// ---------------------------------------------------------------------------

const N: usize = 8; // path graph nodes
const N_EDGES: usize = N - 1;
const N_STEPS: usize = 64;
const T_HORIZON: f64 = 0.5;

// ---------------------------------------------------------------------------
// Helpers: time-varying weights and Laplacian assembly
// ---------------------------------------------------------------------------

/// `w_k(t) = 1 + 0.5 · sin(t + 0.1·k)` — matches the oracle.
fn edge_weights_at(t: f64, n_edges: usize) -> Vec<f64> {
    (0..n_edges)
        .map(|k| 1.0 + 0.5 * (t + 0.1 * k as f64).sin())
        .collect()
}

/// Assemble a combinatorial Laplacian for `path-N` with weights `w`.
fn path_lap_from_weights(n: usize, w: &[f64]) -> Arc<Laplacian<f64>> {
    // cast_possible_truncation: edge indices always < graph node count (small test graphs).
    #[allow(clippy::cast_possible_truncation)]
    let edges: Vec<(u32, u32, f64)> = (0..w.len())
        .map(|e| (e as u32, (e + 1) as u32, w[e]))
        .collect();
    let g = Graph::<f64>::from_edges(n, edges).expect("valid path graph");
    Arc::new(Laplacian::assemble_combinatorial(&g))
}

/// Build the `vals_seq` for `n_steps` steps by sampling the same closure
/// on the GL₄ grid in adjoint-schedule order.
#[allow(unused_variables, clippy::too_many_arguments)]
fn build_vals_seq(
    n: usize,
    n_edges: usize,
    n_steps: usize,
    tau: f64,
    base_nnz: usize,
    base_rp: &[usize],
    base_ci: &[u32],
) -> Vec<f64> {
    let mut vals_seq = vec![0.0_f64; 2 * n_steps * base_nnz];
    let mut times = vec![0.0_f64; 2 * n_steps];
    fill_abscissa_times(T_HORIZON, n_steps, &mut times);

    for k in 0..n_steps {
        for ci in 0..2usize {
            let t_sample = times[2 * k + ci];
            let w = edge_weights_at(t_sample, n_edges);
            let lap = path_lap_from_weights(n, &w);
            // vals from this Laplacian are in the same CSR order as base
            let start = (2 * k + ci) * base_nnz;
            // Reconstruct vals_block by matching col_idx from base pattern
            // The assembled lap has the same pattern as base; just copy vals.
            assert_eq!(lap.col_idx(), base_ci, "topology must not change");
            assert_eq!(lap.row_ptr(), base_rp, "row_ptr must not change");
            vals_seq[start..start + base_nnz].copy_from_slice(lap.vals());
        }
    }
    vals_seq
}

// ---------------------------------------------------------------------------
// Gate: Magnus K=4 variant
// ---------------------------------------------------------------------------

/// `RELEASE_BLOCKING`: pre-sampled and closure paths produce bit-exact results.
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow-tests feature required")]
fn g_graph_adjoint_sampled_parity_magnus() {
    let tau = T_HORIZON / N_STEPS as f64;

    // Build base topology (unit-weight path for CSR pattern only).
    let g_base = Graph::<f64>::path(N);
    let lap_base = Laplacian::assemble_combinatorial(&g_base);
    let base_rp = lap_base.row_ptr().to_vec();
    let base_ci = lap_base.col_idx().to_vec();
    let base_nnz = base_ci.len();

    // --- closure path ---
    let g_arc = Arc::new(Graph::<f64>::path(N));
    let lap_cb: LaplacianAtTime<f64> = {
        let g2 = Arc::clone(&g_arc);
        Box::new(move |t: f64| {
            let w = edge_weights_at(t, N_EDGES);
            let _ = &g2; // keep alive
            path_lap_from_weights(N, &w)
        })
    };
    let mc_closure =
        MagnusGraphHeatChernoff::new(Arc::clone(&g_arc), lap_cb, 4.0, true).expect("closure ctor");

    let lam_n: Vec<f64> = (0..N).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let src = GraphSignal::from_fn(Arc::clone(&g_arc), |i| lam_n[i as usize]);
    let mut dst_closure = GraphSignal::zeros(Arc::clone(&g_arc));
    let mut scratch = ScratchPool::new();
    mc_closure
        .evolve_state_adjoint_into(tau, N_STEPS, &src, &mut dst_closure, &mut scratch)
        .expect("closure evolve");

    // --- presampled path ---
    let vals_seq = build_vals_seq(N, N_EDGES, N_STEPS, tau, base_nnz, &base_rp, &base_ci);
    let seq = PreSampledLaplacianSeq::new(
        base_rp,
        base_ci,
        vals_seq,
        N_STEPS,
        LaplacianKind::Combinatorial,
    )
    .expect("seq ctor");
    let ps_adj =
        MagnusGraphHeatChernoff::<f64>::from_presampled(seq, 4.0, true).expect("presampled ctor");

    let mut dst_sampled = GraphSignal::zeros(Arc::clone(&g_arc));
    let mut scratch2 = ScratchPool::new();
    ps_adj
        .evolve_state_adjoint_into(tau, N_STEPS, &src, &mut dst_sampled, &mut scratch2)
        .expect("presampled evolve");

    // 0 ULP: same float ops, same order.
    assert_eq!(
        dst_sampled.values(),
        dst_closure.values(),
        "G_GRAPH_ADJOINT_SAMPLED_PARITY magnus: presampled != closure (non-zero delta); \
         check abscissa order in build_vals_seq vs GL4 schedule"
    );
}

// ---------------------------------------------------------------------------
// Gate: VarCoef variant
// ---------------------------------------------------------------------------

/// `RELEASE_BLOCKING`: `VarCoef` presampled path is bit-exact vs closure.
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow-tests feature required")]
fn g_graph_adjoint_sampled_parity_varcoef() {
    let tau = T_HORIZON / N_STEPS as f64;

    // Build base topology.
    let g_base = Graph::<f64>::path(N);
    let lap_base = Laplacian::assemble_combinatorial(&g_base);
    let base_rp = lap_base.row_ptr().to_vec();
    let base_ci = lap_base.col_idx().to_vec();
    let base_nnz = base_ci.len();

    // Constant a(t) = 1 for simplicity (VarCoef with a=1 = Magnus).
    let a_fn: WeightAtTime<f64> = Box::new(|_t| vec![1.0_f64; N]);
    let lap_cb: LaplacianAtTime<f64> = Box::new(|t: f64| {
        let w = edge_weights_at(t, N_EDGES);
        path_lap_from_weights(N, &w)
    });

    let mc_closure = VarCoefMagnusGraphHeatChernoff::new(N, lap_cb, a_fn, 4.0, 4.0)
        .expect("varcoef closure ctor");

    let lam_n: Vec<f64> = (0..N).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let g_arc = Arc::new(Graph::<f64>::path(N));
    let src = GraphSignal::from_fn(Arc::clone(&g_arc), |i| lam_n[i as usize]);
    let mut dst_closure = GraphSignal::zeros(Arc::clone(&g_arc));
    let mut scratch = ScratchPool::new();
    mc_closure
        .evolve_state_adjoint_into(tau, N_STEPS, &src, &mut dst_closure, &mut scratch)
        .expect("varcoef closure evolve");

    // Build vals_seq and a_seq on GL4 grid.
    let vals_seq = build_vals_seq(N, N_EDGES, N_STEPS, tau, base_nnz, &base_rp, &base_ci);
    // a_seq: constant 1 over 2*n_steps*n_nodes grid points.
    let a_seq = vec![1.0_f64; 2 * N_STEPS * N];

    let seq = PreSampledLaplacianSeq::new(
        base_rp,
        base_ci,
        vals_seq,
        N_STEPS,
        LaplacianKind::Combinatorial,
    )
    .expect("seq ctor");
    let ps_adj = VarCoefMagnusGraphHeatChernoff::<f64>::from_presampled(seq, a_seq, 4.0, 4.0)
        .expect("varcoef presampled ctor");

    let mut dst_sampled = GraphSignal::zeros(Arc::clone(&g_arc));
    let mut scratch2 = ScratchPool::new();
    ps_adj
        .evolve_state_adjoint_into(tau, N_STEPS, &src, &mut dst_sampled, &mut scratch2)
        .expect("varcoef presampled evolve");

    assert_eq!(
        dst_sampled.values(),
        dst_closure.values(),
        "G_GRAPH_ADJOINT_SAMPLED_PARITY varcoef: presampled != closure; \
         check a_seq ordering and GL4 schedule"
    );
}
