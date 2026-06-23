//! Erdős-Rényi proptest invariants for Wave 2.1B kernels.
//!
//! 64 cases per parameter combination. Verifies:
//!   P1: quasi-contractivity (sup-norm growth bounded by `growth()` envelope).
//!   P2: NaN-free output.
//!   P3: Gershgorin bound ≤ 2 * `max_degree` (combinatorial Laplacian).
//!   P4: `with_zeta_a` vs leading agreement at small τ (order-2 correction bounded).
//!
//! See Wave 2.1B contract §4.4.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use proptest::prelude::*;
use semiflow::{
    graph::{Graph, Laplacian},
    graph_heat::GraphHeatChernoff,
    graph_heat4::GraphHeat4thChernoff,
    graph_signal::GraphSignal,
    strang_graph::StrangSplitGraph,
    ChernoffFunction, ScratchPool, State,
};

fn max_degree(g: &Graph<f64>) -> usize {
    let n = g.n_nodes();
    (0..n)
        .map(|i| g.row_ptr()[i + 1] - g.row_ptr()[i])
        .max()
        .unwrap_or(0)
}

fn max_degree_f32(g: &Graph<f32>) -> usize {
    let n = g.n_nodes();
    (0..n)
        .map(|i| g.row_ptr()[i + 1] - g.row_ptr()[i])
        .max()
        .unwrap_or(0)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    // -----------------------------------------------------------------------
    // P1/P2/P3 for GraphHeatChernoff::with_zeta_a (f64)
    // -----------------------------------------------------------------------
    #[test]
    fn quasi_contractive_zeta_a_f64(
        n in 16usize..64,
        p in 0.05f64..0.30,
        seed in any::<u64>(),
        tau in 1e-5_f64..1e-2_f64,
    ) {
        let g = Arc::new(Graph::<f64>::erdos_renyi(n, p, seed));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));

        // P3: Gershgorin bound ≤ 2 * max_degree
        let rho = lap.spectral_radius_bound();
        let max_deg = max_degree(&g) as f64;
        prop_assert!(rho <= 2.0 * max_deg + 1e-10,
            "P3 fail: rho={rho:.4} > 2*deg={:.4}", 2.0 * max_deg);

        prop_assume!(tau * rho <= 0.5);   // stability envelope

        let kernel = GraphHeatChernoff::with_zeta_a(Arc::clone(&lap));
        let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f64 * 0.17).sin());
        let mut dst = f0.clone();
        let mut scratch = ScratchPool::<f64>::new();
        kernel.apply_into(tau, &f0, &mut dst, &mut scratch).unwrap();

        // P2: NaN-free
        for &v in dst.values() { prop_assert!(v.is_finite()); }

        // P1: quasi-contractivity (analytical bound for Taylor-2)
        let tau2_rho2 = tau * tau * rho * rho;
        let bound = (1.0 + 0.5 * tau2_rho2) * f0.norm_sup() + 1e-12;
        prop_assert!(dst.norm_sup() <= bound,
            "P1 fail: {:.4e} > bound {:.4e}", dst.norm_sup(), bound);
    }

    // -----------------------------------------------------------------------
    // P1/P2 for GraphHeat4thChernoff (f64)
    // -----------------------------------------------------------------------
    #[test]
    fn quasi_contractive_zeta4_f64(
        n in 16usize..64,
        p in 0.05f64..0.30,
        seed in any::<u64>(),
        tau in 1e-5_f64..1e-3_f64,
    ) {
        let g = Arc::new(Graph::<f64>::erdos_renyi(n, p, seed));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        let rho = lap.spectral_radius_bound();

        prop_assume!(tau * rho <= 0.5);

        let kernel = GraphHeat4thChernoff::new(Arc::clone(&lap));
        let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f64 * 0.17).sin());
        let mut dst = f0.clone();
        let mut scratch = ScratchPool::<f64>::new();
        kernel.apply_into(tau, &f0, &mut dst, &mut scratch).unwrap();

        // P2
        for &v in dst.values() { prop_assert!(v.is_finite()); }

        // P1: conservative sup-norm bound from operator-norm analysis
        let bound = 2.0 * f0.norm_sup() + 1e-12;
        prop_assert!(dst.norm_sup() <= bound,
            "P1 fail: {:.4e} > {:.4e}", dst.norm_sup(), bound);
    }

    // -----------------------------------------------------------------------
    // P1/P2 for StrangSplitGraph::new_bipartite_path (path graph, f64)
    // -----------------------------------------------------------------------
    #[test]
    fn quasi_contractive_strang_path_f64(
        n in 4usize..64,
        tau in 1e-5_f64..1e-2_f64,
        seed in any::<u64>(),   // unused for path, kept for proptest arity
    ) {
        let _ = seed;
        let n_even = if n % 2 == 0 { n } else { n + 1 };   // path works for any n >= 2
        let g = Arc::new(Graph::<f64>::path(n_even));
        let lap_full = Arc::new(Laplacian::assemble_combinatorial(&g));
        let rho = lap_full.spectral_radius_bound();

        prop_assume!(tau * rho <= 0.5);

        let strang = StrangSplitGraph::new_bipartite_path(&g).unwrap();
        let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f64 * 0.17).sin());
        let mut dst = f0.clone();
        let mut scratch = ScratchPool::<f64>::new();
        strang.apply_into(tau, &f0, &mut dst, &mut scratch).unwrap();

        // P2
        for &v in dst.values() { prop_assert!(v.is_finite()); }

        // P1: conservative bound
        let bound = 2.0 * f0.norm_sup() + 1e-12;
        prop_assert!(dst.norm_sup() <= bound,
            "P1 fail: {:.4e} > {:.4e}", dst.norm_sup(), bound);
    }

    // -----------------------------------------------------------------------
    // P4: with_zeta_a vs leading — difference bounded by τ² ρ² / 2 term
    // -----------------------------------------------------------------------
    #[test]
    fn p4_zeta_a_vs_leading_f64(
        n in 16usize..64,
        p in 0.05f64..0.30,
        seed in any::<u64>(),
        tau in 1e-5_f64..1e-3_f64,
    ) {
        let g = Arc::new(Graph::<f64>::erdos_renyi(n, p, seed));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        let rho = lap.spectral_radius_bound();

        prop_assume!(tau * rho <= 0.5);

        let leading = GraphHeatChernoff::new(Arc::clone(&lap));
        let zeta_a = GraphHeatChernoff::with_zeta_a(Arc::clone(&lap));

        let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f64 * 0.17).sin());
        let mut dst_l = f0.clone();
        let mut dst_z = f0.clone();
        let mut scratch = ScratchPool::<f64>::new();

        leading.apply_into(tau, &f0, &mut dst_l, &mut scratch).unwrap();
        zeta_a.apply_into(tau, &f0, &mut dst_z, &mut scratch).unwrap();

        // P4: ‖S_zeta_a − S_leading‖ ≤ τ² ρ² / 2 * ‖f0‖ + 1e-10
        let mut diff = dst_z.clone();
        diff.axpy_into(-1.0, &dst_l);
        let expected_bound = 0.5 * tau * tau * rho * rho * f0.norm_sup() + 1e-10;
        prop_assert!(diff.norm_sup() <= expected_bound,
            "P4 fail: diff={:.4e} > bound={:.4e}", diff.norm_sup(), expected_bound);
    }

    // -----------------------------------------------------------------------
    // P1/P2 for StrangSplitGraph::new_bipartite_path (path graph, f32)
    // -----------------------------------------------------------------------
    #[test]
    fn quasi_contractive_strang_path_f32(
        n in 4usize..32,
        tau in 1e-4_f32..1e-2_f32,
    ) {
        let n_even = if n % 2 == 0 { n } else { n + 1 };
        let g = Arc::new(Graph::<f32>::path(n_even));
        let lap_full = Arc::new(Laplacian::assemble_combinatorial(&g));
        let rho = lap_full.spectral_radius_bound();

        prop_assume!(tau * rho <= 0.5);

        let strang = StrangSplitGraph::new_bipartite_path(&g).unwrap();
        let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f32 * 0.17).sin());
        let mut dst = f0.clone();
        let mut scratch = ScratchPool::<f32>::new();
        strang.apply_into(tau, &f0, &mut dst, &mut scratch).unwrap();

        // P2
        for &v in dst.values() { prop_assert!(v.is_finite()); }
        // P1
        let bound = 2.0_f32 * f0.norm_sup() + 1e-5;
        prop_assert!(dst.norm_sup() <= bound,
            "P1 f32 fail: {:.4e} > {:.4e}", dst.norm_sup(), bound);
    }

    // P3 for f32 with_zeta_a
    #[test]
    fn gershgorin_bound_zeta_a_f32(
        n in 4usize..32,
        p in 0.05f64..0.30,
        seed in any::<u64>(),
    ) {
        let g = Arc::new(Graph::<f32>::erdos_renyi(n, p, seed));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        let rho = lap.spectral_radius_bound();
        let max_deg = max_degree_f32(&g) as f32;
        prop_assert!(rho <= 2.0 * max_deg + 1e-5,
            "P3 f32 fail: rho={rho:.4} > 2*deg={:.4}", 2.0 * max_deg);
    }
}
