//! 0-ULP bit-equality tests for `graph_batched` (ADR-0184 D5).
//!
//! For every batched forward kernel and both adjoint paths:
//! `batched([C, N]) == per-channel loop` bit-for-bit.
//!
//! No new math claims — structural identity only.

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};

    use crate::{
        chernoff::ChernoffFunction,
        graph::{Graph, Laplacian, LaplacianKind},
        graph_adjoint_presampled::PreSampledLaplacianSeq,
        graph_batched::{
            adjoint_state_gradient_batched, evolve_batched, evolve_batched_magnus,
            evolve_batched_magnus6, evolve_batched_varcoef_magnus,
        },
        graph_heat::GraphHeatChernoff,
        graph_sensitivity::EdgeWeightSensitivity,
        graph_signal::GraphSignal,
        magnus6_graph::MagnusGraphHeat6thChernoff,
        magnus_graph::MagnusGraphHeatChernoff,
        scratch::ScratchPool,
        varcoef_magnus_graph::VarCoefMagnusGraphHeatChernoff,
    };

    // -----------------------------------------------------------------------
    // Shared fixture
    // -----------------------------------------------------------------------

    const N: usize = 4;
    const N_STEPS: usize = 2;
    const TAU: f64 = 0.05;
    #[allow(clippy::cast_precision_loss)]
    const T_FINAL: f64 = TAU * N_STEPS as f64;
    const RHO: f64 = 4.0;
    const N_CHAN: usize = 3;

    fn test_src() -> Vec<f64> {
        vec![1.0, 0.5, 0.2, 0.0, 0.0, 1.0, 0.0, 0.5, -0.5, 0.3, 0.8, 0.1]
    }

    fn test_src_alt() -> Vec<f64> {
        vec![
            0.3, -0.1, 0.4, 0.9, -0.2, 0.7, -0.3, 0.1, 0.8, -0.5, 0.2, 0.6,
        ]
    }

    fn make_graph() -> Arc<Graph<f64>> {
        Arc::new(Graph::<f64>::path(N))
    }

    fn make_lap(g: &Arc<Graph<f64>>) -> Arc<Laplacian<f64>> {
        Arc::new(Laplacian::assemble_combinatorial(g))
    }

    // -----------------------------------------------------------------------
    // Test 1 — evolve_batched (GraphHeatChernoff, generic path)
    // -----------------------------------------------------------------------

    #[test]
    fn evolve_batched_generic_0ulp() {
        let g = make_graph();
        let lap = make_lap(&g);
        let kernel = GraphHeatChernoff::new(Arc::clone(&lap));
        let src = test_src();
        let mut dst_batched = vec![0.0_f64; src.len()];
        evolve_batched(&kernel, &g, T_FINAL, N_STEPS, &src, &mut dst_batched).unwrap();
        let expected = per_channel_f(&kernel, &g, N_STEPS, TAU, &src);
        assert_eq!(dst_batched, expected, "evolve_batched 0-ULP fail");
    }

    /// Per-channel ping-pong reference for any `ChernoffFunction`.
    #[allow(clippy::many_single_char_names)]
    fn per_channel_f<C>(
        func: &C,
        g: &Arc<Graph<f64>>,
        n_steps: usize,
        tau: f64,
        src_cols: &[f64],
    ) -> Vec<f64>
    where
        C: ChernoffFunction<f64, S = GraphSignal<f64>>,
    {
        let mut out = vec![0.0_f64; src_cols.len()];
        let mut scratch = ScratchPool::new();
        for ch in 0..N_CHAN {
            let s = &src_cols[ch * N..(ch + 1) * N];
            let mut a = GraphSignal::zeros(Arc::clone(g));
            let mut b = GraphSignal::zeros(Arc::clone(g));
            a.axpy_into_slice(1.0, s);
            let mut fwd = true;
            for _ in 0..n_steps {
                if fwd {
                    func.apply_into(tau, &a, &mut b, &mut scratch).unwrap();
                } else {
                    func.apply_into(tau, &b, &mut a, &mut scratch).unwrap();
                }
                fwd = !fwd;
            }
            let r = if fwd { a.values() } else { b.values() };
            out[ch * N..(ch + 1) * N].copy_from_slice(r);
        }
        out
    }

    // -----------------------------------------------------------------------
    // Test 2 — evolve_batched_magnus (MagnusGraphHeatChernoff K=4)
    // -----------------------------------------------------------------------

    fn make_magnus() -> (MagnusGraphHeatChernoff<f64>, Arc<Graph<f64>>) {
        let g = make_graph();
        let lap = make_lap(&g);
        let lap_fn = Box::new(move |_t: f64| Arc::clone(&lap));
        let mc = MagnusGraphHeatChernoff::new(Arc::clone(&g), lap_fn, RHO, false).unwrap();
        (mc, g)
    }

    #[test]
    fn evolve_batched_magnus_0ulp() {
        let (mc, g) = make_magnus();
        let src = test_src();
        let mut dst_batched = vec![0.0_f64; src.len()];
        evolve_batched_magnus(&mc, T_FINAL, N_STEPS, &src, &mut dst_batched).unwrap();
        let expected = per_channel_f(&mc, &g, N_STEPS, TAU, &src);
        assert_eq!(dst_batched, expected, "evolve_batched_magnus 0-ULP fail");
    }

    // -----------------------------------------------------------------------
    // Test 3 — evolve_batched_magnus6 (MagnusGraphHeat6thChernoff K=6)
    // -----------------------------------------------------------------------

    fn make_magnus6() -> (MagnusGraphHeat6thChernoff<f64>, Arc<Graph<f64>>) {
        let g = make_graph();
        let lap = make_lap(&g);
        let lap_fn = Box::new(move |_t: f64| Arc::clone(&lap));
        let mc6 = MagnusGraphHeat6thChernoff::new(Arc::clone(&g), lap_fn, RHO, false).unwrap();
        (mc6, g)
    }

    #[test]
    fn evolve_batched_magnus6_0ulp() {
        let (mc6, g) = make_magnus6();
        let src = test_src();
        let mut dst_batched = vec![0.0_f64; src.len()];
        evolve_batched_magnus6(&mc6, T_FINAL, N_STEPS, &src, &mut dst_batched).unwrap();
        let expected = per_channel_f(&mc6, &g, N_STEPS, TAU, &src);
        assert_eq!(dst_batched, expected, "evolve_batched_magnus6 0-ULP fail");
    }

    // -----------------------------------------------------------------------
    // Test 4 — evolve_batched_varcoef_magnus
    // -----------------------------------------------------------------------

    fn make_vc() -> VarCoefMagnusGraphHeatChernoff<f64> {
        let g = make_graph();
        let lap = make_lap(&g);
        let lap_fn: crate::magnus_graph::LaplacianAtTime<f64> =
            Box::new(move |_t: f64| Arc::clone(&lap));
        let a_fn: crate::varcoef_magnus_graph::WeightAtTime<f64> =
            Box::new(|_t: f64| vec![1.0_f64; N]);
        VarCoefMagnusGraphHeatChernoff::new(N, lap_fn, a_fn, RHO, 1.0)
            .unwrap()
            .with_radius_check(false)
    }

    #[test]
    fn evolve_batched_varcoef_magnus_0ulp() {
        let vc = make_vc();
        let g = make_graph();
        let src = test_src();
        let mut dst_batched = vec![0.0_f64; src.len()];
        evolve_batched_varcoef_magnus(&vc, T_FINAL, N_STEPS, &src, &mut dst_batched).unwrap();
        let expected = per_channel_f(&vc, &g, N_STEPS, TAU, &src);
        assert_eq!(dst_batched, expected, "evolve_batched_varcoef 0-ULP fail");
    }

    // -----------------------------------------------------------------------
    // Test 5 — PreSampledMagnusAdj::evolve_state_adjoint_batched_into
    // -----------------------------------------------------------------------

    fn make_presampled_adj() -> crate::graph_adjoint_presampled::PreSampledMagnusAdj<f64> {
        let g = make_graph();
        let lap = Laplacian::assemble_combinatorial(&g);
        let rp = lap.row_ptr().to_vec();
        let ci = lap.col_idx().to_vec();
        let vs = lap.vals().to_vec();
        let nnz = ci.len();
        let vals_seq: Vec<f64> = vs.iter().copied().cycle().take(nnz * 2 * N_STEPS).collect();
        let seq =
            PreSampledLaplacianSeq::new(rp, ci, vals_seq, N_STEPS, LaplacianKind::Combinatorial)
                .unwrap();
        MagnusGraphHeatChernoff::from_presampled(seq, RHO, false).unwrap()
    }

    #[test]
    fn presampled_adj_batched_0ulp() {
        let adj = make_presampled_adj();
        let g = make_graph();
        let src = test_src();
        let mut dst_batched = vec![0.0_f64; src.len()];
        let mut scratch = ScratchPool::new();
        adj.evolve_state_adjoint_batched_into(TAU, N_STEPS, &src, &mut dst_batched, &mut scratch)
            .unwrap();
        let mut expected = vec![0.0_f64; src.len()];
        for ch in 0..N_CHAN {
            let s = GraphSignal::from_fn(Arc::clone(&g), |i| src[ch * N + i as usize]);
            let mut d = GraphSignal::zeros(Arc::clone(&g));
            adj.evolve_state_adjoint_into(TAU, N_STEPS, &s, &mut d, &mut scratch)
                .unwrap();
            expected[ch * N..(ch + 1) * N].copy_from_slice(d.values());
        }
        assert_eq!(dst_batched, expected, "presampled_adj_batched 0-ULP fail");
    }

    // -----------------------------------------------------------------------
    // Test 6 — PreSampledVarCoefAdj::evolve_state_adjoint_batched_into
    // -----------------------------------------------------------------------

    fn make_presampled_vc_adj() -> crate::graph_adjoint_presampled::PreSampledVarCoefAdj<f64> {
        let g = make_graph();
        let lap = Laplacian::assemble_combinatorial(&g);
        let rp = lap.row_ptr().to_vec();
        let ci = lap.col_idx().to_vec();
        let vs = lap.vals().to_vec();
        let nnz = ci.len();
        let vals_seq: Vec<f64> = vs.iter().copied().cycle().take(nnz * 2 * N_STEPS).collect();
        let seq =
            PreSampledLaplacianSeq::new(rp, ci, vals_seq, N_STEPS, LaplacianKind::Combinatorial)
                .unwrap();
        let a_seq = vec![1.0_f64; 2 * N_STEPS * N];
        VarCoefMagnusGraphHeatChernoff::from_presampled(seq, a_seq, RHO, 1.0).unwrap()
    }

    #[test]
    fn presampled_vc_adj_batched_0ulp() {
        let adj = make_presampled_vc_adj();
        let g = make_graph();
        let src = test_src();
        let mut dst_batched = vec![0.0_f64; src.len()];
        let mut scratch = ScratchPool::new();
        adj.evolve_state_adjoint_batched_into(TAU, N_STEPS, &src, &mut dst_batched, &mut scratch)
            .unwrap();
        let mut expected = vec![0.0_f64; src.len()];
        for ch in 0..N_CHAN {
            let s = GraphSignal::from_fn(Arc::clone(&g), |i| src[ch * N + i as usize]);
            let mut d = GraphSignal::zeros(Arc::clone(&g));
            adj.evolve_state_adjoint_into(TAU, N_STEPS, &s, &mut d, &mut scratch)
                .unwrap();
            expected[ch * N..(ch + 1) * N].copy_from_slice(d.values());
        }
        assert_eq!(
            dst_batched, expected,
            "presampled_vc_adj_batched 0-ULP fail"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7 — adjoint_state_gradient_batched (summed gradient, 0-ULP)
    // -----------------------------------------------------------------------

    #[test]
    fn grad_batched_0ulp() {
        use crate::graph_sensitivity::adjoint_state_gradient;
        let (mc, g) = make_magnus();
        let n_params = N - 1; // path-4 has 3 edges
        let sens = EdgeWeightSensitivity {
            params: vec![(0, 1), (1, 2), (2, 3)],
            n_nodes: N,
        };
        let u0_cols = test_src();
        let dj_cols = test_src_alt();
        let mut grad_batched = vec![0.0_f64; n_params];
        let mut scratch = ScratchPool::new();
        adjoint_state_gradient_batched(
            &mc,
            &u0_cols,
            &dj_cols,
            N_STEPS,
            TAU,
            &sens,
            &mut grad_batched,
            &mut scratch,
        )
        .unwrap();
        // Per-channel reference: sum C individual gradients.
        let mut grad_expected = vec![0.0_f64; n_params];
        for ch in 0..N_CHAN {
            let u0 = GraphSignal::from_fn(Arc::clone(&g), |i| u0_cols[ch * N + i as usize]);
            let dj = GraphSignal::from_fn(Arc::clone(&g), |i| dj_cols[ch * N + i as usize]);
            let mut tmp = vec![0.0_f64; n_params];
            adjoint_state_gradient(&mc, &u0, N_STEPS, TAU, &dj, &sens, &mut tmp, &mut scratch)
                .unwrap();
            for k in 0..n_params {
                grad_expected[k] += tmp[k];
            }
        }
        assert_eq!(grad_batched, grad_expected, "grad_batched 0-ULP fail");
    }

    // -----------------------------------------------------------------------
    // Test 8 — n_steps == 0 copies src unchanged
    // -----------------------------------------------------------------------

    #[test]
    fn evolve_batched_zero_steps() {
        let g = make_graph();
        let lap = make_lap(&g);
        let kernel = GraphHeatChernoff::new(lap);
        let src = test_src();
        let mut dst = vec![99.0_f64; src.len()];
        evolve_batched(&kernel, &g, 0.0, 0, &src, &mut dst).unwrap();
        assert_eq!(dst, src, "zero n_steps must copy src unchanged");
    }

    // -----------------------------------------------------------------------
    // Tests 9–11 — parallel bit-equality (ADR-0184 D5)
    //
    // Each test runs the parallel path (n_cols ≥ 2 → thread::scope workers)
    // and compares against a per-channel reference using the SAME single-channel
    // kernel.  0-ULP identity expected: assert_eq! on f64 slices.
    // -----------------------------------------------------------------------

    #[cfg(feature = "parallel")]
    mod parallel_bit_eq {
        use alloc::{boxed::Box, sync::Arc, vec, vec::Vec};

        use crate::{
            chernoff::ChernoffFunction,
            graph::{Graph, Laplacian},
            graph_batched::{
                adjoint_state_gradient_batched, evolve_batched, evolve_batched_magnus,
            },
            graph_heat::GraphHeatChernoff,
            graph_sensitivity::{adjoint_state_gradient, EdgeWeightSensitivity},
            graph_signal::GraphSignal,
            magnus_graph::MagnusGraphHeatChernoff,
            scratch::ScratchPool,
        };

        // Use 8 channels — well above MIN_CHANNELS_PARALLEL (2) — so workers
        // are actually spawned.  N=8 nodes keeps total buffer tiny.
        const N: usize = 8;
        const N_COLS: usize = 8;
        const N_STEPS: usize = 4;
        const TAU: f64 = 0.02;
        #[allow(clippy::cast_precision_loss)]
        const T_FINAL: f64 = TAU * N_STEPS as f64;
        const RHO: f64 = 8.0;

        #[allow(clippy::cast_precision_loss)]
        fn make_src() -> Vec<f64> {
            (0..N * N_COLS).map(|i| (i as f64) * 0.1 - 0.4).collect()
        }

        #[allow(clippy::cast_precision_loss)]
        fn make_dj() -> Vec<f64> {
            (0..N * N_COLS).map(|i| (i as f64).sin() * 0.5).collect()
        }

        fn make_graph() -> Arc<Graph<f64>> {
            Arc::new(Graph::<f64>::path(N))
        }

        fn make_lap(g: &Arc<Graph<f64>>) -> Arc<Laplacian<f64>> {
            Arc::new(Laplacian::assemble_combinatorial(g))
        }

        fn make_magnus() -> MagnusGraphHeatChernoff<f64> {
            let g = make_graph();
            let lap = make_lap(&g);
            let lap_fn = Box::new(move |_t: f64| Arc::clone(&lap));
            MagnusGraphHeatChernoff::new(Arc::clone(&g), lap_fn, RHO, false).unwrap()
        }

        /// Per-channel reference for `GraphHeatChernoff` (serial, no Sync needed).
        fn reference_heat(
            g: &Arc<Graph<f64>>,
            lap: &Arc<Laplacian<f64>>,
            src: &[f64],
        ) -> Vec<f64> {
            let mut out = vec![0.0_f64; N * N_COLS];
            for ch in 0..N_COLS {
                let kernel = GraphHeatChernoff::new(Arc::clone(lap));
                let mut scr = ScratchPool::new();
                let mut a = GraphSignal::zeros(Arc::clone(g));
                let mut b = GraphSignal::zeros(Arc::clone(g));
                a.axpy_into_slice(1.0, &src[ch * N..(ch + 1) * N]);
                let mut fwd = true;
                for _ in 0..N_STEPS {
                    if fwd {
                        kernel.apply_into(TAU, &a, &mut b, &mut scr).unwrap();
                    } else {
                        kernel.apply_into(TAU, &b, &mut a, &mut scr).unwrap();
                    }
                    fwd = !fwd;
                }
                let r = if fwd { a.values() } else { b.values() };
                out[ch * N..(ch + 1) * N].copy_from_slice(r);
            }
            out
        }

        // Test 9: evolve_batched generic (GraphHeatChernoff, N_COLS=8)
        #[test]
        fn par_evolve_batched_0ulp() {
            let g = make_graph();
            let lap = make_lap(&g);
            let kernel = GraphHeatChernoff::new(Arc::clone(&lap));
            let src = make_src();
            let mut dst_par = vec![0.0_f64; N * N_COLS];
            evolve_batched(&kernel, &g, T_FINAL, N_STEPS, &src, &mut dst_par).unwrap();
            let expected = reference_heat(&g, &lap, &src);
            assert_eq!(dst_par, expected, "par evolve_batched 0-ULP fail (test 9)");
        }

        // Test 10: evolve_batched_magnus (MagnusGraphHeatChernoff K=4, N_COLS=8)
        #[test]
        fn par_evolve_batched_magnus_0ulp() {
            let mc = make_magnus();
            let g = make_graph();
            let src = make_src();
            let mut dst_par = vec![0.0_f64; N * N_COLS];
            evolve_batched_magnus(&mc, T_FINAL, N_STEPS, &src, &mut dst_par).unwrap();
            // Reference: call the ChernoffFunction path per channel.
            let mut expected = vec![0.0_f64; N * N_COLS];
            for ch in 0..N_COLS {
                let mut scr = ScratchPool::new();
                let mut a = GraphSignal::zeros(Arc::clone(&g));
                let mut b = GraphSignal::zeros(Arc::clone(&g));
                a.axpy_into_slice(1.0, &src[ch * N..(ch + 1) * N]);
                let mut fwd = true;
                for _ in 0..N_STEPS {
                    if fwd {
                        mc.apply_into(TAU, &a, &mut b, &mut scr).unwrap();
                    } else {
                        mc.apply_into(TAU, &b, &mut a, &mut scr).unwrap();
                    }
                    fwd = !fwd;
                }
                let r = if fwd { a.values() } else { b.values() };
                expected[ch * N..(ch + 1) * N].copy_from_slice(r);
            }
            assert_eq!(dst_par, expected, "par evolve_batched_magnus 0-ULP fail (test 10)");
        }

        // Test 11: adjoint_state_gradient_batched (edge-weight grad, N_COLS=8)
        #[test]
        fn par_grad_batched_0ulp() {
            let mc = make_magnus();
            let g = make_graph();
            let n_params = N - 1; // path-N has N-1 edges
            let edges: Vec<(usize, usize)> = (0..n_params).map(|i| (i, i + 1)).collect();
            let sens = EdgeWeightSensitivity { params: edges, n_nodes: N };
            let u0_cols = make_src();
            let dj_cols = make_dj();
            let mut grad_par = vec![0.0_f64; n_params];
            let mut scr = ScratchPool::new();
            adjoint_state_gradient_batched(
                &mc, &u0_cols, &dj_cols, N_STEPS, TAU, &sens, &mut grad_par, &mut scr,
            )
            .unwrap();
            // Per-channel serial reference.
            let mut grad_ref = vec![0.0_f64; n_params];
            for ch in 0..N_COLS {
                let u0 = GraphSignal::from_fn(Arc::clone(&g), |i| u0_cols[ch * N + i as usize]);
                let dj = GraphSignal::from_fn(Arc::clone(&g), |i| dj_cols[ch * N + i as usize]);
                let mut tmp = vec![0.0_f64; n_params];
                adjoint_state_gradient(&mc, &u0, N_STEPS, TAU, &dj, &sens, &mut tmp, &mut scr)
                    .unwrap();
                for k in 0..n_params {
                    grad_ref[k] += tmp[k];
                }
            }
            assert_eq!(grad_par, grad_ref, "par grad_batched 0-ULP fail (test 11)");
        }
    }
}
