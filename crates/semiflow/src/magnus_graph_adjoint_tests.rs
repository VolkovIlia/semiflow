// Tests for magnus_graph_adjoint — split from magnus_graph_adjoint.rs (batch H8).
// Included via `include!` so test items see all private symbols from the parent module.

// k (usize loop counter) cast to f64 for time coordinate; k ≪ 2^52.
#[allow(clippy::cast_precision_loss)]
mod tests {
    use alloc::sync::Arc;

    use crate::{
        chernoff::ChernoffFunction,
        graph::{Graph, Laplacian},
        graph_signal::GraphSignal,
        magnus_graph::{LaplacianAtTime, MagnusGraphHeatChernoff},
        scratch::ScratchPool,
        varcoef_magnus_graph::{VarCoefMagnusGraphHeatChernoff, WeightAtTime},
    };

    /// Build a path-graph `MagnusGraphHeatChernoff` for testing.
    fn make_path_mc(n: usize) -> MagnusGraphHeatChernoff<f64> {
        let g = Arc::new(Graph::<f64>::path(n));
        let g2 = Arc::clone(&g);
        let lap: LaplacianAtTime<f64> =
            Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
        MagnusGraphHeatChernoff::new(g, lap, 4.0, true).unwrap()
    }

    /// Build a `VarCoef` with constant a=1 (reduces to path MC).
    fn make_varcoef_mc(n: usize) -> VarCoefMagnusGraphHeatChernoff<f64> {
        let g = Arc::new(Graph::<f64>::path(n));
        let g2 = Arc::clone(&g);
        let lap: LaplacianAtTime<f64> =
            Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
        let a: WeightAtTime<f64> = Box::new(move |_t| alloc::vec![1.0_f64; n]);
        VarCoefMagnusGraphHeatChernoff::new(n, lap, a, 4.0, 1.0).unwrap()
    }

    /// Dual-pairing test: ⟨S u, g⟩ = ⟨u, S⋆ g⟩ on a path-4 graph.
    /// Tests both forward `apply_into` and adjoint `apply_state_adjoint_into`.
    #[test]
    fn dual_pairing_path4() {
        let n = 4;
        let mc = make_path_mc(n);
        let g = Arc::clone(&mc.graph);
        let tau = 0.05;

        let u = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) * 0.3 + 1.0);
        let v = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) * 0.7 - 0.5);

        let mut su = GraphSignal::zeros(Arc::clone(&g));
        let mut s_star_v = GraphSignal::zeros(Arc::clone(&g));
        let mut pool = ScratchPool::<f64>::new();

        mc.apply_into(tau, &u, &mut su, &mut pool).unwrap();
        mc.apply_state_adjoint_into(tau, &v, &mut s_star_v, &mut pool)
            .unwrap();

        let lhs: f64 = su.values().iter().zip(v.values()).map(|(a, b)| a * b).sum();
        let rhs: f64 = u
            .values()
            .iter()
            .zip(s_star_v.values())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            (lhs - rhs).abs() < 1e-12,
            "dual pairing failed: lhs={lhs:.15e}, rhs={rhs:.15e}, diff={}",
            (lhs - rhs).abs()
        );
    }

    /// Wrong-sign control: `VarCoef` with non-uniform a(t) gives [`L_a(t₁)`, `L_a(t₂)`] ≠ 0.
    ///
    /// We use a(t) = [1+t, 1/(1+t), 1+t, 1/(1+t), ...] (alternating pattern).
    /// Then `L_a(t₁)` and `L_a(t₂)` are NOT proportional so their commutator is non-zero,
    /// making S ≠ S⋆.  Using the forward map S on both sides of the duality pairing
    /// must give a different value from using S⋆ on the right side.
    #[test]
    // n, g, g2, tau, mc, … are standard short names for a test with a small graph.
    #[allow(clippy::many_single_char_names)]
    fn wrong_sign_control_fails_pairing_varcoef() {
        let n = 4;
        let g = Arc::new(Graph::<f64>::path(n));
        let g2 = Arc::clone(&g);
        let lap: LaplacianAtTime<f64> =
            Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
        // non-uniform time-varying a: alternating 1+t vs 1/(1+t).
        let a: WeightAtTime<f64> = Box::new(move |t: f64| {
            (0..n)
                .map(|i| if i % 2 == 0 { 1.0 + t } else { 1.0 / (1.0 + t) })
                .collect()
        });
        let mc = VarCoefMagnusGraphHeatChernoff::new(n, lap, a, 6.0, 2.0).unwrap();
        let tau = 0.06;

        let u = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) * 0.5 + 1.0);
        let v = GraphSignal::from_fn(Arc::clone(&g), |i| (f64::from(i) - 1.5).abs() + 0.5);

        let mut su = GraphSignal::zeros(Arc::clone(&g));
        let mut s_star_v = GraphSignal::zeros(Arc::clone(&g));
        let mut sv = GraphSignal::zeros(Arc::clone(&g));
        let mut pool = ScratchPool::<f64>::new();

        mc.apply_into(tau, &u, &mut su, &mut pool).unwrap();
        mc.apply_state_adjoint_into(tau, &v, &mut s_star_v, &mut pool)
            .unwrap();
        mc.apply_into(tau, &v, &mut sv, &mut pool).unwrap();

        let correct_rhs: f64 = u
            .values()
            .iter()
            .zip(s_star_v.values())
            .map(|(a, b)| a * b)
            .sum();
        let wrong_rhs: f64 = u.values().iter().zip(sv.values()).map(|(a, b)| a * b).sum();
        assert!(
            (correct_rhs - wrong_rhs).abs() > 1e-8,
            "expected S⋆ ≠ S for VarCoef non-uniform a(t), diff={}",
            (correct_rhs - wrong_rhs).abs()
        );
    }

    /// Costate recursion: forward then backward should reproduce the inner product.
    #[test]
    fn costate_recursion_path8() {
        let n = 8;
        let mc = make_path_mc(n);
        let g = Arc::clone(&mc.graph);
        let tau = 0.02;
        let steps = 5;

        let u0 = GraphSignal::from_fn(Arc::clone(&g), |i| (f64::from(i) + 1.0).ln());
        let lam_n = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 0.5);

        // Forward: u_n = S^n u_0.
        let mut u_n = u0.clone();
        let mut tmp = GraphSignal::zeros(Arc::clone(&g));
        let mut pool = ScratchPool::<f64>::new();
        for k in 0..steps {
            let t_s = k as f64 * tau;
            mc.apply_into_at(t_s, tau, &u_n, &mut tmp, &mut pool)
                .unwrap();
            core::mem::swap(&mut u_n, &mut tmp);
        }

        // Backward: lam_0 = (S⋆)^n lam_n.
        let mut lam_0 = GraphSignal::zeros(Arc::clone(&g));
        mc.evolve_state_adjoint_into(tau, steps, &lam_n, &mut lam_0, &mut pool)
            .unwrap();

        // ⟨u_n, lam_n⟩ should equal ⟨u_0, lam_0⟩ by adjointness.
        let fwd: f64 = u_n
            .values()
            .iter()
            .zip(lam_n.values())
            .map(|(a, b)| a * b)
            .sum();
        let bwd: f64 = u0
            .values()
            .iter()
            .zip(lam_0.values())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            (fwd - bwd).abs() < 1e-11,
            "costate recursion failed: fwd={fwd:.15e}, bwd={bwd:.15e}",
        );
    }

    /// Dual-pairing test for `VarCoefMagnusGraphHeatChernoff` (a=1 reduces to standard).
    #[test]
    fn dual_pairing_varcoef_path4() {
        let n = 4;
        let mc = make_varcoef_mc(n);
        let g = Arc::new(Graph::<f64>::path(n));
        let tau = 0.04;

        let u = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) * 0.3 + 1.0);
        let v = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) * 0.7 - 0.5);
        let mut su = GraphSignal::zeros(Arc::clone(&g));
        let mut sv = GraphSignal::zeros(Arc::clone(&g));
        let mut pool = ScratchPool::<f64>::new();

        mc.apply_into(tau, &u, &mut su, &mut pool).unwrap();
        mc.apply_state_adjoint_into(tau, &v, &mut sv, &mut pool)
            .unwrap();

        let lhs: f64 = su.values().iter().zip(v.values()).map(|(a, b)| a * b).sum();
        let rhs: f64 = u.values().iter().zip(sv.values()).map(|(a, b)| a * b).sum();
        assert!(
            (lhs - rhs).abs() < 1e-12,
            "VarCoef dual pairing failed: lhs={lhs:.15e}, rhs={rhs:.15e}",
        );
    }
}
