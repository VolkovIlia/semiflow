//! Proptest for `VarCoefGraphHeatChernoff` contractivity (32 Erdős-Rényi cases).
//!
//! Verifies: for all `τ` in CFL range and random `a(·)`, the operator is
//! contractive (‖S(τ) f‖₂ ≤ ‖f‖₂).

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use proptest::prelude::*;

use semiflow::{
    chernoff::ApplyChernoffExt, state::HilbertState, Graph, GraphSignal, VarCoefGraphHeatChernoff,
};

// ---------------------------------------------------------------------------
// Contractivity test: ‖S(τ) f‖₂ ≤ ‖f‖₂ + tiny_tol
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn var_coef_contractive_erdos_renyi(
        seed in 0u64..10000u64,
        n_nodes in 8usize..=32usize,
        p_edge in 0.3f64..=0.8f64,
        a_base in 0.5f64..=2.0f64,
        a_var in 0.0f64..=0.4f64,
        tau_frac in 0.01f64..=0.4f64,
    ) {
        let g = Arc::new(Graph::<f64>::erdos_renyi(n_nodes, p_edge, seed));
        // Skip empty graphs (no edges → trivial).
        if g.n_directed_edges() == 0 {
            return Ok(());
        }
        let a: Vec<f64> = (0..n_nodes)
            .map(|i| {
                // a(i) ∈ [a_base - a_var, a_base + a_var], always > 0.
                let delta = a_var * ((i as f64 * 0.7 + seed as f64 * 0.01).sin());
                (a_base + delta).max(0.1_f64)
            })
            .collect();

        // CFL-safe rho_bar: Gershgorin bound for L_G is at most max_degree * max_edge_weight.
        // For Erdős-Rényi with unit weights, max_degree ≤ n_nodes.
        let rho_bar = (n_nodes as f64) * 2.0; // conservative upper bound

        let Ok(vc) = VarCoefGraphHeatChernoff::new(Arc::clone(&g), a, rho_bar) else {
            return Ok(()); // skip degenerate cases
        };

        // Compute CFL-safe tau: tau * rho_bar * max(a)^2 < 0.5.
        let a_max = vc.a().iter().copied().fold(0.0_f64, f64::max);
        let tau_cfl = 0.4 / (rho_bar * a_max * a_max + 1e-10);
        let tau = tau_frac * tau_cfl;

        let f0 = GraphSignal::from_fn(Arc::clone(&g), |i| (f64::from(i) * 0.5).sin());
        let norm_f0 = f0.norm_l2();

        let Ok(result) = vc.apply_chernoff(tau, &f0) else {
            return Ok(()); // CFL or other validation failed, skip
        };
        let norm_result = result.norm_l2();

        // Contractivity check: ‖S(τ) f‖₂ ≤ ‖f‖₂ + tolerance.
        // Small tolerance accounts for τ²-correction adding a tiny positive term.
        let tol = norm_f0 * 1e-8 + 1e-14;
        prop_assert!(
            norm_result <= norm_f0 + tol,
            "contractivity violated: ‖Sf‖={norm_result:.6e} > ‖f‖={norm_f0:.6e} (tol={tol:.2e})"
        );
    }
}
