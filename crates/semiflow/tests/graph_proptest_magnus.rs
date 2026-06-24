//! Proptest invariants for `MagnusGraphHeatChernoff` (Wave 2.1C, ADR-0051).
//!
//! 32 cases per property. Verifies:
//!   P1: NaN-free output for valid τ and rho_bar_max.
//!   P2: sup-norm contractivity within the Magnus order-4 envelope.
//!   P3: `OutOfMagnusRadius` error returned when τ·ρ̄ ≥ π/2.
//!
//! See Wave 2.1C contract §4 (NORMATIVE) and ADR-0051 §"Convergence radius".

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]

use std::{f64::consts::PI, sync::Arc};

use proptest::prelude::*;
use semiflow::{
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    magnus_graph::{LaplacianAtTime, MagnusGraphHeatChernoff},
    ChernoffFunction, ScratchPool, SemiflowError, State,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a path-graph `MagnusGraphHeatChernoff` with constant weight `w`.
///
/// Returns `(mc, graph_arc)` so callers can construct `GraphSignal` values.
/// The `graph()` accessor returns `&Graph<F>`, not `&Arc<Graph<F>>`, so the
/// Arc must be retained separately.
///
/// Uses `convergence_radius_check = false` to allow the proptest to
/// deliberately trigger the over-radius case via a larger τ.
fn make_mc_const_w(
    n: usize,
    w: f64,
    check_radius: bool,
) -> (MagnusGraphHeatChernoff<f64>, Arc<Graph<f64>>) {
    let edges = (0..n as u32 - 1).map(|i| (i, i + 1, w));
    let topo = Arc::new(Graph::from_edges(n, edges).expect("valid path edges"));
    let topo2 = Arc::clone(&topo);
    let topo3 = Arc::clone(&topo);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t: f64| Arc::new(Laplacian::assemble_combinatorial(&topo2)));
    // rho_bar_max = 2 * w (Gershgorin bound for a path with all weights = w).
    let rho_bar = 2.0 * w;
    let mc = MagnusGraphHeatChernoff::new(topo, lap_at, rho_bar, check_radius)
        .expect("valid inputs to MagnusGraphHeatChernoff::new");
    (mc, topo3)
}

// ---------------------------------------------------------------------------
// Propostests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]

    // -----------------------------------------------------------------------
    // P1 + P2: NaN-free + contractivity within Magnus envelope
    // -----------------------------------------------------------------------

    /// For random path-graph sizes N ∈ [16, 64], random positive edge weight
    /// w ∈ (0.2, 2.0], and random τ ∈ (0, 0.01]:
    ///
    ///   P1: all output values are finite (no NaN or Inf).
    ///   P2: sup-norm bounded by the Taylor-4 error envelope:
    ///       ‖S₄(τ)f‖ ≤ (1 + C·(τρ̄)⁵) · ‖f‖_∞  (conservative C=1 bound)
    ///
    /// Both checks gate the Magnus hot path without triggering OutOfMagnusRadius.
    #[test]
    fn magnus_nan_free_and_contractive(
        n in 16usize..64,
        w in 0.2_f64..2.0_f64,
        tau in 1e-5_f64..0.01_f64,
    ) {
        // Build kernel with radius check enabled.
        let (mc, g) = make_mc_const_w(n, w, true);
        let rho_bar = 2.0 * w;

        // Skip cases that would trigger OutOfMagnusRadius (τ·ρ̄ ≥ π/2).
        prop_assume!(tau * rho_bar < PI / 2.0);

        let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
            (i as f64 * 0.17).sin()
        });
        let mut dst = src.clone();
        let mut pool = ScratchPool::<f64>::new();

        let result = mc.apply_into(tau, &src, &mut dst, &mut pool);
        prop_assert!(result.is_ok(), "apply_into returned error: {:?}", result);

        // P1: NaN-free
        for &v in dst.values() {
            prop_assert!(v.is_finite(), "output contains NaN or Inf: {v}");
        }

        // P2: conservative sup-norm contractivity bound.
        // Magnus degree-4 Taylor residue is O((τρ̄)⁵), so:
        //   ‖S₄(τ)f‖ ≤ (1 + (τρ̄)⁵) · ‖f‖_∞ + 1e-12
        let tau_rho = tau * rho_bar;
        let bound = (1.0 + tau_rho.powi(5)) * src.norm_sup() + 1e-12;
        prop_assert!(
            dst.norm_sup() <= bound,
            "P2 fail: ‖S₄f‖={:.4e} > bound={:.4e}  (τ={tau:.3e}, ρ̄={rho_bar:.3}, n={n})",
            dst.norm_sup(), bound
        );
    }

    // -----------------------------------------------------------------------
    // P3: OutOfMagnusRadius error when τ·ρ̄ ≥ π/2
    // -----------------------------------------------------------------------

    /// For random N ∈ [16, 64] and w ∈ (0.2, 2.0], choose τ such that
    /// τ · rho_bar_max ≥ π/2 (deliberately exceeds the convergence radius).
    /// The kernel MUST return `SemiflowError::OutOfMagnusRadius`.
    #[test]
    fn magnus_out_of_radius_error(
        n in 16usize..64,
        w in 0.2_f64..2.0_f64,
    ) {
        // rho_bar = 2*w; π/2 / rho_bar is the radius boundary.
        let rho_bar = 2.0 * w;
        // τ = π/2 / rho_bar + 0.01 (just over the boundary).
        let tau = PI / (2.0 * rho_bar) + 0.01;

        let (mc, g) = make_mc_const_w(n, w, /* check_radius = */ true);
        let src = GraphSignal::from_fn(Arc::clone(&g), |i| {
            (i as f64 * 0.31).cos()
        });
        let mut dst = src.clone();
        let mut pool = ScratchPool::<f64>::new();

        let result = mc.apply_into(tau, &src, &mut dst, &mut pool);
        prop_assert!(
            matches!(result, Err(SemiflowError::OutOfMagnusRadius { .. })),
            "Expected OutOfMagnusRadius, got: {:?}  (τ={tau:.4}, ρ̄={rho_bar:.4})",
            result
        );
    }
}
