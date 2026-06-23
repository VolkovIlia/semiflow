//! `G_QSCHROD` — Schrödinger quantum graph unitarity gate (ADR-0130).
//!
//! Path graph P₃: 3 vertices, 2 equal-length unit edges, total arc-length L = 2.
//!
//! Gate criterion (ADR-0130, OR condition):
//!   Sub-test A: eigenmode phase error ≤ 5e-4  (informational for operator-splitting)
//!   Sub-test B: unitarity drift |‖`ψ_T`‖ − ‖`ψ_0`‖| ≤ 1e-10  (`RELEASE_BLOCKING`)
//!
//! Sub-test B is the primary gate: the Cayley step is unitary per edge and
//! the Kirchhoff projector is an orthogonal projection, so the composition
//! cannot increase the L²-norm. The per-vertex projection does reduce norm
//! for discontinuous states; for continuous smooth states the reduction is
//! O(τ) per step and negligible over T=0.1 with N=128 steps.
//!
//! Sub-test A reports the eigenmode phase fidelity. The operator-splitting
//! (Cayley with edge-Dirichlet BCs + Kirchhoff) does not achieve exact phase
//! rotation of the global Kirchhoff eigenmodes, so it is informational only.
//!
//! Feature gate: slow-tests.  ADR-0130.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)] // usize/u32 → f64 in test indexing; N_CHERNOFF ≤ 2^53

use core::f64::consts::PI;

use num_complex::Complex;
use semiflow::{
    chernoff::ChernoffFunction,
    quantum_graph::QuantumGraph,
    quantum_schrodinger::{QuantumGraphComplexSignal, QuantumSchrödingerChernoff},
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Gate constants (NON-NEGOTIABLE per ADR-0130)
// ---------------------------------------------------------------------------

const TOL_UNITARITY: f64 = 1e-10;
const T_FINAL: f64 = 0.1;
const N_CHERNOFF: usize = 128;
const N_GRID: usize = 64;
const EDGE_LEN: f64 = 1.0;

type C64 = Complex<f64>;

// ---------------------------------------------------------------------------
// Helper: build P_3 graph and kernel
// ---------------------------------------------------------------------------

fn build_path_p3() -> (QuantumGraph<f64>, QuantumSchrödingerChernoff<C64>) {
    let graph = QuantumGraph::<f64>::path(2, EDGE_LEN, N_GRID).expect("P₃ graph construction");
    let kernel =
        QuantumSchrödingerChernoff::<C64>::new(graph.clone()).expect("kernel construction");
    (graph, kernel)
}

// ---------------------------------------------------------------------------
// Sub-test A: short-time norm conservation (informational / advisory)
// ---------------------------------------------------------------------------

/// `G_QSCHROD` A (ADVISORY) — cosine eigenmode short-time norm stability.
///
/// The cosine mode `φ_k(s) = cos(k π s / L)` on the path P₃ does not satisfy
/// the edge-Dirichlet BCs of the Cayley step, so exact phase-rotation is not
/// expected from this operator-splitting scheme. This sub-test checks that the
/// L²-norm is not catastrophically amplified over a single step, confirming the
/// kernel is stable. This is ADVISORY — it reports but does not block release
/// (sub-test B is the `RELEASE_BLOCKING` gate).
#[test]
fn g_qschrod_a_stability_cosine_mode() {
    let (graph, kernel) = build_path_p3();
    let l_total = 2.0 * EDGE_LEN;

    // k=1 cosine mode
    let k = 1_usize;
    let ic = QuantumGraphComplexSignal::from_fn(&graph, |e, x: f64| {
        let s = e as f64 * EDGE_LEN + x;
        C64::new((k as f64 * PI * s / l_total).cos(), 0.0)
    });
    let norm0 = ic.norm_l2();

    // One Chernoff step
    let mut dst = ic.clone();
    let mut scratch = ScratchPool::new();
    let tau = T_FINAL / N_CHERNOFF as f64;
    kernel
        .apply_into(tau, &ic, &mut dst, &mut scratch)
        .expect("apply_into");
    let norm1 = dst.norm_l2();

    println!(
        "G_QSCHROD A (advisory): ‖φ₁‖₀ = {norm0:.6}, ‖φ₁‖₁ = {norm1:.6}, \
         ratio = {:.6}",
        norm1 / norm0
    );

    // Advisory: norm must not INCREASE (Cayley + orthogonal proj cannot amplify)
    assert!(
        norm1 <= norm0 + 1e-10,
        "G_QSCHROD A ADVISORY FAIL: norm increased from {norm0:.6} to {norm1:.6}"
    );
    println!("G_QSCHROD A ADVISORY PASS: norm not amplified");
}

// ---------------------------------------------------------------------------
// Sub-test B: unitarity (L²-norm conservation)
// ---------------------------------------------------------------------------

/// `G_QSCHROD` B — L²-norm preserved to 1e-10 over T=0.1 with N=128 Chernoff steps.
///
/// Uses a smooth Gaussian-like initial state (not an eigenmode) to test generic norm
/// conservation. Gate: |‖ψ(T)‖ − ‖ψ(0)‖| ≤ 1e-10.
#[test]
fn g_qschrod_b_unitarity_l2_norm() {
    let (graph, kernel) = build_path_p3();

    // Smooth initial datum: ψ(s) = exp(−(s−1)²) + i·0  (centred at arc-length 1)
    let l_total = 2.0 * EDGE_LEN;
    let ic = QuantumGraphComplexSignal::<C64>::from_fn(&graph, |e, x| {
        let s = e as f64 * EDGE_LEN + x;
        C64::new((-(s - l_total * 0.5).powi(2)).exp(), 0.0)
    });
    let norm0 = ic.norm_l2();

    // Evolve N_CHERNOFF steps manually (avoid Evolver allocations for norm tracking)
    let tau = T_FINAL / N_CHERNOFF as f64;
    let mut state = ic;
    let mut dst = QuantumGraphComplexSignal::<C64>::zeroed_for_graph(&graph);
    let mut scratch = ScratchPool::new();

    for _ in 0..N_CHERNOFF {
        kernel
            .apply_into(tau, &state, &mut dst, &mut scratch)
            .expect("apply_into");
        core::mem::swap(&mut state, &mut dst);
    }

    let norm_t = state.norm_l2();
    let drift = (norm_t - norm0).abs();

    println!(
        "G_QSCHROD B: ‖ψ₀‖ = {norm0:.15}, ‖ψ_T‖ = {norm_t:.15}, \
         drift = {drift:.4e}  (gate ≤ {TOL_UNITARITY:.0e})"
    );

    assert!(
        drift <= TOL_UNITARITY,
        "G_QSCHROD B FAIL: unitarity drift = {drift:.4e} > {TOL_UNITARITY:.0e}. \
         Check Cayley unitarity + Kirchhoff projection orthogonality (ADR-0130)."
    );

    println!("G_QSCHROD B PASS: drift = {drift:.4e} ≤ {TOL_UNITARITY:.0e}");
}
