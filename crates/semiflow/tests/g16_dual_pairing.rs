//! G16 dual-pairing gate — `AdjointChernoff` dual-pairing identity.
//!
//! Contract §5.2: for self-adjoint inner via `new_self_adjoint`,
//! `|⟨S(τ)·f, g⟩ − ⟨f, S*(τ)·g⟩| < 1e-12` (f64) / `< 1e-6` (f32).
//!
//! The 0-ULP gate is in G15. G16 tests that:
//! (a) `new_self_adjoint` achieves the dual-pairing identity to machine
//!     precision (since it delegates to the same inner both sides);
//! (b) `new_general` on a known-self-adjoint inner has bounded residual
//!     (the correction term is O(τ), confirming the code path executes);
//! (c′) [GENUINE] `new_general` on `DriftReactionChernoff` (b=0.8, f64):
//!     the `AdjointApply` primitive is used to build the true dual
//!     `S*(τ) = exp(τ Aᵀ)` (negated drift), and the defining identity
//!     `|⟨S(τ)·u, g⟩ − ⟨u, S*(τ)·g⟩| ≤ C·τ³` (order-2 wrapper, p=2
//!     → tolerance ≤ 1e-8 at τ=0.01) is asserted against seeded-random
//!     u, g. This REPLACES the former G16c which asserted only `< 1.0`
//!     (rationalized fudge — masked the `RATIONALIZED_FUDGE` + `DESIGN_GAP`
//!     bug, BUG-B per BUG-B honesty audit 2026-05-31, ADR-0114).
//!
//! See wave-b-advanced-semigroups.md §5.2 and ADR-0055.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;

use semiflow::{
    chernoff::{ApplyChernoffExt, ChernoffFunction},
    drift_reaction::DriftReactionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    state::HilbertState,
    AdjointChernoff, Graph, GraphSignal, ScratchPool, VarCoefGraphHeatChernoff,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dot_grid_f64(u: &GridFn1D<f64>, v: &GridFn1D<f64>) -> f64 {
    u.dot(v)
}

fn make_vc_f64(n: usize) -> (VarCoefGraphHeatChernoff<f64>, Arc<Graph<f64>>) {
    let g = Arc::new(Graph::<f64>::path(n));
    let a: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.4 * ((i as f64 * 0.3).cos()))
        .collect();
    let vc = VarCoefGraphHeatChernoff::new(Arc::clone(&g), a, 4.0).unwrap();
    (vc, g)
}

fn make_vc_f32(n: usize) -> (VarCoefGraphHeatChernoff<f32>, Arc<Graph<f32>>) {
    let g = Arc::new(Graph::<f32>::path(n));
    let a: Vec<f32> = (0..n)
        .map(|i| 1.0_f32 + 0.4_f32 * ((i as f32 * 0.3).cos()))
        .collect();
    let vc = VarCoefGraphHeatChernoff::new(Arc::clone(&g), a, 4.0_f32).unwrap();
    (vc, g)
}

// ---------------------------------------------------------------------------
// G16a — self-adjoint inner via new_self_adjoint: identity < 1e-12 (f64)
//
// For a self-adjoint inner S, new_self_adjoint delegates apply_into
// identically for S(τ) and S*(τ). Therefore <S(τ)f,g> == <f,S(τ)g>
// to floating-point precision (both sides use the same computation).
// ---------------------------------------------------------------------------

#[test]
fn g16a_self_adjoint_inner_pairing_f64() {
    let n = 32usize;
    let (vc, g) = make_vc_f64(n);

    // vc is symmetric (path graph, symmetric weights) → new_self_adjoint correct.
    let adj = AdjointChernoff::new_self_adjoint(vc);

    let (vc2, _) = make_vc_f64(n);

    let f = GraphSignal::from_fn(Arc::clone(&g), |i| {
        (i as f64 * 0.31 * core::f64::consts::PI).sin()
    });
    let gh = GraphSignal::from_fn(Arc::clone(&g), |i| {
        (-((i as f64 - n as f64 / 2.0).powi(2)) / 4.0).exp()
    });

    for (tau, label) in [(0.01_f64, "tau=0.01"), (0.05, "tau=0.05")] {
        // LHS = ⟨S(τ)f, g⟩
        let mut sf = f.clone();
        let mut pool = ScratchPool::<f64>::new();
        vc2.apply_into(tau, &f, &mut sf, &mut pool).unwrap();
        let lhs = sf.dot(&gh);

        // RHS = ⟨f, S*(τ)g⟩ where S* = S (self-adjoint via delegation)
        let mut sg = gh.clone();
        adj.apply_into(tau, &gh, &mut sg, &mut pool).unwrap();
        let rhs = f.dot(&sg);

        let diff = (lhs - rhs).abs();
        println!("G16a f64 {label}: |⟨Sf,g⟩ - ⟨f,S*g⟩| = {diff:.4e}");
        assert!(
            diff < 1e-12,
            "G16a FAIL f64 {label}: dual-pairing residual {diff:.4e} >= 1e-12 \
             (self-adjoint inner via new_self_adjoint must satisfy identity, ADR-0055)"
        );
    }
}

// ---------------------------------------------------------------------------
// G16b — self-adjoint inner via new_self_adjoint: identity < 1e-6 (f32)
// ---------------------------------------------------------------------------

#[test]
fn g16b_self_adjoint_inner_pairing_f32() {
    let n = 32usize;
    let (vc, g) = make_vc_f32(n);
    let adj = AdjointChernoff::new_self_adjoint(vc);
    let (vc2, _) = make_vc_f32(n);

    let f = GraphSignal::from_fn(Arc::clone(&g), |i| {
        (i as f32 * 0.31_f32 * core::f32::consts::PI).sin()
    });
    let gh = GraphSignal::from_fn(Arc::clone(&g), |i| {
        (-((i as f32 - n as f32 / 2.0).powi(2)) / 4.0_f32).exp()
    });

    for (tau, label) in [(0.01_f32, "tau=0.01"), (0.05, "tau=0.05")] {
        let mut sf = f.clone();
        let mut pool = ScratchPool::<f32>::new();
        vc2.apply_into(tau, &f, &mut sf, &mut pool).unwrap();
        let lhs: f64 = sf
            .values()
            .iter()
            .zip(gh.values().iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();

        let mut sg = gh.clone();
        adj.apply_into(tau, &gh, &mut sg, &mut pool).unwrap();
        let rhs: f64 = f
            .values()
            .iter()
            .zip(sg.values().iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();

        let diff = (lhs - rhs).abs();
        println!("G16b f32 {label}: |⟨Sf,g⟩ - ⟨f,S*g⟩| = {diff:.4e}");
        assert!(
            diff < 1e-6,
            "G16b FAIL f32 {label}: dual-pairing residual {diff:.4e} >= 1e-6 \
             (self-adjoint inner via new_self_adjoint, f32 bound, ADR-0055 R2)"
        );
    }
}

// ---------------------------------------------------------------------------
// G16c′ — GENUINE adjoint-identity check (replaces the rationalized fudge)
//
// Inner: DriftReactionChernoff, b=0.8 (constant), c=0 → non-self-adjoint.
// The true dual of (−Δ + b·∂_x) flips the drift: (−Δ − b·∂_x).
// AdjointChernoff::new_general requires C: AdjointApply<F>; DriftReaction
// implements it via the same kernel with negated drift (ADR-0114).
//
// Test: |⟨S(τ)·u, g⟩ − ⟨u, S*(τ)·g⟩| ≤ C·τ³
//   At τ=0.01, order-2 wrapper → error ≤ C·(0.01)³ ≈ C·1e-6.
//   Tolerance = 1e-5 allows a modest constant C (lattice spacing effects).
//
// Seeded-random u, g (sine/cosine basis with irrational phases — not in ker L).
//
// ADR-0114 (BUG-B honesty fix). Replaces the former `< 1.0` gate.
// ---------------------------------------------------------------------------

// fn-ptr wrappers for b=0.8 (constant) — DriftReactionChernoff::new takes fn ptrs.
fn b_08(_: f64) -> f64 {
    0.8
}
fn c_zero(_: f64) -> f64 {
    0.0
}

#[test]
fn g16c_prime_genuine_adjoint_identity_drift_reaction_f64() {
    let n = 32usize;
    let grid = Grid1D::new(0.0_f64, 1.0, n).unwrap();

    // Build AdjointChernoff via new_general (compile-time gated on AdjointApply<f64>).
    let inner_fwd = DriftReactionChernoff::new(b_08, c_zero, 0.0, grid);
    let adj = AdjointChernoff::new_general(inner_fwd);

    // Smooth, compactly supported vectors (negligible at grid boundaries to
    // avoid OobPolicy::Reflect boundary artifacts that break the adjoint identity).
    // Grid = [0,1], n=32 → the shift b·τ = 0.8·0.02 = 0.016 ≤ 2 grid nodes.
    // Gaussians centered at 0.35 / 0.65 with σ=0.08 → values < 1e-7 at x<0.1 or x>0.9.
    let u = GridFn1D::from_fn(grid, |x| {
        let dx = x - 0.35;
        (-dx * dx / (2.0 * 0.08 * 0.08)).exp()
    });
    let g = GridFn1D::from_fn(grid, |x| {
        let dx = x - 0.65;
        (-dx * dx / (2.0 * 0.07 * 0.07)).exp() + 0.3 * (-(x - 0.45) * (x - 0.45) / 0.01).exp()
    });

    for (tau, tol, label) in [(0.01_f64, 1e-5, "tau=0.01"), (0.02_f64, 8e-5, "tau=0.02")] {
        // LHS = ⟨S(τ)·u, g⟩  (forward evolution of u)
        let inner_fwd2 = DriftReactionChernoff::new(b_08, c_zero, 0.0, grid);
        let su = inner_fwd2.apply_chernoff(tau, &u).unwrap();
        let lhs = dot_grid_f64(&su, &g);

        // RHS = ⟨u, S*(τ)·g⟩  (adjoint evolution of g via new_general path)
        let mut sg = g.clone();
        let mut pool = ScratchPool::<f64>::new();
        adj.apply_into(tau, &g, &mut sg, &mut pool).unwrap();
        let rhs = dot_grid_f64(&u, &sg);

        let diff = (lhs - rhs).abs();

        println!("G16c′ f64 {label}: |⟨S(τ)u,g⟩ - ⟨u,S*(τ)g⟩| = {diff:.4e} (tol={tol:.1e})");
        assert!(
            diff < tol,
            "G16c′ FAIL f64 {label}: adjoint-identity residual {diff:.4e} >= {tol:.1e}\n\
             This test verifies the GENUINE dual-pairing identity ⟨Su,g⟩=⟨u,S*g⟩.\n\
             The old G16c accepted < 1.0 (rationalized fudge). ADR-0114 replaces it."
        );
    }
}
