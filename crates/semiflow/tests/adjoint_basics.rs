//! Integration tests for `AdjointChernoff<C, F>` — constructor, order, growth.
//!
//! Covers: `new_general` / `new_self_adjoint` constructors, `order()` caps,
//! `growth()` propagation, and `detect_self_adjointness` input validation.
//!
//! See contract wave-b-advanced-semigroups.md §1 and ADR-0055.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use std::sync::Arc;

use semiflow_core::{
    chernoff::ChernoffFunction, drift_reaction::DriftReactionChernoff,
    graph_heat::GraphHeatChernoff, graph_heat4::GraphHeat4thChernoff, AdjointChernoff, Graph,
    GraphSignal, Grid1D, Laplacian, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_path_heat_f64(n: usize) -> GraphHeatChernoff<f64> {
    let g = Arc::new(Graph::<f64>::path(n));
    let lap = Laplacian::assemble_combinatorial(&g);
    GraphHeatChernoff::from_owned(lap)
}

fn make_path_heat_f32(n: usize) -> GraphHeatChernoff<f32> {
    let g = Arc::new(Graph::<f32>::path(n));
    let lap = Laplacian::assemble_combinatorial(&g);
    GraphHeatChernoff::from_owned(lap)
}

fn make_graph_heat4_f64(n: usize) -> GraphHeat4thChernoff<f64> {
    let g = Arc::new(Graph::<f64>::path(n));
    let lap = Laplacian::assemble_combinatorial(&g);
    GraphHeat4thChernoff::from_owned(lap)
}

// ---------------------------------------------------------------------------
// Constructor tests
// ---------------------------------------------------------------------------

#[test]
fn new_general_sets_flag_false() {
    // DriftReactionChernoff implements AdjointApply<f64> (ADR-0114).
    let grid = Grid1D::new(0.0_f64, 1.0, 8).unwrap();
    let inner = DriftReactionChernoff::new(|_| 0.5_f64, |_| 0.0, 0.0, grid);
    let adj = AdjointChernoff::new_general(inner);
    assert!(!adj.is_self_adjoint());
}

#[test]
fn new_self_adjoint_sets_flag_true() {
    let adj = AdjointChernoff::new_self_adjoint(make_path_heat_f64(8));
    assert!(adj.is_self_adjoint());
}

#[test]
fn inner_borrow_returns_inner() {
    let inner = make_path_heat_f64(8);
    let inner_order = inner.order();
    let adj = AdjointChernoff::new_self_adjoint(inner);
    assert_eq!(adj.inner().order(), inner_order);
}

// ---------------------------------------------------------------------------
// order() tests (contract §1 — ADR-0055 order-preservation rule)
// ---------------------------------------------------------------------------

#[test]
fn order_self_adjoint_preserves_inner_f64() {
    let inner = make_path_heat_f64(8);
    let inner_order = inner.order();
    let adj = AdjointChernoff::new_self_adjoint(inner);
    assert_eq!(
        adj.order(),
        inner_order,
        "self-adjoint wrapper must preserve inner order()"
    );
}

#[test]
fn order_general_capped_at_2_for_order2_inner_f64() {
    // DriftReactionChernoff has order 2; general cap is min(2, 2) = 2.
    // (DriftReactionChernoff implements AdjointApply<f64> — ADR-0114.)
    let grid = Grid1D::new(0.0_f64, 1.0, 8).unwrap();
    let inner = DriftReactionChernoff::new(|_| 0.5_f64, |_| 0.0, 0.0, grid);
    let adj = AdjointChernoff::new_general(inner);
    assert_eq!(
        adj.order(),
        2,
        "general wrapper of order-2 DriftReaction inner must report order 2 (min(2,2)=2)"
    );
}

#[test]
fn order_general_capped_at_2_for_order4_inner_f64() {
    // GraphHeat4thChernoff has order 4; general cap gives min(4,2)=2.
    //
    // This type is an honest AdjointApply implementor: autonomous Padé[0,4]
    // truncated-Taylor semigroup with a constant symmetric combinatorial
    // Laplacian ⟹ S*(τ) = S(τ) exactly (ADR-0114).
    let adj = AdjointChernoff::new_general(make_graph_heat4_f64(8));
    assert_eq!(
        adj.order(),
        2,
        "general wrapper of order-4 GraphHeat4th inner must report order 2 (min(4,2)=2)"
    );
}

#[test]
fn order_self_adjoint_preserves_inner_f32() {
    let inner = make_path_heat_f32(8);
    let inner_order = inner.order();
    let adj = AdjointChernoff::new_self_adjoint(inner);
    assert_eq!(
        adj.order(),
        inner_order,
        "self-adjoint f32 wrapper must preserve inner order()"
    );
}

// ---------------------------------------------------------------------------
// growth() propagation (contract §1.2)
// ---------------------------------------------------------------------------

#[test]
fn growth_propagated_from_inner_f64() {
    let inner = make_path_heat_f64(8);
    let ig = inner.growth();
    let adj = AdjointChernoff::new_self_adjoint(inner);
    let ag = adj.growth();
    // growth() is a bit-identical copy from inner.growth().
    assert!(
        ag.multiplier.to_bits() == ig.multiplier.to_bits(),
        "growth m: {} vs {}",
        ag.multiplier,
        ig.multiplier
    );
    assert!(
        ag.omega.to_bits() == ig.omega.to_bits(),
        "growth w: {} vs {}",
        ag.omega,
        ig.omega
    );
}

#[test]
fn growth_propagated_from_inner_general_f64() {
    let inner = make_graph_heat4_f64(8);
    let ig = inner.growth();
    let adj = AdjointChernoff::new_general(inner);
    let ag = adj.growth();
    assert!(
        ag.multiplier.to_bits() == ig.multiplier.to_bits(),
        "growth m: {} vs {}",
        ag.multiplier,
        ig.multiplier
    );
    assert!(
        ag.omega.to_bits() == ig.omega.to_bits(),
        "growth w: {} vs {}",
        ag.omega,
        ig.omega
    );
}

// ---------------------------------------------------------------------------
// Smoke: self-adjoint wrapper delegates correctly at small tau
// ---------------------------------------------------------------------------

#[test]
fn self_adjoint_wrapper_smoke_f64() {
    let n = 16usize;
    let g = Arc::new(Graph::<f64>::path(n));
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| (i as f64 * 0.2).sin());
    let mut dst_inner = src.clone();
    let mut dst_adj = src.clone();
    let mut pool = ScratchPool::<f64>::new();

    let inner = make_path_heat_f64(n);
    inner
        .apply_into(0.01, &src, &mut dst_inner, &mut pool)
        .unwrap();

    let adj = AdjointChernoff::new_self_adjoint(make_path_heat_f64(n));
    adj.apply_into(0.01, &src, &mut dst_adj, &mut pool).unwrap();

    let diff: f64 = dst_inner
        .values()
        .iter()
        .zip(dst_adj.values())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        diff < 1e-14,
        "self-adjoint wrapper must match inner exactly, diff={diff:.3e}"
    );
}

// ---------------------------------------------------------------------------
// detect_self_adjointness input validation
// ---------------------------------------------------------------------------

#[test]
fn detect_validates_n_samples_zero() {
    let inner = make_path_heat_f64(4);
    assert!(AdjointChernoff::detect_self_adjointness(&inner, 0, 1e-6).is_err());
}

#[test]
fn detect_validates_tol_nonpositive() {
    let inner = make_path_heat_f64(4);
    assert!(AdjointChernoff::detect_self_adjointness(&inner, 1, 0.0).is_err());
    assert!(AdjointChernoff::detect_self_adjointness(&inner, 1, -1e-6).is_err());
}

#[test]
fn detect_valid_inputs_returns_ok() {
    let inner = make_path_heat_f64(4);
    assert!(AdjointChernoff::detect_self_adjointness(&inner, 1, 1e-6).is_ok());
}
