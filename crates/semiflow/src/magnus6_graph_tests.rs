use alloc::sync::Arc;

use super::*;
use crate::{
    chernoff::ChernoffFunction,
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    state::State,
};

fn make_path_k6(n: usize) -> MagnusGraphHeat6thChernoff<f64> {
    let g = Arc::new(Graph::<f64>::path(n));
    let g2 = Arc::clone(&g);
    let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
    let lap_at: LaplacianAtTime<f64> = Box::new(move |_t| Arc::clone(&lap));
    let _ = g2;
    MagnusGraphHeat6thChernoff::new(g, lap_at, 4.0, true).unwrap()
}

#[test]
fn order_is_six() {
    assert_eq!(make_path_k6(8).order(), 6);
}

#[test]
fn gl6_abscissae_sum_to_one() {
    // c1 + c3 = 1 (symmetric around 0.5).
    let sum = GL6_C1 + GL6_C3;
    assert!((sum - 1.0).abs() < 1e-14, "c1+c3 should be 1, got {sum}");
    // c2 = 0.5
    assert!((GL6_C2 - 0.5).abs() < 1e-14);
}

#[test]
fn gl6_weights_sum_to_one() {
    let sum = GL6_B1 + GL6_B2 + GL6_B3;
    assert!(
        (sum - 1.0).abs() < 1e-14,
        "weights should sum to 1, got {sum}"
    );
}

#[test]
fn apply_at_zero_tau_returns_src() {
    let mc = make_path_k6(8);
    let g = Arc::clone(&mc.graph);
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    mc.apply_into(0.0, &src, &mut dst, &mut pool).unwrap();
    let mut diff = dst.clone();
    diff.axpy_into(-1.0, &src);
    assert!(
        diff.norm_sup() < 1e-14,
        "zero-tau should preserve src, got sup_diff = {}",
        diff.norm_sup()
    );
}

#[test]
fn negative_tau_returns_error() {
    let mc = make_path_k6(4);
    let g = Arc::clone(&mc.graph);
    let src = GraphSignal::zeros(Arc::clone(&g));
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    assert!(matches!(
        mc.apply_into(-0.01, &src, &mut dst, &mut pool),
        Err(SemiflowError::DomainViolation { .. })
    ));
}

#[test]
fn radius_violation_returns_error() {
    let mc = make_path_k6(4);
    let g = Arc::clone(&mc.graph);
    let src = GraphSignal::zeros(Arc::clone(&g));
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    // rho_bar_max = 4.0; tau = pi/8 + 0.1 → product > pi/2.
    let tau_over = core::f64::consts::FRAC_PI_2 / 4.0 + 0.1;
    assert!(matches!(
        mc.apply_into(tau_over, &src, &mut dst, &mut pool),
        Err(SemiflowError::OutOfMagnusRadius { .. })
    ));
}

#[test]
fn constructor_rejects_nonpositive_rho() {
    let g = Arc::new(Graph::<f64>::path(4));
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    assert!(MagnusGraphHeat6thChernoff::new(g, lap_at, 0.0, true).is_err());
}

#[test]
fn constructor_rejects_empty_graph() {
    // Graph::from_edges with 0 nodes.
    let edges: Vec<(u32, u32, f64)> = vec![];
    let g = Arc::new(Graph::from_edges(0, edges).unwrap());
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    assert!(MagnusGraphHeat6thChernoff::new(g, lap_at, 1.0, true).is_err());
}
