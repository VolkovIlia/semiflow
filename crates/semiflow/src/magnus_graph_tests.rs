//! Unit smoke tests for [`super::MagnusGraphHeatChernoff`].
#![allow(clippy::unwrap_used)]

use alloc::sync::Arc;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    graph::{Graph, Laplacian},
    graph_signal::GraphSignal,
    magnus_graph::{LaplacianAtTime, MagnusGraphHeatChernoff},
    scratch::ScratchPool,
    state::State,
};

fn make_path_mc(n: usize) -> MagnusGraphHeatChernoff<f64> {
    let g = Arc::new(Graph::<f64>::path(n));
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    MagnusGraphHeatChernoff::new(g, lap_at, 4.0, true).unwrap()
}

#[test]
fn order_is_four() {
    assert_eq!(make_path_mc(8).order(), 4);
}

#[test]
fn apply_at_zero_tau_returns_src() {
    let mc = make_path_mc(8);
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
    let mc = make_path_mc(4);
    let g = Arc::clone(&mc.graph);
    let src = GraphSignal::zeros(Arc::clone(&g));
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    assert!(
        matches!(
            mc.apply_into(-0.1, &src, &mut dst, &mut pool),
            Err(SemiflowError::DomainViolation { .. })
        ),
        "negative tau should return DomainViolation"
    );
}

#[test]
fn radius_violation_returns_error() {
    // rho_bar_max=4.0, tau=pi/2+0.01 → radius = 4*(pi/2+0.01) >> pi/2
    let mc = make_path_mc(4);
    let g = Arc::clone(&mc.graph);
    let src = GraphSignal::zeros(Arc::clone(&g));
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    let tau_violating = core::f64::consts::FRAC_PI_2 + 0.01; // 4 * tau > pi/2
    assert!(
        matches!(
            mc.apply_into(tau_violating, &src, &mut dst, &mut pool),
            Err(SemiflowError::OutOfMagnusRadius { .. })
        ),
        "radius violation should return OutOfMagnusRadius"
    );
}

#[test]
fn constructor_rejects_nonpositive_rho() {
    let g = Arc::new(Graph::<f64>::path(4));
    let g2 = Arc::clone(&g);
    let lap_at: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    assert!(MagnusGraphHeatChernoff::new(g, lap_at, 0.0, true).is_err());
}

#[test]
fn apply_adjoint_into_returns_unsupported() {
    // MagnusGraphHeatChernoff must NOT override apply_adjoint_into; the
    // default trait impl should return UnsupportedOperation loudly instead
    // of silently returning wrong numbers.
    let mc = make_path_mc(4);
    let g = Arc::clone(&mc.graph);
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) + 1.0);
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    assert!(
        matches!(
            mc.apply_adjoint_into(0.1, &src, &mut dst, &mut pool),
            Err(SemiflowError::UnsupportedOperation { .. })
        ),
        "apply_adjoint_into must return UnsupportedOperation, not wrong numbers"
    );
}
