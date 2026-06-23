//! Basic unit tests for `GraphTraj<F>` (ADR-0052).
//!
//! Covers: constructor errors, binary-search lookup, `fixed_topology` degenerate case.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use semiflow_core::{
    graph_traj::{GraphTraj, SegmentWeightFn, MAX_GRAPH_TRAJ_SEGMENTS},
    Graph, Laplacian, SemiflowError,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_lap_fn(g: Arc<Graph<f64>>) -> SegmentWeightFn<f64> {
    Box::new(move |_t: f64| Arc::new(Laplacian::assemble_combinatorial(&g)))
}

fn make_multi_segment_traj(n_seg: usize) -> GraphTraj<f64> {
    let n = 4usize;
    let breakpoints: Vec<f64> = (0..=n_seg).map(|k| k as f64).collect();
    let mut snapshots = Vec::new();
    let mut weight_fns: Vec<SegmentWeightFn<f64>> = Vec::new();
    for _ in 0..n_seg {
        let g = Arc::new(Graph::<f64>::path(n));
        weight_fns.push(make_lap_fn(Arc::clone(&g)));
        snapshots.push(g);
    }
    GraphTraj::new(breakpoints, snapshots, weight_fns).expect("valid traj")
}

// ---------------------------------------------------------------------------
// fixed_topology tests
// ---------------------------------------------------------------------------

#[test]
fn fixed_topology_single_segment() {
    let g = Arc::new(Graph::<f64>::path(8));
    let wfn = make_lap_fn(Arc::clone(&g));
    let traj = GraphTraj::fixed_topology(Arc::clone(&g), wfn, 2.0).unwrap();
    assert_eq!(traj.n_segments(), 1);
    assert_eq!(traj.breakpoints(), &[0.0_f64, 2.0]);
    assert!((traj.t_horizon() - 2.0).abs() < 1e-14);
    assert!(traj.snapshot(0).is_some());
    assert!(traj.snapshot(1).is_none());
}

#[test]
fn fixed_topology_rejects_zero_horizon() {
    let g = Arc::new(Graph::<f64>::path(4));
    let wfn = make_lap_fn(Arc::clone(&g));
    assert!(matches!(
        GraphTraj::fixed_topology(g, wfn, 0.0),
        Err(SemiflowError::DomainViolation { .. })
    ));
}

#[test]
fn fixed_topology_rejects_negative_horizon() {
    let g = Arc::new(Graph::<f64>::path(4));
    let wfn = make_lap_fn(Arc::clone(&g));
    assert!(matches!(
        GraphTraj::fixed_topology(g, wfn, -1.0),
        Err(SemiflowError::DomainViolation { .. })
    ));
}

// ---------------------------------------------------------------------------
// Constructor validation tests
// ---------------------------------------------------------------------------

#[test]
fn new_rejects_empty_snapshots() {
    let result = GraphTraj::<f64>::new(vec![0.0, 1.0], vec![], vec![]);
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}

#[test]
fn new_rejects_breakpoints_len_mismatch() {
    let g = Arc::new(Graph::<f64>::path(4));
    let wfn = make_lap_fn(Arc::clone(&g));
    // 1 snapshot requires exactly 2 breakpoints; giving 3 is wrong.
    let result = GraphTraj::new(vec![0.0, 0.5, 1.0], vec![g], vec![wfn]);
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}

#[test]
fn new_rejects_weight_fns_len_mismatch() {
    let g1 = Arc::new(Graph::<f64>::path(4));
    let g2 = Arc::new(Graph::<f64>::path(4));
    let wfn1 = make_lap_fn(Arc::clone(&g1));
    // 2 snapshots but only 1 weight_fn.
    let result = GraphTraj::new(vec![0.0, 0.5, 1.0], vec![g1, g2], vec![wfn1]);
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}

#[test]
fn new_rejects_non_strictly_increasing_breakpoints() {
    let g1 = Arc::new(Graph::<f64>::path(4));
    let g2 = Arc::new(Graph::<f64>::path(4));
    let wfn1 = make_lap_fn(Arc::clone(&g1));
    let wfn2 = make_lap_fn(Arc::clone(&g2));
    let result = GraphTraj::new(
        vec![0.0, 1.0, 0.5], // not strictly increasing
        vec![g1, g2],
        vec![wfn1, wfn2],
    );
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}

#[test]
fn new_rejects_equal_breakpoints() {
    let g1 = Arc::new(Graph::<f64>::path(4));
    let g2 = Arc::new(Graph::<f64>::path(4));
    let wfn1 = make_lap_fn(Arc::clone(&g1));
    let wfn2 = make_lap_fn(Arc::clone(&g2));
    let result = GraphTraj::new(
        vec![0.0, 0.5, 0.5], // equal = not strictly increasing
        vec![g1, g2],
        vec![wfn1, wfn2],
    );
    assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
}

// ---------------------------------------------------------------------------
// segment_index / binary-search tests
// ---------------------------------------------------------------------------

#[test]
fn segment_index_single_segment() {
    let traj = make_multi_segment_traj(1);
    assert_eq!(traj.segment_index(0.0), Some(0));
    assert_eq!(traj.segment_index(0.5), Some(0));
    assert_eq!(traj.segment_index(1.0), Some(0)); // right endpoint
    assert_eq!(traj.segment_index(-0.1), None);
    assert_eq!(traj.segment_index(1.1), None);
}

#[test]
fn segment_index_right_continuous_at_internal_breakpoints() {
    let traj = make_multi_segment_traj(3); // [0,1,2,3]
    assert_eq!(traj.segment_index(0.0), Some(0)); // left endpoint of [0,1)
    assert_eq!(traj.segment_index(0.999), Some(0));
    assert_eq!(traj.segment_index(1.0), Some(1)); // right-continuous: t=1 → segment 1
    assert_eq!(traj.segment_index(1.5), Some(1));
    assert_eq!(traj.segment_index(2.0), Some(2)); // right-continuous: t=2 → segment 2
    assert_eq!(traj.segment_index(3.0), Some(2)); // right endpoint → last segment
}

#[test]
fn segment_index_out_of_range() {
    let traj = make_multi_segment_traj(2); // [0,1,2]
    assert_eq!(traj.segment_index(-0.01), None);
    assert_eq!(traj.segment_index(2.001), None);
}

#[test]
fn segment_index_five_segments() {
    let traj = make_multi_segment_traj(5); // [0,1,2,3,4,5]
    for seg in 0..5usize {
        let t_mid = seg as f64 + 0.5;
        assert_eq!(traj.segment_index(t_mid), Some(seg));
        let t_left = seg as f64;
        if seg > 0 {
            // Right-continuous: breakpoint belongs to next segment.
            assert_eq!(traj.segment_index(t_left), Some(seg));
        } else {
            assert_eq!(traj.segment_index(t_left), Some(0));
        }
    }
}

// ---------------------------------------------------------------------------
// laplacian_at tests
// ---------------------------------------------------------------------------

#[test]
fn laplacian_at_returns_ok_in_range() {
    let traj = make_multi_segment_traj(2);
    assert!(traj.laplacian_at(0.0).is_ok());
    assert!(traj.laplacian_at(0.5).is_ok());
    assert!(traj.laplacian_at(1.0).is_ok());
    assert!(traj.laplacian_at(2.0).is_ok()); // right endpoint
}

#[test]
fn laplacian_at_errors_out_of_range() {
    let traj = make_multi_segment_traj(2);
    assert!(matches!(
        traj.laplacian_at(-0.1),
        Err(SemiflowError::DomainViolation { .. })
    ));
    assert!(matches!(
        traj.laplacian_at(2.5),
        Err(SemiflowError::DomainViolation { .. })
    ));
}

// ---------------------------------------------------------------------------
// Accessor tests
// ---------------------------------------------------------------------------

#[test]
fn n_segments_and_breakpoints_consistent() {
    let traj = make_multi_segment_traj(7);
    assert_eq!(traj.n_segments(), 7);
    assert_eq!(traj.breakpoints().len(), 8);
}

#[test]
fn snapshot_returns_none_out_of_range() {
    let traj = make_multi_segment_traj(3);
    assert!(traj.snapshot(0).is_some());
    assert!(traj.snapshot(2).is_some());
    assert!(traj.snapshot(3).is_none());
}

#[test]
fn t_horizon_is_last_minus_first() {
    let traj = make_multi_segment_traj(4);
    assert!((traj.t_horizon() - 4.0).abs() < 1e-14);
}

#[test]
fn max_segments_constant_is_correct() {
    assert_eq!(MAX_GRAPH_TRAJ_SEGMENTS, 65_535);
}
