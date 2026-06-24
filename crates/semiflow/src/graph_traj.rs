//! Piecewise-smooth graph trajectory data structure.
//!
//! `GraphTraj<F>` encodes a sequence of K segments `[t_k, t_{k+1})`, each
//! with a frozen graph snapshot (fixed topology within segment) and a smooth
//! edge-weight closure.
//!
//! See math.md §14.1 (NORMATIVE) and ADR-0052 (design).
//!
//! ## Right-continuous convention
//!
//! At `t = breakpoints[k]` for `k ≥ 1`, `laplacian_at(t)` invokes
//! `weight_fns[k]` (matches `snapshots[k]`).

use alloc::{boxed::Box, sync::Arc, vec::Vec};

use crate::{
    error::SemiflowError,
    float::SemiflowFloat,
    graph::{Graph, Laplacian},
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of segments (resource bound per §14.1 NORMATIVE).
///
/// Reflects practical limit of CSR-segment storage on modern hardware
/// (~16 MB at `K=65_535`, average 1024-edge graphs). Constructor-enforced.
pub const MAX_GRAPH_TRAJ_SEGMENTS: usize = 65_535;

// ---------------------------------------------------------------------------
// Type alias
// ---------------------------------------------------------------------------

/// Closure type for per-segment Laplacian sampling.
///
/// `magnus_graph::LaplacianAtTime<F>` is a type alias for this type.
/// Within each segment, the closure MUST return a `Laplacian` whose
/// `row_ptr` / `col_idx` matches the segment snapshot.
pub type SegmentWeightFn<F> = Box<dyn Fn(F) -> Arc<Laplacian<F>> + Send + Sync + 'static>;

// ---------------------------------------------------------------------------
// GraphTraj<F>
// ---------------------------------------------------------------------------

/// Piecewise-smooth graph trajectory.
///
/// Constructed from K segments `[t_k, t_{k+1})`, each with a snapshot
/// graph (fixed topology within segment) and a smooth edge-weight closure.
///
/// Right-continuous convention: at `t = breakpoints[k]` for `k >= 1`,
/// `laplacian_at(t)` invokes `weight_fns[k]` (matches `snapshots[k]`).
///
/// See math.md §14.1 (NORMATIVE) and ADR-0052 (design).
pub struct GraphTraj<F: SemiflowFloat = f64> {
    breakpoints: Vec<F>,
    snapshots: Vec<Arc<Graph<F>>>,
    weight_fns: Vec<SegmentWeightFn<F>>,
}

impl<F: SemiflowFloat> GraphTraj<F> {
    /// Construct from explicit segments.
    ///
    /// # Errors
    /// - `DomainViolation { what: "breakpoints must be strictly increasing", ... }`
    /// - `DomainViolation { what: "breakpoints.len() == snapshots.len() + 1", ... }`
    /// - `DomainViolation { what: "snapshots.len() != weight_fns.len()", ... }`
    /// - `DomainViolation { what: "snapshots.len() must be in [1, MAX_GRAPH_TRAJ_SEGMENTS]", ... }`
    ///
    /// Debug-only: asserts each `weight_fns[k](breakpoints[k])` returns a
    /// Laplacian whose `row_ptr` / `col_idx` matches `snapshots[k]`.
    pub fn new(
        breakpoints: Vec<F>,
        snapshots: Vec<Arc<Graph<F>>>,
        weight_fns: Vec<SegmentWeightFn<F>>,
    ) -> Result<Self, SemiflowError> {
        validate_new_inputs(&breakpoints, &snapshots, &weight_fns)?;

        #[cfg(debug_assertions)]
        check_csr_layout_debug(&breakpoints, &snapshots, &weight_fns);

        Ok(Self {
            breakpoints,
            snapshots,
            weight_fns,
        })
    }

    /// Degenerate single-segment constructor (fixed-topology, full horizon).
    ///
    /// Equivalent to v2.1 Magnus contract input.
    ///
    /// # Errors
    /// - `DomainViolation` if `t_horizon <= 0` or non-finite.
    pub fn fixed_topology(
        graph: Arc<Graph<F>>,
        weight_fn: SegmentWeightFn<F>,
        t_horizon: F,
    ) -> Result<Self, SemiflowError> {
        if !t_horizon.is_finite() || t_horizon <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "GraphTraj::fixed_topology: t_horizon must be finite and > 0",
                value: t_horizon.to_f64().unwrap_or(f64::NAN),
            });
        }
        let breakpoints = alloc::vec![F::zero(), t_horizon];
        let snapshots = alloc::vec![graph];
        let weight_fns = alloc::vec![weight_fn];
        Ok(Self {
            breakpoints,
            snapshots,
            weight_fns,
        })
    }

    /// Read-only borrow of breakpoints. Length `n_segments() + 1`.
    #[must_use]
    pub fn breakpoints(&self) -> &[F] {
        &self.breakpoints
    }

    /// Read-only borrow of snapshot at segment `k`. `None` if `k >= n_segments()`.
    #[must_use]
    pub fn snapshot(&self, k: usize) -> Option<&Arc<Graph<F>>> {
        self.snapshots.get(k)
    }

    /// Number of segments K.
    #[must_use]
    pub fn n_segments(&self) -> usize {
        self.snapshots.len()
    }

    /// Total horizon `breakpoints[K] - breakpoints[0]`.
    #[must_use]
    pub fn t_horizon(&self) -> F {
        self.breakpoints[self.breakpoints.len() - 1] - self.breakpoints[0]
    }

    /// Locate segment containing time `t`. `None` if out of range.
    ///
    /// Right-continuous at internal breakpoints: `t = breakpoints[k]` for
    /// `k >= 1` belongs to segment `k`.
    #[must_use]
    pub fn segment_index(&self, t: F) -> Option<usize> {
        let t0 = self.breakpoints[0];
        let tk = self.breakpoints[self.breakpoints.len() - 1];
        if t < t0 || t > tk {
            return None;
        }
        // Last point: belongs to last segment (right-closed).
        if t == tk {
            return Some(self.n_segments() - 1);
        }
        // Binary search for the rightmost breakpoint <= t.
        Some(segment_index_binary(&self.breakpoints, t))
    }

    /// Sample Laplacian at time `t`. Errors if `t` out of range.
    ///
    /// # Errors
    /// - `DomainViolation` if `t < breakpoints[0]` or `t > breakpoints[K]`.
    pub fn laplacian_at(&self, t: F) -> Result<Arc<Laplacian<F>>, SemiflowError> {
        let k = self
            .segment_index(t)
            .ok_or(SemiflowError::DomainViolation {
                what: "GraphTraj::laplacian_at: t out of trajectory range",
                value: t.to_f64().unwrap_or(f64::NAN),
            })?;
        Ok((self.weight_fns[k])(t))
    }

    /// Borrow the weight function for segment `k` (crate-internal for `evolve_with_traj`).
    ///
    /// # Panics
    /// Panics if `k >= n_segments()` (debug guard — callers MUST validate first).
    pub(crate) fn weight_fns_segment(&self, k: usize) -> &SegmentWeightFn<F> {
        &self.weight_fns[k]
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate inputs for `GraphTraj::new`.
#[allow(clippy::cast_precision_loss)] // usize→f64 for error reporting only; not on hot path
fn validate_new_inputs<F: SemiflowFloat>(
    breakpoints: &[F],
    snapshots: &[Arc<Graph<F>>],
    weight_fns: &[SegmentWeightFn<F>],
) -> Result<(), SemiflowError> {
    let n_seg = snapshots.len();

    if n_seg == 0 || n_seg > MAX_GRAPH_TRAJ_SEGMENTS {
        return Err(SemiflowError::DomainViolation {
            what: "snapshots.len() must be in [1, MAX_GRAPH_TRAJ_SEGMENTS]",
            value: n_seg as f64,
        });
    }
    if breakpoints.len() != n_seg + 1 {
        return Err(SemiflowError::DomainViolation {
            what: "breakpoints.len() == snapshots.len() + 1",
            value: breakpoints.len() as f64,
        });
    }
    if weight_fns.len() != n_seg {
        return Err(SemiflowError::DomainViolation {
            what: "snapshots.len() != weight_fns.len()",
            value: weight_fns.len() as f64,
        });
    }
    // Check strictly increasing.
    for i in 1..breakpoints.len() {
        if breakpoints[i] <= breakpoints[i - 1] {
            return Err(SemiflowError::DomainViolation {
                what: "breakpoints must be strictly increasing",
                value: breakpoints[i].to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}

/// Debug-only CSR layout match check.
#[cfg(debug_assertions)]
fn check_csr_layout_debug<F: SemiflowFloat>(
    breakpoints: &[F],
    snapshots: &[Arc<Graph<F>>],
    weight_fns: &[SegmentWeightFn<F>],
) {
    for k in 0..snapshots.len() {
        let lap = (weight_fns[k])(breakpoints[k]);
        debug_assert_eq!(
            lap.row_ptr().len(),
            snapshots[k].row_ptr().len(),
            "GraphTraj: weight_fns[{k}] returns Laplacian with wrong row_ptr length"
        );
        // col_idx length check (proxy for CSR structure match).
        debug_assert_eq!(
            lap.col_idx().len(),
            snapshots[k].col_idx().len() + snapshots[k].n_nodes(),
            "GraphTraj: weight_fns[{k}] returns Laplacian with wrong col_idx length \
             (Laplacian has n_nodes extra diagonal entries vs Graph col_idx)"
        );
    }
}

/// Binary search: find rightmost segment index where `breakpoints[k] <= t`.
///
/// Precondition: `t >= breakpoints[0]` and `t < breakpoints[last]`.
fn segment_index_binary<F: SemiflowFloat>(breakpoints: &[F], t: F) -> usize {
    // We want the largest k such that breakpoints[k] <= t and k < n_seg.
    let mut lo: usize = 0;
    let mut hi: usize = breakpoints.len() - 1; // exclusive right: segment count
                                               // Binary search over segment indices 0..n_seg.
                                               // breakpoints has n_seg+1 entries; segment k covers [breakpoints[k], breakpoints[k+1]).
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if breakpoints[mid] <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Laplacian;

    #[allow(clippy::cast_precision_loss)]
    fn make_path_traj_f64(n_seg: usize) -> GraphTraj<f64> {
        let n_nodes = 4usize;
        let breakpoints: Vec<f64> = (0..=n_seg).map(|k| k as f64).collect();
        let mut snapshots = Vec::new();
        let mut weight_fns: Vec<SegmentWeightFn<f64>> = Vec::new();
        for _ in 0..n_seg {
            let g = Arc::new(Graph::<f64>::path(n_nodes));
            let g2 = Arc::clone(&g);
            snapshots.push(g);
            let wfn: SegmentWeightFn<f64> =
                Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
            weight_fns.push(wfn);
        }
        GraphTraj::new(breakpoints, snapshots, weight_fns).unwrap()
    }

    #[test]
    fn fixed_topology_degenerate() {
        let g = Arc::new(Graph::<f64>::path(8));
        let g2 = Arc::clone(&g);
        let wfn: SegmentWeightFn<f64> =
            Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
        let traj = GraphTraj::fixed_topology(g, wfn, 1.0).unwrap();
        assert_eq!(traj.n_segments(), 1);
        assert_eq!(traj.breakpoints().len(), 2);
        assert!((traj.t_horizon() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn fixed_topology_nonpositive_t_horizon_rejected() {
        let g = Arc::new(Graph::<f64>::path(4));
        let g2 = Arc::clone(&g);
        let wfn: SegmentWeightFn<f64> =
            Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
        assert!(matches!(
            GraphTraj::fixed_topology(g, wfn, 0.0),
            Err(SemiflowError::DomainViolation { .. })
        ));
    }

    #[test]
    fn new_rejects_non_increasing_breakpoints() {
        let n_nodes = 4usize;
        let g = Arc::new(Graph::<f64>::path(n_nodes));
        let g2 = Arc::clone(&g);
        let wfn: SegmentWeightFn<f64> =
            Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
        // Breakpoints [0.0, 0.5, 0.3] are not strictly increasing.
        let result = GraphTraj::new(
            alloc::vec![0.0, 0.5, 0.3],
            alloc::vec![Arc::clone(&g), Arc::new(Graph::<f64>::path(n_nodes))],
            alloc::vec![wfn, {
                let gg = Arc::new(Graph::<f64>::path(n_nodes));
                let gg2 = Arc::clone(&gg);
                Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&gg2)))
            }],
        );
        assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
    }

    #[test]
    fn segment_index_lookup() {
        let traj = make_path_traj_f64(3); // breakpoints [0,1,2,3]
        assert_eq!(traj.segment_index(0.0), Some(0));
        assert_eq!(traj.segment_index(0.5), Some(0));
        assert_eq!(traj.segment_index(1.0), Some(1)); // right-continuous
        assert_eq!(traj.segment_index(1.5), Some(1));
        assert_eq!(traj.segment_index(2.0), Some(2)); // right-continuous
        assert_eq!(traj.segment_index(3.0), Some(2)); // last point → last segment
        assert_eq!(traj.segment_index(-0.1), None);
        assert_eq!(traj.segment_index(3.1), None);
    }

    #[test]
    fn laplacian_at_in_range() {
        let traj = make_path_traj_f64(2);
        let lap = traj.laplacian_at(0.5);
        assert!(lap.is_ok());
        let lap = traj.laplacian_at(1.5);
        assert!(lap.is_ok());
    }

    #[test]
    fn laplacian_at_out_of_range_errors() {
        let traj = make_path_traj_f64(2);
        assert!(matches!(
            traj.laplacian_at(-0.1),
            Err(SemiflowError::DomainViolation { .. })
        ));
        assert!(matches!(
            traj.laplacian_at(2.5),
            Err(SemiflowError::DomainViolation { .. })
        ));
    }

    #[test]
    fn n_segments_and_breakpoints_consistent() {
        let traj = make_path_traj_f64(5);
        assert_eq!(traj.n_segments(), 5);
        assert_eq!(traj.breakpoints().len(), 6);
        assert!(traj.snapshot(4).is_some());
        assert!(traj.snapshot(5).is_none());
    }

    #[test]
    fn new_rejects_wrong_breakpoints_len() {
        let n_nodes = 4usize;
        let g = Arc::new(Graph::<f64>::path(n_nodes));
        let g2 = Arc::clone(&g);
        let wfn: SegmentWeightFn<f64> =
            Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
        // Only 1 snapshot but 3 breakpoints (requires 2 breakpoints).
        let result = GraphTraj::new(alloc::vec![0.0, 0.5, 1.0], alloc::vec![g], alloc::vec![wfn]);
        assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
    }
}
