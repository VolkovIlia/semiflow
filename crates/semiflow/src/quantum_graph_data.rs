//! Internal helpers for quantum graph heat Chernoff (ADR-0078, math §29).
//!
//! Extracted from `quantum_graph.rs` to keep the main file within the 500-line
//! suckless limit.  All items are `pub(crate)`.

// Graph vertex/degree indices (usize) cast to f64 for Kirchhoff projector entries; ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid::Grid1D,
};

use super::quantum_graph::QuantumGraph;

// ── Kirchhoff vertex ─────────────────────────────────────────────────────────

/// Kirchhoff vertex: mean-averaging matrix `Q_v` = (1/d)·1·1^T (ADR-0078, math §29.2).
#[derive(Debug, Clone)]
pub struct KirchhoffVertex<F: SemiflowFloat = f64> {
    /// Index into `QuantumGraph::n_vertices`.
    pub vertex_index: usize,
    /// Degree (number of incident edges).
    pub degree: usize,
    /// Incident edge indices.
    pub incident_edges: Vec<usize>,
    /// Row-major d×d mean-averaging matrix Q = (1/d)·1·1^T.
    pub projection_matrix: Vec<F>,
    pub(crate) _f: PhantomData<F>,
}

impl<F: SemiflowFloat> KirchhoffVertex<F> {
    /// Build the projector at `vertex_index`.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] if `vertex_index >= graph.n_vertices`.
    pub fn new(graph: &QuantumGraph<F>, vertex_index: usize) -> Result<Self, SemiflowError> {
        if vertex_index >= graph.n_vertices {
            return Err(SemiflowError::DomainViolation {
                what: "vertex_index must be < graph.n_vertices",
                value: vertex_index as f64,
            });
        }
        let incident_edges = graph.vertex_adjacency[vertex_index].clone();
        let degree = incident_edges.len();
        let projection_matrix = build_kirchhoff_projector::<F>(degree);
        Ok(Self {
            vertex_index,
            degree,
            incident_edges,
            projection_matrix,
            _f: PhantomData,
        })
    }
}

// ── Graph construction helpers ────────────────────────────────────────────────

pub(crate) fn validate_graph_inputs<F: SemiflowFloat>(
    ep: &[(usize, usize)],
    lengths: &[F],
    n_grid: usize,
) -> Result<(), SemiflowError> {
    if ep.len() != lengths.len() {
        return Err(SemiflowError::DomainViolation {
            what: "edge_endpoints.len() must equal edge_lengths.len()",
            value: ep.len() as f64,
        });
    }
    if ep.is_empty() {
        return Err(SemiflowError::DomainViolation {
            what: "graph must have at least 1 edge",
            value: 0.0,
        });
    }
    if n_grid < 4 {
        return Err(SemiflowError::DomainViolation {
            what: "edge_n_grid_points must be >= 4 (Catmull-Rom stencil)",
            value: n_grid as f64,
        });
    }
    for (i, &len) in lengths.iter().enumerate() {
        if !len.is_finite() || len <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "all edge lengths must be finite and positive",
                value: i as f64,
            });
        }
    }
    Ok(())
}

pub(crate) fn max_vertex_index(ep: &[(usize, usize)]) -> usize {
    ep.iter().flat_map(|&(a, b)| [a, b]).max().unwrap_or(0)
}

pub(crate) fn build_edge_grids<F: SemiflowFloat>(
    lengths: &[F],
    n: usize,
) -> Result<Vec<Grid1D<F>>, SemiflowError> {
    lengths
        .iter()
        .map(|&l| Grid1D::new_generic(F::zero(), l, n))
        .collect()
}

pub(crate) fn build_adjacency(ep: &[(usize, usize)], n_v: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n_v];
    for (e, &(a, b)) in ep.iter().enumerate() {
        adj[a].push(e);
        adj[b].push(e);
    }
    adj
}

// ── Kirchhoff projection helpers ──────────────────────────────────────────────

/// d×d mean-averaging projector Q = (1/d)·1·1^T (row-major).
/// Q·1 = 1 (preserves constants, enforces V1 continuity). Distinct from
/// mean-zeroing P = I−Q used in spectral theory (`T_QG` sub-check 1).
pub(crate) fn build_kirchhoff_projector<F: SemiflowFloat>(d: usize) -> Vec<F> {
    if d == 0 {
        return Vec::new();
    }
    let inv_d = from_f64::<F>(1.0 / d as f64);
    vec![inv_d; d * d]
}

/// Phase 2: gather → GEMV with `P_v` → scatter, for each vertex with degree ≥ 2.
pub(crate) fn apply_kirchhoff_projection<F: SemiflowFloat>(
    graph: &QuantumGraph<F>,
    vertices: &[KirchhoffVertex<F>],
    dst: &mut super::quantum_graph::QuantumGraphSignal<F>,
) {
    for vtx in vertices {
        let d = vtx.degree;
        if d <= 1 {
            continue;
        }
        let mut buf: Vec<F> = vtx
            .incident_edges
            .iter()
            .map(|&e| dst.edge_endpoint_value(graph, e, vtx.vertex_index))
            .collect();
        kirchhoff_gemv_inplace(&vtx.projection_matrix, &mut buf, d);
        for (k, &e) in vtx.incident_edges.iter().enumerate() {
            dst.set_edge_endpoint_value(graph, e, vtx.vertex_index, buf[k]);
        }
    }
}

/// In-place GEMV: y ← P·y (row-major, d×d).
pub(crate) fn kirchhoff_gemv_inplace<F: SemiflowFloat>(mat: &[F], y: &mut [F], d: usize) {
    let mut out = vec![F::zero(); d];
    for row in 0..d {
        let mut acc = F::zero();
        for col in 0..d {
            acc += mat[row * d + col] * y[col];
        }
        out[row] = acc;
    }
    y.copy_from_slice(&out);
}

// ── Unit tests (moved from quantum_graph.rs to stay within 500-line limit) ────

#[cfg(test)]
// Exact float comparisons in tests verify norm_sup == 0 after exact cancellation.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::chernoff::ChernoffFunction;
    use crate::quantum_graph::{QuantumGraph, QuantumGraphHeatChernoff, QuantumGraphSignal};
    use crate::scratch::ScratchPool;
    use crate::state::State;
    use core::f64::consts::PI;

    #[test]
    fn path_graph_topology() {
        let g = QuantumGraph::<f64>::path(2, 1.0, 16).unwrap();
        assert_eq!(g.n_vertices, 3);
        assert_eq!(g.n_edges, 2);
        assert_eq!(g.vertex_adjacency[1].len(), 2);
    }

    #[test]
    fn star_graph_topology() {
        let g = QuantumGraph::<f64>::star(3, 2.0, 8).unwrap();
        assert_eq!(g.vertex_adjacency[0].len(), 3);
        assert_eq!(g.vertex_adjacency[1].len(), 1);
    }

    #[test]
    fn projector_idempotent_d2() {
        let q = build_kirchhoff_projector::<f64>(2);
        let d = 2;
        let mut qq = vec![0.0f64; d * d];
        for row in 0..d {
            for col in 0..d {
                qq[row * d + col] = (0..d).map(|k| q[row * d + k] * q[k * d + col]).sum();
            }
        }
        for i in 0..d * d {
            assert!((qq[i] - q[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn projector_preserves_constant() {
        let q = build_kirchhoff_projector::<f64>(2);
        let mut ones = vec![1.0f64; 2];
        kirchhoff_gemv_inplace(&q, &mut ones, 2);
        for r in &ones {
            assert!((r - 1.0).abs() < 1e-14, "Q·1 ≠ 1: {r}");
        }
    }

    #[test]
    fn projector_averages_unequal() {
        let q = build_kirchhoff_projector::<f64>(2);
        let mut y = vec![3.0f64, 5.0f64];
        kirchhoff_gemv_inplace(&q, &mut y, 2);
        assert!((y[0] - 4.0).abs() < 1e-14);
        assert!((y[1] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn eigenmode_k1_zero_at_midpoint() {
        let g = QuantumGraph::<f64>::path(2, 1.0, 65).unwrap();
        let sig = QuantumGraphSignal::from_eigenmode(&g, 1);
        let n = sig.per_edge[0].values.len();
        assert!(sig.per_edge[0].values[n - 1].abs() < 1e-10);
    }

    #[test]
    fn chernoff_smoke_constant_eigenmode() {
        let g = QuantumGraph::<f64>::path(2, 1.0, 16).unwrap();
        let kernel = QuantumGraphHeatChernoff::new(g.clone()).unwrap();
        let src = QuantumGraphSignal::from_eigenmode(&g, 0);
        let mut dst = src.clone();
        let mut scratch = ScratchPool::new();
        kernel
            .apply_into(0.01, &src, &mut dst, &mut scratch)
            .unwrap();
        let sup = State::<f64>::norm_sup(&dst);
        assert!(sup > 0.5 && sup <= 1.0 + 1e-10, "sup={sup}");
    }

    #[test]
    fn state_axpy_and_norm() {
        let g = QuantumGraph::<f64>::path(1, 1.0, 8).unwrap();
        let sig = QuantumGraphSignal::from_eigenmode(&g, 0);
        let mut u = sig.clone();
        State::<f64>::axpy_into(&mut u, -1.0, &sig);
        assert_eq!(State::<f64>::norm_sup(&u), 0.0);
    }

    #[test]
    fn eigenvalue_k1_decay_sanity() {
        let scale = (-(PI * PI / 8.0) * 0.1_f64).exp();
        assert!(0.8 < scale && scale < 0.95, "decay={scale}");
    }
}
