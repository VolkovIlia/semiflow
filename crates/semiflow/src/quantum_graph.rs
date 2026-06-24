//! v3.1 B7 — Quantum graphs, Kirchhoff vertex condition (math §29, ADR-0078).
//!
//! Heat on a metric graph: `u_t` = `½∂²_x` u per edge, u continuous at vertices,
//! Σ ∂_x u (in-flowing) = 0 (Kirchhoff). Algorithm (math §29.3): edgewise heat
//! via `ShiftChernoff1D<F>` + per-vertex orthogonal projection `P_v` = I - (1/d)·1·1^T.
//!
//! Cite: Friedlander 2005 *Ann. Inst. Fourier* + Kuchment 2004 *Waves Random Media*.
//!
//! Internal helpers (`KirchhoffVertex`, projector arithmetic, graph builders) live in
//! `quantum_graph_data` to keep this file within the 500-line suckless limit.

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;
use alloc::{vec, vec::Vec};

pub use crate::quantum_graph_data::KirchhoffVertex;
use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    grid::Grid1D,
    grid_fn::GridFn1D,
    quantum_graph_data::{
        apply_kirchhoff_projection, build_adjacency, build_edge_grids, max_vertex_index,
        validate_graph_inputs,
    },
    scratch::ScratchPool,
    shift1d::ShiftChernoff1D,
    state::State,
};

// ── Public types ──────────────────────────────────────────────────────────────

/// Metric graph: vertices + edges with lengths + per-edge grids (ADR-0078, math §29.6).
#[derive(Debug, Clone)]
pub struct QuantumGraph<F: SemiflowFloat = f64> {
    /// Number of vertices.
    pub n_vertices: usize,
    /// Number of edges.
    pub n_edges: usize,
    /// Per-edge (`vertex_a`, `vertex_b`).
    pub edge_endpoints: Vec<(usize, usize)>,
    /// Per-edge lengths `ℓ_i` > 0.
    pub edge_lengths: Vec<F>,
    /// Per-edge grids over [0, `ℓ_i`].
    pub edge_grids: Vec<Grid1D<F>>,
    /// Per-vertex incident edge indices.
    pub vertex_adjacency: Vec<Vec<usize>>,
}

impl<F: SemiflowFloat> QuantumGraph<F> {
    /// Build from edge endpoints, lengths, and grid resolution.
    ///
    /// # Errors
    /// Returns [`SemiflowError`] on endpoint/length mismatch, non-positive length, or `n < 4`.
    pub fn new(
        edge_endpoints: Vec<(usize, usize)>,
        edge_lengths: Vec<F>,
        edge_n_grid_points: usize,
    ) -> Result<Self, SemiflowError> {
        validate_graph_inputs(&edge_endpoints, &edge_lengths, edge_n_grid_points)?;
        let n_edges = edge_endpoints.len();
        let n_vertices = max_vertex_index(&edge_endpoints) + 1;
        let edge_grids = build_edge_grids(&edge_lengths, edge_n_grid_points)?;
        let vertex_adjacency = build_adjacency(&edge_endpoints, n_vertices);
        Ok(Self {
            n_vertices,
            n_edges,
            edge_endpoints,
            edge_lengths,
            edge_grids,
            vertex_adjacency,
        })
    }

    /// Path graph P_{n+1}: `n_edges+1` vertices, equal-length edges.
    /// G30 oracle: `path(2, 1.0, 64)`.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] if `n_edges == 0`, or from `new`.
    pub fn path(n_edges: usize, edge_length: F, n_grid: usize) -> Result<Self, SemiflowError> {
        if n_edges == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "path graph requires n_edges >= 1",
                value: 0.0,
            });
        }
        Self::new(
            (0..n_edges).map(|i| (i, i + 1)).collect(),
            vec![edge_length; n_edges],
            n_grid,
        )
    }

    /// Star graph: central vertex 0 + `n_arms` leaf vertices.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] if `n_arms == 0`, or from `new`.
    pub fn star(n_arms: usize, edge_length: F, n_grid: usize) -> Result<Self, SemiflowError> {
        if n_arms == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "star graph requires n_arms >= 1",
                value: 0.0,
            });
        }
        Self::new(
            (0..n_arms).map(|i| (0, i + 1)).collect(),
            vec![edge_length; n_arms],
            n_grid,
        )
    }
}

// ── QuantumGraphSignal ─────────────────────────────────────────────────────────

/// Per-edge signal for `QuantumGraphHeatChernoff` (ADR-0078, math §29.6).
#[derive(Debug, Clone)]
pub struct QuantumGraphSignal<F: SemiflowFloat = f64> {
    /// Per-edge sampled functions. Length == `n_edges`.
    pub per_edge: Vec<GridFn1D<F>>,
}

impl<F: SemiflowFloat> QuantumGraphSignal<F> {
    /// Zero signal sized to `graph`.
    #[must_use]
    pub fn zeroed_for_graph(graph: &QuantumGraph<F>) -> Self {
        let per_edge = graph
            .edge_grids
            .iter()
            .map(|g| GridFn1D::from_fn_generic(*g, |_| F::zero()))
            .collect();
        Self { per_edge }
    }

    /// Sample `f(edge_index, x_along_edge)` on each edge.
    pub fn from_fn(graph: &QuantumGraph<F>, mut f: impl FnMut(usize, F) -> F) -> Self {
        let per_edge = graph
            .edge_grids
            .iter()
            .enumerate()
            .map(|(e, g)| GridFn1D::from_fn_generic(*g, |x| f(e, x)))
            .collect();
        Self { per_edge }
    }

    /// Friedlander eigenmode `φ_k(s)` = cos(k·π·s/L) on arc-length s (math §29.4).
    #[must_use]
    pub fn from_eigenmode(graph: &QuantumGraph<F>, k: usize) -> Self {
        let total = total_arc_length(graph);
        let pi = from_f64::<F>(core::f64::consts::PI);
        let k_f = from_f64::<F>(k as f64);
        let mut offset = F::zero();
        let per_edge = graph
            .edge_grids
            .iter()
            .enumerate()
            .map(|(e, g)| {
                let off = offset;
                offset += graph.edge_lengths[e];
                GridFn1D::from_fn_generic(*g, |x| (k_f * pi * (off + x) / total).cos())
            })
            .collect();
        Self { per_edge }
    }

    /// Scaled eigenmode `scale · φ_k(s)` — G30 reference solution.
    pub fn from_eigenmode_scaled(graph: &QuantumGraph<F>, k: usize, scale: F) -> Self {
        let mut s = Self::from_eigenmode(graph, k);
        for e in &mut s.per_edge {
            for v in &mut e.values {
                *v *= scale;
            }
        }
        s
    }

    /// Value at the endpoint of edge `e` incident to vertex `v`.
    #[inline]
    #[must_use]
    pub fn edge_endpoint_value(&self, graph: &QuantumGraph<F>, e: usize, v: usize) -> F {
        let vals = &self.per_edge[e].values;
        if graph.edge_endpoints[e].0 == v {
            vals[0]
        } else {
            vals[vals.len() - 1]
        }
    }

    /// Set the value at the endpoint of edge `e` incident to vertex `v`.
    #[inline]
    pub fn set_edge_endpoint_value(&mut self, graph: &QuantumGraph<F>, e: usize, v: usize, val: F) {
        let vals = &mut self.per_edge[e].values;
        if graph.edge_endpoints[e].0 == v {
            vals[0] = val;
        } else {
            let n = vals.len();
            vals[n - 1] = val;
        }
    }
}

impl<F: SemiflowFloat> State<F> for QuantumGraphSignal<F> {
    fn len(&self) -> usize {
        self.per_edge.iter().map(|e| e.values.len()).sum()
    }

    fn axpy_into(&mut self, alpha: F, src: &Self) {
        for (d, s) in self.per_edge.iter_mut().zip(src.per_edge.iter()) {
            State::<F>::axpy_into(d, alpha, s);
        }
    }

    fn copy_from(&mut self, src: &Self) {
        for (d, s) in self.per_edge.iter_mut().zip(src.per_edge.iter()) {
            State::<F>::copy_from(d, s);
        }
    }

    fn zero_into(&mut self) {
        for e in &mut self.per_edge {
            State::<F>::zero_into(e);
        }
    }

    fn norm_sup(&self) -> F {
        self.per_edge.iter().fold(F::zero(), |acc, e| {
            let n = State::<F>::norm_sup(e);
            if n > acc {
                n
            } else {
                acc
            }
        })
    }

    fn scale_into(&mut self, k: F) {
        for e in &mut self.per_edge {
            State::<F>::scale_into(e, k);
        }
    }
}

// ── QuantumGraphHeatChernoff ───────────────────────────────────────────────────

/// Quantum graph heat Chernoff (ADR-0078, math §29.3): `½∂²_x` per edge + Kirchhoff.
/// Phase 1: combined-domain `ShiftChernoff1D` on `[0,L_total]` (uniform graphs) or
/// per-edge fallback. Phase 2: mean-averaging `Q_v` = (1/d)·1·1^T at each vertex.
/// Note: `Debug` not derived — `ShiftChernoff1D<F>` stores fn-pointer fields.
#[derive(Clone)]
pub struct QuantumGraphHeatChernoff<F: SemiflowFloat = f64> {
    /// Metric graph geometry.
    pub graph: QuantumGraph<F>,
    /// Per-vertex Kirchhoff projectors.
    pub vertices: Vec<KirchhoffVertex<F>>,
    /// Per-edge heat kernels — fallback for non-uniform / non-path graphs.
    pub edge_kernels: Vec<ShiftChernoff1D<F>>,
    /// Combined-domain kernel for uniform path graphs (`n_per_edge` grid points).
    /// `Some((kernel, n_per_edge))` iff all edges have equal length and grid size.
    combined_kernel: Option<(ShiftChernoff1D<F>, usize)>,
}

impl<F: SemiflowFloat> QuantumGraphHeatChernoff<F> {
    /// Build from a `QuantumGraph`.
    ///
    /// # Errors
    /// Returns [`SemiflowError`] if vertex or edge kernel construction fails.
    pub fn new(graph: QuantumGraph<F>) -> Result<Self, SemiflowError> {
        let vertices = (0..graph.n_vertices)
            .map(|v| KirchhoffVertex::new(&graph, v))
            .collect::<Result<Vec<_>, _>>()?;
        let edge_kernels = build_edge_kernels::<F>(&graph);
        let combined_kernel = build_combined_kernel::<F>(&graph)?;
        Ok(Self {
            graph,
            vertices,
            edge_kernels,
            combined_kernel,
        })
    }
}

impl<F: SemiflowFloat> ChernoffFunction<F> for QuantumGraphHeatChernoff<F> {
    type S = QuantumGraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        _scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        // Phase 1: heat evolution — combined domain if available, else per-edge.
        if let Some((ck, n)) = &self.combined_kernel {
            apply_combined_heat(&self.graph, ck, *n, tau, src, dst)?;
        } else {
            for (e, kernel) in self.edge_kernels.iter().enumerate() {
                let result = kernel.apply_f(tau, &src.per_edge[e])?;
                State::<F>::copy_from(&mut dst.per_edge[e], &result);
            }
        }
        // Phase 2: per-vertex Kirchhoff projection (math §29.3).
        apply_kirchhoff_projection(&self.graph, &self.vertices, dst);
        Ok(())
    }

    fn order(&self) -> u32 {
        1
    }
    fn growth(&self) -> Growth<F> {
        Growth::new(F::one(), F::zero())
    }
}

// ── Private helpers ────────────────────────────────────────────────────────────

fn coeff_half<F: SemiflowFloat>(_: F) -> F {
    from_f64::<F>(0.5)
}
fn coeff_zero<F: SemiflowFloat>(_: F) -> F {
    F::zero()
}

fn build_edge_kernels<F: SemiflowFloat>(graph: &QuantumGraph<F>) -> Vec<ShiftChernoff1D<F>> {
    graph
        .edge_grids
        .iter()
        .map(|&g| {
            ShiftChernoff1D::new_generic(coeff_half::<F>, coeff_zero::<F>, coeff_zero::<F>, 0.0, g)
        })
        .collect()
}

/// Return `true` iff the graph is a simple path (chain) topology.
///
/// A path graph has every vertex with degree ≤ 2 and is acyclic.
/// Only path-topology graphs may use the combined-domain `[0, L_total]`
/// heat kernel: their edges genuinely concatenate into a single interval.
/// Star graphs and general graphs with degree-≥3 vertices do NOT — concatenating
/// arms into one interval creates spurious adjacency between arm-endpoints that
/// are physically distinct vertices, injecting fictitious heat and breaking mass
/// conservation. (Issue #15, ADR-0078 §29.5).
fn is_path_topology(graph: &QuantumGraph<impl SemiflowFloat>) -> bool {
    graph.vertex_adjacency.iter().all(|adj| adj.len() <= 2)
}

/// Combined-domain kernel iff all edges have equal length and grid size
/// AND the graph is a path topology (simple chain).
///
/// `Some((kernel, n_per_edge))` for uniform path graphs; `None` otherwise.
/// Star and general graphs fall back to per-edge kernels (always correct).
fn build_combined_kernel<F: SemiflowFloat>(
    graph: &QuantumGraph<F>,
) -> Result<Option<(ShiftChernoff1D<F>, usize)>, SemiflowError> {
    if graph.n_edges < 2 {
        return Ok(None);
    }
    // Issue #15 fix: combined-domain is only valid for path topologies.
    if !is_path_topology(graph) {
        return Ok(None);
    }
    let n0 = graph.edge_grids[0].n;
    let l0 = graph.edge_lengths[0];
    let uniform = graph
        .edge_grids
        .iter()
        .zip(graph.edge_lengths.iter())
        .all(|(g, &l)| g.n == n0 && (l - l0).abs() < from_f64::<F>(1e-12));
    if !uniform {
        return Ok(None);
    }
    let n_total = graph.n_edges * (n0 - 1) + 1;
    let grid = Grid1D::new_generic(F::zero(), total_arc_length(graph), n_total)?;
    let k =
        ShiftChernoff1D::new_generic(coeff_half::<F>, coeff_zero::<F>, coeff_zero::<F>, 0.0, grid);
    Ok(Some((k, n0)))
}

fn total_arc_length<F: SemiflowFloat>(graph: &QuantumGraph<F>) -> F {
    graph.edge_lengths.iter().fold(F::zero(), |a, &b| a + b)
}

/// Phase 1 (uniform graphs): gather per-edge values → combined kernel → scatter back.
/// Edge `e` occupies combined[e*(n-1) .. e*(n-1)+n] (interior vertices overlap).
fn apply_combined_heat<F: SemiflowFloat>(
    graph: &QuantumGraph<F>,
    kernel: &ShiftChernoff1D<F>,
    n: usize,
    tau: F,
    src: &QuantumGraphSignal<F>,
    dst: &mut QuantumGraphSignal<F>,
) -> Result<(), SemiflowError> {
    let ne = graph.n_edges;
    let n_total = ne * (n - 1) + 1;
    let mut combined = vec![F::zero(); n_total];
    for e in 0..ne {
        let base = e * (n - 1);
        combined[base..(base + n)].copy_from_slice(&src.per_edge[e].values[..n]);
    }
    // Two half-steps S(τ/2)² for second-order accuracy at fixed N.
    let ht = tau / from_f64::<F>(2.0);
    let gf1 = kernel.apply_f(
        ht,
        &GridFn1D {
            values: combined,
            grid: kernel.grid,
        },
    )?;
    let gf_out = kernel.apply_f(ht, &gf1)?;
    for e in 0..ne {
        let base = e * (n - 1);
        for i in 0..n {
            dst.per_edge[e].values[i] = gf_out.values[base + i];
        }
    }
    Ok(())
}

// Unit tests are in `quantum_graph_data.rs` (co-located with helper fns to
// avoid exceeding the 500-line suckless limit in this file).
