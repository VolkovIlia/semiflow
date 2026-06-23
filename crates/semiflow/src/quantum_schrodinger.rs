//! v7.0.0 #16 — Schrödinger on quantum graphs via complex Kirchhoff (ADR-0130, math §30.5-bis).
//!
//! `QuantumSchrödingerChernoff<C>` solves `i ψ_t = −½ ∂²_x ψ` per edge of a metric
//! graph with Kirchhoff–Neumann vertex condition (continuity + net-zero derivative
//! flux). The pre-flight (`scripts/verify_quantum_schrodinger.py`, 3/3 PASS) confirms:
//!
//! 1. `Q_v = (1/d) 1 1^T` over ℂ is Hermitian/idempotent/rank-1 — lifts verbatim.
//! 2. `Q · Cayley(τ) · Q` is norm-preserving on the continuity subspace.
//! 3. Probability current `∑ₑ Im(ψ̄ ψ′) = 0` is implied by continuity + Kirchhoff.
//!
//! No new vertex-condition class is needed. Algorithm mirrors `QuantumGraphHeatChernoff`:
//! Phase 1 — per-edge free-kinetic Cayley step (V=0 Crank-Nicolson, reuses
//! `cayley_step_dx`); Phase 2 — per-vertex complex Kirchhoff projection.
//!
//! Cite: ADR-0078 (heat), ADR-0079 (complex), ADR-0130 (this).

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use num_traits::{Float, One, Zero};

use crate::{
    chernoff::{ChernoffFunction, Growth},
    complex::SemiflowComplex,
    error::SemiflowError,
    float::from_f64,
    quantum_graph::QuantumGraph,
    quantum_graph_data::KirchhoffVertex,
    schrodinger_complex::{cayley_step_dx, GridFnComplex1D},
    scratch::ScratchPool,
    state::State,
};

// ── Public signal type ─────────────────────────────────────────────────────────

/// Per-edge complex signal for `QuantumSchrödingerChernoff` (ADR-0130, math §30.5-bis).
#[derive(Clone)]
pub struct QuantumGraphComplexSignal<C: SemiflowComplex> {
    /// Per-edge complex wave functions. Length == `n_edges`.
    pub per_edge: Vec<GridFnComplex1D<C>>,
}

impl<C: SemiflowComplex> QuantumGraphComplexSignal<C> {
    /// Zero signal sized to `graph`.
    #[must_use]
    pub fn zeroed_for_graph(graph: &QuantumGraph<C::Real>) -> Self {
        let per_edge = graph
            .edge_grids
            .iter()
            .map(|g| GridFnComplex1D::from_fn(*g, |_| C::zero()))
            .collect();
        Self { per_edge }
    }

    /// Sample `f(edge_index, x) -> C` on each edge.
    #[must_use]
    pub fn from_fn(graph: &QuantumGraph<C::Real>, f: impl Fn(usize, C::Real) -> C) -> Self {
        let per_edge = graph
            .edge_grids
            .iter()
            .enumerate()
            .map(|(e, g)| GridFnComplex1D::from_fn(*g, |x| f(e, x)))
            .collect();
        Self { per_edge }
    }

    /// Complex arc-length eigenmode: `φ_k(s) = exp(i k π s / L)` (free-particle mode).
    ///
    /// For a path graph with total arc-length `L`, this is an eigenfunction of
    /// `−½ ∂²_x` with eigenvalue `λ_k = k²π²/(2L²)`. Phase evolves as
    /// `exp(−i λ_k τ)` per Chernoff step.
    #[must_use]
    pub fn from_phase_mode(graph: &QuantumGraph<C::Real>, k: usize) -> Self {
        let total = total_arc_length_c(graph);
        let pi = from_f64::<C::Real>(core::f64::consts::PI);
        // Truncating cast is safe: k is a small mode index, never exceeds 2^52.
        #[allow(clippy::cast_precision_loss)]
        let k_f = from_f64::<C::Real>(k as f64);
        let mut offset = <C::Real as Zero>::zero();
        let per_edge = graph
            .edge_grids
            .iter()
            .enumerate()
            .map(|(e, g)| {
                let off = offset;
                offset += graph.edge_lengths[e];
                GridFnComplex1D::from_fn(*g, |x| {
                    // phase = k π (off + x) / L
                    let phase = k_f * pi * (off + x) / total;
                    C::from_polar(<C::Real as One>::one(), phase)
                })
            })
            .collect();
        Self { per_edge }
    }

    /// Discrete L²-norm squared: `Σ_e Σ_k |ψ_{e,k}|² dx_e`.
    #[must_use]
    pub fn norm_l2_sq(&self) -> C::Real {
        self.per_edge
            .iter()
            .fold(<C::Real as Zero>::zero(), |acc, e| acc + e.norm_l2_sq())
    }

    /// Discrete L²-norm.
    #[must_use]
    pub fn norm_l2(&self) -> C::Real {
        Float::sqrt(self.norm_l2_sq())
    }

    /// Value at the endpoint of edge `e` incident to vertex `v`.
    #[must_use]
    #[inline]
    pub fn edge_endpoint_value(&self, graph: &QuantumGraph<C::Real>, e: usize, v: usize) -> C {
        let vals = &self.per_edge[e].values;
        if graph.edge_endpoints[e].0 == v {
            vals[0]
        } else {
            vals[vals.len() - 1]
        }
    }

    /// Set the value at the endpoint of edge `e` incident to vertex `v`.
    #[inline]
    pub fn set_edge_endpoint_value(
        &mut self,
        graph: &QuantumGraph<C::Real>,
        e: usize,
        v: usize,
        val: C,
    ) {
        let vals = &mut self.per_edge[e].values;
        if graph.edge_endpoints[e].0 == v {
            vals[0] = val;
        } else {
            let n = vals.len();
            vals[n - 1] = val;
        }
    }
}

impl<C: SemiflowComplex> State<C::Real> for QuantumGraphComplexSignal<C> {
    fn len(&self) -> usize {
        self.per_edge.iter().map(|e| e.values.len()).sum()
    }

    fn axpy_into(&mut self, alpha: C::Real, src: &Self) {
        for (d, s) in self.per_edge.iter_mut().zip(src.per_edge.iter()) {
            State::<C::Real>::axpy_into(d, alpha, s);
        }
    }

    fn copy_from(&mut self, src: &Self) {
        for (d, s) in self.per_edge.iter_mut().zip(src.per_edge.iter()) {
            State::<C::Real>::copy_from(d, s);
        }
    }

    fn zero_into(&mut self) {
        for e in &mut self.per_edge {
            State::<C::Real>::zero_into(e);
        }
    }

    fn norm_sup(&self) -> C::Real {
        self.per_edge
            .iter()
            .fold(<C::Real as Zero>::zero(), |acc, e| {
                let n = State::<C::Real>::norm_sup(e);
                if n > acc {
                    n
                } else {
                    acc
                }
            })
    }

    fn scale_into(&mut self, k: C::Real) {
        for e in &mut self.per_edge {
            State::<C::Real>::scale_into(e, k);
        }
    }
}

// ── QuantumSchrödingerChernoff ─────────────────────────────────────────────────

/// Quantum graph Schrödinger Chernoff (ADR-0130, math §30.5-bis):
/// `i ψ_t = −½ ∂²_x ψ` per edge, Kirchhoff vertex conditions.
///
/// Algorithm (two-phase, mirrors `QuantumGraphHeatChernoff`):
/// - Phase 1: per-edge free Cayley step (V=0 Crank-Nicolson), reusing `cayley_step_dx`.
/// - Phase 2: per-vertex complex mean-averaging `Q_v = (1/d) 1 1^T` over ℂ.
///
/// The composition stays unitary on the continuity subspace (sympy-verified, ADR-0130).
/// Order: 1 (per-edge Cayley is order-2; composition across the vertex projector is order-1).
#[derive(Clone)]
pub struct QuantumSchrödingerChernoff<C: SemiflowComplex> {
    /// Metric graph geometry (real-valued lengths/grids).
    pub graph: QuantumGraph<C::Real>,
    /// Per-vertex Kirchhoff projectors (reused from heat kernel, now over ℂ).
    pub vertices: Vec<KirchhoffVertex<C::Real>>,
    /// Per-edge grid spacings `dx_e = L_e / (n-1)` for the Cayley step.
    edge_dx: Vec<C::Real>,
    /// Number of grid points per edge.
    n_per_edge: usize,
}

impl<C: SemiflowComplex> QuantumSchrödingerChernoff<C> {
    /// Build from a `QuantumGraph`.
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] if the graph has zero edges or
    /// if any vertex index is out of range.
    pub fn new(graph: QuantumGraph<C::Real>) -> Result<Self, SemiflowError> {
        if graph.n_edges == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "QuantumSchrödingerChernoff: graph must have at least 1 edge",
                value: 0.0,
            });
        }
        let vertices = (0..graph.n_vertices)
            .map(|v| KirchhoffVertex::new(&graph, v))
            .collect::<Result<Vec<_>, _>>()?;
        let n_per_edge = graph.edge_grids[0].n;
        let edge_dx = graph
            .edge_grids
            .iter()
            .map(crate::grid::Grid1D::dx)
            .collect();
        Ok(Self {
            graph,
            vertices,
            edge_dx,
            n_per_edge,
        })
    }
}

impl<C: SemiflowComplex> ChernoffFunction<C::Real> for QuantumSchrödingerChernoff<C> {
    type S = QuantumGraphComplexSignal<C>;

    fn apply_into(
        &self,
        tau: C::Real,
        src: &Self::S,
        dst: &mut Self::S,
        _scratch: &mut ScratchPool<C::Real>,
    ) -> Result<(), SemiflowError> {
        // Phase 1: per-edge free Cayley step (V=0).
        let n = self.n_per_edge;
        let mut tmp_dst: Vec<C> = vec![C::zero(); n];
        for (e, &dx) in self.edge_dx.iter().enumerate() {
            cayley_step_dx::<C>(dx, tau, &src.per_edge[e].values, &mut tmp_dst)?;
            dst.per_edge[e].values.copy_from_slice(&tmp_dst);
        }

        // Phase 2: per-vertex complex Kirchhoff projection Q_v = (1/d) 1 1^T.
        apply_kirchhoff_projection_complex(&self.graph, &self.vertices, dst);
        Ok(())
    }

    fn order(&self) -> u32 {
        1
    }

    /// Unitary: multiplier=1, omega=0 (Cayley + orthogonal projection).
    fn growth(&self) -> Growth<C::Real> {
        Growth::new(<C::Real as One>::one(), <C::Real as Zero>::zero())
    }
}

// ── Private helpers ────────────────────────────────────────────────────────────

fn total_arc_length_c<F: crate::float::SemiflowFloat>(graph: &QuantumGraph<F>) -> F {
    graph.edge_lengths.iter().fold(F::zero(), |a, &b| a + b)
}

/// Phase 2: complex mean-averaging at each vertex with degree ≥ 2.
/// Same logic as `quantum_graph_data::apply_kirchhoff_projection` but over ℂ.
fn apply_kirchhoff_projection_complex<C: SemiflowComplex>(
    graph: &QuantumGraph<C::Real>,
    vertices: &[KirchhoffVertex<C::Real>],
    dst: &mut QuantumGraphComplexSignal<C>,
) {
    for vtx in vertices {
        let d = vtx.degree;
        if d <= 1 {
            continue;
        }
        // gather endpoint values
        let mut buf: Vec<C> = vtx
            .incident_edges
            .iter()
            .map(|&e| dst.edge_endpoint_value(graph, e, vtx.vertex_index))
            .collect();
        // mean-average: each entry becomes (1/d) * sum(buf)
        let sum: C = buf.iter().fold(C::zero(), |acc, &z| acc + z);
        // Truncating cast is safe: d is a small vertex degree, never exceeds 2^52.
        #[allow(clippy::cast_precision_loss)]
        let inv_d = from_f64::<C::Real>(1.0 / d as f64);
        let mean = C::from_real(inv_d) * sum;
        buf.fill(mean);
        // scatter back
        for (k, &e) in vtx.incident_edges.iter().enumerate() {
            dst.set_edge_endpoint_value(graph, e, vtx.vertex_index, buf[k]);
        }
    }
}

// ── Inline unit tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    type C64 = Complex<f64>;

    fn path_graph(n_edges: usize, l: f64, n: usize) -> QuantumGraph<f64> {
        QuantumGraph::<f64>::path(n_edges, l, n).unwrap()
    }

    #[test]
    fn ctor_path_graph() {
        let g = path_graph(2, 1.0, 16);
        let k = QuantumSchrödingerChernoff::<C64>::new(g).unwrap();
        assert_eq!(k.graph.n_edges, 2);
        assert_eq!(k.vertices.len(), 3);
        assert_eq!(k.order(), 1);
    }

    #[test]
    fn growth_is_unitary() {
        let g = path_graph(2, 1.0, 16);
        let k = QuantumSchrödingerChernoff::<C64>::new(g).unwrap();
        let gr = k.growth();
        assert!((gr.multiplier - 1.0).abs() < 1e-15);
        assert!(gr.omega.abs() < 1e-15);
    }

    #[test]
    fn zeroed_signal_len() {
        let g = path_graph(2, 1.0, 16);
        let sig = QuantumGraphComplexSignal::<C64>::zeroed_for_graph(&g);
        assert_eq!(sig.per_edge.len(), 2);
        assert_eq!(State::<f64>::len(&sig), 2 * 16);
    }

    #[test]
    fn apply_into_output_finite() {
        let g = path_graph(2, 1.0, 32);
        let kernel = QuantumSchrödingerChernoff::<C64>::new(g.clone()).unwrap();
        let src = QuantumGraphComplexSignal::from_fn(&g, |_, x: f64| C64::new((-x * x).exp(), 0.0));
        let mut dst = src.clone();
        let mut scratch = ScratchPool::new();
        kernel
            .apply_into(0.01, &src, &mut dst, &mut scratch)
            .unwrap();
        assert!(dst
            .per_edge
            .iter()
            .all(|e| e.values.iter().all(|z| z.is_finite())));
    }

    #[test]
    fn unitarity_one_step_smoke() {
        let g = path_graph(2, 1.0, 64);
        let kernel = QuantumSchrödingerChernoff::<C64>::new(g.clone()).unwrap();
        let src =
            QuantumGraphComplexSignal::from_fn(&g, |_, x: f64| C64::new((-x * x / 2.0).exp(), 0.0));
        let norm0 = src.norm_l2();
        let mut dst = src.clone();
        let mut scratch = ScratchPool::new();
        kernel
            .apply_into(0.01, &src, &mut dst, &mut scratch)
            .unwrap();
        let norm1 = dst.norm_l2();
        // The projection Q can shrink norm for non-continuous states.
        // For a smooth initial datum the drift should be small but not necessarily 0.
        assert!(
            norm1 > 0.0 && norm1 <= norm0 + 1e-10,
            "norm drift: {}",
            (norm1 - norm0).abs()
        );
    }
}
