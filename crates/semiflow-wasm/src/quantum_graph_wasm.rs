//! WASM bindings for quantum graph heat (`full` feature, Round 11).
//!
//! | JS class          | Core type                    | Python mirror     |
//! |-------------------|------------------------------|-------------------|
//! | `QuantumGraph`    | `QuantumGraph<f64>`          | `QuantumGraph`    |
//! | `QuantumGraphHeat`| `QuantumGraphHeatChernoff<f64>` | `QuantumGraphHeat` |
//!
//! ## State layout
//!
//! Flat `Float64Array` of length `n_edges * n_per_edge` — per-edge values
//! concatenated in edge order (same as Python).
//!
//! ## Error model
//!
//! `.kind`-tagged JS `Error` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1); no `catch_unwind`.

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use semiflow::{
    ChernoffSemigroup, GridFn1D, QuantumGraph as CoreQuantumGraph, QuantumGraphHeatChernoff,
    QuantumGraphSignal,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// QuantumGraph handle
// ---------------------------------------------------------------------------

/// Metric graph for quantum graph PDE solvers.
///
/// Stores vertices, edges with arc-lengths, and per-edge grids.
/// Factories: [`QuantumGraph.path`] and [`QuantumGraph.star`].
///
/// # Errors
/// - `.kind = "OutOfDomain"` — zero edges, non-positive edge length, or
///   `n_grid < 4`.
#[wasm_bindgen(js_name = "QuantumGraph")]
pub struct QuantumGraphWasm {
    inner: CoreQuantumGraph<f64>,
}

#[wasm_bindgen(js_class = "QuantumGraph")]
impl QuantumGraphWasm {
    /// Path graph `P_{n_edges+1}`: `n_edges` equal-length edges.
    ///
    /// ## Parameters
    /// - `n_edges`     — number of edges; must be ≥ 1.
    /// - `edge_length` — length of each edge; must be finite and > 0.
    /// - `n_grid`      — grid nodes per edge; must be ≥ 4.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` if any precondition is violated.
    #[wasm_bindgen(constructor)]
    pub fn new_path(
        n_edges: u32,
        edge_length: f64,
        n_grid: u32,
    ) -> Result<QuantumGraphWasm, JsValue> {
        let g = CoreQuantumGraph::<f64>::path(n_edges as usize, edge_length, n_grid as usize)
            .map_err(|e| err_to_js(&e))?;
        Ok(Self { inner: g })
    }

    /// Star graph: central vertex + `n_arms` leaf vertices.
    ///
    /// ## Parameters
    /// - `n_arms`      — number of arms; must be ≥ 1.
    /// - `edge_length` — arm length; must be finite and > 0.
    /// - `n_grid`      — grid nodes per edge; must be ≥ 4.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` if any precondition is violated.
    #[wasm_bindgen(js_name = "star")]
    pub fn new_star(
        n_arms: u32,
        edge_length: f64,
        n_grid: u32,
    ) -> Result<QuantumGraphWasm, JsValue> {
        let g = CoreQuantumGraph::<f64>::star(n_arms as usize, edge_length, n_grid as usize)
            .map_err(|e| err_to_js(&e))?;
        Ok(Self { inner: g })
    }

    /// Number of vertices in the graph.
    #[must_use]
    pub fn n_vertices(&self) -> u32 {
        self.inner.n_vertices as u32
    }

    /// Number of edges in the graph.
    #[must_use]
    pub fn n_edges(&self) -> u32 {
        self.inner.n_edges as u32
    }

    /// Number of grid nodes per edge (uniform grids only).
    #[must_use]
    pub fn n_per_edge(&self) -> u32 {
        self.inner.edge_grids[0].n as u32
    }
}

// ---------------------------------------------------------------------------
// QuantumGraphHeat
// ---------------------------------------------------------------------------

/// Quantum graph heat Chernoff: `∂_t u = ½∂²_x u` per edge, Kirchhoff BCs.
///
/// Algorithm: Phase 1 = edgewise `ShiftChernoff1D` (combined-domain for uniform
/// path graphs); Phase 2 = per-vertex mean-averaging `Q_v = (1/d) 1 1^T`.
///
/// State layout: flat `Float64Array` of length `n_edges * n_per_edge` —
/// per-edge values concatenated in edge order.
///
/// Mirrors Python `QuantumGraphHeat`.
///
/// # Errors
/// - `.kind = "OutOfDomain"` — graph has zero edges.
/// - `.kind = "GridMismatch"` — `f0.length != n_edges * n_per_edge`.
/// - `.kind = "NanInf"` — `f0` contains NaN or Inf.
/// - `.kind = "OutOfDomain"` — `t < 0`, non-finite, or `n_steps == 0`.
#[wasm_bindgen(js_name = "QuantumGraphHeat")]
pub struct QuantumGraphHeatWasm {
    kernel: QuantumGraphHeatChernoff<f64>,
    graph: CoreQuantumGraph<f64>,
    n_edges: usize,
    n_per_edge: usize,
}

#[wasm_bindgen(js_class = "QuantumGraphHeat")]
impl QuantumGraphHeatWasm {
    /// Build a quantum graph heat solver from a `QuantumGraph`.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` if kernel construction fails.
    #[wasm_bindgen(constructor)]
    pub fn new(qg: &QuantumGraphWasm) -> Result<QuantumGraphHeatWasm, JsValue> {
        let graph = qg.inner.clone();
        let n_edges = graph.n_edges;
        let n_per_edge = graph.edge_grids[0].n;
        let kernel = QuantumGraphHeatChernoff::new(graph.clone()).map_err(|e| err_to_js(&e))?;
        Ok(Self {
            kernel,
            graph,
            n_edges,
            n_per_edge,
        })
    }

    /// Evolve `f0` by time `t` using `n_steps` Chernoff steps.
    ///
    /// Returns a freshly allocated `Float64Array` of length `n_edges * n_per_edge`.
    /// Stateless: each call uses the explicit `f0` initial condition.
    ///
    /// # Errors
    /// See struct-level error table.
    pub fn evolve(&self, t: f64, n_steps: u32, f0: &[f64]) -> Result<Vec<f64>, JsValue> {
        let total = self.n_edges * self.n_per_edge;
        validate_qg_inputs(f0, total, t, n_steps)?;
        let sg = ChernoffSemigroup::new(self.kernel.clone(), n_steps as usize)
            .map_err(|e| err_to_js(&e))?;
        let src = signal_from_flat(&self.graph, f0, self.n_per_edge);
        let out = sg.evolve(t, &src).map_err(|e| err_to_js(&e))?;
        Ok(gather_flat(&out, self.n_per_edge))
    }

    /// Total number of grid nodes (`n_edges * n_per_edge`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> u32 {
        (self.n_edges * self.n_per_edge) as u32
    }

    /// Number of edges.
    #[must_use]
    pub fn n_edges(&self) -> u32 {
        self.n_edges as u32
    }

    /// Grid nodes per edge.
    #[must_use]
    pub fn n_per_edge(&self) -> u32 {
        self.n_per_edge as u32
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn signal_from_flat(
    graph: &CoreQuantumGraph<f64>,
    flat: &[f64],
    n_per_edge: usize,
) -> QuantumGraphSignal<f64> {
    let per_edge = graph
        .edge_grids
        .iter()
        .enumerate()
        .map(|(e, &g)| {
            let base = e * n_per_edge;
            GridFn1D {
                values: flat[base..base + n_per_edge].to_vec(),
                grid: g,
            }
        })
        .collect();
    QuantumGraphSignal { per_edge }
}

fn gather_flat(sig: &QuantumGraphSignal<f64>, n_per_edge: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(sig.per_edge.len() * n_per_edge);
    for e in &sig.per_edge {
        out.extend_from_slice(&e.values);
    }
    out
}

fn validate_qg_inputs(f0: &[f64], total: usize, t: f64, n_steps: u32) -> Result<(), JsValue> {
    if f0.len() != total {
        return Err(make_js_error(
            "GridMismatch",
            "f0.length must equal n_edges * n_per_edge",
        ));
    }
    for &v in f0 {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", "f0 contains NaN or Inf"));
        }
    }
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}
