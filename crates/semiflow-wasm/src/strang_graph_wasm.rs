//! WASM binding for `StrangSplitGraph` (`full` feature, Round 11).
//!
//! | JS class     | Core type                                | Python mirror |
//! |--------------|------------------------------------------|---------------|
//! | `StrangGraph`| `StrangSplitGraph<GraphHeatChernoff<f64>,…>` | `StrangGraph` |
//!
//! Palindromic Strang split on graph signals: `S(τ) f = A(τ/2)∘B(τ)∘A(τ/2)·f`.
//! Two safe constructors for guaranteed-commuting decompositions via edge-parity
//! 2-coloring: `fromPath` (path graph) and `fromCycle` (even-length cycle).
//!
//! State: flat `Float64Array` of length `n_nodes`.
//!
//! ## Error model
//!
//! `.kind`-tagged JS `Error` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1); no `catch_unwind`.

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use semiflow::{
    ChernoffFunction, ChernoffSemigroup, Graph, GraphHeatChernoff, GraphSignal, StrangSplitGraph,
};
use wasm_bindgen::prelude::*;

use crate::{
    error::{err_to_js, make_js_error},
    graph_wasm::GraphPath,
};

// ---------------------------------------------------------------------------
// StrangGraph
// ---------------------------------------------------------------------------

/// Palindromic Strang split for graph heat Chernoff kernels (math §12.8).
///
/// `S(τ) f = A(τ/2) ∘ B(τ) ∘ A(τ/2) · f` on `GraphSignal<f64>`.
///
/// Uses edge-parity 2-coloring to guarantee `[L_A, L_B] = 0`, yielding global
/// order-2 convergence. Mirrors Python `StrangGraph`.
///
/// Factories:
/// - [`StrangGraph.fromPath`] — path graph `P_n` (n ≥ 2).
/// - [`StrangGraph.fromCycle`] — even-length cycle `C_n` (n ≥ 4, n even).
///
/// # Errors
/// - `.kind = "OutOfDomain"` — graph too small for the chosen factory.
/// - `.kind = "GridMismatch"` — `f0.length != n_nodes` in `evolve`.
/// - `.kind = "NanInf"` — `f0` contains NaN or Inf.
/// - `.kind = "OutOfDomain"` — `t_final < 0`, non-finite, or `n_steps == 0`.
#[wasm_bindgen(js_name = "StrangGraph")]
pub struct StrangGraphWasm {
    strang: StrangSplitGraph<GraphHeatChernoff<f64>, GraphHeatChernoff<f64>, f64>,
    graph: Arc<Graph<f64>>,
    n_nodes: usize,
}

#[wasm_bindgen(js_class = "StrangGraph")]
impl StrangGraphWasm {
    /// Build from a path graph `P_n` via even/odd edge-parity 2-coloring.
    ///
    /// Requires `n_nodes >= 2`.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` if `n_nodes < 2`.
    #[wasm_bindgen(js_name = "fromPath")]
    pub fn from_path(graph: &GraphPath) -> Result<StrangGraphWasm, JsValue> {
        let n = graph.n_nodes() as usize;
        let g = Arc::new(Graph::<f64>::path(n));
        let strang = StrangSplitGraph::new_bipartite_path(&g).map_err(|e| err_to_js(&e))?;
        Ok(Self {
            strang,
            graph: g,
            n_nodes: n,
        })
    }

    /// Build from an even-length cycle graph `C_n` via edge-parity coloring.
    ///
    /// Requires `n_nodes >= 4` and `n_nodes % 2 == 0`.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` if `n_nodes < 4` or odd.
    #[wasm_bindgen(js_name = "fromCycle")]
    pub fn from_cycle(graph: &GraphPath) -> Result<StrangGraphWasm, JsValue> {
        let n = graph.n_nodes() as usize;
        let g = Arc::new(Graph::<f64>::path(n));
        let strang = StrangSplitGraph::new_bipartite_cycle(&g).map_err(|e| err_to_js(&e))?;
        Ok(Self {
            strang,
            graph: g,
            n_nodes: n,
        })
    }

    /// Evolve signal `f0` to time `t_final` with `n_steps` Strang steps.
    ///
    /// Returns a freshly allocated `Float64Array` of length `n_nodes`.
    ///
    /// # Errors
    /// See struct-level error table.
    pub fn evolve(&self, t_final: f64, n_steps: u32, f0: &[f64]) -> Result<Vec<f64>, JsValue> {
        validate_strang_inputs(f0, self.n_nodes, t_final, n_steps)?;
        let sg = ChernoffSemigroup::new(self.strang.clone(), n_steps as usize)
            .map_err(|e| err_to_js(&e))?;
        let signal = GraphSignal::from_fn(Arc::clone(&self.graph), |i| f0[i as usize]);
        let out = sg.evolve(t_final, &signal).map_err(|e| err_to_js(&e))?;
        Ok(out.values().to_vec())
    }

    /// Approximation order (2 when commutativity holds).
    #[must_use]
    pub fn order(&self) -> u32 {
        self.strang.order()
    }

    /// Number of nodes in the graph.
    #[must_use]
    pub fn n_nodes(&self) -> u32 {
        self.n_nodes as u32
    }
}

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

fn validate_strang_inputs(
    f0: &[f64],
    n_nodes: usize,
    t_final: f64,
    n_steps: u32,
) -> Result<(), JsValue> {
    if f0.len() != n_nodes {
        return Err(make_js_error(
            "GridMismatch",
            "f0.length must equal n_nodes",
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
    if !t_final.is_finite() || t_final < 0.0 {
        return Err(make_js_error(
            "OutOfDomain",
            "t_final must be finite and >= 0",
        ));
    }
    Ok(())
}
