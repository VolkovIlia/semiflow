//! Extra graph heat WASM bindings: `GraphHeat4th` and
//! `VarCoefGraphHeat` — static (time-independent) kernels.
//!
//! Mirrors `semiflow-py` `GraphHeat4th` and `VarCoefGraphHeat` from
//! `crates/semiflow-py/src/graph_extra_heat.rs`.
//!
//! Behind `#[cfg(feature = "full")]`.  Both classes accept a `GraphPath`
//! handle (same as `GraphHeat` and `GraphHeat6`) plus extra parameters.
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` as all other graph WASM classes.
//! `panic = "abort"` (ADR-0028 Amendment 1).  No `catch_unwind`.
//!
//! ## f64 only
//!
//! Graph PDE WASM bindings are f64-only (ADR-0059 §"Out of scope").
//!
//! ## Example (JS)
//!
//! ```js
//! import init, { GraphPath, GraphHeat4th, VarCoefGraphHeat } from "@semiflow/wasm";
//! await init();
//! const g  = new GraphPath(32);
//! const h4 = new GraphHeat4th(g, 4.0);
//! const f0 = new Float64Array(32).fill(0.0); f0[0] = 1.0;
//! const r  = h4.evolve(0.25, 20, f0);
//!
//! const a  = new Float64Array(32).fill(2.0);
//! const vc = new VarCoefGraphHeat(g, a, 8.0);
//! const r2 = vc.evolve(0.1, 10, f0);
//! ```

#![allow(unsafe_code)]

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};
use crate::graph_wasm::GraphPath;
use semiflow::{
    ChernoffSemigroup, Graph, GraphSignal, Laplacian,
    GraphHeat4thChernoff as CoreGraphHeat4th,
    VarCoefGraphHeatChernoff as CoreVarCoefGraphHeat,
};

// ---------------------------------------------------------------------------
// GraphHeat4th
// ---------------------------------------------------------------------------

/// Order-4 graph heat Chernoff: `∂ₜu = −L_G u` via Padé[0,4] Taylor truncation.
///
/// `S₄(τ) f = Σ_{k=0}^{4} (−τ L_G)^k / k! · f`
///
/// Mirrors Python `GraphHeat4th`.  For time-varying `L_G(t)` use
/// `MagnusGraphHeat` (K=4) or `MagnusGraphHeat6` (K=6).
///
/// # Errors
/// - `.kind = "OutOfDomain"` — `rho_bar <= 0` or non-finite.
/// - `.kind = "GridMismatch"` — `f0.length != n_nodes` in `evolve`.
/// - `.kind = "NanInf"` — `f0` contains NaN or Inf.
/// - `.kind = "OutOfDomain"` — `t_final < 0`, non-finite, or `n_steps == 0`.
/// - `.kind = "ConvergenceFailed"` — semigroup convergence failure.
#[wasm_bindgen(js_name = "GraphHeat4th")]
pub struct GraphHeat4thWasm {
    // Stored as Arc so the Chernoff can be cheaply rebuilt each evolve call.
    lap: Arc<Laplacian<f64>>,
    graph: Arc<Graph<f64>>,
    n_nodes: usize,
}

#[wasm_bindgen(js_class = "GraphHeat4th")]
impl GraphHeat4thWasm {
    /// Build an order-4 graph heat Chernoff from a `GraphPath`.
    ///
    /// ## Parameters
    /// - `graph`   — borrow of a `GraphPath` (must have ≥ 1 node).
    /// - `rho_bar` — Gershgorin spectral-radius upper bound `ρ̄ ≥ ρ(L_G)`;
    ///   must be finite and > 0.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `rho_bar <= 0` or non-finite.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &GraphPath, rho_bar: f64) -> Result<GraphHeat4thWasm, JsValue> {
        if !rho_bar.is_finite() || rho_bar <= 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "rho_bar must be finite and > 0",
            ));
        }
        let n_nodes = graph.n_nodes() as usize;
        let g = Arc::new(Graph::<f64>::path(n_nodes));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        Ok(Self { lap, graph: g, n_nodes })
    }

    /// Evolve `f0` by `t_final` seconds with `n_steps` order-4 Chernoff steps.
    ///
    /// Returns a freshly allocated `Float64Array` (length `n_nodes`).
    /// Stateless: each call uses the explicit `f0` initial condition.
    ///
    /// # Errors
    /// See struct-level error table.
    pub fn evolve(&self, t_final: f64, n_steps: u32, f0: &[f64]) -> Result<Vec<f64>, JsValue> {
        validate_evolve_inputs(f0, self.n_nodes, t_final, n_steps)?;
        // Rebuild Chernoff from stored Arc<Laplacian> (cheap — Arc::clone only).
        let chernoff = CoreGraphHeat4th::new(Arc::clone(&self.lap));
        let signal = GraphSignal::from_fn(Arc::clone(&self.graph), |i| f0[i as usize]);
        let sg = ChernoffSemigroup::new(chernoff, n_steps as usize)
            .map_err(|e| err_to_js(&e))?;
        let result = sg.evolve(t_final, &signal).map_err(|e| err_to_js(&e))?;
        Ok(result.values().to_vec())
    }

    /// Number of nodes the kernel acts on.
    #[must_use]
    #[wasm_bindgen(js_name = "n_nodes")]
    pub fn n_nodes(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        let n = self.n_nodes as u32;
        n
    }
}

// ---------------------------------------------------------------------------
// VarCoefGraphHeat
// ---------------------------------------------------------------------------

/// Variable-coefficient graph heat: `∂ₜu = −L_a u`,
/// `L_a = A^{1/2} L_G A^{1/2}`, `A = diag(a)`.
///
/// Order-2 Chernoff approximation.  Mirrors Python `VarCoefGraphHeat`.
/// For time-varying `a(t)` use `VarCoefMagnusGraph`.
///
/// # Errors
/// - `.kind = "OutOfDomain"` — `rho_bar <= 0`, `a` non-positive, or non-finite.
/// - `.kind = "GridMismatch"` — `a.length != n_nodes` or `f0.length != n_nodes`.
/// - `.kind = "NanInf"` — `f0` contains NaN or Inf.
/// - `.kind = "OutOfDomain"` — `t_final < 0`, non-finite, or `n_steps == 0`.
/// - `.kind = "ConvergenceFailed"` — semigroup convergence failure.
#[wasm_bindgen(js_name = "VarCoefGraphHeat")]
pub struct VarCoefGraphHeatWasm {
    graph: Arc<Graph<f64>>,
    a: Vec<f64>,
    rho_bar: f64,
    n_nodes: usize,
}

#[wasm_bindgen(js_class = "VarCoefGraphHeat")]
impl VarCoefGraphHeatWasm {
    /// Build a variable-coefficient order-2 graph heat state.
    ///
    /// ## Parameters
    /// - `graph`   — borrow of a `GraphPath` (must have ≥ 1 node).
    /// - `a`       — conductivity vector; `Float64Array` of length `n_nodes`.
    ///   All elements must be finite and strictly positive.
    /// - `rho_bar` — Gershgorin bound for `L_a`; must be finite and > 0.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `rho_bar <= 0`, any `a[i] <= 0`, non-finite.
    /// - `.kind = "GridMismatch"` — `a.length != n_nodes`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &GraphPath, a: &[f64], rho_bar: f64) -> Result<VarCoefGraphHeatWasm, JsValue> {
        if !rho_bar.is_finite() || rho_bar <= 0.0 {
            return Err(make_js_error("OutOfDomain", "rho_bar must be finite and > 0"));
        }
        let n_nodes = graph.n_nodes() as usize;
        if a.len() != n_nodes {
            return Err(make_js_error("GridMismatch", "a.length must equal n_nodes"));
        }
        for &v in a {
            if !v.is_finite() || v <= 0.0 {
                return Err(make_js_error("OutOfDomain", "a must contain finite positive values"));
            }
        }
        let g = Arc::new(Graph::<f64>::path(n_nodes));
        // Validate via core constructor.
        CoreVarCoefGraphHeat::new(Arc::clone(&g), a.to_vec(), rho_bar)
            .map_err(|e| err_to_js(&e))?;
        Ok(Self {
            graph: g,
            a: a.to_vec(),
            rho_bar,
            n_nodes,
        })
    }

    /// Evolve `f0` by `t_final` seconds with `n_steps` order-2 Chernoff steps.
    ///
    /// Returns a freshly allocated `Float64Array` (length `n_nodes`).
    ///
    /// # Errors
    /// See struct-level error table.
    pub fn evolve(&self, t_final: f64, n_steps: u32, f0: &[f64]) -> Result<Vec<f64>, JsValue> {
        validate_evolve_inputs(f0, self.n_nodes, t_final, n_steps)?;
        let graph = Arc::clone(&self.graph);
        let g2 = Arc::clone(&self.graph);
        let chernoff = CoreVarCoefGraphHeat::new(graph, self.a.clone(), self.rho_bar)
            .map_err(|e| err_to_js(&e))?;
        let sg = ChernoffSemigroup::new(chernoff, n_steps as usize)
            .map_err(|e| err_to_js(&e))?;
        #[allow(clippy::cast_possible_truncation)]
        let signal = GraphSignal::from_fn(g2, |i| f0[i as usize]);
        let result = sg.evolve(t_final, &signal).map_err(|e| err_to_js(&e))?;
        Ok(result.values().to_vec())
    }

    /// Number of nodes the kernel acts on.
    #[must_use]
    #[wasm_bindgen(js_name = "n_nodes")]
    pub fn n_nodes(&self) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        let n = self.n_nodes as u32;
        n
    }
}

// ---------------------------------------------------------------------------
// Shared validation helper
// ---------------------------------------------------------------------------

/// Validate common evolve parameters: f0 length, finiteness, `n_steps`, `t_final`.
fn validate_evolve_inputs(
    f0: &[f64],
    n_nodes: usize,
    t_final: f64,
    n_steps: u32,
) -> Result<(), JsValue> {
    if f0.len() != n_nodes {
        return Err(make_js_error("GridMismatch", "f0.length must equal n_nodes"));
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
        return Err(make_js_error("OutOfDomain", "t_final must be finite and >= 0"));
    }
    Ok(())
}
