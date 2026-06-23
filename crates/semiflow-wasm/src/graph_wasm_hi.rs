//! High-order graph WASM bindings for v2.4 (ADR-0062, ADR-0064).
//!
//! Exposes one JS class for the new order-6 spatial graph heat kernel:
//!
//!   `new GraphHeat6(graph, rho_bar)` — order-6 heat Chernoff on a graph.
//!   `heat.evolve(t_final, n_steps, f0)` — advance `t_final` seconds,
//!       return a `Float64Array` of the final state.
//!
//! **Scope (v2.4, ADR-0064)**: only `GraphHeat6` is shipped to WASM in v2.4.
//! Time-dependent variants (`MagnusGraphHeat`, `MagnusGraphHeat6`,
//! `VarCoefMagnusGraph`) require JS-side callbacks for `lap_at_t` /
//! `a_at_t`; the unsafe Send+Sync wrappers around `js_sys::Function` plus
//! JS-call-per-quadrature-point overhead are deferred to v2.5 per
//! ADR-0064 §"Out of scope (v2.4)".
//!
//! Mirrors the design of `graph_wasm::GraphHeat` (order-2 ζ-A Taylor) — same
//! error model, same panic policy (`panic = abort`, no `catch_unwind`),
//! f64-only.

#![allow(unsafe_code)]

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};
use crate::graph_wasm::GraphPath;
use semiflow::{ChernoffSemigroup, Graph, GraphHeat6thChernoff, GraphSignal, Laplacian};

// ---------------------------------------------------------------------------
// GraphHeat6
// ---------------------------------------------------------------------------

/// Order-6 graph heat Chernoff state for `∂ₜu = −L_G u` (ADR-0062).
///
/// `S_6(τ) f = Σ_{k=0}^{6} (−τ L_G)^k / k! · f`.
///
/// Uses [`GraphHeat6thChernoff`] over `f64`. The Laplacian is assembled once
/// at construction and reused across all `evolve` calls.
///
/// # Stability domain
/// Caller is responsible for keeping `τ · ρ̄(L_G) < 6` (the radius of clean
/// degree-6 Taylor truncation on the negative real axis; see math.md §19.3).
/// For `P_n` with unit weights, `ρ̄ < 4`, so the user should choose `n_steps`
/// such that `τ = t_final / n_steps < 6/4 = 1.5`.
///
/// # Lifecycle
/// ```js
/// import init, { GraphPath, GraphHeat6 } from "@semiflow/wasm";
/// await init();
/// const g = new GraphPath(64);
/// const heat = new GraphHeat6(g, 4.0);
/// const f0 = new Float64Array(64);
/// f0[0] = 1.0;
/// const out = heat.evolve(0.5, 50, f0);
/// ```
///
/// # Errors
/// Throws JS `Error` with `.kind` matching `SemiflowStatus`:
/// - `.kind = "OutOfDomain"` — `rho_bar <= 0` or non-finite.
/// - `.kind = "GridMismatch"` — `f0.length != n_nodes` in `evolve`.
/// - `.kind = "NanInf"` — `f0` contains NaN or Inf.
/// - `.kind = "OutOfDomain"` — `t_final < 0`, non-finite, or `n_steps == 0`.
/// - `.kind = "ConvergenceFailed"` — semigroup convergence failure.
#[wasm_bindgen]
pub struct GraphHeat6 {
    chernoff: GraphHeat6thChernoff<f64>,
    n_nodes: usize,
}

#[wasm_bindgen]
impl GraphHeat6 {
    /// Build an order-6 graph heat Chernoff from a `GraphPath`.
    ///
    /// ## Parameters
    /// - `graph` — borrow of a `GraphPath` (must have ≥ 1 node).
    /// - `rho_bar` — Gershgorin spectral-radius upper bound. For `P_n` with
    ///   unit edge weights, `4.0` is a safe conservative value. Only used for
    ///   diagnostics in this binding; the order-6 Chernoff itself relies on
    ///   caller-side CFL discipline (see §"Stability domain").
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `rho_bar <= 0` or non-finite.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &GraphPath, rho_bar: f64) -> Result<GraphHeat6, JsValue> {
        if !rho_bar.is_finite() || rho_bar <= 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "rho_bar must be finite and > 0",
            ));
        }
        let n_nodes = graph.n_nodes() as usize;
        // Reconstruct the path graph from n_nodes (cheap O(n)) since GraphPath's
        // inner field is private. Matches the pattern in graph_wasm::GraphHeat::new.
        let g = Arc::new(Graph::<f64>::path(n_nodes));
        let lap = Laplacian::assemble_combinatorial(&g);
        let chernoff = GraphHeat6thChernoff::from_owned(lap);
        Ok(GraphHeat6 { chernoff, n_nodes })
    }

    /// Evolve the graph heat equation by `t_final` seconds using `n_steps`
    /// Chernoff iterations.
    ///
    /// Returns a freshly allocated `Float64Array` (length `n_nodes`).
    /// Stateless: each call uses the explicit `f0` initial condition.
    ///
    /// # Errors
    /// See struct-level docs.
    pub fn evolve(&self, t_final: f64, n_steps: u32, f0: &[f64]) -> Result<Vec<f64>, JsValue> {
        if f0.len() != self.n_nodes {
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

        let graph_arc = Arc::new(Graph::<f64>::path(self.n_nodes));
        #[allow(clippy::cast_possible_truncation)]
        let signal = GraphSignal::from_fn(Arc::clone(&graph_arc), |i| f0[i as usize]);

        let semigroup = ChernoffSemigroup::new(self.chernoff.clone(), n_steps as usize)
            .map_err(|e| err_to_js(&e))?;
        let result = semigroup
            .evolve(t_final, &signal)
            .map_err(|e| err_to_js(&e))?;

        Ok(result.values().to_vec())
    }

    /// Number of nodes the kernel acts on.
    #[wasm_bindgen(js_name = "n_nodes")]
    #[must_use]
    pub fn n_nodes(&self) -> u32 {
        // n_nodes ≤ u32::MAX in WASM (32-bit address space).
        #[allow(clippy::cast_possible_truncation)]
        let n = self.n_nodes as u32;
        n
    }
}
