//! Graph PDE WASM bindings for `semiflow-core` (v2.2 Wave C).
//!
//! Exposes two JS classes for graph heat diffusion:
//!   `new GraphPath(n_nodes)` — path graph `0−1−…−(n−1)` with unit weights.
//!   `new GraphHeat(graph, rho_bar)` — order-2 heat Chernoff on the graph.
//!   `heat.evolve(t_final, n_steps, f0)` — advance `t_final` seconds,
//!       return a `Float64Array` copy of the final state.
//!
//! # Design (mirrors `Heat1D` from v0.10.0 Wave C — see ADR-0059)
//!
//! The WASM binding wraps `GraphHeatChernoff` (order-2, ζ-A Taylor) driven
//! by `ChernoffSemigroup::evolve`.  `GraphPath` is a thin holder for the
//! `Arc<Graph<f64>>` that `GraphHeat` internally consumes.
//!
//! # Error model
//!
//! All fallible methods throw a JS `Error` with a `.kind` property matching the
//! `SemiflowStatus` naming used by `semiflow-ffi` and `semiflow-py` (same table
//! as in `error.rs`).  This is the same convention used by `Heat1D`.
//!
//! # Panic boundary
//!
//! Uses workspace `[profile.release]` (`panic = "abort"`) — same as `Heat1D`
//! (ADR-0028 Amendment 1 / ADR-0029).  Do NOT add `catch_unwind` here;
//! panics abort rather than unwind on this profile.  Error paths use early
//! `return Err(JsValue::from_str(...))` instead.
//!
//! # f64 only
//!
//! All graph PDE bindings are `f64`-only per ADR-0059 §"Out of scope".
//! `f32` graph bindings are deferred to v2.3+.
//!
//! # Example (JS/TS)
//! ```js
//! import init, { GraphPath, GraphHeat, panic_hook_init } from "@semiflow/wasm";
//! await init();
//! panic_hook_init();
//! const g = new GraphPath(64);               // P_64 path graph
//! const heat = new GraphHeat(g, 4.0);        // rho_bar = 4.0
//! const f0 = new Float64Array(64);
//! f0[0] = 1.0;
//! const result = heat.evolve(0.5, 50, f0);   // Float64Array copy
//! ```

#![allow(unsafe_code)]

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};
use semiflow_core::{ChernoffSemigroup, Graph, GraphHeatChernoff, GraphSignal, Laplacian};

// ---------------------------------------------------------------------------
// GraphPath
// ---------------------------------------------------------------------------

/// Path graph `0 − 1 − … − (n_nodes − 1)` with unit edge weights.
///
/// This is the simplest non-trivial graph topology and mirrors the `P_n` path
/// graph used in all cross-binding identity gates (ADR-0059 §2.5).
///
/// # Lifecycle
/// ```js
/// const g = new GraphPath(64);   // create P_64
/// console.log(g.n_nodes());      // 64
/// ```
///
/// # Errors
/// - `.kind = "OutOfDomain"` — `n_nodes == 0`.
#[wasm_bindgen]
pub struct GraphPath {
    inner: Arc<Graph<f64>>,
}

#[wasm_bindgen]
impl GraphPath {
    /// Create a path graph on `n_nodes` nodes with unit edge weights.
    ///
    /// ## Parameters
    /// - `n_nodes` — number of nodes; must be ≥ 1.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `n_nodes == 0` (empty graph not allowed).
    #[wasm_bindgen(constructor)]
    pub fn new(n_nodes: u32) -> Result<GraphPath, JsValue> {
        if n_nodes == 0 {
            return Err(make_js_error("OutOfDomain", "n_nodes must be >= 1"));
        }
        let g = Graph::<f64>::path(n_nodes as usize);
        Ok(GraphPath { inner: Arc::new(g) })
    }

    /// Number of nodes in the graph.
    #[must_use]
    #[wasm_bindgen(js_name = "n_nodes")]
    pub fn n_nodes(&self) -> u32 {
        // n_nodes is always <= u32::MAX in practice (WASM is 32-bit).
        #[allow(clippy::cast_possible_truncation)]
        let n = self.inner.n_nodes() as u32;
        n
    }
}

// ---------------------------------------------------------------------------
// GraphHeat
// ---------------------------------------------------------------------------

/// Order-2 graph heat Chernoff state for `∂ₜu = −L_G u`.
///
/// Uses [`GraphHeatChernoff`] with the ζ-A Taylor-2 variant:
/// `S(τ) f = f − τ·L_G·f + (τ²/2)·L_G²·f` (Wave 2.1B).
///
/// The Laplacian is assembled once at construction (combinatorial) and reused
/// across all `evolve` calls — no allocation per step in steady state.
///
/// # Lifecycle
/// ```js
/// const g = new GraphPath(64);
/// const heat = new GraphHeat(g, 4.0);          // rho_bar ≥ spectral radius
/// const f0 = new Float64Array(64);
/// f0[0] = 1.0;
/// const out = heat.evolve(0.5, 50, f0);        // Float64Array copy of result
/// ```
///
/// # Error model
/// All methods throw a JS `Error` with `.kind` matching `SemiflowStatus` names.
///
/// # `rho_bar` parameter
/// `rho_bar` is the Gershgorin spectral-radius upper bound for the Laplacian
/// `L_G`.  For a path graph `P_n` with unit weights, the largest eigenvalue is
/// `2(1 − cos(π/n)) < 4`, so `rho_bar = 4.0` is safe.  Pass a tighter bound
/// for better CFL acceptance in `evolve`.
///
/// # Panic boundary
/// Uses `[profile.release]` `panic=abort`.  No `catch_unwind` wrapper.
/// Caller errors use `Err(JsValue)` instead.
///
/// # Errors
/// - `.kind = "OutOfDomain"` — `rho_bar <= 0` or non-finite.
/// - `.kind = "GridMismatch"` — `f0.length != n_nodes` in `evolve`.
/// - `.kind = "NanInf"` — `f0` contains NaN or Inf.
/// - `.kind = "OutOfDomain"` — `t_final < 0`, non-finite, or `n_steps == 0`.
/// - `.kind = "ConvergenceFailed"` — CFL / convergence condition violated.
#[wasm_bindgen]
pub struct GraphHeat {
    chernoff: GraphHeatChernoff<f64>,
    n_nodes: usize,
}

#[wasm_bindgen]
impl GraphHeat {
    /// Build an order-2 graph heat Chernoff from a `GraphPath`.
    ///
    /// Assembles the combinatorial Laplacian `L = D − W` once at construction.
    /// The `GraphPath` object may be dropped afterwards; the Laplacian is
    /// stored in an `Arc` internally.
    ///
    /// ## Parameters
    /// - `graph` — borrow of a `GraphPath`; topology must have ≥ 1 node.
    /// - `rho_bar` — Gershgorin spectral-radius upper bound `ρ̄ ≥ ρ(L_G)`.
    ///   For `P_n` with unit weights, `4.0` is a safe conservative value.
    ///   Must be > 0 and finite (only used for error diagnostics in this binding;
    ///   the order-2 Chernoff itself does not enforce a CFL via `rho_bar`).
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `rho_bar <= 0` or non-finite.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &GraphPath, rho_bar: f64) -> Result<GraphHeat, JsValue> {
        if !rho_bar.is_finite() || rho_bar <= 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "rho_bar must be finite and > 0",
            ));
        }
        let lap = Laplacian::assemble_combinatorial(&graph.inner);
        let lap_arc = Arc::new(lap);
        let chernoff = GraphHeatChernoff::with_zeta_a(lap_arc);
        let n_nodes = graph.inner.n_nodes();
        Ok(GraphHeat { chernoff, n_nodes })
    }

    /// Evolve the graph heat equation by `t_final` seconds using `n_steps`
    /// Chernoff iterations, starting from initial condition `f0`.
    ///
    /// Returns a freshly allocated `Float64Array` (length `n_nodes`) with the
    /// solution at time `t_final`.  Does NOT mutate the `GraphHeat` object —
    /// each call is stateless (initial condition is passed explicitly).
    ///
    /// ## Parameters
    /// - `t_final` — time horizon; must be ≥ 0 and finite.
    /// - `n_steps` — number of Chernoff steps; must be ≥ 1.
    /// - `f0` — `Float64Array` of length exactly `n_nodes`.  All elements
    ///   must be finite.
    ///
    /// ## Note on `t_final = 0`
    /// Applying the Chernoff kernel `n_steps` times with `τ = 0` underflows
    /// to near-zero rather than recovering the initial condition.  Callers who
    /// need the identity should skip the call or return `f0` directly.
    ///
    /// # Errors
    /// - `.kind = "GridMismatch"` — `f0.length != n_nodes`.
    /// - `.kind = "NanInf"` — `f0` contains NaN or Inf.
    /// - `.kind = "OutOfDomain"` — `t_final < 0`, non-finite, or
    ///   `n_steps == 0`.
    /// - `.kind = "ConvergenceFailed"` — semigroup convergence failure
    ///   (step `τ = t_final / n_steps` too large for the chosen kernel).
    pub fn evolve(&self, t_final: f64, n_steps: u32, f0: &[f64]) -> Result<Vec<f64>, JsValue> {
        // Early returns — validate before any allocation.
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

        // Build initial GraphSignal from the input slice.
        // Rebuild graph Arc from Laplacian topology for GraphSignal.
        // GraphHeatChernoff stores Arc<Laplacian>; we need Arc<Graph> for the
        // signal.  Reconstruct path graph with correct n_nodes (cheap: O(n)).
        let graph_arc = Arc::new(Graph::<f64>::path(self.n_nodes));
        #[allow(clippy::cast_possible_truncation)]
        let signal = GraphSignal::from_fn(Arc::clone(&graph_arc), |i| f0[i as usize]);

        // Wrap in ChernoffSemigroup and evolve.
        let semigroup = ChernoffSemigroup::new(self.chernoff.clone(), n_steps as usize)
            .map_err(|e| err_to_js(&e))?;
        let result = semigroup
            .evolve(t_final, &signal)
            .map_err(|e| err_to_js(&e))?;

        Ok(result.values().to_vec())
    }
}
