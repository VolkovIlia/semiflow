//! Time-varying graph heat WASM bindings: Magnus K=4, Magnus K=6, and
//! variable-coefficient Magnus K=4 kernels, all behind `#[cfg(feature = "full")]`.
//!
//! Mirrors `semiflow-py`:
//!   - `MagnusGraphHeat`    → Python `MagnusGraphHeat`  (`magnus_graph_py.rs`)
//!   - `MagnusGraphHeat6`   → Python `MagnusGraphHeat6` (`magnus6.rs`)
//!   - `VarCoefMagnusGraph` → Python `VarCoefMagnusGraph` (`graph_v2_4.rs`)
//!
//! ## Callback contract (JS side)
//!
//! **`lap_at_t_js`** — `(t: number) => Float64Array`
//! Returns undirected edge weights at time `t`. For a path graph `P_n` the
//! array length must equal `n − 1` (edges `0–1, 1–2, …, (n-2)–(n-1)`).
//! Weights must be finite and > 0.  Falls back to unit weights on error.
//!
//! **`a_at_t_js`** (`VarCoefMagnusGraph` only) — `(t: number) => Float64Array`
//! Returns conductivity vector; length = `n_nodes`; all values > 0.
//! Falls back to all-ones on error.
//!
//! ## Error model / panic policy
//!
//! Same `.kind`-tagged JS `Error` as all other graph WASM classes.
//! `panic = "abort"` (ADR-0028 Amendment 1).  No `catch_unwind`.
//!
//! ## f64 only (ADR-0059)

#![allow(unsafe_code)]

use std::sync::Arc;

use js_sys::Float64Array;
use semiflow::{
    magnus_graph::LaplacianAtTime,
    varcoef_magnus_graph::{VarCoefMagnusGraphHeatChernoff as CoreVarCoefMagnus, WeightAtTime},
    Graph, GraphSignal, Laplacian, MagnusGraphHeat6thChernoff as CoreMagnusK6,
    MagnusGraphHeatChernoff as CoreMagnusK4, ScratchPool,
};
use wasm_bindgen::prelude::*;

use crate::{
    error::{err_to_js, make_js_error},
    graph_wasm::GraphPath,
};

mod graph_magnus_wasm_helpers;
use graph_magnus_wasm_helpers::{make_a_at_t, make_lap_at_t, validate_evolve, JsLapCb};

// ---------------------------------------------------------------------------
// MagnusGraphHeat — Magnus K=4 time-varying graph heat
// ---------------------------------------------------------------------------

/// Magnus K=4 graph heat: `∂ₜu = −L_G(t) u` with time-varying edge weights.
///
/// Uses `MagnusGraphHeatChernoff` (GL₂ quadrature, 2× JS callback per step).
/// Mirrors Python `MagnusGraphHeat`.
///
/// ## JS callback
/// `lap_at_t_js(t) => Float64Array` — undirected edge weights.
/// Length = `n_nodes − 1` for a path graph.  Falls back to unit weights on error.
///
/// # Errors
/// - `.kind = "OutOfDomain"` — `rho_bar_max <= 0` or non-finite.
/// - `.kind = "GridMismatch"` / `.kind = "NanInf"` — bad `f0` in `evolve`.
/// - `.kind = "ConvergenceFailed"` — convergence-radius condition violated.
#[wasm_bindgen(js_name = "MagnusGraphHeat")]
pub struct MagnusGraphHeatWasm {
    graph: Arc<Graph<f64>>,
    rho_bar_max: f64,
    convergence_check: bool,
    lap_cb: Arc<LaplacianAtTime<f64>>,
    n_nodes: usize,
}

#[wasm_bindgen(js_class = "MagnusGraphHeat")]
impl MagnusGraphHeatWasm {
    /// Build a Magnus K=4 graph heat state.
    ///
    /// ## Parameters
    /// - `graph`            — `GraphPath` for topology.
    /// - `lap_at_t_js`      — JS `(t) => Float64Array` of undirected weights.
    /// - `rho_bar_max`      — spectral-radius upper bound over all `t`; > 0.
    /// - `convergence_check`— if `true`, gate `rho_bar_max * τ < π/2` per step.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `rho_bar_max <= 0` or non-finite.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &GraphPath,
        lap_at_t_js: js_sys::Function,
        rho_bar_max: f64,
        convergence_check: bool,
    ) -> Result<MagnusGraphHeatWasm, JsValue> {
        if !rho_bar_max.is_finite() || rho_bar_max <= 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "rho_bar_max must be finite and > 0",
            ));
        }
        let n_nodes = graph.n_nodes() as usize;
        let g = Arc::new(Graph::<f64>::path(n_nodes));
        let cb = JsLapCb(lap_at_t_js);
        let lap_fn = make_lap_at_t(cb, &g);
        Ok(Self {
            graph: g,
            rho_bar_max,
            convergence_check,
            lap_cb: Arc::new(lap_fn),
            n_nodes,
        })
    }

    /// Evolve `f0` by `t_final` seconds with `n_steps` Magnus K=4 steps.
    ///
    /// # Errors
    /// See struct-level error table.
    pub fn evolve(&self, t_final: f64, n_steps: u32, f0: &[f64]) -> Result<Vec<f64>, JsValue> {
        validate_evolve(f0, self.n_nodes, t_final, n_steps)?;
        let graph = Arc::clone(&self.graph);
        let lap_cb = Arc::clone(&self.lap_cb);
        let rho_bar_max = self.rho_bar_max;
        let convergence_check = self.convergence_check;
        let n_st = n_steps as usize;
        let lap_at_t: LaplacianAtTime<f64> = Box::new(move |t| lap_cb(t));
        let mghc = CoreMagnusK4::new(Arc::clone(&graph), lap_at_t, rho_bar_max, convergence_check)
            .map_err(|e| err_to_js(&e))?;
        #[allow(clippy::cast_precision_loss)]
        let tau = t_final / n_st as f64;
        let mut state = GraphSignal::from_fn(graph, |i| f0[i as usize]);
        let mut scratch = ScratchPool::new();
        for step in 0..n_st {
            #[allow(clippy::cast_precision_loss)]
            let t_start = step as f64 * tau;
            let mut next = state.clone();
            mghc.apply_into_at(t_start, tau, &state, &mut next, &mut scratch)
                .map_err(|e| err_to_js(&e))?;
            state = next;
        }
        Ok(state.values().to_vec())
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
// MagnusGraphHeat6 — Magnus K=6 time-varying graph heat
// ---------------------------------------------------------------------------

/// Magnus K=6 graph heat: `∂ₜu = −L_G(t) u` — sixth-order GL₃ expansion.
///
/// Uses `MagnusGraphHeat6thChernoff` (3× JS callback per step).
/// f64 only.  Mirrors Python `MagnusGraphHeat6`.
///
/// # Errors
/// Same `.kind` set as `MagnusGraphHeat`.
#[wasm_bindgen(js_name = "MagnusGraphHeat6")]
pub struct MagnusGraphHeat6Wasm {
    graph: Arc<Graph<f64>>,
    rho_bar_max: f64,
    convergence_check: bool,
    lap_cb: Arc<LaplacianAtTime<f64>>,
    n_nodes: usize,
}

#[wasm_bindgen(js_class = "MagnusGraphHeat6")]
impl MagnusGraphHeat6Wasm {
    /// Build a Magnus K=6 graph heat state.
    ///
    /// ## Parameters
    /// - `graph`            — `GraphPath` for topology.
    /// - `lap_at_t_js`      — JS `(t) => Float64Array` of undirected weights.
    /// - `rho_bar_max`      — spectral-radius upper bound; > 0.
    /// - `convergence_check`— gate per-step `rho_bar_max * τ < π/2` check.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `rho_bar_max <= 0` or non-finite.
    #[wasm_bindgen(constructor)]
    pub fn new(
        graph: &GraphPath,
        lap_at_t_js: js_sys::Function,
        rho_bar_max: f64,
        convergence_check: bool,
    ) -> Result<MagnusGraphHeat6Wasm, JsValue> {
        if !rho_bar_max.is_finite() || rho_bar_max <= 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "rho_bar_max must be finite and > 0",
            ));
        }
        let n_nodes = graph.n_nodes() as usize;
        let g = Arc::new(Graph::<f64>::path(n_nodes));
        let cb = JsLapCb(lap_at_t_js);
        let lap_fn = make_lap_at_t(cb, &g);
        Ok(Self {
            graph: g,
            rho_bar_max,
            convergence_check,
            lap_cb: Arc::new(lap_fn),
            n_nodes,
        })
    }

    /// Evolve `f0` by `t_final` seconds with `n_steps` Magnus K=6 steps.
    ///
    /// # Errors
    /// See struct-level error table.
    pub fn evolve(&self, t_final: f64, n_steps: u32, f0: &[f64]) -> Result<Vec<f64>, JsValue> {
        validate_evolve(f0, self.n_nodes, t_final, n_steps)?;
        let graph = Arc::clone(&self.graph);
        let lap_cb = Arc::clone(&self.lap_cb);
        let rho_bar_max = self.rho_bar_max;
        let convergence_check = self.convergence_check;
        let n_st = n_steps as usize;
        let lap_at_t: LaplacianAtTime<f64> = Box::new(move |t| lap_cb(t));
        let mgh6 = CoreMagnusK6::new(Arc::clone(&graph), lap_at_t, rho_bar_max, convergence_check)
            .map_err(|e| err_to_js(&e))?;
        #[allow(clippy::cast_precision_loss)]
        let tau = t_final / n_st as f64;
        let mut state = GraphSignal::from_fn(graph, |i| f0[i as usize]);
        let mut scratch = ScratchPool::new();
        for step in 0..n_st {
            #[allow(clippy::cast_precision_loss)]
            let t_start = step as f64 * tau;
            let mut next = state.clone();
            mgh6.apply_into_at(t_start, tau, &state, &mut next, &mut scratch)
                .map_err(|e| err_to_js(&e))?;
            state = next;
        }
        Ok(state.values().to_vec())
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
// VarCoefMagnusGraph — variable-a × time-dep Magnus K=4
// ---------------------------------------------------------------------------

/// Variable-coefficient × time-dependent graph Magnus K=4:
/// `∂_t u = −L_a(t) u`, `L_a(t) = sqrt(a(t)) ⊙ L_G(t) ⊙ sqrt(a(t))`.
///
/// Mirrors Python `VarCoefMagnusGraph`.  Both callbacks called 2× per step.
/// f64 only.
///
/// ## JS callbacks
/// - `lap_at_t_js(t) => Float64Array` — undirected edge weights.
/// - `a_at_t_js(t) => Float64Array`   — conductivity vector; length `n_nodes`.
///
/// # Errors
/// - `.kind = "OutOfDomain"` — invalid `n_nodes`, `rho_bar_max`, or `a_sup_max`.
/// - `.kind = "GridMismatch"` / `.kind = "NanInf"` — bad `f0`.
/// - `.kind = "ConvergenceFailed"` — convergence-radius condition violated.
#[wasm_bindgen(js_name = "VarCoefMagnusGraph")]
pub struct VarCoefMagnusGraphWasm {
    n_nodes: usize,
    graph: Arc<Graph<f64>>,
    rho_bar_max: f64,
    a_sup_max: f64,
    convergence_check: bool,
    lap_cb: Arc<LaplacianAtTime<f64>>,
    a_cb: Arc<WeightAtTime<f64>>,
}

#[wasm_bindgen(js_class = "VarCoefMagnusGraph")]
impl VarCoefMagnusGraphWasm {
    /// Build a variable-coefficient Magnus K=4 graph heat state.
    ///
    /// ## Parameters
    /// - `n_nodes`          — number of nodes; must be ≥ 1.
    /// - `lap_at_t_js`      — JS `(t) => Float64Array` of undirected edge weights.
    /// - `a_at_t_js`        — JS `(t) => Float64Array` of conductivities.
    /// - `rho_bar_max`      — upper bound on `ρ̄(L_G(t))`; > 0.
    /// - `a_sup_max`        — upper bound on `sqrt(max_i a_i(t))`; > 0.
    /// - `convergence_check`— gate per-step radius check.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — invalid numeric params or `n_nodes == 0`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        n_nodes: u32,
        lap_at_t_js: js_sys::Function,
        a_at_t_js: js_sys::Function,
        rho_bar_max: f64,
        a_sup_max: f64,
        convergence_check: bool,
    ) -> Result<VarCoefMagnusGraphWasm, JsValue> {
        if n_nodes == 0 {
            return Err(make_js_error("OutOfDomain", "n_nodes must be >= 1"));
        }
        if !rho_bar_max.is_finite() || rho_bar_max <= 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "rho_bar_max must be finite and > 0",
            ));
        }
        if !a_sup_max.is_finite() || a_sup_max <= 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "a_sup_max must be finite and > 0",
            ));
        }
        let n = n_nodes as usize;
        let g = Arc::new(Graph::<f64>::path(n));
        let lap_fn = make_lap_at_t(JsLapCb(lap_at_t_js), &g);
        let a_fn = make_a_at_t(JsLapCb(a_at_t_js), n);
        Ok(Self {
            n_nodes: n,
            graph: g,
            rho_bar_max,
            a_sup_max,
            convergence_check,
            lap_cb: Arc::new(lap_fn),
            a_cb: Arc::new(a_fn),
        })
    }

    /// Evolve `f0` by `t_final` seconds with `n_steps` Magnus K=4 steps.
    ///
    /// `t_start` (default `0.0`) offsets absolute time for stitched trajectories.
    ///
    /// # Errors
    /// See struct-level error table.
    pub fn evolve(
        &self,
        t_final: f64,
        n_steps: u32,
        f0: &[f64],
        t_start: f64,
    ) -> Result<Vec<f64>, JsValue> {
        validate_evolve(f0, self.n_nodes, t_final, n_steps)?;
        if !t_start.is_finite() {
            return Err(make_js_error("OutOfDomain", "t_start must be finite"));
        }
        let graph = Arc::clone(&self.graph);
        let lap_cb = Arc::clone(&self.lap_cb);
        let a_cb = Arc::clone(&self.a_cb);
        let n = self.n_nodes;
        let rho_bar_max = self.rho_bar_max;
        let a_sup_max = self.a_sup_max;
        let convergence_check = self.convergence_check;
        let n_st = n_steps as usize;
        let lap_fn: LaplacianAtTime<f64> = Box::new(move |t| lap_cb(t));
        let a_fn: WeightAtTime<f64> = Box::new(move |t| a_cb(t));
        let mc = CoreVarCoefMagnus::<f64>::new(n, lap_fn, a_fn, rho_bar_max, a_sup_max)
            .map_err(|e| err_to_js(&e))?
            .with_radius_check(convergence_check);
        #[allow(clippy::cast_precision_loss)]
        let tau = t_final / n_st as f64;
        let mut state = GraphSignal::from_fn(graph, |i| f0[i as usize]);
        let mut scratch = ScratchPool::<f64>::new();
        for step in 0..n_st {
            #[allow(clippy::cast_precision_loss)]
            let t = t_start + step as f64 * tau;
            let mut next = state.clone();
            mc.apply_into_at(t, tau, &state, &mut next, &mut scratch)
                .map_err(|e| err_to_js(&e))?;
            state = next;
        }
        Ok(state.values().to_vec())
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
