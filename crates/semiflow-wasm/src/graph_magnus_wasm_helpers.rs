//! Shared helpers for `graph_magnus_wasm.rs` — extracted to keep file ≤500 lines.

use std::sync::Arc;

use js_sys::Float64Array;
use semiflow::{
    magnus_graph::LaplacianAtTime, varcoef_magnus_graph::WeightAtTime, Graph, Laplacian,
};
use wasm_bindgen::prelude::*;

use crate::error::make_js_error;

/// Newtype for `js_sys::Function` with unsafe `Send + Sync`.
///
/// # Safety
///
/// `wasm32-unknown-unknown` is single-threaded by spec — no OS threads share
/// WASM linear memory.  `Send` and `Sync` are vacuously safe (ADR-0034).
pub(super) struct JsLapCb(pub(super) js_sys::Function);

// Safety: wasm32-unknown-unknown is single-threaded (ADR-0034).
unsafe impl Send for JsLapCb {}
unsafe impl Sync for JsLapCb {}

impl JsLapCb {
    /// Call JS callback with one `f64` argument → `Float64Array` or `None`.
    pub(super) fn call_arr(&self, t: f64) -> Option<Vec<f64>> {
        let arg = JsValue::from_f64(t);
        let ret = self.0.call1(&JsValue::NULL, &arg).ok()?;
        if !ret.is_instance_of::<Float64Array>() {
            return None;
        }
        let arr = Float64Array::from(ret);
        let mut buf = vec![0.0_f64; arr.length() as usize];
        arr.copy_to(&mut buf);
        Some(buf)
    }
}

/// Unit-weight fallback Laplacian for a path graph with `n_nodes` nodes.
pub(super) fn unit_lap(n_nodes: usize) -> Arc<Laplacian<f64>> {
    let g = Graph::<f64>::path(n_nodes);
    Arc::new(Laplacian::assemble_combinatorial(&g))
}

/// Build `LaplacianAtTime<f64>` from a JS callback + stored graph topology.
pub(super) fn make_lap_at_t(cb: JsLapCb, base_graph: &Graph<f64>) -> LaplacianAtTime<f64> {
    let n_nodes = base_graph.n_nodes();
    let row_ptr = base_graph.row_ptr().to_vec();
    let col_idx = base_graph.col_idx().to_vec();

    Box::new(move |t: f64| -> Arc<Laplacian<f64>> {
        let Some(weights) = cb.call_arr(t) else {
            return unit_lap(n_nodes);
        };
        let mut edges: Vec<(u32, u32, f64)> = Vec::new();
        let mut w_idx = 0usize;
        for u in 0..n_nodes {
            for &nb in &col_idx[row_ptr[u]..row_ptr[u + 1]] {
                let v = nb as usize;
                if u < v {
                    if w_idx >= weights.len() {
                        return unit_lap(n_nodes);
                    }
                    let w = weights[w_idx];
                    if !w.is_finite() || w <= 0.0 {
                        return unit_lap(n_nodes);
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    edges.push((u as u32, v as u32, w));
                    w_idx += 1;
                }
            }
        }
        if w_idx != weights.len() {
            return unit_lap(n_nodes);
        }
        match Graph::<f64>::from_edges(n_nodes, edges) {
            Ok(g) => Arc::new(Laplacian::assemble_combinatorial(&g)),
            Err(_) => unit_lap(n_nodes),
        }
    })
}

/// Build `WeightAtTime<f64>` from a JS callback returning a `Float64Array`.
pub(super) fn make_a_at_t(cb: JsLapCb, n_nodes: usize) -> WeightAtTime<f64> {
    Box::new(move |t: f64| -> Vec<f64> {
        let fallback = vec![1.0_f64; n_nodes];
        let Some(weights) = cb.call_arr(t) else {
            return fallback;
        };
        if weights.len() != n_nodes {
            return fallback;
        }
        for &v in &weights {
            if !v.is_finite() || v <= 0.0 {
                return fallback;
            }
        }
        weights
    })
}

/// Validate common evolve parameters.
pub(super) fn validate_evolve(
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
