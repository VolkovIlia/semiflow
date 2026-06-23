//! WASM JS classes for `Laplacian` introspection + `GraphTraj` (degenerate)
//! (C-parity pass, `full` feature, ADR-0028/0171).
//!
//! Mirrors `semiflow-py` `laplacian_introspect.rs` + `structured_traj.rs`.
//!
//! ## JS classes
//!
//! ### `Laplacian`
//!
//! ```js
//! import init, { GraphPath, Laplacian } from "@semiflow/wasm";
//! await init();
//! const g   = new GraphPath(8);
//! const lap = Laplacian.combinatorial(g);
//! console.log(lap.nNodes());          // 8
//! console.log(lap.isCombinatorial()); // true
//! console.log(lap.spectralBound());   // e.g. 3.414…
//! const rp = lap.rowPtr();            // Uint32Array length n+1
//! const ci = lap.colIdx();            // Uint32Array length nnz
//! const v  = lap.vals();              // Float64Array length nnz
//! const D  = lap.toDense();           // Float64Array length n*n (row-major)
//! // D.length / rp.length === n  →  n = rp.length - 1
//! ```
//!
//! ### `GraphTraj`
//!
//! ```js
//! const g    = new GraphPath(8);
//! const traj = new GraphTraj(g, 1.0);  // t_horizon = 1.0
//! console.log(traj.nNodes());          // 8
//! console.log(traj.tHorizon());        // 1.0
//! console.log(traj.nSegments());       // 1
//! ```
//!
//! ## Error model
//!
//! `.kind`-tagged JS `Error` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1); no `catch_unwind`.

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use crate::error::make_js_error;
use crate::graph_wasm::GraphPath;
use semiflow_core::{Graph, Laplacian, LaplacianKind};

// ---------------------------------------------------------------------------
// Laplacian JS class
// ---------------------------------------------------------------------------

/// Sparse Laplacian in CSR layout assembled from a `GraphPath`.
///
/// **Factories**: `Laplacian.combinatorial(graph)`, `Laplacian.normalized(graph)`.
///
/// # Errors
/// All fallible methods throw `.kind`-tagged JS `Error`.
#[wasm_bindgen(js_name = "Laplacian")]
pub struct LaplacianWasm {
    inner: Arc<Laplacian<f64>>,
    // Keep topology carrier for GraphSignal usage in downstream callers.
    #[allow(dead_code)]
    graph: Arc<Graph<f64>>,
}

#[wasm_bindgen(js_class = "Laplacian")]
impl LaplacianWasm {
    /// Assemble the combinatorial Laplacian `L = D − W` from `graph`.
    ///
    /// ## Parameters
    /// - `graph` — borrow of a `GraphPath`; must have ≥ 1 node.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `graph.n_nodes() == 0`.
    #[wasm_bindgen(js_name = "combinatorial")]
    pub fn combinatorial(graph: &GraphPath) -> Result<LaplacianWasm, JsValue> {
        let n = graph.n_nodes() as usize;
        if n == 0 {
            return Err(make_js_error("OutOfDomain", "n_nodes must be >= 1"));
        }
        let g = Arc::new(Graph::<f64>::path(n));
        let lap = Arc::new(Laplacian::assemble_combinatorial(&g));
        Ok(LaplacianWasm { inner: lap, graph: g })
    }

    /// Assemble the symmetric normalized Laplacian
    /// `L_sym = I − D^{−½} W D^{−½}` from `graph`.
    ///
    /// ## Parameters
    /// - `graph` — borrow of a `GraphPath`; must have ≥ 1 node.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `graph.n_nodes() == 0`.
    #[wasm_bindgen(js_name = "normalized")]
    pub fn normalized(graph: &GraphPath) -> Result<LaplacianWasm, JsValue> {
        let n = graph.n_nodes() as usize;
        if n == 0 {
            return Err(make_js_error("OutOfDomain", "n_nodes must be >= 1"));
        }
        let g = Arc::new(Graph::<f64>::path(n));
        let lap = Arc::new(Laplacian::assemble_normalized(&g));
        Ok(LaplacianWasm { inner: lap, graph: g })
    }

    /// Number of nodes in the Laplacian.
    #[must_use]
    #[wasm_bindgen(js_name = "nNodes")]
    pub fn n_nodes(&self) -> u32 {
        self.inner.n_nodes() as u32
    }

    /// Returns `true` iff this is the combinatorial Laplacian.
    #[must_use]
    #[wasm_bindgen(js_name = "isCombinatorial")]
    pub fn is_combinatorial(&self) -> bool {
        self.inner.kind() == LaplacianKind::Combinatorial
    }

    /// Returns `true` iff this is the symmetric-normalized Laplacian.
    #[must_use]
    #[wasm_bindgen(js_name = "isNormalized")]
    pub fn is_normalized(&self) -> bool {
        self.inner.kind() == LaplacianKind::SymNormalized
    }

    /// Gershgorin spectral-radius upper bound `ρ̄ ≥ ρ(L_G)` (cached).
    #[must_use]
    #[wasm_bindgen(js_name = "spectralBound")]
    pub fn spectral_bound(&self) -> f64 {
        self.inner.spectral_radius_bound()
    }

    /// CSR row-pointer array (copy), length `n_nodes + 1`.
    ///
    /// Returns a `Uint32Array`. Values are `usize` widened to `u32`; graphs
    /// are bounded to u32::MAX nodes so no truncation occurs in practice.
    ///
    /// # Errors
    /// Throws on allocation failure (extremely unlikely).
    #[wasm_bindgen(js_name = "rowPtr")]
    pub fn row_ptr(&self) -> Vec<u32> {
        self.inner.row_ptr().iter().map(|&x| x as u32).collect()
    }

    /// CSR column-index array (copy), length `n_directed_edges`.
    ///
    /// Returns a `Uint32Array`.
    #[wasm_bindgen(js_name = "colIdx")]
    pub fn col_idx(&self) -> Vec<u32> {
        self.inner.col_idx().iter().map(|&x| x).collect()
    }

    /// CSR values array (copy), length `n_directed_edges`.
    ///
    /// Returns a `Float64Array`.
    #[wasm_bindgen(js_name = "vals")]
    pub fn vals(&self) -> Vec<f64> {
        self.inner.vals().to_vec()
    }

    /// Dense `n × n` row-major matrix reconstructed from CSR.
    ///
    /// Returns a `Float64Array` of length `n * n`.
    /// The matrix dimension `n = nNodes()`.
    ///
    /// Memory: O(n²).
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `n * n` overflows `usize`.
    #[wasm_bindgen(js_name = "toDense")]
    pub fn to_dense(&self) -> Result<Vec<f64>, JsValue> {
        let n = self.inner.n_nodes();
        let size = n.checked_mul(n).ok_or_else(|| {
            make_js_error("OutOfDomain", "n * n overflows usize")
        })?;
        let mut buf = vec![0.0_f64; size];
        let row_ptr = self.inner.row_ptr();
        let col_idx = self.inner.col_idx();
        let vals = self.inner.vals();
        for row in 0..n {
            for k in row_ptr[row]..row_ptr[row + 1] {
                let col = col_idx[k] as usize;
                buf[row * n + col] = vals[k];
            }
        }
        Ok(buf)
    }
}

// ---------------------------------------------------------------------------
// GraphTraj JS class
// ---------------------------------------------------------------------------

/// Degenerate fixed-topology graph trajectory (1 segment, constant Laplacian).
///
/// Mirrors Python `GraphTraj(graph, t_horizon)` degenerate constructor.
///
/// Full multi-segment trajectories with JS-callable weight functions
/// cannot cross the `wasm-bindgen` boundary (closures are not `Send+Sync`).
/// This class exposes the most useful degenerate constructor.
///
/// # Errors
/// - `.kind = "NanInf"` — `t_horizon` is not finite.
/// - `.kind = "OutOfDomain"` — `t_horizon <= 0`.
#[wasm_bindgen(js_name = "GraphTraj")]
pub struct GraphTrajWasm {
    n_nodes: usize,
    t_horizon: f64,
    n_segments: usize,
}

#[wasm_bindgen(js_class = "GraphTraj")]
impl GraphTrajWasm {
    /// Build a degenerate fixed-topology `GraphTraj`.
    ///
    /// ## Parameters
    /// - `graph`     — borrow of a `GraphPath`.
    /// - `t_horizon` — total horizon; must be finite and > 0.
    ///
    /// # Errors
    /// - `.kind = "NanInf"` — `t_horizon` is NaN or Inf.
    /// - `.kind = "OutOfDomain"` — `t_horizon <= 0`.
    #[wasm_bindgen(constructor)]
    pub fn new(graph: &GraphPath, t_horizon: f64) -> Result<GraphTrajWasm, JsValue> {
        if !t_horizon.is_finite() {
            return Err(make_js_error("NanInf", "t_horizon must be finite"));
        }
        if t_horizon <= 0.0 {
            return Err(make_js_error("OutOfDomain", "t_horizon must be > 0"));
        }
        let n = graph.n_nodes() as usize;
        Ok(GraphTrajWasm { n_nodes: n, t_horizon, n_segments: 1 })
    }

    /// Number of nodes in the trajectory's graph.
    #[must_use]
    #[wasm_bindgen(js_name = "nNodes")]
    pub fn n_nodes(&self) -> u32 {
        self.n_nodes as u32
    }

    /// Total time horizon.
    #[must_use]
    #[wasm_bindgen(js_name = "tHorizon")]
    pub fn t_horizon(&self) -> f64 {
        self.t_horizon
    }

    /// Number of segments (always 1 for fixed-topology).
    #[must_use]
    #[wasm_bindgen(js_name = "nSegments")]
    pub fn n_segments(&self) -> u32 {
        self.n_segments as u32
    }
}
