//! `GraphAdjointPresampled` — pre-sampled graph state-adjoint WASM binding (ADR-0180).
//!
//! ## JS API
//!
//! ```js
//! // Step 1: get sample times (2 * n_steps Float64Array)
//! const times = GraphAdjointPresampled.abscissaTimes(tHorizon, nSteps);
//!
//! // Step 2: build valsSeq — flat Float64Array of length 2*n_steps*nnz
//! //   (call your Laplacian-weight function at each time in times[])
//!
//! // Step 3: construct
//! const adj = GraphAdjointPresampled.fromPresampled(
//!     graph, rowPtr, colIdx, valsSeq, nSteps, tHorizon, rhoBar, kind);
//!
//! // Step 4: sweep
//! const lambda0 = adj.evolveStateAdjoint(lambdaN, nSteps);
//! ```
//!
//! ## GL₄ layout invariant (ADR-0180)
//!
//! `valsSeq[2k·nnz .. (2k+1)·nnz]` = Laplacian weights at `times[2k]` (c₁ abscissa
//! for adjoint step k). `valsSeq[(2k+1)·nnz .. (2k+2)·nnz]` = c₂ abscissa.
//! Use `abscissaTimes()` to get the exact sample times.
//!
//! ## Panic policy
//!
//! `panic = "abort"` (WASM release, ADR-0028 Amendment 1). No `catch_unwind`.

#[cfg(feature = "full")]
pub use full::*;

#[cfg(feature = "full")]
mod full {
    use js_sys::Float64Array;
    use wasm_bindgen::prelude::*;

    use semiflow::{
        graph::Graph,
        graph_adjoint_presampled::{
            fill_abscissa_times, PreSampledLaplacianSeq, PreSampledMagnusAdj,
        },
        graph_signal::GraphSignal,
        LaplacianKind, MagnusGraphHeatChernoff,
    };
    use semiflow::scratch::ScratchPool;
    use std::sync::Arc;

    use crate::error::{err_to_js, make_js_error};

    // -------------------------------------------------------------------------
    // Helper: kind from u32
    // -------------------------------------------------------------------------

    fn kind_from_u32(kind: u32) -> LaplacianKind {
        if kind == 1 { LaplacianKind::SymNormalized } else { LaplacianKind::Combinatorial }
    }

    // -------------------------------------------------------------------------
    // GraphAdjointPresampled
    // -------------------------------------------------------------------------

    /// Pre-sampled graph state-adjoint (ADR-0180, Magnus K=4).
    ///
    /// Construct with `GraphAdjointPresampled.fromPresampled(...)`.
    /// The `valsSeq` array is the flat pre-sampled Laplacian weight sequence
    /// with `2·nSteps·nnz` elements in adjoint-schedule order — use
    /// `GraphAdjointPresampled.abscissaTimes(tHorizon, nSteps)` to get the
    /// exact sample times.
    #[wasm_bindgen]
    pub struct GraphAdjointPresampled {
        ps: PreSampledMagnusAdj<f64>,
        /// Dummy graph for GraphSignal allocator (n_nodes only).
        graph: Arc<Graph<f64>>,
        tau: f64,
        n_steps: usize,
    }

    #[wasm_bindgen]
    impl GraphAdjointPresampled {
        /// Fill and return a `Float64Array` of `2 * nSteps` GL₄ abscissa sample times
        /// in adjoint-schedule order.
        ///
        /// The host must sample the Laplacian at exactly these times to build `valsSeq`.
        pub fn abscissaTimes(t_horizon: f64, n_steps: u32) -> Result<Float64Array, JsValue> {
            let ns = n_steps as usize;
            if ns == 0 || !t_horizon.is_finite() || t_horizon <= 0.0 {
                return Err(make_js_error("OutOfDomain", "tHorizon must be finite > 0, nSteps >= 1"));
            }
            let mut buf = vec![0.0_f64; 2 * ns];
            fill_abscissa_times(t_horizon, ns, &mut buf);
            Ok(Float64Array::from(buf.as_slice()))
        }

        /// Construct from a pre-sampled Laplacian weight sequence.
        ///
        /// Parameters
        /// ----------
        /// nNodes     — number of graph nodes.
        /// rowPtr     — CSR row pointer (`Uint32Array`, length `nNodes + 1`).
        /// colIdx     — CSR column indices (`Uint32Array`, length `nnz`).
        /// valsSeq    — flat weight sequence (`Float64Array`, length `2*nSteps*nnz`).
        /// nSteps     — number of adjoint time steps.
        /// tHorizon   — total time horizon.
        /// rhoBar     — Gershgorin spectral bound.
        /// kind       — 0 = combinatorial, 1 = sym-normalized.
        #[allow(clippy::too_many_arguments)]
        pub fn fromPresampled(
            n_nodes: u32,
            row_ptr: &[u32],
            col_idx: &[u32],
            vals_seq: &[f64],
            n_steps: u32,
            t_horizon: f64,
            rho_bar: f64,
            kind: u32,
        ) -> Result<GraphAdjointPresampled, JsValue> {
            let ns = n_steps as usize;
            let nn = n_nodes as usize;
            if ns == 0 || !t_horizon.is_finite() || t_horizon <= 0.0 {
                return Err(make_js_error("OutOfDomain", "tHorizon > 0 and nSteps >= 1 required"));
            }
            let rp: Vec<usize> = row_ptr.iter().map(|&x| x as usize).collect();
            let ci: Vec<u32> = col_idx.to_vec();
            let nnz = ci.len();
            if vals_seq.len() != 2 * ns * nnz {
                return Err(make_js_error(
                    "GridMismatch",
                    &format!("valsSeq.length must equal 2*nSteps*nnz = {}", 2 * ns * nnz),
                ));
            }
            let lk = kind_from_u32(kind);
            let seq = PreSampledLaplacianSeq::new(rp, ci, vals_seq.to_vec(), ns, lk)
                .map_err(|e| err_to_js(&e))?;
            let ps = MagnusGraphHeatChernoff::<f64>::from_presampled(seq, rho_bar, false)
                .map_err(|e| err_to_js(&e))?;
            let graph = Arc::new(Graph::<f64>::path(nn.max(1)));
            let tau = t_horizon / ns as f64;
            Ok(GraphAdjointPresampled { ps, graph, tau, n_steps: ns })
        }

        /// Backward costate sweep `lambdaN → lambda0`.
        ///
        /// `nSteps` must match the construction value.
        pub fn evolveStateAdjoint(
            &self,
            lambda_n: &[f64],
            n_steps: u32,
        ) -> Result<Float64Array, JsValue> {
            let ns = n_steps as usize;
            if ns != self.n_steps {
                return Err(make_js_error(
                    "OutOfDomain",
                    &format!("nSteps={ns} != construction nSteps={}", self.n_steps),
                ));
            }
            if lambda_n.len() != self.ps.n_nodes() {
                return Err(make_js_error(
                    "GridMismatch",
                    "lambdaN.length must equal nNodes",
                ));
            }
            let src = GraphSignal::from_fn(Arc::clone(&self.graph), |i| lambda_n[i as usize]);
            let mut dst = GraphSignal::zeros(Arc::clone(&self.graph));
            let mut scratch = ScratchPool::new();
            self.ps
                .evolve_state_adjoint_into(self.tau, ns, &src, &mut dst, &mut scratch)
                .map_err(|e| err_to_js(&e))?;
            Ok(Float64Array::from(dst.values()))
        }

        /// Number of graph nodes.
        pub fn nNodes(&self) -> usize {
            self.ps.n_nodes()
        }

        /// Number of construction-time steps.
        pub fn nSteps(&self) -> usize {
            self.n_steps
        }
    }
}
