//! Wave P6 — `GraphTraj` + `StrangGraph` (M22).
//!
//! Fixed-topology graph trajectory and bipartite Strang split on graph signals.
//! Split from `structured.rs` for suckless file-size compliance.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    graph::Graph, graph_heat::GraphHeatChernoff, graph_signal::GraphSignal, graph_traj::GraphTraj,
    strang_graph::StrangSplitGraph, ChernoffFunction, ChernoffSemigroup,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_extra::PyGraph,
    graph_py::extract_f64_vec,
    panic::catch_panic_py,
};

// ===========================================================================
// GraphTraj — fixed-topology graph trajectory (M22)
// ===========================================================================

/// Fixed-topology graph trajectory (M22, ADR-0052, math §14.1).
///
/// A piecewise-smooth trajectory over ``[0, t_horizon]`` with a single
/// segment and constant combinatorial Laplacian assembled from ``graph``.
///
/// Full multi-segment trajectories with Python-callable weight functions
/// cannot cross the `PyO3` boundary (closures are not ``Send+Sync``).
/// This class exposes the most useful degenerate constructor.
///
/// Parameters
/// ----------
/// graph : Graph
///     Topology (combinatorial Laplacian assembled internally).
/// `t_horizon` : float
///     Total horizon (must be > 0, finite).
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`OutOfDomain`' if `t_horizon` <= 0 or non-finite.
#[pyclass(name = "GraphTraj")]
pub struct PyGraphTraj {
    n_nodes: usize,
    t_horizon: f64,
    n_segments: usize,
    // Kept for potential future multi-step PyO3 use; not read directly in Python methods.
    #[allow(dead_code)]
    graph: Arc<Graph<f64>>,
}

#[pymethods]
impl PyGraphTraj {
    #[new]
    fn new(graph: &PyGraph, t_horizon: f64) -> PyResult<Self> {
        catch_panic_py!({
            if !t_horizon.is_finite() || t_horizon <= 0.0 {
                return Err(new_pyerr("OutOfDomain", "t_horizon must be finite and > 0"));
            }
            let n = graph.inner.n_nodes();
            // Validate via core constructor (fixed-topology degenerate).
            let g = Arc::clone(&graph.inner);
            let g2 = Arc::clone(&g);
            let wfn: semiflow::graph_traj::SegmentWeightFn<f64> = Box::new(move |_t| {
                Arc::new(semiflow::graph::Laplacian::assemble_combinatorial(&g2))
            });
            let _traj = GraphTraj::fixed_topology(Arc::clone(&g), wfn, t_horizon)
                .map_err(|e| from_core(&e))?;
            Ok(Self {
                n_nodes: n,
                t_horizon,
                n_segments: 1,
                graph: g,
            })
        })
    }

    /// Number of nodes in the trajectory's graph.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Total time horizon.
    #[getter]
    fn t_horizon(&self) -> f64 {
        self.t_horizon
    }

    /// Number of segments (always 1 for fixed-topology).
    #[getter]
    fn n_segments(&self) -> usize {
        self.n_segments
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphTraj(n_nodes={}, t_horizon={:.4}, n_segments={})",
            self.n_nodes, self.t_horizon, self.n_segments,
        )
    }
}

// ===========================================================================
// StrangGraph — bipartite Strang split on graph signals (M22)
// ===========================================================================

/// Palindromic Strang split for graph heat Chernoff kernels (M22, math §12.8).
///
/// ``S(τ) f = A(τ/2) ∘ B(τ) ∘ A(τ/2) · f`` on ``GraphSignal<f64>``.
///
/// Uses edge-parity 2-coloring to guarantee commutativity ``[L_A, L_B] = 0``
/// (A-edges and B-edges are on disjoint node pairs).  This yields global
/// order-2 convergence (palindromic Strang + commuting ⟹ BCH residue cancels).
///
/// Factories
/// ---------
/// :meth:`from_path` — path graph bipartite coloring.
/// :meth:`from_cycle` — even-length cycle graph bipartite coloring.
///
/// The state type is a flat float64 node-signal (``len = n_nodes``).
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`OutOfDomain`' if graph is too small for the chosen factory.
#[pyclass(name = "StrangGraph")]
pub struct PyStrangGraph {
    strang: StrangSplitGraph<GraphHeatChernoff<f64>, GraphHeatChernoff<f64>, f64>,
    graph: Arc<Graph<f64>>,
    n_nodes: usize,
}

#[pymethods]
impl PyStrangGraph {
    /// Build from a path graph ``P_n`` via even/odd edge-parity 2-coloring.
    ///
    /// Requires ``n_nodes >= 2``.
    #[staticmethod]
    fn from_path(graph: &PyGraph) -> PyResult<Self> {
        catch_panic_py!({
            let g = Arc::clone(&graph.inner);
            let n = g.n_nodes();
            let strang = StrangSplitGraph::new_bipartite_path(&g).map_err(|e| from_core(&e))?;
            Ok(Self {
                strang,
                graph: g,
                n_nodes: n,
            })
        })
    }

    /// Build from an even-length cycle graph ``C_n`` via edge-parity coloring.
    ///
    /// Requires ``n_nodes >= 4`` and ``n_nodes % 2 == 0``.
    #[staticmethod]
    fn from_cycle(graph: &PyGraph) -> PyResult<Self> {
        catch_panic_py!({
            let g = Arc::clone(&graph.inner);
            let n = g.n_nodes();
            let strang = StrangSplitGraph::new_bipartite_cycle(&g).map_err(|e| from_core(&e))?;
            Ok(Self {
                strang,
                graph: g,
                n_nodes: n,
            })
        })
    }

    /// Evolve signal ``f0`` to time ``t_final`` with ``n_steps`` Strang steps.
    ///
    /// GIL released during compute (ADR-0031).
    ///
    /// Parameters
    /// ----------
    /// `t_final` : float
    ///     Target time (>= 0, finite).
    /// `n_steps` : int
    ///     Number of Strang steps (>= 1).
    /// f0 : array-like
    ///     Initial node signal; float64 array of length ``n_nodes``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray[float64]
    ///     Evolved signal (copy), length ``n_nodes``.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if `t_final` < 0, non-finite, or `n_steps` == 0.
    ///     kind='`GridMismatch`' if len(f0) != `n_nodes`.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        t_final: f64,
        n_steps: u32,
        f0: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            if !t_final.is_finite() || t_final < 0.0 {
                return Err(new_pyerr("OutOfDomain", "t_final must be finite and >= 0"));
            }
            if n_steps == 0 {
                return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
            }
            let input = extract_f64_vec(f0).map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("f0 must be a float64 array")
            })?;
            if input.len() != self.n_nodes {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("f0 length {}, expected {}", input.len(), self.n_nodes),
                ));
            }
            let strang = self.strang.clone();
            let graph = Arc::clone(&self.graph);
            let n_st = n_steps as usize;
            let result: Result<Vec<f64>, semiflow::SemiflowError> =
                py.detach(|| compute_strang(strang, graph, &input, t_final, n_st));
            Ok(result.map_err(|e| from_core(&e))?.as_slice().to_pyarray(py))
        })
    }

    /// Return approximation order (2 when commutativity holds).
    fn order(&self) -> u32 {
        self.strang.order()
    }

    /// Number of nodes in the graph.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    fn __repr__(&self) -> String {
        format!(
            "StrangGraph(n_nodes={}, order={})",
            self.n_nodes,
            self.strang.order(),
        )
    }
}

// ---------------------------------------------------------------------------
// StrangGraph GIL-free compute
// ---------------------------------------------------------------------------

fn compute_strang(
    strang: StrangSplitGraph<GraphHeatChernoff<f64>, GraphHeatChernoff<f64>, f64>,
    graph: Arc<Graph<f64>>,
    input: &[f64],
    t_final: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(strang, n_steps)?;
    let f0 = GraphSignal::from_fn(Arc::clone(&graph), |i| input[i as usize]);
    let result = sg.evolve(t_final, &f0)?;
    Ok(result.values().to_vec())
}
