//! Wave P6 — quantum graphs + matrix diffusion + point-eval + graph trajectories.
//!
//! ADR-0111 parity items M16–M22.
//!
//! | pyclass / fn          | Core type                                     | M# | Module |
//! |-----------------------|-----------------------------------------------|----|--------|
//! | `QuantumGraph`        | `QuantumGraph<f64>`                           | M16 | here |
//! | `QuantumGraphHeat`    | `QuantumGraphHeatChernoff<f64>`               | M16 | here |
//! | `MatrixDiffusion1D`   | `MatrixDiffusionChernoff<f64, 2>`             | M17 | `structured_matrix` |
//! | `PointEval`           | `DiffusionChernoff<f64>` w/ `PointEval` trait | M18 | `structured_point` |
//! | `sample_gridfn2d`     | free fn `sample_gridfn2d`                     | M18 | `structured_point` |
//! | `GraphTraj`           | `GraphTraj<f64>` (single-segment, fixed topo) | M22 | `structured_traj` |
//! | `StrangGraph`         | `StrangSplitGraph<GH, GH, f64>`               | M22 | `structured_traj` |
//!
//! ## Design notes — see original structured.rs header (history preserved)

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_pass_by_value
)]

use std::sync::Arc;

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;

use semiflow_core::{
    quantum_graph::{
        QuantumGraph as CoreQuantumGraph, QuantumGraphHeatChernoff, QuantumGraphSignal,
    },
    ChernoffSemigroup,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_py::extract_f64_vec,
    panic::catch_panic_py,
};

// Re-export sibling classes so `register()` can reference them.
pub(crate) use crate::structured_matrix::PyMatrixDiffusion1D;
pub(crate) use crate::structured_point::{sample_gridfn2d, PyPointEval};
pub(crate) use crate::structured_traj::{PyGraphTraj, PyStrangGraph};

// ===========================================================================
// QuantumGraph — topology/metric data class (M16)
// ===========================================================================

/// Metric graph for quantum graph heat equations (M16, ADR-0078, math §29).
///
/// A metric graph is a combinatorial graph whose edges carry continuous
/// lengths ``ℓ_e > 0``.  Heat evolves per-edge via ``½∂²_x`` with Kirchhoff
/// vertex conditions (continuity + sum of inward derivatives = 0).
///
/// Factories
/// ---------
/// :meth:`path` — path graph ``P_{n+1}`` (``n_edges+1`` vertices).
/// :meth:`star` — star graph (one hub, ``n_arms`` leaves).
/// :meth:`from_edges` — arbitrary topology from triplet array.
///
/// All factories require ``edge_length > 0`` and ``n_grid >= 4``.
#[pyclass(name = "QuantumGraph")]
pub struct PyQuantumGraph {
    pub(crate) inner: Arc<CoreQuantumGraph<f64>>,
}

#[pymethods]
impl PyQuantumGraph {
    /// Path graph ``P_{n_edges+1}``: ``n_edges+1`` vertices, equal-length edges.
    ///
    /// ``n_edges >= 1``, ``edge_length > 0``, ``n_grid >= 4``.
    #[staticmethod]
    #[pyo3(signature = (n_edges, edge_length = 1.0_f64, n_grid = 32_usize))]
    fn path(n_edges: usize, edge_length: f64, n_grid: usize) -> PyResult<Self> {
        catch_panic_py!({
            let g = CoreQuantumGraph::<f64>::path(n_edges, edge_length, n_grid)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner: Arc::new(g) })
        })
    }

    /// Star graph: central vertex 0, ``n_arms`` leaf vertices.
    ///
    /// ``n_arms >= 1``, ``edge_length > 0``, ``n_grid >= 4``.
    #[staticmethod]
    #[pyo3(signature = (n_arms, edge_length = 1.0_f64, n_grid = 32_usize))]
    fn star(n_arms: usize, edge_length: f64, n_grid: usize) -> PyResult<Self> {
        catch_panic_py!({
            let g = CoreQuantumGraph::<f64>::star(n_arms, edge_length, n_grid)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner: Arc::new(g) })
        })
    }

    /// Build from explicit edges.
    ///
    /// ``edges`` is a flat float64 array of triplets
    /// ``[vertex_a, vertex_b, edge_length, ...]``.  Length must be divisible
    /// by 3.  ``n_grid >= 4``.
    #[staticmethod]
    #[pyo3(signature = (edges, n_grid = 32_usize))]
    fn from_edges(edges: &Bound<'_, PyAny>, n_grid: usize) -> PyResult<Self> {
        catch_panic_py!({
            let raw = extract_f64_vec(edges).map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "edges must be a float64 array of triplets [va, vb, length, ...]",
                )
            })?;
            if raw.len() % 3 != 0 {
                return Err(new_pyerr(
                    "OutOfDomain",
                    "edges array length must be divisible by 3",
                ));
            }
            let ep: Vec<(usize, usize)> = raw
                .chunks_exact(3)
                .map(|c| (c[0] as usize, c[1] as usize))
                .collect();
            let lengths: Vec<f64> = raw.chunks_exact(3).map(|c| c[2]).collect();
            let g = CoreQuantumGraph::<f64>::new(ep, lengths, n_grid).map_err(|e| from_core(&e))?;
            Ok(Self { inner: Arc::new(g) })
        })
    }

    /// Number of vertices in the metric graph.
    #[getter]
    fn n_vertices(&self) -> usize {
        self.inner.n_vertices
    }

    /// Number of edges in the metric graph.
    #[getter]
    fn n_edges(&self) -> usize {
        self.inner.n_edges
    }

    /// Sum of all edge lengths (total arc-length).
    #[getter]
    fn total_arc_length(&self) -> f64 {
        self.inner.edge_lengths.iter().sum()
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumGraph(n_vertices={}, n_edges={}, total_arc_length={:.4})",
            self.inner.n_vertices,
            self.inner.n_edges,
            self.inner.edge_lengths.iter().sum::<f64>(),
        )
    }
}

// ===========================================================================
// QuantumGraphHeat — quantum graph heat Chernoff (M16)
// ===========================================================================

/// Quantum graph heat Chernoff approximation (M16, ADR-0078, math §29).
///
/// Solves ``∂_t u = ½ ∂²_x u`` per edge with Kirchhoff vertex conditions
/// (continuity + sum of inward derivatives = 0).
///
/// Algorithm: Phase 1 = edgewise ``ShiftChernoff1D`` (combined-domain for
/// uniform graphs); Phase 2 = per-vertex mean-averaging projection
/// ``Q_v = (1/d) 1 1^T``.
///
/// State layout: flat float64 array; concatenation of per-edge sampled values
/// in edge order.  Edge ``e`` occupies ``values[offset_e : offset_e + n_e]``
/// where ``n_e`` is the number of grid nodes on edge ``e`` (same for all edges
/// if constructed via ``path``/``star``).
///
/// Parameters
/// ----------
/// qgraph : `QuantumGraph`
///     Metric graph to evolve on.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`OutOfDomain`' if the graph has fewer than 1 edge.
#[pyclass(name = "QuantumGraphHeat")]
pub struct PyQuantumGraphHeat {
    graph: Arc<CoreQuantumGraph<f64>>,
    kernel: QuantumGraphHeatChernoff<f64>,
    current: QuantumGraphSignal<f64>,
    n_per_edge: usize,
    n_edges: usize,
}

#[pymethods]
impl PyQuantumGraphHeat {
    #[new]
    fn new(qgraph: &PyQuantumGraph) -> PyResult<Self> {
        catch_panic_py!({
            let g = Arc::clone(&qgraph.inner);
            let kernel = QuantumGraphHeatChernoff::new((*g).clone()).map_err(|e| from_core(&e))?;
            let n_per_edge = g.edge_grids[0].n;
            let n_edges = g.n_edges;
            let current = QuantumGraphSignal::zeroed_for_graph(&g);
            Ok(Self {
                graph: g,
                kernel,
                current,
                n_per_edge,
                n_edges,
            })
        })
    }

    /// Set the initial condition from a flat float64 array.
    ///
    /// Length must equal ``len(self)`` (sum of per-edge grid sizes).
    /// Returns ``None``; mutates state in place.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`GridMismatch`' if length mismatches.
    ///     kind='`NanInf`' if any value is non-finite.
    fn set_state(&mut self, u0: &Bound<'_, PyAny>) -> PyResult<()> {
        catch_panic_py!({
            let vals = extract_f64_vec(u0).map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("u0 must be a float64 array")
            })?;
            let expected = self.__len__();
            if vals.len() != expected {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("u0 length {}, expected {}", vals.len(), expected),
                ));
            }
            for &v in &vals {
                if !v.is_finite() {
                    return Err(new_pyerr("NanInf", "u0 contains NaN or Inf"));
                }
            }
            scatter_flat_to_signal(&mut self.current, &vals, self.n_per_edge);
            Ok(())
        })
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// GIL released during inner Rust compute (ADR-0031).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if t < 0, non-finite, or `n_steps` == 0.
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_t_nsteps(t, n_steps)?;
            let kernel = self.kernel.clone();
            let graph = Arc::clone(&self.graph);
            let n_per_edge = self.n_per_edge;
            // Collect flat values for GIL-free compute.
            let flat: Vec<f64> = gather_signal_to_flat(&self.current, n_per_edge);
            let result: Result<Vec<f64>, semiflow_core::SemiflowError> =
                py.detach(|| evolve_quantum(kernel, graph, flat, t, n_steps));
            let out = result.map_err(|e| from_core(&e))?;
            scatter_flat_to_signal(&mut self.current, &out, self.n_per_edge);
            Ok(())
        })
    }

    /// Return current signal as flat ``numpy.ndarray[float64]`` (copy).
    ///
    /// Values are concatenated per-edge in edge order.  Length = ``len(self)``.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            let flat = gather_signal_to_flat(&self.current, self.n_per_edge);
            Ok(flat.as_slice().to_pyarray(py))
        })
    }

    /// Return total number of grid nodes (sum of per-edge sizes).
    fn __len__(&self) -> usize {
        self.n_edges * self.n_per_edge
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumGraphHeat(n_edges={}, n_per_edge={})",
            self.n_edges, self.n_per_edge,
        )
    }
}

// ---------------------------------------------------------------------------
// QuantumGraphHeat helpers — GIL-free compute
// ---------------------------------------------------------------------------

fn gather_signal_to_flat(sig: &QuantumGraphSignal<f64>, n_per_edge: usize) -> Vec<f64> {
    let mut flat = Vec::with_capacity(sig.per_edge.len() * n_per_edge);
    for e in &sig.per_edge {
        flat.extend_from_slice(&e.values);
    }
    flat
}

fn scatter_flat_to_signal(sig: &mut QuantumGraphSignal<f64>, flat: &[f64], n_per_edge: usize) {
    for (e_idx, edge) in sig.per_edge.iter_mut().enumerate() {
        let base = e_idx * n_per_edge;
        edge.values.copy_from_slice(&flat[base..base + n_per_edge]);
    }
}

fn evolve_quantum(
    kernel: QuantumGraphHeatChernoff<f64>,
    graph: Arc<CoreQuantumGraph<f64>>,
    flat: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let n_per_edge = graph.edge_grids[0].n;
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let mut src = QuantumGraphSignal::zeroed_for_graph(&graph);
    scatter_flat_to_signal_owned(&mut src, &flat, n_per_edge);
    let out = sg.evolve(t, &src)?;
    Ok(gather_signal_to_flat(&out, n_per_edge))
}

fn scatter_flat_to_signal_owned(
    sig: &mut QuantumGraphSignal<f64>,
    flat: &[f64],
    n_per_edge: usize,
) {
    for (e_idx, edge) in sig.per_edge.iter_mut().enumerate() {
        let base = e_idx * n_per_edge;
        edge.values.copy_from_slice(&flat[base..base + n_per_edge]);
    }
}

// ===========================================================================
// Shared validation helper (pub so structured_matrix / structured_traj can use it)
// ===========================================================================

pub(crate) fn validate_t_nsteps(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

// ===========================================================================
// Registration
// ===========================================================================

/// Register Wave P6 pyclasses and free functions into the `semiflow` module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyQuantumGraph>()?;
    m.add_class::<PyQuantumGraphHeat>()?;
    m.add_class::<PyMatrixDiffusion1D>()?;
    m.add_class::<PyPointEval>()?;
    m.add_class::<PyGraphTraj>()?;
    m.add_class::<PyStrangGraph>()?;
    m.add_function(wrap_pyfunction!(sample_gridfn2d, m)?)?;
    Ok(())
}
