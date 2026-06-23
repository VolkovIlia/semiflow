//! Graph PDE Python classes for `semiflow-py`.
//!
//! Exposes [`GraphPath`], [`GraphHeat`], and [`MagnusGraphHeat`] to Python,
//! mirroring the v0.10.0 `Heat1D` pyclass pattern (commit c3a6fe5) and the
//! v0.11.0 GIL-release pattern (ADR-0031, commit 07a4689).
//!
//! ## Python API
//!
//! ```python
//! import numpy as np
//! from semiflow import GraphPath, GraphHeat, MagnusGraphHeat, SemiflowError
//!
//! g = GraphPath(64)                             # P_64 path graph
//! gh = GraphHeat(g, rho_bar=4.0)               # order-1 heat
//! u0 = np.exp(-np.arange(64)**2 / 64.0)
//! u1 = gh.evolve(t_final=0.5, n_steps=50, f0=u0)
//!
//! def lap_at_t(t):
//!     return g                                  # time-independent shortcut
//! mghc = MagnusGraphHeat(g, lap_at_t, rho_bar=4.0)
//! u2 = mghc.evolve(t_final=0.5, n_steps=50, f0=u0)
//! ```
//!
//! ## GIL policy (ADR-0031)
//!
//! [`GraphHeat::evolve`] and [`MagnusGraphHeat::evolve`] use the three-phase
//! pattern:
//! 1. **Pre-flight (GIL held)**: validate, copy `f0` into owned `Vec<f64>`.
//! 2. **Compute (GIL released via `py.detach`)**: pure-Rust kernel.
//! 3. **Post-flight (GIL held)**: convert result to `PyArray1<f64>`.
//!
//! **[`MagnusGraphHeat`] callback note**: the Python `lap_at_t` callable is
//! invoked from within the GIL-released window via `Python::attach` to
//! re-acquire the GIL just for the callback (ADR-0059 R2 mitigation: at most
//! 4 GIL re-acquires per Magnus K=4 step).  Total GIL overhead is
//! `O(4 · n_steps)` callback entries; typical latency ~2-5 µs each.
//!
//! ## dtype support (Issue #3, ADR-0115)
//!
//! `GraphHeat` and `MagnusGraphHeat` accept an optional `dtype="f32"` kwarg.
//! The default `"f64"` path is unchanged.  fp16 is explicitly REJECTED.

#![allow(unsafe_code)]

use std::sync::Arc;

use numpy::ToPyArray;
use pyo3::prelude::*;

use semiflow::{ChernoffSemigroup, Graph, GraphHeatChernoff, GraphSignal, Laplacian};

use crate::dtype_dispatch::{cast_f64_to_f32, parse_dtype, Dtype};
use crate::graph_heat_f32::compute_graph_heat_f32;

use crate::{
    error::{from_core, new_pyerr},
    graph_extra::{PyGraph, PyLaplacian},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// GraphPath — P_n path graph
// ---------------------------------------------------------------------------

/// Path graph on `n_nodes` nodes with unit edge weights.
///
/// Represents the path graph `0 — 1 — 2 — … — (n-1)` with all edge
/// weights equal to 1.  Used as the topology for [`GraphHeat`] and
/// [`MagnusGraphHeat`].
///
/// Parameters
/// ----------
/// `n_nodes` : int
///     Number of nodes.  Must be ≥ 1.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='OutOfDomain'`` if ``n_nodes == 0``.
#[pyclass(name = "GraphPath")]
pub struct GraphPath {
    pub(crate) inner: Arc<Graph<f64>>,
}

#[pymethods]
impl GraphPath {
    /// Create a path graph on `n_nodes` nodes.
    #[new]
    fn new(n_nodes: u32) -> PyResult<Self> {
        catch_panic_py!({
            if n_nodes == 0 {
                return Err(new_pyerr("OutOfDomain", "n_nodes must be >= 1"));
            }
            let g = Graph::<f64>::path(n_nodes as usize);
            Ok(GraphPath { inner: Arc::new(g) })
        })
    }

    /// Number of nodes in the graph.
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    /// Number of directed edge entries (= 2 × undirected edges).
    fn n_directed_edges(&self) -> usize {
        self.inner.n_directed_edges()
    }
}

// ---------------------------------------------------------------------------
// GraphHeat — order-1 Chernoff heat on a path graph
// ---------------------------------------------------------------------------

/// Order-1 graph heat state: `∂ₜu = −L_G u` via `S(τ)f = f − τ L_G f`.
///
/// Accepts ``graph`` (`GraphPath` or `Graph`) or ``laplacian`` (`Laplacian`);
/// ``laplacian`` takes precedence.  ``rho_bar`` must be > 0.
/// GIL released during `evolve` (ADR-0031).
/// Optional ``dtype="f32"`` runs the kernel in single precision (Issue #3).
#[pyclass(name = "GraphHeat")]
pub struct GraphHeat {
    chernoff: GraphHeatChernoff<f64>,
    graph: Arc<Graph<f64>>,
    /// Parsed dtype choice; stored so `evolve` can dispatch f32 vs f64.
    dtype: Dtype,
}

#[pymethods]
impl GraphHeat {
    /// Create a graph heat state from a graph or a pre-assembled Laplacian.
    ///
    /// Accepts ``graph`` (`GraphPath` or Graph) or ``laplacian`` (Laplacian).
    /// ``laplacian`` takes precedence.  The graph is used as topology carrier.
    ///
    /// Parameters
    /// ----------
    /// dtype : str, optional
    ///     ``"f64"`` (default) or ``"f32"``.  When ``"f32"``, the Chernoff
    ///     kernel runs in single precision; ``evolve()`` returns ``float32``.
    #[new]
    #[pyo3(signature = (graph=None, laplacian=None, *, rho_bar, dtype="f64"))]
    fn new(
        graph: Option<&Bound<'_, PyAny>>,
        laplacian: Option<&Bound<'_, PyAny>>,
        rho_bar: f64,
        dtype: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            validate_rho_bar(rho_bar)?;
            let dt = parse_dtype(dtype)?;

            // Laplacian takes precedence over graph
            let mut gh = if let Some(lap_obj) = laplacian {
                build_graph_heat_from_laplacian_any(lap_obj)?
            } else if let Some(graph_obj) = graph {
                build_graph_heat_from_graph_any(graph_obj)?
            } else {
                return Err(new_pyerr(
                    "OutOfDomain",
                    "GraphHeat requires graph= or laplacian=",
                ));
            };
            gh.dtype = dt;
            Ok(gh)
        })
    }

    /// Evolve ``f0`` to ``t=t_final`` using ``n_steps`` Chernoff steps.
    /// GIL released during compute (ADR-0031).
    /// Returns ``float32`` array when ``dtype="f32"`` was set at construction.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        t_final: f64,
        n_steps: u32,
        f0: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        catch_panic_py!({
            validate_t_final(t_final)?;
            validate_n_steps(n_steps)?;
            let input = extract_f64_vec(f0)?;
            validate_signal_len(&input, self.graph.n_nodes())?;
            let n_steps_usize = n_steps as usize;

            match self.dtype {
                Dtype::F64 => {
                    let chernoff = self.chernoff.clone();
                    let graph = Arc::clone(&self.graph);
                    let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(|| {
                        compute_graph_heat(chernoff, graph, &input, t_final, n_steps_usize)
                    });
                    let arr = result.map_err(|e| from_core(&e))?.as_slice().to_pyarray(py);
                    Ok(arr.into_any())
                }
                Dtype::F32 => {
                    let graph = Arc::clone(&self.graph);
                    let result: Result<Vec<f64>, semiflow::SemiflowError> =
                        py.detach(|| compute_graph_heat_f32(graph, &input, t_final, n_steps_usize));
                    let out_f64 = result.map_err(|e| from_core(&e))?;
                    let out_f32: Vec<f32> = cast_f64_to_f32(&out_f64);
                    let arr = out_f32.as_slice().to_pyarray(py);
                    Ok(arr.into_any())
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Phase-2 helpers: pure-Rust kernels (no Python types, called in py.detach)
// ---------------------------------------------------------------------------

/// Order-1 graph heat evolution: `(S(t/n))^n f0`.
///
/// Wraps `f0` in a `GraphSignal`, builds [`ChernoffSemigroup`], calls `evolve`.
/// No GIL held during this function.
fn compute_graph_heat(
    chernoff: GraphHeatChernoff<f64>,
    graph: Arc<Graph<f64>>,
    input: &[f64],
    t_final: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(chernoff, n_steps)?;
    let f0 = GraphSignal::from_fn(graph, |i| input[i as usize]);
    let result = sg.evolve(t_final, &f0)?;
    Ok(result.values().to_vec())
}

// ---------------------------------------------------------------------------
// Topology resolver — shared by MagnusGraphHeat (K=4) and MagnusGraphHeat6 (K=6)
// ---------------------------------------------------------------------------

/// Resolve a fixed-topology `Arc<Graph<f64>>` from optional `graph` / `laplacian` args.
///
/// `graph` may be `Graph` (`PyGraph`) or legacy `GraphPath`.
/// At least one must be provided.  `graph` takes precedence over `laplacian`.
/// When only `laplacian` is given, a path graph of matching size is used as the
/// topology carrier (same contract as `resolve_graph` in magnus6.rs).
pub(crate) fn resolve_graph_from_any(
    graph: Option<&Bound<'_, PyAny>>,
    laplacian: Option<&PyLaplacian>,
) -> PyResult<Arc<Graph<f64>>> {
    if let Some(obj) = graph {
        // Accept PyGraph (Graph) or legacy GraphPath
        if let Ok(g) = obj.extract::<PyRef<'_, PyGraph>>() {
            return Ok(Arc::clone(&g.inner));
        }
        if let Ok(gp) = obj.extract::<PyRef<'_, GraphPath>>() {
            return Ok(Arc::clone(&gp.inner));
        }
        return Err(new_pyerr(
            "OutOfDomain",
            "graph must be a Graph or GraphPath",
        ));
    }
    if let Some(l) = laplacian {
        let n = l.inner.n_nodes();
        return Ok(Arc::new(Graph::<f64>::path(n.max(1))));
    }
    Err(new_pyerr(
        "OutOfDomain",
        "provide either graph= or laplacian=",
    ))
}

/// Resolve a fixed-topology `Arc<Graph<f64>>` from optional `PyGraph` / `laplacian` args.
///
/// Used by `MagnusGraphHeat6` (which already has `PyGraph` in its signature).
pub(crate) fn resolve_graph_topology(
    graph: Option<&PyGraph>,
    laplacian: Option<&PyLaplacian>,
) -> PyResult<Arc<Graph<f64>>> {
    match (graph, laplacian) {
        (Some(g), _) => Ok(Arc::clone(&g.inner)),
        (None, Some(l)) => {
            let n = l.inner.n_nodes();
            Ok(Arc::new(Graph::<f64>::path(n.max(1))))
        }
        (None, None) => Err(new_pyerr(
            "OutOfDomain",
            "provide either graph= or laplacian=",
        )),
    }
}

// ---------------------------------------------------------------------------
// Validation helpers (pub(crate) — reused by graph_extra.rs and magnus6.rs)
// ---------------------------------------------------------------------------

pub(crate) fn validate_rho_bar(rho_bar: f64) -> PyResult<()> {
    if !rho_bar.is_finite() || rho_bar <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "rho_bar must be finite and > 0"));
    }
    Ok(())
}

pub(crate) fn validate_t_final(t: f64) -> PyResult<()> {
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t_final must be finite and >= 0"));
    }
    Ok(())
}

pub(crate) fn validate_n_steps(n: u32) -> PyResult<()> {
    if n == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    Ok(())
}

pub(crate) fn validate_signal_len(v: &[f64], n_nodes: usize) -> PyResult<()> {
    if v.len() != n_nodes {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("f0.len()={} != graph.n_nodes()={}", v.len(), n_nodes),
        ));
    }
    Ok(())
}

pub(crate) fn extract_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "f0 must be a numpy.ndarray[float64] or a sequence of floats",
        )
    })
}

/// Extract edge triplets from either a 2-D ``(M, 3)`` float64 array (natural
/// user-facing layout) **or** a flat 1-D ``(3M,)`` float64 array (back-compat).
///
/// Returns a flat ``Vec<f64>`` of length ``3M`` ready to be ``chunks_exact(3)``-d.
///
/// Error semantics:
/// - 2-D array with wrong second dimension (not 3) → ``TypeError``
/// - 1-D array whose length is not divisible by 3 → caller checks separately
/// - Any other shape / dtype → ``TypeError``
pub(crate) fn extract_edges_flat(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};

    // --- Try 2-D path first: accept (M, 3) float64 ---
    if let Ok(arr2) = obj.extract::<Bound<'_, PyArray2<f64>>>() {
        let shape = arr2.shape();
        if shape[1] != 3 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "edges array must have shape (M, 3), got ({}, {})",
                shape[0], shape[1]
            )));
        }
        // as_array() works for any memory layout; iter() yields row-major order.
        return Ok(arr2.readonly().as_array().iter().copied().collect());
    }

    // --- Fall back to flat 1-D path ---
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "edges must be a float64 array: shape (M, 3) or flat 1-D of length 3*M",
        )
    })
}

// ---------------------------------------------------------------------------
// GraphHeat construction helpers (late-bound to avoid circular imports)
// ---------------------------------------------------------------------------

/// Build [`GraphHeat`] from a `PyAny` that is `GraphPath`, `Graph`, or `Laplacian`.
/// `dtype` is always set to the default `F64`; the caller may override it.
pub(crate) fn build_graph_heat_from_graph_any(obj: &Bound<'_, PyAny>) -> PyResult<GraphHeat> {
    use crate::graph_extra::{PyGraph, PyLaplacian};

    // Try legacy GraphPath
    if let Ok(gp) = obj.extract::<PyRef<'_, GraphPath>>() {
        let lap = Laplacian::assemble_combinatorial(&gp.inner);
        let chernoff = GraphHeatChernoff::from_owned(lap);
        return Ok(GraphHeat {
            chernoff,
            graph: Arc::clone(&gp.inner),
            dtype: Dtype::F64,
        });
    }
    // Try new Graph type
    if let Ok(g) = obj.extract::<PyRef<'_, PyGraph>>() {
        let lap = Laplacian::assemble_combinatorial(&g.inner);
        let chernoff = GraphHeatChernoff::from_owned(lap);
        return Ok(GraphHeat {
            chernoff,
            graph: Arc::clone(&g.inner),
            dtype: Dtype::F64,
        });
    }
    // Try Laplacian directly passed as graph= (unusual but allowed)
    if let Ok(l) = obj.extract::<PyRef<'_, PyLaplacian>>() {
        let n = l.inner.n_nodes();
        let g = Arc::new(Graph::<f64>::path(n.max(1)));
        let chernoff = GraphHeatChernoff::new(Arc::clone(&l.inner));
        return Ok(GraphHeat {
            chernoff,
            graph: g,
            dtype: Dtype::F64,
        });
    }
    Err(new_pyerr(
        "OutOfDomain",
        "graph must be a GraphPath, Graph, or Laplacian",
    ))
}

/// Build [`GraphHeat`] from a `PyAny` that is a `Laplacian` (or `Graph`/`GraphPath`).
pub(crate) fn build_graph_heat_from_laplacian_any(obj: &Bound<'_, PyAny>) -> PyResult<GraphHeat> {
    use crate::graph_extra::PyLaplacian;

    // Try Laplacian first
    if let Ok(l) = obj.extract::<PyRef<'_, PyLaplacian>>() {
        let n = l.inner.n_nodes();
        let g = Arc::new(Graph::<f64>::path(n.max(1)));
        let chernoff = GraphHeatChernoff::new(Arc::clone(&l.inner));
        return Ok(GraphHeat {
            chernoff,
            graph: g,
            dtype: Dtype::F64,
        });
    }
    // Fall through: accept Graph or GraphPath as laplacian= (assemble comb.).
    build_graph_heat_from_graph_any(obj)
}

/// Build a `LaplacianAtTime<f32>` from a Python callback (Issue #3 f32 path).
///
/// The callback returns `Graph|Laplacian|GraphPath` (f64); we extract the f64
/// Laplacian then rebuild a combinatorial f32 Laplacian with the same topology.
/// This preserves varying edge weights cast to f32 precision.
pub(crate) fn make_lap_at_t_f32(
    callback: Py<PyAny>,
    graph: Arc<Graph<f64>>,
) -> semiflow::LaplacianAtTime<f32> {
    use crate::graph_heat_f32::build_lap_f32_from_lap_f64;
    // LaplacianAtTime<f32> = Box<dyn Fn(f32) -> Arc<Laplacian<f32>> + ...>
    Box::new(move |t: f32| {
        Python::attach(|py| {
            let lap64: Arc<Laplacian<f64>> = match callback.call1(py, (f64::from(t),)) {
                Ok(v) => extract_laplacian_arc(v.bind(py), &graph),
                Err(_) => Arc::new(Laplacian::assemble_combinatorial(&graph)),
            };
            build_lap_f32_from_lap_f64(&lap64)
        })
    })
}

/// Extract `Arc<Laplacian<f64>>` from `PyAny` (`Laplacian`, `Graph`, or `GraphPath`).
/// Falls back to fixed-topology combinatorial Laplacian on failure.
pub(crate) fn extract_laplacian_arc(
    obj: &Bound<'_, PyAny>,
    fallback: &Arc<Graph<f64>>,
) -> Arc<Laplacian<f64>> {
    use crate::graph_extra::{PyGraph, PyLaplacian};

    if let Ok(l) = obj.extract::<PyRef<'_, PyLaplacian>>() {
        return Arc::clone(&l.inner);
    }
    if let Ok(g) = obj.extract::<PyRef<'_, PyGraph>>() {
        return Arc::new(Laplacian::assemble_combinatorial(&g.inner));
    }
    if let Ok(gp) = obj.extract::<PyRef<'_, GraphPath>>() {
        return Arc::new(Laplacian::assemble_combinatorial(&gp.inner));
    }
    Arc::new(Laplacian::assemble_combinatorial(fallback))
}
