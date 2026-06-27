//! Extended graph Python classes: `Graph`, `Laplacian`, `GraphHeat4th`,
//! `VarCoefGraphHeat`.
//!
//! ## Python API
//!
//! ```python
//! import numpy as np
//! from semiflow import Graph, Laplacian, GraphHeat4th, VarCoefGraphHeat
//!
//! g   = Graph.path(64)
//! g2  = Graph.cycle(64)
//! L   = Laplacian.combinatorial(g)
//! gh4 = GraphHeat4th(laplacian=L, rho_bar=2.0)
//! u0  = np.exp(-np.arange(64)**2 / 64.0)
//! u1  = gh4.evolve(t_final=0.5, n_steps=50, f0=u0)
//! ```
//!
//! ## GIL policy (ADR-0031)
//!
//! `GraphHeat4th::evolve` and `VarCoefGraphHeat::evolve` use the three-phase
//! pattern: validate (GIL held) → compute (`py.detach`) → build array (GIL held).

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_pass_by_value,
    clippy::type_complexity
)]

use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use semiflow::{Graph, Laplacian, LaplacianKind};

// Re-export heat kernels from their own module.
pub(crate) use crate::graph_extra_heat::{GraphHeat4th, VarCoefGraphHeat};
use crate::{
    error::{from_core, new_pyerr},
    graph_py::extract_edges_flat,
    laplacian_introspect::{
        laplacian_col_idx, laplacian_row_ptr, laplacian_to_dense, laplacian_vals,
    },
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Graph — factory + introspection
// ---------------------------------------------------------------------------

/// Sparse weighted graph in symmetric CSR layout.
///
/// **Factories**: :meth:`path`, :meth:`cycle`, :meth:`from_edges`,
/// :meth:`erdos_renyi`.
///
/// See ADR-0048 for invariants (I1–I7).
#[pyclass(name = "Graph")]
pub struct PyGraph {
    pub(crate) inner: Arc<Graph<f64>>,
}

#[pymethods]
impl PyGraph {
    /// Create path graph ``0 — 1 — … — (n-1)`` with unit edge weights.
    /// ``n_nodes`` must be >= 1; raises ``SemiflowError(OutOfDomain)`` otherwise.
    #[staticmethod]
    fn path(n_nodes: u32) -> PyResult<Self> {
        catch_panic_py!({
            if n_nodes == 0 {
                return Err(new_pyerr("OutOfDomain", "n_nodes must be >= 1"));
            }
            Ok(PyGraph {
                inner: Arc::new(Graph::<f64>::path(n_nodes as usize)),
            })
        })
    }

    /// Create cycle graph (path + wrap-around edge) with unit edge weights.
    /// ``n_nodes`` must be >= 3; raises ``SemiflowError(OutOfDomain)`` otherwise.
    #[staticmethod]
    fn cycle(n_nodes: u32) -> PyResult<Self> {
        catch_panic_py!({
            if n_nodes < 3 {
                return Err(new_pyerr("OutOfDomain", "cycle requires n_nodes >= 3"));
            }
            Ok(PyGraph {
                inner: Arc::new(Graph::<f64>::cycle(n_nodes as usize)),
            })
        })
    }

    /// Create graph from an edge array of triplets ``[u, v, w]``.
    ///
    /// ``edges`` may be:
    ///
    /// * a 2-D ``numpy.ndarray`` of shape ``(M, 3)`` and dtype ``float64``
    ///   (natural layout, recommended), **or**
    /// * a flat 1-D ``numpy.ndarray`` of length ``3*M`` and dtype ``float64``
    ///   (back-compatible layout).
    ///
    /// ``n_nodes`` must be > 0.  Raises ``SemiflowError(OutOfDomain)`` on
    /// self-loop, duplicate edge, non-positive weight, or out-of-range index.
    #[staticmethod]
    fn from_edges(n_nodes: u32, edges: &Bound<'_, PyAny>) -> PyResult<Self> {
        catch_panic_py!({
            if n_nodes == 0 {
                return Err(new_pyerr("OutOfDomain", "n_nodes must be >= 1"));
            }
            let raw = extract_edges_flat(edges)?;
            if raw.len() % 3 != 0 {
                return Err(new_pyerr(
                    "OutOfDomain",
                    "edges flat array length must be divisible by 3",
                ));
            }
            let iter = raw
                .chunks_exact(3)
                .map(|row| (row[0] as u32, row[1] as u32, row[2]));
            let g = Graph::<f64>::from_edges(n_nodes as usize, iter).map_err(|e| from_core(&e))?;
            Ok(PyGraph { inner: Arc::new(g) })
        })
    }

    /// Create Erdős–Rényi ``G(n, p)`` random graph (unit weights, splitmix64).
    ///
    /// ``n_nodes >= 1``, ``p`` in ``[0, 1]``, ``seed`` default ``42``.
    #[staticmethod]
    #[pyo3(signature = (n_nodes, p, seed = 42))]
    fn erdos_renyi(n_nodes: u32, p: f64, seed: u64) -> PyResult<Self> {
        catch_panic_py!({
            if n_nodes == 0 {
                return Err(new_pyerr("OutOfDomain", "n_nodes must be >= 1"));
            }
            let g = Graph::<f64>::erdos_renyi(n_nodes as usize, p, seed);
            Ok(PyGraph { inner: Arc::new(g) })
        })
    }

    /// Number of nodes in the graph.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    /// Number of directed edge entries (= 2 × undirected edges).
    #[getter]
    fn n_directed_edges(&self) -> usize {
        self.inner.n_directed_edges()
    }

    /// Degree of ``node`` (0-based).  Raises ``OutOfDomain`` if ``node >= n_nodes``.
    fn degree(&self, node: u32) -> PyResult<usize> {
        if node as usize >= self.inner.n_nodes() {
            return Err(new_pyerr("OutOfDomain", "node index >= n_nodes"));
        }
        let rp = self.inner.row_ptr();
        Ok(rp[node as usize + 1] - rp[node as usize])
    }
}

// ---------------------------------------------------------------------------
// Laplacian — factory + introspection
// ---------------------------------------------------------------------------

/// Sparse Laplacian in CSR layout assembled from a :class:`Graph`.
///
/// **Factories**: :meth:`combinatorial`, :meth:`normalized`.
///
/// See ADR-0048 invariants L1–L4.
#[pyclass(name = "Laplacian")]
pub struct PyLaplacian {
    pub(crate) inner: Arc<Laplacian<f64>>,
}

#[pymethods]
impl PyLaplacian {
    /// Assemble combinatorial Laplacian ``L = D − W``.  ``is_combinatorial == True``.
    #[staticmethod]
    fn combinatorial(graph: &PyGraph) -> Self {
        PyLaplacian {
            inner: Arc::new(Laplacian::assemble_combinatorial(&graph.inner)),
        }
    }

    /// Assemble normalized Laplacian ``L_sym = I − D^{−½} W D^{−½}``.
    /// ``is_normalized == True``.
    #[staticmethod]
    fn normalized(graph: &PyGraph) -> Self {
        PyLaplacian {
            inner: Arc::new(Laplacian::assemble_normalized(&graph.inner)),
        }
    }

    /// Build a :class:`Laplacian` directly from symmetric CSR arrays (issue #13 convenience).
    ///
    /// This is a lower-level constructor than :meth:`combinatorial` or :meth:`normalized`.
    /// The matrix is stored as ``LaplacianKind::GeneralSymmetric``; Gershgorin bound
    /// is computed automatically.
    ///
    /// Parameters
    /// ----------
    /// indptr : ndarray[int64, shape (n+1,)]
    ///     CSR row-pointer array.
    /// indices : ndarray[int32, shape (nnz,)]
    ///     CSR column-index array (``0 ≤ col < n``).
    /// data : ndarray[float64, shape (nnz,)]
    ///     Non-zero values.
    /// n : int
    ///     Number of nodes.
    ///
    /// Raises ``SemiflowError`` on shape or range violations.
    #[staticmethod]
    fn from_csr(
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f64>,
        n: usize,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let row_ptr: Vec<usize> = indptr
                .as_slice()
                .map_err(|_| new_pyerr("GridMismatch", "indptr must be contiguous"))?
                .iter()
                .map(|&v| v as usize)
                .collect();
            let col_idx: Vec<u32> = indices
                .as_slice()
                .map_err(|_| new_pyerr("GridMismatch", "indices must be contiguous"))?
                .iter()
                .map(|&v| v as u32)
                .collect();
            let vals: Vec<f64> = data
                .as_slice()
                .map_err(|_| new_pyerr("GridMismatch", "data must be contiguous"))?
                .to_vec();
            let inner = Laplacian::from_csr_parts(
                n,
                row_ptr,
                col_idx,
                vals,
                LaplacianKind::GeneralSymmetric,
            )
            .map_err(|e| from_core(&e))?;
            Ok(PyLaplacian { inner: Arc::new(inner) })
        })
    }

    /// Number of nodes.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    /// ``True`` iff this is the combinatorial Laplacian ``L = D − W``.
    #[getter]
    fn is_combinatorial(&self) -> bool {
        self.inner.kind() == LaplacianKind::Combinatorial
    }

    /// ``True`` iff this is the symmetric normalized Laplacian.
    #[getter]
    fn is_normalized(&self) -> bool {
        self.inner.kind() == LaplacianKind::SymNormalized
    }

    /// Gershgorin spectral-radius upper bound ``ρ̄ ≥ ρ(L_G)`` (cached).
    #[getter]
    fn spectral_bound(&self) -> f64 {
        self.inner.spectral_radius_bound()
    }

    // --- Issue #5 introspection (ADR-0115) — helpers in laplacian_introspect.rs ---

    /// Dense ``n × n`` float64 matrix reconstructed from CSR (row-major copy).
    ///
    /// Memory: O(n²).  Raises ``SemiflowError(OutOfDomain)`` if ``n*n``
    /// overflows ``usize``.
    fn to_dense<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        catch_panic_py!({ laplacian_to_dense(py, &self.inner) })
    }

    /// CSR row-pointer array (copy), length ``n_nodes + 1``, dtype ``int64``.
    fn row_ptr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
        catch_panic_py!({ laplacian_row_ptr(py, &self.inner) })
    }

    /// CSR column-index array (copy), length ``n_directed_edges``, dtype ``int64``.
    fn col_idx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
        catch_panic_py!({ laplacian_col_idx(py, &self.inner) })
    }

    /// CSR values array (copy), length ``n_directed_edges``, dtype ``float64``.
    fn vals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        catch_panic_py!({ laplacian_vals(py, &self.inner) })
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Resolve ``(laplacian, graph)`` optional pair into
/// ``(Arc<Laplacian>, Arc<Graph>)``.
///
/// At least one of ``laplacian`` or ``graph`` must be provided.
/// When only ``laplacian`` is given, a path graph of the same size is used
/// as the topology carrier for ``GraphSignal``.
pub(crate) fn resolve_lap_and_graph(
    laplacian: Option<&PyLaplacian>,
    graph: Option<&PyGraph>,
) -> PyResult<(Arc<Laplacian<f64>>, Arc<Graph<f64>>)> {
    match (laplacian, graph) {
        (Some(l), _) => {
            // Topology carrier: use a path graph of matching size.
            let n = l.inner.n_nodes();
            let g = Arc::new(Graph::<f64>::path(n.max(1)));
            Ok((Arc::clone(&l.inner), g))
        }
        (None, Some(g)) => {
            let lap = Laplacian::assemble_combinatorial(&g.inner);
            Ok((Arc::new(lap), Arc::clone(&g.inner)))
        }
        (None, None) => Err(new_pyerr(
            "OutOfDomain",
            "provide either laplacian= or graph=",
        )),
    }
}
