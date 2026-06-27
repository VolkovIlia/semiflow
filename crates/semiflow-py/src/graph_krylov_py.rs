//! Python binding for `GraphKrylovChernoff` (A1) and `graph_expmv_frechet` (A2).
//!
//! ## Python API
//!
//! ```python
//! lap  = semiflow.Laplacian.combinatorial(graph)
//! gk   = semiflow.GraphKrylov(lap, path="chebyshev", tol=1e-10)
//! out  = gk.evolve_batched(t, features_NC)           # [N, C]
//! grad = semiflow.graph_expmv_frechet(gk, u0_NC, dj_NC, t=0.3, params=[(0,1),(1,2)])
//! ```
//!
//! GIL policy: three-phase ADR-0031 (validate → `py.detach` → scatter);
//! ONE `py.detach` per call; `[N,C]↔[C,N]` boundary copy fused into gather/scatter.

#![allow(unsafe_code, clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
use semiflow::{EdgeWeightSensitivity, Graph, GraphKrylovChernoff, KrylovPath, ScratchPool};

use crate::{
    error::{from_core, new_pyerr},
    graph_extra::PyLaplacian,
    graph_py::{gather_nc_to_cn, scatter_cn_to_nc, validate_batched_shape, validate_t_final},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// GraphKrylov — Python class wrapping `GraphKrylovChernoff`
// ---------------------------------------------------------------------------

/// Depth-independent graph action `e^{-t L_G}·v` via Krylov methods (A1, §54, ADR-0185).
///
/// Accepts a symmetric ``Laplacian`` (combinatorial or normalised).
/// ``evolve_batched`` applies it to a ``[N, C]`` feature matrix in ONE FFI hop
/// with ONE GIL release (ADR-0031, ADR-0184 D1).
///
/// Parameters
/// ----------
/// laplacian : Laplacian
///     Pre-assembled symmetric Laplacian (:meth:`Laplacian.combinatorial`).
/// path : str
///     ``"chebyshev"`` (default, O(1) work vectors, Bessel-tail degree) or
///     ``"lanczos"`` (m-dim Krylov basis + Padé, adaptive depth).
/// tol : float
///     Target accuracy ε.  Default ``1e-10``.
/// `m_max` : int
///     Max Krylov dimension for ``path="lanczos"`` (ignored for Chebyshev).
///     Capped at 18 internally.  Default ``18``.
#[pyclass(name = "GraphKrylov")]
pub struct GraphKrylov {
    pub(crate) gk: GraphKrylovChernoff<f64>,
    /// Dummy no-edge graph with `n_nodes` nodes; domain carrier for
    /// `GraphSignal` buffer allocation inside `evolve_batched` (ADR-0184 §1a).
    pub(crate) dummy_graph: Arc<Graph<f64>>,
}

#[pymethods]
impl GraphKrylov {
    /// Construct from a symmetric ``Laplacian``.
    ///
    /// Raises ``SemiflowError(OutOfDomain)`` if ``tol ≤ 0``, ``tol`` is not
    /// finite, or ``path`` is neither ``"chebyshev"`` nor ``"lanczos"``.
    #[new]
    #[pyo3(signature = (laplacian, *, path = "chebyshev", tol = 1e-10_f64, m_max = 18_u32))]
    fn new(laplacian: &PyLaplacian, path: &str, tol: f64, m_max: u32) -> PyResult<Self> {
        catch_panic_py!({
            let kpath = parse_krylov_path(path, m_max)?;
            let lap_arc = Arc::clone(&laplacian.inner);
            let n = lap_arc.n_nodes();
            let gk =
                GraphKrylovChernoff::new(lap_arc, kpath, tol).map_err(|e| from_core(&e))?;
            let dummy_graph = build_dummy_graph(n);
            Ok(GraphKrylov { gk, dummy_graph })
        })
    }

    /// Apply `e^{-t L_G}` to ``features_nc`` (``[N, C]``); single GIL release.
    ///
    /// Implemented as one Chernoff step (``n_steps = 1``) at time ``t`` — the
    /// depth-independent single Krylov solve (§54.4).
    /// Returns ``[N, C]`` float64 array.
    ///
    /// Raises ``SemiflowError(OutOfDomain)`` on layout mismatch or ``t < 0``.
    fn evolve_batched<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        features_nc: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        validate_t_final(t)?;
        let n = self.gk.n_nodes();
        let [n_nodes, n_cols] = validate_batched_shape(features_nc.shape(), n)?;
        let src_cn = gather_nc_to_cn(&features_nc.as_array(), n_nodes, n_cols);
        let gk = self.gk.clone();
        let dummy = Arc::clone(&self.dummy_graph);
        let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(|| {
            let mut dst_cn = vec![0.0_f64; n_nodes * n_cols];
            // n_steps = 1: single Krylov solve at t (§54.4 depth-flat action).
            semiflow::graph_batched::evolve_batched(&gk, &dummy, t, 1, &src_cn, &mut dst_cn)?;
            Ok(dst_cn)
        });
        let dst_cn = result.map_err(|e| from_core(&e))?;
        Ok(scatter_cn_to_nc(&dst_cn, n_nodes, n_cols, py))
    }

    /// Number of graph nodes (= rows/columns of the Laplacian).
    fn n_nodes(&self) -> usize {
        self.gk.n_nodes()
    }
}

// ---------------------------------------------------------------------------
// graph_expmv_frechet — module-level function (A2, §54.5, ADR-0185)
// ---------------------------------------------------------------------------

/// VJP gradient `∂J/∂w` via the Fréchet–Duhamel integral (A2, §54.5, ADR-0185).
///
/// Computes `∂J/∂w_k = t ∫₀¹ ⟨e^{−(1−s)tL}dj_c, (∂A/∂w_k) e^{−stL}u0_c⟩ ds`
/// summed over channels using 8-point Gauss-Legendre quadrature.  Exact for all
/// graph topologies including non-commuting edge directions.
///
/// Parameters
/// ----------
/// gk : `GraphKrylov`
///     Krylov solver owning the Laplacian ``L``.
/// u0 : ndarray[float64, shape (N, C)]
///     Batched initial conditions.
/// dj : ndarray[float64, shape (N, C)]
///     Batched loss-gradient vectors ``∂J/∂u_final``.
/// t : float
///     Evolution time ``t > 0``.
/// params : list[tuple[int, int]]
///     Explicit undirected edge pairs ``(i, j)`` (0-indexed).  Each pair
///     identifies one edge-weight parameter ``w_k``.
///     ``"all_edges"`` is **not** accepted here — pass an explicit list.
///
/// Returns
/// -------
/// np.ndarray[float64]
///     Summed ``∂J/∂w`` over all channels, length ``len(params)``.
#[pyfunction]
#[pyo3(name = "graph_expmv_frechet", signature = (gk, u0, dj, *, t, params))]
#[allow(clippy::too_many_arguments)]
pub fn graph_expmv_frechet_py<'py>(
    py: Python<'py>,
    gk: &GraphKrylov,
    u0: PyReadonlyArray2<'py, f64>,
    dj: PyReadonlyArray2<'py, f64>,
    t: f64,
    params: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    catch_panic_py!({
        if !t.is_finite() || t <= 0.0 {
            return Err(new_pyerr(
                "OutOfDomain",
                "graph_expmv_frechet: t must be finite and positive",
            ));
        }
        let n_nodes = gk.gk.n_nodes();
        let [n, c] = validate_batched_shape(u0.shape(), n_nodes)?;
        let [n2, c2] = validate_batched_shape(dj.shape(), n_nodes)?;
        if n2 != n || c2 != c {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("graph_expmv_frechet: u0 shape [{n},{c}] != dj shape [{n2},{c2}]"),
            ));
        }
        let edge_pairs = extract_explicit_edge_pairs(params, n_nodes)?;
        let n_params = edge_pairs.len();
        let u0_cn = gather_nc_to_cn(&u0.as_array(), n, c);
        let dj_cn = gather_nc_to_cn(&dj.as_array(), n, c);
        let inner_gk = gk.gk.clone();
        let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
            let sens = EdgeWeightSensitivity { params: edge_pairs, n_nodes };
            let mut grad = vec![0.0_f64; n_params];
            let mut scratch = ScratchPool::new();
            semiflow::graph_expmv_frechet(&inner_gk, &u0_cn, &dj_cn, c, t, &sens, &mut grad, &mut scratch)?;
            Ok(grad)
        });
        let out = result.map_err(|e| from_core(&e))?;
        Ok(out.as_slice().to_pyarray(py))
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map ``"chebyshev"`` / ``"lanczos"`` string to [`KrylovPath`].
fn parse_krylov_path(path: &str, m_max: u32) -> PyResult<KrylovPath> {
    match path {
        "chebyshev" => Ok(KrylovPath::Chebyshev),
        "lanczos" => Ok(KrylovPath::Lanczos { m_max: m_max as usize }),
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!("path must be 'chebyshev' or 'lanczos', got '{other}'"),
        )),
    }
}

/// No-edge graph with ``n`` nodes — carries domain for `GraphSignal` buffer alloc.
fn build_dummy_graph(n: usize) -> Arc<Graph<f64>> {
    Arc::new(
        Graph::<f64>::from_edges(n, core::iter::empty())
            .expect("zero-edge graph construction is infallible"),
    )
}

/// Extract explicit ``(i, j)`` pairs from Python; reject ``"all_edges"`` strings.
fn extract_explicit_edge_pairs(
    params: &Bound<'_, PyAny>,
    n_nodes: usize,
) -> PyResult<Vec<(usize, usize)>> {
    if params.extract::<String>().is_ok() {
        return Err(new_pyerr(
            "OutOfDomain",
            "graph_expmv_frechet: 'all_edges' not supported — pass explicit list[(i,j)] pairs",
        ));
    }
    let pairs = params.extract::<Vec<(usize, usize)>>().map_err(|_| {
        new_pyerr("OutOfDomain", "graph_expmv_frechet: params must be list[(int, int)]")
    })?;
    for &(i, j) in &pairs {
        if i >= n_nodes || j >= n_nodes {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("graph_expmv_frechet: edge ({i},{j}) out of range for n_nodes={n_nodes}"),
            ));
        }
    }
    Ok(pairs)
}

/// Register [`GraphKrylov`] class and [`graph_expmv_frechet_py`] function.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<GraphKrylov>()?;
    m.add_function(wrap_pyfunction!(graph_expmv_frechet_py, m)?)?;
    Ok(())
}
