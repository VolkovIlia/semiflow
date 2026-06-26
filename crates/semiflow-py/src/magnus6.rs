//! Magnus K=6 graph heat Python class: `MagnusGraphHeat6`.
//!
//! Mirrors the `MagnusGraphHeat` (K=4) pattern from `graph_py.rs` with the
//! sixth-order three-point GL₆ expansion (`MagnusGraphHeat6thChernoff`).
//!
//! ## Python API
//!
//! ```python
//! from semiflow import Graph, Laplacian, MagnusGraphHeat6
//! import numpy as np
//!
//! g = Graph.path(64)
//! def lap_at_t(t):
//!     return g                          # time-independent shortcut
//!
//! mgh6 = MagnusGraphHeat6(graph=g, lap_at_t=lap_at_t, rho_bar_max=2.5)
//! u0   = np.exp(-np.arange(64)**2 / 64.0)
//! u1   = mgh6.evolve(t_final=0.5, n_steps=20, f0=u0)
//! ```
//!
//! ## GIL policy (ADR-0031 / ADR-0059 R2)
//!
//! The `lap_at_t` callback is called via `Python::attach` inside the
//! GIL-released compute window — at most 3 GIL re-acquires per K=6 step
//! (one per GL₆ abscissa).  Total overhead is `O(3 · n_steps)` Python calls.
//!
//! ## f64 only (ADR-0056)
//!
//! `MagnusGraphHeat6thChernoff` does not implement `ChernoffFunction<f32>`.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    Graph, GraphSignal, Laplacian, LaplacianAtTime, MagnusGraphHeat6thChernoff, ScratchPool,
};

use crate::{
    error::from_core,
    graph_extra::{PyGraph, PyLaplacian},
    graph_py::{
        extract_f64_vec, extract_laplacian_arc, gather_nc_to_cn, resolve_graph_topology,
        scatter_cn_to_nc, validate_batched_shape, validate_n_steps, validate_rho_bar,
        validate_signal_len, validate_t_final,
    },
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// MagnusGraphHeat6 — K=6 graph heat with time-varying Laplacian
// ---------------------------------------------------------------------------

/// Magnus K=6 graph heat equation: ``∂ₜu = −L_G(t) u``.
///
/// Uses ``MagnusGraphHeat6thChernoff`` (sixth-order GL₆ three-point expansion)
/// for time-varying graph Laplacians.  For time-independent problems,
/// :class:`GraphHeat4th` is faster.
///
/// **f64 only** — ``MagnusGraphHeat6thChernoff`` does not support f32.
/// See ADR-0056.
///
/// **Callback overhead** (ADR-0059 R2): ``lap_at_t`` is called 3× per step
/// (one per GL₆ abscissa); 3 GIL re-acquires per step (~2–5 µs each).
///
/// Parameters
/// ----------
/// graph : Graph, optional
///     Fixed-topology graph.  Either ``graph`` or ``laplacian`` is required.
/// laplacian : Laplacian, optional
///     Pre-assembled Laplacian for the topology.
/// `lap_at_t` : callable
///     ``t: float -> Graph | Laplacian | GraphPath``  — return the Laplacian
///     (or graph) at absolute time ``t``.  The topology (``row_ptr``,
///     ``col_idx``) MUST match ``graph`` at every ``t``.
/// `rho_bar_max` : float
///     Upper bound on ``ρ̄(L_G(t))`` for all ``t``.  Must be > 0.
/// `convergence_check` : bool, optional
///     If ``True`` (default), each step checks ``rho_bar_max * tau < π/2``.
///     Raises ``SemiflowError(kind='ConvergenceFailed')`` on violation.
///
/// Raises
/// ------
/// SemiflowError
///     ``kind='OutOfDomain'`` if ``rho_bar_max <= 0`` or no graph provided.
///     ``kind='ConvergenceFailed'`` if convergence-radius condition violated.
#[pyclass(name = "MagnusGraphHeat6")]
pub struct MagnusGraphHeat6 {
    graph: Arc<Graph<f64>>,
    rho_bar_max: f64,
    convergence_check: bool,
    lap_callback: Py<PyAny>,
}

#[pymethods]
impl MagnusGraphHeat6 {
    /// Create a Magnus K=6 graph heat state.
    #[new]
    #[pyo3(signature = (graph=None, laplacian=None, *, lap_at_t, rho_bar_max, convergence_check=true))]
    fn new(
        graph: Option<&PyGraph>,
        laplacian: Option<&PyLaplacian>,
        lap_at_t: Py<PyAny>,
        rho_bar_max: f64,
        convergence_check: bool,
    ) -> PyResult<Self> {
        catch_panic_py!({
            validate_rho_bar(rho_bar_max)?;
            let g = resolve_graph_topology(graph, laplacian)?;
            Ok(MagnusGraphHeat6 {
                graph: g,
                rho_bar_max,
                convergence_check,
                lap_callback: lap_at_t,
            })
        })
    }

    /// Evolve ``f0`` from ``t=0`` to ``t=t_final`` using ``n_steps`` Magnus K=6
    /// steps.
    ///
    /// The GIL is released during the Rust compute loop.  ``lap_at_t`` is
    /// called from within the GIL-released window via ``Python::attach``
    /// (ADR-0031 / ADR-0059 R2 pattern).
    ///
    /// Parameters
    /// ----------
    /// `t_final` : float
    ///     Time horizon.  Must be finite and >= 0.
    /// `n_steps` : int
    ///     Number of Magnus K=6 steps.  Must be >= 1.
    /// f0 : numpy.ndarray[float64]
    ///     Initial condition; length ``n_nodes``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray[float64]
    ///     Result at ``t = t_final``; length ``n_nodes``.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``OutOfDomain`` / ``ConvergenceFailed`` on invalid parameters.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        t_final: f64,
        n_steps: u32,
        f0: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            // Phase 1: validate + extract (GIL held)
            validate_t_final(t_final)?;
            validate_n_steps(n_steps)?;
            let input = extract_f64_vec(f0)?;
            validate_signal_len(&input, self.graph.n_nodes())?;

            let graph = Arc::clone(&self.graph);
            let rho_bar_max = self.rho_bar_max;
            let convergence_check = self.convergence_check;
            let callback = self.lap_callback.clone_ref(py);
            let n_steps_usize = n_steps as usize;

            // Phase 2: compute (GIL released); callbacks via Python::attach.
            let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(|| {
                compute_magnus6(
                    &graph,
                    callback,
                    rho_bar_max,
                    convergence_check,
                    &input,
                    t_final,
                    n_steps_usize,
                )
            });
            let result = result.map_err(|e| from_core(&e))?;

            // Phase 3: build numpy array (GIL held)
            Ok(result.as_slice().to_pyarray(py))
        })
    }

    /// Evolve ``f0`` (``[N, C]``) for ``n_steps`` Magnus K=6 steps; single GIL release.
    ///
    /// GL₆ Laplacian samples are hoisted ONCE and shared across all ``C`` channels.
    #[allow(clippy::needless_pass_by_value)]
    fn evolve_batched<'py>(
        &self,
        py: Python<'py>,
        t_final: f64,
        n_steps: u32,
        f0: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        validate_t_final(t_final)?;
        validate_n_steps(n_steps)?;
        let [n_nodes, n_cols] = validate_batched_shape(f0.shape(), self.graph.n_nodes())?;
        let src = gather_nc_to_cn(&f0.as_array(), n_nodes, n_cols);
        let graph = Arc::clone(&self.graph);
        let cb = self.lap_callback.clone_ref(py);
        let rho = self.rho_bar_max;
        let cc = self.convergence_check;
        let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(|| {
            let mc = build_magnus6_for_batched(&graph, cb, rho, cc)?;
            let mut dst = vec![0.0f64; n_nodes * n_cols];
            semiflow::graph_batched::evolve_batched_magnus6(
                &mc,
                t_final,
                n_steps as usize,
                &src,
                &mut dst,
            )?;
            Ok(dst)
        });
        let dst = result.map_err(|e| from_core(&e))?;
        Ok(scatter_cn_to_nc(&dst, n_nodes, n_cols, py))
    }
}

// ---------------------------------------------------------------------------
// Phase-2 helper (no Python types; called in py.detach)
// ---------------------------------------------------------------------------

/// Magnus K=6 graph heat evolution with Python callback for `L_G(t)`.
///
/// Accepts `Graph`, `Laplacian`, or `GraphPath` from the Python callback;
/// assembles combinatorial Laplacian as needed.  Falls back to fixed topology
/// on Python exceptions (defensive; mimics the K=4 pattern).
fn compute_magnus6(
    graph: &Arc<Graph<f64>>,
    callback: Py<PyAny>,
    rho_bar_max: f64,
    convergence_check: bool,
    input: &[f64],
    t_final: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let graph2 = Arc::clone(graph);
    let graph3 = Arc::clone(graph);

    let lap_at_t: LaplacianAtTime<f64> = Box::new(move |t: f64| {
        // Re-acquire GIL to call the Python callable (ADR-0059 R2 pattern).
        Python::attach(|py| match callback.call1(py, (t,)) {
            Ok(py_any) => extract_laplacian_arc(py_any.bind(py), &graph2),
            Err(_) => Arc::new(Laplacian::assemble_combinatorial(&graph2)),
        })
    });

    let mgh6 = MagnusGraphHeat6thChernoff::new(
        Arc::clone(graph),
        lap_at_t,
        rho_bar_max,
        convergence_check,
    )?;

    #[allow(clippy::cast_precision_loss)]
    let tau = t_final / n_steps as f64;
    let mut state = GraphSignal::from_fn(graph3, |i| input[i as usize]);
    let mut scratch = ScratchPool::new();

    for step in 0..n_steps {
        #[allow(clippy::cast_precision_loss)]
        let t_start = step as f64 * tau;
        let mut next = state.clone();
        mgh6.apply_into_at(t_start, tau, &state, &mut next, &mut scratch)?;
        state = next;
    }

    Ok(state.values().to_vec())
}

/// Build a [`MagnusGraphHeat6thChernoff`] inside `py.detach` for the batched path.
///
/// `lap_at_t` re-acquires the GIL via `Python::attach`.
/// `evolve_batched_magnus6` hoists it ONCE (three GL₆ abscissae total).
fn build_magnus6_for_batched(
    graph: &Arc<Graph<f64>>,
    callback: Py<PyAny>,
    rho_bar_max: f64,
    convergence_check: bool,
) -> Result<MagnusGraphHeat6thChernoff<f64>, semiflow::SemiflowError> {
    let g2 = Arc::clone(graph);
    let lap_at_t: LaplacianAtTime<f64> = Box::new(move |t: f64| {
        Python::attach(|py| match callback.call1(py, (t,)) {
            Ok(v) => extract_laplacian_arc(v.bind(py), &g2),
            Err(_) => Arc::new(Laplacian::assemble_combinatorial(&g2)),
        })
    });
    MagnusGraphHeat6thChernoff::new(Arc::clone(graph), lap_at_t, rho_bar_max, convergence_check)
}
