//! v2.4 graph Python bindings: `GraphHeat6` (ADR-0062) and
//! `VarCoefMagnusGraph` (ADR-0063).
//!
//! ## Python API
//!
//! ```python
//! import numpy as np
//! from semiflow import Graph, Laplacian, GraphHeat6, VarCoefMagnusGraph
//!
//! # Order-6 static graph heat
//! g   = Graph.path(64)
//! gh6 = GraphHeat6(graph=g, rho_bar=4.0)
//! u0  = np.exp(-(np.arange(64) - 32.0)**2 / 32.0)
//! u1  = gh6.evolve(t_final=0.5, n_steps=20, f0=u0)
//!
//! # Variable-coefficient × time-dependent Magnus K=4
//! n = 32
//! def lap_at_t(t):
//!     return Graph.path(n)               # time-independent topology shortcut
//! def a_at_t(t):
//!     return np.ones(n) * (1.0 + 0.5 * np.sin(np.pi * t))
//!
//! vcm = VarCoefMagnusGraph(n_nodes=n, lap_at_t=lap_at_t, a_at_t=a_at_t,
//!                          rho_bar_max=3.0, a_sup_max=1.5**0.5)
//! u   = vcm.evolve(t_final=0.5, n_steps=20, f0=u0[:n])
//! ```
//!
//! ## GIL policy (ADR-0031 / ADR-0059 R2)
//!
//! Both classes release the GIL during compute via `py.detach`. The Python
//! callbacks (`lap_at_t`, `a_at_t`) re-acquire the GIL via `Python::attach`
//! at each GL₂ quadrature point.
//!
//! ## f64 only
//!
//! Per the Python-bindings precision policy (ADR-0056 precedent applied to
//! all graph kernels), both classes are f64-only. The Rust core types remain
//! generic over `F: SemiflowFloat`.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use std::sync::Arc;

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;

use semiflow::{
    varcoef_magnus_graph::{
        compute_rho_bar as core_compute_rho_bar, VarCoefMagnusGraphHeatChernoff, WeightAtTime,
    },
    ChernoffSemigroup, Graph, GraphHeat6thChernoff, GraphSignal, Laplacian, LaplacianAtTime,
    ScratchPool,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_extra::{resolve_lap_and_graph, PyGraph, PyLaplacian},
    graph_py::{
        extract_f64_vec, extract_laplacian_arc, validate_n_steps, validate_rho_bar,
        validate_signal_len, validate_t_final,
    },
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// GraphHeat6 — order-6 static spatial graph heat (ADR-0062, math.md §19)
// ---------------------------------------------------------------------------

/// Order-6 graph heat equation: ``∂ₜu = −L_G u`` via degree-6 operator-Taylor.
///
/// ``S₆(τ) f = Σ_{k=0}^{6} (−τ L_G)^k / k! · f``.  6 `SpMV` per step;
/// 2 ping-pong scratch buffers; zero heap alloc in steady state.
///
/// For time-dependent ``L_G(t)``, use :class:`MagnusGraphHeat` (K=4) or
/// :class:`MagnusGraphHeat6` (K=6 time-dep). For static ``L_G`` with order-4
/// accuracy, use :class:`GraphHeat4th`. **f64 only** in Python; the Rust
/// core type is generic over ``F: SemiflowFloat``.
///
/// Parameters
/// ----------
/// laplacian : Laplacian, optional
///     Pre-assembled Laplacian.  Either ``laplacian`` or ``graph`` is required.
/// graph : Graph, optional
///     Topology; combinatorial Laplacian assembled internally if provided.
/// `rho_bar` : float
///     Gershgorin spectral-radius bound.  Must be > 0 and finite.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='OutOfDomain'`` if ``rho_bar <= 0`` or neither ``laplacian``
///     nor ``graph`` is provided.
#[pyclass(name = "GraphHeat6")]
pub struct GraphHeat6 {
    laplacian: Arc<Laplacian<f64>>,
    graph: Arc<Graph<f64>>,
    n_nodes: usize,
}

#[pymethods]
impl GraphHeat6 {
    /// Create a `GraphHeat6` state from a Laplacian or Graph.
    #[new]
    #[pyo3(signature = (laplacian=None, graph=None, *, rho_bar))]
    fn new(
        laplacian: Option<&PyLaplacian>,
        graph: Option<&PyGraph>,
        rho_bar: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            validate_rho_bar(rho_bar)?;
            let (lap_arc, g_arc) = resolve_lap_and_graph(laplacian, graph)?;
            let n = lap_arc.n_nodes();
            Ok(GraphHeat6 {
                laplacian: lap_arc,
                graph: g_arc,
                n_nodes: n,
            })
        })
    }

    /// Evolve ``f0`` to ``t=t_final`` using ``n_steps`` Chernoff steps.
    /// GIL released during compute (ADR-0031).
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        t_final: f64,
        n_steps: u32,
        f0: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            validate_t_final(t_final)?;
            validate_n_steps(n_steps)?;
            let input = extract_f64_vec(f0)?;
            validate_signal_len(&input, self.n_nodes)?;
            let lap = Arc::clone(&self.laplacian);
            let graph = Arc::clone(&self.graph);
            let n_st = n_steps as usize;
            let result: Result<Vec<f64>, semiflow::SemiflowError> =
                py.detach(|| compute_heat6(lap, graph, &input, t_final, n_st));
            Ok(result.map_err(|e| from_core(&e))?.as_slice().to_pyarray(py))
        })
    }

    /// Number of nodes the kernel acts on.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }
}

/// Order-6 graph heat evolution.  No GIL held.
fn compute_heat6(
    lap: Arc<Laplacian<f64>>,
    graph: Arc<Graph<f64>>,
    input: &[f64],
    t_final: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let chernoff = GraphHeat6thChernoff::new(lap);
    let sg = ChernoffSemigroup::new(chernoff, n_steps)?;
    let f0 = GraphSignal::from_fn(graph, |i| input[i as usize]);
    let result = sg.evolve(t_final, &f0)?;
    Ok(result.values().to_vec())
}

// ---------------------------------------------------------------------------
// VarCoefMagnusGraph — variable-a × time-dep Magnus K=4 (ADR-0063, math.md §20)
// ---------------------------------------------------------------------------

/// Variable-coefficient × time-dependent graph Magnus K=4:
/// ``∂_t u = −L_a(t) u``, ``L_a(t) = sqrt(a(t)) ⊙ L_G(t) ⊙ sqrt(a(t))``.
///
/// Order-4 Magnus expansion with GL₂ quadrature; samples BOTH ``a(t)`` and
/// ``L_G(t)`` at each abscissa per step.  Convergence radius:
/// ``rho_bar_max * a_sup_max² * τ < π/2``; violations raise
/// :class:`SemiflowError` with ``kind='ConvergenceFailed'``.
///
/// **f64 only** in Python; the Rust core type is generic over ``F``.
///
/// **Callback overhead** (ADR-0059 R2): both ``lap_at_t`` and ``a_at_t`` are
/// called twice per step (one per GL₂ abscissa); 4 GIL re-acquires per step.
///
/// Parameters
/// ----------
/// `n_nodes` : int
///     Number of nodes (must match output of ``lap_at_t(t).n_nodes()`` and
///     ``len(a_at_t(t))`` at every sampled ``t``).
/// `lap_at_t` : callable
///     ``t: float -> Graph | Laplacian | GraphPath``.  Topology
///     (``row_ptr``, ``col_idx``) MUST be invariant in ``t``.
/// `a_at_t` : callable
///     ``t: float -> numpy.ndarray[float64]`` of length ``n_nodes``.  All
///     entries strictly positive and finite.
/// `rho_bar_max` : float
///     Upper bound on ``ρ̄(L_G(t))`` over the integration interval.
/// `a_sup_max` : float
///     Upper bound on ``sqrt(max_i a_i(t))`` over the integration interval.
/// `convergence_check` : bool, optional
///     If ``True`` (default), each step validates the convergence-radius
///     inequality.
#[pyclass(name = "VarCoefMagnusGraph")]
pub struct VarCoefMagnusGraph {
    n_nodes: usize,
    graph: Arc<Graph<f64>>,
    rho_bar_max: f64,
    a_sup_max: f64,
    convergence_check: bool,
    lap_callback: Py<PyAny>,
    a_callback: Py<PyAny>,
}

#[pymethods]
impl VarCoefMagnusGraph {
    /// Create a `VarCoefMagnusGraph` state.
    #[new]
    #[pyo3(signature = (n_nodes, *, lap_at_t, a_at_t, rho_bar_max, a_sup_max, convergence_check=true))]
    fn new(
        n_nodes: u32,
        lap_at_t: Py<PyAny>,
        a_at_t: Py<PyAny>,
        rho_bar_max: f64,
        a_sup_max: f64,
        convergence_check: bool,
    ) -> PyResult<Self> {
        catch_panic_py!({
            if n_nodes == 0 {
                return Err(new_pyerr("OutOfDomain", "n_nodes must be >= 1"));
            }
            validate_rho_bar(rho_bar_max)?;
            if !a_sup_max.is_finite() || a_sup_max <= 0.0 {
                return Err(new_pyerr("OutOfDomain", "a_sup_max must be > 0 and finite"));
            }
            let g = Arc::new(Graph::<f64>::path(n_nodes as usize));
            Ok(VarCoefMagnusGraph {
                n_nodes: n_nodes as usize,
                graph: g,
                rho_bar_max,
                a_sup_max,
                convergence_check,
                lap_callback: lap_at_t,
                a_callback: a_at_t,
            })
        })
    }

    /// Evolve ``f0`` from ``t=t_start`` to ``t=t_start + t_final`` using
    /// ``n_steps`` Magnus K=4 steps.  GIL released during compute.
    ///
    /// ``t_start`` defaults to 0.  Use a non-zero ``t_start`` for stitched
    /// trajectories where this call continues from an earlier step.
    #[pyo3(signature = (t_final, n_steps, f0, *, t_start=0.0))]
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        t_final: f64,
        n_steps: u32,
        f0: &Bound<'_, PyAny>,
        t_start: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            validate_t_final(t_final)?;
            validate_n_steps(n_steps)?;
            if !t_start.is_finite() {
                return Err(new_pyerr("OutOfDomain", "t_start must be finite"));
            }
            let input = extract_f64_vec(f0)?;
            validate_signal_len(&input, self.n_nodes)?;
            let graph = Arc::clone(&self.graph);
            let rho_bar_max = self.rho_bar_max;
            let a_sup_max = self.a_sup_max;
            let convergence_check = self.convergence_check;
            let lap_cb = self.lap_callback.clone_ref(py);
            let a_cb = self.a_callback.clone_ref(py);
            let n_st = n_steps as usize;
            let n = self.n_nodes;
            let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(|| {
                compute_varcoef_magnus(
                    &graph,
                    lap_cb,
                    a_cb,
                    n,
                    rho_bar_max,
                    a_sup_max,
                    convergence_check,
                    &input,
                    t_final,
                    t_start,
                    n_st,
                )
            });
            Ok(result.map_err(|e| from_core(&e))?.as_slice().to_pyarray(py))
        })
    }

    /// Number of nodes the kernel acts on.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Estimate ``(rho_bar_max, a_sup_max)`` over ``[t0, t1]`` via
    /// ``n_samples`` equispaced sample points.
    ///
    /// Wraps :rust:`varcoef_magnus_graph::compute_rho_bar`. Returns a tuple
    /// suitable for direct use in the :class:`VarCoefMagnusGraph` constructor.
    #[staticmethod]
    #[pyo3(signature = (lap_at_t, a_at_t, t0, t1, n_nodes, *, n_samples=32))]
    fn compute_rho_bar(
        py: Python<'_>,
        lap_at_t: Py<PyAny>,
        a_at_t: Py<PyAny>,
        t0: f64,
        t1: f64,
        n_nodes: u32,
        n_samples: u32,
    ) -> PyResult<(f64, f64)> {
        catch_panic_py!({
            if n_nodes == 0 {
                return Err(new_pyerr("OutOfDomain", "n_nodes must be >= 1"));
            }
            if n_samples == 0 {
                return Err(new_pyerr("OutOfDomain", "n_samples must be >= 1"));
            }
            let graph = Arc::new(Graph::<f64>::path(n_nodes as usize));
            let lap_fn = make_lap_at_t_py(lap_at_t.clone_ref(py), Arc::clone(&graph));
            let a_fn = make_a_at_t_py(a_at_t.clone_ref(py), n_nodes as usize);
            let (rho, a_sup) =
                py.detach(|| core_compute_rho_bar(&lap_fn, &a_fn, (t0, t1), n_samples as usize));
            Ok((rho, a_sup))
        })
    }
}

// ---------------------------------------------------------------------------
// Phase-2 helpers (no Python types in body; called inside py.detach)
// ---------------------------------------------------------------------------

/// Wrap a Python ``t -> Graph | Laplacian | GraphPath`` callable into a
/// ``LaplacianAtTime<f64>`` closure suitable for the Rust core.
fn make_lap_at_t_py(callback: Py<PyAny>, graph: Arc<Graph<f64>>) -> LaplacianAtTime<f64> {
    Box::new(move |t: f64| {
        Python::attach(|py| match callback.call1(py, (t,)) {
            Ok(v) => extract_laplacian_arc(v.bind(py), &graph),
            Err(_) => Arc::new(Laplacian::assemble_combinatorial(&graph)),
        })
    })
}

/// Wrap a Python ``t -> numpy.ndarray[float64]`` callable into a
/// ``WeightAtTime<f64>`` closure.  Falls back to all-ones on Python errors
/// (defensive; matches `make_lap_at_t_py` pattern).
fn make_a_at_t_py(callback: Py<PyAny>, n_nodes: usize) -> WeightAtTime<f64> {
    Box::new(move |t: f64| {
        Python::attach(|py| match callback.call1(py, (t,)) {
            Ok(v) => match extract_f64_vec(v.bind(py)) {
                Ok(a) if a.len() == n_nodes => a,
                _ => vec![1.0_f64; n_nodes],
            },
            Err(_) => vec![1.0_f64; n_nodes],
        })
    })
}

/// `VarCoef` Magnus K=4 evolution.  No GIL held outside the wrapped callbacks.
#[allow(clippy::too_many_arguments)]
fn compute_varcoef_magnus(
    graph: &Arc<Graph<f64>>,
    lap_cb: Py<PyAny>,
    a_cb: Py<PyAny>,
    n_nodes: usize,
    rho_bar_max: f64,
    a_sup_max: f64,
    convergence_check: bool,
    input: &[f64],
    t_final: f64,
    t_start: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let lap_fn = make_lap_at_t_py(lap_cb, Arc::clone(graph));
    let a_fn = make_a_at_t_py(a_cb, n_nodes);
    let mc =
        VarCoefMagnusGraphHeatChernoff::<f64>::new(n_nodes, lap_fn, a_fn, rho_bar_max, a_sup_max)?
            .with_radius_check(convergence_check);

    let graph2 = Arc::clone(graph);
    #[allow(clippy::cast_precision_loss)]
    let tau = t_final / n_steps as f64;
    let mut state = GraphSignal::from_fn(graph2, |i| input[i as usize]);
    let mut scratch = ScratchPool::<f64>::new();
    for step in 0..n_steps {
        #[allow(clippy::cast_precision_loss)]
        let t = t_start + step as f64 * tau;
        let mut next = state.clone();
        mc.apply_into_at(t, tau, &state, &mut next, &mut scratch)?;
        state = next;
    }
    Ok(state.values().to_vec())
}
