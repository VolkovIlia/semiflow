//! `MagnusGraphHeat` — Magnus K=4 graph heat with time-varying weights.
//!
//! Split from `graph_py.rs` for suckless file-size compliance.
//! The GIL callback pattern and f32 dispatch are the most complex parts;
//! keeping them isolated makes both modules easier to audit.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3 wrapper patterns.
#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use numpy::ToPyArray;
use pyo3::prelude::*;

use semiflow_core::{Graph, GraphSignal, Laplacian, LaplacianAtTime, MagnusGraphHeatChernoff};

use crate::dtype_dispatch::{cast_f64_to_f32, parse_dtype, Dtype};
use crate::graph_heat_f32::compute_magnus_graph_f32;

use crate::{
    error::from_core,
    graph_extra::PyLaplacian,
    graph_py::{
        extract_f64_vec, extract_laplacian_arc, make_lap_at_t_f32, resolve_graph_from_any,
        validate_n_steps, validate_rho_bar, validate_signal_len, validate_t_final,
    },
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// MagnusGraphHeat — Magnus K=4 heat with time-varying weights  (D1 + D4)
// ---------------------------------------------------------------------------

/// Magnus K=4 graph heat equation: ``∂ₜu = −L_G(t) u``.
///
/// Uses ``MagnusGraphHeatChernoff`` (fourth-order Magnus expansion with
/// two-point Gauss-Legendre quadrature) for time-varying graph Laplacians.
/// For time-independent problems, :class:`GraphHeat` is faster (no Python callback).
///
/// **D1 — Signature change**: The constructor now mirrors :class:`MagnusGraphHeat6`.
/// Accept ``graph`` (``Graph``/``GraphPath``) or ``laplacian`` (``Laplacian``) plus
/// the ``lap_at_t`` callback that may return a Laplacian with **varying edge weights**
/// (only topology must be fixed).
///
/// **D4 — Naming**: ``rho_bar_max`` (keyword-only) for all time-varying Magnus kernels;
/// ``rho_bar`` (keyword-only) is reserved for static-bound kernels.
///
/// **Callback overhead (ADR-0059 R2)**: ``lap_at_t`` is called 2× per
/// Magnus step (GL₄ quadrature nodes ``c₁``, ``c₂``), plus work for the
/// commutator term — at most 4 GIL re-acquires per step (~2–5 µs each).
///
/// Parameters
/// ----------
/// graph : Graph or `GraphPath`, optional
///     Fixed-topology graph.  Either ``graph`` or ``laplacian`` is required.
/// laplacian : Laplacian, optional
///     Pre-assembled Laplacian for the topology.
/// `lap_at_t` : callable
///     ``t: float -> Graph | Laplacian | GraphPath``  — return the Laplacian
///     (or graph) at absolute time ``t``.  The topology (``row_ptr``,
///     ``col_idx``) MUST match ``graph`` at every ``t``.  Edge weights may vary.
/// `rho_bar_max` : float
///     Upper bound on ``ρ̄(L_G(t))`` for all ``t``.  Must be > 0.
/// `convergence_check` : bool, optional
///     If ``True`` (default), each step checks ``rho_bar_max * tau < π/2``.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='OutOfDomain'`` if ``rho_bar_max <= 0`` or no graph provided.
///     ``kind='ConvergenceFailed'`` if convergence-radius condition violated.
#[pyclass(name = "MagnusGraphHeat")]
pub struct MagnusGraphHeat {
    graph: Arc<Graph<f64>>,
    rho_bar_max: f64,
    convergence_check: bool,
    lap_callback: Py<PyAny>,
    dtype: Dtype,
}

#[pymethods]
impl MagnusGraphHeat {
    /// Create a Magnus K=4 graph heat state.
    #[new]
    #[pyo3(signature = (graph=None, laplacian=None, *, lap_at_t, rho_bar_max, convergence_check=true, dtype="f64"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        graph: Option<&Bound<'_, PyAny>>,
        laplacian: Option<&PyLaplacian>,
        lap_at_t: Py<PyAny>,
        rho_bar_max: f64,
        convergence_check: bool,
        dtype: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            validate_rho_bar(rho_bar_max)?;
            let dt = parse_dtype(dtype)?;
            let g = resolve_graph_from_any(graph, laplacian)?;
            Ok(MagnusGraphHeat {
                graph: g,
                rho_bar_max,
                convergence_check,
                lap_callback: lap_at_t,
                dtype: dt,
            })
        })
    }

    /// Evolve ``f0`` from ``t=0`` to ``t=t_final`` using ``n_steps`` Magnus K=4 steps.
    ///
    /// The GIL is released during the Rust compute loop.  The ``lap_at_t``
    /// callable is called from within the GIL-released window via
    /// ``Python::attach`` (ADR-0031 / ADR-0059 R2 pattern).
    ///
    /// Returns ``float32`` array when ``dtype="f32"`` was set at construction.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        t_final: f64,
        n_steps: u32,
        f0: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        catch_panic_py!({
            // Phase 1: validate + extract (GIL held)
            validate_t_final(t_final)?;
            validate_n_steps(n_steps)?;
            let input = extract_f64_vec(f0)?;
            validate_signal_len(&input, self.graph.n_nodes())?;
            let graph = Arc::clone(&self.graph);
            let rho_bar_max = self.rho_bar_max;
            let convergence_check = self.convergence_check;
            let n_steps_usize = n_steps as usize;
            match self.dtype {
                Dtype::F64 => evolve_f64(
                    py,
                    graph,
                    self.lap_callback.clone_ref(py),
                    rho_bar_max,
                    convergence_check,
                    &input,
                    t_final,
                    n_steps_usize,
                ),
                Dtype::F32 => evolve_f32(
                    py,
                    graph,
                    self.lap_callback.clone_ref(py),
                    rho_bar_max,
                    convergence_check,
                    &input,
                    t_final,
                    n_steps_usize,
                ),
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Evolve dispatch helpers (one per dtype, called from evolve())
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn evolve_f64<'py>(
    py: Python<'py>,
    graph: Arc<Graph<f64>>,
    callback: Py<PyAny>,
    rho_bar_max: f64,
    convergence_check: bool,
    input: &[f64],
    t_final: f64,
    n_steps: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let result: Result<Vec<f64>, semiflow_core::SemiflowError> = py.detach(|| {
        compute_magnus_graph(
            &graph,
            callback,
            rho_bar_max,
            convergence_check,
            input,
            t_final,
            n_steps,
        )
    });
    let arr = result.map_err(|e| from_core(&e))?.as_slice().to_pyarray(py);
    Ok(arr.into_any())
}

fn evolve_f32<'py>(
    py: Python<'py>,
    graph: Arc<Graph<f64>>,
    callback: Py<PyAny>,
    rho_bar_max: f64,
    convergence_check: bool,
    input: &[f64],
    t_final: f64,
    n_steps: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let graph2 = Arc::clone(&graph);
    let lap_f32 = make_lap_at_t_f32(callback, graph2);
    let result: Result<Vec<f64>, semiflow_core::SemiflowError> = py.detach(|| {
        compute_magnus_graph_f32(
            graph,
            lap_f32,
            rho_bar_max,
            convergence_check,
            input,
            t_final,
            n_steps,
        )
    });
    let out_f32: Vec<f32> = cast_f64_to_f32(&result.map_err(|e| from_core(&e))?);
    Ok(out_f32.as_slice().to_pyarray(py).into_any())
}

// ---------------------------------------------------------------------------
// Phase-2 helpers: pure-Rust kernel (no Python types, called in py.detach)
// ---------------------------------------------------------------------------

/// Magnus K=4 graph heat evolution (time-varying `L_G(t)`).
///
/// Builds a `LaplacianAtTime` closure that calls the Python `lap_at_t`
/// callable via `Python::attach` (one GIL re-acquire per sample).
/// Runs a manual loop calling `apply_into_at` with correct absolute time
/// `t_start` for each step.
///
/// `convergence_check` gates the per-step `rho_bar_max * tau < π/2` guard.
fn compute_magnus_graph(
    graph: &Arc<Graph<f64>>,
    callback: Py<PyAny>,
    rho_bar_max: f64,
    convergence_check: bool,
    input: &[f64],
    t_final: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    use semiflow_core::ScratchPool;

    let graph2 = Arc::clone(graph);
    let lap_at_t: LaplacianAtTime<f64> = Box::new(move |t: f64| {
        // Re-acquire GIL to call Python lap_at_t(t).
        Python::attach(|py| {
            match callback.call1(py, (t,)) {
                Ok(py_any) => extract_laplacian_arc(py_any.bind(py), &graph2),
                // Python exception — fall back to fixed topology.
                Err(_) => Arc::new(Laplacian::assemble_combinatorial(&graph2)),
            }
        })
    });

    let mghc =
        MagnusGraphHeatChernoff::new(Arc::clone(graph), lap_at_t, rho_bar_max, convergence_check)?;

    #[allow(clippy::cast_precision_loss)]
    let tau = t_final / n_steps as f64;
    let mut state = GraphSignal::from_fn(Arc::clone(graph), |i| input[i as usize]);
    let mut scratch = ScratchPool::new();

    for step in 0..n_steps {
        #[allow(clippy::cast_precision_loss)]
        let t_start = step as f64 * tau;
        let mut next = state.clone();
        mghc.apply_into_at(t_start, tau, &state, &mut next, &mut scratch)?;
        state = next;
    }

    Ok(state.values().to_vec())
}
