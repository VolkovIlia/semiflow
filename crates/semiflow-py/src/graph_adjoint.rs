//! `GraphAdjoint` — graph state-adjoint Python class (Issue #2, ADR-0115).
//!
//! Computes `S⋆(τ) · λ` — the transpose of the implemented degree-4 Taylor map
//! for the Magnus K=4 graph heat equation (math.md §42, Theorem 42.1).
//!
//! Supports `kernel="magnus_graph"` and `kernel="varcoef_magnus_graph"`.
//!
//! ## GIL policy (ADR-0031)
//!
//! Three-phase: validate (GIL held), compute (GIL released via `py.detach`),
//! return array (GIL held).  Python callbacks re-acquire GIL via `Python::attach`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use std::sync::Arc;

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow_core::{
    Graph, GraphSignal, Laplacian, LaplacianAtTime, MagnusGraphHeatChernoff, ScratchPool,
    VarCoefMagnusGraphHeatChernoff,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_extra::PyLaplacian,
    graph_py::{
        extract_f64_vec, extract_laplacian_arc, resolve_graph_from_any, validate_n_steps,
        validate_rho_bar, validate_signal_len, validate_t_final,
    },
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Configuration enum — stored on the Python object
// ---------------------------------------------------------------------------

/// Describes which variant of the graph adjoint to use.
pub(crate) enum KernelConfig {
    Magnus {
        rho_bar: f64,
        convergence_check: bool,
    },
    VarCoef {
        rho_bar: f64,
        a_sup_max: f64,
    },
}

// ---------------------------------------------------------------------------
// GraphAdjoint pyclass
// ---------------------------------------------------------------------------

/// Graph state-adjoint for the truncated Magnus K=4 map (Issue #2, ADR-0115).
///
/// Computes the backward costate sweep `λ_0 = S⋆_1 ⋯ S⋆_n · λ_n` where each
/// step applies `S⋆(τ) = Σ_{m=0..4} (Ω₄ᵀ)^m/m!` — the transpose of the
/// degree-4 Taylor map (math.md §42 Theorem 42.1, sign-flipped commutator).
///
/// Parameters
/// ----------
/// graph : Graph or `GraphPath`, optional
///     Fixed-topology graph.
/// laplacian : Laplacian, optional
///     Pre-assembled Laplacian for the topology.
/// `lap_at_t` : callable
///     ``t: float -> Graph | Laplacian | GraphPath``
/// `rho_bar` : float
///     Upper bound on ``ρ̄(L_G(t))``.
/// a : callable, optional
///     ``t: float -> list[float]`` — node weights.
///     Required for ``kernel="varcoef_magnus_graph"``.
/// kernel : str, optional
///     ``"magnus_graph"`` (default) or ``"varcoef_magnus_graph"``.
/// `convergence_check` : bool, optional
///     Enable convergence-radius guard (default ``True``).
#[pyclass(name = "GraphAdjoint")]
pub struct GraphAdjoint {
    config: KernelConfig,
    lap_callback: Py<PyAny>,
    a_callback: Option<Py<PyAny>>,
    graph: Arc<Graph<f64>>,
}

#[pymethods]
impl GraphAdjoint {
    #[new]
    #[pyo3(signature = (graph=None, laplacian=None, *, lap_at_t, rho_bar,
                        a=None, kernel="magnus_graph", convergence_check=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        graph: Option<&Bound<'_, PyAny>>,
        laplacian: Option<&PyLaplacian>,
        lap_at_t: Py<PyAny>,
        rho_bar: f64,
        a: Option<Py<PyAny>>,
        kernel: &str,
        convergence_check: bool,
    ) -> PyResult<Self> {
        catch_panic_py!({
            validate_rho_bar(rho_bar)?;
            let g = resolve_graph_from_any(graph, laplacian)?;
            let config = match kernel {
                "magnus_graph" => KernelConfig::Magnus {
                    rho_bar,
                    convergence_check,
                },
                "varcoef_magnus_graph" => {
                    if a.is_none() {
                        return Err(new_pyerr(
                            "OutOfDomain",
                            "varcoef_magnus_graph requires a= callback",
                        ));
                    }
                    KernelConfig::VarCoef {
                        rho_bar,
                        a_sup_max: rho_bar,
                    }
                }
                other => {
                    return Err(new_pyerr(
                        "OutOfDomain",
                        &format!(
                            "unknown kernel '{other}'; expected magnus_graph|varcoef_magnus_graph"
                        ),
                    ))
                }
            };
            Ok(GraphAdjoint {
                config,
                lap_callback: lap_at_t,
                a_callback: a,
                graph: g,
            })
        })
    }

    /// Backward costate sweep: `n_steps` adjoint steps of total time `t`.
    ///
    /// Terminal costate `lambda_n` → initial costate `λ_0`.
    /// The GIL is released during the Rust compute loop (ADR-0031).
    #[pyo3(signature = (lambda_n, t, n_steps = 100))]
    fn evolve_state_adjoint<'py>(
        &self,
        py: Python<'py>,
        lambda_n: &Bound<'_, PyAny>,
        t: f64,
        n_steps: u32,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            validate_t_final(t)?;
            validate_n_steps(n_steps)?;
            let lam = extract_f64_vec(lambda_n)?;
            validate_signal_len(&lam, self.graph.n_nodes())?;

            let g = Arc::clone(&self.graph);
            let n = n_steps as usize;
            #[allow(clippy::cast_precision_loss)]
            let tau = t / n as f64;

            let result: Result<Vec<f64>, semiflow_core::SemiflowError> = match &self.config {
                KernelConfig::Magnus {
                    rho_bar,
                    convergence_check,
                } => {
                    let cb = self.lap_callback.clone_ref(py);
                    let rho = *rho_bar;
                    let cc = *convergence_check;
                    py.detach(|| adjoint_magnus(g, cb, rho, cc, &lam, tau, n))
                }
                KernelConfig::VarCoef { rho_bar, a_sup_max } => {
                    let cb_lap = self.lap_callback.clone_ref(py);
                    let cb_a = self.a_callback.as_ref().unwrap().clone_ref(py);
                    let rho = *rho_bar;
                    let a_sup = *a_sup_max;
                    py.detach(|| adjoint_varcoef(g, cb_lap, cb_a, rho, a_sup, &lam, tau, n))
                }
            };
            let out = result.map_err(|e| from_core(&e))?;
            Ok(out.as_slice().to_pyarray(py))
        })
    }

    /// Number of graph nodes.
    fn n_nodes(&self) -> usize {
        self.graph.n_nodes()
    }
}

// ---------------------------------------------------------------------------
// Phase-2 helpers (no Python types; called inside py.detach)
// ---------------------------------------------------------------------------

fn adjoint_magnus(
    graph: Arc<Graph<f64>>,
    callback: Py<PyAny>,
    rho_bar: f64,
    convergence_check: bool,
    lam: &[f64],
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let g2 = Arc::clone(&graph);
    let lap: LaplacianAtTime<f64> = Box::new(move |t: f64| {
        Python::attach(|py| match callback.call1(py, (t,)) {
            Ok(v) => extract_laplacian_arc(v.bind(py), &g2),
            Err(_) => Arc::new(Laplacian::assemble_combinatorial(&g2)),
        })
    });
    let g = Arc::clone(&graph);
    let mc = MagnusGraphHeatChernoff::new(graph, lap, rho_bar, convergence_check)?;
    let src = GraphSignal::from_fn(Arc::clone(&g), |i| lam[i as usize]);
    let mut dst = GraphSignal::zeros(Arc::clone(&g));
    let mut scratch = ScratchPool::new();
    mc.evolve_state_adjoint_into(tau, n_steps, &src, &mut dst, &mut scratch)?;
    Ok(dst.values().to_vec())
}

fn adjoint_varcoef(
    graph: Arc<Graph<f64>>,
    cb_lap: Py<PyAny>,
    cb_a: Py<PyAny>,
    rho_bar: f64,
    a_sup_max: f64,
    lam: &[f64],
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let n = graph.n_nodes();
    let g2 = Arc::clone(&graph);
    let lap: LaplacianAtTime<f64> = Box::new(move |t: f64| {
        Python::attach(|py| match cb_lap.call1(py, (t,)) {
            Ok(v) => extract_laplacian_arc(v.bind(py), &g2),
            Err(_) => Arc::new(Laplacian::assemble_combinatorial(&g2)),
        })
    });
    let a_fn: semiflow_core::varcoef_magnus_graph::WeightAtTime<f64> = Box::new(move |t: f64| {
        Python::attach(|py| match cb_a.call1(py, (t,)) {
            Ok(v) => v
                .bind(py)
                .extract::<Vec<f64>>()
                .unwrap_or_else(|_| vec![1.0_f64; n]),
            Err(_) => vec![1.0_f64; n],
        })
    });
    let mc = VarCoefMagnusGraphHeatChernoff::new(n, lap, a_fn, rho_bar, a_sup_max)?;
    let src = GraphSignal::from_fn(Arc::clone(&graph), |i| lam[i as usize]);
    let mut dst = GraphSignal::zeros(Arc::clone(&graph));
    let mut scratch = ScratchPool::new();
    mc.evolve_state_adjoint_into(tau, n_steps, &src, &mut dst, &mut scratch)?;
    Ok(dst.values().to_vec())
}
