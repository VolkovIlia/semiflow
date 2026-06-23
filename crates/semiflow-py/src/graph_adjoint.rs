//! `GraphAdjoint` — graph state-adjoint Python class (Issue #2, ADR-0115).
//!
//! Computes `S⋆(τ) · λ` — the transpose of the implemented degree-4 Taylor map
//! for the Magnus K=4 graph heat equation (math.md §42, Theorem 42.1).
//!
//! Supports `kernel="magnus_graph"` and `kernel="varcoef_magnus_graph"`.
//!
//! ## Pre-sampled path (ADR-0180)
//!
//! `GraphAdjoint.from_presampled(...)` samples the `lap_at_t` callback ONCE
//! under GIL at construction (at the `2·n_steps` GL₄ abscissa times), stores
//! the weight sequence, then runs `evolve_state_adjoint` FULLY in `py.detach`
//! with NO per-step Python re-entry. This closes the PyO3-only deferral from
//! the v0.9.0-beta launch.
//!
//! ## GIL policy (ADR-0031)
//!
//! Closure path: validate (GIL), compute+callback (GIL released, per-step reattach).
//! Pre-sampled path: validate+sample (GIL held once), replay evolve (fully detached).

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use std::sync::Arc;

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    graph_adjoint_presampled::{
        fill_abscissa_times, PreSampledLaplacianSeq, PreSampledMagnusAdj, PreSampledVarCoefAdj,
    },
    Graph, GraphSignal, Laplacian, LaplacianAtTime, LaplacianKind, MagnusGraphHeatChernoff,
    ScratchPool, VarCoefMagnusGraphHeatChernoff,
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

            let result: Result<Vec<f64>, semiflow::SemiflowError> = match &self.config {
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
// GraphAdjointPresampled — pre-sampled path (ADR-0180)
// ---------------------------------------------------------------------------

/// Pre-sampled graph state-adjoint — no live Python callback during evolve.
///
/// Construct via `GraphAdjointPresampled.from_presampled(...)`. The
/// `lap_at_t` callback is called ONCE under GIL at construction on the
/// `2·n_steps` GL₄ abscissa times. The backward sweep in
/// `evolve_state_adjoint` runs fully in `py.detach` (no GIL reacquisition
/// per step) — see ADR-0031 §3, ADR-0180.
///
/// Parameters (from_presampled)
/// ----------------------------
/// graph : Graph or GraphPath
///     Graph topology.
/// lap_at_t : callable
///     ``t: float -> Graph | Laplacian | GraphPath``.  Called 2·n_steps times
///     at construction; never called during evolve.
/// rho_bar : float
///     Gershgorin upper bound on ``ρ̄(L_G(t))``.
/// n_steps : int
///     Number of adjoint time steps (must match `evolve_state_adjoint`).
/// t_horizon : float
///     Total time horizon (``τ = t_horizon / n_steps``).
/// a : callable, optional
///     ``t: float -> list[float]`` — node weights for VarCoef kernel.
/// kernel : str, optional
///     ``"magnus_graph"`` (default) or ``"varcoef_magnus_graph"``.
/// convergence_check : bool, optional
///     Enable Magnus radius guard (default ``True``).
#[pyclass(name = "GraphAdjointPresampled")]
pub struct GraphAdjointPresampled {
    variant: PresampledVariant,
    graph: Arc<Graph<f64>>,
    n_steps: usize,
    tau: f64,
}

enum PresampledVariant {
    Magnus(PreSampledMagnusAdj<f64>),
    VarCoef(PreSampledVarCoefAdj<f64>),
}

#[pymethods]
impl GraphAdjointPresampled {
    /// Construct by pre-sampling `lap_at_t` at `2·n_steps` GL₄ abscissa times.
    ///
    /// The callback is called under GIL. The returned object runs
    /// `evolve_state_adjoint` without any GIL reacquisition.
    #[classmethod]
    #[pyo3(signature = (graph, lap_at_t, rho_bar, n_steps, t_horizon,
                        a=None, kernel="magnus_graph", convergence_check=true))]
    #[allow(clippy::too_many_arguments)]
    fn from_presampled(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        graph: &Bound<'_, PyAny>,
        lap_at_t: Py<PyAny>,
        rho_bar: f64,
        n_steps: u32,
        t_horizon: f64,
        a: Option<Py<PyAny>>,
        kernel: &str,
        convergence_check: bool,
    ) -> PyResult<Self> {
        use crate::graph_py::{extract_laplacian_arc, resolve_graph_from_any, validate_n_steps,
                               validate_rho_bar, validate_t_final};
        validate_rho_bar(rho_bar)?;
        validate_n_steps(n_steps)?;
        validate_t_final(t_horizon)?;
        let g = resolve_graph_from_any(Some(graph), None)?;
        let ns = n_steps as usize;
        let tau = t_horizon / ns as f64;
        // Build base CSR pattern from the static (t=0) Laplacian to get row_ptr/col_idx.
        let base_lap = lap_at_t.call1(py, (0.0_f64,))
            .map_err(|e| e)?;
        let base_arc = extract_laplacian_arc(base_lap.bind(py), &g);
        let row_ptr = base_arc.row_ptr().to_vec();
        let col_idx = base_arc.col_idx().to_vec();
        let nnz = col_idx.len();
        // Sample at 2*n_steps GL4 abscissa times (GIL held throughout).
        let mut times = vec![0.0_f64; 2 * ns];
        fill_abscissa_times(t_horizon, ns, &mut times);
        let mut vals_seq = vec![0.0_f64; 2 * ns * nnz];
        for (idx, &t) in times.iter().enumerate() {
            let lap_obj = lap_at_t.call1(py, (t,))?;
            let lap_arc = extract_laplacian_arc(lap_obj.bind(py), &g);
            let block = &lap_arc.vals();
            vals_seq[idx * nnz..(idx + 1) * nnz].copy_from_slice(block);
        }
        let kind = LaplacianKind::Combinatorial;
        let seq = PreSampledLaplacianSeq::new(row_ptr, col_idx, vals_seq, ns, kind)
            .map_err(|e| crate::error::from_core(&e))?;
        match kernel {
            "magnus_graph" => {
                let ps = MagnusGraphHeatChernoff::<f64>::from_presampled(
                    seq, rho_bar, convergence_check,
                ).map_err(|e| crate::error::from_core(&e))?;
                Ok(GraphAdjointPresampled {
                    variant: PresampledVariant::Magnus(ps),
                    graph: g, n_steps: ns, tau,
                })
            }
            "varcoef_magnus_graph" => {
                let a_cb = a.ok_or_else(|| {
                    crate::error::new_pyerr("OutOfDomain",
                        "varcoef_magnus_graph requires a= callback")
                })?;
                let n = g.n_nodes();
                let mut a_seq = vec![0.0_f64; 2 * ns * n];
                for (idx, &t) in times.iter().enumerate() {
                    let a_vals: Vec<f64> = a_cb.call1(py, (t,))?
                        .bind(py).extract()?;
                    a_seq[idx * n..(idx + 1) * n].copy_from_slice(&a_vals);
                }
                let ps = VarCoefMagnusGraphHeatChernoff::<f64>::from_presampled(
                    seq, a_seq, rho_bar, rho_bar,
                ).map_err(|e| crate::error::from_core(&e))?;
                Ok(GraphAdjointPresampled {
                    variant: PresampledVariant::VarCoef(ps),
                    graph: g, n_steps: ns, tau,
                })
            }
            other => Err(crate::error::new_pyerr("OutOfDomain",
                &format!("unknown kernel '{other}'"))),
        }
    }

    /// Backward costate sweep: `lambda_n → lambda_0`.
    ///
    /// Runs fully in `py.detach` — no GIL reacquisition per step.
    /// `n_steps` must equal the value supplied at construction.
    #[pyo3(signature = (lambda_n, n_steps=None))]
    fn evolve_state_adjoint<'py>(
        &self,
        py: Python<'py>,
        lambda_n: &Bound<'_, PyAny>,
        n_steps: Option<u32>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use crate::graph_py::{extract_f64_vec, validate_signal_len};
        let lam = extract_f64_vec(lambda_n)?;
        validate_signal_len(&lam, self.graph.n_nodes())?;
        let ns = n_steps.map(|v| v as usize).unwrap_or(self.n_steps);
        if ns != self.n_steps {
            return Err(crate::error::new_pyerr(
                "OutOfDomain",
                &format!("n_steps={ns} != construction n_steps={}", self.n_steps),
            ));
        }
        let g = Arc::clone(&self.graph);
        let tau = self.tau;
        let out: Result<Vec<f64>, semiflow::SemiflowError> = match &self.variant {
            PresampledVariant::Magnus(ps) => {
                py.detach(|| presampled_magnus_evolve(ps, g, &lam, tau, ns))
            }
            PresampledVariant::VarCoef(ps) => {
                py.detach(|| presampled_varcoef_evolve(ps, g, &lam, tau, ns))
            }
        };
        let result = out.map_err(|e| crate::error::from_core(&e))?;
        Ok(result.as_slice().to_pyarray(py))
    }

    /// Number of graph nodes.
    fn n_nodes(&self) -> usize {
        self.graph.n_nodes()
    }

    /// Number of construction-time steps.
    fn n_steps(&self) -> usize {
        self.n_steps
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
) -> Result<Vec<f64>, semiflow::SemiflowError> {
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
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let n = graph.n_nodes();
    let g2 = Arc::clone(&graph);
    let lap: LaplacianAtTime<f64> = Box::new(move |t: f64| {
        Python::attach(|py| match cb_lap.call1(py, (t,)) {
            Ok(v) => extract_laplacian_arc(v.bind(py), &g2),
            Err(_) => Arc::new(Laplacian::assemble_combinatorial(&g2)),
        })
    });
    let a_fn: semiflow::varcoef_magnus_graph::WeightAtTime<f64> = Box::new(move |t: f64| {
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

// ---------------------------------------------------------------------------
// Pre-sampled evolve helpers (no Python types, called inside py.detach)
// ---------------------------------------------------------------------------

fn presampled_magnus_evolve(
    ps: &PreSampledMagnusAdj<f64>,
    graph: Arc<Graph<f64>>,
    lam: &[f64],
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let src = GraphSignal::from_fn(Arc::clone(&graph), |i| lam[i as usize]);
    let mut dst = GraphSignal::zeros(Arc::clone(&graph));
    let mut scratch = ScratchPool::new();
    ps.evolve_state_adjoint_into(tau, n_steps, &src, &mut dst, &mut scratch)?;
    Ok(dst.values().to_vec())
}

fn presampled_varcoef_evolve(
    ps: &PreSampledVarCoefAdj<f64>,
    graph: Arc<Graph<f64>>,
    lam: &[f64],
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let src = GraphSignal::from_fn(Arc::clone(&graph), |i| lam[i as usize]);
    let mut dst = GraphSignal::zeros(Arc::clone(&graph));
    let mut scratch = ScratchPool::new();
    ps.evolve_state_adjoint_into(tau, n_steps, &src, &mut dst, &mut scratch)?;
    Ok(dst.values().to_vec())
}
