//! `GraphAdjoint` / `GraphAdjointPresampled` — graph state-adjoint (Issue #2, ADR-0115, ADR-0180).
//!
//! Computes `S⋆(τ) · λ` for Magnus K=4; batched variant added in Issue #10.
//! Kernels: `magnus_graph`, `varcoef_magnus_graph`.
//!
//! GIL: closure path releases per-step; pre-sampled path releases fully (ADR-0031).

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]

use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
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
        extract_f64_vec, extract_laplacian_arc, gather_nc_to_cn, resolve_graph_from_any,
        scatter_cn_to_nc, validate_batched_shape, validate_n_steps, validate_rho_bar,
        validate_signal_len, validate_t_final,
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

/// Graph state-adjoint for the Magnus K=4 map (Issue #2, ADR-0115).
///
/// Backward costate sweep `λ_0 = S⋆_1 ⋯ S⋆_n · λ_n`.
/// See Python `.pyi` stub for full API docs.
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

// GraphAdjointPresampled — pre-sampled path (ADR-0180)

/// Pre-sampled graph state-adjoint — no live Python callback during evolve (ADR-0180).
///
/// Construct via `from_presampled`. Callbacks called ONCE at construction;
/// backward sweep runs fully in `py.detach`. See `.pyi` stub for full API.
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
        use crate::graph_py::{
            resolve_graph_from_any, validate_n_steps, validate_rho_bar, validate_t_final,
        };
        validate_rho_bar(rho_bar)?;
        validate_n_steps(n_steps)?;
        validate_t_final(t_horizon)?;
        let g = resolve_graph_from_any(Some(graph), None)?;
        let ns = n_steps as usize;
        #[allow(clippy::cast_precision_loss)]
        let tau = t_horizon / ns as f64;
        let (seq, times) = build_presampled_seq(py, &g, &lap_at_t, ns, t_horizon)?;
        build_adjoint_variant(
            py,
            g,
            seq,
            &times,
            ns,
            tau,
            rho_bar,
            a,
            kernel,
            convergence_check,
        )
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
        let ns = n_steps.map_or(self.n_steps, |v| v as usize);
        if ns != self.n_steps {
            return Err(crate::error::new_pyerr(
                "OutOfDomain",
                &format!("n_steps={ns} != construction n_steps={}", self.n_steps),
            ));
        }
        let g = Arc::clone(&self.graph);
        let tau = self.tau;
        let out: Result<Vec<f64>, semiflow::SemiflowError> = match &self.variant {
            PresampledVariant::Magnus(ps) => py.detach(|| {
                let src = GraphSignal::from_fn(Arc::clone(&g), |i| lam[i as usize]);
                let mut dst = GraphSignal::zeros(Arc::clone(&g));
                ps.evolve_state_adjoint_into(tau, ns, &src, &mut dst, &mut ScratchPool::new())?;
                Ok(dst.values().to_vec())
            }),
            PresampledVariant::VarCoef(ps) => py.detach(|| {
                let src = GraphSignal::from_fn(Arc::clone(&g), |i| lam[i as usize]);
                let mut dst = GraphSignal::zeros(Arc::clone(&g));
                ps.evolve_state_adjoint_into(tau, ns, &src, &mut dst, &mut ScratchPool::new())?;
                Ok(dst.values().to_vec())
            }),
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

    /// Batched backward costate sweep: ``lambda_cols`` (``[N, C]``) → ``[N, C]``.
    ///
    /// Each channel ``c`` is evolved independently; single GIL release (ADR-0031).
    /// The presampled Laplacian sequence is shared across channels.
    #[pyo3(signature = (lambda_cols, n_steps=None))]
    fn evolve_state_adjoint_batched<'py>(
        &self,
        py: Python<'py>,
        lambda_cols: PyReadonlyArray2<'py, f64>,
        n_steps: Option<u32>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.graph.n_nodes();
        let [n_nodes, n_cols] = validate_batched_shape(lambda_cols.shape(), n)?;
        let ns = n_steps.map_or(self.n_steps, |v| v as usize);
        if ns != self.n_steps {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("n_steps={ns} != construction n_steps={}", self.n_steps),
            ));
        }
        let src = gather_nc_to_cn(&lambda_cols.as_array(), n_nodes, n_cols);
        let tau = self.tau;
        let out: Result<Vec<f64>, semiflow::SemiflowError> = match &self.variant {
            PresampledVariant::Magnus(ps) => py.detach(|| {
                let mut dst = vec![0.0f64; n_nodes * n_cols];
                ps.evolve_state_adjoint_batched_into(
                    tau,
                    ns,
                    &src,
                    &mut dst,
                    &mut ScratchPool::new(),
                )?;
                Ok(dst)
            }),
            PresampledVariant::VarCoef(ps) => py.detach(|| {
                let mut dst = vec![0.0f64; n_nodes * n_cols];
                ps.evolve_state_adjoint_batched_into(
                    tau,
                    ns,
                    &src,
                    &mut dst,
                    &mut ScratchPool::new(),
                )?;
                Ok(dst)
            }),
        };
        let result = out.map_err(|e| from_core(&e))?;
        Ok(scatter_cn_to_nc(&result, n_nodes, n_cols, py))
    }
}

// Phase-2 helpers (no Python types; called inside py.detach)

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

// Pre-sampled construction helpers

/// Sample `lap_at_t` at 2·`n_steps` GL₄ abscissa times and build a
/// `PreSampledLaplacianSeq`.  Returns `(seq, times)` so the caller
/// can reuse `times` for the optional `a`-coefficient sampling.
fn build_presampled_seq(
    py: Python<'_>,
    g: &Arc<Graph<f64>>,
    lap_at_t: &Py<PyAny>,
    ns: usize,
    t_horizon: f64,
) -> PyResult<(PreSampledLaplacianSeq<f64>, Vec<f64>)> {
    use crate::graph_py::extract_laplacian_arc;
    let base_lap = lap_at_t.call1(py, (0.0_f64,))?;
    let base_arc = extract_laplacian_arc(base_lap.bind(py), g);
    let row_ptr = base_arc.row_ptr().to_vec();
    let col_idx = base_arc.col_idx().to_vec();
    let nnz = col_idx.len();
    let mut times = vec![0.0_f64; 2 * ns];
    fill_abscissa_times(t_horizon, ns, &mut times);
    let mut vals_seq = vec![0.0_f64; 2 * ns * nnz];
    for (idx, &t) in times.iter().enumerate() {
        let lap_obj = lap_at_t.call1(py, (t,))?;
        let lap_arc = extract_laplacian_arc(lap_obj.bind(py), g);
        vals_seq[idx * nnz..(idx + 1) * nnz].copy_from_slice(lap_arc.vals());
    }
    let kind = LaplacianKind::Combinatorial;
    let seq = PreSampledLaplacianSeq::new(row_ptr, col_idx, vals_seq, ns, kind)
        .map_err(|e| crate::error::from_core(&e))?;
    Ok((seq, times))
}

/// Dispatch to Magnus or `VarCoef` variant and return a `GraphAdjointPresampled`.
#[allow(clippy::too_many_arguments)]
fn build_adjoint_variant(
    py: Python<'_>,
    g: Arc<Graph<f64>>,
    seq: PreSampledLaplacianSeq<f64>,
    times: &[f64],
    ns: usize,
    tau: f64,
    rho_bar: f64,
    a: Option<Py<PyAny>>,
    kernel: &str,
    convergence_check: bool,
) -> PyResult<GraphAdjointPresampled> {
    match kernel {
        "magnus_graph" => {
            let ps =
                MagnusGraphHeatChernoff::<f64>::from_presampled(seq, rho_bar, convergence_check)
                    .map_err(|e| crate::error::from_core(&e))?;
            Ok(GraphAdjointPresampled {
                variant: PresampledVariant::Magnus(ps),
                graph: g,
                n_steps: ns,
                tau,
            })
        }
        "varcoef_magnus_graph" => build_varcoef_adjoint(py, g, seq, times, ns, tau, rho_bar, a),
        other => Err(crate::error::new_pyerr(
            "OutOfDomain",
            &format!("unknown kernel '{other}'"),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_varcoef_adjoint(
    py: Python<'_>,
    g: Arc<Graph<f64>>,
    seq: PreSampledLaplacianSeq<f64>,
    times: &[f64],
    ns: usize,
    tau: f64,
    rho_bar: f64,
    a: Option<Py<PyAny>>,
) -> PyResult<GraphAdjointPresampled> {
    let a_cb = a.ok_or_else(|| {
        crate::error::new_pyerr("OutOfDomain", "varcoef_magnus_graph requires a= callback")
    })?;
    let n = g.n_nodes();
    let mut a_seq = vec![0.0_f64; 2 * ns * n];
    for (idx, &t) in times.iter().enumerate() {
        let a_vals: Vec<f64> = a_cb.call1(py, (t,))?.bind(py).extract()?;
        a_seq[idx * n..(idx + 1) * n].copy_from_slice(&a_vals);
    }
    let ps = VarCoefMagnusGraphHeatChernoff::<f64>::from_presampled(seq, a_seq, rho_bar, rho_bar)
        .map_err(|e| crate::error::from_core(&e))?;
    Ok(GraphAdjointPresampled {
        variant: PresampledVariant::VarCoef(ps),
        graph: g,
        n_steps: ns,
        tau,
    })
}
