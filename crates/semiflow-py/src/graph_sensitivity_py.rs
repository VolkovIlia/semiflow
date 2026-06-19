//! Python binding for `edge_weight_grad` — adjoint-state parameter-sensitivity
//! (Issue #1, ADR-0115).
//!
//! Exposes the MATH primitive + building blocks. NO autograd hook.
//!
//! ## Python API
//!
//! ```python
//! semiflow.edge_weight_grad(
//!     graph, a, u0, dj_du_n, t, n_steps, rho_bar, params
//! ) -> np.ndarray[float64]
//! ```
//!
//! `params` = list of `(int,int)` edge pairs or the string `"all_edges"`.
//! Returns `∂J/∂w` for each requested edge.
//!
//! ## GIL policy (ADR-0031)
//!
//! Three-phase: validate (GIL held), compute (`py.detach`), build array (GIL held).

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value, clippy::needless_range_loop)]

use std::sync::Arc;

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow_core::{
    adjoint_state_gradient, EdgeWeightSensitivity, Graph, GraphSignal, Laplacian, LaplacianAtTime,
    MagnusGraphHeatChernoff, ScratchPool,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_extra::PyLaplacian,
    graph_py::{
        extract_f64_vec, resolve_graph_from_any, validate_n_steps, validate_rho_bar,
        validate_signal_len, validate_t_final,
    },
    panic::catch_panic_py,
};

/// Compute `∂J/∂w` for each requested edge via the discrete adjoint-state method.
///
/// `J(u_n) = ½‖u_n − target‖²` is implicit; the caller provides `dj_du_n = u_n − target`.
///
/// Parameters
/// ----------
/// graph : Graph, `GraphPath`, or Laplacian
///     Fixed-topology graph.
/// a : None
///     Reserved for future varcoef path; must be ``None`` (unit timescale).
/// u0 : `array_like`[float64]
///     Initial condition.
/// `dj_du_n` : `array_like`[float64]
///     Terminal sensitivity ``∂J/∂u_n``.
/// t : float
///     Total evolution time.
/// `n_steps` : int
///     Number of Magnus K=4 steps.
/// `rho_bar` : float
///     Upper bound on ``ρ̄(L_G)``.
/// params : list[tuple[int,int]] or "`all_edges`"
///     Which edge weights to differentiate.  ``"all_edges"`` includes every
///     undirected edge once (in CSR row-major order, `i < j`).
///
/// Returns
/// -------
/// np.ndarray[float64]
///     ``∂J/∂w`` for each requested edge (same order as ``params``).
#[pyfunction]
#[pyo3(signature = (graph=None, a=None, *, u0, dj_du_n, t, n_steps, rho_bar, params))]
#[allow(clippy::too_many_arguments)]
pub fn edge_weight_grad<'py>(
    py: Python<'py>,
    graph: Option<&Bound<'_, PyAny>>,
    a: Option<&Bound<'_, PyAny>>,
    u0: &Bound<'_, PyAny>,
    dj_du_n: &Bound<'_, PyAny>,
    t: f64,
    n_steps: u32,
    rho_bar: f64,
    params: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    catch_panic_py!({
        // Validate.
        validate_t_final(t)?;
        validate_n_steps(n_steps)?;
        validate_rho_bar(rho_bar)?;
        if a.is_some() {
            return Err(new_pyerr(
                "OutOfDomain",
                "edge_weight_grad: a != None (varcoef path not yet implemented; pass a=None)",
            ));
        }
        let g = resolve_graph_from_any(graph, None::<&PyLaplacian>)?;
        let u0_v = extract_f64_vec(u0)?;
        let dj_v = extract_f64_vec(dj_du_n)?;
        validate_signal_len(&u0_v, g.n_nodes())?;
        validate_signal_len(&dj_v, g.n_nodes())?;

        // Parse params.
        let edge_pairs = parse_params(params, &g)?;

        let n = n_steps as usize;
        #[allow(clippy::cast_precision_loss)]
        let tau = t / n as f64;

        let result: Result<Vec<f64>, semiflow_core::SemiflowError> = {
            let gc = Arc::clone(&g);
            py.detach(move || compute_edge_weight_grad(gc, u0_v, dj_v, tau, n, rho_bar, edge_pairs))
        };
        let out = result.map_err(|e| from_core(&e))?;
        Ok(out.as_slice().to_pyarray(py))
    })
}

/// Parse `params` from Python: list[(i,j)] or "`all_edges`".
fn parse_params(params: &Bound<'_, PyAny>, g: &Graph<f64>) -> PyResult<Vec<(usize, usize)>> {
    // Check for "all_edges" string.
    if let Ok(s) = params.extract::<String>() {
        if s != "all_edges" {
            return Err(new_pyerr(
                "OutOfDomain",
                r#"params string must be "all_edges""#,
            ));
        }
        return Ok(all_edges(g));
    }
    // Otherwise must be a list/sequence of (int,int) pairs.
    let list = params.extract::<Vec<(usize, usize)>>().map_err(|_| {
        new_pyerr(
            "OutOfDomain",
            r#"params must be list[(int,int)] or "all_edges""#,
        )
    })?;
    Ok(list)
}

/// Collect all undirected edges from the graph (i < j only, CSR order).
fn all_edges(g: &Graph<f64>) -> Vec<(usize, usize)> {
    let row_ptr = g.row_ptr();
    let col_idx = g.col_idx();
    let mut edges = Vec::new();
    for i in 0..g.n_nodes() {
        for ptr in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[ptr] as usize;
            if j > i {
                edges.push((i, j));
            }
        }
    }
    edges
}

/// Pure-Rust compute (called inside `py.detach`).
#[allow(clippy::too_many_arguments)]
fn compute_edge_weight_grad(
    graph: Arc<Graph<f64>>,
    u0_v: Vec<f64>,
    dj_v: Vec<f64>,
    tau: f64,
    n_steps: usize,
    rho_bar: f64,
    edge_pairs: Vec<(usize, usize)>,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let g2 = Arc::clone(&graph);
    let lap: LaplacianAtTime<f64> =
        Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
    let mc = MagnusGraphHeatChernoff::new(Arc::clone(&graph), lap, rho_bar, true)?;
    let u0 = GraphSignal::from_fn(Arc::clone(&graph), |i| u0_v[i as usize]);
    let dj = GraphSignal::from_fn(Arc::clone(&graph), |i| dj_v[i as usize]);
    let n_params = edge_pairs.len();
    let sens = EdgeWeightSensitivity {
        params: edge_pairs,
        n_nodes: graph.n_nodes(),
    };
    let mut grad = vec![0.0_f64; n_params];
    let mut scratch = ScratchPool::new();
    adjoint_state_gradient(&mc, &u0, n_steps, tau, &dj, &sens, &mut grad, &mut scratch)?;
    Ok(grad)
}

/// Register `edge_weight_grad` in the Python module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_function(wrap_pyfunction!(edge_weight_grad, m)?)?;
    Ok(())
}
