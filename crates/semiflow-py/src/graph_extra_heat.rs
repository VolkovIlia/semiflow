//! `GraphHeat4th` and `VarCoefGraphHeat` Python classes.
//!
//! Split from `graph_extra.rs` for suckless file-size compliance.

#![allow(unsafe_code)]

use std::sync::Arc;

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;

use semiflow_core::{
    ChernoffSemigroup, Graph, GraphHeat4thChernoff, GraphSignal, Laplacian,
    VarCoefGraphHeatChernoff,
};

use crate::{
    dtype_dispatch::{cast_f64_to_f32, parse_dtype, Dtype},
    error::{from_core, new_pyerr},
    graph_extra::{resolve_lap_and_graph, PyGraph, PyLaplacian},
    graph_heat_f32::compute_var_coef_f32,
    graph_py::{
        extract_f64_vec, validate_n_steps, validate_rho_bar, validate_signal_len, validate_t_final,
    },
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// GraphHeat4th — order-4 Chernoff heat
// ---------------------------------------------------------------------------

/// Order-4 graph heat equation: ``∂ₜu = −L_G u`` via Padé[0,4] Taylor
/// truncation of ``exp(−τ L_G)``.
///
/// ``S₄(τ) f = Σ_{k=0}^{4} (−τ L_G)^k / k! · f``
///
/// The GIL is released during :meth:`evolve` (ADR-0031 three-phase pattern).
///
/// Parameters
/// ----------
/// laplacian : Laplacian, optional
///     Pre-assembled Laplacian.  Preferred over ``graph``.
/// graph : Graph, optional
///     Topology; combinatorial Laplacian assembled internally if provided.
/// `rho_bar` : float
///     Gershgorin spectral-radius bound ``ρ̄ ≥ ρ(L_G)``.  Must be > 0.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='OutOfDomain'`` if ``rho_bar <= 0`` or neither ``laplacian``
///     nor ``graph`` is provided.
#[pyclass(name = "GraphHeat4th")]
pub struct GraphHeat4th {
    laplacian: Arc<Laplacian<f64>>,
    graph: Arc<Graph<f64>>,
    n_nodes: usize,
}

#[pymethods]
impl GraphHeat4th {
    /// Create a `GraphHeat4th` state from a Laplacian or Graph.
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
            Ok(GraphHeat4th {
                laplacian: lap_arc,
                graph: g_arc,
                n_nodes: n,
            })
        })
    }

    /// Evolve ``f0`` to ``t=t_final`` using ``n_steps`` Chernoff steps.
    /// GIL released during compute (ADR-0031).  Raises ``OutOfDomain`` /
    /// ``GridMismatch`` on invalid inputs.
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
            let result: Result<Vec<f64>, semiflow_core::SemiflowError> =
                py.detach(|| compute_heat4(lap, graph, &input, t_final, n_st));
            Ok(result.map_err(|e| from_core(&e))?.as_slice().to_pyarray(py))
        })
    }
}

// ---------------------------------------------------------------------------
// VarCoefGraphHeat — variable-coefficient order-2 heat
// ---------------------------------------------------------------------------

/// Variable-coefficient graph heat: ``∂ₜu = −L_a u``,
/// ``L_a = A^{1/2} L_G A^{1/2}``, ``A = diag(a)``.
///
/// Order-2 Chernoff approximation.  See ADR-0053 and math.md §14.2.
///
/// The GIL is released during :meth:`evolve` (ADR-0031 three-phase pattern).
///
/// Parameters
/// ----------
/// graph : Graph
///     Topology; combinatorial Laplacian assembled internally.
/// a : numpy.ndarray[float64]
///     Conductivity vector ``a[i] > 0``; length must equal ``n_nodes``.
/// `rho_bar` : float
///     Gershgorin spectral-radius bound for ``L_a``.  Must be > 0.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='OutOfDomain'`` if ``rho_bar <= 0``, ``len(a) != n_nodes``,
///     or any ``a[i]`` is non-positive or non-finite.
#[pyclass(name = "VarCoefGraphHeat")]
pub struct VarCoefGraphHeat {
    graph: Arc<Graph<f64>>,
    a: Vec<f64>,
    rho_bar: f64,
    n_nodes: usize,
    dtype: Dtype,
}

#[pymethods]
impl VarCoefGraphHeat {
    /// Create a `VarCoefGraphHeat` state.
    ///
    /// D4 (ADR-0113): ``rho_bar`` is keyword-only.
    /// Optional ``dtype="f32"`` runs the kernel in single precision (Issue #3).
    #[new]
    #[pyo3(signature = (graph, a, *, rho_bar, dtype="f64"))]
    fn new(graph: &PyGraph, a: &Bound<'_, PyAny>, rho_bar: f64, dtype: &str) -> PyResult<Self> {
        catch_panic_py!({
            validate_rho_bar(rho_bar)?;
            let dt = parse_dtype(dtype)?;
            let a_vec = extract_f64_vec(a)?;
            validate_a_values(&a_vec)?;
            let g = Arc::clone(&graph.inner);
            let n = g.n_nodes();
            // Validate by constructing (returns error for wrong length etc.)
            VarCoefGraphHeatChernoff::new(Arc::clone(&g), a_vec.clone(), rho_bar)
                .map_err(|e| from_core(&e))?;
            Ok(VarCoefGraphHeat {
                graph: g,
                a: a_vec,
                rho_bar,
                n_nodes: n,
                dtype: dt,
            })
        })
    }

    /// Evolve ``f0`` to ``t=t_final`` using ``n_steps`` Chernoff steps.
    /// GIL released during compute (ADR-0031).
    /// Returns ``float32`` when ``dtype="f32"`` was set at construction.
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
            validate_signal_len(&input, self.n_nodes)?;
            let graph = Arc::clone(&self.graph);
            let a_vec = self.a.clone();
            let rho = self.rho_bar;
            let n_st = n_steps as usize;
            match self.dtype {
                Dtype::F64 => {
                    let result: Result<Vec<f64>, semiflow_core::SemiflowError> =
                        py.detach(|| compute_var_coef(graph, a_vec, rho, &input, t_final, n_st));
                    let arr = result.map_err(|e| from_core(&e))?.as_slice().to_pyarray(py);
                    Ok(arr.into_any())
                }
                Dtype::F32 => {
                    let result: Result<Vec<f64>, semiflow_core::SemiflowError> = py
                        .detach(|| compute_var_coef_f32(graph, &a_vec, rho, &input, t_final, n_st));
                    let out_f32: Vec<f32> = cast_f64_to_f32(&result.map_err(|e| from_core(&e))?);
                    let arr = out_f32.as_slice().to_pyarray(py);
                    Ok(arr.into_any())
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Phase-2 helpers (called inside py.detach — no Python types allowed)
// ---------------------------------------------------------------------------

/// Order-4 graph heat evolution.  No GIL held.
fn compute_heat4(
    lap: Arc<Laplacian<f64>>,
    graph: Arc<Graph<f64>>,
    input: &[f64],
    t_final: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let chernoff = GraphHeat4thChernoff::new(lap);
    let sg = ChernoffSemigroup::new(chernoff, n_steps)?;
    let f0 = GraphSignal::from_fn(graph, |i| input[i as usize]);
    let result = sg.evolve(t_final, &f0)?;
    Ok(result.values().to_vec())
}

/// Variable-coefficient graph heat evolution.  No GIL held.
fn compute_var_coef(
    graph: Arc<Graph<f64>>,
    a: Vec<f64>,
    rho_bar: f64,
    input: &[f64],
    t_final: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let g2 = Arc::clone(&graph);
    let chernoff = VarCoefGraphHeatChernoff::new(graph, a, rho_bar)?;
    let sg = ChernoffSemigroup::new(chernoff, n_steps)?;
    let f0 = GraphSignal::from_fn(g2, |i| input[i as usize]);
    let result = sg.evolve(t_final, &f0)?;
    Ok(result.values().to_vec())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Validate that all conductivity values are finite and positive.
fn validate_a_values(a: &[f64]) -> PyResult<()> {
    for &v in a {
        if !v.is_finite() || v <= 0.0 {
            return Err(new_pyerr(
                "OutOfDomain",
                "a must contain finite positive values",
            ));
        }
    }
    Ok(())
}
