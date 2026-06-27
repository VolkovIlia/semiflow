//! Python bindings for conservative (FV divergence-form) diffusion (#11).
//!
//! ## Python API
//!
//! ```python
//! # Chernoff semigroup via Thomas tridiagonal solver
//! op = semiflow.ConservativeDiffusionChernoff.from_k_array(
//!         n=64, x_lo=0.0, x_hi=1.0, k_nodes=k_arr)
//! sym_op = op.to_symmetric_operator()  # → SymmetricOperator (Krylov bridge)
//!
//! # Direct Krylov assembly
//! sym_op = semiflow.assemble_conservative_csr_1d(
//!         n=64, x_lo=0.0, x_hi=1.0, k_nodes=k_arr)  # → SymmetricOperator
//! ```
//!
//! GIL policy: ADR-0031 three-phase (validate → `py.detach` → scatter).

#![allow(
    unsafe_code,
    clippy::doc_markdown,
    clippy::needless_pass_by_value,
)]

use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use semiflow::{
    assemble_conservative_csr_1d, BoundaryPolicy, ConservativeDiffusionChernoff, Grid1D,
};

use crate::{
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
    symmetric_op_py::PySymmetricOperator,
};

// ---------------------------------------------------------------------------
// parse helpers
// ---------------------------------------------------------------------------

/// Parse Python `boundary` kwarg for conservative diffusion.
///
/// Accepted: `"neumann"` (default, zero-flux), `"dirichlet:<value>"`.
fn parse_conservative_boundary(s: &str) -> PyResult<BoundaryPolicy<f64>> {
    let sl = s.to_ascii_lowercase();
    if sl == "neumann" {
        return Ok(BoundaryPolicy::Neumann);
    }
    if let Some(val_str) = sl.strip_prefix("dirichlet:") {
        let v = val_str.parse::<f64>().map_err(|_| {
            new_pyerr("OutOfDomain", &format!("invalid Dirichlet value in '{s}'"))
        })?;
        return Ok(BoundaryPolicy::Dirichlet { value: v });
    }
    Err(new_pyerr(
        "OutOfDomain",
        &format!(
            "unknown conservative boundary '{s}'; valid: \"neumann\", \"dirichlet:<value>\""
        ),
    ))
}

fn make_grid(n: usize, x_lo: f64, x_hi: f64) -> PyResult<Grid1D<f64>> {
    if n < 2 {
        return Err(new_pyerr("GridMismatch", "n must be >= 2"));
    }
    if !x_lo.is_finite() || !x_hi.is_finite() || x_lo >= x_hi {
        return Err(new_pyerr("OutOfDomain", "need finite x_lo < x_hi"));
    }
    Grid1D::new(x_lo, x_hi, n).map_err(|e| from_core(&e))
}

fn parse_k_nodes(arr: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    Ok(arr
        .as_slice()
        .map_err(|_| new_pyerr("GridMismatch", "k_nodes must be contiguous"))?
        .to_vec())
}

fn parse_r_contact(arr: Option<PyReadonlyArray1<'_, f64>>) -> PyResult<Option<Vec<f64>>> {
    match arr {
        None => Ok(None),
        Some(a) => Ok(Some(
            a.as_slice()
                .map_err(|_| new_pyerr("GridMismatch", "r_contact must be contiguous"))?
                .to_vec(),
        )),
    }
}

// ---------------------------------------------------------------------------
// PyConservativeDiffusionChernoff pyclass
// ---------------------------------------------------------------------------

/// Order-2 conservative (FV divergence-form) variable-coefficient diffusion (§56).
///
/// Generator: ``L_k u = ∂_x(k(x) ∂_x u)`` with harmonic-mean face conductivities.
/// Chernoff step: Crank–Nicolson ``(I − ½τL_k)⁻¹(I + ½τL_k)`` via O(n) Thomas solve.
///
/// Use :meth:`to_symmetric_operator` to obtain a :class:`SymmetricOperator`
/// consumable by :meth:`SymmetricOperator.evolve_batched` (Krylov, issue #13).
#[pyclass(name = "ConservativeDiffusionChernoff")]
pub struct PyConservativeDiffusionChernoff {
    inner: Arc<ConservativeDiffusionChernoff<f64>>,
}

#[pymethods]
impl PyConservativeDiffusionChernoff {
    /// Build from node-sampled conductivities ``k_nodes[i] = k(x_i) > 0``.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Grid size (``n >= 2``).
    /// x_lo, x_hi : float
    ///     Domain endpoints (``x_lo < x_hi``).
    /// k_nodes : ndarray[float64, shape (n,)]
    ///     Positive conductivities at each node.
    /// r_contact : ndarray[float64, shape (n-1,)] or None
    ///     Optional contact resistance at each face (default: None → no resistance).
    /// boundary : str
    ///     ``"neumann"`` (default, zero-flux) or ``"dirichlet:<value>"``.
    ///
    /// Raises ``SemiflowError`` on invalid inputs.
    #[staticmethod]
    #[pyo3(signature = (n, x_lo, x_hi, k_nodes, r_contact = None, boundary = "neumann"))]
    fn from_k_array(
        n: usize,
        x_lo: f64,
        x_hi: f64,
        k_nodes: PyReadonlyArray1<'_, f64>,
        r_contact: Option<PyReadonlyArray1<'_, f64>>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let grid = make_grid(n, x_lo, x_hi)?;
            let k = parse_k_nodes(&k_nodes)?;
            let rc = parse_r_contact(r_contact)?;
            let bc = parse_conservative_boundary(boundary)?;
            let inner = ConservativeDiffusionChernoff::from_k_array(
                grid,
                &k,
                rc.as_deref(),
                bc,
            )
            .map_err(|e| from_core(&e))?;
            Ok(Self { inner: Arc::new(inner) })
        })
    }

    /// Assemble ``A = −L_k`` as a :class:`SymmetricOperator` (Krylov bridge, §56.2).
    ///
    /// Uses Neumann BCs for the CSR carrier (natural for Krylov).
    ///
    /// Returns :class:`SymmetricOperator`.
    fn to_symmetric_operator(&self) -> PyResult<PySymmetricOperator> {
        catch_panic_py!({
            let op = self.inner.to_symmetric_operator().map_err(|e| from_core(&e))?;
            Ok(PySymmetricOperator { op: Arc::new(op) })
        })
    }

    /// Number of grid nodes.
    fn n(&self) -> usize {
        self.inner.grid.n
    }

    /// Grid step size ``Δx = (x_hi − x_lo) / (n − 1)``.
    fn dx(&self) -> f64 {
        self.inner.grid.dx()
    }
}

// ---------------------------------------------------------------------------
// assemble_conservative_csr_1d — free function
// ---------------------------------------------------------------------------

/// Directly assemble ``A = −L_k`` as :class:`SymmetricOperator` from CSR (§56.1–56.2).
///
/// Skips building the Chernoff object; useful when you only need the Krylov carrier.
/// Always uses Neumann (zero-flux) BCs.
///
/// Parameters
/// ----------
/// n : int
///     Grid size (``n >= 2``).
/// x_lo, x_hi : float
///     Domain endpoints.
/// k_nodes : ndarray[float64, shape (n,)]
///     Positive conductivities at each node.
/// r_contact : ndarray[float64, shape (n-1,)] or None
///     Optional contact resistance per face.
///
/// Returns :class:`SymmetricOperator`.
#[pyfunction]
#[pyo3(name = "assemble_conservative_csr_1d",
       signature = (n, x_lo, x_hi, k_nodes, r_contact = None))]
pub fn assemble_conservative_csr_1d_py(
    n: usize,
    x_lo: f64,
    x_hi: f64,
    k_nodes: PyReadonlyArray1<'_, f64>,
    r_contact: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<PySymmetricOperator> {
    catch_panic_py!({
        let grid = make_grid(n, x_lo, x_hi)?;
        let k = parse_k_nodes(&k_nodes)?;
        let rc = parse_r_contact(r_contact)?;
        let op = assemble_conservative_csr_1d(grid, &k, rc.as_deref(), BoundaryPolicy::Neumann)
            .map_err(|e| from_core(&e))?;
        Ok(PySymmetricOperator { op: Arc::new(op) })
    })
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConservativeDiffusionChernoff>()?;
    m.add_function(wrap_pyfunction!(assemble_conservative_csr_1d_py, m)?)?;
    Ok(())
}
