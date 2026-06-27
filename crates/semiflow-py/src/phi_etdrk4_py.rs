//! Python bindings for φ-function actions and ETDRK4 driver (#12).
//!
//! ## Python API
//!
//! ```python
//! # φ_k(τA) · v  for a single k
//! out = semiflow.phi_action(op, k=1, tau=0.1, v=v_arr)
//!
//! # All φ_0 … φ_p simultaneously
//! outs = semiflow.phi_action_batched(op, p=3, tau=0.1, v=v_arr)
//! # outs.shape = (p+1, n)
//!
//! # ETDRK4 driver (menu-based nonlinearity, ADR-0189)
//! driver = semiflow.Etdrk4.from_symmetric_op(op, nonlinearity="allen_cahn", h=0.01)
//! u_final = driver.integrate(u0, n_steps=100)
//! ```
//!
//! GIL policy: ADR-0031 three-phase (validate → `py.detach` → scatter).
//! Nonlinearity menu: ADR-0189 §D3 — NO arbitrary Python callbacks.

#![allow(
    unsafe_code,
    clippy::doc_markdown,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::type_complexity,
)]

use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    AllenCahn, Etdrk4, NegLaplacianGenerator, ScratchPool, SymmetricOperator, phi_action,
    phi_action_batched, PHI_MAX,
};

use crate::{
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
    symmetric_op_py::PySymmetricOperator,
};

// ---------------------------------------------------------------------------
// phi_action — single φ_k
// ---------------------------------------------------------------------------

/// Compute ``φ_k(τA) · v`` for a single index ``k`` (§58, ADR-0189).
///
/// ``A = −L`` where ``L`` is the operator from ``op``
/// (NegLaplacianGenerator wraps it so ``A`` is the generator, not the Laplacian).
///
/// Parameters
/// ----------
/// op : SymmetricOperator
///     Base symmetric operator (represents ``L``).
/// k : int
///     φ-index in ``[0, PHI_MAX]`` (``PHI_MAX = 3``).
/// tau : float
///     Time step ``τ``.
/// v : ndarray[float64, shape (n,)]
///     Input vector.
///
/// Returns ndarray[float64, shape (n,)].
#[pyfunction]
#[pyo3(name = "phi_action", signature = (op, k, tau, v))]
pub fn phi_action_py<'py>(
    py: Python<'py>,
    op: &PySymmetricOperator,
    k: usize,
    tau: f64,
    v: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    catch_panic_py!({
        if k > PHI_MAX {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("k = {k} > PHI_MAX = {PHI_MAX}"),
            ));
        }
        if !tau.is_finite() {
            return Err(new_pyerr("OutOfDomain", "tau must be finite"));
        }
        let n = op.op.n();
        let v_vec: Vec<f64> = v
            .as_slice()
            .map_err(|_| new_pyerr("GridMismatch", "v must be contiguous"))?
            .to_vec();
        if v_vec.len() != n {
            return Err(new_pyerr(
                "GridMismatch",
                &format!("v length {} != op.n() {}", v_vec.len(), n),
            ));
        }
        let op_c = Arc::clone(&op.op);
        let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
            let gen = NegLaplacianGenerator::new(op_c.as_ref().clone());
            let mut out = vec![0.0_f64; n];
            let mut scratch = ScratchPool::new();
            phi_action(&gen, k, tau, &v_vec, &mut out, &mut scratch)?;
            Ok(out)
        });
        let out = result.map_err(|e| from_core(&e))?;
        Ok(out.as_slice().to_pyarray(py))
    })
}

// ---------------------------------------------------------------------------
// phi_action_batched — all φ_0 … φ_p
// ---------------------------------------------------------------------------

/// Compute ``φ_k(τA) · v`` for all ``k = 0 … p`` simultaneously (§58, ADR-0189).
///
/// Parameters
/// ----------
/// op : SymmetricOperator
///     Base symmetric operator (represents ``L``).
/// p : int
///     Max φ-index; must satisfy ``p <= PHI_MAX = 3``.
/// tau : float
///     Time step ``τ``.
/// v : ndarray[float64, shape (n,)]
///     Input vector.
///
/// Returns ndarray[float64, shape (p+1, n)].
/// ``result[k, :]`` is ``φ_k(τA) · v``.
#[pyfunction]
#[pyo3(name = "phi_action_batched", signature = (op, p, tau, v))]
pub fn phi_action_batched_py<'py>(
    py: Python<'py>,
    op: &PySymmetricOperator,
    p: usize,
    tau: f64,
    v: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    catch_panic_py!({
        if p > PHI_MAX {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("p = {p} > PHI_MAX = {PHI_MAX}"),
            ));
        }
        if !tau.is_finite() {
            return Err(new_pyerr("OutOfDomain", "tau must be finite"));
        }
        let n = op.op.n();
        let v_vec: Vec<f64> = v
            .as_slice()
            .map_err(|_| new_pyerr("GridMismatch", "v must be contiguous"))?
            .to_vec();
        if v_vec.len() != n {
            return Err(new_pyerr(
                "GridMismatch",
                &format!("v length {} != op.n() {}", v_vec.len(), n),
            ));
        }
        let op_c = Arc::clone(&op.op);
        let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
            let gen = NegLaplacianGenerator::new(op_c.as_ref().clone());
            let mut out = vec![0.0_f64; (p + 1) * n];
            let mut scratch = ScratchPool::new();
            phi_action_batched(&gen, p, tau, &v_vec, &mut out, &mut scratch)?;
            Ok(out)
        });
        let flat = result.map_err(|e| from_core(&e))?;
        // flat layout: out[k*n .. (k+1)*n] = φ_k(τA)v; reshape to (p+1, n)
        let arr = numpy::PyArray1::from_vec(py, flat);
        arr.reshape([p + 1, n]).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    })
}

// ---------------------------------------------------------------------------
// PyEtdrk4 — ETDRK4 driver (menu-based, ADR-0189 §D3)
// ---------------------------------------------------------------------------

/// ETDRK4 semilinear time-stepping driver (§58, ADR-0189).
///
/// Solves ``∂_t u = L u + N(u)``, ``u(0) = u₀``, where:
///
/// - ``L`` is the linear generator wrapped inside a :class:`SymmetricOperator` with
///   ``A = −L`` convention (``NegLaplacianGenerator``).
/// - ``N`` is a menu-based nonlinearity (no Python callbacks per ADR-0189 §D3).
///
/// **Nonlinearity menu** (``nonlinearity`` kwarg):
///
/// - ``"allen_cahn"`` — ``N(u) = u − u³``
///
/// Build with :meth:`from_symmetric_op`.
#[pyclass(name = "Etdrk4")]
pub struct PyEtdrk4 {
    driver: Arc<
        Etdrk4<
            f64,
            NegLaplacianGenerator<f64, SymmetricOperator<f64>>,
            AllenCahn<f64>,
        >,
    >,
    n: usize,
}

#[pymethods]
impl PyEtdrk4 {
    /// Build ETDRK4 driver from a :class:`SymmetricOperator`.
    ///
    /// Parameters
    /// ----------
    /// op : SymmetricOperator
    ///     Linear part ``L`` (operator represents ``L``; generator ``A = −L``).
    /// nonlinearity : str
    ///     Nonlinearity name.  Currently only ``"allen_cahn"`` (``N(u) = u − u³``).
    /// h : float
    ///     Fixed time-step size ``h > 0``.
    ///
    /// Raises ``SemiflowError(OutOfDomain)`` for unknown nonlinearity or ``h <= 0``.
    #[staticmethod]
    #[pyo3(signature = (op, nonlinearity, h))]
    fn from_symmetric_op(
        op: &PySymmetricOperator,
        nonlinearity: &str,
        h: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let nl = parse_nonlinearity(nonlinearity)?;
            if !h.is_finite() || h <= 0.0 {
                return Err(new_pyerr("OutOfDomain", "h must be finite and positive"));
            }
            let n = op.op.n();
            let gen = NegLaplacianGenerator::new(op.op.as_ref().clone());
            let driver = Etdrk4::new(gen, nl, h).map_err(|e| from_core(&e))?;
            Ok(Self { driver: Arc::new(driver), n })
        })
    }

    /// Advance ``u`` by one step of size ``h`` (set in :meth:`from_symmetric_op`).
    ///
    /// Parameters
    /// ----------
    /// u : ndarray[float64, shape (n,)]
    ///     Current state.
    ///
    /// Returns ndarray[float64, shape (n,)] — next state.
    fn step<'py>(
        &self,
        py: Python<'py>,
        u: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            let n = self.n;
            let u_vec: Vec<f64> = u
                .as_slice()
                .map_err(|_| new_pyerr("GridMismatch", "u must be contiguous"))?
                .to_vec();
            if u_vec.len() != n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("u length {} != op.n() {}", u_vec.len(), n),
                ));
            }
            let drv = Arc::clone(&self.driver);
            let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
                let mut u_next = vec![0.0_f64; n];
                let mut scratch = ScratchPool::new();
                drv.step(&u_vec, &mut u_next, &mut scratch)?;
                Ok(u_next)
            });
            let out = result.map_err(|e| from_core(&e))?;
            Ok(out.as_slice().to_pyarray(py))
        })
    }

    /// Integrate ``n_steps`` steps from ``u0``.
    ///
    /// Parameters
    /// ----------
    /// u0 : ndarray[float64, shape (n,)]
    ///     Initial condition.
    /// n_steps : int
    ///     Number of steps to advance.
    ///
    /// Returns ndarray[float64, shape (n,)] — final state ``u(n_steps · h)``.
    fn integrate<'py>(
        &self,
        py: Python<'py>,
        u0: PyReadonlyArray1<'py, f64>,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            let n = self.n;
            let u0_vec: Vec<f64> = u0
                .as_slice()
                .map_err(|_| new_pyerr("GridMismatch", "u0 must be contiguous"))?
                .to_vec();
            if u0_vec.len() != n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("u0 length {} != op.n() {}", u0_vec.len(), n),
                ));
            }
            let drv = Arc::clone(&self.driver);
            let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
                let mut out = vec![0.0_f64; n];
                let mut scratch = ScratchPool::new();
                drv.integrate(&u0_vec, n_steps, &mut out, &mut scratch)?;
                Ok(out)
            });
            let final_state = result.map_err(|e| from_core(&e))?;
            Ok(final_state.as_slice().to_pyarray(py))
        })
    }

    /// Operator dimension.
    fn n(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Nonlinearity menu parser
// ---------------------------------------------------------------------------

fn parse_nonlinearity(s: &str) -> PyResult<AllenCahn<f64>> {
    match s {
        "allen_cahn" => Ok(AllenCahn::new()),
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!(
                "unknown nonlinearity '{other}'; valid choices: \"allen_cahn\""
            ),
        )),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEtdrk4>()?;
    m.add_function(wrap_pyfunction!(phi_action_py, m)?)?;
    m.add_function(wrap_pyfunction!(phi_action_batched_py, m)?)?;
    Ok(())
}
