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
//! # ADR-0202: Σ τ^k φ_k(τG) w_k in one sweep; mass= and GeneralOperator on all three
//! y = semiflow.phi_combination(op, tau, w, mass=None)   # w.shape = (p+1, n)
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
    clippy::type_complexity
)]

use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{exceptions::PyTypeError, prelude::*};
use semiflow::{
    general_operator::GeneralOperator, phi_action, phi_action_batched, phi_combination, AllenCahn,
    CsrGenerator, Etdrk4, NegLaplacianGenerator, ScratchPool, SymmetricOperator, PHI_MAX,
};

use crate::{
    error::{from_core, new_pyerr},
    general_op_py::PyGeneralOperator,
    panic::catch_panic_py,
    spd_resolvent_py::contiguous_vec,
    symmetric_op_py::PySymmetricOperator,
};

// ---------------------------------------------------------------------------
// Operator argument: SymmetricOperator | GeneralOperator (ADR-0202 D3)
// ---------------------------------------------------------------------------

/// The generator source of a φ call. Cloning is `Arc`-cheap (phase 1, GIL held).
enum PhiOp {
    Sym(Arc<SymmetricOperator<f64>>),
    General(Arc<GeneralOperator<f64>>),
}

impl PhiOp {
    fn extract(op: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(sym) = op.extract::<PyRef<'_, PySymmetricOperator>>() {
            return Ok(Self::Sym(Arc::clone(&sym.op)));
        }
        if let Ok(gen) = op.extract::<PyRef<'_, PyGeneralOperator>>() {
            return Ok(Self::General(Arc::clone(&gen.inner)));
        }
        Err(PyTypeError::new_err(
            "op must be a SymmetricOperator or a GeneralOperator",
        ))
    }

    fn n(&self) -> usize {
        match self {
            Self::Sym(op) => op.n(),
            Self::General(op) => op.n(),
        }
    }

    /// `G = −M⁻¹A` (`mass = None`: `M = I`); bit-identical to `NegLaplacianGenerator` then.
    fn generator(
        &self,
        mass: Option<&[f64]>,
    ) -> Result<CsrGenerator<f64>, semiflow::SemiflowError> {
        match self {
            Self::Sym(op) => CsrGenerator::from_symmetric(op, mass),
            Self::General(op) => CsrGenerator::from_general(op, mass),
        }
    }
}

/// Phase-1 validation shared by the three φ entry points; returns `(op, mass)`.
fn phi_inputs(
    op: &Bound<'_, PyAny>,
    tau: f64,
    mass: Option<&PyReadonlyArray1<'_, f64>>,
) -> PyResult<(PhiOp, Option<Vec<f64>>)> {
    let op = PhiOp::extract(op)?;
    if !tau.is_finite() {
        return Err(new_pyerr("OutOfDomain", "tau must be finite"));
    }
    let mass = mass.map(|m| contiguous_vec(m, "mass")).transpose()?;
    if let Some(m) = &mass {
        if m.len() != op.n() {
            return Err(new_pyerr(
                "GridMismatch",
                &format!("mass length {} != op.n() {}", m.len(), op.n()),
            ));
        }
    }
    Ok((op, mass))
}

fn check_vec_len(v: &[f64], n: usize, name: &str) -> PyResult<()> {
    if v.len() != n {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("{name} length {} != op.n() {n}", v.len()),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// phi_action — single φ_k
// ---------------------------------------------------------------------------

/// Compute ``φ_k(τG) · v`` for a single index ``k``, ``G = −M⁻¹A`` (§58, §62.3).
///
/// Parameters
/// ----------
/// op : SymmetricOperator or GeneralOperator
///     The operator ``A`` (PSD / accretive; evolution ``e^{−tA}``).
/// k : int
///     φ-index in ``[0, PHI_MAX]`` (``PHI_MAX = 3``).
/// tau : float
///     Time step ``τ``.
/// v : ndarray[float64, shape (n,)]
///     Input vector.
/// mass : ndarray[float64, shape (n,)] or None
///     Diagonal mass ``M > 0`` (default identity).
///
/// Returns ndarray[float64, shape (n,)].
#[pyfunction]
#[pyo3(name = "phi_action", signature = (op, k, tau, v, mass = None))]
pub fn phi_action_py<'py>(
    py: Python<'py>,
    op: &Bound<'py, PyAny>,
    k: usize,
    tau: f64,
    v: PyReadonlyArray1<'py, f64>,
    mass: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    catch_panic_py!({
        if k > PHI_MAX {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("k = {k} > PHI_MAX = {PHI_MAX}"),
            ));
        }
        let (op, mass) = phi_inputs(op, tau, mass.as_ref())?;
        let n = op.n();
        let v_vec = contiguous_vec(&v, "v")?;
        check_vec_len(&v_vec, n, "v")?;
        let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
            let gen = op.generator(mass.as_deref())?;
            let mut out = vec![0.0_f64; n];
            phi_action(&gen, k, tau, &v_vec, &mut out, &mut ScratchPool::new())?;
            Ok(out)
        });
        let out = result.map_err(|e| from_core(&e))?;
        Ok(out.as_slice().to_pyarray(py))
    })
}

// ---------------------------------------------------------------------------
// phi_action_batched — all φ_0 … φ_p
// ---------------------------------------------------------------------------

/// Compute ``φ_k(τG) · v`` for all ``k = 0 … p`` simultaneously (§58, §62.3).
///
/// Parameters
/// ----------
/// op : SymmetricOperator or GeneralOperator
///     The operator ``A``.
/// p : int
///     Max φ-index; must satisfy ``p <= PHI_MAX = 3``.
/// tau : float
///     Time step ``τ``.
/// v : ndarray[float64, shape (n,)]
///     Input vector.
/// mass : ndarray[float64, shape (n,)] or None
///     Diagonal mass ``M > 0`` (default identity).
///
/// Returns ndarray[float64, shape (p+1, n)].
/// ``result[k, :]`` is ``φ_k(τG) · v``.
#[pyfunction]
#[pyo3(name = "phi_action_batched", signature = (op, p, tau, v, mass = None))]
pub fn phi_action_batched_py<'py>(
    py: Python<'py>,
    op: &Bound<'py, PyAny>,
    p: usize,
    tau: f64,
    v: PyReadonlyArray1<'py, f64>,
    mass: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    catch_panic_py!({
        if p > PHI_MAX {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("p = {p} > PHI_MAX = {PHI_MAX}"),
            ));
        }
        let (op, mass) = phi_inputs(op, tau, mass.as_ref())?;
        let n = op.n();
        let v_vec = contiguous_vec(&v, "v")?;
        check_vec_len(&v_vec, n, "v")?;
        let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
            let gen = op.generator(mass.as_deref())?;
            let mut out = vec![0.0_f64; (p + 1) * n];
            phi_action_batched(&gen, p, tau, &v_vec, &mut out, &mut ScratchPool::new())?;
            Ok(out)
        });
        let flat = result.map_err(|e| from_core(&e))?;
        // flat layout: out[k*n .. (k+1)*n] = φ_k(τG)v; reshape to (p+1, n)
        let arr = numpy::PyArray1::from_vec(py, flat);
        arr.reshape([p + 1, n])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    })
}

// ---------------------------------------------------------------------------
// phi_combination — Σ τ^k φ_k(τG) w_k in one augmented sweep
// ---------------------------------------------------------------------------

/// ``y = Σ_{k=0}^{p} τ^k φ_k(τG) w_k`` in one augmented Horner sweep (§62.4).
///
/// ``G = −M⁻¹A``. With the recipes of §62.5 this gives exact affine evolution
/// (``M u' = −A u + s``: ``w = [u₀, M⁻¹ s]``), ETD-RK2 and ETDRK4 steps, with the
/// nonlinearity evaluated by the caller (no Python callbacks cross the boundary).
///
/// Parameters
/// ----------
/// op : SymmetricOperator or GeneralOperator
///     The operator ``A``.
/// tau : float
///     Step ``τ ≥ 0``.
/// w : ndarray[float64, shape (p+1, n)]
///     Rows ``w_0 … w_p``, ``p <= PHI_MAX = 3``.
/// mass : ndarray[float64, shape (n,)] or None
///     Diagonal mass ``M > 0`` (default identity).
///
/// Returns ndarray[float64, shape (n,)].
#[pyfunction]
#[pyo3(name = "phi_combination", signature = (op, tau, w, mass = None))]
pub fn phi_combination_py<'py>(
    py: Python<'py>,
    op: &Bound<'py, PyAny>,
    tau: f64,
    w: PyReadonlyArray2<'py, f64>,
    mass: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    catch_panic_py!({
        let (op, mass) = phi_inputs(op, tau, mass.as_ref())?;
        let n = op.n();
        let view = w.as_array();
        if view.shape()[1] != n {
            return Err(new_pyerr(
                "GridMismatch",
                &format!("w has {} columns, expected n = {n}", view.shape()[1]),
            ));
        }
        let rows: Vec<Vec<f64>> = view.rows().into_iter().map(|r| r.to_vec()).collect();
        let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
            let gen = op.generator(mass.as_deref())?;
            let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
            let mut out = vec![0.0_f64; n];
            phi_combination(&gen, tau, &refs, &mut out, &mut ScratchPool::new())?;
            Ok(out)
        });
        let out = result.map_err(|e| from_core(&e))?;
        Ok(out.as_slice().to_pyarray(py))
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
    driver: Arc<Etdrk4<f64, NegLaplacianGenerator<f64, SymmetricOperator<f64>>, AllenCahn<f64>>>,
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
    fn from_symmetric_op(op: &PySymmetricOperator, nonlinearity: &str, h: f64) -> PyResult<Self> {
        catch_panic_py!({
            let nl = parse_nonlinearity(nonlinearity)?;
            if !h.is_finite() || h <= 0.0 {
                return Err(new_pyerr("OutOfDomain", "h must be finite and positive"));
            }
            let n = op.op.n();
            let gen = NegLaplacianGenerator::new(op.op.as_ref().clone());
            let driver = Etdrk4::new(gen, nl, h).map_err(|e| from_core(&e))?;
            Ok(Self {
                driver: Arc::new(driver),
                n,
            })
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
            &format!("unknown nonlinearity '{other}'; valid choices: \"allen_cahn\""),
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
    m.add_function(wrap_pyfunction!(phi_combination_py, m)?)?;
    Ok(())
}
