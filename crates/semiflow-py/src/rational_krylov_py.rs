//! Python binding of the rational (shift-and-invert) Krylov action (ADR-0206).
//!
//! ```python
//! rk = op.rational_krylov(t, mass=None, tol=1e-10, gamma=None, m_max=64)
//! y, report = rk.apply(v)          # e^{-t M^{-1} A} v, report: dict
//! Y = rk.apply_batched(V_nc)       # column by column, one factor
//! Y = op.evolve_batched(t, V_nc, path="rational")
//! Y = semiflow.mass_lumped_evolve(op, m, t, V_nc, path="rational")
//! ```
//!
//! GIL policy: ADR-0031 three-phase (validate → `py.detach` → scatter).

// PyO3 extractors are passed by value by convention (cf. `symmetric_op_py.rs`).
#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::{prelude::*, types::PyDict};
use semiflow::{
    Precond, RationalKrylov, RationalOptions, RationalReport, ScratchPool, SemiflowError,
    SpdSolver, SymmetricOperator,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_py::{gather_nc_to_cn, scatter_cn_to_nc, validate_batched_shape},
    panic::catch_panic_py,
};

/// Default Krylov dimension of the rational path.
const RATIONAL_M_MAX: usize = 64;

/// `e^{-t·M⁻¹A}·v` by shift-and-invert Lanczos with one factor of ``M/γ + A``
/// (ADR-0206). Build with :meth:`SymmetricOperator.rational_krylov`.
///
/// The number of solves is independent of ``λ_max·t``: the a-priori Krylov
/// dimension of ``tol`` (27 at ``tol = 1e-10`` with ``γ = t/10``), fewer on an
/// invariant subspace.
#[pyclass(name = "RationalKrylov", frozen)]
pub struct PyRationalKrylov {
    inner: Arc<RationalKrylov<f64>>,
}

/// Settings shared by every rational entry point.
pub(crate) struct RationalSettings {
    pub(crate) tol: f64,
    pub(crate) gamma: Option<f64>,
    pub(crate) m_max: usize,
    pub(crate) solver: String,
    pub(crate) precond: String,
}

/// `path=` of `evolve_batched` / `mass_lumped_evolve`, which (unlike
/// `MassKOperator.evolve`) also accept `"rational"`: an unknown name is reported with
/// the full menu here, before the shared polynomial-path parser sees it.
pub(crate) fn check_rational_menu(path: &str) -> PyResult<()> {
    if matches!(path, "chebyshev" | "lanczos" | "implicit" | "rational") {
        return Ok(());
    }
    Err(new_pyerr(
        "Unsupported",
        &format!("path must be 'chebyshev', 'lanczos', 'implicit' or 'rational', got '{path}'"),
    ))
}

/// Rational-path settings from the `evolve_batched` keyword set.
pub(crate) fn rational_settings(
    tol: f64,
    gamma: Option<f64>,
    m_max: Option<u32>,
) -> RationalSettings {
    RationalSettings {
        tol,
        gamma,
        m_max: m_max.map_or(RATIONAL_M_MAX, |m| m as usize),
        solver: "auto".into(),
        precond: "ic0".into(),
    }
}

/// `(solver, fallback)` for a solver menu entry: `"auto"` tries the tridiagonal
/// `LDLᵀ` and falls back to PCG (the `SymmetricOperator.resolvent` menu).
fn solver_menu(solver: &str, precond: &str) -> PyResult<(SpdSolver, Option<SpdSolver>)> {
    let precond = match precond {
        "ic0" => Precond::Ic0,
        "jacobi" => Precond::Jacobi,
        other => {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("precond must be 'ic0' or 'jacobi', got '{other}'"),
            ))
        }
    };
    let pcg = SpdSolver::Pcg {
        precond,
        max_iter: None,
    };
    match solver {
        "auto" => Ok((SpdSolver::Tridiagonal, Some(pcg))),
        "tridiagonal" => Ok((SpdSolver::Tridiagonal, None)),
        "pcg" => Ok((pcg, None)),
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!("solver must be 'auto', 'tridiagonal' or 'pcg', got '{other}'"),
        )),
    }
}

/// Validate the menu under the GIL; the factor is built by [`build`].
pub(crate) fn plan(s: &RationalSettings) -> PyResult<(RationalOptions, Option<SpdSolver>)> {
    let (primary, fallback) = solver_menu(&s.solver, &s.precond)?;
    let mut opts = RationalOptions::default()
        .with_tol(s.tol)
        .with_m_max(s.m_max)
        .with_solver(primary);
    if let Some(g) = s.gamma {
        opts = opts.with_gamma(g);
    }
    Ok((opts, fallback))
}

/// Build the factor (GIL released by the caller).
pub(crate) fn build(
    op: &SymmetricOperator<f64>,
    mass: Option<&[f64]>,
    t: f64,
    (opts, fallback): (RationalOptions, Option<SpdSolver>),
) -> Result<RationalKrylov<f64>, SemiflowError> {
    match (RationalKrylov::new(op, mass, t, opts), fallback) {
        (Err(SemiflowError::Unsupported { .. }), Some(fb)) => {
            RationalKrylov::new(op, mass, t, opts.with_solver(fb))
        }
        (other, _) => other,
    }
}

/// Apply to every column of a column-major `[C][N]` buffer.
pub(crate) fn apply_columns(
    rk: &RationalKrylov<f64>,
    src_cn: &[f64],
    n: usize,
) -> Result<Vec<f64>, SemiflowError> {
    let mut out = vec![0.0; src_cn.len()];
    let mut scratch = ScratchPool::new();
    for (src, dst) in src_cn.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        rk.apply_into(src, dst, &mut scratch)?;
    }
    Ok(out)
}

/// `SymmetricOperator.evolve_batched(path="rational")` and
/// `mass_lumped_evolve(path="rational")`: one factor, every column.
pub(crate) fn rational_batched<'py>(
    py: Python<'py>,
    op: &Arc<SymmetricOperator<f64>>,
    mass: Option<Vec<f64>>,
    t: f64,
    v_nc: &PyReadonlyArray2<'py, f64>,
    settings: &RationalSettings,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let menu = plan(settings)?;
    let [n_nodes, n_cols] = validate_batched_shape(v_nc.shape(), op.n())?;
    let src_cn = gather_nc_to_cn(&v_nc.as_array(), n_nodes, n_cols);
    let op_c = Arc::clone(op);
    let result = py.detach(move || {
        let rk = build(&op_c, mass.as_deref(), t, menu)?;
        apply_columns(&rk, &src_cn, n_nodes)
    });
    let dst_cn = result.map_err(|e| from_core(&e))?;
    Ok(scatter_cn_to_nc(&dst_cn, n_nodes, n_cols, py))
}

/// Body of `SymmetricOperator.rational_krylov`.
pub(crate) fn new_rational(
    py: Python<'_>,
    op: &Arc<SymmetricOperator<f64>>,
    t: f64,
    mass: Option<Vec<f64>>,
    settings: &RationalSettings,
) -> PyResult<PyRationalKrylov> {
    let menu = plan(settings)?;
    let op_c = Arc::clone(op);
    let built = py.detach(move || build(&op_c, mass.as_deref(), t, menu));
    Ok(PyRationalKrylov {
        inner: Arc::new(built.map_err(|e| from_core(&e))?),
    })
}

fn report_dict<'py>(py: Python<'py>, r: &RationalReport) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("solves", r.solves)?;
    d.set_item("krylov_dim", r.krylov_dim)?;
    d.set_item("est_error", r.est_error)?;
    d.set_item("inner_iterations", r.inner_iterations)?;
    Ok(d)
}

#[pymethods]
impl PyRationalKrylov {
    /// ``(y, report)`` with ``y = e^{-t M^{-1} A} v`` and ``report`` a dict
    /// ``{solves, krylov_dim, est_error, inner_iterations}``.
    ///
    /// Raises ``SemiflowError``: ``GridMismatch`` for a wrong length,
    /// ``ConvergenceFailed`` if the a-priori dimension for ``tol`` exceeds ``m_max``.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyDict>)> {
        catch_panic_py!({
            let src = crate::spd_resolvent_py::contiguous_vec(&v, "v")?;
            if src.len() != self.inner.n() {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("v length {} != n {}", src.len(), self.inner.n()),
                ));
            }
            let rk = Arc::clone(&self.inner);
            let result = py.detach(move || {
                let mut out = vec![0.0; src.len()];
                let rep = rk.apply_into(&src, &mut out, &mut ScratchPool::new())?;
                Ok::<_, SemiflowError>((out, rep))
            });
            let (out, rep) = result.map_err(|e| from_core(&e))?;
            Ok((out.to_pyarray(py), report_dict(py, &rep)?))
        })
    }

    /// Apply to every column of ``v_nc`` (shape ``(N, C)``); returns ``(N, C)``.
    fn apply_batched<'py>(
        &self,
        py: Python<'py>,
        v_nc: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        catch_panic_py!({
            let [n_nodes, n_cols] = validate_batched_shape(v_nc.shape(), self.inner.n())?;
            let src_cn = gather_nc_to_cn(&v_nc.as_array(), n_nodes, n_cols);
            let rk = Arc::clone(&self.inner);
            let result = py.detach(move || apply_columns(&rk, &src_cn, n_nodes));
            let dst_cn = result.map_err(|e| from_core(&e))?;
            Ok(scatter_cn_to_nc(&dst_cn, n_nodes, n_cols, py))
        })
    }

    /// Time ``t`` of the factor.
    #[getter]
    fn t(&self) -> f64 {
        self.inner.t()
    }

    /// Shift ``γ`` in use (default ``t/10``).
    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    /// Operator dimension.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }
}

/// Register `RationalKrylov`.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRationalKrylov>()
}
