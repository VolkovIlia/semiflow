//! Python bindings for the SPD resolvent / steady solve (ADR-0202, §62).
//!
//! ## Python API
//!
//! ```python
//! R = op.resolvent(lam=0.0, mass=None, solver="auto", precond="ic0",
//!                  tol=1e-12, max_iter=None)      # SymmetricOperator.resolvent
//! x = R.solve(b)                                  # (n,) -> (n,)
//! X = R.solve_batched(B)                          # (n, nc) -> (n, nc)
//! x, iters, rel_res = R.solve_info(b)
//! R.method   # "tridiagonal" | "pcg-ic0" | "pcg-jacobi"
//! R.n
//! ```
//!
//! GIL policy: ADR-0031 three-phase (validate → `py.detach` → scatter). No Python
//! callbacks cross the boundary (ADR-0189 D3).

#![allow(clippy::doc_markdown, clippy::needless_pass_by_value)]

use std::sync::Arc;

use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::prelude::*;
use semiflow::{
    Precond, ResolventMethod, ScratchPool, SemiflowError as CoreError, SpdResolvent, SpdSolver,
    SymmetricOperator,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_py::{gather_nc_to_cn, scatter_cn_to_nc, validate_batched_shape},
    panic::catch_panic_py,
};

/// Solver-menu entry point (`solver`, `precond`, `max_iter`) shared by `resolvent`.
pub(crate) fn parse_solver(
    solver: &str,
    precond: &str,
    max_iter: Option<usize>,
) -> PyResult<SpdSolver> {
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
    match solver {
        "auto" => Ok(SpdSolver::Auto),
        "tridiagonal" => Ok(SpdSolver::Tridiagonal),
        "pcg" => Ok(SpdSolver::Pcg { precond, max_iter }),
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!("solver must be 'auto', 'tridiagonal' or 'pcg', got '{other}'"),
        )),
    }
}

/// Contiguous float64 vector → owned `Vec` (the GIL-boundary copy).
pub(crate) fn contiguous_vec(arr: &PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<Vec<f64>> {
    arr.as_slice().map(<[f64]>::to_vec).map_err(|_| {
        new_pyerr(
            "GridMismatch",
            &format!("{name} must be contiguous float64"),
        )
    })
}

/// Build the resolvent inside `py.detach` (factorisation is `O(n)`–`O(nnz)`).
pub(crate) fn build_resolvent(
    py: Python<'_>,
    op: &Arc<SymmetricOperator<f64>>,
    lam: f64,
    mass: Option<Vec<f64>>,
    solver: SpdSolver,
    tol: f64,
) -> PyResult<PySpdResolvent> {
    let op_c = Arc::clone(op);
    let built: Result<SpdResolvent<f64>, CoreError> =
        py.detach(move || SpdResolvent::new(&op_c, lam, mass.as_deref(), solver, tol));
    let inner = built.map_err(|e| from_core(&e))?;
    Ok(PySpdResolvent {
        inner: Arc::new(inner),
    })
}

/// Factor-once / solve-many resolvent ``x = (λM + A)⁻¹ b`` of an SPD operator (§62).
///
/// Build with :meth:`SymmetricOperator.resolvent`. ``λ ≥ 0``, ``M = diag(mass) > 0``
/// (default identity). Tridiagonal operators use an exact ``O(n)`` LDLᵀ; everything
/// else uses PCG with IC(0) (or Jacobi). The factor is reused by every solve.
#[pyclass(name = "SpdResolvent", frozen)]
pub struct PySpdResolvent {
    inner: Arc<SpdResolvent<f64>>,
}

#[pymethods]
impl PySpdResolvent {
    /// Solve ``(λM + A) x = b``. Returns ndarray[float64, shape (n,)].
    ///
    /// Raises ``SemiflowError``: ``GridMismatch`` (wrong length / non-contiguous),
    /// ``NanInf`` (non-finite ``b``), ``ConvergenceFailed`` (PCG cap or breakdown).
    fn solve<'py>(
        &self,
        py: Python<'py>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x, _, _) = self.solve_one(py, &b)?;
        Ok(x.as_slice().to_pyarray(py))
    }

    /// Solve for ``nc`` right-hand sides at once. ``B`` is ``(n, nc)``; result too.
    fn solve_batched<'py>(
        &self,
        py: Python<'py>,
        b: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        catch_panic_py!({
            let n = self.inner.n();
            let [rows, cols] = validate_batched_shape(b.shape(), n)?;
            let src_cn = gather_nc_to_cn(&b.as_array(), rows, cols);
            let res = Arc::clone(&self.inner);
            let result: Result<Vec<f64>, CoreError> = py.detach(move || {
                let mut dst_cn = vec![0.0_f64; rows * cols];
                let mut scratch = ScratchPool::new();
                for c in 0..cols {
                    let span = c * rows..(c + 1) * rows;
                    res.solve_into(&src_cn[span.clone()], &mut dst_cn[span], &mut scratch)?;
                }
                Ok(dst_cn)
            });
            let dst_cn = result.map_err(|e| from_core(&e))?;
            Ok(scatter_cn_to_nc(&dst_cn, rows, cols, py))
        })
    }

    /// Solve and report ``(x, iterations, rel_residual)``.
    ///
    /// ``iterations`` is 0 on the direct path. ``rel_residual`` is the TRUE
    /// ``‖b − (λM+A)x‖₂ / ‖b‖₂``; on PCG it may exceed ``tol`` near the ``ε·κ`` floor.
    fn solve_info<'py>(
        &self,
        py: Python<'py>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, f64)> {
        let (x, iters, rel) = self.solve_one(py, &b)?;
        Ok((x.as_slice().to_pyarray(py), iters, rel))
    }

    /// Method in use: ``"tridiagonal"``, ``"pcg-ic0"`` or ``"pcg-jacobi"``.
    #[getter]
    fn method(&self) -> &'static str {
        match self.inner.method() {
            ResolventMethod::Tridiagonal => "tridiagonal",
            ResolventMethod::PcgIc0 => "pcg-ic0",
            ResolventMethod::PcgJacobi => "pcg-jacobi",
        }
    }

    /// Operator dimension.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpdResolvent(n={}, method='{}')",
            self.inner.n(),
            self.method()
        )
    }
}

impl PySpdResolvent {
    /// Validate → `py.detach` → return `(x, iterations, rel_residual)`.
    fn solve_one(
        &self,
        py: Python<'_>,
        b: &PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(Vec<f64>, usize, f64)> {
        catch_panic_py!({
            let n = self.inner.n();
            let b_vec = contiguous_vec(b, "b")?;
            if b_vec.len() != n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("b length {} != n {}", b_vec.len(), n),
                ));
            }
            let res = Arc::clone(&self.inner);
            let result: Result<(Vec<f64>, usize, f64), CoreError> = py.detach(move || {
                let mut x = vec![0.0_f64; n];
                let report = res.solve_into(&b_vec, &mut x, &mut ScratchPool::new())?;
                Ok((x, report.iterations, report.rel_residual))
            });
            result.map_err(|e| from_core(&e))
        })
    }
}

/// Register the pyclass on the module.
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpdResolvent>()?;
    Ok(())
}
