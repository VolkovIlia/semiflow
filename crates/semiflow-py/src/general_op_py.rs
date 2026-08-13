//! `GeneralOperator` — non-symmetric CSR operator action (#24, ADR-0194).
//!
//! `SymmetricOperator.from_csr` validates symmetry, so the whole Krylov surface
//! is closed to non-self-adjoint generators. This opens it via the
//! symmetry-agnostic scaled-truncated-Taylor engine.
//!
//! **There is deliberately no `path=` argument.** Chebyshev needs a real
//! spectrum in `[0, λ_max]` and is not fixable; Lanczos is structurally
//! symmetric-only (a 3-term recurrence with no reorthogonalisation, projecting
//! onto a force-symmetrised tridiagonal). Encoding that as an absent parameter
//! makes reaching those paths an `AttributeError`, not a runtime tolerance
//! failure.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value)]

use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use semiflow::general_operator::GeneralOperator;

use crate::{
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

/// Externally-assembled **possibly non-symmetric** sparse operator.
///
/// Accepts any real CSR with finite entries: no symmetry check, no diagonal-sign
/// check. Opens `e^{-tA}v` to drifted Fokker-Planck
/// ``∂_t p = ∂_x(D ∂_x p) − ∂_x(μ p)`` and to inventory-ladder generators, both
/// of which `SymmetricOperator` rejects.
///
/// Honest limits
/// -------------
/// - Cost is ``Θ(t·‖A‖_∞)`` matvecs — **linear in the depth, not flat**. The
///   depth-independence `GraphKrylov` has does not transfer to this path.
/// - Only the **backward** error is certified. For a severely non-normal ``A``
///   the forward error can exceed the backward radius by ``κ(V)``, which is not
///   estimated.
/// - No `path=` argument: Chebyshev and Lanczos are structurally unavailable.
/// - No conservation claim. Discrete mass conservation for a Fokker-Planck
///   assembly holds only if the caller's ``A`` has exactly zero column sums.
#[pyclass(name = "GeneralOperator")]
pub struct PyGeneralOperator {
    inner: GeneralOperator<f64>,
}

#[pymethods]
impl PyGeneralOperator {
    /// Build from CSR arrays ``(indptr, indices, data)`` for an ``n x n`` matrix.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='GridMismatch'`` on a malformed CSR; ``kind='NanInf'`` on a
    ///     non-finite entry.
    #[staticmethod]
    #[pyo3(signature = (n, indptr, indices, data))]
    fn from_csr(
        n: usize,
        indptr: Vec<usize>,
        indices: Vec<u32>,
        data: Vec<f64>,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let inner = GeneralOperator::<f64>::from_csr(n, &indptr, &indices, &data)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Operator dimension.
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Row-sum bound ``‖A‖_∞``.
    ///
    /// A **norm** bound, not a spectral interval — the name is deliberate.
    fn norm_inf_bound(&self) -> f64 {
        self.inner.norm_inf_bound()
    }

    /// ``e^{-tA}·v`` for `C` right-hand sides. ``v_nc`` is ``[N, C]``; result too.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='GridMismatch'`` if ``v_nc`` is not 2-D with ``shape[0] == n``;
    ///     ``kind='OutOfDomain'`` if ``t`` is negative or non-finite.
    #[pyo3(signature = (t, v_nc))]
    fn evolve_batched<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        v_nc: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        catch_panic_py!({
            if !t.is_finite() || t < 0.0 {
                return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
            }
            let view = v_nc.as_array();
            let n = self.inner.n();
            if view.shape()[0] != n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("v_nc has shape[0]={}, expected n={n}", view.shape()[0]),
                ));
            }
            let n_cols = view.shape()[1];
            let src_cn = crate::graph_py::gather_nc_to_cn(&view, n, n_cols);
            let kernel = self.inner.expmv();
            let mut dst_cn = vec![0.0_f64; n * n_cols];
            let result: Result<(), semiflow::SemiflowError> = py.detach(|| {
                for c in 0..n_cols {
                    kernel.action_into_slice(
                        t,
                        &src_cn[c * n..(c + 1) * n],
                        &mut dst_cn[c * n..(c + 1) * n],
                    )?;
                }
                Ok(())
            });
            result.map_err(|e| from_core(&e))?;
            Ok(crate::graph_py::scatter_cn_to_nc(&dst_cn, n, n_cols, py))
        })
    }

    /// ``Aᵀ·v`` — a real transpose, not the self-adjoint shortcut.
    fn apply_transpose<'py>(
        &self,
        py: Python<'py>,
        v: Vec<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        catch_panic_py!({
            let n = self.inner.n();
            if v.len() != n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("v has length {}, expected n={n}", v.len()),
                ));
            }
            let mut out = vec![0.0_f64; n];
            self.inner.apply_transpose_into_slice(&v, &mut out);
            Ok(out.as_slice().to_pyarray(py))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneralOperator(n={}, norm_inf={:.4e})",
            self.inner.n(),
            self.inner.norm_inf_bound()
        )
    }
}

/// Register the pyclass on the module.
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGeneralOperator>()?;
    Ok(())
}
