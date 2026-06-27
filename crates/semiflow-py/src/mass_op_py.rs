//! Python bindings for lumped + consistent mass operators (#14).
//!
//! ## Python API
//!
//! ```python
//! # Diagonal / lumped-mass fast path
//! out = semiflow.mass_lumped_evolve(k_op, m_diag, t=0.3, v_nc=V,
//!                                    path="chebyshev", tol=1e-10)
//!
//! # Consistent-mass operator Â = R⁻ᵀ K R⁻¹ (small n, gate tests)
//! mo = semiflow.MassKOperator.from_k_and_mass(k_op, m_dense)
//! out = mo.evolve(t=0.3, v=v_1d, path="chebyshev", tol=1e-10)
//! ```
//!
//! GIL policy: ADR-0031 three-phase (validate → `py.detach` → scatter).

#![allow(
    unsafe_code,
    clippy::doc_markdown,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
)]

use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    graph_batched, MassKOperator, ScratchPool, SymmetricLinearOp, TriangularFactor,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_py::{gather_nc_to_cn, scatter_cn_to_nc, validate_batched_shape, validate_t_final},
    panic::catch_panic_py,
    symmetric_op_py::{krylov_path, no_edge_graph, PySymmetricOperator},
};

// ---------------------------------------------------------------------------
// mass_lumped_evolve — free function, fast diagonal-mass path (§55.3)
// ---------------------------------------------------------------------------

/// Evolve ``e^{-t M⁻¹ K} V`` for diagonal mass matrix ``M = diag(m_diag)`` (§55.3).
///
/// Maps to the congruence ``Â = D^{-½} K D^{-½}`` and a Krylov solve on ``Â``,
/// with pre/post diagonal scaling.  Cheaper than building a full :class:`MassKOperator`
/// for the lumped-mass case.
///
/// Parameters
/// ----------
/// k_op : SymmetricOperator
///     Stiffness operator ``K``.
/// m_diag : ndarray[float64, shape (n,)]
///     Positive diagonal mass entries.
/// t : float
///     Evolution time ``t ≥ 0``.
/// v_nc : ndarray[float64, shape (N, C)]
///     Input vectors; ``N == k_op.n()``.
/// path : str
///     ``"chebyshev"`` (default) or ``"lanczos"``.
/// tol : float
///     Krylov accuracy (default ``1e-10``).
/// m_max : int
///     Max Krylov dimension (Lanczos only; default ``18``).
///
/// Returns ndarray[float64, shape (N, C)].
#[pyfunction]
#[pyo3(name = "mass_lumped_evolve",
       signature = (k_op, m_diag, t, v_nc, path = "chebyshev", tol = 1e-10_f64, m_max = 18_u32))]
pub fn mass_lumped_evolve_py<'py>(
    py: Python<'py>,
    k_op: &PySymmetricOperator,
    m_diag: PyReadonlyArray1<'py, f64>,
    t: f64,
    v_nc: PyReadonlyArray2<'py, f64>,
    path: &str,
    tol: f64,
    m_max: u32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    catch_panic_py!({
        validate_t_final(t)?;
        let kpath = krylov_path(path, m_max)?;
        let n = k_op.op.n();
        let [n_nodes, n_cols] = validate_batched_shape(v_nc.shape(), n)?;
        let masses = extract_masses(&m_diag, n)?;
        let src_nc = gather_nc_to_cn(&v_nc.as_array(), n_nodes, n_cols);
        let op_c = Arc::clone(&k_op.op);
        let dummy = no_edge_graph(n);
        let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
            lumped_inner(&op_c, &masses, t, kpath, tol, &src_nc, &dummy, n_nodes, n_cols)
        });
        let dst_cn = result.map_err(|e| from_core(&e))?;
        Ok(scatter_cn_to_nc(&dst_cn, n_nodes, n_cols, py))
    })
}

/// Extract and validate the diagonal mass array.
fn extract_masses(m_diag: &PyReadonlyArray1<'_, f64>, n: usize) -> PyResult<Vec<f64>> {
    let masses: Vec<f64> = m_diag
        .as_slice()
        .map_err(|_| new_pyerr("GridMismatch", "m_diag must be contiguous"))?
        .to_vec();
    if masses.len() != n {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("m_diag length {} != k_op.n() {}", masses.len(), n),
        ));
    }
    for (i, &m) in masses.iter().enumerate() {
        if !m.is_finite() || m <= 0.0 {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("m_diag[{i}] = {m} is not finite-positive"),
            ));
        }
    }
    Ok(masses)
}

/// Pure-Rust batched lumped-mass evolution (runs outside GIL).
fn lumped_inner(
    op: &semiflow::SymmetricOperator<f64>,
    masses: &[f64],
    t: f64,
    kpath: semiflow::KrylovPath,
    tol: f64,
    src_cn: &[f64],
    dummy: &Arc<semiflow::Graph<f64>>,
    n_nodes: usize,
    n_cols: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let a_hat = op.lumped_congruence(masses)?;
    let gk = a_hat.krylov(kpath, tol)?;
    let mut w_cn = vec![0.0_f64; n_nodes * n_cols];
    for c in 0..n_cols {
        for i in 0..n_nodes {
            w_cn[c * n_nodes + i] = src_cn[c * n_nodes + i] * masses[i].sqrt();
        }
    }
    let mut out_cn = vec![0.0_f64; n_nodes * n_cols];
    graph_batched::evolve_batched(&gk, dummy, t, 1, &w_cn, &mut out_cn)?;
    for c in 0..n_cols {
        for i in 0..n_nodes {
            out_cn[c * n_nodes + i] /= masses[i].sqrt();
        }
    }
    Ok(out_cn)
}

// ---------------------------------------------------------------------------
// PyMassKOperator — consistent-mass operator Â = R⁻ᵀ K R⁻¹ (§55.4)
// ---------------------------------------------------------------------------

/// Consistent-mass operator ``Â = R⁻ᵀ K R⁻¹`` where ``M = Rᵀ R`` (§55.4).
///
/// Suitable for generalized eigenvalue problems with a consistent (non-diagonal)
/// mass matrix ``M``.  For large ``n``, prefer :func:`mass_lumped_evolve`.
///
/// Build with :meth:`from_k_and_mass`.
#[pyclass(name = "MassKOperator")]
pub struct PyMassKOperator {
    op: Arc<MassKOperator<f64>>,
}

#[pymethods]
impl PyMassKOperator {
    /// Build from stiffness operator ``K`` and dense mass matrix ``M``.
    ///
    /// Parameters
    /// ----------
    /// k_op : SymmetricOperator
    ///     Stiffness operator ``K`` (symmetric PSD).
    /// m_dense : ndarray[float64, shape (n*n,)]
    ///     Dense symmetric positive-definite mass matrix ``M`` in row-major order.
    ///     Length must be ``n*n`` where ``n = k_op.n()``.
    ///
    /// Raises ``SemiflowError(OutOfDomain)`` if Cholesky factorization fails
    /// (i.e. M is not positive-definite).
    #[staticmethod]
    fn from_k_and_mass(
        k_op: &PySymmetricOperator,
        m_dense: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let n = k_op.op.n();
            let flat = m_dense
                .as_slice()
                .map_err(|_| new_pyerr("GridMismatch", "m_dense must be contiguous"))?
                .to_vec();
            if flat.len() != n * n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("m_dense length {} != n*n = {}*{} = {}", flat.len(), n, n, n * n),
                ));
            }
            let r = TriangularFactor::dense_cholesky_spd(&flat, n).map_err(|e| from_core(&e))?;
            let k_owned = k_op.op.as_ref().clone();
            let op = MassKOperator::new(k_owned, r);
            Ok(Self { op: Arc::new(op) })
        })
    }

    /// Apply ``e^{-t M⁻¹ K}`` to a single vector ``v`` (shape ``(n,)``).
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Evolution time ``t ≥ 0``.
    /// v : ndarray[float64, shape (n,)]
    ///     Input vector.
    /// path : str
    ///     ``"chebyshev"`` (default) or ``"lanczos"``.
    /// tol : float
    ///     Krylov accuracy (default ``1e-10``).
    /// m_max : int
    ///     Max Krylov dimension (Lanczos only; default ``18``).
    ///
    /// Returns ndarray[float64, shape (n,)].
    #[pyo3(signature = (t, v, path = "chebyshev", tol = 1e-10_f64, m_max = 18_u32))]
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        v: PyReadonlyArray1<'py, f64>,
        path: &str,
        tol: f64,
        m_max: u32,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            validate_t_final(t)?;
            let kpath = krylov_path(path, m_max)?;
            let n = SymmetricLinearOp::n(self.op.as_ref());
            let v_sl = v.as_slice()
                .map_err(|_| new_pyerr("GridMismatch", "v must be contiguous"))?
                .to_vec();
            if v_sl.len() != n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("v length {} != op.n() {}", v_sl.len(), n),
                ));
            }
            let op_c = Arc::clone(&self.op);
            let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
                let mut out = vec![0.0_f64; n];
                let mut scratch = ScratchPool::new();
                op_c.evolve(t, &v_sl, &mut out, kpath, tol, &mut scratch)?;
                Ok(out)
            });
            let out = result.map_err(|e| from_core(&e))?;
            Ok(out.as_slice().to_pyarray(py))
        })
    }

    /// Operator dimension.
    /// Operator dimension.
    fn n(&self) -> usize {
        // Trait method SymmetricLinearOp::n() — brought into scope above.
        SymmetricLinearOp::n(self.op.as_ref())
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMassKOperator>()?;
    m.add_function(wrap_pyfunction!(mass_lumped_evolve_py, m)?)?;
    Ok(())
}
