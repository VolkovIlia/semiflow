//! Python bindings for `SymmetricOperator` (#13) + entry-sensitivity Fréchet.
//!
//! ## Python API
//!
//! ```python
//! op = semiflow.SymmetricOperator.from_csr(indptr, indices, data, n)
//! out = op.evolve_batched(t, V_nc, path="chebyshev", tol=1e-10, m_max=18)
//! grad = semiflow.symmetric_op_expmv_frechet(op, u0_nc, dj_nc, t=0.3, entries=[(0,1)])
//!
//! # ADR-0202: operator composition + SPD resolvent
//! op2 = op.with_diagonal(c)                     # A + diag(c), c >= 0
//! indptr, indices, data = op.to_csr()           # int64, int32, float64
//! op_hat = op.lumped_congruence(masses)         # M^{-1/2} A M^{-1/2}
//! R = op.resolvent(lam=0.0, mass=None, solver="auto", precond="ic0", tol=1e-12)
//! ```
//!
//! GIL policy: ADR-0031 three-phase (validate → `py.detach` → scatter).

#![allow(
    unsafe_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::prelude::*;
use semiflow::{
    graph_batched, graph_expmv_frechet, EntrySensitivity, Graph, KrylovPath, ScratchPool,
    SymmetricOperator,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_py::{gather_nc_to_cn, scatter_cn_to_nc, validate_batched_shape, validate_t_final},
    panic::catch_panic_py,
    spd_resolvent_py::{build_resolvent, contiguous_vec, parse_solver, PySpdResolvent},
};

/// `(indptr int64, indices int32, data float64)` as returned by `to_csr`.
type CsrArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f64>>,
);

// ---------------------------------------------------------------------------
// PySymmetricOperator pyclass
// ---------------------------------------------------------------------------

/// Validated externally-assembled symmetric positive-semidefinite sparse operator (§55).
///
/// Built from standard CSR arrays.  Feeds directly into
/// :meth:`evolve_batched` (Krylov expmv) and
/// :func:`symmetric_op_expmv_frechet` (entry-gradient VJP).
///
/// See also :func:`assemble_conservative_csr_1d` to build an operator
/// from a 1-D variable-coefficient conservative-diffusion stencil (#11).
///
/// Parameters
/// ----------
/// indptr : ndarray[int64] of shape ``(n+1,)``
///     CSR row-pointer array.
/// indices : ndarray[int32] of shape ``(nnz,)``
///     CSR column-index array.
/// data : ndarray[float64] of shape ``(nnz,)``
///     Non-zero values.  Matrix must be symmetric; diagonal entries ≥ 0.
/// n : int
///     Operator dimension.
/// sym_tol : float
///     Symmetry tolerance (default ``1e-10``).
#[pyclass(name = "SymmetricOperator")]
pub struct PySymmetricOperator {
    pub(crate) op: Arc<SymmetricOperator<f64>>,
}

#[pymethods]
impl PySymmetricOperator {
    /// Build from CSR arrays.  Validates finiteness, diagonal ≥ 0, symmetry.
    ///
    /// Raises ``SemiflowError(OutOfDomain)`` on violations.
    #[staticmethod]
    #[pyo3(signature = (indptr, indices, data, n, sym_tol = 1e-10_f64))]
    fn from_csr(
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i32>,
        data: PyReadonlyArray1<'_, f64>,
        n: usize,
        sym_tol: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let row_ptr = csr_row_ptr(&indptr)?;
            let col_idx = csr_col_idx(&indices)?;
            let vals = csr_vals(&data)?;
            let op = SymmetricOperator::from_csr(n, &row_ptr, &col_idx, &vals, sym_tol)
                .map_err(|e| from_core(&e))?;
            Ok(Self { op: Arc::new(op) })
        })
    }

    /// Apply ``e^{-t A}`` to batched input ``v_nc`` (shape ``[N, C]``).
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time ``t ≥ 0``.
    /// v_nc : ndarray[float64, shape (N, C)]
    ///     Input vectors.
    /// path : str
    ///     ``"chebyshev"`` (default), ``"lanczos"``, or ``"implicit"`` (§59).
    /// tol : float
    ///     Krylov accuracy (default ``1e-10``).
    /// m_max : int
    ///     Max Krylov dimension (Lanczos only; default ``18``).
    /// n_steps : int
    ///     Backward-Euler sub-steps (implicit path only; default ``100``).
    /// cg_max_iter : int or None
    ///     CG iteration cap per sub-step (implicit path only; default ``None``).
    ///     ``None`` uses the theory-derived bound ``ceil(√κ·ln(2/tol))`` (§59.4).
    ///     Pass an explicit integer to override, e.g. when the Gershgorin bound
    ///     is loose and fewer iterations are known to suffice.
    ///
    /// Returns ndarray[float64, shape (N, C)].
    #[pyo3(signature = (t, v_nc, path = "chebyshev", tol = 1e-10_f64, m_max = 18_u32, n_steps = 100_usize, cg_max_iter = None))]
    fn evolve_batched<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        v_nc: PyReadonlyArray2<'py, f64>,
        path: &str,
        tol: f64,
        m_max: u32,
        n_steps: usize,
        cg_max_iter: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        catch_panic_py!({
            validate_t_final(t)?;
            let kpath = krylov_path(path, m_max, n_steps, cg_max_iter)?;
            let n = self.op.n();
            let [n_nodes, n_cols] = validate_batched_shape(v_nc.shape(), n)?;
            let src_cn = gather_nc_to_cn(&v_nc.as_array(), n_nodes, n_cols);
            let op_c = Arc::clone(&self.op);
            let dummy = no_edge_graph(n);
            let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
                let gk = op_c.krylov(kpath, tol)?;
                let mut dst_cn = vec![0.0_f64; n_nodes * n_cols];
                graph_batched::evolve_batched(&gk, &dummy, t, 1, &src_cn, &mut dst_cn)?;
                Ok(dst_cn)
            });
            let dst_cn = result.map_err(|e| from_core(&e))?;
            Ok(scatter_cn_to_nc(&dst_cn, n_nodes, n_cols, py))
        })
    }

    /// ``A + diag(c)`` (ADR-0202 D2). Inserts structurally missing diagonal entries.
    ///
    /// Raises ``SemiflowError``: ``OutOfDomain`` for ``c_i < 0``, ``NanInf`` for a
    /// non-finite entry, ``GridMismatch`` for a wrong length.
    fn with_diagonal(&self, py: Python<'_>, c: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        catch_panic_py!({
            let c_vec = contiguous_vec(&c, "c")?;
            let n = self.op.n();
            if c_vec.len() != n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("c length {} != n {}", c_vec.len(), n),
                ));
            }
            let op_c = Arc::clone(&self.op);
            let result = py.detach(move || op_c.with_diagonal(&c_vec));
            Ok(Self {
                op: Arc::new(result.map_err(|e| from_core(&e))?),
            })
        })
    }

    /// CSR arrays ``(indptr int64, indices int32, data float64)``; ``from_csr`` round-trips.
    fn to_csr<'py>(&self, py: Python<'py>) -> PyResult<CsrArrays<'py>> {
        let (row_ptr, col_idx, vals) = self.op.csr();
        let too_big = || {
            new_pyerr(
                "OutOfDomain",
                "CSR index does not fit the int64/int32 export",
            )
        };
        let indptr = row_ptr
            .iter()
            .map(|&v| i64::try_from(v))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| too_big())?;
        let indices = col_idx
            .iter()
            .map(|&v| i32::try_from(v))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| too_big())?;
        Ok((
            indptr.as_slice().to_pyarray(py),
            indices.as_slice().to_pyarray(py),
            vals.to_pyarray(py),
        ))
    }

    /// Lumped congruence ``M^{-1/2} A M^{-1/2}`` for ``M = diag(masses)`` (§55.3).
    fn lumped_congruence(
        &self,
        py: Python<'_>,
        masses: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let m_vec = contiguous_vec(&masses, "masses")?;
            let n = self.op.n();
            if m_vec.len() != n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("masses length {} != n {}", m_vec.len(), n),
                ));
            }
            let op_c = Arc::clone(&self.op);
            let result = py.detach(move || op_c.lumped_congruence(&m_vec));
            Ok(Self {
                op: Arc::new(result.map_err(|e| from_core(&e))?),
            })
        })
    }

    /// Factor ``(λM + A)`` once for repeated solves (ADR-0202, §62).
    ///
    /// Parameters
    /// ----------
    /// lam : float
    ///     Shift ``λ ≥ 0``. ``λ = 0`` requires ``A`` positive definite.
    /// mass : ndarray[float64, shape (n,)] or None
    ///     Diagonal mass ``M > 0`` (default identity).
    /// solver : str
    ///     ``"auto"`` (default), ``"tridiagonal"`` or ``"pcg"``.
    /// precond : str
    ///     ``"ic0"`` (default) or ``"jacobi"``; used on the PCG path only.
    /// tol : float
    ///     PCG relative-residual target in ``(0, 1)``; ignored by the direct path.
    /// max_iter : int or None
    ///     PCG iteration cap (``None`` → ``2n + 16``).
    ///
    /// Returns :class:`SpdResolvent`.
    #[pyo3(signature = (lam = 0.0_f64, mass = None, solver = "auto", precond = "ic0", tol = 1e-12_f64, max_iter = None))]
    fn resolvent(
        &self,
        py: Python<'_>,
        lam: f64,
        mass: Option<PyReadonlyArray1<'_, f64>>,
        solver: &str,
        precond: &str,
        tol: f64,
        max_iter: Option<usize>,
    ) -> PyResult<PySpdResolvent> {
        catch_panic_py!({
            let kind = parse_solver(solver, precond, max_iter)?;
            let mass_vec = mass.map(|m| contiguous_vec(&m, "mass")).transpose()?;
            if let Some(m) = &mass_vec {
                if m.len() != self.op.n() {
                    return Err(new_pyerr(
                        "GridMismatch",
                        &format!("mass length {} != n {}", m.len(), self.op.n()),
                    ));
                }
            }
            build_resolvent(py, &self.op, lam, mass_vec, kind, tol)
        })
    }

    /// Operator dimension.
    fn n(&self) -> usize {
        self.op.n()
    }

    /// Gershgorin spectral-radius upper bound ``ρ̄ ≥ ρ(A)``.
    fn lambda_max_bound(&self) -> f64 {
        self.op.lambda_max_bound()
    }
}

// ---------------------------------------------------------------------------
// symmetric_op_expmv_frechet — entry-sensitivity Fréchet VJP
// ---------------------------------------------------------------------------

/// Entry-sensitivity VJP ``∂J/∂A_{ij}`` via Fréchet–Duhamel (§55.5, ADR-0186).
///
/// Uses ``EntrySensitivity`` stencil within the ``graph_expmv_frechet``
/// Gauss-Legendre 8-point quadrature.
///
/// Parameters
/// ----------
/// op : SymmetricOperator
///     The assembled operator.
/// u0 : ndarray[float64, shape (N, C)]
///     Initial conditions.
/// dj : ndarray[float64, shape (N, C)]
///     Loss-gradient ``∂J/∂u_final``.
/// t : float
///     Evolution time ``t > 0``.
/// entries : list[tuple[int, int]]
///     ``(i, j)`` pairs with ``i ≤ j`` (one parameter per pair).
///
/// Returns ndarray[float64] of length ``len(entries)``.
#[pyfunction]
#[pyo3(name = "symmetric_op_expmv_frechet", signature = (op, u0, dj, *, t, entries))]
pub fn symmetric_op_expmv_frechet_py<'py>(
    py: Python<'py>,
    op: &PySymmetricOperator,
    u0: PyReadonlyArray2<'py, f64>,
    dj: PyReadonlyArray2<'py, f64>,
    t: f64,
    entries: Vec<(usize, usize)>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    catch_panic_py!({
        if !t.is_finite() || t <= 0.0 {
            return Err(new_pyerr("OutOfDomain", "t must be finite and positive"));
        }
        let n = op.op.n();
        let [rows, n_cols] = validate_batched_shape(u0.shape(), n)?;
        let [r2, c2] = validate_batched_shape(dj.shape(), n)?;
        if r2 != rows || c2 != n_cols {
            return Err(new_pyerr("OutOfDomain", "u0 and dj shapes must match"));
        }
        check_entries(&entries, n)?;
        let n_params = entries.len();
        let u0_cn = gather_nc_to_cn(&u0.as_array(), rows, n_cols);
        let dj_cn = gather_nc_to_cn(&dj.as_array(), rows, n_cols);
        let op_c = Arc::clone(&op.op);
        let result: Result<Vec<f64>, semiflow::SemiflowError> = py.detach(move || {
            let gk = op_c.krylov(KrylovPath::Chebyshev, 1e-12)?;
            let sens = EntrySensitivity {
                entries,
                n_nodes: n,
            };
            let mut grad = vec![0.0_f64; n_params];
            let mut scratch = ScratchPool::new();
            graph_expmv_frechet(
                &gk,
                &u0_cn,
                &dj_cn,
                n_cols,
                t,
                &sens,
                &mut grad,
                &mut scratch,
            )?;
            Ok(grad)
        });
        let out = result.map_err(|e| from_core(&e))?;
        Ok(out.as_slice().to_pyarray(py))
    })
}

// ---------------------------------------------------------------------------
// Helpers (private)
// ---------------------------------------------------------------------------

pub(crate) fn csr_row_ptr(arr: &PyReadonlyArray1<'_, i64>) -> PyResult<Vec<usize>> {
    let sl = arr
        .as_slice()
        .map_err(|_| new_pyerr("GridMismatch", "indptr must be contiguous"))?;
    Ok(sl.iter().map(|&v| v as usize).collect())
}

pub(crate) fn csr_col_idx(arr: &PyReadonlyArray1<'_, i32>) -> PyResult<Vec<u32>> {
    let sl = arr
        .as_slice()
        .map_err(|_| new_pyerr("GridMismatch", "indices must be contiguous"))?;
    Ok(sl.iter().map(|&v| v as u32).collect())
}

pub(crate) fn csr_vals(arr: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    Ok(arr
        .as_slice()
        .map_err(|_| new_pyerr("GridMismatch", "data must be contiguous"))?
        .to_vec())
}

pub(crate) fn krylov_path(
    path: &str,
    m_max: u32,
    n_steps: usize,
    cg_max_iter: Option<usize>,
) -> PyResult<KrylovPath> {
    match path {
        "chebyshev" => Ok(KrylovPath::Chebyshev),
        "lanczos" => Ok(KrylovPath::Lanczos {
            m_max: m_max as usize,
        }),
        "implicit" => Ok(KrylovPath::ImplicitEuler {
            n_steps,
            cg_max_iter,
        }),
        other => Err(new_pyerr(
            "Unsupported",
            &format!("path must be 'chebyshev', 'lanczos', or 'implicit', got '{other}'"),
        )),
    }
}

fn check_entries(entries: &[(usize, usize)], n: usize) -> PyResult<()> {
    for &(i, j) in entries {
        if i >= n || j >= n {
            return Err(new_pyerr(
                "OutOfDomain",
                &format!("entry ({i},{j}) out of range for n={n}"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn no_edge_graph(n: usize) -> Arc<Graph<f64>> {
    Arc::new(
        Graph::<f64>::from_edges(n, core::iter::empty()).expect("zero-edge graph is infallible"),
    )
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PySymmetricOperator>()?;
    m.add_function(wrap_pyfunction!(symmetric_op_expmv_frechet_py, m)?)?;
    Ok(())
}
