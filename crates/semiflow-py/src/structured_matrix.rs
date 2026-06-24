//! Wave P6 — `MatrixDiffusion1D` (M17).
//!
//! Coupled 2-component 1D diffusion via `MatrixDiffusionChernoff<f64, 2>`.
//! Split from `structured.rs` for suckless file-size compliance.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value, clippy::unused_self)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    matrix_system::{MatrixDiffusionChernoff, MatrixGridFn1D},
    ChernoffSemigroup, Grid1D,
};

use crate::{
    error::{from_core, new_pyerr},
    graph_py::extract_f64_vec,
    panic::catch_panic_py,
};

// ===========================================================================
// MatrixDiffusion1D — coupled 2-component 1D diffusion (M17)
// ===========================================================================

/// Coupled 2-component 1D diffusion (M17, ADR-0082, math §33).
///
/// Solves ``∂_t u_i = Σ_j a_ij(x) ∂²u_j + Σ_j b_ij(x) ∂u_j + Σ_j c_ij(x) u_j``
/// for ``u ∈ ℝ²``.  Three-phase palindromic Strang:
/// ``R(τ/2) ∘ D(τ) ∘ R(τ/2)``.
///
/// This binding specialises to M=2 (const-generic crossing over `PyO3` boundary).
///
/// State layout: flat float64 array of length ``2*n``; component ``i`` at grid
/// point ``k`` is at index ``k*2+i`` (row-major, component inner).
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     Grid boundaries (must be finite and xmin < xmax).
/// n : int
///     Number of grid nodes (must be >= 5 per ADR-0082).
/// `a_diag` : float, optional
///     Diagonal diffusion coefficient ``a_00 = a_11`` (default 1.0, off-diagonal = 0).
///     Must be > 0.
/// `c_coupling` : float, optional
///     Off-diagonal reaction ``c_01 = c_10`` (default 0.0, symmetric coupling).
/// u0 : array-like
///     Initial condition; float64 array of length ``2*n``  (row-major).
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if grid params invalid or u0 length mismatch.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if n < 5 or `a_diag` <= 0.
#[pyclass(name = "MatrixDiffusion1D")]
pub struct PyMatrixDiffusion1D {
    grid: Grid1D<f64>,
    a_diag: f64,
    c_coupling: f64,
    current: MatrixGridFn1D<f64, 2>,
    n: usize,
}

#[pymethods]
impl PyMatrixDiffusion1D {
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, a_diag = 1.0_f64, c_coupling = 0.0_f64))]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        a_diag: f64,
        c_coupling: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            if n < 5 {
                return Err(new_pyerr(
                    "OutOfDomain",
                    "n must be >= 5 (block-CN stencil)",
                ));
            }
            if !a_diag.is_finite() || a_diag <= 0.0 {
                return Err(new_pyerr("OutOfDomain", "a_diag must be finite and > 0"));
            }
            let vals = extract_f64_vec(u0).map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err("u0 must be a float64 array")
            })?;
            if vals.len() != 2 * n {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("u0 length {}, expected 2*n={}", vals.len(), 2 * n),
                ));
            }
            for &v in &vals {
                if !v.is_finite() {
                    return Err(new_pyerr("NanInf", "u0 contains NaN or Inf"));
                }
            }
            let grid = Grid1D::new(xmin, xmax, n).map_err(|e| from_core(&e))?;
            // Validate kernel construction at construction time.
            let _check =
                build_matrix_kernel(a_diag, c_coupling, grid).map_err(|e| from_core(&e))?;
            let mut current = MatrixGridFn1D::<f64, 2>::new(grid);
            current.values.copy_from_slice(&vals);
            Ok(Self {
                grid,
                a_diag,
                c_coupling,
                current,
                n,
            })
        })
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// GIL released during inner Rust compute (ADR-0031).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if t < 0, non-finite, or `n_steps` == 0.
    ///     kind='Unsupported' if M >= 5 (deferred to v4.x).
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            crate::structured::validate_t_nsteps(t, n_steps)?;
            let grid = self.grid;
            let a_d = self.a_diag;
            let c_c = self.c_coupling;
            let vals: Vec<f64> = self.current.values.clone();
            let result: Result<Vec<f64>, semiflow::SemiflowError> =
                py.detach(|| evolve_matrix(a_d, c_c, grid, vals, t, n_steps));
            let out = result.map_err(|e| from_core(&e))?;
            self.current.values.copy_from_slice(&out);
            Ok(())
        })
    }

    /// Return current state as flat ``numpy.ndarray[float64]`` (copy).
    ///
    /// Length is ``2 * n``.  Component ``i`` at grid point ``k`` is at
    /// index ``k*2+i`` (row-major, component-inner layout).
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.current.values.as_slice().to_pyarray(py)) })
    }

    /// Return approximation order (always 2 — palindromic Strang).
    fn order(&self) -> u32 {
        2
    }

    /// Return ``n`` (number of grid nodes).
    fn __len__(&self) -> usize {
        self.n
    }

    fn __repr__(&self) -> String {
        format!(
            "MatrixDiffusion1D(n={}, M=2, a_diag={:.3}, c_coupling={:.3}, order=2)",
            self.n, self.a_diag, self.c_coupling,
        )
    }
}

// ---------------------------------------------------------------------------
// MatrixDiffusion1D helpers
// ---------------------------------------------------------------------------

pub(crate) fn build_matrix_kernel(
    a_diag: f64,
    c_coupling: f64,
    grid: Grid1D<f64>,
) -> Result<MatrixDiffusionChernoff<f64, 2>, semiflow::SemiflowError> {
    let a_d = a_diag;
    let c_c = c_coupling;
    MatrixDiffusionChernoff::<f64, 2>::new(
        move |_x, mat| {
            mat[0][0] = a_d;
            mat[0][1] = 0.0;
            mat[1][0] = 0.0;
            mat[1][1] = a_d;
        },
        |_x, mat| {
            // zero drift
            mat[0][0] = 0.0;
            mat[0][1] = 0.0;
            mat[1][0] = 0.0;
            mat[1][1] = 0.0;
        },
        move |_x, mat| {
            mat[0][0] = 0.0;
            mat[0][1] = c_c;
            mat[1][0] = c_c;
            mat[1][1] = 0.0;
        },
        grid,
    )
}

fn evolve_matrix(
    a_diag: f64,
    c_coupling: f64,
    grid: Grid1D<f64>,
    vals: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let kernel = build_matrix_kernel(a_diag, c_coupling, grid)?;
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let mut src = MatrixGridFn1D::<f64, 2>::new(grid);
    src.values.copy_from_slice(&vals);
    let out = sg.evolve(t, &src)?;
    Ok(out.values)
}
