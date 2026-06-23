//! `#[pyclass] Heat2D` — 2-D heat equation state exposed to Python.
//!
//! Wraps `Semiflow2DStateInner` (palindromic Strang splitting on `Grid2D`)
//! and exposes an idiomatic Python API:
//!   `Heat2D(xmin, xmax, nx, ymin, ymax, ny)` — constructor
//!   `evolve(u0, tau, n_steps)` — advance `n_steps` of size `tau` (GIL released)
//!   `evolve_into(buf, tau, n_steps)` — in-place zero-copy variant (ADR-0045 Wave 5)
//!
//! The flat `nx × ny` output is row-major (x is the fast axis):
//! `values[j * nx + i] ≈ u(x_i, y_j)`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::too_many_arguments)]

use numpy::{PyArray1, PyReadwriteArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::ScratchPool;

use crate::{
    boundary::parse_boundary,
    error::{from_core, new_pyerr},
    handle::{build_heat_2d, compute_evolve_2d, compute_evolve_2d_inplace, Semiflow2DStateInner},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Heat2D Python class
// ---------------------------------------------------------------------------

/// 2-D heat equation state.
///
/// Solves `∂_t u = ∂_{xx} u + ∂_{yy} u` (unit diffusion `a = 1`) on
/// `[xmin, xmax] × [ymin, ymax]` using palindromic Strang splitting
/// (`Strang2D`).  The flat `nx × ny` output is row-major (x is the fast axis):
/// `values[j * nx + i] ≈ u(x_i, y_j)`.
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     X-axis boundary; must be finite and `xmin < xmax`.
/// nx : int
///     Number of X-axis nodes (must be ≥ 4).
/// ymin, ymax : float
///     Y-axis boundary; must be finite and `ymin < ymax`.
/// ny : int
///     Number of Y-axis nodes (must be ≥ 4).
///
/// Raises
/// ------
/// `SemiflowError`
///     `kind='GridMismatch'` — any axis has `n < 4` or `xmin >= xmax`.
#[pyclass(name = "Heat2D")]
pub struct Heat2D {
    inner: Semiflow2DStateInner,
    nx: usize,
    ny: usize,
}

#[pymethods]
impl Heat2D {
    /// Construct unit-coefficient `Heat2D` on a `Grid2D`.
    ///
    /// Parameters
    /// ----------
    /// boundary : str, optional
    ///     Boundary policy (keyword-only).  One of ``"reflect"`` (default),
    ///     ``"periodic"``, ``"zero"``, ``"linear"``.  Applied to both axes.
    #[new]
    #[pyo3(signature = (xmin, xmax, nx, ymin, ymax, ny, *, boundary = "reflect"))]
    fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let inner =
                build_heat_2d(xmin, xmax, nx, ymin, ymax, ny, policy).map_err(|e| from_core(&e))?;
            Ok(Heat2D { inner, nx, ny })
        })
    }

    /// Evolve `u0` (flat row-major, length `nx * ny`) by `n_steps` of size `tau`.
    ///
    /// The GIL is released via `py.detach` during the inner `Strang2D` compute
    /// loop (ADR-0031 three-phase pattern).
    ///
    /// Parameters
    /// ----------
    /// u0 : array-like[float64]
    ///     Flat initial condition, row-major, length exactly `nx * ny`.
    /// tau : float
    ///     Step size.  Must be finite and > 0.
    /// `n_steps` : int
    ///     Number of `Strang2D` steps.  Must be ≥ 1.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray[float64]
    ///     Flat row-major array, shape `(nx * ny,)`.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     `kind='GridMismatch'` — `len(u0) != nx * ny`.
    ///     `kind='OutOfDomain'` — `tau <= 0`, non-finite, or `n_steps == 0`.
    #[pyo3(signature = (u0, tau, n_steps))]
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        u0: &Bound<'_, PyAny>,
        tau: f64,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            // --- Phase 1: validate + extract (under GIL) ---
            validate_evolve_2d_params(tau, n_steps)?;
            let input: Vec<f64> = u0.extract::<Vec<f64>>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "u0 must be a numpy.ndarray[float64] or sequence of floats",
                )
            })?;
            let expected = self.nx * self.ny;
            if input.len() != expected {
                return Err(new_pyerr(
                    "GridMismatch",
                    &format!("u0 length {} != nx*ny={}", input.len(), expected),
                ));
            }
            let strang = self.inner.strang.clone();
            let grid = self.inner.grid;

            // --- Phase 2: pure-Rust compute (GIL released) ---
            let result: Result<Vec<f64>, _> =
                py.detach(|| compute_evolve_2d(&strang, grid, input, tau, n_steps));
            let result = result.map_err(|e| from_core(&e))?;

            // --- Phase 3: return as numpy array (under GIL) ---
            let arr = result.as_slice().to_pyarray(py);
            Ok(arr)
        })
    }

    /// Evolve `buf` (flat row-major, length `nx * ny`) in place by `n_steps` of
    /// size `tau`.
    ///
    /// Zero-copy path (ADR-0045 Wave 5): when `buf` is a contiguous C-order
    /// `float64` array of the right length, the result is written directly back
    /// into `buf` without a second allocation. A non-contiguous or wrong-length
    /// buffer falls back to an allocation-plus-copy path and emits a
    /// `tracing::warn!` at target `"semiflow::zero_copy"`.
    ///
    /// Parameters
    /// ----------
    /// buf : numpy.ndarray[float64], writable
    ///     Flat row-major, length exactly `nx * ny`. **Modified in place.**
    /// tau : float
    ///     Step size. Must be finite and > 0.
    /// `n_steps` : int
    ///     Number of `Strang2D` steps. Must be ≥ 1.
    ///
    /// Returns
    /// -------
    /// None
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     `kind='GridMismatch'` — buf length != nx * ny.
    ///     `kind='OutOfDomain'` — tau <= 0, non-finite, or `n_steps` == 0.
    #[pyo3(signature = (buf, tau, n_steps))]
    fn evolve_into<'py>(
        &self,
        py: Python<'py>,
        mut buf: PyReadwriteArray1<'py, f64>,
        tau: f64,
        n_steps: usize,
    ) -> PyResult<()> {
        catch_panic_py!({
            validate_evolve_2d_params(tau, n_steps)?;
            let expected_len = self.nx * self.ny;
            match buf.as_slice_mut() {
                Ok(slice) if slice.len() == expected_len => {
                    // --- Zero-copy happy path ---
                    let strang = self.inner.strang.clone();
                    let grid = self.inner.grid;
                    // Cast to usize to make the closure Send + Sync.
                    // SAFETY: py.detach releases GIL; the buffer backing `slice`
                    // is owned by `buf` which lives for 'py.  No aliasing during
                    // the detach window (GIL is off, Python cannot mutate memory).
                    let raw_addr: usize = slice.as_mut_ptr() as usize;
                    let raw_len = slice.len();
                    let result = py.detach(|| {
                        // SAFETY: addr came from a valid &mut [f64]; len unchanged.
                        let s = unsafe {
                            std::slice::from_raw_parts_mut(raw_addr as *mut f64, raw_len)
                        };
                        let mut scratch = ScratchPool::<f64>::new();
                        compute_evolve_2d_inplace(&strang, grid, s, tau, n_steps, &mut scratch)
                    });
                    result.map_err(|e| from_core(&e))?;
                    Ok(())
                }
                _ => {
                    // --- Copy fallback ---
                    tracing::warn!(
                        target: "semiflow::zero_copy",
                        expected_len,
                        "Heat2D::evolve_into falling back to copy mode"
                    );
                    self.evolve_into_copy_fallback(py, buf, tau, n_steps)
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Heat2D::evolve_into copy fallback (non-public helper, not a pymethod)
// ---------------------------------------------------------------------------

impl Heat2D {
    fn evolve_into_copy_fallback<'py>(
        &self,
        py: Python<'py>,
        mut buf: PyReadwriteArray1<'py, f64>,
        tau: f64,
        n_steps: usize,
    ) -> PyResult<()> {
        let owned: Vec<f64> = buf.as_array().to_vec();
        let strang = self.inner.strang.clone();
        let grid = self.inner.grid;
        let result: Result<Vec<f64>, _> =
            py.detach(|| compute_evolve_2d(&strang, grid, owned, tau, n_steps));
        let result = result.map_err(|e| from_core(&e))?;
        let mut view = buf.as_array_mut();
        let dst_slice = view.as_slice_mut().ok_or_else(|| {
            new_pyerr(
                "OutOfDomain",
                "destination buffer is not contiguous; cannot write back",
            )
        })?;
        if dst_slice.len() != result.len() {
            return Err(new_pyerr(
                "GridMismatch",
                "result length != buffer length after fallback",
            ));
        }
        dst_slice.copy_from_slice(&result);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Heat2D input validation helper
// ---------------------------------------------------------------------------

/// Validate `tau` and `n_steps` for `Heat2D::evolve` and `Heat2D::evolve_into`.
fn validate_evolve_2d_params(tau: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "tau must be finite and > 0"));
    }
    Ok(())
}
