//! v8.1.0 `PyO3` binding for `ResolventJumpChernoff` (F2, ADR-0138, ADR-0134).
//!
//! Implements `ResolventJumpV8` — a stateless-per-call Python class that
//! evaluates `e^{tA}g` for the 1D unit-diffusion heat kernel via the TWS
//! parabolic-contour inverse Laplace quadrature (math.md §47).
//!
//! ## NARROW scope (§47.4, ADR-0134 NORMATIVE)
//!
//! Self-adjoint / sectorial generators only (diffusion family).
//! Non-self-adjoint / advection-dominated generators are OUT of scope.
//! `m_nodes >= 6` is enforced at construction.
//!
//! ## ABI-safety invariant (ADR-0138 hard constraint)
//!
//! `Complex<f64>` / TWS contour arithmetic stays sealed inside core.
//! Only `jump(t, g) -> GridFn1D<f64>` is exposed; the surface receives
//! and returns flat `f64` numpy arrays.
//!
//! ## GIL policy (ADR-0031 three-phase)
//!
//! `.jump` releases the GIL via `py.detach` around the M-node contour
//! solve (M complex Thomas sweeps — the heavy compute).
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util with semiflow-ffi/wasm.

use numpy::ToPyArray;
use pyo3::prelude::*;
use semiflow::{DiffusionChernoff, Grid1D, GridFn1D, ResolventJumpChernoff};

use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;

// ---------------------------------------------------------------------------
// Inner Rust state
// ---------------------------------------------------------------------------

/// Heap-owned kernel.  All fields are `Copy`/`Clone` (the kernel is `&self` for jump).
struct ResolventJumpInner {
    kernel: ResolventJumpChernoff<DiffusionChernoff<f64>, f64>,
}

// ---------------------------------------------------------------------------
// ResolventJumpV8 pyclass
// ---------------------------------------------------------------------------

/// Resolvent time-jump evaluator for the 1D unit-diffusion heat kernel (v8.1.0).
///
/// Evaluates `e^{tA}g` for a LARGE step `t` via the TWS parabolic-contour
/// inverse Laplace quadrature (math.md §47, ADR-0134).  Suitable for large
/// `t` where a many-step Chernoff product would be expensive.
///
/// **NARROW scope**: self-adjoint / sectorial generators only (diffusion family).
/// Non-self-adjoint / advection-dominated generators are OUT of scope
/// (math.md §47.4 NORMATIVE).  `m_nodes >= 6` required.
///
/// Parameters
/// ----------
/// `domain_lo` : float
///     Left boundary (finite).
/// `domain_hi` : float
///     Right boundary (finite, > `domain_lo`).
/// `n_grid` : int
///     Number of grid nodes (>= 4).
/// `m_nodes` : int
///     Number of TWS contour nodes (>= 6; M=16 recommended for |t|≤1).
///
/// Raises
/// ------
/// `SemiflowError`
///     `kind='GridMismatch'`  — invalid grid geometry.
///     `kind='OutOfDomain'`   — `m_nodes` < 6.
#[pyclass(name = "ResolventJumpV8")]
pub struct PyResolventJumpV8 {
    inner: ResolventJumpInner,
}

#[pymethods]
impl PyResolventJumpV8 {
    #[new]
    fn new(domain_lo: f64, domain_hi: f64, n_grid: usize, m_nodes: usize) -> PyResult<Self> {
        catch_panic_py!({
            let inner =
                build_inner(domain_lo, domain_hi, n_grid, m_nodes).map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Evaluate `e^{tA}g` and return the result as a numpy float64 array.
    ///
    /// The GIL is released during the M-node TWS contour solve (ADR-0031).
    /// The complex contour arithmetic stays sealed in core (ADR-0138).
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time step (> 0, finite).
    /// g : array-like
    ///     Initial condition, 1-D float64, length `n_grid`.
    ///
    /// Returns
    /// -------
    /// np.ndarray
    ///     Result `e^{tA}g`, float64, length `n_grid`.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     `kind='GridMismatch'` — len(g) != `n_grid`.
    ///     `kind='OutOfDomain'`  — t <= 0 or non-finite.
    fn jump<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        g: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        catch_panic_py!({
            validate_t(t)?;
            // Phase 1: extract input under GIL.
            let g_vec = extract_f64_vec(g)?;
            let grid = self.inner.kernel.grid;
            let n = grid.n;
            if g_vec.len() != n {
                return Err(new_pyerr("GridMismatch", "len(g) must equal n_grid"));
            }
            // Phase 2: contour solve — release GIL.
            let result = py.detach(|| run_jump(grid, &g_vec, self.inner.kernel.m_nodes, t));
            let values = result.map_err(|e| from_core(&e))?;
            // Phase 3: marshal to numpy under GIL.
            Ok(values.as_slice().to_pyarray(py))
        })
    }

    /// Return the number of grid nodes.
    fn size(&self) -> usize {
        self.inner.kernel.grid.n
    }

    /// Return the number of TWS contour nodes.
    fn m_nodes(&self) -> usize {
        self.inner.kernel.m_nodes
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust contour solve (GIL-off)
// ---------------------------------------------------------------------------

/// Rebuild the kernel and evaluate jump — runs GIL-off under `py.detach`.
///
/// Re-construction is cheap (no heap allocation of grid points) and ensures
/// Send + Sync compliance for `py.detach` (per-crate dup, ADR-0028 Amdt 2).
fn run_jump(
    grid: Grid1D<f64>,
    g_vals: &[f64],
    m_nodes: usize,
    t: f64,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let chernoff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    let kernel = ResolventJumpChernoff::new(chernoff, m_nodes, grid)?;
    let g = GridFn1D::new(grid, g_vals.to_vec())?;
    let result = kernel.jump(t, &g)?;
    Ok(result.values)
}

// ---------------------------------------------------------------------------
// Builder and validators
// ---------------------------------------------------------------------------

fn build_inner(
    lo: f64,
    hi: f64,
    n_grid: usize,
    m_nodes: usize,
) -> Result<ResolventJumpInner, semiflow::SemiflowError> {
    let grid = Grid1D::new(lo, hi, n_grid)?;
    let chernoff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    let kernel = ResolventJumpChernoff::new(chernoff, m_nodes, grid)?;
    Ok(ResolventJumpInner { kernel })
}

fn extract_f64_vec(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "g must be a numpy.ndarray[float64] or a sequence of floats",
        )
    })
}

fn validate_t(t: f64) -> PyResult<()> {
    if !t.is_finite() || t <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and > 0"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `ResolventJumpV8` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResolventJumpV8>()?;
    Ok(())
}
