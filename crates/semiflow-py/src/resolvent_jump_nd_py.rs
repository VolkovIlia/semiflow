//! v8.3.0 `PyO3` bindings for `ResolventJumpChernoff2D`/`3D` (F2 ND, ADR-0153, ADR-0148).
//!
//! Implements `ResolventJump2DV8` and `ResolventJump3DV8` — stateless-per-call
//! Python classes that evaluate `e^{tA}g` for the 2D/3D unit-diffusion heat
//! kernel via the TWS parabolic-contour inverse Laplace quadrature (math.md §47.8).
//!
//! ## NARROW scope (§47.8, ADR-0148 NORMATIVE)
//!
//! Self-adjoint / sectorial generators only (diffusion family, 2D/3D).
//! `m_nodes >= 6` enforced at construction.
//!
//! ## ND layout contract (§3.1, `V8_3_TIER3_BINDING_DESIGN.md` — NORMATIVE)
//!
//! Rust `GridFn2D`/`GridFn3D` are **axis-0-fastest** (Fortran / column-major):
//!   `idx(i,j) = j·nx + i`            (2D),
//!   `idx(i,j,k) = k·nx·ny + j·nx + i` (3D).
//! The `PyO3` layer accepts an ND `np.ndarray` (shape `(nx,ny[,nz])`) and
//! calls `g.ravel(order="F")` before passing to Rust, then returns
//! `out.reshape((nx,ny[,nz]), order="F")`.  This is the NORMATIVE fix for
//! the v8.1.0 C1/F4 C-vs-Fortran-order bug (ADR-0153).  A flat 1-D array of
//! length `nx·ny[·nz]` is accepted unchanged (caller is responsible for layout).
//!
//! ## GIL policy (ADR-0031 three-phase)
//!
//! `.jump` releases the GIL via `py.detach` around the M-node banded complex
//! LU solve (the heavy compute).
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util with semiflow-ffi/wasm.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::too_many_arguments)]

use numpy::{npyffi::NPY_ORDER, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use semiflow_core::{
    Grid1D, Grid2D, Grid3D, GridFn2D, GridFn3D, ResolventJumpChernoff2D, ResolventJumpChernoff3D,
};

use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;

// ---------------------------------------------------------------------------
// Helpers shared within this file (per-crate dup, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

fn validate_t(t: f64) -> PyResult<()> {
    if !t.is_finite() || t <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and > 0"));
    }
    Ok(())
}

/// Extract a flat `Vec<f64>` from a numpy array or sequence.
///
/// If the input is already 1-D the values are used directly.
/// If it is ND, it is raveled in Fortran order (axis-0-fastest = column-major)
/// to match the Rust internal layout.
fn extract_flat_f64(obj: &Bound<'_, pyo3::PyAny>, expected_len: usize) -> PyResult<Vec<f64>> {
    // Try direct flat extraction first (already 1-D or a plain Vec).
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        if v.len() != expected_len {
            return Err(new_pyerr(
                "GridMismatch",
                "g flat length does not match grid size",
            ));
        }
        return Ok(v);
    }
    // ND numpy array: ravel in Fortran order.
    let ravel_f: Bound<'_, pyo3::PyAny> = {
        let kwargs = pyo3::types::PyDict::new(obj.py());
        kwargs.set_item("order", "F")?;
        obj.call_method("ravel", (), Some(&kwargs))?
    };
    let v: Vec<f64> = ravel_f.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "g must be a numpy.ndarray[float64] or a sequence of floats",
        )
    })?;
    if v.len() != expected_len {
        return Err(new_pyerr(
            "GridMismatch",
            "g length does not match grid size after ravel",
        ));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// ResolventJump2DV8 pyclass
// ---------------------------------------------------------------------------

/// Resolvent time-jump evaluator for 2D unit-diffusion heat (v8.3.0, ADR-0153).
///
/// Evaluates ``e^{tA}g`` for a 2D LARGE step ``t`` via the TWS parabolic-contour
/// inverse Laplace quadrature (math.md §47.8, ADR-0148).
///
/// **NARROW scope**: self-adjoint / sectorial parabolic generators only (§47.8).
/// Non-self-adjoint / advection-dominated generators are OUT of scope.
/// ``m_nodes >= 6`` required.
///
/// ## ND layout (NORMATIVE, §3.1 `V8_3_TIER3_BINDING_DESIGN.md`)
///
/// Pass ``g`` as shape ``(nx, ny)`` — the binding calls ``g.ravel(order="F")``
/// internally to match the Rust axis-0-fastest layout (``idx(i,j) = j·nx + i``).
/// The returned array has shape ``(nx, ny)`` reshaped with ``order="F"``.
/// A pre-raveled flat array of length ``nx·ny`` is also accepted.
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     x-axis bounds (finite, xmin < xmax).
/// nx : int
///     Number of x-axis grid nodes (>= 4).
/// ymin, ymax : float
///     y-axis bounds (finite, ymin < ymax).
/// ny : int
///     Number of y-axis grid nodes (>= 4).
/// `m_nodes` : int
///     TWS contour node count (>= 6; M=8 recommended for |t|≤1).
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` — invalid grid geometry.
///     ``kind='OutOfDomain'`` — `m_nodes` < 6.
#[pyclass(name = "ResolventJump2DV8")]
pub struct PyResolventJump2DV8 {
    kernel: ResolventJumpChernoff2D<f64>,
}

#[pymethods]
impl PyResolventJump2DV8 {
    #[new]
    fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        m_nodes: usize,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let kernel = build_kernel_2d(xmin, xmax, nx, ymin, ymax, ny, m_nodes)
                .map_err(|e| from_core(&e))?;
            Ok(Self { kernel })
        })
    }

    /// Evaluate ``e^{tA}g`` and return the result reshaped to ``(nx, ny)``.
    ///
    /// The GIL is released during the M-node banded complex LU solve (ADR-0031).
    /// The complex contour arithmetic stays sealed in core (ADR-0138).
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time step (> 0, finite).
    /// g : array-like
    ///     Initial condition.  Shape ``(nx, ny)`` (raveled ``order="F"``
    ///     internally) or flat ``float64`` array of length ``nx·ny``.
    ///
    /// Returns
    /// -------
    /// np.ndarray
    ///     Result, shape ``(nx, ny)``, ``float64``, Fortran-order layout.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='GridMismatch'`` — g length != nx·ny.
    ///     ``kind='OutOfDomain'`` — t <= 0 or non-finite.
    fn jump<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        g: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        catch_panic_py!({
            validate_t(t)?;
            let grid = self.kernel.grid;
            let n = grid.len();
            // Phase 1: extract + ravel under GIL.
            let g_vec = extract_flat_f64(g, n)?;
            let nx = grid.x.n;
            let ny = grid.y.n;
            // Phase 2: banded LU solve — release GIL.
            let result = py.detach(|| run_jump_2d(grid, &g_vec, self.kernel.m_nodes, t));
            let values = result.map_err(|e| from_core(&e))?;
            // Phase 3: marshal to numpy shape (nx, ny) Fortran-order.
            let flat = values.as_slice().to_pyarray(py);
            // Reshape to (nx, ny) with Fortran (column-major) order — matches
            // the axis-0-fastest internal layout (§3.1 NORMATIVE).
            let reshaped = flat
                .reshape_with_order([nx, ny], NPY_ORDER::NPY_FORTRANORDER)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(reshaped)
        })
    }

    /// Return the ``(nx, ny)`` shape tuple.
    fn shape(&self) -> (usize, usize) {
        (self.kernel.grid.x.n, self.kernel.grid.y.n)
    }

    /// Return the number of TWS contour nodes.
    fn m_nodes(&self) -> usize {
        self.kernel.m_nodes
    }
}

// ---------------------------------------------------------------------------
// ResolventJump3DV8 pyclass
// ---------------------------------------------------------------------------

/// Resolvent time-jump evaluator for 3D unit-diffusion heat (v8.3.0, ADR-0153).
///
/// Evaluates ``e^{tA}g`` for a 3D LARGE step ``t`` via the TWS parabolic-contour
/// inverse Laplace quadrature (math.md §47.8, ADR-0148).
///
/// **NARROW scope**: self-adjoint / sectorial parabolic generators only (§47.8).
/// ``m_nodes >= 6`` required.
///
/// ## ND layout (NORMATIVE, §3.1 `V8_3_TIER3_BINDING_DESIGN.md`)
///
/// Pass ``g`` as shape ``(nx, ny, nz)`` — raveled ``order="F"`` internally to
/// match ``idx(i,j,k) = k·nx·ny + j·nx + i``. Returns shape ``(nx, ny, nz)``
/// with ``order="F"``.
///
/// Parameters
/// ----------
/// xmin, xmax : float  — x-axis bounds.
/// nx : int            — x-axis grid nodes (>= 4).
/// ymin, ymax : float  — y-axis bounds.
/// ny : int            — y-axis grid nodes (>= 4).
/// zmin, zmax : float  — z-axis bounds.
/// nz : int            — z-axis grid nodes (>= 4).
/// `m_nodes` : int       — TWS contour nodes (>= 6).
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` — invalid grid geometry.
///     ``kind='OutOfDomain'`` — `m_nodes` < 6.
#[pyclass(name = "ResolventJump3DV8")]
pub struct PyResolventJump3DV8 {
    kernel: ResolventJumpChernoff3D<f64>,
}

#[pymethods]
impl PyResolventJump3DV8 {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        zmin: f64,
        zmax: f64,
        nz: usize,
        m_nodes: usize,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let kernel = build_kernel_3d(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, m_nodes)
                .map_err(|e| from_core(&e))?;
            Ok(Self { kernel })
        })
    }

    /// Evaluate ``e^{tA}g`` and return the result reshaped to ``(nx, ny, nz)``.
    ///
    /// GIL released during the M-node banded complex LU solve (ADR-0031).
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time step (> 0, finite).
    /// g : array-like
    ///     Shape ``(nx, ny, nz)`` (raveled ``order="F"``) or flat length ``nx·ny·nz``.
    ///
    /// Returns
    /// -------
    /// np.ndarray
    ///     Result, shape ``(nx, ny, nz)``, ``float64``, Fortran-order layout.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='GridMismatch'`` — g length != nx·ny·nz.
    ///     ``kind='OutOfDomain'`` — t <= 0 or non-finite.
    fn jump<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        g: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Bound<'py, numpy::PyArray3<f64>>> {
        catch_panic_py!({
            validate_t(t)?;
            let grid = self.kernel.grid;
            let n = grid.len();
            // Phase 1: extract + ravel under GIL.
            let g_vec = extract_flat_f64(g, n)?;
            let nx = grid.x.n;
            let ny = grid.y.n;
            let nz = grid.z.n;
            // Phase 2: banded LU solve — release GIL.
            let result = py.detach(|| run_jump_3d(grid, &g_vec, self.kernel.m_nodes, t));
            let values = result.map_err(|e| from_core(&e))?;
            // Phase 3: marshal to numpy shape (nx, ny, nz) Fortran-order.
            let flat = values.as_slice().to_pyarray(py);
            // Reshape with Fortran order — matches axis-0-fastest layout (§3.1 NORMATIVE).
            let reshaped = flat
                .reshape_with_order([nx, ny, nz], NPY_ORDER::NPY_FORTRANORDER)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(reshaped)
        })
    }

    /// Return the ``(nx, ny, nz)`` shape tuple.
    fn shape(&self) -> (usize, usize, usize) {
        (
            self.kernel.grid.x.n,
            self.kernel.grid.y.n,
            self.kernel.grid.z.n,
        )
    }

    /// Return the number of TWS contour nodes.
    fn m_nodes(&self) -> usize {
        self.kernel.m_nodes
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust contour solves (GIL-off, per-crate dup ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

/// Rebuild 2D kernel and evaluate jump — runs GIL-off under `py.detach`.
fn run_jump_2d(
    grid: Grid2D<f64>,
    g_vals: &[f64],
    m_nodes: usize,
    t: f64,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let kernel = ResolventJumpChernoff2D::new(grid, m_nodes)?;
    let g = GridFn2D {
        values: g_vals.to_vec(),
        grid,
    };
    let result = kernel.jump(t, &g)?;
    Ok(result.values)
}

/// Rebuild 3D kernel and evaluate jump — runs GIL-off under `py.detach`.
fn run_jump_3d(
    grid: Grid3D<f64>,
    g_vals: &[f64],
    m_nodes: usize,
    t: f64,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let kernel = ResolventJumpChernoff3D::new(grid, m_nodes)?;
    let g = GridFn3D {
        values: g_vals.to_vec(),
        grid,
    };
    let result = kernel.jump(t, &g)?;
    Ok(result.values)
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

fn build_kernel_2d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    m_nodes: usize,
) -> Result<ResolventJumpChernoff2D<f64>, semiflow_core::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    let grid = Grid2D::new(gx, gy);
    ResolventJumpChernoff2D::new(grid, m_nodes)
}

#[allow(clippy::too_many_arguments)]
fn build_kernel_3d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    zmin: f64,
    zmax: f64,
    nz: usize,
    m_nodes: usize,
) -> Result<ResolventJumpChernoff3D<f64>, semiflow_core::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    let gz = Grid1D::new(zmin, zmax, nz)?;
    let grid = Grid3D::new(gx, gy, gz)?;
    ResolventJumpChernoff3D::new(grid, m_nodes)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `ResolventJump2DV8` and `ResolventJump3DV8` into `semiflow`.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResolventJump2DV8>()?;
    m.add_class::<PyResolventJump3DV8>()?;
    Ok(())
}
