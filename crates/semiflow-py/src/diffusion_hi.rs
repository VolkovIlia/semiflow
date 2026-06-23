//! `Heat1D4th` and `Heat1D6th` — 4th/6th-order 1-D diffusion kernels for Python.
//!
//! Same three-phase GIL pattern (ADR-0031) as `Heat1D`: validate under GIL,
//! compute inside `py.detach`, update state under GIL.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_precision_loss, clippy::similar_names)]

use numpy::{PyArray1, ToPyArray};
use pyo3::{prelude::*, types::PyAnyMethods};
use semiflow::{ChernoffSemigroup, Diffusion4thChernoff, Diffusion6thChernoff, GridFn1D};

use crate::{
    boundary::parse_boundary,
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// Array-based builders + compute helpers extracted to keep this file ≤500 lines (batch H8).
// Included directly so helpers share the same module namespace.
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/diffusion_hi_helpers.rs"
));

// ---------------------------------------------------------------------------
// Inner state types
// ---------------------------------------------------------------------------

/// Inner state for `Heat1D4th`.
pub(crate) struct Diff4StateInner {
    /// The Chernoff semigroup wrapping `Diffusion4thChernoff`.
    pub semigroup: ChernoffSemigroup<Diffusion4thChernoff<f64>, GridFn1D<f64>>,
    /// Current grid function state.
    pub current: GridFn1D<f64>,
}

/// Inner state for `Heat1D6th`.
pub(crate) struct Diff6StateInner {
    /// The Chernoff semigroup wrapping `Diffusion6thChernoff`.
    pub semigroup: ChernoffSemigroup<Diffusion6thChernoff<f64>, GridFn1D<f64>>,
    /// Current grid function state.
    pub current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Heat1D4th Python class
// ---------------------------------------------------------------------------

/// 1-D diffusion state using the 4th-order Chernoff kernel.
///
/// Solves `∂_t u = a(x)∂²u`. Default: unit `a = 1`.
/// Uses `Diffusion4thChernoff` — global spatial order 4.
///
/// Raises `SemiflowError` with `kind='GridMismatch'` or `'NanInf'` on invalid inputs.
#[pyclass(name = "Heat1D4th")]
pub struct Heat1D4th {
    inner: Diff4StateInner,
}

#[pymethods]
impl Heat1D4th {
    /// Create a new `Heat1D4th` state with unit diffusion coefficient.
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, boundary = "reflect"))]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let slice = extract_f64_slice(u0)?;
            let inner =
                build_diff4_unit(xmin, xmax, n, 100, &slice, policy).map_err(|e| from_core(&e))?;
            Ok(Heat1D4th { inner })
        })
    }

    /// Advance state by time `t` using `n_steps` Chernoff iterations.
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_evolve_params(t, n_steps)?;
            let func = self.inner.semigroup.func.clone();
            let grid = self.inner.current.grid;
            let input: Vec<f64> = self.inner.current.values.clone();
            let func_for_sg = func.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| compute_evolve_4th(func, grid, input, t, n_steps));
            let result = result.map_err(|e| from_core(&e))?;
            self.inner.current.values = result;
            let sg = ChernoffSemigroup::new(func_for_sg, n_steps).map_err(|e| from_core(&e))?;
            self.inner.semigroup = sg;
            Ok(())
        })
    }

    /// Return the current grid values as `numpy.ndarray[float64]`.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Number of grid nodes.
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }

    /// Create `Heat1D4th` from pre-sampled arrays `a`, `u0`.
    ///
    /// `a_prime`, `a_double_prime` default to 4th-order FD if ``None``.
    /// `a_norm_bound` defaults to ``1.1 * max(a)``.
    /// See `Heat1D.with_a_array` for full parameter docs.
    #[staticmethod]
    #[pyo3(signature = (xmin, xmax, n, a, u0, *,
                        a_prime = None, a_double_prime = None,
                        a_norm_bound = None, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn with_a_array(
        xmin: f64,
        xmax: f64,
        n: usize,
        a: &Bound<'_, PyAny>,
        u0: &Bound<'_, PyAny>,
        a_prime: Option<&Bound<'_, PyAny>>,
        a_double_prime: Option<&Bound<'_, PyAny>>,
        a_norm_bound: Option<f64>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            build_heat4th_from_arrays(
                xmin,
                xmax,
                n,
                a,
                u0,
                a_prime,
                a_double_prime,
                a_norm_bound,
                boundary,
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Heat1D6th Python class
// ---------------------------------------------------------------------------

/// 1-D diffusion state using the 6th-order Chernoff kernel.
///
/// Same API as `Heat1D4th` but backed by `Diffusion6thChernoff` (spatial order 6).
/// Solves `∂_t u = a(x)∂²u`. Default: unit `a = 1`.
#[pyclass(name = "Heat1D6th")]
pub struct Heat1D6th {
    inner: Diff6StateInner,
}

#[pymethods]
impl Heat1D6th {
    /// Create a new `Heat1D6th` state with unit diffusion coefficient.
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, boundary = "reflect"))]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = parse_boundary(boundary)?;
            let slice = extract_f64_slice(u0)?;
            let inner =
                build_diff6_unit(xmin, xmax, n, 100, &slice, policy).map_err(|e| from_core(&e))?;
            Ok(Heat1D6th { inner })
        })
    }

    /// Advance state by time `t` using `n_steps` Chernoff iterations.
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_evolve_params(t, n_steps)?;
            let func = self.inner.semigroup.func.clone();
            let grid = self.inner.current.grid;
            let input: Vec<f64> = self.inner.current.values.clone();
            let func_for_sg = func.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| compute_evolve_6th(func, grid, input, t, n_steps));
            let result = result.map_err(|e| from_core(&e))?;
            self.inner.current.values = result;
            let sg = ChernoffSemigroup::new(func_for_sg, n_steps).map_err(|e| from_core(&e))?;
            self.inner.semigroup = sg;
            Ok(())
        })
    }

    /// Return the current grid values as `numpy.ndarray[float64]`.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Number of grid nodes.
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }

    /// Create `Heat1D6th` from pre-sampled arrays. Same parameters as `Heat1D4th.with_a_array`.
    #[staticmethod]
    #[pyo3(signature = (xmin, xmax, n, a, u0, *,
                        a_prime = None, a_double_prime = None,
                        a_norm_bound = None, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn with_a_array(
        xmin: f64,
        xmax: f64,
        n: usize,
        a: &Bound<'_, PyAny>,
        u0: &Bound<'_, PyAny>,
        a_prime: Option<&Bound<'_, PyAny>>,
        a_double_prime: Option<&Bound<'_, PyAny>>,
        a_norm_bound: Option<f64>,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            build_heat6th_from_arrays(
                xmin,
                xmax,
                n,
                a,
                u0,
                a_prime,
                a_double_prime,
                a_norm_bound,
                boundary,
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Unit-coefficient builders
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_d(_: f64) -> f64 {
    0.0
}

/// Build `Diff4StateInner` with constant `a = 1.0`.
///
/// # Errors
/// Propagates `SemiflowError` from grid or semigroup construction.
pub(crate) fn build_diff4_unit(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    u0: &[f64],
    boundary: semiflow::BoundaryPolicy,
) -> Result<Diff4StateInner, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = semiflow::Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let chernoff = Diffusion4thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Diff4StateInner { semigroup, current })
}

/// Build `Diff6StateInner` with constant `a = 1.0`.
///
/// # Errors
/// Propagates `SemiflowError` from grid or semigroup construction.
pub(crate) fn build_diff6_unit(
    xmin: f64,
    xmax: f64,
    n: usize,
    n_steps: usize,
    u0: &[f64],
    boundary: semiflow::BoundaryPolicy,
) -> Result<Diff6StateInner, semiflow::SemiflowError> {
    validate_u0_finite(u0)?;
    let grid = semiflow::Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let chernoff = Diffusion6thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, n_steps)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Diff6StateInner { semigroup, current })
}

// ---------------------------------------------------------------------------
// Shared validation helpers
// ---------------------------------------------------------------------------

pub(super) fn validate_evolve_params(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

pub(super) fn validate_u0_finite(u0: &[f64]) -> Result<(), semiflow::SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(semiflow::SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}

/// Convert any Python array-like to a `Vec<f64>`.
pub(super) fn extract_f64_slice(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "u0 must be a numpy.ndarray[float64] or a sequence of floats",
        )
    })
}
