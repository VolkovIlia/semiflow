//! Wave P5 — `HypoellipticChernoffKolmogorov` (M14) and
//! `HypoellipticChernoffEngel` (M15).
//!
//! Split from `geometry.rs` for suckless file-size compliance.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_precision_loss,
    clippy::items_after_statements,
    clippy::too_many_arguments,
    clippy::unused_self
)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    grid_nd::{GridFnND, GridND},
    hormander::{HypoellipticChernoff, KolmogorovHypoelliptic, KolmogorovPhaseSpace},
    ChernoffSemigroup, Grid1D, Grid2D, GridFn2D,
};

use crate::{
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Shared helpers (re-exported from geometry.rs)
// ---------------------------------------------------------------------------

pub(crate) fn validate_params_geo(n_steps: usize, t: f64) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

pub(crate) fn extract_f64_vec_geo(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "u0 must be a numpy.ndarray[float64] or a sequence of floats",
        )
    })
}

pub(crate) fn validate_u0_geo(u0: &[f64]) -> Result<(), semiflow::SemiflowError> {
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

// ---------------------------------------------------------------------------
// HypoellipticChernoffKolmogorov inner state
//
// `HypoellipticChernoff` is NOT Clone (contains Box<dyn VectorField>).
// We reconstruct the canonical Kolmogorov kernel inside py.detach — this
// is just struct construction, no I/O or heavy computation.
// ---------------------------------------------------------------------------

struct KolmogorovInner {
    current: GridFn2D<f64>,
}

// ---------------------------------------------------------------------------
// HypoellipticChernoffKolmogorov pyclass (M14)
// ---------------------------------------------------------------------------

/// Kolmogorov phase-space hypoelliptic Chernoff approximation (M14).
///
/// Wraps ``KolmogorovHypoelliptic<f64>``
/// (= ``HypoellipticChernoff<f64, 2, 1>``) with the palindromic
/// Strang-Hörmander decomposition:
///
/// ``F(τ) = exp(τX₀/2) ∘ exp(τ/2·∂²_v) ∘ exp(τX₀/2)``
///
/// where ``X₀ = v·∂_x`` (advection), ``X₁ = ∂_v`` (diffusion in v).
/// Models ``∂_t p = v·∂_x p + ½·∂²_v p`` — the Kolmogorov equation
/// (Kolmogorov 1934 *Math. Annalen* 108, math.md §28.4.A, ADR-0077).
///
/// State type: ``GridFn2D<f64>`` (2D phase-space chart).
/// Row-major flat array of length ``nx * ny``.
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     x-axis boundary (must be finite and xmin < xmax).
/// nx : int
///     Number of nodes on x-axis (must be >= 4).
/// vmin, vmax : float
///     v-axis boundary (velocity space; must be finite and vmin < vmax).
/// nv : int
///     Number of nodes on v-axis (must be >= 4).
/// u0 : array-like
///     Initial condition; float64 array of length nx*nv (row-major).
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if any grid axis is invalid or u0 length mismatch.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if the Hörmander bracket condition fails at origin
///         (should never occur with canonical Kolmogorov fields).
#[pyclass(name = "HypoellipticChernoffKolmogorov")]
pub struct PyHypoellipticChernoffKolmogorov {
    inner: KolmogorovInner,
}

#[pymethods]
impl PyHypoellipticChernoffKolmogorov {
    #[new]
    #[pyo3(signature = (xmin, xmax, nx, vmin, vmax, nv, u0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        vmin: f64,
        vmax: f64,
        nv: usize,
        u0: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let u0_vec = extract_f64_vec_geo(u0)?;
            let inner = build_kolmogorov(xmin, xmax, nx, vmin, vmax, nv, &u0_vec)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (always 2 — palindromic Strang-Hörmander).
    fn order(&self) -> u32 {
        2
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// GIL released during inner Rust compute (ADR-0031).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if t < 0, non-finite, or `n_steps` == 0.
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_params_geo(n_steps, t)?;
            let grid = self.inner.current.grid;
            let values: Vec<f64> = self.inner.current.values.clone();
            // HypoellipticChernoff is !Clone; reconstruct canonical fields inside
            // py.detach (cheap — struct + bracket check only).
            let result: Result<Vec<f64>, _> =
                py.detach(|| evolve_kolmogorov(grid, values, t, n_steps));
            self.inner.current.values = result.map_err(|e| from_core(&e))?;
            Ok(())
        })
    }

    /// Return current phase-space values as ``numpy.ndarray[float64]`` (copy).
    ///
    /// Length is ``nx * nv``, row-major.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Return ``nx * nv`` (total number of phase-space grid nodes).
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }

    fn __repr__(&self) -> &'static str {
        "HypoellipticChernoffKolmogorov(order=2, D=2, M=1)"
    }
}

// ---------------------------------------------------------------------------
// HypoellipticChernoffEngel inner state
//
// Same !Clone constraint as Kolmogorov — kernel is reconstructed in py.detach.
// ---------------------------------------------------------------------------

struct EngelInner {
    current: GridFnND<f64, 4>,
    n: usize,
}

// ---------------------------------------------------------------------------
// HypoellipticChernoffEngel pyclass (M15)
// ---------------------------------------------------------------------------

/// Engel step-3 Carnot group hypoelliptic Chernoff approximation (M15).
///
/// Wraps ``HypoellipticChernoff<f64, 4, 2>`` (Engel group, D=4, M=2
/// generators) with the palindromic Strang-Hörmander decomposition:
///
/// ``F(τ) = exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)``
///
/// on ``ℝ⁴`` (math.md §28.bis.2, ADR-0095).  The bracket structure is
/// verified at construction: ``[X₁, X₂] ≈ X₃`` and ``[X₁, X₃] ≈ X₄``
/// (step-3 Carnot; `T_HORM` 5/5 sympy PASS).
///
/// State type: ``GridFnND<f64, 4>`` — flat ``n⁴`` float64 array (copy).
/// Axis 0 is fastest.  For ``n=8`` the grid has 4096 points.
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     Common axis boundary for all 4 axes (must be finite and xmin < xmax).
/// n : int
///     Per-axis node count (must be >= 4).  All 4 axes share ``n``.
/// u0 : array-like
///     Initial condition; float64 array of length n**4.
///
/// Notes
/// -----
/// A useful oracle: the Engel Chernoff step preserves total integral
/// (mass conservation) for smooth, compactly supported initial data.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if grid params are invalid or u0 length mismatch.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if the Engel bracket check fails at the origin.
#[pyclass(name = "HypoellipticChernoffEngel")]
pub struct PyHypoellipticChernoffEngel {
    inner: EngelInner,
}

#[pymethods]
impl PyHypoellipticChernoffEngel {
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0))]
    fn new(xmin: f64, xmax: f64, n: usize, u0: &Bound<'_, PyAny>) -> PyResult<Self> {
        catch_panic_py!({
            let u0_vec = extract_f64_vec_geo(u0)?;
            let inner = build_engel(xmin, xmax, n, &u0_vec).map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (always 2 — palindromic Strang-Hörmander).
    fn order(&self) -> u32 {
        2
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// GIL released during inner Rust compute (ADR-0031).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if t < 0, non-finite, or `n_steps` == 0.
    #[pyo3(signature = (t, n_steps = 10))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_params_geo(n_steps, t)?;
            let grid = self.inner.current.grid.clone();
            let values: Vec<f64> = self.inner.current.values.clone();
            // Reconstruct Engel kernel inside py.detach (not Clone).
            let result: Result<Vec<f64>, _> = py.detach(|| evolve_engel(grid, values, t, n_steps));
            self.inner.current.values = result.map_err(|e| from_core(&e))?;
            Ok(())
        })
    }

    /// Return current 4D-grid values as ``numpy.ndarray[float64]`` (copy).
    ///
    /// Length is ``n**4``, flat row-major (axis 0 fastest).
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Return ``n**4`` (total number of grid nodes).
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "HypoellipticChernoffEngel(order=2, n={}, D=4, M=2)",
            self.inner.n
        )
    }
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

fn build_kolmogorov(
    xmin: f64,
    xmax: f64,
    nx: usize,
    vmin: f64,
    vmax: f64,
    nv: usize,
    u0: &[f64],
) -> Result<KolmogorovInner, semiflow::SemiflowError> {
    validate_u0_geo(u0)?;
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gv = Grid1D::new(vmin, vmax, nv)?;
    let grid = Grid2D::new(gx, gv);
    let expected = nx * nv;
    if u0.len() != expected {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "u0 length must equal nx * nv",
            value: u0.len() as f64,
        });
    }
    // Verify bracket condition at construction time (fails loudly if Hörmander fails).
    extern crate alloc;
    let x0 = alloc::boxed::Box::new(KolmogorovPhaseSpace::x0_drift());
    let x1 = alloc::boxed::Box::new(KolmogorovPhaseSpace::x1_diffusion());
    let _ = KolmogorovHypoelliptic::<f64>::new(x0, [x1])?;
    let current = GridFn2D::new_generic(grid, u0.to_vec())?;
    Ok(KolmogorovInner { current })
}

fn build_engel(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
) -> Result<EngelInner, semiflow::SemiflowError> {
    validate_u0_geo(u0)?;
    let ax = Grid1D::new(xmin, xmax, n)?;
    let grid = GridND::<f64, 4>::new([ax, ax, ax, ax])?;
    let expected = n * n * n * n;
    if u0.len() != expected {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "u0 length must equal n**4",
            value: u0.len() as f64,
        });
    }
    // Verify Engel bracket condition at construction time.
    let _ = HypoellipticChernoff::<f64, 4, 2>::new_engel()?;
    let current = GridFnND {
        values: u0.to_vec(),
        grid,
    };
    Ok(EngelInner { current, n })
}

// ---------------------------------------------------------------------------
// GIL-free compute helpers
// ---------------------------------------------------------------------------

fn evolve_kolmogorov(
    grid: Grid2D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    extern crate alloc;
    // Reconstruct canonical Kolmogorov kernel (cheap: struct + bracket check).
    let x0 = alloc::boxed::Box::new(KolmogorovPhaseSpace::x0_drift());
    let x1 = alloc::boxed::Box::new(KolmogorovPhaseSpace::x1_diffusion());
    let kernel = KolmogorovHypoelliptic::<f64>::new(x0, [x1])?;
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let f = GridFn2D::new_generic(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}

fn evolve_engel(
    grid: GridND<f64, 4>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    // Reconstruct Engel kernel (cheap: struct + bracket check).
    let kernel = HypoellipticChernoff::<f64, 4, 2>::new_engel()?;
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let f = GridFnND { values, grid };
    Ok(sg.evolve(t, &f)?.values)
}
