//! Wave P5 — geometry: manifold backend (`geometry.rs`).
//!
//! ADR-0111 parity items M13–M15.
//!
//! | pyclass | Core type | M# | Module |
//! |---------|-----------|-----|--------|
//! | `Manifold2D` | `ManifoldChernoff<M, f64>` | M13 | here |
//! | `HypoellipticChernoffKolmogorov` | `KolmogorovHypoelliptic<f64>` | M14 | `geometry_hypoelliptic` |
//! | `HypoellipticChernoffEngel` | `HypoellipticChernoff<f64,4,2>` | M15 | `geometry_hypoelliptic` |
//!
//! ## Design notes
//!
//! ### `Manifold2D` backend enum
//!
//! `BoundedGeometryManifold` is a non-object-safe trait (const-generic `D` on `Torus`).
//! To avoid monomorphising a separate pyclass per backend while keeping the wrapper
//! `Send + Sync`, we introduce a `ManifoldEnum` newtype that owns one of the three
//! concrete backends and dispatches `ChernoffFunction<f64>` by matching on the variant.
//! This mirrors the `SubordinatorEnum` pattern from `time_dependent.rs` (Wave P4).
//!
//! ## GIL policy (ADR-0031)
//!
//! All `evolve` methods: validate + copy under GIL → `py.detach` compute →
//! write result under GIL.  All three kernel types are `Send + Sync` (verified
//! in `send_assertions.rs`).

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_precision_loss, clippy::too_many_arguments)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    manifold::{Hyperbolic2, Sphere2, Torus},
    manifold_chernoff::ManifoldChernoff,
    ChernoffFunction, ChernoffSemigroup, Grid1D, Grid2D, GridFn2D, ScratchPool,
};

use crate::{
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// Re-export hypoelliptic classes for registration.
pub(crate) use crate::geometry_hypoelliptic::{
    PyHypoellipticChernoffEngel, PyHypoellipticChernoffKolmogorov,
};

// Shared helpers used by geometry_hypoelliptic.rs.
pub(crate) use crate::geometry_hypoelliptic::{
    extract_f64_vec_geo, validate_params_geo, validate_u0_geo,
};

// ---------------------------------------------------------------------------
// Manifold2D — ManifoldEnum dispatch (binding-side enum, mirrors P4 pattern)
// ---------------------------------------------------------------------------

/// Binding-side enum wrapping all three `ManifoldChernoff` concrete types.
///
/// `BoundedGeometryManifold` is not object-safe (const-generic `D` on `Torus`).
/// This enum implements `ChernoffFunction<f64>` by matching on the variant,
/// keeping the wrapper `Send + Sync` without dynamic dispatch overhead.
#[derive(Clone)]
enum ManifoldEnum {
    Torus(ManifoldChernoff<Torus<f64, 2>, f64>),
    Sphere(ManifoldChernoff<Sphere2<f64>, f64>),
    Hyperbolic(ManifoldChernoff<Hyperbolic2<f64>, f64>),
}

// Safety: all three inner types are Send+Sync (BoundedGeometryManifold is Send+Sync,
// ManifoldChernoff<M,F> contains only M and PhantomData<F>).
unsafe impl Send for ManifoldEnum {}
unsafe impl Sync for ManifoldEnum {}

impl ChernoffFunction<f64> for ManifoldEnum {
    type S = GridFn2D<f64>;

    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn2D<f64>,
        dst: &mut GridFn2D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow::SemiflowError> {
        match self {
            ManifoldEnum::Torus(k) => k.apply_into(tau, src, dst, scratch),
            ManifoldEnum::Sphere(k) => k.apply_into(tau, src, dst, scratch),
            ManifoldEnum::Hyperbolic(k) => k.apply_into(tau, src, dst, scratch),
        }
    }

    fn order(&self) -> u32 {
        match self {
            ManifoldEnum::Torus(k) => k.order(),
            ManifoldEnum::Sphere(k) => k.order(),
            ManifoldEnum::Hyperbolic(k) => k.order(),
        }
    }

    fn growth(&self) -> semiflow::chernoff::Growth<f64> {
        match self {
            ManifoldEnum::Torus(k) => k.growth(),
            ManifoldEnum::Sphere(k) => k.growth(),
            ManifoldEnum::Hyperbolic(k) => k.growth(),
        }
    }
}

fn parse_manifold(
    manifold: &str,
    radius: f64,
    curvature_correction: bool,
) -> PyResult<ManifoldEnum> {
    match manifold {
        "torus" => {
            let m = Torus::<f64, 2>::unit();
            Ok(ManifoldEnum::Torus(ManifoldChernoff::new(
                m,
                curvature_correction,
            )))
        }
        "sphere2" => {
            let m = Sphere2::with_radius(radius).map_err(|e| from_core(&e))?;
            Ok(ManifoldEnum::Sphere(ManifoldChernoff::new(
                m,
                curvature_correction,
            )))
        }
        "hyperbolic2" => {
            let m = Hyperbolic2::with_scale(radius).map_err(|e| from_core(&e))?;
            Ok(ManifoldEnum::Hyperbolic(ManifoldChernoff::new(
                m,
                curvature_correction,
            )))
        }
        other => Err(new_pyerr(
            "Unsupported",
            &format!("Unknown manifold '{other}'. Must be 'torus', 'sphere2', or 'hyperbolic2'."),
        )),
    }
}

// ---------------------------------------------------------------------------
// Manifold2D inner state
// ---------------------------------------------------------------------------

struct Manifold2DInner {
    kernel: ManifoldEnum,
    current: GridFn2D<f64>,
    nx: usize,
    ny: usize,
}

// ---------------------------------------------------------------------------
// Manifold2D pyclass (M13)
// ---------------------------------------------------------------------------

/// 2-D Riemannian manifold Chernoff approximation (M13).
///
/// Wraps ``ManifoldChernoff<M, f64>`` (MMRS 2023 *Math. Nachr.* Thm 1,
/// math.md §24, ADR-0071).  The backend manifold is selected via the
/// ``manifold=`` string kwarg.
///
/// Backends
/// --------
/// - ``"torus"``      — flat 2-torus T² (R ≡ 0; curvature correction is no-op).
/// - ``"sphere2"``    — 2-sphere S²(r) (R ≡ 2/r²; ``radius`` sets r).
/// - ``"hyperbolic2"``— Poincaré disk H²(s) (R ≡ -2/s²; ``radius`` sets s).
///
/// State type is ``GridFn2D<f64>`` — a function sampled on a 2D chart grid
/// ``[x0min, x0max] × [x1min, x1max]``.  Row-major flat float64 of length
/// ``nx * ny``.
///
/// Parameters
/// ----------
/// x0min, x0max : float
///     Chart-axis-0 boundary (must be finite and x0min < x0max).
/// nx : int
///     Number of nodes on axis 0 (must be >= 4).
/// x1min, x1max : float
///     Chart-axis-1 boundary (must be finite and x1min < x1max).
/// ny : int
///     Number of nodes on axis 1 (must be >= 4).
/// u0 : array-like
///     Initial condition; float64 array of length nx*ny (row-major).
/// manifold : str, optional
///     Backend: ``"torus"`` (default), ``"sphere2"``, or ``"hyperbolic2"``.
/// radius : float, optional
///     Sphere radius (for sphere2) or scale (for hyperbolic2); default 1.0.
///     Must be > 0.  Ignored for torus.
/// `curvature_correction` : bool, optional
///     If True, applies the R/12 curvature correction (MMRS 2023 Thm 1),
///     lifting order from 1 to 2.  Default True.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if any grid axis is invalid or u0 length mismatch.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if radius <= 0 or non-finite.
///     kind='Unsupported' if manifold string is not recognised.
#[pyclass(name = "Manifold2D")]
pub struct PyManifold2D {
    inner: Manifold2DInner,
}

#[pymethods]
impl PyManifold2D {
    #[new]
    #[pyo3(signature = (
        x0min, x0max, nx,
        x1min, x1max, ny,
        u0, *,
        manifold = "torus",
        radius = 1.0_f64,
        curvature_correction = true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x0min: f64,
        x0max: f64,
        nx: usize,
        x1min: f64,
        x1max: f64,
        ny: usize,
        u0: &Bound<'_, PyAny>,
        manifold: &str,
        radius: f64,
        curvature_correction: bool,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let u0_vec = extract_f64_vec_geo(u0)?;
            let kernel = parse_manifold(manifold, radius, curvature_correction)?;
            let inner = build_manifold2d(x0min, x0max, nx, x1min, x1max, ny, &u0_vec, kernel)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order.
    ///
    /// Returns 2 when ``curvature_correction=True`` for sphere2/hyperbolic2;
    /// returns 1 for torus (R ≡ 0, correction is identity) or when
    /// ``curvature_correction=False``.
    fn order(&self) -> u32 {
        self.inner.kernel.order()
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
            let kernel = self.inner.kernel.clone();
            let grid = self.inner.current.grid;
            let values: Vec<f64> = self.inner.current.values.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| evolve_manifold2d(kernel, grid, values, t, n_steps));
            self.inner.current.values = result.map_err(|e| from_core(&e))?;
            Ok(())
        })
    }

    /// Return current chart values as ``numpy.ndarray[float64]`` (copy).
    ///
    /// Length is ``nx * ny``, row-major (axis 0 fast).
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Return ``nx * ny`` (total number of grid nodes).
    fn __len__(&self) -> usize {
        self.inner.nx * self.inner.ny
    }

    fn __repr__(&self) -> String {
        format!(
            "Manifold2D(nx={}, ny={}, order={})",
            self.inner.nx,
            self.inner.ny,
            self.inner.kernel.order()
        )
    }
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

fn build_manifold2d(
    x0min: f64,
    x0max: f64,
    nx: usize,
    x1min: f64,
    x1max: f64,
    ny: usize,
    u0: &[f64],
    kernel: ManifoldEnum,
) -> Result<Manifold2DInner, semiflow::SemiflowError> {
    validate_u0_geo(u0)?;
    let gx = Grid1D::new(x0min, x0max, nx)?;
    let gy = Grid1D::new(x1min, x1max, ny)?;
    let grid = Grid2D::new(gx, gy);
    let expected = nx * ny;
    if u0.len() != expected {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "u0 length must equal nx * ny",
            value: u0.len() as f64,
        });
    }
    let current = GridFn2D::new_generic(grid, u0.to_vec())?;
    Ok(Manifold2DInner {
        kernel,
        current,
        nx,
        ny,
    })
}

// ---------------------------------------------------------------------------
// GIL-free compute helper
// ---------------------------------------------------------------------------

fn evolve_manifold2d(
    kernel: ManifoldEnum,
    grid: Grid2D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let f = GridFn2D::new_generic(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register Wave P5 pyclasses into the `semiflow` module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = py;
    m.add_class::<PyManifold2D>()?;
    m.add_class::<PyHypoellipticChernoffKolmogorov>()?;
    m.add_class::<PyHypoellipticChernoffEngel>()?;
    Ok(())
}
