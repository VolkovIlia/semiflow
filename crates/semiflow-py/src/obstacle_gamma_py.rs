//! v8.3.0 TIER-2 `PyO3` bindings for obstacle Γ and `ObstacleND` (ADR-0153, §4).
//!
//! ## `ObstacleGammaV8` — inactive-set Γ = V″ two-output primitive
//!
//! Wraps `ObstacleChernoff::apply_inactive_gamma_into` (math §44.5.bis).
//! Returns a Python tuple `(gamma: np.ndarray[float64], defined: np.ndarray[bool],
//! count: int)`.
//!
//! **Honesty (NORMATIVE, math §44.5.bis)**
//!
//! `defined[i] == False` means Γ is **REFUSED** at node `i` (active set /
//! contact line / one-node guard band). It does NOT mean "Γ = 0". Callers MUST
//! consult `defined[i]` before reading `gamma[i]`. Γ JUMPS across the free
//! boundary `x*` (perpetual-put witness Γ(S*⁺)≈4.90, Γ(S*⁻)=0); there is NO
//! classical global Γ. D=1 only; multi-asset Γ deferred (§44.5.ter, ADR-0153).
//!
//! ## `ObstacleNDV8` — D=2 forward evolution (Fortran-order ND state)
//!
//! Wraps `ObstacleChernoffND<AnisotropicShiftChernoffND<f64,2>, ConstantObstacle, f64, 2>`
//! — a unit-isotropic-diffusion inner for 2D forward evolution.
//! ND state obeys the §3.1 Fortran-order contract: input `v` must be flat
//! axis-0-fastest; output is also flat axis-0-fastest.
//! D=3 deferred (same posture as WASM/FFI ND, ADR-0153).
//!
//! ## FFI / WASM scope note
//!
//! FFI (`obstacle_gamma_ffi.rs`) and WASM (`obstacle_gamma_wasm.rs`) obstacle
//! bindings are DEFERRED (ADR-0153 TIER-2 opportunistic). This `PyO3` file is
//! the TIER-2 SHIP deliverable. No shared util with semiflow-ffi / semiflow-wasm
//! (ADR-0028 Amendment 2).
//!
//! ## GIL policy (ADR-0031)
//!
//! `inactive_gamma`: no GIL release — cheap O(N) single-pass stencil.
//! `ObstacleNDV8.apply`: GIL released via `py.detach` around the ND sweep.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::needless_pass_by_value)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow_core::{
    AnisotropicShiftChernoffND, ChernoffFunction, ConstantObstacle, DiffusionChernoff, Grid1D,
    GridFn1D, GridFnND, GridND, ObstacleChernoff, ObstacleChernoffND, ScratchPool, SquareMatrix,
};

use crate::{
    error::{from_core, new_pyerr},
    obstacle_py::ArrayObstacle,
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Type aliases for the ObstacleGammaV8 inner
// ---------------------------------------------------------------------------

type GammaConst = ObstacleChernoff<DiffusionChernoff<f64>, ConstantObstacle<f64>, f64>;
type GammaArray = ObstacleChernoff<DiffusionChernoff<f64>, ArrayObstacle, f64>;

/// Dispatch over constant / array obstacle variants for Γ.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
enum GammaVariant {
    Const(GammaConst),
    Array(GammaArray),
}

impl GammaVariant {
    fn apply_gamma(
        &self,
        v: &GridFn1D<f64>,
        gamma: &mut GridFn1D<f64>,
        defined: &mut [bool],
    ) -> Result<usize, semiflow_core::SemiflowError> {
        match self {
            Self::Const(k) => k.apply_inactive_gamma_into(v, gamma, defined),
            Self::Array(k) => k.apply_inactive_gamma_into(v, gamma, defined),
        }
    }
}

// ---------------------------------------------------------------------------
// ObstacleGammaV8 pyclass
// ---------------------------------------------------------------------------

/// Inactive-set Γ = V″ primitive for obstacle problems (v8.3.0, ADR-0153, §4.1).
///
/// Exposes ``ObstacleChernoff::apply_inactive_gamma_into`` as a Python class.
///
/// .. `warning::`
///
///     **Honesty (NORMATIVE, math §44.5.bis)**: Γ is **discontinuous** at the
///     free boundary ``x*``.  ``defined[i] == False`` means Γ is **REFUSED**
///     (active set / contact line / one-node guard band) — it does NOT mean
///     ``gamma[i] == 0``.  Callers MUST consult ``defined`` before reading
///     ``gamma``.  Perpetual-put witness: Γ(S*⁺) ≈ 4.90, Γ(S*⁻) = 0.
///     No classical global Γ exists.  **D = 1 only** (multi-asset Γ deferred
///     §44.5.ter, ADR-0153).
///
/// Parameters
/// ----------
/// `domain_lo` : float
///     Left boundary (finite, < `domain_hi`).
/// `domain_hi` : float
///     Right boundary (finite, > `domain_lo`).
/// `n_grid` : int
///     Number of grid nodes (>= 4; required by the `Grid1D` constructor).
/// level : float, optional
///     Constant obstacle floor ``g ≡ level`` (keyword-only).
/// `obstacle_array` : array-like, optional
///     Per-node obstacle floor, length ``n_grid`` (keyword-only).
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` — invalid grid or length mismatch.
///     ``kind='OutOfDomain'`` — invalid domain or ``n_grid < 4``.
///     ``kind='NanInf'`` — obstacle array contains NaN / Inf.
#[pyclass(name = "ObstacleGammaV8")]
pub struct PyObstacleGammaV8 {
    kernel: GammaVariant,
    grid: Grid1D<f64>,
}

#[pymethods]
impl PyObstacleGammaV8 {
    #[new]
    #[pyo3(signature = (domain_lo, domain_hi, n_grid, *, level = f64::NAN,
                        obstacle_array = None))]
    fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        level: f64,
        obstacle_array: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        catch_panic_py!({
            validate_domain(domain_lo, domain_hi)?;
            if n_grid < 4 {
                return Err(new_pyerr(
                    "OutOfDomain",
                    "n_grid must be >= 4 (Grid1D requirement)",
                ));
            }
            let grid = Grid1D::new(domain_lo, domain_hi, n_grid).map_err(|e| from_core(&e))?;
            let diff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0_f64, grid);
            let kernel = build_gamma_kernel(diff, grid, level, obstacle_array)?;
            Ok(Self { kernel, grid })
        })
    }

    /// Compute inactive-set Γ = V″ on the OPEN continuation set.
    ///
    /// Returns ``(gamma, defined, count)`` where:
    ///
    /// - ``gamma`` — ``np.ndarray[float64]`` of length ``n_grid``.  Valid
    ///   (central-difference V″) where ``defined[i] == True``.
    /// - ``defined`` — ``np.ndarray[bool]`` of length ``n_grid``.
    ///   ``defined[i] == False`` means Γ is **REFUSED**.  Never interpret as "Γ=0".
    /// - ``count`` — number of nodes where Γ is defined.
    ///
    /// Parameters
    /// ----------
    /// v : array-like[float64]
    ///     Current value field, length ``n_grid``.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='GridMismatch'`` — ``len(v) != n_grid``.
    fn inactive_gamma<'py>(
        &self,
        py: Python<'py>,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        catch_panic_py!({
            let v_vec = extract_f64_vec(v)?;
            let n = self.grid.n;
            if v_vec.len() != n {
                return Err(new_pyerr("GridMismatch", "v length does not match n_grid"));
            }
            let v_fn = GridFn1D::new(self.grid, v_vec).map_err(|e| from_core(&e))?;
            let mut gamma_fn = v_fn.zeroed_like();
            let mut defined = vec![false; n];
            let count = self
                .kernel
                .apply_gamma(&v_fn, &mut gamma_fn, &mut defined)
                .map_err(|e| from_core(&e))?;
            // Marshal gamma -> float64 numpy array.
            let gamma_arr = gamma_fn.values.as_slice().to_pyarray(py);
            // Marshal defined -> bool numpy array (genuine numpy bool, NOT int8).
            // PyArray1::from_slice creates an owned 1-D array with the correct dtype.
            let defined_arr: Bound<'py, numpy::PyArray1<bool>> =
                numpy::PyArray1::<bool>::from_slice(py, &defined);
            // Return (gamma, defined, count) as a Python tuple.
            let tuple = pyo3::types::PyTuple::new(
                py,
                [
                    gamma_arr.into_any(),
                    defined_arr.into_any(),
                    count.into_pyobject(py)?.into_any(),
                ],
            )?;
            Ok(tuple)
        })
    }

    /// Return the number of grid nodes.
    fn size(&self) -> usize {
        self.grid.n
    }
}

// ---------------------------------------------------------------------------
// ObstacleNDV8 pyclass (D=2, axis-0-fastest layout)
// ---------------------------------------------------------------------------

/// D=2 projective-splitting obstacle evolver (v8.3.0, ADR-0153, §4.2).
///
/// Wraps ``ObstacleChernoffND`` with an isotropic (unit) diffusion inner
/// ``AnisotropicShiftChernoffND<f64, 2>`` (a=I, b=0, c=0).
///
/// .. note::
///
///     **ND layout (NORMATIVE, §3.1 V8_3_TIER3_BINDING_DESIGN.md)**
///     Input/output are flat axis-0-fastest buffers of length ``nx*ny``
///     (``idx(i,j) = j*nx + i``).  Pass ``v.ravel(order="F")`` if ``v`` has
///     shape ``(nx, ny)``; reshape the output with ``order="F"``.
///
/// .. note::
///
///     **Scope (§44.5.ter / ADR-0153)**: D=2 forward evolution only.
///     D=3, active-set adjoint, and inactive-set Γ remain D=1.
///     FFI/WASM ND deferred.  ``nx * ny >= 25`` required (5^D = 25 minimum
///     for ``AnisotropicShiftChernoffND``).
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     X-axis bounds (finite, xmin < xmax).
/// nx : int
///     Number of x-axis grid nodes (>= 5 recommended; must give nx*ny >= 25).
/// ymin, ymax : float
///     Y-axis bounds (finite, ymin < ymax).
/// ny : int
///     Number of y-axis grid nodes (>= 5 recommended).
/// level : float
///     Constant obstacle floor ``g ≡ level`` (finite).
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` — invalid grid or too few nodes.
///     ``kind='NanInf'`` — non-finite level.
///     ``kind='OutOfDomain'`` — invalid bounds.
#[pyclass(name = "ObstacleNDV8")]
pub struct PyObstacleNDV8 {
    level: f64,
    grid_nd: GridND<f64, 2>,
    nx: usize,
    ny: usize,
}

#[pymethods]
impl PyObstacleNDV8 {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        level: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            validate_domain(xmin, xmax)?;
            validate_domain(ymin, ymax)?;
            if !level.is_finite() {
                return Err(new_pyerr("NanInf", "level must be finite"));
            }
            let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| from_core(&e))?;
            let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| from_core(&e))?;
            let grid_nd = GridND::<f64, 2>::new([gx, gy]).map_err(|e| from_core(&e))?;
            Ok(Self {
                level,
                grid_nd,
                nx,
                ny,
            })
        })
    }

    /// Apply one Chernoff step ``Π_g ∘ S(Δτ)`` to the value field ``v``.
    ///
    /// Parameters
    /// ----------
    /// tau : float
    ///     Time step (> 0, finite).
    /// v : array-like[float64]
    ///     Flat axis-0-fastest buffer of length ``nx*ny``.
    ///
    /// Returns
    /// -------
    /// np.ndarray[float64]
    ///     Flat axis-0-fastest array of length ``nx*ny``.
    ///     Use ``out.reshape((nx, ny), order="F")`` to recover 2D layout.
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     ``kind='GridMismatch'`` — ``v`` length != ``nx*ny``.
    ///     ``kind='OutOfDomain'`` — ``tau <= 0`` or non-finite.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        tau: f64,
        v: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            validate_tau(tau)?;
            let n = self.nx * self.ny;
            let v_vec = extract_flat_f64_nd(v, n)?;
            let level = self.level;
            let grid_nd = self.grid_nd.clone();
            // GIL release around the ND sweep.
            let result = py.detach(|| run_nd_step_2d(grid_nd, &v_vec, level, tau));
            let out = result.map_err(|e| from_core(&e))?;
            Ok(out.as_slice().to_pyarray(py))
        })
    }

    /// Return the ``(nx, ny)`` shape tuple.
    fn shape(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }
}

// ---------------------------------------------------------------------------
// Type alias for the concrete D=2 obstacle kernel
// ---------------------------------------------------------------------------

type Nd2DKernel =
    ObstacleChernoffND<AnisotropicShiftChernoffND<f64, 2>, ConstantObstacle<f64>, f64, 2>;

/// Run one Chernoff step on a 2D axis-0-fastest value field (GIL-free).
///
/// Per-crate duplication required (ADR-0028 Amdt 2).
fn run_nd_step_2d(
    grid_nd: GridND<f64, 2>,
    v_vals: &[f64],
    level: f64,
    tau: f64,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let src = GridFnND::new(grid_nd.clone(), v_vals.to_vec())?;
    let mut dst = GridFnND::new(grid_nd.clone(), vec![0.0_f64; v_vals.len()])?;
    // Unit-isotropic diffusion inner: a = I, b = 0, c = 0.
    let inner = AnisotropicShiftChernoffND::<f64, 2>::new(
        |_x: &[f64; 2], a: &mut SquareMatrix<f64, 2>| {
            a.set(0, 0, 1.0);
            a.set(1, 1, 1.0);
            a.set(0, 1, 0.0);
            a.set(1, 0, 0.0);
        },
        |_x: &[f64; 2], b: &mut [f64; 2]| {
            b[0] = 0.0;
            b[1] = 0.0;
        },
        |_x: &[f64; 2]| 0.0_f64,
        grid_nd.clone(),
    )?;
    let obs = ConstantObstacle::new(level)?;
    let kernel: Nd2DKernel = ObstacleChernoffND::new(inner, obs)?;
    let mut scratch = ScratchPool::new();
    kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
    Ok(dst.values)
}

// ---------------------------------------------------------------------------
// Validation helpers (per-crate dup, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

fn validate_domain(lo: f64, hi: f64) -> PyResult<()> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(new_pyerr("NanInf", "domain bounds must be finite"));
    }
    if lo >= hi {
        return Err(new_pyerr("OutOfDomain", "domain_lo must be < domain_hi"));
    }
    Ok(())
}

fn validate_tau(tau: f64) -> PyResult<()> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "tau must be finite and > 0"));
    }
    Ok(())
}

fn extract_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected numpy.ndarray[float64] or sequence of floats",
        )
    })
}

/// Extract a flat Vec<f64> from a 1D or ND numpy array (Fortran-order ravel).
fn extract_flat_f64_nd(obj: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<f64>> {
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        if v.len() != expected {
            return Err(new_pyerr(
                "GridMismatch",
                "v flat length does not match grid size",
            ));
        }
        return Ok(v);
    }
    let kwargs = pyo3::types::PyDict::new(obj.py());
    kwargs.set_item("order", "F")?;
    let flat: Bound<'_, PyAny> = obj.call_method("ravel", (), Some(&kwargs))?;
    let v: Vec<f64> = flat
        .extract::<Vec<f64>>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("v must be numpy.ndarray[float64]"))?;
    if v.len() != expected {
        return Err(new_pyerr("GridMismatch", "v length mismatch after ravel"));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// GammaVariant builder
// ---------------------------------------------------------------------------

fn build_gamma_kernel(
    diff: DiffusionChernoff<f64>,
    grid: Grid1D<f64>,
    level: f64,
    obstacle_array: Option<&Bound<'_, PyAny>>,
) -> PyResult<GammaVariant> {
    if let Some(arr) = obstacle_array {
        let vals: Vec<f64> = arr.extract::<Vec<f64>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("obstacle_array must be float64 array")
        })?;
        if vals.len() != grid.n {
            return Err(new_pyerr(
                "GridMismatch",
                "obstacle_array length must equal n_grid",
            ));
        }
        let obs = ArrayObstacle::new(vals).map_err(|e| from_core(&e))?;
        let kernel: GammaArray = ObstacleChernoff::new(diff, obs).map_err(|e| from_core(&e))?;
        Ok(GammaVariant::Array(kernel))
    } else {
        if !level.is_finite() {
            return Err(new_pyerr("NanInf", "level must be finite"));
        }
        let obs = ConstantObstacle::new(level).map_err(|e| from_core(&e))?;
        let kernel: GammaConst = ObstacleChernoff::new(diff, obs).map_err(|e| from_core(&e))?;
        Ok(GammaVariant::Const(kernel))
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `ObstacleGammaV8` and `ObstacleNDV8` into `semiflow`.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyObstacleGammaV8>()?;
    m.add_class::<PyObstacleNDV8>()?;
    Ok(())
}
