//! `ObstacleChernoff` — projective-splitting Chernoff for obstacle problems (math §44).
//!
//! Exposes `Π_g ∘ S(Δτ)` as a Python class backed by a concrete
//! `ObstacleChernoff<Inner, ObstacleVariant, f64>` where Inner is one of:
//! - `DiffusionChernoff<f64>` — when b=0, c=0 (fast path, backward compat)
//! - `StrangSplit<DiffusionChernoff, DriftReactionChernoff>` — when b≠0 or c≠0
//!
//! ## Obstacle variants
//!
//! | Python kwarg | Core type | Description |
//! |---|---|---|
//! | `level=` float | `ConstantObstacle<f64>` | Flat floor `g ≡ level` |
//! | `obstacle_array=` ndarray | `ArrayObstacle` | Per-node floor from numpy array |
//!
//! ## Generator
//!
//! ``L = a·∂_xx + b·∂_x + c·`` with constant coefficients.
//! `b` and `c` default to 0.0. When both are zero the fast-path pure-diffusion
//! kernel is used; otherwise a `StrangSplit<DiffusionChernoff, DriftReactionChernoff>`
//! is constructed (obstacle projection caps global order to 1 regardless).
//!
//! ## GIL policy (ADR-0031)
//!
//! Three-phase: validate + copy under GIL → `py.detach` compute → write result.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::assigning_clones,
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::unused_self
)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    ChernoffFunction, ConstantObstacle, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D,
    ObstacleChernoff, ScratchPool, StrangSplit,
};

use crate::{
    error::{from_core, new_pyerr},
    obstacle_build,
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// ArrayObstacle — per-node floor from a pre-sampled Vec<f64>
// ---------------------------------------------------------------------------

/// Array-backed obstacle `g(x_i) = values[i]` (1-D, node-indexed).
#[derive(Debug, Clone)]
pub(crate) struct ArrayObstacle {
    pub(crate) values: Vec<f64>,
}

impl ArrayObstacle {
    /// Validate that `values` has no NaN/Inf.
    pub(crate) fn new(values: Vec<f64>) -> Result<Self, semiflow::SemiflowError> {
        for &v in &values {
            if !v.is_finite() {
                return Err(semiflow::SemiflowError::DomainViolation {
                    what: "ArrayObstacle: obstacle_array contains NaN or Inf",
                    value: v,
                });
            }
        }
        Ok(Self { values })
    }
}

impl semiflow::Obstacle<f64> for ArrayObstacle {
    fn value_at(&self, _point: &[f64]) -> f64 {
        0.0
    }

    fn project_in_place(&self, dst: &mut GridFn1D<f64>) -> Result<(), semiflow::SemiflowError> {
        if dst.values.len() != self.values.len() {
            return Err(semiflow::SemiflowError::DomainViolation {
                what: "ArrayObstacle::project_in_place: obstacle length != grid n",
                value: self.values.len() as f64,
            });
        }
        for (v, &g) in dst.values.iter_mut().zip(self.values.iter()) {
            if *v < g {
                *v = g;
            }
        }
        Ok(())
    }

    fn active_set_into(
        &self,
        w: &GridFn1D<f64>,
        active: &mut [bool],
    ) -> Result<(), semiflow::SemiflowError> {
        if active.len() != w.grid.n || active.len() != self.values.len() {
            return Err(semiflow::SemiflowError::DomainViolation {
                what: "ArrayObstacle::active_set_into: length mismatch",
                value: active.len() as f64,
            });
        }
        for i in 0..w.grid.n {
            active[i] = w.values[i] > self.values[i];
        }
        Ok(())
    }

    fn dim(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------
// Type aliases for concrete kernel combinations
// ---------------------------------------------------------------------------

pub(crate) type DiffUnit = DiffusionChernoff<f64>;
type DrUnit = DriftReactionChernoff<f64>;
pub(crate) type StrangUnit = StrangSplit<DiffUnit, DrUnit, f64>;
pub(crate) type ConstKernel = ObstacleChernoff<DiffUnit, ConstantObstacle<f64>, f64>;
pub(crate) type ArrayKernel = ObstacleChernoff<DiffUnit, ArrayObstacle, f64>;
pub(crate) type StrangConstKernel = ObstacleChernoff<StrangUnit, ConstantObstacle<f64>, f64>;
pub(crate) type StrangArrayKernel = ObstacleChernoff<StrangUnit, ArrayObstacle, f64>;

// ---------------------------------------------------------------------------
// ObstacleVariant — erases concrete obstacle+inner type (no dyn needed)
// ---------------------------------------------------------------------------

/// Concrete obstacle kernel dispatch — all four combos of obstacle×inner.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub(crate) enum ObstacleVariant {
    /// Pure diffusion, constant obstacle (default / fast path).
    Const(ConstKernel),
    /// Pure diffusion, array obstacle.
    Array(ArrayKernel),
    /// Strang(diffusion + drift + reaction), constant obstacle.
    Strang(StrangConstKernel),
    /// Strang(diffusion + drift + reaction), array obstacle.
    StrangArray(StrangArrayKernel),
}

impl ObstacleVariant {
    pub(crate) fn apply_step(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow::SemiflowError> {
        match self {
            Self::Const(k) => k.apply_into(tau, src, dst, scratch),
            Self::Array(k) => k.apply_into(tau, src, dst, scratch),
            Self::Strang(k) => k.apply_into(tau, src, dst, scratch),
            Self::StrangArray(k) => k.apply_into(tau, src, dst, scratch),
        }
    }

    pub(crate) fn apply_active_set_adjoint(
        &self,
        tau: f64,
        w_fwd: &GridFn1D<f64>,
        lam: &GridFn1D<f64>,
        lam_next: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow::SemiflowError> {
        match self {
            Self::Const(k) => k.apply_active_set_adjoint_into(tau, w_fwd, lam, lam_next, scratch),
            Self::Array(k) => k.apply_active_set_adjoint_into(tau, w_fwd, lam, lam_next, scratch),
            Self::Strang(k) => k.apply_active_set_adjoint_into(tau, w_fwd, lam, lam_next, scratch),
            Self::StrangArray(k) => {
                k.apply_active_set_adjoint_into(tau, w_fwd, lam, lam_next, scratch)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inner state for ObstacleChernoff pyclass
// ---------------------------------------------------------------------------

pub(crate) struct ObstaclePyInner {
    pub(crate) kernel: ObstacleVariant,
    pub(crate) current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// ObstacleChernoff Python class
// ---------------------------------------------------------------------------

/// 1-D obstacle / variational-inequality Chernoff evolver (math §44).
///
/// Generator: ``L = a u_xx + b u_x + c u`` (constant coefficients).
///
/// When ``b == 0`` and ``c == 0`` (default) the fast-path
/// ``DiffusionChernoff`` kernel is used.  For ``b ≠ 0`` or ``c ≠ 0`` a
/// Strang-split ``D(τ/2)∘R(τ)∘D(τ/2)`` is used (global order 2 for the
/// linear part; obstacle projection caps to 1 regardless of order).
///
/// Parameters
/// ----------
/// xmin, xmax : float  Grid endpoints.
/// n : int             Grid nodes (≥ 4).
/// u0 : array-like     Initial condition, length n, float64.
/// a : float           Diffusion coefficient (> 0, default 1.0).
/// b : float           Drift coefficient (default 0.0).
/// c : float           Reaction coefficient (default 0.0).
/// level : float       Constant obstacle ``g ≡ level``.
/// `obstacle_array` : array-like  Per-node obstacle, length n.
///
/// Raises
/// ------
/// `SemiflowError`   kind='`GridMismatch`', '`NanInf`', or '`OutOfDomain`' on bad params.
#[pyclass(name = "ObstacleChernoff")]
pub struct PyObstacleChernoff {
    inner: ObstaclePyInner,
}

#[pymethods]
impl PyObstacleChernoff {
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, a = 1.0, b = 0.0, c = 0.0,
                        level = f64::NAN, obstacle_array = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        a: f64,
        b: f64,
        c: f64,
        level: f64,
        obstacle_array: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let u0_vec = extract_f64_vec(u0)?;
            validate_u0(&u0_vec).map_err(|e| from_core(&e))?;
            validate_a(a)?;
            validate_bc(b, c)?;
            let inner =
                obstacle_build::build_inner(xmin, xmax, n, &u0_vec, a, b, c, level, obstacle_array)
                    .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Approximation order (always 1; §44.4 projection cap).
    fn order(&self) -> u32 {
        1
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// Returns new grid values as ``numpy.ndarray[float64]`` (copy).
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            validate_params(t, n_steps)?;
            let grid = self.inner.current.grid;
            let input: Vec<f64> = self.inner.current.values.clone();
            let kernel = self.inner.kernel.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| run_evolve(kernel, grid, input, t, n_steps));
            let out = result.map_err(|e| from_core(&e))?;
            self.inner.current.values = out.clone();
            Ok(out.as_slice().to_pyarray(py))
        })
    }

    /// Return current grid values as ``numpy.ndarray[float64]`` (copy).
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Active-set adjoint step (math §44.5). Raises ``Unsupported`` — see docs.
    fn evolve_active_set_adjoint<'py>(
        &self,
        py: Python<'py>,
        w_fwd: &Bound<'_, PyAny>,
        lam: &Bound<'_, PyAny>,
        tau: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            let w_vec = extract_f64_vec(w_fwd)?;
            let lam_vec = extract_f64_vec(lam)?;
            let grid = self.inner.current.grid;
            if w_vec.len() != grid.n || lam_vec.len() != grid.n {
                return Err(new_pyerr(
                    "GridMismatch",
                    "w_fwd and lam must have length n",
                ));
            }
            if !tau.is_finite() || tau <= 0.0 {
                return Err(new_pyerr("OutOfDomain", "tau must be finite and > 0"));
            }
            let kernel = self.inner.kernel.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| run_adjoint_step(kernel, grid, w_vec, lam_vec, tau));
            let out = result.map_err(|e| from_core(&e))?;
            Ok(out.as_slice().to_pyarray(py))
        })
    }

    /// Number of grid nodes.
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }
}

// ---------------------------------------------------------------------------
// GIL-free compute helpers
// ---------------------------------------------------------------------------

fn run_evolve(
    kernel: ObstacleVariant,
    grid: Grid1D<f64>,
    input: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    #[allow(clippy::cast_precision_loss)]
    let tau = t / n_steps as f64;
    let mut src = GridFn1D::new(grid, input)?;
    let mut dst = src.zeroed_like();
    let mut scratch = ScratchPool::new();
    for _ in 0..n_steps {
        kernel.apply_step(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

fn run_adjoint_step(
    kernel: ObstacleVariant,
    grid: Grid1D<f64>,
    w_fwd_vals: Vec<f64>,
    lam_vals: Vec<f64>,
    tau: f64,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let w_fwd = GridFn1D::new(grid, w_fwd_vals)?;
    let lam = GridFn1D::new(grid, lam_vals)?;
    let mut lam_next = lam.zeroed_like();
    let mut scratch = ScratchPool::new();
    kernel.apply_active_set_adjoint(tau, &w_fwd, &lam, &mut lam_next, &mut scratch)?;
    Ok(lam_next.values)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_params(t: f64, n_steps: usize) -> PyResult<()> {
    if n_steps == 0 {
        return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn validate_u0(u0: &[f64]) -> Result<(), semiflow::SemiflowError> {
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

fn validate_a(a: f64) -> PyResult<()> {
    if !a.is_finite() || a <= 0.0 {
        return Err(new_pyerr("OutOfDomain", "a must be finite and > 0"));
    }
    Ok(())
}

fn validate_bc(b: f64, c: f64) -> PyResult<()> {
    if !b.is_finite() {
        return Err(new_pyerr("OutOfDomain", "b must be finite"));
    }
    if !c.is_finite() {
        return Err(new_pyerr("OutOfDomain", "c must be finite"));
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

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register the obstacle pyclass into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyObstacleChernoff>()?;
    Ok(())
}
