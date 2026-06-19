//! `AdaptivePI` — Python-accessible PI adaptive integrator (v2.3 Phase 4).
//!
//! Wraps `semiflow_core::AdaptivePI<C, f64>` for each of the 5 supported 1-D
//! kernels via a concrete enum.  `AdaptivePI` is NOT a `ChernoffFunction` and
//! MUST NOT be wrapped in a `ChernoffSemigroup` — it integrates via adaptive
//! substeps, not a fixed-n product (see `adaptive.rs` module doc, ADR-0044).
//!
//! ## GIL policy
//!
//! Same three-phase pattern as `Heat1D` (ADR-0031).

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow_core::{
    AdaptiveOutcome, Diffusion4thChernoff, Diffusion6thChernoff, DiffusionChernoff,
    DriftReactionChernoff, GridFn1D, ShiftChernoff1D,
};

use crate::{
    diffusion_hi::{build_diff4_unit, build_diff6_unit},
    drift_reaction_py::build_drift_scalar,
    error::{from_core, new_pyerr},
    handle::unit_diffusion_1d,
    panic::catch_panic_py,
};

// Re-export the core type with f64 specialised.
type CoreAdaptivePI<C> = semiflow_core::AdaptivePI<C, f64>;

// ---------------------------------------------------------------------------
// 5-kernel enum (avoids Box<dyn ChernoffFunction>)
// ---------------------------------------------------------------------------

/// Enum over 5 `AdaptivePI` variants to avoid `Box<dyn ChernoffFunction>`.
#[allow(clippy::large_enum_variant)]
pub(crate) enum AdaptiveVariant {
    /// Standard 2nd-order diffusion.
    Diff2(CoreAdaptivePI<DiffusionChernoff<f64>>),
    /// 4th-order diffusion.
    Diff4(CoreAdaptivePI<Diffusion4thChernoff<f64>>),
    /// 6th-order diffusion.
    Diff6(CoreAdaptivePI<Diffusion6thChernoff<f64>>),
    /// Drift+reaction `b(x)∂_x + c(x)`.
    DriftReaction(CoreAdaptivePI<DriftReactionChernoff<f64>>),
    /// Universal shift `a(x)∂² + b(x)∂ + c(x)`.
    Shift(CoreAdaptivePI<ShiftChernoff1D<f64>>),
}

impl AdaptiveVariant {
    /// Set tolerances (builder-style mutating).
    fn set_tolerance(&mut self, abs: f64, rel: f64) {
        match self {
            Self::Diff2(k) => {
                k.tol_abs = abs;
                k.tol_rel = rel;
            }
            Self::Diff4(k) => {
                k.tol_abs = abs;
                k.tol_rel = rel;
            }
            Self::Diff6(k) => {
                k.tol_abs = abs;
                k.tol_rel = rel;
            }
            Self::DriftReaction(k) => {
                k.tol_abs = abs;
                k.tol_rel = rel;
            }
            Self::Shift(k) => {
                k.tol_abs = abs;
                k.tol_rel = rel;
            }
        }
    }

    /// Run adaptive integration. Returns `(Vec<f64>, steps_accepted, steps_rejected)`.
    fn evolve_adaptive(
        &mut self,
        t: f64,
        u0: &GridFn1D<f64>,
    ) -> Result<AdaptiveOutcome<GridFn1D<f64>>, semiflow_core::SemiflowError> {
        match self {
            Self::Diff2(k) => k.evolve_adaptive(t, u0),
            Self::Diff4(k) => k.evolve_adaptive(t, u0),
            Self::Diff6(k) => k.evolve_adaptive(t, u0),
            Self::DriftReaction(k) => k.evolve_adaptive(t, u0),
            Self::Shift(k) => k.evolve_adaptive(t, u0),
        }
    }
}

// ---------------------------------------------------------------------------
// AdaptivePI Python class
// ---------------------------------------------------------------------------

/// PI-controller adaptive-step integrator for any supported 1-D kernel.
///
/// `AdaptivePI` is **NOT** a fixed-step Chernoff product — it selects
/// substep sizes automatically to meet the mixed tolerance
/// ``tol_abs + tol_rel * ‖u‖``. See ADR-0044 and math.md §11.1.bis.
///
/// Parameters
/// ----------
/// xmin, xmax : float
///     Domain boundaries; `xmin < xmax`.
/// n : int
///     Number of grid nodes (>= 4).
/// u0 : numpy.ndarray[float64]
///     Initial condition, length `n`.
/// kernel : str, optional
///     Inner kernel; one of ``"heat2"`` (default), ``"heat4"``,
///     ``"heat6"``, ``"drift"``, ``"shift"``.
/// `tol_abs` : float, optional
///     Absolute component of the mixed tolerance (default 1e-6).
/// `tol_rel` : float, optional
///     Relative component (default 1e-4).
/// boundary : str, optional
///     Boundary policy; default ``"reflect"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` for invalid grid or IC-length mismatches.
///     ``kind='NanInf'`` if `u0` contains NaN or Inf.
///     ``kind='OutOfDomain'`` if `t <= 0` or non-finite.
///     ``kind='CflViolated'`` if the adaptive integrator exceeds `max_substeps`.
#[pyclass(name = "AdaptivePI")]
pub struct AdaptivePI {
    /// Concrete adaptive integrator variant.
    integrator: AdaptiveVariant,
    /// Current 1-D grid function (updated after each `evolve` call).
    current: GridFn1D<f64>,
}

#[pymethods]
impl AdaptivePI {
    /// Construct `AdaptivePI` for the given kernel.
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, kernel = "heat2",
                        tol_abs = 1e-6, tol_rel = 1e-4, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        kernel: &str,
        tol_abs: f64,
        tol_rel: f64,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = crate::boundary::parse_boundary(boundary)?;
            let u0_vals: Vec<f64> = u0
                .extract::<Vec<f64>>()
                .map_err(|_| new_pyerr("GridMismatch", "u0 must be numpy.ndarray[float64]"))?;
            let grid_fn =
                build_initial(xmin, xmax, n, &u0_vals, policy).map_err(|e| from_core(&e))?;
            let mut iv = build_adaptive(xmin, xmax, n, &u0_vals, kernel, policy)?;
            iv.set_tolerance(tol_abs, tol_rel);
            Ok(AdaptivePI {
                integrator: iv,
                current: grid_fn,
            })
        })
    }

    /// Evolve the current state by time `t` using adaptive PI substeps.
    ///
    /// Returns a flat `numpy.ndarray[float64]` of length `n`.
    /// Also exposes the substep diagnostics as `steps_accepted` and
    /// `steps_rejected` on the most recent call via :attr:`last_steps_accepted`
    /// and :attr:`last_steps_rejected`.
    ///
    /// The GIL is released during the adaptive integration loop.
    #[pyo3(signature = (t))]
    fn evolve<'py>(&mut self, py: Python<'py>, t: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            if !t.is_finite() || t <= 0.0 {
                return Err(new_pyerr("OutOfDomain", "t must be finite and > 0"));
            }
            let input = self.current.clone();
            // Safety: AdaptiveVariant and GridFn1D are Send.
            let result: Result<AdaptiveOutcome<GridFn1D<f64>>, _> =
                py.detach(|| self.integrator.evolve_adaptive(t, &input));
            let outcome = result.map_err(|e| from_core(&e))?;
            self.current = outcome.final_state.clone();
            Ok(outcome.final_state.values.as_slice().to_pyarray(py))
        })
    }

    /// Number of grid nodes.
    fn __len__(&self) -> usize {
        self.current.values.len()
    }
}

// ---------------------------------------------------------------------------
// Private builders (each ≤50 lines)
// ---------------------------------------------------------------------------

/// Build the initial `GridFn1D` from validated `u0`.
fn build_initial(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    policy: semiflow_core::BoundaryPolicy,
) -> Result<GridFn1D<f64>, semiflow_core::SemiflowError> {
    use semiflow_core::Grid1D;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(policy);
    GridFn1D::new(grid, u0.to_vec())
}

/// Dispatch to the appropriate `AdaptiveVariant` based on `kernel` string.
fn build_adaptive(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    kernel: &str,
    policy: semiflow_core::BoundaryPolicy,
) -> PyResult<AdaptiveVariant> {
    use semiflow_core::{AdaptivePI as CorePI, Grid1D};
    let grid = Grid1D::new(xmin, xmax, n)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    match kernel {
        "heat2" => {
            let inner = unit_diffusion_1d(grid);
            Ok(AdaptiveVariant::Diff2(CorePI::new(inner)))
        }
        "heat4" => {
            let st = build_diff4_unit(xmin, xmax, n, 1, u0, policy).map_err(|e| from_core(&e))?;
            Ok(AdaptiveVariant::Diff4(CorePI::new(st.semigroup.func)))
        }
        "heat6" => {
            let st = build_diff6_unit(xmin, xmax, n, 1, u0, policy).map_err(|e| from_core(&e))?;
            Ok(AdaptiveVariant::Diff6(CorePI::new(st.semigroup.func)))
        }
        "drift" => {
            let st = build_drift_scalar(xmin, xmax, n, 1, u0, 0.5, 0.0, policy)
                .map_err(|e| from_core(&e))?;
            Ok(AdaptiveVariant::DriftReaction(CorePI::new(
                st.semigroup.func,
            )))
        }
        "shift" => {
            let inner = ShiftChernoff1D::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
            Ok(AdaptiveVariant::Shift(CorePI::new(inner)))
        }
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!("unknown kernel '{other}'; expected heat2|heat4|heat6|drift|shift"),
        )),
    }
}
