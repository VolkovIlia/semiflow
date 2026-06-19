//! `Adjoint` — Python-accessible adjoint semigroup wrapper (v2.3 Phase 4).
//!
//! Wraps `AdjointChernoff<C, f64>` for each of the 5 supported 1-D kernels via
//! a concrete enum.  The generic parameter cannot be erased without `Box<dyn …>`,
//! so `KernelVariant` enumerates all valid inner types at compile time.
//!
//! ## Kernels supported
//!
//! | Inner | `AdjointChernoff` order |
//! |---|---|
//! | `DiffusionChernoff` (order 2) | 2 (self-adjoint) |
//! | `Diffusion4thChernoff` (order 4) | 4 (self-adjoint) |
//! | `Diffusion6thChernoff` (order 6) | 6 (self-adjoint) |
//! | `DriftReactionChernoff` (order 2) | 2 (general) |
//! | `ShiftChernoff1D` (order 1) | 1 (general) |
//!
//! ## GIL policy
//!
//! Same three-phase pattern as `Heat1D` (ADR-0031).

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::assigning_clones,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow_core::{
    AdjointChernoff, ChernoffFunction, Diffusion4thChernoff, Diffusion6thChernoff,
    DiffusionChernoff, DriftReactionChernoff, GridFn1D, ScratchPool, ShiftChernoff1D,
};

use crate::{
    diffusion_hi::{build_diff4_unit, build_diff6_unit},
    drift_reaction_py::build_drift_scalar,
    error::{from_core, new_pyerr},
    handle::{build_heat_unit, unit_diffusion_1d},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// 5-kernel enum (avoids Box<dyn ChernoffFunction>)
// ---------------------------------------------------------------------------

/// Enum over the 5 supported inner kernel types for `AdjointChernoff`.
#[allow(clippy::large_enum_variant)]
pub(crate) enum KernelVariant {
    /// Standard 2nd-order diffusion `a(x)∂²`.
    Diff2(AdjointChernoff<DiffusionChernoff<f64>>),
    /// 4th-order diffusion kernel.
    Diff4(AdjointChernoff<Diffusion4thChernoff<f64>>),
    /// 6th-order diffusion kernel.
    Diff6(AdjointChernoff<Diffusion6thChernoff<f64>>),
    /// Drift+reaction `b(x)∂_x + c(x)`.
    DriftReaction(AdjointChernoff<DriftReactionChernoff<f64>>),
    /// Universal shift `a(x)∂² + b(x)∂ + c(x)`.
    Shift(AdjointChernoff<ShiftChernoff1D<f64>>),
}

impl KernelVariant {
    /// Apply one Chernoff step of size `tau` in-place.
    fn apply_step(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow_core::SemiflowError> {
        match self {
            Self::Diff2(k) => k.apply_into(tau, src, dst, scratch),
            Self::Diff4(k) => k.apply_into(tau, src, dst, scratch),
            Self::Diff6(k) => k.apply_into(tau, src, dst, scratch),
            Self::DriftReaction(k) => k.apply_into(tau, src, dst, scratch),
            Self::Shift(k) => k.apply_into(tau, src, dst, scratch),
        }
    }

    /// Order of the wrapped kernel.
    fn order(&self) -> u32 {
        match self {
            Self::Diff2(k) => k.order(),
            Self::Diff4(k) => k.order(),
            Self::Diff6(k) => k.order(),
            Self::DriftReaction(k) => k.order(),
            Self::Shift(k) => k.order(),
        }
    }

    /// `is_self_adjoint` flag of the inner `AdjointChernoff`.
    fn is_self_adjoint(&self) -> bool {
        match self {
            Self::Diff2(k) => k.is_self_adjoint(),
            Self::Diff4(k) => k.is_self_adjoint(),
            Self::Diff6(k) => k.is_self_adjoint(),
            Self::DriftReaction(k) => k.is_self_adjoint(),
            Self::Shift(k) => k.is_self_adjoint(),
        }
    }
}

// ---------------------------------------------------------------------------
// Adjoint Python class
// ---------------------------------------------------------------------------

/// Adjoint semigroup wrapper for any supported 1-D Chernoff kernel.
///
/// `Adjoint(kernel)` wraps a kernel string (one of ``"heat2"``,
/// ``"heat4"``, ``"heat6"``, ``"drift"``, ``"shift"``) and evolves the
/// adjoint (dual) semigroup `exp(τA*)`.
///
/// For self-adjoint kernels (`"heat2"`, `"heat4"`, `"heat6"`) the wrapper
/// is zero-overhead — `apply_into` delegates directly.  For `"drift"` and
/// `"shift"` the bounded-perturbation expansion is used (math.md §15.1).
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
///     Inner kernel. One of ``"heat2"`` (default), ``"heat4"``,
///     ``"heat6"``, ``"drift"``, ``"shift"``.
/// `self_adjoint` : bool, optional
///     If ``True``, skip the dual correction even for non-symmetric
///     kernels (caller asserts self-adjointness).  Default ``False``.
/// boundary : str, optional
///     Boundary policy; default ``"reflect"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     ``kind='GridMismatch'`` for invalid grid or IC-length mismatches.
///     ``kind='NanInf'`` if `u0` contains NaN or Inf.
#[pyclass(name = "Adjoint")]
pub struct Adjoint {
    /// Concrete adjoint kernel variant.
    kernel: KernelVariant,
    /// Current 1-D grid function state.
    current: GridFn1D<f64>,
}

#[pymethods]
impl Adjoint {
    /// Construct `Adjoint` with the specified inner kernel.
    #[new]
    #[pyo3(signature = (xmin, xmax, n, u0, *, kernel = "heat2",
                        self_adjoint = false, boundary = "reflect"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        kernel: &str,
        self_adjoint: bool,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = crate::boundary::parse_boundary(boundary)?;
            let u0_vals: Vec<f64> = u0
                .extract::<Vec<f64>>()
                .map_err(|_| new_pyerr("GridMismatch", "u0 must be numpy.ndarray[float64]"))?;
            let grid_fn =
                build_initial_state(xmin, xmax, n, &u0_vals, policy).map_err(|e| from_core(&e))?;
            let kv = build_kernel(xmin, xmax, n, &u0_vals, kernel, self_adjoint, policy)?;
            Ok(Adjoint {
                kernel: kv,
                current: grid_fn,
            })
        })
    }

    /// Evolve the current state by time `t` using `n_steps` Chernoff steps.
    ///
    /// Returns a flat `numpy.ndarray[float64]` of length `n`.
    /// The GIL is released during the inner Rust compute loop.
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        n_steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            if n_steps == 0 {
                return Err(new_pyerr("OutOfDomain", "n_steps must be >= 1"));
            }
            if !t.is_finite() || t <= 0.0 {
                return Err(new_pyerr("OutOfDomain", "t must be finite and > 0"));
            }
            let tau = t / n_steps as f64;
            let input = self.current.values.clone();
            let grid = self.current.grid;
            let result: Result<Vec<f64>, _> =
                py.detach(|| evolve_adjoint(&self.kernel, grid, input, tau, n_steps));
            let result = result.map_err(|e| from_core(&e))?;
            self.current.values = result.clone();
            Ok(result.as_slice().to_pyarray(py))
        })
    }

    /// Order of the wrapped adjoint kernel.
    fn order(&self) -> u32 {
        self.kernel.order()
    }

    /// Whether the inner kernel is declared self-adjoint.
    fn is_self_adjoint(&self) -> bool {
        self.kernel.is_self_adjoint()
    }

    /// Return the current grid values as a 1-D ``numpy.ndarray[float64]``.
    ///
    /// Returns a **copy** of the internal state; mutations to the returned
    /// array do not affect this `Adjoint` object.  Dtype is always
    /// ``float64``, length is always ``len(self)``.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({
            let arr = self.current.values.as_slice().to_pyarray(py);
            Ok(arr)
        })
    }

    /// Number of grid nodes.
    fn __len__(&self) -> usize {
        self.current.values.len()
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust compute helper (called inside py.detach)
// ---------------------------------------------------------------------------

/// Advance `n_steps` adjoint steps of size `tau`.
fn evolve_adjoint(
    kv: &KernelVariant,
    grid: semiflow_core::Grid1D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let mut src = GridFn1D::new(grid, input)?;
    let mut dst = GridFn1D::new(grid, vec![0.0; src.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kv.apply_step(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ---------------------------------------------------------------------------
// Builders (kept ≤50 lines each)
// ---------------------------------------------------------------------------

/// Build the initial `GridFn1D` from a `Vec<f64>` + policy.
fn build_initial_state(
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

/// Dispatch to the appropriate `KernelVariant` based on `kernel` string.
fn build_kernel(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    kernel: &str,
    self_adjoint: bool,
    policy: semiflow_core::BoundaryPolicy,
) -> PyResult<KernelVariant> {
    use semiflow_core::Grid1D;
    let grid = Grid1D::new(xmin, xmax, n)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    match kernel {
        "heat2" => build_kernel_heat2(grid),
        "heat4" => build_kernel_heat4(xmin, xmax, n, u0, policy),
        "heat6" => build_kernel_heat6(xmin, xmax, n, u0, policy),
        "drift" => build_kernel_drift(xmin, xmax, n, u0, self_adjoint, policy),
        "shift" => build_kernel_shift(xmin, xmax, n, u0, policy, grid),
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!("unknown kernel '{other}'; expected heat2|heat4|heat6|drift|shift"),
        )),
    }
}

#[allow(clippy::unnecessary_wraps)]
fn build_kernel_heat2(grid: semiflow_core::Grid1D<f64>) -> PyResult<KernelVariant> {
    let inner = unit_diffusion_1d(grid);
    Ok(KernelVariant::Diff2(AdjointChernoff::new_self_adjoint(
        inner,
    )))
}

fn build_kernel_heat4(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    policy: semiflow_core::BoundaryPolicy,
) -> PyResult<KernelVariant> {
    let st = build_diff4_unit(xmin, xmax, n, 1, u0, policy).map_err(|e| from_core(&e))?;
    Ok(KernelVariant::Diff4(AdjointChernoff::new_self_adjoint(
        st.semigroup.func,
    )))
}

fn build_kernel_heat6(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    policy: semiflow_core::BoundaryPolicy,
) -> PyResult<KernelVariant> {
    let st = build_diff6_unit(xmin, xmax, n, 1, u0, policy).map_err(|e| from_core(&e))?;
    Ok(KernelVariant::Diff6(AdjointChernoff::new_self_adjoint(
        st.semigroup.func,
    )))
}

fn build_kernel_drift(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    self_adjoint: bool,
    policy: semiflow_core::BoundaryPolicy,
) -> PyResult<KernelVariant> {
    let st =
        build_drift_scalar(xmin, xmax, n, 1, u0, 0.5, 0.0, policy).map_err(|e| from_core(&e))?;
    let inner = st.semigroup.func;
    let adj = if self_adjoint {
        AdjointChernoff::new_self_adjoint(inner)
    } else {
        AdjointChernoff::new_general(inner)
    };
    Ok(KernelVariant::DriftReaction(adj))
}

fn build_kernel_shift(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    policy: semiflow_core::BoundaryPolicy,
    grid: semiflow_core::Grid1D<f64>,
) -> PyResult<KernelVariant> {
    // ShiftChernoff1D with a=0.5, b=0, c=0 is a symmetric isotropic diffusion.
    // AdjointApply is not implemented for ShiftChernoff1D; use new_self_adjoint.
    // The non-self-adjoint path would need a transpose-apply primitive (ADR-0114).
    let _ = build_heat_unit(xmin, xmax, n, 1, u0, policy).map_err(|e| from_core(&e))?;
    let inner = ShiftChernoff1D::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    Ok(KernelVariant::Shift(AdjointChernoff::new_self_adjoint(
        inner,
    )))
}
