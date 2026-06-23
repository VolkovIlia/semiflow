//! Wave P4 — `Subordinated1D` (M12).
//!
//! 1-D subordinated heat semigroup via Bochner-Phillips calculus.
//! Split from `time_dependent.rs` for suckless file-size compliance.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::too_many_arguments, clippy::unused_self)]

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use semiflow::{
    diffusion::DiffusionChernoff,
    subordinated::{
        GammaSubordinator, InverseGaussianSubordinator, LevySubordinator, StableSubordinator,
        SubordinatedChernoff,
    },
    ChernoffSemigroup, Grid1D, GridFn1D,
};

use crate::{
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
    time_dependent::{extract_f64_vec_td, unit_a_td, validate_params_td, validate_u0_td, zero_td},
};

// ---------------------------------------------------------------------------
// SubordinatorEnum — binding-side enum implementing LevySubordinator<f64>
//
// This avoids trait-object overhead (no Box<dyn>), keeps the type Send+Sync,
// and lets SubordinatedChernoff remain fully monomorphic.
// ---------------------------------------------------------------------------

/// Binding-side discriminant for the three Lévy subordinator backends.
///
/// `LevySubordinator` is a non-bindable trait (ADR-0111 §3); Python users
/// select the backend via the `subordinator=` string kwarg.
#[derive(Debug, Clone)]
pub enum SubordinatorEnum {
    Stable(StableSubordinator<f64>),
    Gamma(GammaSubordinator<f64>),
    InverseGaussian(InverseGaussianSubordinator<f64>),
}

impl LevySubordinator<f64> for SubordinatorEnum {
    fn laplace_exponent(&self, lambda: f64) -> f64 {
        match self {
            SubordinatorEnum::Stable(s) => s.laplace_exponent(lambda),
            SubordinatorEnum::Gamma(s) => s.laplace_exponent(lambda),
            SubordinatorEnum::InverseGaussian(s) => s.laplace_exponent(lambda),
        }
    }

    fn quadrature(&self, tau: f64, n_nodes: usize) -> (Vec<f64>, Vec<f64>) {
        match self {
            SubordinatorEnum::Stable(s) => s.quadrature(tau, n_nodes),
            SubordinatorEnum::Gamma(s) => s.quadrature(tau, n_nodes),
            SubordinatorEnum::InverseGaussian(s) => s.quadrature(tau, n_nodes),
        }
    }
}

// SubordinatorEnum inherits Send+Sync from all three inner types.
// StableSubordinator<f64>, GammaSubordinator<f64>, InverseGaussianSubordinator<f64>
// all contain only f64 scalars — trivially Send + Sync.

fn parse_subordinator(kind: &str, alpha: f64, c: f64) -> PyResult<SubordinatorEnum> {
    match kind {
        "stable" => {
            // alpha must be in (0, 1) strict.
            let sub = StableSubordinator::new(alpha).map_err(|e| from_core(&e))?;
            Ok(SubordinatorEnum::Stable(sub))
        }
        "gamma" => {
            // c > 0.
            let sub = GammaSubordinator::new(c).map_err(|e| from_core(&e))?;
            Ok(SubordinatorEnum::Gamma(sub))
        }
        "inverse_gaussian" => {
            // c > 0.
            let sub = InverseGaussianSubordinator::new(c).map_err(|e| from_core(&e))?;
            Ok(SubordinatorEnum::InverseGaussian(sub))
        }
        other => Err(new_pyerr(
            "Unsupported",
            &format!(
                "Unknown subordinator '{other}'. Must be 'stable', 'gamma', or 'inverse_gaussian'."
            ),
        )),
    }
}

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

type DiffUnit = DiffusionChernoff<f64>;
type SubordinatedKernel = SubordinatedChernoff<DiffUnit, SubordinatorEnum, f64>;

// ---------------------------------------------------------------------------
// Subordinated1D inner state
// ---------------------------------------------------------------------------

struct Subordinated1DInner {
    semigroup: ChernoffSemigroup<SubordinatedKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

// ---------------------------------------------------------------------------
// Subordinated1D pyclass (M12)
// ---------------------------------------------------------------------------

/// 1-D subordinated heat semigroup via Bochner-Phillips calculus (M12).
///
/// Wraps ``SubordinatedChernoff<DiffusionChernoff<f64>, Subordinator, f64>``
/// (Butko 2018 Thm 2.1, math.md §37, ADR-0103).  The Lévy subordinator is
/// selected by a string kwarg:
///
/// - ``"stable"`` — α-stable subordinator ``φ(λ) = λ^α``.  Requires ``alpha ∈ (0,1)``.
/// - ``"gamma"`` — Gamma subordinator ``φ(λ) = log(1 + λ/c)``.  Requires ``c > 0``.
/// - ``"inverse_gaussian"`` — IG subordinator ``φ(λ) = √(c²+2λ) − c``.  Requires ``c > 0``.
///
/// The subordinated semigroup approximates ``exp(t φ(−∂²))``:
/// - stable → fractional heat ``exp(−t(−∂²)^α)``.
/// - gamma → relativistic heat ``exp(−t log(I − Δ/c))``.
/// - `inverse_gaussian` → inverse-Gaussian heat.
///
/// Parameters
/// ----------
/// xmin : float
///     Left boundary.
/// xmax : float
///     Right boundary (must be > xmin).
/// n : int
///     Number of grid nodes (must be >= 4).
/// u0 : array-like
///     Initial condition; float64 array of length n.
/// subordinator : str, optional
///     Backend selector: ``"stable"`` (default), ``"gamma"``, or
///     ``"inverse_gaussian"``.
/// alpha : float, optional
///     Stability index for ``"stable"``; must be in ``(0, 1)``.  Default 0.5.
/// c : float, optional
///     Rate/drift parameter for ``"gamma"``/``"inverse_gaussian"``; must be > 0.
///     Default 1.0.
/// `n_nodes` : int, optional
///     Number of GL-32 quadrature nodes (1–32); default 32.
/// boundary : str, optional
///     Boundary policy for the inner diffusion kernel; default ``"reflect"``.
///
/// Raises
/// ------
/// `SemiflowError`
///     kind='`GridMismatch`' if grid or u0 are invalid.
///     kind='`NanInf`' if u0 contains NaN or Inf.
///     kind='`OutOfDomain`' if `alpha/c/n_nodes` are out of valid range, or
///         subordinator string is not recognised (kind='Unsupported').
#[pyclass(name = "Subordinated1D")]
pub struct PySubordinated1D {
    inner: Subordinated1DInner,
}

#[pymethods]
impl PySubordinated1D {
    #[new]
    #[pyo3(signature = (
        xmin, xmax, n, u0, *,
        subordinator = "stable",
        alpha = 0.5_f64,
        c = 1.0_f64,
        n_nodes = 32_usize,
        boundary = "reflect"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Bound<'_, PyAny>,
        subordinator: &str,
        alpha: f64,
        c: f64,
        n_nodes: usize,
        boundary: &str,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let policy = crate::boundary::parse_boundary(boundary)?;
            let u0_vec = extract_f64_vec_td(u0)?;
            let sub = parse_subordinator(subordinator, alpha, c)?;
            let inner = build_subordinated(xmin, xmax, n, &u0_vec, sub, n_nodes, policy)
                .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Return the approximation order (always 1 — Butko 2018 Theorem 2.1).
    fn order(&self) -> u32 {
        1
    }

    /// Advance state by time ``t`` using ``n_steps`` Chernoff iterations.
    ///
    /// GIL released during inner Rust compute (ADR-0031).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     kind='`OutOfDomain`' if t < 0, t is non-finite, or `n_steps` == 0.
    #[pyo3(signature = (t, n_steps = 100))]
    fn evolve(&mut self, py: Python<'_>, t: f64, n_steps: usize) -> PyResult<()> {
        catch_panic_py!({
            validate_params_td(n_steps, t)?;
            let func = self.inner.semigroup.func.clone();
            let grid = self.inner.current.grid;
            let values: Vec<f64> = self.inner.current.values.clone();
            let func_clone = func.clone();
            let result: Result<Vec<f64>, _> =
                py.detach(|| evolve_subordinated(func, grid, values, t, n_steps));
            self.inner.current.values = result.map_err(|e| from_core(&e))?;
            let sg = ChernoffSemigroup::new(func_clone, n_steps).map_err(|e| from_core(&e))?;
            self.inner.semigroup = sg;
            Ok(())
        })
    }

    /// Return current grid values as ``numpy.ndarray[float64]`` (copy).
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_panic_py!({ Ok(self.inner.current.values.as_slice().to_pyarray(py)) })
    }

    /// Number of grid nodes.
    fn __len__(&self) -> usize {
        self.inner.current.values.len()
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fn build_subordinated(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    sub: SubordinatorEnum,
    n_nodes: usize,
    boundary: semiflow::BoundaryPolicy,
) -> Result<Subordinated1DInner, semiflow::SemiflowError> {
    validate_u0_td(u0)?;
    let grid = Grid1D::new(xmin, xmax, n)?.with_boundary(boundary);
    let diff = DiffusionChernoff::new(unit_a_td, zero_td, zero_td, 1.0, grid);
    let wrapper = SubordinatedChernoff::with_n_nodes(diff, sub, n_nodes)?;
    let semigroup = ChernoffSemigroup::new(wrapper, 100)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(Subordinated1DInner { semigroup, current })
}

// ---------------------------------------------------------------------------
// GIL-free compute helper
// ---------------------------------------------------------------------------

fn evolve_subordinated(
    func: SubordinatedKernel,
    grid: Grid1D<f64>,
    values: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, values)?;
    Ok(sg.evolve(t, &f)?.values)
}
