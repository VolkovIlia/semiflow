//! v9.0.0 `PyO3` binding for `ReverseChernoff.value_and_grad` (math §51.5,
//! ADR-0156, Shift B).
//!
//! Exposes `ReverseHeat1D` — a thin wrapper around
//! `semiflow::ReverseChernoff<f64>` for the constant-a
//! `DiffusionChernoff` kernel (narrow scope, §51.5).
//!
//! ## API
//!
//! ```python
//! from semiflow import ReverseHeat1D
//!
//! rc = ReverseHeat1D(theta=0.4, xmin=-4.0, xmax=4.0, n_grid=24, n_steps=8)
//! u0     = np.exp(-x**2)         # shape (n_grid,), float64
//! target = np.zeros(n_grid)
//! value, grad = rc.value_and_grad(tau=0.05, u0=u0, target=target)
//! # value: float, grad: float (scalar θ-gradient of L² loss)
//! ```
//!
//! ## Error model
//!
//! Raises `SemiflowError` on invalid inputs.
//!
//! ## GIL policy
//!
//! `value_and_grad` runs entirely in pure Rust; the GIL is held throughout
//! (the compute is cheap — K=1 Dual forward pass over `n_steps` steps; see §51.4).
//! For K>>1 or large grids, GIL release may be warranted in a later revision.
//!
//! ## Scope (NARROW — §51.5)
//!
//! Constant-a `DiffusionChernoff` only; θ is the uniform diffusivity.
//! Variable-coefficient and nonlinear kernels are out of scope for v9.0.0.
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication of kernel construction required; no shared util with
//! semiflow-ffi/wasm.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::similar_names)]

use pyo3::prelude::*;
use semiflow::{
    error::SemiflowError, CheckpointSchedule, DiffusionChernoff, Dual, Grid1D, GridFn1D,
    InterpKind, ReverseChernoff,
};

use crate::{
    error::{from_core, new_pyerr},
    panic::catch_panic_py,
};

// ---------------------------------------------------------------------------
// Kernel construction helpers (per-crate duplicate, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

/// Build a `DiffusionChernoff<f64>` for constant `a(x) ≡ theta` (`CubicHermite`).
fn build_f64_kernel(theta: f64, grid: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::with_closure(move |_| theta, |_| 0.0_f64, |_| 0.0_f64, theta, grid)
}

/// Construct a `ReverseChernoff<f64>` from scalar params.
///
/// Both f64 and Dual<f64> grids are built here (`SepticHermite`, §46.5.bis).
fn build_reverse_chernoff(
    theta: f64,
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_steps: usize,
) -> Result<ReverseChernoff<f64>, SemiflowError> {
    // f64 grid.
    let grid_f64 = Grid1D::<f64>::new(xmin, xmax, n_grid)?.with_interp(InterpKind::CubicHermite);

    // Dual<f64> grid.
    let grid_dual =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(xmin), Dual::constant(xmax), n_grid)?
            .with_interp(InterpKind::CubicHermite);

    let kernel_f64 = build_f64_kernel(theta, grid_f64);

    let kernel_dual = DiffusionChernoff::<Dual<f64>>::with_closure(
        move |_: Dual<f64>| Dual::variable(theta),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        theta,
        grid_dual,
    );

    let schedule = CheckpointSchedule::sqrt_n(n_steps);
    Ok(ReverseChernoff::new(kernel_f64, kernel_dual, schedule))
}

// ---------------------------------------------------------------------------
// ReverseHeat1D pyclass
// ---------------------------------------------------------------------------

/// Reverse-mode AD evolver for constant-a 1-D heat (v9.0.0, math §51, ADR-0156).
///
/// Computes `(J, ∂J/∂θ)` where `J(θ) = ‖(F_θ(τ))ⁿ u₀ − target‖²` via the
/// K=1 forward-mode Dual path (§51.4; 0-ULP parity with forward AD by construction).
///
/// ## NARROW scope (§51.5)
///
/// Constant-a `DiffusionChernoff` ONLY; θ is the uniform diffusivity.
/// Variable-coefficient and nonlinear kernels are out of scope for v9.0.0.
///
/// Parameters
/// ----------
/// theta : float  — diffusivity parameter θ > 0 (finite).
/// xmin  : float  — left domain boundary.
/// xmax  : float  — right domain boundary (xmax > xmin).
/// `n_grid` : int   — grid nodes (>= 4).
/// `n_steps` : int  — Chernoff steps per `value_and_grad` call (>= 1).
///
/// Raises
/// ------
/// `SemiflowError`
///     `kind='GridMismatch'`  — `n_grid` < 4 or xmin >= xmax.
///     `kind='OutOfDomain'`   — theta <= 0, `n_steps` == 0.
#[pyclass(name = "ReverseHeat1D")]
pub struct PyReverseHeat1D {
    theta: f64,
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_steps: usize,
}

#[pymethods]
impl PyReverseHeat1D {
    /// Construct a `ReverseHeat1D` evolver.
    #[new]
    fn new(theta: f64, xmin: f64, xmax: f64, n_grid: usize, n_steps: usize) -> PyResult<Self> {
        catch_panic_py!({
            if !theta.is_finite() || theta <= 0.0 {
                return Err(new_pyerr(
                    "OutOfDomain",
                    "ReverseHeat1D: theta must be finite and > 0",
                ));
            }
            if n_steps == 0 {
                return Err(new_pyerr(
                    "OutOfDomain",
                    "ReverseHeat1D: n_steps must be >= 1",
                ));
            }
            // Eagerly validate grid by constructing it; discard the result.
            Grid1D::<f64>::new(xmin, xmax, n_grid).map_err(|e| from_core(&e))?;
            Ok(Self {
                theta,
                xmin,
                xmax,
                n_grid,
                n_steps,
            })
        })
    }

    /// Compute `(J, ∂J/∂θ)` for the scalar diffusivity parameter.
    ///
    /// **K=1 path**: uses the forward-mode `Dual<f64>` pass (§51.4), which
    /// guarantees **0-ULP parity** with the forward-mode reference.
    ///
    /// Parameters
    /// ----------
    /// tau : float         — per-step time increment (> 0, finite).
    /// u0 : array-like     — initial condition, float64, length `n_grid`.
    /// target : array-like — target state, float64, length `n_grid`.
    ///
    /// Returns
    /// -------
    /// (value, grad) : (float, float)
    ///     `value` is the L² loss `‖(F_θ(τ))ⁿ u₀ − target‖²`.
    ///     `grad` is `∂J/∂θ` (scalar, forward-mode Dual, 0-ULP vs core).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     `kind='OutOfDomain'`   — tau <= 0 or non-finite.
    ///     `kind='GridMismatch'`  — u0/target length != `n_grid`.
    ///     `kind='NanInf'`        — NaN/Inf in u0 or target.
    fn value_and_grad<'py>(
        &self,
        _py: Python<'py>,
        tau: f64,
        u0: &Bound<'py, pyo3::types::PyAny>,
        target: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<(f64, f64)> {
        catch_panic_py!({
            validate_tau_py(tau, "ReverseHeat1D.value_and_grad")?;
            let u0_vec = extract_f64_vec(u0)?;
            let target_vec = extract_f64_vec(target)?;
            let (rc, u0_fn, target_fn) = build_rc_inputs(
                self.theta,
                self.xmin,
                self.xmax,
                self.n_grid,
                self.n_steps,
                u0_vec,
                target_vec,
                "ReverseHeat1D.value_and_grad",
            )?;
            let (value, grad) = rc
                .value_and_grad_k1(tau, self.n_steps, &u0_fn, &target_fn)
                .map_err(|e| from_core(&e))?;
            Ok((value, grad))
        })
    }

    /// Compute `(J, grad_vec)` for a K-vector of parameters in ONE backward pass.
    ///
    /// **K-vector path (§51.9, ADR-0156 Amendment 1)**: runs the genuine cotangent
    /// backward sweep once, accumulating all K gradient components `∂J/∂θ_p` in the
    /// single backward walk — O(1) trajectory passes independent of K (vs O(K) for
    /// forward dual-AD). This is the capability tier asserted by `G_REVERSE_AD_ADVANTAGE`.
    ///
    /// ## 0-ULP parity
    ///
    /// For K=1, this method is byte-identical to `value_and_grad` (same arithmetic path).
    /// For K>1, the gradient vector is byte-identical to K independent `value_and_grad`
    /// calls (same backward sweep with K accumulations). All three bindings (Rust, `PyO3`,
    /// WASM) produce the same bits by `G_BINDING_REVERSE_AD_PARITY` sub-test 4.
    ///
    /// Parameters
    /// ----------
    /// tau : float         — per-step time increment (> 0, finite).
    /// `theta_vec` : list    — K diffusivity parameters, float64, len >= 1.
    /// u0 : array-like     — initial condition, float64, length `n_grid`.
    /// target : array-like — target state, float64, length `n_grid`.
    ///
    /// Returns
    /// -------
    /// (value, `grad_vec`) : (float, list[float])
    ///     `value` is the L² loss `‖(F_θ(τ))ⁿ u₀ − target‖²`.
    ///     `grad_vec` is `[∂J/∂θ_0, …, ∂J/∂θ_{K-1}]` (len K).
    ///
    /// Raises
    /// ------
    /// `SemiflowError`
    ///     `kind='OutOfDomain'`  — tau <= 0, non-finite, or `theta_vec` empty.
    ///     `kind='GridMismatch'` — u0/target length != `n_grid`.
    ///     `kind='NanInf'`       — NaN/Inf in u0, target, or `theta_vec`.
    fn value_and_grad_kvec<'py>(
        &self,
        _py: Python<'py>,
        tau: f64,
        theta_vec: &Bound<'py, pyo3::types::PyAny>,
        u0: &Bound<'py, pyo3::types::PyAny>,
        target: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<(f64, Vec<f64>)> {
        catch_panic_py!({
            validate_tau_py(tau, "ReverseHeat1D.value_and_grad_kvec")?;
            let theta_v = extract_f64_vec(theta_vec)?;
            if theta_v.is_empty() {
                return Err(new_pyerr(
                    "OutOfDomain",
                    "ReverseHeat1D.value_and_grad_kvec: theta_vec must be non-empty",
                ));
            }
            validate_finite(&theta_v, "theta_vec")?;
            let u0_vec = extract_f64_vec(u0)?;
            let target_vec = extract_f64_vec(target)?;
            // Build ReverseChernoff using the first theta (narrow scope: constant-a).
            let (rc, u0_fn, target_fn) = build_rc_inputs(
                self.theta,
                self.xmin,
                self.xmax,
                self.n_grid,
                self.n_steps,
                u0_vec,
                target_vec,
                "ReverseHeat1D.value_and_grad_kvec",
            )?;
            // K-vector backward sweep — ONE pass for all K gradients.
            let (value, grad_vec) = rc
                .value_and_grad(tau, self.n_steps, &u0_fn, &target_fn, &theta_v)
                .map_err(|e| from_core(&e))?;
            Ok((value, grad_vec))
        })
    }

    /// Return the diffusivity parameter θ.
    fn theta(&self) -> f64 {
        self.theta
    }

    /// Return the number of Chernoff steps.
    fn n_steps(&self) -> usize {
        self.n_steps
    }

    /// Return the number of grid nodes.
    fn n_grid(&self) -> usize {
        self.n_grid
    }
}

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

/// Validate `tau > 0` and finite; returns `OutOfDomain` on failure.
fn validate_tau_py(tau: f64, ctx: &str) -> PyResult<()> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(new_pyerr(
            "OutOfDomain",
            &format!("{ctx}: tau must be finite and > 0"),
        ));
    }
    Ok(())
}

/// Validate array lengths match `n_grid`, validate finite, build `ReverseChernoff` + `GridFn1D` pair.
#[allow(clippy::too_many_arguments)]
fn build_rc_inputs(
    theta: f64,
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_steps: usize,
    u0_vec: Vec<f64>,
    target_vec: Vec<f64>,
    ctx: &str,
) -> PyResult<(ReverseChernoff<f64>, GridFn1D<f64>, GridFn1D<f64>)> {
    if u0_vec.len() != n_grid {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("{ctx}: u0 length must equal n_grid"),
        ));
    }
    if target_vec.len() != n_grid {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("{ctx}: target length must equal n_grid"),
        ));
    }
    validate_finite(&u0_vec, "u0")?;
    validate_finite(&target_vec, "target")?;
    let rc =
        build_reverse_chernoff(theta, xmin, xmax, n_grid, n_steps).map_err(|e| from_core(&e))?;
    let grid = Grid1D::<f64>::new(xmin, xmax, n_grid)
        .map_err(|e| from_core(&e))?
        .with_interp(InterpKind::CubicHermite);
    let u0_fn = GridFn1D::new(grid, u0_vec).map_err(|e| from_core(&e))?;
    let target_fn = GridFn1D::new(grid, target_vec).map_err(|e| from_core(&e))?;
    Ok((rc, u0_fn, target_fn))
}

fn extract_f64_vec(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<f64>> {
    obj.extract::<Vec<f64>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected a numpy.ndarray[float64] or list of floats",
        )
    })
}

fn validate_finite(v: &[f64], name: &str) -> PyResult<()> {
    for &x in v {
        if !x.is_finite() {
            return Err(new_pyerr(
                "NanInf",
                &format!("ReverseHeat1D: {name} contains NaN or Inf ({x})"),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `ReverseHeat1D` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReverseHeat1D>()?;
    Ok(())
}
