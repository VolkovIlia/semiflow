//! `PyO3` binding for Dual-AD Greeks (Δ, Γ) of 1D unit-diffusion heat.
//!
//! Implements `EvolverHeat1DGreeksV3` (§3, `V8_PHASE5_BINDING_GREEKS_DESIGN.md`)
//! and `KilledDirichlet1D` (TIER 2, §7 — thin wrapper, opportunistic).
//!
//! ## AD path
//!
//! A single `Dual<Dual<f64>>` hyper-dual sweep computes value/Δ/Γ together
//! (§1, math.md §46.4).  θ is the diffusion-scale; only the `scale` arg is
//! seeded — `a`/`a'`/`a''` fn-ptrs carry constant duals (tangents = 0).
//!
//! - `value[i]  = u.values[i].value.value`
//! - `delta[i]  = u.values[i].tangent.value`   (∂u/∂θ, outer tangent)
//! - `gamma[i]  = u.values[i].tangent.tangent` (∂²u/∂θ², hyper-dual)
//!
//! Profile: `[profile.release-ffi]` (`panic = "unwind"`) — same as v3.rs.
//! GIL: released via `py.detach` three-phase pattern (ADR-0031).
//! Errors: `SemiflowError` (kind-discriminated, same error module as v3.rs).
//!
//! ADR-0028 Amendment 2: per-crate duplication — no shared util between
//! semiflow-ffi/semiflow-py/semiflow-wasm.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::assigning_clones, clippy::type_complexity)]

use numpy::ToPyArray;
use pyo3::prelude::*;
use semiflow_core::{
    dual::Dual, DiffusionChernoff, Evolver, Grid1D, GridFn1D, KilledDirichletChernoff,
};

use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;

// ---------------------------------------------------------------------------
// Type alias for readability
// ---------------------------------------------------------------------------

type HyperDual = Dual<Dual<f64>>;

// ---------------------------------------------------------------------------
// GreeksInner — heap-owned state for EvolverHeat1DGreeksV3
// ---------------------------------------------------------------------------

/// Inner Rust state: hyper-dual evolver + value-only chain state + θ seed.
struct GreeksInner {
    /// Number of Chernoff iterations.
    n_chernoff: usize,
    /// Grid geometry (f64; reused for value extraction).
    grid: Grid1D<f64>,
    /// Diffusion-scale θ at which Δ/Γ are evaluated.
    theta: f64,
    /// Current state (value-only, f64; updated after each `.greeks()` call).
    current: Vec<f64>,
}

// ---------------------------------------------------------------------------
// EvolverHeat1DGreeksV3 pyclass
// ---------------------------------------------------------------------------

/// Hyper-dual Greeks evolver for unit-diffusion heat (v8.0.0, ADR-0133 A3).
///
/// Computes `(value, delta, gamma)` — the solution and its first/second
/// derivatives w.r.t. the diffusion-scale parameter θ — via a single
/// `Dual<Dual<f64>>` hyper-dual sweep (math §46.4).
///
/// Parameters
/// ----------
/// `domain_lo` : float
///     Left boundary (finite).
/// `domain_hi` : float
///     Right boundary (finite, > `domain_lo`).
/// `n_grid` : int
///     Number of grid nodes (>= 4).
/// u0 : array-like
///     Initial state, 1-D float64, length `n_grid`.
/// `n_chernoff` : int
///     Chernoff iteration count (>= 1).
/// `scale_theta` : float, optional
///     Diffusion-scale θ (default 0.5).
///
/// Raises
/// ------
/// `SemiflowError`
///     `kind='GridMismatch'` — invalid geometry or len(u0) != `n_grid`.
///     `kind='NanInf'`       — non-finite value in u0 or `scale_theta`.
///     `kind='OutOfDomain'`  — `n_chernoff` == 0.
#[pyclass(name = "EvolverHeat1DGreeksV3")]
pub struct PyEvolverHeat1DGreeksV3 {
    inner: GreeksInner,
}

#[pymethods]
impl PyEvolverHeat1DGreeksV3 {
    #[new]
    #[pyo3(signature = (domain_lo, domain_hi, n_grid, u0, n_chernoff, scale_theta = 0.5))]
    fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        u0: &Bound<'_, pyo3::types::PyAny>,
        n_chernoff: usize,
        scale_theta: f64,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let u0_vec = extract_f64_vec(u0)?;
            let inner = build_greeks_inner(
                domain_lo,
                domain_hi,
                n_grid,
                &u0_vec,
                n_chernoff,
                scale_theta,
            )
            .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Advance by `t` and return `(value, delta, gamma)` as three numpy arrays.
    ///
    /// Each array has length `size()` and dtype `float64`.  The internal
    /// current state is updated to the primal-value result.  The GIL is
    /// released during the hyper-dual Chernoff sweep (ADR-0031).
    ///
    /// Parameters
    /// ----------
    /// t : float
    ///     Time step (>= 0, finite).
    ///
    /// Returns
    /// -------
    /// tuple[np.ndarray, np.ndarray, np.ndarray]
    ///     `(value, delta, gamma)` — each float64 array of length `n_grid`.
    fn greeks<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray1<f64>>,
    )> {
        catch_panic_py!({
            validate_t(t)?;
            // --- Phase 1: clone Send data under GIL ---
            let n_chernoff = self.inner.n_chernoff;
            let grid_f64 = self.inner.grid;
            let theta = self.inner.theta;
            let u0_vals = self.inner.current.clone();

            // --- Phase 2: hyper-dual sweep (GIL released) ---
            let result =
                py.detach(|| run_hyper_dual_sweep(grid_f64, &u0_vals, n_chernoff, theta, t));

            let (values, deltas, gammas) = result.map_err(|e| from_core(&e))?;

            // --- Phase 3: update chain state + marshal to numpy (under GIL) ---
            self.inner.current = values.clone();
            let arr_v = values.as_slice().to_pyarray(py);
            let arr_d = deltas.as_slice().to_pyarray(py);
            let arr_g = gammas.as_slice().to_pyarray(py);
            Ok((arr_v, arr_d, arr_g))
        })
    }

    /// Return the number of grid nodes.
    fn size(&self) -> usize {
        self.inner.current.len()
    }

    /// Return the Chernoff iteration count.
    fn n_chernoff(&self) -> usize {
        self.inner.n_chernoff
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust hyper-dual sweep (runs GIL-off)
// ---------------------------------------------------------------------------

/// Run one `Dual<Dual<f64>>` Chernoff sweep; return demultiplexed buffers.
///
/// `DiffusionChernoff<F>` implements `ChernoffFunction` only for `F = f64`
/// (SIMD path, ADR-0018). For `F = HyperDual` we call `apply_f` directly in a
/// loop — the same scalar generic path used in `dual.rs` tests (§46.4).
///
/// θ-seeding: unit diffusion means `a(x) = θ`. Since fn-ptrs cannot capture θ,
/// we use `with_closure` to capture `Dual::variable(Dual::variable(theta))` and
/// return it from `a`. `a'` and `a''` return the constant zero dual.
fn run_hyper_dual_sweep(
    grid_f64: Grid1D<f64>,
    u0_f64: &[f64],
    n_chernoff: usize,
    theta: f64,
    t: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), semiflow_core::SemiflowError> {
    // Build hyper-dual grid (same geometry, dual field).
    let lo = Dual::constant(Dual::constant(grid_f64.xmin));
    let hi = Dual::constant(Dual::constant(grid_f64.xmax));
    let grid_hd = Grid1D::<HyperDual>::new_generic(lo, hi, grid_f64.n)?;

    // Seed θ: both inner and outer tangents = 1 → captures ∂/∂θ and ∂²/∂θ².
    let theta_seeded: HyperDual = Dual::variable(Dual::variable(theta));

    // Build the hyper-dual DiffusionChernoff with θ-seeded `a` closure.
    // a'(x) = 0, a''(x) = 0 (unit diffusion, constant a = θ).
    let chernoff = DiffusionChernoff::with_closure(
        move |_: HyperDual| theta_seeded,
        |_: HyperDual| Dual::constant(Dual::constant(0.0_f64)),
        |_: HyperDual| Dual::constant(Dual::constant(0.0_f64)),
        theta, // a_norm_bound: f64 — primal value of θ
        grid_hd,
    );

    // u0 is θ-independent: all tangents = 0.
    let u0_vals_hd: Vec<HyperDual> = u0_f64
        .iter()
        .map(|&v| Dual::constant(Dual::constant(v)))
        .collect();
    let u0_hd: GridFn1D<HyperDual> = GridFn1D::new_generic(grid_hd, u0_vals_hd)?;

    // Time-step per iteration: τ = t / n_chernoff (constant dual, tangent = 0).
    #[allow(clippy::cast_precision_loss)]
    let tau_f64 = t / (n_chernoff as f64);
    let tau: HyperDual = Dual::constant(Dual::constant(tau_f64));

    // Iterate apply_f n_chernoff times.
    let mut current: GridFn1D<HyperDual> = u0_hd;
    for _ in 0..n_chernoff {
        current = chernoff.apply_f(tau, &current)?;
    }
    Ok(demux_hyper_dual(&current.values))
}

/// Demultiplex a hyper-dual buffer into (value, delta, gamma) f64 vecs (§1, math §46.4).
///
/// - `value[i]  = h.value.value`   — primal
/// - `delta[i]  = h.tangent.value` — ∂u/∂θ (outer tangent primal)
/// - `gamma[i]  = h.tangent.tangent` — ∂²u/∂θ² (hyper component)
fn demux_hyper_dual(hds: &[HyperDual]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = hds.len();
    let mut values = Vec::with_capacity(n);
    let mut deltas = Vec::with_capacity(n);
    let mut gammas = Vec::with_capacity(n);
    for hd in hds {
        values.push(hd.value.value);
        deltas.push(hd.tangent.value);
        gammas.push(hd.tangent.tangent);
    }
    (values, deltas, gammas)
}

// ---------------------------------------------------------------------------
// Builder helper
// ---------------------------------------------------------------------------

fn build_greeks_inner(
    lo: f64,
    hi: f64,
    n_grid: usize,
    u0: &[f64],
    n_chernoff: usize,
    theta: f64,
) -> Result<GreeksInner, semiflow_core::SemiflowError> {
    validate_u0_finite(u0)?;
    validate_theta(theta)?;
    if n_chernoff == 0 {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "n_chernoff must be >= 1",
            value: 0.0,
        });
    }
    let grid = Grid1D::new(lo, hi, n_grid)?;
    Ok(GreeksInner {
        n_chernoff,
        grid,
        theta,
        current: u0.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn extract_f64_vec(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        return Ok(v);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "u0 must be a numpy.ndarray[float64] or a sequence of floats",
    ))
}

fn validate_t(t: f64) -> PyResult<()> {
    if !t.is_finite() || t < 0.0 {
        return Err(new_pyerr("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn validate_u0_finite(u0: &[f64]) -> Result<(), semiflow_core::SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(semiflow_core::SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}

fn validate_theta(theta: f64) -> Result<(), semiflow_core::SemiflowError> {
    if !theta.is_finite() {
        return Err(semiflow_core::SemiflowError::DomainViolation {
            what: "scale_theta must be finite",
            value: theta,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// TIER 2: KilledDirichlet1D — thin PyO3 wrapper (~35 LoC body)
// ---------------------------------------------------------------------------

/// `PyO3` wrapper for `KilledDirichletChernoff` (TIER 2, §7).
///
/// Crank–Nicolson Cayley map of the killed Dirichlet generator on
/// `[domain_lo, domain_hi]` with absorbing endpoints (u|∂R = 0).
/// Order-2 (math §44.ter, ADR-0135 Amendment 2).
///
/// Parameters
/// ----------
/// `domain_lo`, `domain_hi` : float  — domain boundaries.
/// `n_grid` : int                  — grid nodes (>= 3).
/// `n_chernoff` : int              — Chernoff iteration count (>= 1).
///
/// Raises
/// ------
/// `SemiflowError`  — `GridMismatch`, `OutOfDomain`, `NanInf`.
#[pyclass(name = "KilledDirichlet1D")]
pub struct PyKilledDirichlet1D {
    evolver: Evolver<KilledDirichletChernoff>,
    current: GridFn1D<f64>,
}

#[pymethods]
impl PyKilledDirichlet1D {
    #[new]
    fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        u0: &Bound<'_, pyo3::types::PyAny>,
        n_chernoff: usize,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let u0_vec = extract_f64_vec(u0)?;
            let grid = Grid1D::new(domain_lo, domain_hi, n_grid).map_err(|e| from_core(&e))?;
            let kernel = KilledDirichletChernoff::new(|_| 1.0_f64, |_| 0.0_f64, grid)
                .map_err(|e| from_core(&e))?;
            let evolver = Evolver::new(kernel, n_chernoff).map_err(|e| from_core(&e))?;
            let current = GridFn1D::new(grid, u0_vec).map_err(|e| from_core(&e))?;
            Ok(Self { evolver, current })
        })
    }

    /// Advance by `t`; return evolved grid as numpy array.
    fn apply<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        catch_panic_py!({
            validate_t(t)?;
            let result = self
                .evolver
                .evolve(t, &self.current)
                .map_err(|e| from_core(&e))?;
            self.current = result;
            Ok(self.current.values.as_slice().to_pyarray(py))
        })
    }

    /// Number of grid nodes.
    fn size(&self) -> usize {
        self.current.values.len()
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register Greeks pyclasses into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEvolverHeat1DGreeksV3>()?;
    m.add_class::<PyKilledDirichlet1D>()?;
    Ok(())
}
