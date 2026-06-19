//! v8.3.0 `PyO3` binding for `DynamicWentzellChernoff` (C-9, ADR-0153, ADR-0151).
//!
//! Implements `WentzellV8` (primary schedule API + `from_family` sugar) and
//! `GammaFamily` (ergonomic sugar for 90% use cases; expands to a schedule).
//!
//! ## γ-schedule ABI (ADR-0153 Decision 1)
//!
//! `WentzellV8.__init__` takes `gamma_schedule: np.ndarray[float64]`, length
//! `n_steps`.  The host pre-samples its arbitrary γ at `t_k = t_offset + k·τ`
//! BEFORE calling — the GIL-off kernel reads `schedule[k]` per step.
//! `GammaFamily` is sugar that expands `Constant/Linear/Exponential` to a
//! schedule internally ("covers 90% ergonomically; use the schedule overload
//! for arbitrary γ"; §1 design doc).
//!
//! ## NARROW scope (ADR-0151 NORMATIVE)
//!
//! 1D half-line collapse only (`dst.values[0]` = boundary trace DOF).
//! Multi-D true-product state deferred (math §49.7).  Order = 1.
//!
//! ## GIL policy (ADR-0031 three-phase)
//!
//! `WentzellV8.evolve` releases the GIL via `py.detach` around the `n_steps`
//! Chernoff sweep.  Phase 1: clone Send data (grid, schedule, u0, c) under GIL.
//! Phase 2: schedule-backed sweep GIL-off.  Phase 3: marshal to numpy.
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util with semiflow-ffi/wasm.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::assigning_clones,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use numpy::ToPyArray;
use pyo3::prelude::*;
use semiflow_core::{
    error::SemiflowError, scratch::ScratchPool, DiffusionChernoff, DynamicWentzellChernoff, Grid1D,
    GridFn1D, TimedChernoffFunction,
};

use crate::error::{from_core, new_pyerr};
use crate::panic::catch_panic_py;
use crate::wentzell_helpers::{
    extract_f64_vec, validate_c_reaction, validate_schedule, validate_t, validate_u0_finite,
    ScheduledWentzellRegion,
};

// ---------------------------------------------------------------------------
// Inner state (heap-owned, Send across GIL boundary)
// ---------------------------------------------------------------------------

/// γ-source: either an explicit pre-sampled schedule or a stored family
/// that is LAZILY expanded at each `evolve(t, t_offset)` call.
///
/// Lazy expansion is necessary for `from_family`: the schedule must be
/// sampled at `t_k = t_offset + k·(t/n_steps)` — the actual arguments
/// supplied to `evolve`, not a frozen t=1.0 template (Howland §49.2).
#[derive(Clone)]
enum GammaSource {
    /// Explicit schedule: expanded once by the caller, swept as-is.
    Explicit(Vec<f64>),
    /// Stored family: expanded lazily inside `evolve(t, t_offset)`.
    Family { kind: GammaKind, n_steps: usize },
}

impl GammaSource {
    /// Materialise the schedule for this `(t, t_offset)` pair.
    fn schedule(&self, t: f64, t_offset: f64) -> Vec<f64> {
        match self {
            GammaSource::Explicit(v) => v.clone(),
            GammaSource::Family { kind, n_steps } => {
                let tau = t / *n_steps as f64;
                (0..*n_steps)
                    .map(|k| {
                        let t_k = t_offset + k as f64 * tau;
                        match kind {
                            GammaKind::Constant(c) => *c,
                            GammaKind::Linear(a, b) => a + b * t_k,
                            GammaKind::Exponential(r) => (r * t_k).exp(),
                        }
                    })
                    .collect()
            }
        }
    }

    fn n_steps(&self) -> usize {
        match self {
            GammaSource::Explicit(v) => v.len(),
            GammaSource::Family { n_steps, .. } => *n_steps,
        }
    }
}

struct WentzellInner {
    grid: Grid1D<f64>,
    gamma_source: GammaSource,
    c_reaction: f64,
    current: Vec<f64>,
}

// ---------------------------------------------------------------------------
// GammaFamily pyclass
// ---------------------------------------------------------------------------

/// Ergonomic γ-schedule family for `WentzellV8` (v8.3.0, ADR-0153).
///
/// Expands Constant/Linear/Exponential to a pre-sampled schedule of length
/// `n_steps` at `t_k = t_offset + k·τ`, `τ = t / n_steps` (left-endpoint freeze).
/// "Covers 90% ergonomically; use `WentzellV8(... gamma_schedule=...)` for
/// arbitrary γ."
///
/// **NARROW**: 1D half-line only; multi-D Wentzell deferred (math §49.7).
#[pyclass(name = "GammaFamily")]
pub struct PyGammaFamily {
    kind: GammaKind,
}

#[derive(Clone)]
enum GammaKind {
    Constant(f64),
    Linear(f64, f64),
    Exponential(f64),
}

#[pymethods]
impl PyGammaFamily {
    /// Constant γ(t) = c.  `c >= 0`.
    #[staticmethod]
    fn constant(c: f64) -> PyResult<Self> {
        if c < 0.0 || !c.is_finite() {
            return Err(new_pyerr(
                "OutOfDomain",
                "GammaFamily.constant: c must be finite and >= 0",
            ));
        }
        Ok(Self {
            kind: GammaKind::Constant(c),
        })
    }

    /// Linear γ(t) = a + b·t.  Must satisfy γ(t) ≥ 0 at t=0 (a ≥ 0).
    #[staticmethod]
    fn linear(a: f64, b: f64) -> PyResult<Self> {
        if a < 0.0 || !a.is_finite() || !b.is_finite() {
            return Err(new_pyerr(
                "OutOfDomain",
                "GammaFamily.linear: a must be finite and >= 0; b finite",
            ));
        }
        Ok(Self {
            kind: GammaKind::Linear(a, b),
        })
    }

    /// Exponential γ(t) = exp(rate·t).  Always ≥ 1.
    #[staticmethod]
    fn exponential(rate: f64) -> PyResult<Self> {
        if !rate.is_finite() {
            return Err(new_pyerr(
                "OutOfDomain",
                "GammaFamily.exponential: rate must be finite",
            ));
        }
        Ok(Self {
            kind: GammaKind::Exponential(rate),
        })
    }
}

// ---------------------------------------------------------------------------
// WentzellV8 pyclass
// ---------------------------------------------------------------------------

/// Dynamic Wentzell/Robin BC evolver for 1D unit-diffusion heat (v8.3.0).
///
/// Advances `∂_t u = ∂_xx u` on `[domain_lo, domain_hi]` (half-line) with
/// the dynamic Wentzell BC `∂_t u + γ(t)·∂_ν u + c·u = 0` at `domain_lo`,
/// implemented via a bulk–boundary Cayley Lie split (math §49, ADR-0151).
///
/// ## γ-schedule (primary API)
///
/// `gamma_schedule`: `np.ndarray[float64]`, length `n_steps`.  Host pre-samples
/// its arbitrary γ at `t_k = t_offset + k·τ` (`τ = t / n_steps`) BEFORE evolving.
/// **NORMATIVE**: sampling must match the left-endpoint freeze point exactly (§49.2),
/// or a silent order-1 error results.  Each `γ[k] ≥ 0` and finite.
///
/// ## Ergonomic sugar (`from_family`)
///
/// Use `WentzellV8.from_family(... family=GammaFamily.linear(0.5, 0.1))` to
/// expand standard families to a schedule automatically.
///
/// ## NARROW scope
///
/// 1D half-line only.  Multi-D Wentzell (true-product state) is deferred
/// to v8.x (math §49.7 NORMATIVE).  Order = 1.
///
/// Parameters
/// ----------
/// `domain_lo` : float  — left boundary (half-line origin).
/// `domain_hi` : float  — right boundary.
/// `n_grid` : int       — grid nodes (>= 4).
/// u0 : array-like    — initial condition, float64, length `n_grid`, all finite.
/// `n_steps` : int      — Chernoff steps per `evolve` call (>= 1).
/// `c_reaction` : float — boundary reaction c ≥ 0 (finite).
/// `gamma_schedule` : array-like — float64, length `n_steps`, all ≥ 0 and finite.
///
/// Raises
/// ------
/// `SemiflowError`
///     `kind='GridMismatch'`  — geometry invalid or len mismatch.
///     `kind='NanInf'`        — non-finite value in u0 or schedule.
///     `kind='OutOfDomain'`   — c < 0, γ < 0, or `n_steps` == 0.
#[pyclass(name = "WentzellV8")]
pub struct PyWentzellV8 {
    inner: WentzellInner,
}

#[pymethods]
impl PyWentzellV8 {
    #[new]
    fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        u0: &Bound<'_, pyo3::types::PyAny>,
        n_steps: usize,
        c_reaction: f64,
        gamma_schedule: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let u0_vec = extract_f64_vec(u0)?;
            let sched_vec = extract_f64_vec(gamma_schedule)?;
            let inner = build_wentzell_inner(
                domain_lo, domain_hi, n_grid, &u0_vec, n_steps, c_reaction, &sched_vec,
            )
            .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Construct from a `GammaFamily` (ergonomic sugar; schedule expanded lazily).
    ///
    /// The `GammaFamily` (kind + params) is stored in the evolver.  The γ-schedule
    /// is expanded LAZILY inside each `evolve(t, t_offset)` call using the ACTUAL
    /// time arguments: `γ[k] = family.eval(t_offset + k·(t/n_steps))`.
    ///
    /// This ensures `from_family(...).evolve(t, t_offset)` produces the correct
    /// Howland left-endpoint freeze for Linear/Exponential families regardless of
    /// `t` and `t_offset` (fixes the frozen-at-t=1.0 template bug).
    ///
    /// The result is 0-ULP equivalent to constructing with an explicit schedule
    /// sampled at the same `(t, t_offset)`.
    ///
    /// Parameters
    /// ----------
    /// family : `GammaFamily`  — schedule expansion family.
    /// (other params same as `__init__`)
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_family(
        _cls: &Bound<'_, pyo3::types::PyType>,
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        u0: &Bound<'_, pyo3::types::PyAny>,
        n_steps: usize,
        c_reaction: f64,
        family: &Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Self> {
        catch_panic_py!({
            let u0_vec = extract_f64_vec(u0)?;
            let fam: PyRef<PyGammaFamily> = family.extract()?;
            let inner = build_wentzell_inner_from_family(
                domain_lo,
                domain_hi,
                n_grid,
                &u0_vec,
                n_steps,
                c_reaction,
                fam.kind.clone(),
            )
            .map_err(|e| from_core(&e))?;
            Ok(Self { inner })
        })
    }

    /// Advance by `t` and return evolved grid as numpy float64 array.
    ///
    /// Sweeps γ-schedule once (`n_steps` Chernoff steps), reading `schedule[k]`
    /// per step (left-endpoint freeze, `t_k = t_offset + k·τ`).  The GIL is
    /// released during the sweep (ADR-0031).  Internal state updated in-place.
    ///
    /// Parameters
    /// ----------
    /// t : float       — time step (> 0, finite).
    /// `t_offset` : float — absolute start time for γ sampling (default 0.0).
    ///
    /// Returns
    /// -------
    /// np.ndarray  — evolved state, float64, length `n_grid`.
    #[pyo3(signature = (t, t_offset = 0.0))]
    fn evolve<'py>(
        &mut self,
        py: Python<'py>,
        t: f64,
        t_offset: f64,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        catch_panic_py!({
            validate_t(t)?;
            // Phase 1: clone Send data under GIL.
            // Lazy expansion: family sources produce the schedule for THIS (t, t_offset).
            let grid = self.inner.grid;
            let sched = self.inner.gamma_source.schedule(t, t_offset);
            let c = self.inner.c_reaction;
            let u0_vals = self.inner.current.clone();
            // Phase 2: schedule sweep GIL-off.
            let result = py.detach(|| run_wentzell_sweep(grid, &u0_vals, &sched, c, t, t_offset));
            let new_vals = result.map_err(|e| from_core(&e))?;
            // Phase 3: update state + marshal to numpy.
            self.inner.current = new_vals.clone();
            Ok(new_vals.as_slice().to_pyarray(py))
        })
    }

    /// Return the number of grid nodes.
    fn size(&self) -> usize {
        self.inner.current.len()
    }

    /// Return the number of Chernoff steps.
    fn n_steps(&self) -> usize {
        self.inner.gamma_source.n_steps()
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust sweep (GIL-off)
// ---------------------------------------------------------------------------

fn run_wentzell_sweep(
    grid: Grid1D<f64>,
    u0: &[f64],
    schedule: &[f64],
    c: f64,
    t: f64,
    t_offset: f64,
) -> Result<Vec<f64>, SemiflowError> {
    let n_steps = schedule.len();
    let tau = t / n_steps as f64;
    let mut state = GridFn1D::new(grid, u0.to_vec())?;
    let mut scratch = ScratchPool::new();
    for k in 0..n_steps {
        let t_k = t_offset + k as f64 * tau;
        let inner = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
        let region = ScheduledWentzellRegion::new(schedule[k], c)?;
        let wrapper = DynamicWentzellChernoff::new(inner, region)?;
        let src = state.clone();
        wrapper.apply_at(t_k, tau, &src, &mut state, &mut scratch)?;
    }
    Ok(state.values)
}

// ---------------------------------------------------------------------------
// Builder and validators
// ---------------------------------------------------------------------------

fn build_wentzell_inner(
    lo: f64,
    hi: f64,
    n_grid: usize,
    u0: &[f64],
    n_steps: usize,
    c_reaction: f64,
    schedule: &[f64],
) -> Result<WentzellInner, SemiflowError> {
    validate_u0_finite(u0)?;
    validate_c_reaction(c_reaction)?;
    validate_schedule(schedule, n_steps)?;
    let grid = Grid1D::new(lo, hi, n_grid)?;
    if u0.len() != n_grid {
        return Err(SemiflowError::DomainViolation {
            what: "u0 length must equal n_grid",
            value: u0.len() as f64,
        });
    }
    Ok(WentzellInner {
        grid,
        gamma_source: GammaSource::Explicit(schedule.to_vec()),
        c_reaction,
        current: u0.to_vec(),
    })
}

/// Build a `WentzellInner` that stores a `GammaFamily` for lazy schedule expansion.
///
/// Validation of `n_steps` > 0 and `c_reaction` >= 0 still runs at construction time;
/// γ-value validation (non-negative, finite) runs per-step inside `run_wentzell_sweep`.
fn build_wentzell_inner_from_family(
    lo: f64,
    hi: f64,
    n_grid: usize,
    u0: &[f64],
    n_steps: usize,
    c_reaction: f64,
    kind: GammaKind,
) -> Result<WentzellInner, SemiflowError> {
    validate_u0_finite(u0)?;
    validate_c_reaction(c_reaction)?;
    if n_steps == 0 {
        return Err(SemiflowError::DomainViolation {
            what: "n_steps must be >= 1",
            value: 0.0,
        });
    }
    let grid = Grid1D::new(lo, hi, n_grid)?;
    if u0.len() != n_grid {
        return Err(SemiflowError::DomainViolation {
            what: "u0 length must equal n_grid",
            value: u0.len() as f64,
        });
    }
    Ok(WentzellInner {
        grid,
        gamma_source: GammaSource::Family { kind, n_steps },
        c_reaction,
        current: u0.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register `WentzellV8` and `GammaFamily` into the `semiflow` module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWentzellV8>()?;
    m.add_class::<PyGammaFamily>()?;
    Ok(())
}
