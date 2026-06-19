//! v8.3.0 WASM binding for `DynamicWentzellChernoff` (C-9, ADR-0153, ADR-0151).
//!
//! Implements `WentzellV8` (primary schedule API + `fromFamily` static) and
//! `GammaFamily` (ergonomic sugar that expands to a schedule).
//!
//! ## γ-schedule ABI (ADR-0153 Decision 1)
//!
//! `WentzellV8` constructor takes `gammaSchedule: Float64Array` (length `nSteps`).
//! The JS caller pre-samples its γ at `t_k = tOffset + k·τ`, `τ = t / nSteps`
//! BEFORE constructing the evolver.  The kernel reads `schedule[k]` per step.
//! Constant-γ = a flat schedule of identical values.
//!
//! **NORMATIVE**: host MUST sample at left-endpoint freeze points or a silent
//! order-1 error results.  Each `γ[k] ≥ 0` and finite.
//!
//! ## NARROW scope (ADR-0151 NORMATIVE)
//!
//! 1D half-line only; multi-D true-product state deferred (math §49.7).
//! Order = 1.
//!
//! ## Panic boundary (ADR-0028 Amendment 1)
//!
//! `[profile.release]` uses `panic = "abort"` — NO `catch_unwind`.
//! All error paths return `Err(JsValue)` with `.kind` discriminator.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::assigning_clones,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_errors_doc,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use wasm_bindgen::prelude::*;

use semiflow_core::{
    error::SemiflowError,
    reflection::{HalfSpaceRegion, ReflectingRegion},
    robin::RobinRegion,
    scratch::ScratchPool,
    wentzell::WentzellRegion,
    DiffusionChernoff, DynamicWentzellChernoff, Grid1D, GridFn1D, TimedChernoffFunction,
};

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Schedule-backed WentzellRegion (per-crate duplicate, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

struct ScheduledWentzellRegion {
    gamma_val: f64,
    c: f64,
    half_space: HalfSpaceRegion<f64, 1>,
}

impl ScheduledWentzellRegion {
    fn new(gamma_val: f64, c: f64) -> Result<Self, SemiflowError> {
        Ok(Self {
            gamma_val,
            c,
            half_space: HalfSpaceRegion::<f64, 1>::new([0.0], [1.0])?,
        })
    }
}

impl ReflectingRegion<f64> for ScheduledWentzellRegion {
    fn dim(&self) -> usize {
        self.half_space.dim()
    }
    fn is_inside(&self, point: &[f64]) -> bool {
        self.half_space.is_inside(point)
    }
    fn reflect_in_place(
        &self,
        dst: &mut GridFn1D<f64>,
        src: &GridFn1D<f64>,
    ) -> Result<(), SemiflowError> {
        self.half_space.reflect_in_place(dst, src)
    }
}

impl RobinRegion<f64> for ScheduledWentzellRegion {
    fn robin_coeffs(&self) -> (f64, f64) {
        (self.c, self.gamma_val)
    }
}

impl WentzellRegion<f64> for ScheduledWentzellRegion {
    fn gamma_at(&self, _t: f64) -> f64 {
        self.gamma_val
    }
    fn reaction(&self) -> f64 {
        self.c
    }
}

// ---------------------------------------------------------------------------
// GammaFamily JS class
// ---------------------------------------------------------------------------

/// Ergonomic γ-schedule family for `WentzellV8` (v8.3.0, ADR-0153).
///
/// The family kind is STORED in `WentzellV8` when constructed via `fromFamily`,
/// and the γ-schedule is expanded LAZILY inside each `evolve(t, tOffset)` call
/// using the ACTUAL time arguments: `γ[k] = family.eval(tOffset + k·(t/nSteps))`.
/// This ensures Linear/Exponential families are sampled at the correct time grid
/// (Howland §49.2 left-endpoint freeze) — not a frozen `t=1.0` template.
///
/// "Covers 90% ergonomically; use `new WentzellV8(... gammaSchedule)` for
/// arbitrary γ."
///
/// ## Error model (`.kind`)
///
/// - `"OutOfDomain"` — negative constant, or non-finite rate/a.
#[wasm_bindgen]
pub struct GammaFamily {
    kind: GammaKindWasm,
}

enum GammaKindWasm {
    Constant(f64),
    Linear(f64, f64),
    Exponential(f64),
}

// ---------------------------------------------------------------------------
// γ-source (per-crate duplicate, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

/// γ-source: either an explicit pre-sampled schedule or a stored family
/// that is LAZILY expanded at each `evolve(t, tOffset)` call.
enum GammaSourceWasm {
    Explicit(Vec<f64>),
    Family { kind: GammaKindWasm, n_steps: usize },
}

impl GammaSourceWasm {
    fn schedule(&self, t: f64, t_offset: f64) -> Vec<f64> {
        match self {
            GammaSourceWasm::Explicit(v) => v.clone(),
            GammaSourceWasm::Family { kind, n_steps } => {
                let tau = t / *n_steps as f64;
                (0..*n_steps)
                    .map(|k| {
                        let t_k = t_offset + k as f64 * tau;
                        match kind {
                            GammaKindWasm::Constant(c) => *c,
                            GammaKindWasm::Linear(a, b) => a + b * t_k,
                            GammaKindWasm::Exponential(r) => (r * t_k).exp(),
                        }
                    })
                    .collect()
            }
        }
    }

    fn n_steps(&self) -> usize {
        match self {
            GammaSourceWasm::Explicit(v) => v.len(),
            GammaSourceWasm::Family { n_steps, .. } => *n_steps,
        }
    }
}

#[wasm_bindgen]
impl GammaFamily {
    /// Constant γ(t) = c.
    pub fn constant(c: f64) -> Result<GammaFamily, JsValue> {
        if c < 0.0 || !c.is_finite() {
            return Err(make_js_error(
                "OutOfDomain",
                "GammaFamily.constant: c must be finite and >= 0",
            ));
        }
        Ok(Self {
            kind: GammaKindWasm::Constant(c),
        })
    }

    /// Linear γ(t) = a + b·t.
    pub fn linear(a: f64, b: f64) -> Result<GammaFamily, JsValue> {
        if a < 0.0 || !a.is_finite() || !b.is_finite() {
            return Err(make_js_error(
                "OutOfDomain",
                "GammaFamily.linear: a must be finite and >= 0; b finite",
            ));
        }
        Ok(Self {
            kind: GammaKindWasm::Linear(a, b),
        })
    }

    /// Exponential γ(t) = exp(rate·t).
    pub fn exponential(rate: f64) -> Result<GammaFamily, JsValue> {
        if !rate.is_finite() {
            return Err(make_js_error(
                "OutOfDomain",
                "GammaFamily.exponential: rate must be finite",
            ));
        }
        Ok(Self {
            kind: GammaKindWasm::Exponential(rate),
        })
    }
}

// ---------------------------------------------------------------------------
// WentzellV8 JS class
// ---------------------------------------------------------------------------

/// Dynamic Wentzell/Robin BC evolver for 1D unit-diffusion heat (v8.3.0).
///
/// Advances `∂_t u = ∂_xx u` on `[domainLo, domainHi]` (half-line) with
/// the dynamic Wentzell BC `∂_t u + γ(t)·∂_ν u + c·u = 0` at `domainLo`,
/// via bulk–boundary Cayley Lie split (math §49, ADR-0151).
///
/// ## γ-schedule (primary API)
///
/// `gammaSchedule: Float64Array`, length `nSteps`.  JS caller pre-samples γ at
/// `t_k = tOffset + k·τ` (`τ = t / nSteps`) before constructing the evolver.
/// **NORMATIVE**: sampling must match the left-endpoint freeze exactly or silent
/// order-1 error results.  Each `γ[k] ≥ 0` and finite.
///
/// ## NARROW scope
///
/// 1D half-line only; multi-D Wentzell deferred (math §49.7 NORMATIVE).
///
/// ## Error model (`.kind`)
///
/// - `"GridMismatch"` — geometry invalid or u0.length != nGrid.
/// - `"NanInf"`       — non-finite value in u0 or schedule.
/// - `"OutOfDomain"`  — cReaction < 0, γ < 0, nSteps == 0, t <= 0.
#[wasm_bindgen]
pub struct WentzellV8 {
    grid: Grid1D<f64>,
    gamma_source: GammaSourceWasm,
    c_reaction: f64,
    current: Vec<f64>,
}

#[wasm_bindgen]
impl WentzellV8 {
    /// Construct a Wentzell evolver from an explicit γ-schedule.
    #[wasm_bindgen(constructor)]
    pub fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        u0: &js_sys::Float64Array,
        n_steps: usize,
        c_reaction: f64,
        gamma_schedule: &js_sys::Float64Array,
    ) -> Result<WentzellV8, JsValue> {
        if u0.length() as usize != n_grid {
            return Err(make_js_error("GridMismatch", "u0.length must equal nGrid"));
        }
        if gamma_schedule.length() as usize != n_steps {
            return Err(make_js_error(
                "GridMismatch",
                "gammaSchedule.length must equal nSteps",
            ));
        }
        let mut u0_buf = vec![0.0f64; n_grid];
        u0.copy_to(&mut u0_buf);
        let mut sched_buf = vec![0.0f64; n_steps];
        gamma_schedule.copy_to(&mut sched_buf);
        build_wentzell_wasm_explicit(
            domain_lo, domain_hi, n_grid, &u0_buf, n_steps, c_reaction, &sched_buf,
        )
        .map_err(|e| err_to_js(&e))
    }

    /// Construct from a `GammaFamily` (ergonomic sugar; schedule expanded lazily).
    ///
    /// The `GammaFamily` kind is STORED in the evolver. The γ-schedule is
    /// expanded LAZILY inside each `evolve(t, tOffset)` call using the ACTUAL
    /// time arguments: `γ[k] = family.eval(tOffset + k·(t/nSteps))`.
    ///
    /// This ensures `fromFamily(...).evolve(t, tOffset)` produces the correct
    /// Howland left-endpoint freeze for Linear/Exponential families.  The result
    /// is 0-ULP equivalent to constructing with an explicit schedule sampled at
    /// the same `(t, tOffset)`.
    #[wasm_bindgen(js_name = "fromFamily")]
    pub fn from_family(
        domain_lo: f64,
        domain_hi: f64,
        n_grid: usize,
        u0: &js_sys::Float64Array,
        n_steps: usize,
        c_reaction: f64,
        family: &GammaFamily,
    ) -> Result<WentzellV8, JsValue> {
        if u0.length() as usize != n_grid {
            return Err(make_js_error("GridMismatch", "u0.length must equal nGrid"));
        }
        if n_steps == 0 {
            return Err(make_js_error("OutOfDomain", "nSteps must be >= 1"));
        }
        let mut u0_buf = vec![0.0f64; n_grid];
        u0.copy_to(&mut u0_buf);
        build_wentzell_wasm_from_family(
            domain_lo,
            domain_hi,
            n_grid,
            &u0_buf,
            n_steps,
            c_reaction,
            &family.kind,
        )
        .map_err(|e| err_to_js(&e))
    }

    /// Advance by `t`; return evolved state as `Float64Array` of length `size()`.
    ///
    /// Sweeps γ-schedule once, reading `schedule[k]` at step k
    /// (`t_k = tOffset + k·τ`).  Internal state updated in-place.
    /// For family-backed evolvers the schedule is expanded lazily here.
    pub fn evolve(&mut self, t: f64, t_offset: f64) -> Result<js_sys::Float64Array, JsValue> {
        if !t.is_finite() || t <= 0.0 {
            return Err(make_js_error("OutOfDomain", "t must be finite and > 0"));
        }
        let sched = self.gamma_source.schedule(t, t_offset);
        let new_vals = run_wentzell_sweep_wasm(
            self.grid,
            &self.current,
            &sched,
            self.c_reaction,
            t,
            t_offset,
        )
        .map_err(|e| err_to_js(&e))?;
        self.current = new_vals.clone();
        let arr = js_sys::Float64Array::new_with_length(new_vals.len() as u32);
        arr.copy_from(&new_vals);
        Ok(arr)
    }

    /// Return the number of grid nodes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.current.len()
    }

    /// Return the number of Chernoff steps.
    #[wasm_bindgen(js_name = "nSteps")]
    #[must_use]
    pub fn n_steps(&self) -> usize {
        self.gamma_source.n_steps()
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust sweep
// ---------------------------------------------------------------------------

fn run_wentzell_sweep_wasm(
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

fn build_wentzell_wasm_explicit(
    lo: f64,
    hi: f64,
    n_grid: usize,
    u0: &[f64],
    n_steps: usize,
    c_reaction: f64,
    schedule: &[f64],
) -> Result<WentzellV8, SemiflowError> {
    validate_u0_finite_wasm(u0)?;
    validate_c_reaction_wasm(c_reaction)?;
    validate_schedule_wasm(schedule, n_steps)?;
    let grid = Grid1D::new(lo, hi, n_grid)?;
    Ok(WentzellV8 {
        grid,
        gamma_source: GammaSourceWasm::Explicit(schedule.to_vec()),
        c_reaction,
        current: u0.to_vec(),
    })
}

fn build_wentzell_wasm_from_family(
    lo: f64,
    hi: f64,
    n_grid: usize,
    u0: &[f64],
    n_steps: usize,
    c_reaction: f64,
    kind: &GammaKindWasm,
) -> Result<WentzellV8, SemiflowError> {
    validate_u0_finite_wasm(u0)?;
    validate_c_reaction_wasm(c_reaction)?;
    let grid = Grid1D::new(lo, hi, n_grid)?;
    let kind_owned = match kind {
        GammaKindWasm::Constant(c) => GammaKindWasm::Constant(*c),
        GammaKindWasm::Linear(a, b) => GammaKindWasm::Linear(*a, *b),
        GammaKindWasm::Exponential(r) => GammaKindWasm::Exponential(*r),
    };
    Ok(WentzellV8 {
        grid,
        gamma_source: GammaSourceWasm::Family {
            kind: kind_owned,
            n_steps,
        },
        c_reaction,
        current: u0.to_vec(),
    })
}

fn validate_u0_finite_wasm(u0: &[f64]) -> Result<(), SemiflowError> {
    for &v in u0 {
        if !v.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "u0 contains NaN or Inf",
                value: v,
            });
        }
    }
    Ok(())
}

fn validate_c_reaction_wasm(c: f64) -> Result<(), SemiflowError> {
    if !c.is_finite() || c < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "cReaction must be finite and >= 0",
            value: c,
        });
    }
    Ok(())
}

fn validate_schedule_wasm(sched: &[f64], n_steps: usize) -> Result<(), SemiflowError> {
    if sched.len() != n_steps {
        return Err(SemiflowError::DomainViolation {
            what: "gammaSchedule.length must equal nSteps",
            value: sched.len() as f64,
        });
    }
    for &g in sched {
        if !g.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "gammaSchedule contains NaN or Inf",
                value: g,
            });
        }
        if g < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "gammaSchedule values must be >= 0",
                value: g,
            });
        }
    }
    Ok(())
}
