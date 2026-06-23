//! v8.3.0 FFI surface for `DynamicWentzellChernoff` (C-9, ADR-0153, ADR-0151).
//!
//! Exposes a Wentzell evolver for the 1D unit-diffusion heat half-line via three
//! `extern "C"` entry points with the `_v3`-suffix naming convention (additive
//! alongside the existing v3 surface; per ADR-0076 Approach A).
//!
//! ## γ-schedule ABI (ADR-0153 Decision 1, §1 `V8_3_TIER3_BINDING_DESIGN.md`)
//!
//! The kernel samples γ at the LEFT endpoint of each Chernoff step:
//! `t_k = t_offset + k·τ`, `τ = t / n_steps`, `k = 0..n_steps`.
//! A host-level `γ: fn(t)->f64` closure CANNOT cross the ABI soundly; instead
//! the caller pre-samples its γ at every `t_k` BEFORE calling this function and
//! passes `gamma_schedule: *const f64` (length `n_steps`).  The kernel reads
//! `schedule[k]` per step.  Constant-γ = a flat schedule of identical values.
//!
//! **NORMATIVE**: host MUST sample at `t_k = t_offset + k·τ` (left-endpoint freeze)
//! or a silent order-1 error will result.  `γ[k] ≥ 0` and finite is REQUIRED.
//!
//! ## NARROW scope (ADR-0151 NORMATIVE)
//!
//! 1D half-line collapse only; multi-D true-product state is deferred (§49.7).
//! Order = 1 (bulk↔boundary Lie split commutator nonzero, §49.8).
//!
//! ## ABI notes
//!
//! - NO `GammaFamily` on FFI (signature uniformity, ADR-0153 §2.1).
//! - Null-check BEFORE `catch_panic!`; `catch_panic!` every body.
//! - Build with `--profile release-ffi` (`panic = "unwind"`).
//! - `[profile.release-ffi]` is required; workspace release uses `panic = "abort"`.
//!
//! ## Ownership model
//!
//! `smf_wentzell_evolver_new_heat_1d_unit_v3` allocates a `Box<WentzellInnerV3>`
//! and returns ownership as `*mut SmfWentzellEvolverV3`.
//! Free with `smf_wentzell_evolver_free_v3`.  Null is always safe.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_precision_loss, clippy::too_many_arguments)]

use std::os::raw::c_double;

use semiflow::{
    error::SemiflowError,
    reflection::{HalfSpaceRegion, ReflectingRegion},
    robin::RobinRegion,
    scratch::ScratchPool,
    wentzell::WentzellRegion,
    DiffusionChernoff, DynamicWentzellChernoff, Grid1D, GridFn1D, TimedChernoffFunction,
};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a `DynamicWentzellChernoff` evolver for 1D half-line heat.
///
/// C callers receive this from `smf_wentzell_evolver_new_heat_1d_unit_v3`
/// and pass it to `smf_wentzell_evolve_v3` / `smf_wentzell_evolver_free_v3`.
/// Do not dereference or heap-allocate this struct from C.
#[repr(C)]
pub struct SmfWentzellEvolverV3 {
    _private: [u8; 0],
}

/// Inner Rust state (heap-allocated, cast to/from `*mut SmfWentzellEvolverV3`).
struct WentzellInnerV3 {
    /// Grid geometry.
    grid: Grid1D<f64>,
    /// Pre-sampled γ-schedule (length `n_steps`).
    gamma_schedule: Vec<f64>,
    /// Boundary reaction coefficient `c ≥ 0`.
    c_reaction: f64,
    /// Current evolved state.
    current: GridFn1D<f64>,
    /// Scratch pool (reused across evolve calls).
    scratch: ScratchPool<f64>,
}

// ---------------------------------------------------------------------------
// Schedule-backed WentzellRegion (per-crate duplicate, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

/// Schedule-backed `WentzellRegion`.
///
/// `gamma_at` ignores the time argument and returns the pre-sampled constant
/// `gamma_val` stored for the current step.  The Wentzell binding reconstructs
/// this newtype per step from the caller-supplied `gamma_schedule`.
struct ScheduledWentzellRegion {
    gamma_val: f64,
    c: f64,
    half_space: HalfSpaceRegion<f64, 1>,
}

impl ScheduledWentzellRegion {
    fn new(gamma_val: f64, c: f64) -> Result<Self, SemiflowError> {
        let half_space = HalfSpaceRegion::<f64, 1>::new([0.0], [1.0])?;
        Ok(Self {
            gamma_val,
            c,
            half_space,
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
// Constructor
// ---------------------------------------------------------------------------

/// Construct a Wentzell evolver for 1D unit-diffusion heat on a half-line.
///
/// Constructs `DynamicWentzellChernoff` for the equation
/// `∂_t u = ∂_xx u` on `[xmin, xmax]` (half-line, boundary at `xmin`),
/// with Wentzell/Robin BC `∂_t u + γ(t)·∂_ν u + c·u = 0` at `xmin`.
/// `c_reaction ≥ 0` (boundary reaction); `n_steps` Chernoff steps per `evolve`.
///
/// `gamma_schedule`: caller-owned `f64` buffer, length `n_steps`.
/// Host pre-samples γ at `t_k = t_offset + k·τ` (left endpoint) BEFORE calling.
/// `gamma_schedule[k] ≥ 0` and finite; length must equal `n_steps`.
///
/// On success `*out_ev` is set to a non-null opaque pointer.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// - `gamma_schedule` must point to `gamma_len` readable contiguous `f64` values.
/// - `u0` must point to `u0_len` readable contiguous `f64` values.
/// - `out_ev` must be a valid writable `*mut *mut SmfWentzellEvolverV3`.
#[no_mangle]
pub unsafe extern "C" fn smf_wentzell_evolver_new_heat_1d_unit_v3(
    xmin: c_double,
    xmax: c_double,
    n_grid: usize,
    n_steps: usize,
    c_reaction: c_double,
    gamma_schedule: *const c_double,
    gamma_len: usize,
    u0: *const c_double,
    u0_len: usize,
    out_ev: *mut *mut SmfWentzellEvolverV3,
) -> SemiflowStatus {
    if gamma_schedule.is_null() || u0.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let sched = unsafe { std::slice::from_raw_parts(gamma_schedule, gamma_len) };
        let u0_slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_wentzell_inner(xmin, xmax, n_grid, n_steps, c_reaction, sched, u0_slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfWentzellEvolverV3>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

/// Advance the evolver by time `t` and write the evolved grid to `out`.
///
/// Sweeps the γ-schedule once (`n_steps` Chernoff steps), sampling
/// `gamma_schedule[k]` at step `k` (left-endpoint freeze: `t_k = t_offset + k·τ`).
/// The internal current state is updated in-place (chainable).
///
/// `out_len` must equal `n_grid` (the size passed to the constructor).
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// - `ev` must be a live pointer from `smf_wentzell_evolver_new_*_v3`.
/// - `out` must be valid for `out_len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_wentzell_evolve_v3(
    ev: *mut SmfWentzellEvolverV3,
    t: c_double,
    t_offset: c_double,
    out: *mut c_double,
    out_len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &mut *ev.cast::<WentzellInnerV3>() };
        if out_len != inner.current.values.len() {
            return SemiflowStatus::GridMismatch;
        }
        if !t.is_finite() || t <= 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        if let Err(e) = run_wentzell_sweep(inner, t, t_offset) {
            SemiflowStatus::from(&e)
        } else {
            unsafe {
                let out_slice = std::slice::from_raw_parts_mut(out, out_len);
                out_slice.copy_from_slice(&inner.current.values);
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

/// Free a Wentzell evolver handle.  Null-safe; do not use after this call.
///
/// # Safety
/// - `ev` must be null or a live pointer from `smf_wentzell_evolver_new_*_v3`.
#[no_mangle]
pub unsafe extern "C" fn smf_wentzell_evolver_free_v3(ev: *mut SmfWentzellEvolverV3) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<WentzellInnerV3>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Build `WentzellInnerV3` from validated parameters.
fn build_wentzell_inner(
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_steps: usize,
    c_reaction: f64,
    gamma_schedule: &[f64],
    u0: &[f64],
) -> Result<WentzellInnerV3, SemiflowError> {
    validate_schedule(gamma_schedule, n_steps)?;
    validate_u0_len(u0, n_grid)?;
    crate::handle::validate_u0_finite(u0)?;
    validate_c_reaction(c_reaction)?;
    let grid = Grid1D::new(xmin, xmax, n_grid)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok(WentzellInnerV3 {
        grid,
        gamma_schedule: gamma_schedule.to_vec(),
        c_reaction,
        current,
        scratch: ScratchPool::new(),
    })
}

/// Sweep `n_steps` Chernoff steps using the stored γ-schedule.
fn run_wentzell_sweep(
    inner: &mut WentzellInnerV3,
    t: f64,
    t_offset: f64,
) -> Result<(), SemiflowError> {
    let n = inner.gamma_schedule.len();
    let tau = t / n as f64;
    for k in 0..n {
        let t_k = t_offset + k as f64 * tau;
        let gamma_k = inner.gamma_schedule[k];
        let chernoff =
            DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, inner.grid);
        let region = ScheduledWentzellRegion::new(gamma_k, inner.c_reaction)?;
        let wrapper = DynamicWentzellChernoff::new(chernoff, region)?;
        let src = inner.current.clone();
        wrapper.apply_at(t_k, tau, &src, &mut inner.current, &mut inner.scratch)?;
    }
    Ok(())
}

/// Validate γ-schedule: all values finite and ≥ 0, length == `n_steps`.
fn validate_schedule(sched: &[f64], n_steps: usize) -> Result<(), SemiflowError> {
    if sched.len() != n_steps {
        return Err(SemiflowError::DomainViolation {
            what: "gamma_schedule length must equal n_steps",
            value: sched.len() as f64,
        });
    }
    for &g in sched {
        if !g.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "gamma_schedule contains NaN or Inf",
                value: g,
            });
        }
        if g < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "gamma_schedule values must be >= 0",
                value: g,
            });
        }
    }
    Ok(())
}

/// Validate `c_reaction ≥ 0` and finite.
fn validate_c_reaction(c: f64) -> Result<(), SemiflowError> {
    if !c.is_finite() || c < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "c_reaction must be finite and >= 0",
            value: c,
        });
    }
    Ok(())
}

/// Validate `u0.len() == n_grid`.
fn validate_u0_len(u0: &[f64], n_grid: usize) -> Result<(), SemiflowError> {
    if u0.len() != n_grid {
        return Err(SemiflowError::DomainViolation {
            what: "u0_len must equal n_grid",
            value: u0.len() as f64,
        });
    }
    Ok(())
}
