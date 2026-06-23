//! v8.0.0 Greeks FFI surface (ADR-0028 Amendment 2, ADR-0133 A1).
//!
//! Exposes forward-mode Dual-AD value / Δ / Γ for the 1-D unit-diffusion
//! heat kernel via three `extern "C"` entry points with the `_v3`-suffix
//! naming convention (additive alongside the existing v3 surface;
//! per ADR-0076 Approach A).
//!
//! ## AD seam (design doc §1, math.md §46.4)
//!
//! A single `DiffusionChernoff<Dual<Dual<f64>>>` sweep is run for `n_chernoff`
//! Chernoff steps using `apply_f`.  The diffusion-scale θ is seeded as
//! `Dual::variable(Dual::variable(θ))` in the `a(x)` coefficient closure;
//! u0 values carry zero tangents (θ-independent initial data).
//! After the sweep the three output buffers are demultiplexed:
//!   `value[i]  = result[i].value.value`
//!   `delta[i]  = result[i].tangent.value`   (∂u/∂θ, §46 Δ)
//!   `gamma[i]  = result[i].tangent.tangent` (∂²u/∂θ², §46.4 Γ)
//!
//! ## Panic safety
//!
//! Every `extern "C"` body is wrapped in `catch_panic!`.
//! Build with `--profile release-ffi` (`panic = "unwind"`).
//!
//! ## Ownership model
//!
//! `smf_greeks_evolver_new_heat_1d_unit_v3` allocates a `Box<GreeksInnerV3>`
//! and hands ownership to the caller as a `*mut SmfGreeksEvolverV3`.
//! Free with `smf_greeks_evolver_free_v3`.  Null is always safe.
//!
//! ## Safety invariants (mirrors the existing v3 surface)
//!
//! 1. Null-check BEFORE `catch_panic!`.
//! 2. `*mut SmfGreeksEvolverV3` is always a live `Box<GreeksInnerV3>`.
//! 3. `smf_greeks_evolver_free_v3` is null-safe and idempotent.
//! 4. `(ptr, len)` slice pairs are caller-guaranteed valid for the call.

#![allow(unsafe_code)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::os::raw::c_double;
use std::sync::Arc;

use semiflow::{DiffusionChernoff, Dual, Grid1D, GridFn1D};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Field type alias
// ---------------------------------------------------------------------------

/// Hyper-dual scalar field for second-order forward-mode AD (§46.4).
type HyperDual = Dual<Dual<f64>>;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a hyper-dual Greeks evolver (`Dual<Dual<f64>>` field).
///
/// C callers receive this from `smf_greeks_evolver_new_heat_1d_unit_v3`
/// and pass it to `smf_heat1d_greeks_v3` / `smf_greeks_evolver_free_v3`.
/// Do not dereference or heap-allocate this struct from C.
#[repr(C)]
pub struct SmfGreeksEvolverV3 {
    _private: [u8; 0],
}

/// Inner data (Rust-private).  Holds the hyper-dual kernel + initial state.
pub(crate) struct GreeksInnerV3 {
    /// Hyper-dual Chernoff kernel with seeded θ in `a(x)`.
    chernoff: DiffusionChernoff<HyperDual>,
    /// Initial condition with zero tangents (θ-independent).
    u0: GridFn1D<HyperDual>,
    /// Number of Chernoff iterations per evaluation.
    n_chernoff: usize,
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Create a hyper-dual Greeks evolver for unit-diffusion heat.
///
/// Solves `∂_t u = θ · ∂_xx u` on `[xmin, xmax]` with `n_grid` nodes.
/// `scale_theta` (θ) is seeded for Dual-AD so a single sweep yields
/// value, Δ = ∂u/∂θ, and Γ = ∂²u/∂θ² simultaneously.
///
/// On success `*out_ev` is set.  On any error `*out_ev` is unchanged.
///
/// ## Preconditions
/// - `xmin < xmax`, both finite.
/// - `n_grid >= 4`.
/// - `n_chernoff >= 1`.
/// - `scale_theta` finite and `> 0`.
/// - `u0` non-null; `u0_len == n_grid`; all elements finite.
/// - `out_ev` non-null.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `NanInf` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// - `u0` must point to `u0_len` readable contiguous `f64` values.
/// - `out_ev` must be a valid writable `*mut *mut SmfGreeksEvolverV3`.
#[no_mangle]
pub unsafe extern "C" fn smf_greeks_evolver_new_heat_1d_unit_v3(
    xmin: c_double,
    xmax: c_double,
    n_grid: usize,
    n_chernoff: usize,
    scale_theta: c_double,
    u0: *const c_double,
    u0_len: usize,
    out_ev: *mut *mut SmfGreeksEvolverV3,
) -> SemiflowStatus {
    if u0.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let slice = unsafe { std::slice::from_raw_parts(u0, u0_len) };
        match build_greeks_inner(xmin, xmax, n_grid, n_chernoff, scale_theta, slice) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(inner) => {
                let raw = Box::into_raw(Box::new(inner)).cast::<SmfGreeksEvolverV3>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

/// Free a Greeks evolver handle.  Null-safe; after the call the pointer is
/// dangling — do not use it again.
///
/// # Safety
/// - `ev` must be null or a live pointer from `smf_greeks_evolver_new_*_v3`.
#[no_mangle]
pub unsafe extern "C" fn smf_greeks_evolver_free_v3(ev: *mut SmfGreeksEvolverV3) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<GreeksInnerV3>())) };
    }));
}

// ---------------------------------------------------------------------------
// Greeks evaluation
// ---------------------------------------------------------------------------

/// Advance by `t` and write value, Δ, and Γ to three caller-owned buffers.
///
/// Runs `n_chernoff` hyper-dual Chernoff steps and demultiplexes the lanes:
///   `value[i] = result[i].value.value`
///   `delta[i] = result[i].tangent.value`
///   `gamma[i] = result[i].tangent.tangent`
///
/// All three output pointers must be non-null and each length `len`.
/// `len` must equal `n_grid` (the size passed to the constructor).
///
/// ## Preconditions
/// - `ev` non-null, from `smf_greeks_evolver_new_*_v3`.
/// - `t >= 0`, finite.
/// - `out_value`, `out_delta`, `out_gamma` non-null, each length `len`.
/// - `len == n_grid`.
///
/// ## Return values
/// `Ok` | `NullPtr` | `GridMismatch` | `OutOfDomain` | `Panic`.
///
/// # Safety
/// - `ev` must be a live Greeks evolver handle.
/// - All three `out_*` must be valid for `len` writable `f64`s.
#[no_mangle]
pub unsafe extern "C" fn smf_heat1d_greeks_v3(
    ev: *mut SmfGreeksEvolverV3,
    t: c_double,
    out_value: *mut c_double,
    out_delta: *mut c_double,
    out_gamma: *mut c_double,
    len: usize,
) -> SemiflowStatus {
    if ev.is_null() || out_value.is_null() || out_delta.is_null() || out_gamma.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let inner = unsafe { &*ev.cast::<GreeksInnerV3>() };
        if len != inner.u0.values.len() {
            return SemiflowStatus::GridMismatch;
        }
        match run_hyper_dual_sweep(inner, t) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(result) => unsafe {
                write_output_buffers(&result, out_value, out_delta, out_gamma, len)
            },
        }
    })
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Build `GreeksInnerV3` for unit-diffusion heat with seeded θ.
fn build_greeks_inner(
    xmin: f64,
    xmax: f64,
    n_grid: usize,
    n_chernoff: usize,
    scale_theta: f64,
    u0_f64: &[f64],
) -> Result<GreeksInnerV3, semiflow::SemiflowError> {
    crate::handle::validate_u0_finite(u0_f64)?;
    validate_n_chernoff(n_chernoff)?;
    validate_scale(scale_theta)?;
    validate_u0_len(u0_f64, n_grid)?;

    // Build hyper-dual grid: xmin/xmax as pure constants (no θ-seed in geometry).
    let xmin_dd = hd_const(Dual::constant(xmin));
    let xmax_dd = hd_const(Dual::constant(xmax));
    let grid = Grid1D::<HyperDual>::new_generic(xmin_dd, xmax_dd, n_grid)?;

    // Seed θ: Dual::variable(Dual::variable(θ)) so outer AND inner tangent = 1.
    let theta_seeded = Dual::variable(Dual::variable(scale_theta));
    let theta_arc = Arc::new(theta_seeded);

    // Coefficient `a(x) = θ` (constant in x, but carries the θ-seed).
    // `a'` and `a''` are zero (constant diffusion coefficient).
    let theta_arc2 = Arc::clone(&theta_arc);
    let chernoff = DiffusionChernoff::with_closure(
        move |_x: HyperDual| *theta_arc2,
        |_x: HyperDual| hd_const(Dual::constant(0.0)),
        |_x: HyperDual| hd_const(Dual::constant(0.0)),
        scale_theta, // norm bound: |θ| for the constant diffusion coefficient
        grid,
    );

    // Initial condition with zero tangents (u0 is θ-independent).
    let u0 = GridFn1D::from_fn_generic(grid, |xi| {
        let v = u0_f64[grid_index(xi, xmin, xmax, n_grid)];
        hd_const(Dual::constant(v))
    });

    Ok(GreeksInnerV3 {
        chernoff,
        u0,
        n_chernoff,
    })
}

/// Run `n_chernoff` Chernoff steps at hyper-dual precision.
fn run_hyper_dual_sweep(
    inner: &GreeksInnerV3,
    t: f64,
) -> Result<GridFn1D<HyperDual>, semiflow::SemiflowError> {
    if !t.is_finite() || t < 0.0 {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "t must be finite and >= 0",
            value: t,
        });
    }
    let n = inner.n_chernoff;
    #[allow(clippy::cast_precision_loss)]
    let tau_f64 = t / (n as f64);
    let tau = hd_const(Dual::constant(tau_f64));
    let mut state = inner.u0.clone();
    for _ in 0..n {
        state = inner.chernoff.apply_f(tau, &state)?;
    }
    Ok(state)
}

/// Demultiplex hyper-dual lanes into three `f64` output buffers.
///
/// # Safety
/// Caller guarantees all three pointers are valid for `len` writable `f64`s.
unsafe fn write_output_buffers(
    result: &GridFn1D<HyperDual>,
    out_value: *mut c_double,
    out_delta: *mut c_double,
    out_gamma: *mut c_double,
    len: usize,
) -> SemiflowStatus {
    let vals = unsafe { std::slice::from_raw_parts_mut(out_value, len) };
    let delts = unsafe { std::slice::from_raw_parts_mut(out_delta, len) };
    let gams = unsafe { std::slice::from_raw_parts_mut(out_gamma, len) };
    for (i, &dd) in result.values.iter().enumerate() {
        vals[i] = dd.value.value;
        delts[i] = dd.tangent.value;
        gams[i] = dd.tangent.tangent;
    }
    SemiflowStatus::Ok
}

/// Recover the grid index for a hyper-dual grid point (by position).
///
/// Returns the nearest node index given the primal x-coordinate value.
/// Used to look up the stored f64 u0 slice when building the dual u0 grid.
#[inline]
fn grid_index(xi: HyperDual, xmin: f64, xmax: f64, n: usize) -> usize {
    let x = xi.value.value;
    let dx = (xmax - xmin) / ((n - 1) as f64);
    let idx = ((x - xmin) / dx).round() as isize;
    idx.clamp(0, (n - 1) as isize) as usize
}

/// Wrap a `Dual<f64>` as a constant `Dual<Dual<f64>>` (zero tangents).
#[inline]
fn hd_const(v: Dual<f64>) -> HyperDual {
    Dual::constant(v)
}

/// Validate `n_chernoff >= 1`.
fn validate_n_chernoff(n: usize) -> Result<(), semiflow::SemiflowError> {
    if n == 0 {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "n_chernoff must be >= 1",
            value: 0.0,
        });
    }
    Ok(())
}

/// Validate `scale_theta` is finite and positive.
fn validate_scale(theta: f64) -> Result<(), semiflow::SemiflowError> {
    if !theta.is_finite() || theta <= 0.0 {
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "scale_theta must be finite and positive",
            value: theta,
        });
    }
    Ok(())
}

/// Validate `u0.len() == n_grid`.
fn validate_u0_len(u0: &[f64], n_grid: usize) -> Result<(), semiflow::SemiflowError> {
    if u0.len() != n_grid {
        #[allow(clippy::cast_precision_loss)]
        return Err(semiflow::SemiflowError::DomainViolation {
            what: "u0_len must equal n_grid",
            value: u0.len() as f64,
        });
    }
    Ok(())
}
