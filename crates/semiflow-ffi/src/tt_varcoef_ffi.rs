//! FFI surface for `VarCoefTt` — additive-separable variable-coefficient
//! TT evolver (ADR-0178, math §52.10, issue #2).
//!
//! Mirrors `tt_ffi.rs` for `SmfTtEvolver` / `SmfTtState`.
//!
//! ## Ragged-array convention (C-2, mirrors `tt_ffi.rs`)
//!
//! Three CSR-flattened ragged arrays (a / b / v) each carry
//! `(data: *const f64, offsets: *const usize, n_axes: usize)`.
//! Axis `j` occupies `data[offsets[j] .. offsets[j+1]]`.
//!
//! `SmfTtState` is defined in `tt_ffi.rs` and re-used here.
//!
//! ## Error mapping
//!
//! `VarCoefOutOfClass` → `SemiflowStatus::OutOfDomain` (pre-checked, no panic).
//!
//! ## Build requirement
//!
//! Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(clippy::cast_precision_loss, clippy::too_many_arguments)]

use semiflow::{TtState, VarCoefTt};

use crate::{status::SemiflowStatus, tt_ffi::SmfTtState};

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque C handle to `Box<VarCoefTt<f64>>`.
#[repr(C)]
pub struct SmfVarCoefTtEvolver {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// smf_varcoef_tt_evolver_new
// ---------------------------------------------------------------------------

/// Construct a `VarCoefTt<f64>` evolver.
///
/// Three CSR-ragged arrays carry the per-axis coefficient vectors:
/// - `a_data` / `a_off` — diffusion `aⱼ(xⱼ)` (all > 0).
/// - `b_data` / `b_off` — drift `bⱼ(xⱼ)`.
/// - `v_data` / `v_off` — reaction `vⱼ(xⱼ)` (empty axis = zero).
/// - `dom_lo` / `dom_hi` — per-axis `(xmin, xmax)`, length `n_axes`.
/// - `eps_round` — TT-rounding tolerance.
///
/// Returns `OutOfDomain` for shape / parabolicity violations.
/// Returns `NullPtr` if any required pointer is null.
///
/// # Safety
/// All pointers must be valid for the documented lengths.
#[no_mangle]
pub unsafe extern "C" fn smf_varcoef_tt_evolver_new(
    a_data: *const f64,
    a_off: *const usize,
    b_data: *const f64,
    b_off: *const usize,
    v_data: *const f64,
    v_off: *const usize,
    dom_lo: *const f64,
    dom_hi: *const f64,
    n_axes: usize,
    eps_round: f64,
    out_ev: *mut *mut SmfVarCoefTtEvolver,
) -> SemiflowStatus {
    if a_data.is_null()
        || a_off.is_null()
        || b_data.is_null()
        || b_off.is_null()
        || v_data.is_null()
        || v_off.is_null()
        || dom_lo.is_null()
        || dom_hi.is_null()
        || out_ev.is_null()
    {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: null-check above; caller guarantees lengths match n_axes.
        unsafe {
            build_varcoef_evolver(
                a_data, a_off, b_data, b_off, v_data, v_off, dom_lo, dom_hi, n_axes, eps_round,
                out_ev,
            )
        }
    })
}

/// Load offset+data slices and validate offsets.
///
/// # Safety
/// All pointers must be non-null and valid for `n_axes` / the lengths implied by offset arrays.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
unsafe fn load_varcoef_slices<'a>(
    a_data: *const f64,
    a_off: *const usize,
    b_data: *const f64,
    b_off: *const usize,
    v_data: *const f64,
    v_off: *const usize,
    dom_lo: *const f64,
    dom_hi: *const f64,
    n_axes: usize,
) -> Result<
    (
        &'a [f64],
        &'a [usize],
        &'a [f64],
        &'a [usize],
        &'a [f64],
        &'a [usize],
        &'a [f64],
        &'a [f64],
    ),
    SemiflowStatus,
> {
    let a_offs = unsafe { std::slice::from_raw_parts(a_off, n_axes + 1) };
    let b_offs = unsafe { std::slice::from_raw_parts(b_off, n_axes + 1) };
    let v_offs = unsafe { std::slice::from_raw_parts(v_off, n_axes + 1) };
    if validate_offsets(a_offs, n_axes).is_err()
        || validate_offsets(b_offs, n_axes).is_err()
        || validate_offsets(v_offs, n_axes).is_err()
    {
        return Err(SemiflowStatus::GridMismatch);
    }
    let a_flat = unsafe { std::slice::from_raw_parts(a_data, a_offs[n_axes]) };
    let b_flat = unsafe { std::slice::from_raw_parts(b_data, b_offs[n_axes]) };
    let v_flat = unsafe { std::slice::from_raw_parts(v_data, v_offs[n_axes]) };
    let lo_s = unsafe { std::slice::from_raw_parts(dom_lo, n_axes) };
    let hi_s = unsafe { std::slice::from_raw_parts(dom_hi, n_axes) };
    Ok((a_flat, a_offs, b_flat, b_offs, v_flat, v_offs, lo_s, hi_s))
}

/// Inner safe-ish helper called from `smf_varcoef_tt_evolver_new`.
///
/// # Safety
/// All pointer preconditions inherited from the outer `unsafe extern "C"` fn.
unsafe fn build_varcoef_evolver(
    a_data: *const f64,
    a_off: *const usize,
    b_data: *const f64,
    b_off: *const usize,
    v_data: *const f64,
    v_off: *const usize,
    dom_lo: *const f64,
    dom_hi: *const f64,
    n_axes: usize,
    eps_round: f64,
    out_ev: *mut *mut SmfVarCoefTtEvolver,
) -> SemiflowStatus {
    if n_axes == 0 {
        return SemiflowStatus::OutOfDomain;
    }
    if !eps_round.is_finite() {
        return SemiflowStatus::NanInf;
    }
    let slices = unsafe {
        load_varcoef_slices(
            a_data, a_off, b_data, b_off, v_data, v_off, dom_lo, dom_hi, n_axes,
        )
    };
    let (a_flat, a_offs, b_flat, b_offs, v_flat, v_offs, lo_s, hi_s) = match slices {
        Ok(s) => s,
        Err(s) => return s,
    };
    let (a, b, v) = match decode_ragged_abv(a_flat, a_offs, b_flat, b_offs, v_flat, v_offs, n_axes)
    {
        Ok(t) => t,
        Err(s) => return s,
    };
    let domain = match build_domain(lo_s, hi_s) {
        Ok(d) => d,
        Err(s) => return s,
    };
    match VarCoefTt::<f64>::new(a, b, v, domain, eps_round) {
        Err(_) => SemiflowStatus::OutOfDomain,
        Ok(ev) => {
            let raw = Box::into_raw(Box::new(ev)).cast::<SmfVarCoefTtEvolver>();
            unsafe { *out_ev = raw };
            SemiflowStatus::Ok
        }
    }
}

/// Build domain pairs with validation.
fn build_domain(lo_s: &[f64], hi_s: &[f64]) -> Result<Vec<(f64, f64)>, SemiflowStatus> {
    lo_s.iter()
        .zip(hi_s.iter())
        .map(|(&lo, &hi)| {
            if !lo.is_finite() || !hi.is_finite() || lo >= hi {
                Err(SemiflowStatus::NanInf)
            } else {
                Ok((lo, hi))
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// smf_varcoef_tt_evolver_evolve
// ---------------------------------------------------------------------------

/// Evolve `state` for time `t_final` using `n_steps` steps (in-place).
///
/// Returns `OutOfDomain` if `n_steps == 0`, `t_final` non-finite/negative,
/// or `ev.ndim() != state.ndim()`.
///
/// # Safety
/// `ev` and `state` must be non-null live pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_varcoef_tt_evolver_evolve(
    ev: *const SmfVarCoefTtEvolver,
    state: *mut SmfTtState,
    t_final: f64,
    n_steps: usize,
) -> SemiflowStatus {
    if ev.is_null() || state.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if n_steps == 0 || !t_final.is_finite() || t_final < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        let evolver = unsafe { &*ev.cast::<VarCoefTt<f64>>() };
        let s = unsafe { &mut *state.cast::<TtState<f64>>() };
        if evolver.ndim() != s.ndim() {
            return SemiflowStatus::OutOfDomain;
        }
        evolver.evolve(t_final, n_steps, s);
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// smf_varcoef_tt_evolver_ndim
// ---------------------------------------------------------------------------

/// Return the number of axes. Returns 0 if `ev` is null.
///
/// # Safety
/// `ev` must be null or a live `SmfVarCoefTtEvolver` pointer.
#[no_mangle]
pub unsafe extern "C" fn smf_varcoef_tt_evolver_ndim(ev: *const SmfVarCoefTtEvolver) -> usize {
    if ev.is_null() {
        return 0;
    }
    let evolver = unsafe { &*ev.cast::<VarCoefTt<f64>>() };
    evolver.ndim()
}

// ---------------------------------------------------------------------------
// smf_varcoef_tt_evolver_free
// ---------------------------------------------------------------------------

/// Free a `SmfVarCoefTtEvolver` handle. Null-safe; do not use after this call.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_varcoef_tt_evolver_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_varcoef_tt_evolver_free(ev: *mut SmfVarCoefTtEvolver) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<VarCoefTt<f64>>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Validate C-2 offset array (strictly monotone, offsets[0]==0).
fn validate_offsets(off: &[usize], n_axes: usize) -> Result<(), SemiflowStatus> {
    if n_axes == 0 || off[0] != 0 {
        return Err(SemiflowStatus::GridMismatch);
    }
    for i in 0..n_axes {
        if off[i + 1] <= off[i] {
            return Err(SemiflowStatus::GridMismatch);
        }
    }
    Ok(())
}

/// Decode flat buffer + offsets → `Vec<Vec<f64>>` with finite checks.
fn decode_ragged(
    flat: &[f64],
    off: &[usize],
    n_axes: usize,
) -> Result<Vec<Vec<f64>>, SemiflowStatus> {
    let mut out = Vec::with_capacity(n_axes);
    for j in 0..n_axes {
        let s = off[j];
        let e = off[j + 1];
        // Allow empty slice (v_axis may be zero-length per axis)
        let sl = flat.get(s..e).ok_or(SemiflowStatus::GridMismatch)?.to_vec();
        for &v in &sl {
            if !v.is_finite() {
                return Err(SemiflowStatus::NanInf);
            }
        }
        out.push(sl);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn decode_ragged_abv(
    a_flat: &[f64],
    a_offs: &[usize],
    b_flat: &[f64],
    b_offs: &[usize],
    v_flat: &[f64],
    v_offs: &[usize],
    n_axes: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>), SemiflowStatus> {
    let a = decode_ragged(a_flat, a_offs, n_axes)?;
    let b = decode_ragged(b_flat, b_offs, n_axes)?;
    let v = decode_ragged(v_flat, v_offs, n_axes)?;
    Ok((a, b, v))
}
