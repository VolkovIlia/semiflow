//! S³ tensor-train FFI — `SmfTtState` and `SmfTtEvolver` handles.
//!
//! Implements the `smf_ttstate_*` and `smf_tt_evolver_*` groups from
//! `contracts/semiflow-ffi.s3-carrier-handle.yaml` (v9.2.0, ADR-0171).
//!
//! ## Conventions
//!
//! - Ragged arrays cross as `(data, offsets, n_axes)` (C-2).
//! - Constructors take `out_handle: *mut *mut Smf*` and write on Ok (C-5).
//! - Every `extern "C"` body is wrapped in `catch_panic!`; null-checks before.
//! - Build with `--profile release-ffi` (`panic = "unwind"`).

#![allow(unsafe_code)]
#![allow(clippy::cast_precision_loss, clippy::too_many_arguments)]

use semiflow::{TtChernoff, TtState};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque C handle to `Box<TtState<f64>>`.  Curse-escape: the d-dim tensor is
/// never materialised; only TT cores live behind the handle.
#[repr(C)]
pub struct SmfTtState {
    _private: [u8; 0],
}

/// Opaque C handle to `Box<TtChernoff<f64>>`.
#[repr(C)]
pub struct SmfTtEvolver {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// smf_ttstate_new_separable
// ---------------------------------------------------------------------------

/// Build a rank-1 separable `TtState<f64>` from per-axis slice data.
///
/// `data` / `offsets` follow C-2 ragged-array flattening:
///   axis `j` occupies `data[offsets[j] .. offsets[j+1]]`.
/// `n_axes` must be ≥ 1; `offsets[0]` must be 0; all `n_j` must be ≥ 1.
///
/// On success, `*out_state` is set to the freshly allocated handle.
///
/// # Safety
/// `data`, `offsets`, `out_state` must be valid non-null pointers with the
/// stated lengths.
#[no_mangle]
pub unsafe extern "C" fn smf_ttstate_new_separable(
    data: *const f64,
    offsets: *const usize,
    n_axes: usize,
    out_state: *mut *mut SmfTtState,
) -> SemiflowStatus {
    if data.is_null() || offsets.is_null() || out_state.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let off = unsafe { std::slice::from_raw_parts(offsets, n_axes + 1) };
        if let Err(s) = validate_offsets(off, n_axes) {
            return s;
        }
        let total = off[n_axes];
        let flat = unsafe { std::slice::from_raw_parts(data, total) };
        match build_slices_from_ragged(flat, off, n_axes) {
            Err(s) => s,
            Ok(slices) => {
                let state = TtState::<f64>::rank1_separable(slices);
                let raw = Box::into_raw(Box::new(state)).cast::<SmfTtState>();
                unsafe { *out_state = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// smf_ttstate_free
// ---------------------------------------------------------------------------

/// Free a `SmfTtState` handle. Null-safe; do not use after this call.
///
/// # Safety
/// `state` must be null or a live pointer from `smf_ttstate_new_separable`.
#[no_mangle]
pub unsafe extern "C" fn smf_ttstate_free(state: *mut SmfTtState) {
    if state.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(state.cast::<TtState<f64>>())) };
    }));
}

// ---------------------------------------------------------------------------
// smf_ttstate_ndim
// ---------------------------------------------------------------------------

/// Return the number of modes (d). Returns 0 if `state` is null.
///
/// # Safety
/// `state` must be null or a live `SmfTtState` pointer.
#[no_mangle]
pub unsafe extern "C" fn smf_ttstate_ndim(state: *const SmfTtState) -> usize {
    if state.is_null() {
        return 0;
    }
    let s = unsafe { &*state.cast::<TtState<f64>>() };
    s.ndim()
}

// ---------------------------------------------------------------------------
// smf_ttstate_n_j
// ---------------------------------------------------------------------------

/// Write the mode size of `axis` into `*out_n`. Returns `OutOfDomain` if
/// `axis >= ndim`.
///
/// # Safety
/// `state` and `out_n` must be non-null valid pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_ttstate_n_j(
    state: *const SmfTtState,
    axis: usize,
    out_n: *mut usize,
) -> SemiflowStatus {
    if state.is_null() || out_n.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*state.cast::<TtState<f64>>() };
        if axis >= s.ndim() {
            return SemiflowStatus::OutOfDomain;
        }
        unsafe { *out_n = s.n_j(axis) };
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// smf_ttstate_peak_rank
// ---------------------------------------------------------------------------

/// Return the peak bond rank. Returns 0 if `state` is null.
///
/// # Safety
/// `state` must be null or a live `SmfTtState` pointer.
#[no_mangle]
pub unsafe extern "C" fn smf_ttstate_peak_rank(state: *const SmfTtState) -> usize {
    if state.is_null() {
        return 0;
    }
    let s = unsafe { &*state.cast::<TtState<f64>>() };
    s.peak_rank()
}

// ---------------------------------------------------------------------------
// smf_ttstate_storage_size
// ---------------------------------------------------------------------------

/// Return the total stored scalar count. Returns 0 if `state` is null.
///
/// # Safety
/// `state` must be null or a live `SmfTtState` pointer.
#[no_mangle]
pub unsafe extern "C" fn smf_ttstate_storage_size(state: *const SmfTtState) -> usize {
    if state.is_null() {
        return 0;
    }
    let s = unsafe { &*state.cast::<TtState<f64>>() };
    s.storage_size()
}

// ---------------------------------------------------------------------------
// smf_ttstate_inner_separable
// ---------------------------------------------------------------------------

/// Compute the scalar projection `⟨f, u⟩` for a separable functional f.
///
/// `data`/`offsets`/`n_axes` follow C-2 ragged-array convention; `n_axes`
/// must equal `state.ndim()` and each functional length must equal `n_j(axis)`.
///
/// # Safety
/// All non-null pointer arguments must be valid for the stated lengths.
#[no_mangle]
pub unsafe extern "C" fn smf_ttstate_inner_separable(
    state: *const SmfTtState,
    data: *const f64,
    offsets: *const usize,
    n_axes: usize,
    out_value: *mut f64,
) -> SemiflowStatus {
    if state.is_null() || data.is_null() || offsets.is_null() || out_value.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let s = unsafe { &*state.cast::<TtState<f64>>() };
        if n_axes != s.ndim() {
            return SemiflowStatus::GridMismatch;
        }
        let off = unsafe { std::slice::from_raw_parts(offsets, n_axes + 1) };
        if let Err(st) = validate_offsets(off, n_axes) {
            return st;
        }
        let total = off[n_axes];
        let flat = unsafe { std::slice::from_raw_parts(data, total) };
        match build_slices_from_ragged(flat, off, n_axes) {
            Err(st) => st,
            Ok(funcs) => {
                // Pre-check functional lengths against n_j before inner_separable.
                for (j, fj) in funcs.iter().enumerate() {
                    if fj.len() != s.n_j(j) {
                        return SemiflowStatus::GridMismatch;
                    }
                }
                let val = s.inner_separable(&funcs);
                unsafe { *out_value = val };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// smf_tt_evolver_new
// ---------------------------------------------------------------------------

/// Construct a separable `TtChernoff<f64>` evolver.
///
/// `a[j]` / `b[j]` / `dom_min[j]` / `dom_max[j]` are per-axis arrays of
/// length `n_axes`.  `eps_round` is the TT-rounding tolerance.
///
/// # Safety
/// All pointer arguments must be valid non-null pointers with `n_axes` f64 entries
/// (or `*mut *mut SmfTtEvolver` for `out_ev`).
#[no_mangle]
pub unsafe extern "C" fn smf_tt_evolver_new(
    a: *const f64,
    b: *const f64,
    c: f64,
    dom_min: *const f64,
    dom_max: *const f64,
    n_axes: usize,
    eps_round: f64,
    out_ev: *mut *mut SmfTtEvolver,
) -> SemiflowStatus {
    if a.is_null() || b.is_null() || dom_min.is_null() || dom_max.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        if n_axes == 0 {
            return SemiflowStatus::GridMismatch;
        }
        let a_s = unsafe { std::slice::from_raw_parts(a, n_axes) };
        let b_s = unsafe { std::slice::from_raw_parts(b, n_axes) };
        let min_s = unsafe { std::slice::from_raw_parts(dom_min, n_axes) };
        let max_s = unsafe { std::slice::from_raw_parts(dom_max, n_axes) };
        match build_tt_evolver(a_s, b_s, c, min_s, max_s, eps_round) {
            Err(s) => s,
            Ok(ev) => {
                let raw = Box::into_raw(Box::new(ev)).cast::<SmfTtEvolver>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// smf_tt_evolver_evolve
// ---------------------------------------------------------------------------

/// Evolve `state` for time `t_final` using `n_steps` Chernoff steps (in-place).
///
/// Returns `OutOfDomain` if `n_steps == 0`, `t_final` is non-finite/negative, or
/// `ev.ndim() != state.ndim()`.
///
/// # Safety
/// `ev` and `state` must be non-null live pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_tt_evolver_evolve(
    ev: *const SmfTtEvolver,
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
        let evolver = unsafe { &*ev.cast::<TtChernoff<f64>>() };
        let s = unsafe { &mut *state.cast::<TtState<f64>>() };
        if evolver.ndim() != s.ndim() {
            return SemiflowStatus::OutOfDomain;
        }
        evolver.evolve(t_final, n_steps, s);
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// smf_tt_evolver_free
// ---------------------------------------------------------------------------

/// Free a `SmfTtEvolver` handle. Null-safe; do not use after this call.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_tt_evolver_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_tt_evolver_free(ev: *mut SmfTtEvolver) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<TtChernoff<f64>>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Validate the C-2 offset array (must be strictly monotone, offsets[0]==0).
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

/// Extract per-axis slices from flat buffer + offset prefix-sum array.
fn build_slices_from_ragged(
    flat: &[f64],
    off: &[usize],
    n_axes: usize,
) -> Result<Vec<Vec<f64>>, SemiflowStatus> {
    let mut slices: Vec<Vec<f64>> = Vec::with_capacity(n_axes);
    for j in 0..n_axes {
        let s = off[j];
        let e = off[j + 1];
        let slice = flat[s..e].to_vec();
        for &v in &slice {
            if !v.is_finite() {
                return Err(SemiflowStatus::NanInf);
            }
        }
        slices.push(slice);
    }
    Ok(slices)
}

/// Validate per-axis f64 array: must be finite, and satisfy `pred`.
fn validate_f64_axis(vals: &[f64], pred: impl Fn(f64) -> bool) -> SemiflowStatus {
    for &v in vals {
        if !v.is_finite() {
            return SemiflowStatus::NanInf;
        }
        if !pred(v) {
            return SemiflowStatus::NanInf;
        }
    }
    SemiflowStatus::Ok
}

/// Build a `TtChernoff<f64>` from validated parameter slices.
fn build_tt_evolver(
    a: &[f64],
    b: &[f64],
    c: f64,
    dom_min: &[f64],
    dom_max: &[f64],
    eps_round: f64,
) -> Result<TtChernoff<f64>, SemiflowStatus> {
    if !c.is_finite() || !eps_round.is_finite() {
        return Err(SemiflowStatus::NanInf);
    }
    let st = validate_f64_axis(a, |v| v >= 0.0);
    if st != SemiflowStatus::Ok {
        return Err(st);
    }
    let st = validate_f64_axis(b, |_| true);
    if st != SemiflowStatus::Ok {
        return Err(st);
    }
    let domain: Vec<(f64, f64)> = dom_min
        .iter()
        .zip(dom_max.iter())
        .map(|(&lo, &hi)| {
            if !lo.is_finite() || !hi.is_finite() || lo >= hi {
                Err(SemiflowStatus::NanInf)
            } else {
                Ok((lo, hi))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ev = TtChernoff::new(a.to_vec(), b.to_vec(), c, domain, eps_round);
    Ok(ev)
}

// SmfTtState is defined in this module and accessible as `crate::tt_ffi::SmfTtState`.
