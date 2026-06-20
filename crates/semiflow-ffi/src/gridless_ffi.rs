//! S³ gridless-particle FFI — `SmfMeasureState` and `SmfGridlessEvolver` handles.
//!
//! Implements the `smf_measurestate_*` and `smf_gridless_*` groups from
//! `contracts/semiflow-ffi.s3-carrier-handle.yaml` (v9.2.0, ADR-0171).
//!
//! ## D-monomorphism
//!
//! v9.2.0 ships D=1 as the compiled dimension.  Runtime `dim` arguments are
//! validated against `COMPILED_D = 1`; passing `dim != 1` returns `Unsupported`.
//!
//! ## Particle ABI
//!
//! `n_part` Diracs cross as two parallel flat arrays:
//!   `positions[n_part * D]` (row-major) + `weights[n_part]`.
//! Read-out (marginal) writes the projection onto `axis` into two caller buffers.
//!
//! ## Curse-escape
//!
//! The dense 3^D particle tree is NEVER materialised across the ABI — only the
//! sparse marginal (one axis projection) crosses out (C-1 invariant preserved).

#![allow(unsafe_code)]
#![allow(clippy::cast_precision_loss, clippy::too_many_arguments)]

use semiflow_core::{GridlessChernoff, MeasureState, ParticleReduction, ScratchPool};

use crate::status::SemiflowStatus;

// ---------------------------------------------------------------------------
// Compiled dimension (monomorphic v9.2.0 D=1 build)
// ---------------------------------------------------------------------------

/// The compiled dimension for this build (D = 1 for v9.2.0).
const COMPILED_D: usize = 1;

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque C handle to `Box<MeasureState<f64, 1>>`.
///
/// The dense 3^D particle tree is NEVER materialised — only sparse
/// marginals and scalar observables cross the ABI (curse-escape, C-1).
#[repr(C)]
pub struct SmfMeasureState {
    _private: [u8; 0],
}

/// Opaque C handle to `Box<GridlessChernoff<f64, 1>>`.
#[repr(C)]
pub struct SmfGridlessEvolver {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// smf_measurestate_new
// ---------------------------------------------------------------------------

/// Construct a `MeasureState<f64,1>` from particle buffers.
///
/// `dim` must equal the compiled D (= 1 for this build); otherwise returns
/// `Unsupported`.  `n_part` must be ≥ 1.  All positions/weights must be finite.
///
/// On success, `*out_state` receives the freshly allocated handle.
///
/// # Safety
/// `positions` (length `n_part * dim`), `weights` (length `n_part`), and
/// `out_state` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_measurestate_new(
    positions: *const f64,
    weights: *const f64,
    n_part: usize,
    dim: usize,
    out_state: *mut *mut SmfMeasureState,
) -> SemiflowStatus {
    if positions.is_null() || weights.is_null() || out_state.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if dim != COMPILED_D {
        return SemiflowStatus::Unsupported;
    }
    if n_part == 0 {
        return SemiflowStatus::GridMismatch;
    }
    catch_panic!({
        let pos = unsafe { std::slice::from_raw_parts(positions, n_part * COMPILED_D) };
        let wts = unsafe { std::slice::from_raw_parts(weights, n_part) };
        match build_measure_state(pos, wts, n_part) {
            Err(s) => s,
            Ok(ms) => {
                let raw = Box::into_raw(Box::new(ms)).cast::<SmfMeasureState>();
                unsafe { *out_state = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// smf_measurestate_free
// ---------------------------------------------------------------------------

/// Free a `SmfMeasureState` handle. Null-safe.
///
/// # Safety
/// `state` must be null or a live pointer from `smf_measurestate_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_measurestate_free(state: *mut SmfMeasureState) {
    if state.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(state.cast::<MeasureState<f64, 1>>())) };
    }));
}

// ---------------------------------------------------------------------------
// smf_measurestate_n_diracs
// ---------------------------------------------------------------------------

/// Return the number of Dirac atoms. Returns 0 if `state` is null.
///
/// # Safety
/// `state` must be null or a live `SmfMeasureState` pointer.
#[no_mangle]
pub unsafe extern "C" fn smf_measurestate_n_diracs(state: *const SmfMeasureState) -> usize {
    if state.is_null() {
        return 0;
    }
    let ms = unsafe { &*state.cast::<MeasureState<f64, 1>>() };
    ms.n_diracs()
}

// ---------------------------------------------------------------------------
// smf_measurestate_total_variation
// ---------------------------------------------------------------------------

/// Write the total-variation norm `‖ρ‖_TV` into `*out_tv`.
///
/// # Safety
/// `state` and `out_tv` must be non-null valid pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_measurestate_total_variation(
    state: *const SmfMeasureState,
    out_tv: *mut f64,
) -> SemiflowStatus {
    if state.is_null() || out_tv.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let ms = unsafe { &*state.cast::<MeasureState<f64, 1>>() };
        unsafe { *out_tv = ms.total_variation() };
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// smf_measurestate_second_moment
// ---------------------------------------------------------------------------

/// Write the second moment `⟨x², ρ⟩` into `*out_m2`.
///
/// # Safety
/// `state` and `out_m2` must be non-null valid pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_measurestate_second_moment(
    state: *const SmfMeasureState,
    out_m2: *mut f64,
) -> SemiflowStatus {
    if state.is_null() || out_m2.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        let ms = unsafe { &*state.cast::<MeasureState<f64, 1>>() };
        unsafe { *out_m2 = ms.second_moment() };
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// smf_measurestate_marginal
// ---------------------------------------------------------------------------

/// Write the 1-D marginal onto `axis` into caller-owned buffers.
///
/// The marginal is the set of `(positions[i][axis], weights[i])` pairs for all
/// Diracs `i`.  Curse-escape preserved: only a sparse marginal crosses the ABI,
/// never a dense grid.
///
/// Returns `GridMismatch` with `*out_n = n_diracs` if `cap < n_diracs` (retry
/// signal).  Returns `OutOfDomain` if `axis >= COMPILED_D`.
///
/// # Safety
/// All pointer arguments must be non-null. `out_pos`/`out_wt` must be writable
/// for `cap` f64 elements.
#[no_mangle]
pub unsafe extern "C" fn smf_measurestate_marginal(
    state: *const SmfMeasureState,
    out_pos: *mut f64,
    out_wt: *mut f64,
    cap: usize,
    axis: usize,
    out_n: *mut usize,
) -> SemiflowStatus {
    if state.is_null() || out_pos.is_null() || out_wt.is_null() || out_n.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if axis >= COMPILED_D {
        return SemiflowStatus::OutOfDomain;
    }
    catch_panic!({
        let ms = unsafe { &*state.cast::<MeasureState<f64, 1>>() };
        let diracs = ms.diracs();
        let n = diracs.len();
        if cap < n {
            // Signal required size for retry.
            unsafe { *out_n = n };
            return SemiflowStatus::GridMismatch;
        }
        let pos_sl = unsafe { std::slice::from_raw_parts_mut(out_pos, cap) };
        let wt_sl = unsafe { std::slice::from_raw_parts_mut(out_wt, cap) };
        for (i, (pos, w)) in diracs.iter().enumerate() {
            pos_sl[i] = pos[axis];
            wt_sl[i] = *w;
        }
        unsafe { *out_n = n };
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// smf_gridless_new
// ---------------------------------------------------------------------------

/// Construct a `GridlessChernoff<f64,1>` evolver.
///
/// `dim` must equal 1 (compiled D); otherwise returns `Unsupported`.
/// `reduction_tag`: 0 = `WeightedVoronoi { cap: voronoi_cap }`,
///                  1 = `GaussianBackground` (pass-through stub).
///
/// # Safety
/// `a`, `b`, and `out_ev` must be valid non-null pointers with `dim` elements.
#[no_mangle]
pub unsafe extern "C" fn smf_gridless_new(
    a: *const f64,
    b: *const f64,
    c: f64,
    dim: usize,
    reduction_tag: u32,
    voronoi_cap: usize,
    out_ev: *mut *mut SmfGridlessEvolver,
) -> SemiflowStatus {
    if a.is_null() || b.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if dim != COMPILED_D {
        return SemiflowStatus::Unsupported;
    }
    catch_panic!({
        let a_s = unsafe { std::slice::from_raw_parts(a, COMPILED_D) };
        let b_s = unsafe { std::slice::from_raw_parts(b, COMPILED_D) };
        match build_gridless_evolver(a_s, b_s, c, reduction_tag, voronoi_cap) {
            Err(s) => s,
            Ok(ev) => {
                let raw = Box::into_raw(Box::new(ev)).cast::<SmfGridlessEvolver>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// smf_gridless_apply
// ---------------------------------------------------------------------------

/// Apply one Chernoff step of size `tau`: `dst` is overwritten with the
/// push-forward of `src`.
///
/// `src` is borrowed read-only; `dst` is overwritten entirely.  A fresh
/// `ScratchPool` is created per call (matches v3 `evolve_into` pattern).
///
/// # Safety
/// `ev`, `src`, and `dst` must be non-null live pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_gridless_apply(
    ev: *const SmfGridlessEvolver,
    tau: f64,
    src: *const SmfMeasureState,
    dst: *mut SmfMeasureState,
) -> SemiflowStatus {
    if ev.is_null() || src.is_null() || dst.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if !tau.is_finite() || tau < 0.0 {
        return SemiflowStatus::OutOfDomain;
    }
    catch_panic!({
        use semiflow_core::chernoff::ChernoffFunction;
        let evolver = unsafe { &*ev.cast::<GridlessChernoff<f64, 1>>() };
        let src_ms = unsafe { &*src.cast::<MeasureState<f64, 1>>() };
        let dst_ms = unsafe { &mut *dst.cast::<MeasureState<f64, 1>>() };
        let mut pool = ScratchPool::<f64>::new();
        match evolver.apply_into(tau, src_ms, dst_ms, &mut pool) {
            Ok(()) => SemiflowStatus::Ok,
            Err(e) => SemiflowStatus::from(&e),
        }
    })
}

// ---------------------------------------------------------------------------
// smf_gridless_evolve
// ---------------------------------------------------------------------------

/// Evolve `state` in-place for time `t_final` using `n_steps` Chernoff steps.
///
/// Uses two alternating scratch buffers to avoid re-allocating on each step.
/// The final result is written back into `state`.
///
/// # Safety
/// `ev` and `state` must be non-null live pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_gridless_evolve(
    ev: *const SmfGridlessEvolver,
    t_final: f64,
    n_steps: usize,
    state: *mut SmfMeasureState,
) -> SemiflowStatus {
    if ev.is_null() || state.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if n_steps == 0 || !t_final.is_finite() || t_final < 0.0 {
        return SemiflowStatus::OutOfDomain;
    }
    catch_panic!({
        use semiflow_core::chernoff::ChernoffFunction;
        let evolver = unsafe { &*ev.cast::<GridlessChernoff<f64, 1>>() };
        let ms = unsafe { &mut *state.cast::<MeasureState<f64, 1>>() };
        let tau = t_final / n_steps as f64;
        // Two owned scratch buffers — no raw-pointer aliasing.
        let mut buf_a: MeasureState<f64, 1> = ms.clone();
        let mut buf_b: MeasureState<f64, 1> = ms.clone();
        let mut pool = ScratchPool::<f64>::new();
        let mut a_is_src = true;
        for _ in 0..n_steps {
            let status = if a_is_src {
                evolver.apply_into(tau, &buf_a, &mut buf_b, &mut pool)
            } else {
                evolver.apply_into(tau, &buf_b, &mut buf_a, &mut pool)
            };
            if let Err(e) = status {
                return SemiflowStatus::from(&e);
            }
            a_is_src = !a_is_src;
        }
        // Copy result back into `state`.
        *ms = if a_is_src { buf_a } else { buf_b };
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// smf_gridless_free
// ---------------------------------------------------------------------------

/// Free a `SmfGridlessEvolver` handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_gridless_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_gridless_free(ev: *mut SmfGridlessEvolver) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<GridlessChernoff<f64, 1>>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Validate particle buffers and build `MeasureState<f64,1>`.
fn build_measure_state(
    pos: &[f64],
    wts: &[f64],
    n_part: usize,
) -> Result<MeasureState<f64, 1>, SemiflowStatus> {
    let mut particles: Vec<([f64; 1], f64)> = Vec::with_capacity(n_part);
    for i in 0..n_part {
        let p = pos[i * COMPILED_D];
        let w = wts[i];
        if !p.is_finite() || !w.is_finite() {
            return Err(SemiflowStatus::NanInf);
        }
        particles.push(([p], w));
    }
    Ok(MeasureState::<f64, 1>::from_particles(&particles))
}

/// Build `GridlessChernoff<f64,1>` from validated inputs.
fn build_gridless_evolver(
    a: &[f64],
    b: &[f64],
    c: f64,
    reduction_tag: u32,
    voronoi_cap: usize,
) -> Result<GridlessChernoff<f64, 1>, SemiflowStatus> {
    if !c.is_finite() {
        return Err(SemiflowStatus::NanInf);
    }
    for &v in a {
        if !v.is_finite() || v < 0.0 {
            return Err(SemiflowStatus::NanInf);
        }
    }
    for &v in b {
        if !v.is_finite() {
            return Err(SemiflowStatus::NanInf);
        }
    }
    let reduction = match reduction_tag {
        0 => {
            if voronoi_cap == 0 {
                return Err(SemiflowStatus::OutOfDomain);
            }
            ParticleReduction::WeightedVoronoi { cap: voronoi_cap }
        }
        1 => ParticleReduction::GaussianBackground,
        _ => return Err(SemiflowStatus::OutOfDomain),
    };
    let a_arr: [f64; 1] = [a[0]];
    let b_arr: [f64; 1] = [b[0]];
    Ok(GridlessChernoff::<f64, 1>::new(a_arr, b_arr, c, reduction))
}
