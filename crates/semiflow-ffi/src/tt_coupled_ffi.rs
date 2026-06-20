//! S³ coupled tensor-train FFI — `SmfTtCoupledEvolver` handle.
//!
//! Implements the `smf_tt_coupled_*` group from
//! `contracts/semiflow-ffi.s3-carrier-handle.yaml` (v9.2.0, ADR-0171).
//!
//! The coupled evolver advances the SAME `SmfTtState` carrier as the separable
//! evolver (Gate C: `CouplingTopology::None` is bit-identical to `TtChernoff`).
//!
//! ## Fail-loud walls (pre-checked, NOT panicked — C-4)
//!
//! - `b_j ≠ 0` (drift deferred, `tt_coupled.rs:127`) → `OutOfDomain`.
//! - Non-adjacent pairs `|k-j| != 1` (`tt_coupled.rs:141`) → `OutOfDomain`.
//! - Non-SPD pair block `det(B) ≤ 0` → `OutOfDomain`.

#![allow(unsafe_code)]
#![allow(clippy::cast_precision_loss, clippy::too_many_arguments)]

use semiflow_core::{CoupledTtChernoff, CouplingTopology};

use crate::{status::SemiflowStatus, tt_ffi::SmfTtState};

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque C handle to `Box<CoupledTtChernoff<f64>>`.
///
/// The coupled evolver advances `SmfTtState` in-place (same carrier).
/// Free with `smf_tt_coupled_free`.
#[repr(C)]
pub struct SmfTtCoupledEvolver {
    _private: [u8; 0],
}

// ---------------------------------------------------------------------------
// smf_tt_coupled_new
// ---------------------------------------------------------------------------

/// Construct a `CoupledTtChernoff<f64>` evolver.
///
/// Coupling topology is specified via `coupling_tag` (0=None, 1=Tridiagonal,
/// 2=Pairs). Pairs cross as `(pairs_jk[2*n_pairs], pairs_rho[n_pairs])`.
///
/// **Fail-loud walls pre-checked here** (see contract C-4 / ADR-0162):
/// any `b_j ≠ 0`, non-adjacent pair, or non-SPD block → `OutOfDomain`.
///
/// # Safety
/// All non-null pointer arguments must be valid for the stated lengths.
/// Null-check guard for `smf_tt_coupled_new` pointer args.
fn coupled_new_null_check(
    a: *const f64,
    b: *const f64,
    dom_min: *const f64,
    dom_max: *const f64,
    out_ev: *mut *mut SmfTtCoupledEvolver,
    coupling_tag: u32,
    pairs_jk: *const usize,
    pairs_rho: *const f64,
) -> SemiflowStatus {
    if a.is_null() || b.is_null() || dom_min.is_null() || dom_max.is_null() || out_ev.is_null() {
        return SemiflowStatus::NullPtr;
    }
    if coupling_tag == 2 && (pairs_jk.is_null() || pairs_rho.is_null()) {
        return SemiflowStatus::NullPtr;
    }
    SemiflowStatus::Ok
}

/// Construct a `CoupledTtChernoff<f64>` evolver.
///
/// Coupling topology is specified via `coupling_tag` (0=None, 1=Tridiagonal,
/// 2=Pairs). Pairs cross as `(pairs_jk[2*n_pairs], pairs_rho[n_pairs])`.
///
/// **Fail-loud walls pre-checked here** (see contract C-4 / ADR-0162):
/// any `b_j ≠ 0`, non-adjacent pair, or non-SPD block → `OutOfDomain`.
///
/// # Safety
/// All non-null pointer arguments must be valid for the stated lengths.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn smf_tt_coupled_new(
    a: *const f64,
    b: *const f64,
    c: f64,
    coupling_tag: u32,
    tridiag_rho: f64,
    pairs_jk: *const usize,
    pairs_rho: *const f64,
    n_pairs: usize,
    dom_min: *const f64,
    dom_max: *const f64,
    n_axes: usize,
    eps_round: f64,
    out_ev: *mut *mut SmfTtCoupledEvolver,
) -> SemiflowStatus {
    let nc = coupled_new_null_check(a, b, dom_min, dom_max, out_ev, coupling_tag, pairs_jk, pairs_rho);
    if nc != SemiflowStatus::Ok {
        return nc;
    }
    catch_panic!({
        if n_axes == 0 {
            return SemiflowStatus::GridMismatch;
        }
        let a_s = unsafe { std::slice::from_raw_parts(a, n_axes) };
        let b_s = unsafe { std::slice::from_raw_parts(b, n_axes) };
        let min_s = unsafe { std::slice::from_raw_parts(dom_min, n_axes) };
        let max_s = unsafe { std::slice::from_raw_parts(dom_max, n_axes) };
        match build_coupled_evolver(a_s, b_s, c, coupling_tag, tridiag_rho,
                pairs_jk, pairs_rho, n_pairs, min_s, max_s, eps_round, n_axes) {
            Err(s) => s,
            Ok(ev) => {
                let raw = Box::into_raw(Box::new(ev)).cast::<SmfTtCoupledEvolver>();
                unsafe { *out_ev = raw };
                SemiflowStatus::Ok
            }
        }
    })
}

// ---------------------------------------------------------------------------
// smf_tt_coupled_evolve
// ---------------------------------------------------------------------------

/// Evolve `state` for time `t_final` using `n_steps` Chernoff steps (in-place).
///
/// Same carrier (`SmfTtState`) as the separable evolver — Gate C: `None` topology
/// is bit-identical to `smf_tt_evolver_evolve`.
///
/// # Safety
/// `ev` and `state` must be non-null live pointers.
#[no_mangle]
pub unsafe extern "C" fn smf_tt_coupled_evolve(
    ev: *const SmfTtCoupledEvolver,
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
        let evolver = unsafe { &*ev.cast::<CoupledTtChernoff<f64>>() };
        let s = unsafe { &mut *state.cast::<semiflow_core::TtState<f64>>() };
        if evolver.ndim() != s.ndim() {
            return SemiflowStatus::OutOfDomain;
        }
        evolver.evolve(t_final, n_steps, s);
        SemiflowStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// smf_tt_coupled_free
// ---------------------------------------------------------------------------

/// Free a `SmfTtCoupledEvolver` handle. Null-safe.
///
/// # Safety
/// `ev` must be null or a live pointer from `smf_tt_coupled_new`.
#[no_mangle]
pub unsafe extern "C" fn smf_tt_coupled_free(ev: *mut SmfTtCoupledEvolver) {
    if ev.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unsafe { drop(Box::from_raw(ev.cast::<CoupledTtChernoff<f64>>())) };
    }));
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Reconstruct `CouplingTopology<f64>` from the ABI tag + pair buffers.
///
/// # Safety
/// Caller guarantees that `pairs_jk` and `pairs_rho` point to valid memory
/// when `coupling_tag == 2`.
unsafe fn decode_topology(
    coupling_tag: u32,
    tridiag_rho: f64,
    pairs_jk: *const usize,
    pairs_rho: *const f64,
    n_pairs: usize,
    n_axes: usize,
) -> Result<CouplingTopology<f64>, SemiflowStatus> {
    match coupling_tag {
        0 => Ok(CouplingTopology::None),
        1 => {
            if !tridiag_rho.is_finite() {
                return Err(SemiflowStatus::NanInf);
            }
            Ok(CouplingTopology::Tridiagonal(tridiag_rho))
        }
        2 => {
            let jk = unsafe { std::slice::from_raw_parts(pairs_jk, 2 * n_pairs) };
            let rho = unsafe { std::slice::from_raw_parts(pairs_rho, n_pairs) };
            let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(n_pairs);
            for i in 0..n_pairs {
                let j = jk[2 * i];
                let k = jk[2 * i + 1];
                let r = rho[i];
                if !r.is_finite() {
                    return Err(SemiflowStatus::NanInf);
                }
                if j >= n_axes || k >= n_axes {
                    return Err(SemiflowStatus::GridMismatch);
                }
                pairs.push((j, k, r));
            }
            Ok(CouplingTopology::Pairs(pairs))
        }
        _ => Err(SemiflowStatus::OutOfDomain),
    }
}

/// Pre-check fail-loud walls before calling `CoupledTtChernoff::new`.
fn precheck_coupled_walls(
    b: &[f64],
    topology: &CouplingTopology<f64>,
    a: &[f64],
) -> SemiflowStatus {
    // Wall 1: drift b != 0 deferred (ADR-0162).
    for &bj in b {
        if bj != 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
    }
    // Wall 2: non-adjacent pairs.
    // Wall 3: non-SPD pair block.
    if let CouplingTopology::Pairs(ref ps) = topology {
        for &(j, k, rho) in ps {
            let (lo, hi) = if j < k { (j, k) } else { (k, j) };
            if hi != lo + 1 {
                return SemiflowStatus::OutOfDomain;
            }
            // SPD: det = a_j * a_k - rho^2 * a_j * a_k = a_j*a_k*(1 - rho^2)
            // Strictly: det(B) = a[lo]*a[hi] - rho^2 > 0
            let det = a[lo] * a[hi] - rho * rho;
            if det <= 0.0 {
                return SemiflowStatus::OutOfDomain;
            }
        }
    }
    SemiflowStatus::Ok
}

/// Validate scalar coefficients and build domain vec.
fn validate_coeffs_and_domain(
    a: &[f64],
    b: &[f64],
    c: f64,
    eps_round: f64,
    min_s: &[f64],
    max_s: &[f64],
) -> Result<Vec<(f64, f64)>, SemiflowStatus> {
    if !c.is_finite() || !eps_round.is_finite() {
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
    min_s
        .iter()
        .zip(max_s.iter())
        .map(|(&lo, &hi)| {
            if !lo.is_finite() || !hi.is_finite() || lo >= hi {
                Err(SemiflowStatus::NanInf)
            } else {
                Ok((lo, hi))
            }
        })
        .collect::<Result<Vec<_>, _>>()
}

/// Build `CoupledTtChernoff<f64>` from validated inputs.
#[allow(clippy::too_many_arguments)]
unsafe fn build_coupled_evolver(
    a: &[f64],
    b: &[f64],
    c: f64,
    coupling_tag: u32,
    tridiag_rho: f64,
    pairs_jk: *const usize,
    pairs_rho: *const f64,
    n_pairs: usize,
    min_s: &[f64],
    max_s: &[f64],
    eps_round: f64,
    n_axes: usize,
) -> Result<CoupledTtChernoff<f64>, SemiflowStatus> {
    let domain = validate_coeffs_and_domain(a, b, c, eps_round, min_s, max_s)?;
    let topology = unsafe {
        decode_topology(coupling_tag, tridiag_rho, pairs_jk, pairs_rho, n_pairs, n_axes)
    }?;
    let wall_st = precheck_coupled_walls(b, &topology, a);
    if wall_st != SemiflowStatus::Ok {
        return Err(wall_st);
    }
    // CoupledTtChernoff::new panics on the same conditions pre-checked above.
    // catch_panic! at the call site handles any residual panic from other assertions.
    Ok(CoupledTtChernoff::new(a.to_vec(), b.to_vec(), c, topology, domain, eps_round))
}
