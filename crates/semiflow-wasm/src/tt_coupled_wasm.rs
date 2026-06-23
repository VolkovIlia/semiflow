//! v9.0.0-beta WASM bindings for `CoupledTtChernoff` (Shift C, ADR-0171).
//!
//! Exposes `TtCoupledEvolver` — a JS class wrapping `CoupledTtChernoff<f64>`.
//! Advances the same `TtState` carrier used by `TtEvolver` (Gate C invariant).
//!
//! ## Fail-loud walls (pre-checked — ADR-0162)
//!
//! - Any `b_j ≠ 0` (drift deferred) → `OutOfDomain`.
//! - Non-adjacent pairs (`|k-j| != 1`) → `OutOfDomain`.
//! - Non-SPD pair block (`det ≤ 0`) → `OutOfDomain`.
//!
//! ## Coupling topology (JS tag)
//!
//! `couplingTag`: `0` = None, `1` = Tridiagonal, `2` = Pairs.
//! - None / Tridiagonal: pass `tridiagRho` (ignored for None).
//! - Pairs: pass `pairsJk: Uint32Array` (`2·n_pairs`) + `pairsRho: Float64Array` (`n_pairs`).
//!
//! ## Panic boundary (ADR-0028 Amendment 1)
//!
//! Workspace `[profile.release]` uses `panic = "abort"`.  No `catch_unwind`.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::missing_errors_doc,
    clippy::too_many_arguments
)]

use wasm_bindgen::prelude::*;

use semiflow::{CoupledTtChernoff, CouplingTopology};

use crate::error::make_js_error;
use crate::tt_wasm::TtState;

// ---------------------------------------------------------------------------
// TtCoupledEvolver JS class
// ---------------------------------------------------------------------------

/// Cross-axis TT-Chernoff evolver with `CouplingTopology` (v9.0.0-beta, §52).
///
/// Advances a `TtState` in-place — the same carrier as `TtEvolver`.
/// Gate C: `couplingTag = 0` (None) is bit-identical to `TtEvolver`.
///
/// ## Error model (`.kind`)
///
/// - `"GridMismatch"` — `n_axes == 0` or array length mismatch.
/// - `"NanInf"`       — non-finite coefficients, rho, or domain bounds.
/// - `"OutOfDomain"`  — fail-loud walls (`b≠0`, non-adjacent, non-SPD) or
///                      `n_steps == 0` / `t_final < 0` / ndim mismatch.
#[wasm_bindgen]
pub struct TtCoupledEvolver {
    inner: CoupledTtChernoff<f64>,
}

#[wasm_bindgen]
impl TtCoupledEvolver {
    /// Construct a `TtCoupledEvolver`.
    ///
    /// `a`           — `Float64Array`: per-axis diffusion (length `n_axes`).
    /// `b`           — `Float64Array`: per-axis drift — MUST be all 0 (v9 wall).
    /// `c`           — scalar reaction.
    /// `couplingTag` — `0`=None, `1`=Tridiagonal, `2`=Pairs.
    /// `tridiagRho`  — coupling strength (used iff `tag == 1`).
    /// `pairsJk`     — `Uint32Array` of length `2·n_pairs`: `[j0,k0, j1,k1, …]`.
    /// `pairsRho`    — `Float64Array` of length `n_pairs`: per-pair correlation.
    /// `domMin`      — `Float64Array`: per-axis `x_min`.
    /// `domMax`      — `Float64Array`: per-axis `x_max`.
    /// `epsRound`    — TT-rounding tolerance.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(constructor)]
    pub fn new(
        a: &js_sys::Float64Array,
        b: &js_sys::Float64Array,
        c: f64,
        coupling_tag: u32,
        tridiag_rho: f64,
        pairs_jk: &js_sys::Uint32Array,
        pairs_rho: &js_sys::Float64Array,
        dom_min: &js_sys::Float64Array,
        dom_max: &js_sys::Float64Array,
        eps_round: f64,
    ) -> Result<TtCoupledEvolver, JsValue> {
        let n = a.length() as usize;
        if n == 0 {
            return Err(make_js_error("GridMismatch", "TtCoupledEvolver: n_axes must be >= 1"));
        }
        for arr in &[b, dom_min, dom_max] {
            if arr.length() as usize != n {
                return Err(make_js_error(
                    "GridMismatch",
                    "TtCoupledEvolver: a, b, domMin, domMax must have equal length",
                ));
            }
        }
        if !c.is_finite() || !eps_round.is_finite() {
            return Err(make_js_error("NanInf", "TtCoupledEvolver: c and epsRound must be finite"));
        }
        let ev = build_coupled_evolver_wasm(
            a, b, c, coupling_tag, tridiag_rho, pairs_jk, pairs_rho, dom_min, dom_max, n,
            eps_round,
        )?;
        Ok(TtCoupledEvolver { inner: ev })
    }

    /// Evolve `state` in-place for time `t_final` with `n_steps` Chernoff steps.
    pub fn evolve(
        &self,
        state: &mut TtState,
        t_final: f64,
        n_steps: usize,
    ) -> Result<(), JsValue> {
        if n_steps == 0 {
            return Err(make_js_error(
                "OutOfDomain",
                "TtCoupledEvolver.evolve: n_steps must be >= 1",
            ));
        }
        if !t_final.is_finite() || t_final < 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "TtCoupledEvolver.evolve: t_final must be finite and >= 0",
            ));
        }
        if self.inner.ndim() != state.ndim() {
            return Err(make_js_error(
                "OutOfDomain",
                "TtCoupledEvolver.evolve: evolver ndim must match state ndim",
            ));
        }
        self.inner.evolve(t_final, n_steps, state.inner_mut());
        Ok(())
    }

    /// Return the number of tensor modes this evolver was built for.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Decode `CouplingTopology<f64>` from JS tag + pair arrays.
fn decode_topology_wasm(
    coupling_tag: u32,
    tridiag_rho: f64,
    pairs_jk: &js_sys::Uint32Array,
    pairs_rho: &js_sys::Float64Array,
    n_axes: usize,
) -> Result<CouplingTopology<f64>, JsValue> {
    match coupling_tag {
        0 => Ok(CouplingTopology::None),
        1 => {
            if !tridiag_rho.is_finite() {
                return Err(make_js_error("NanInf", "TtCoupledEvolver: tridiagRho must be finite"));
            }
            Ok(CouplingTopology::Tridiagonal(tridiag_rho))
        }
        2 => {
            let n_pairs = pairs_rho.length() as usize;
            if pairs_jk.length() as usize != 2 * n_pairs {
                return Err(make_js_error(
                    "GridMismatch",
                    "TtCoupledEvolver: pairsJk.length must equal 2 * pairsRho.length",
                ));
            }
            let mut jk = vec![0u32; 2 * n_pairs];
            let mut rho = vec![0.0f64; n_pairs];
            pairs_jk.copy_to(&mut jk);
            pairs_rho.copy_to(&mut rho);
            let mut pairs = Vec::with_capacity(n_pairs);
            for i in 0..n_pairs {
                let j = jk[2 * i] as usize;
                let k = jk[2 * i + 1] as usize;
                let r = rho[i];
                if !r.is_finite() {
                    return Err(make_js_error("NanInf", "TtCoupledEvolver: pairsRho contains NaN/Inf"));
                }
                if j >= n_axes || k >= n_axes {
                    return Err(make_js_error(
                        "GridMismatch",
                        "TtCoupledEvolver: pair index out of range",
                    ));
                }
                pairs.push((j, k, r));
            }
            Ok(CouplingTopology::Pairs(pairs))
        }
        _ => Err(make_js_error("OutOfDomain", "TtCoupledEvolver: unknown couplingTag")),
    }
}

/// Pre-check fail-loud walls (b != 0, non-adjacent, non-SPD) → `OutOfDomain`.
fn precheck_walls_wasm(
    b: &[f64],
    topology: &CouplingTopology<f64>,
    a: &[f64],
) -> Result<(), JsValue> {
    for &bj in b {
        if bj != 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "TtCoupledEvolver: drift b must be all-zero (v9 wall, ADR-0162)",
            ));
        }
    }
    if let CouplingTopology::Pairs(ref ps) = topology {
        for &(j, k, rho) in ps {
            let (lo, hi) = if j < k { (j, k) } else { (k, j) };
            if hi != lo + 1 {
                return Err(make_js_error(
                    "OutOfDomain",
                    "TtCoupledEvolver: non-adjacent pairs not supported (v9 wall)",
                ));
            }
            let det = a[lo] * a[hi] - rho * rho;
            if det <= 0.0 {
                return Err(make_js_error(
                    "OutOfDomain",
                    "TtCoupledEvolver: pair block not SPD (|rho| too large)",
                ));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_coupled_evolver_wasm(
    a_js: &js_sys::Float64Array,
    b_js: &js_sys::Float64Array,
    c: f64,
    coupling_tag: u32,
    tridiag_rho: f64,
    pairs_jk: &js_sys::Uint32Array,
    pairs_rho: &js_sys::Float64Array,
    dom_min: &js_sys::Float64Array,
    dom_max: &js_sys::Float64Array,
    n: usize,
    eps_round: f64,
) -> Result<CoupledTtChernoff<f64>, JsValue> {
    let mut a_v = vec![0.0f64; n];
    let mut b_v = vec![0.0f64; n];
    let mut lo_v = vec![0.0f64; n];
    let mut hi_v = vec![0.0f64; n];
    a_js.copy_to(&mut a_v);
    b_js.copy_to(&mut b_v);
    dom_min.copy_to(&mut lo_v);
    dom_max.copy_to(&mut hi_v);
    for &v in a_v.iter().chain(b_v.iter()).chain(lo_v.iter()).chain(hi_v.iter()) {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", "TtCoupledEvolver: coefficient/domain is NaN/Inf"));
        }
    }
    for &ai in &a_v {
        if ai < 0.0 {
            return Err(make_js_error("NanInf", "TtCoupledEvolver: diffusion a_j must be >= 0"));
        }
    }
    let domain: Vec<(f64, f64)> = lo_v
        .iter()
        .zip(hi_v.iter())
        .map(|(&lo, &hi)| {
            if lo >= hi {
                Err(make_js_error("NanInf", "TtCoupledEvolver: dom_min must be < dom_max"))
            } else {
                Ok((lo, hi))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let topology = decode_topology_wasm(coupling_tag, tridiag_rho, pairs_jk, pairs_rho, n)?;
    precheck_walls_wasm(&b_v, &topology, &a_v)?;
    Ok(CoupledTtChernoff::new(a_v, b_v, c, topology, domain, eps_round))
}
