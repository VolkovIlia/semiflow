//! WASM binding for `VarCoefTt` — additive-separable variable-coefficient
//! TT evolver (ADR-0178, math §52.10, issue #2).
//!
//! Mirrors `tt_wasm.rs` for `TtEvolver` / `TtState`.
//!
//! ## JS API
//!
//! ```js
//! const n = 16; const d = 2;
//! // Build per-axis a/b/v arrays (same length n per axis)
//! const off = new Uint32Array([0, n, 2*n]);       // 2 axes, n points each
//! const a   = new Float64Array(2*n).fill(0.5);     // const diffusion
//! const b   = new Float64Array(2*n).fill(0.0);     // zero drift
//! const v   = new Float64Array(2*n).fill(0.0);     // zero reaction
//! const lo  = new Float64Array([-3.0, -3.0]);
//! const hi  = new Float64Array([ 3.0,  3.0]);
//!
//! const ev  = new VarCoefTtEvolver(a, off, b, off, v, off, lo, hi, 1e-10);
//!
//! // TtState constructed per tt_wasm.rs pattern (reused as-is)
//! const state = new TtState(data, stateOffsets);
//! ev.evolve(state, 0.1, 4);
//! ```
//!
//! ## Ragged-array convention
//!
//! Each coefficient takes `(data: Float64Array, offsets: Uint32Array)`.
//! Mirrors the C-2 convention from `tt_ffi.rs` / `tt_varcoef_ffi.rs`.
//!
//! ## Panic boundary (ADR-0028 Amendment 1)
//!
//! `panic = "abort"` — no `catch_unwind`.  Validate before dispatching.

#![allow(clippy::too_many_arguments)]

use js_sys::{Float64Array, Uint32Array};
use semiflow::VarCoefTt;
use wasm_bindgen::prelude::*;

use crate::error::make_js_error;
use crate::tt_wasm::TtState;

// ---------------------------------------------------------------------------
// VarCoefTtEvolver JS class
// ---------------------------------------------------------------------------

/// Additive-separable variable-coefficient TT evolver (ADR-0178, math §52.10).
///
/// Advances a `TtState` in-place via `evolve(state, t_final, n_steps)`.
///
/// ## Constructor parameters (Float64Array + Uint32Array pairs)
///
/// Each ragged array is passed as `(data, offsets)` where `offsets` is a
/// `Uint32Array` of length `n_axes + 1` (C-2 prefix-sum convention):
///
/// - `aData` / `aOffsets` — per-axis diffusion `aⱼ(xⱼ)` (all > 0).
/// - `bData` / `bOffsets` — per-axis drift `bⱼ(xⱼ)`.
/// - `vData` / `vOffsets` — per-axis reaction `vⱼ(xⱼ)` (empty = zero).
/// - `domLo` / `domHi`    — `Float64Array` of length `n_axes`.
/// - `epsRound`            — TT-rounding tolerance (finite, >= 0).
///
/// ## Error model (`.kind`)
///
/// - `"GridMismatch"` — zero axes or invalid offsets.
/// - `"NanInf"`       — non-finite coefficient or domain.
/// - `"OutOfDomain"`  — parabolicity (`aⱼ <= 0`), shape mismatch, `nⱼ < 2`.
#[wasm_bindgen(js_name = "VarCoefTtEvolver")]
pub struct WasmVarCoefTtEvolver {
    inner: VarCoefTt<f64>,
}

#[wasm_bindgen(js_class = "VarCoefTtEvolver")]
impl WasmVarCoefTtEvolver {
    /// Construct a `VarCoefTtEvolver`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        a_data: &Float64Array,
        a_offsets: &Uint32Array,
        b_data: &Float64Array,
        b_offsets: &Uint32Array,
        v_data: &Float64Array,
        v_offsets: &Uint32Array,
        dom_lo: &Float64Array,
        dom_hi: &Float64Array,
        eps_round: f64,
    ) -> Result<WasmVarCoefTtEvolver, JsValue> {
        let a_off = decode_offsets(a_offsets, "VarCoefTtEvolver.aOffsets")?;
        let n_axes = a_off.len() - 1;
        if n_axes == 0 {
            return Err(make_js_error("GridMismatch", "VarCoefTtEvolver: n_axes must be >= 1"));
        }
        let b_off = decode_offsets(b_offsets, "VarCoefTtEvolver.bOffsets")?;
        let v_off = decode_offsets(v_offsets, "VarCoefTtEvolver.vOffsets")?;
        if b_off.len() - 1 != n_axes || v_off.len() - 1 != n_axes {
            return Err(make_js_error(
                "GridMismatch",
                "VarCoefTtEvolver: all offset arrays must have the same n_axes",
            ));
        }
        if !eps_round.is_finite() {
            return Err(make_js_error("NanInf", "VarCoefTtEvolver: epsRound must be finite"));
        }
        let a = extract_ragged(a_data, &a_off, n_axes, "aData")?;
        let b = extract_ragged(b_data, &b_off, n_axes, "bData")?;
        let v = extract_ragged(v_data, &v_off, n_axes, "vData")?;
        let domain = extract_domain(dom_lo, dom_hi, n_axes)?;
        let ev = VarCoefTt::<f64>::new(a, b, v, domain, eps_round)
            .map_err(|e| make_js_error("OutOfDomain", &format!("{e}")))?;
        Ok(WasmVarCoefTtEvolver { inner: ev })
    }

    /// Number of axes this evolver was built for.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Evolve `state` in-place for `tFinal` using `nSteps` steps.
    ///
    /// Throws `OutOfDomain` if `nSteps == 0`, `tFinal < 0`, or ndim mismatch.
    pub fn evolve(
        &self,
        state: &mut TtState,
        t_final: f64,
        n_steps: usize,
    ) -> Result<(), JsValue> {
        if n_steps == 0 {
            return Err(make_js_error("OutOfDomain", "VarCoefTtEvolver.evolve: nSteps must be >= 1"));
        }
        if !t_final.is_finite() || t_final < 0.0 {
            return Err(make_js_error("OutOfDomain", "VarCoefTtEvolver.evolve: tFinal must be finite >= 0"));
        }
        if self.inner.ndim() != state.inner_mut().ndim() {
            return Err(make_js_error("OutOfDomain", "VarCoefTtEvolver.evolve: ndim mismatch"));
        }
        self.inner.evolve(t_final, n_steps, state.inner_mut());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Validate and decode a `Uint32Array` into prefix-sum offsets.
fn decode_offsets(offsets: &Uint32Array, ctx: &str) -> Result<Vec<usize>, JsValue> {
    let len = offsets.length() as usize;
    if len < 2 {
        return Err(make_js_error("GridMismatch", &format!("{ctx}: offsets must have length >= 2")));
    }
    let v: Vec<usize> = (0..len).map(|i| offsets.get_index(i as u32) as usize).collect();
    if v[0] != 0 {
        return Err(make_js_error("GridMismatch", &format!("{ctx}: offsets[0] must be 0")));
    }
    for win in v.windows(2) {
        // Allow equal adjacent offsets (empty axis = zero-length reaction)
        if win[1] < win[0] {
            return Err(make_js_error("GridMismatch", &format!("{ctx}: offsets must be non-decreasing")));
        }
    }
    Ok(v)
}

/// Extract per-axis slices from `Float64Array` + offset vec.
fn extract_ragged(
    data: &Float64Array,
    off: &[usize],
    n_axes: usize,
    ctx: &str,
) -> Result<Vec<Vec<f64>>, JsValue> {
    let total = off[n_axes];
    if data.length() as usize != total {
        return Err(make_js_error("GridMismatch", &format!("{ctx}.length must equal offsets[n_axes]")));
    }
    let mut raw = vec![0.0f64; total];
    data.copy_to(&mut raw);
    let mut slices = Vec::with_capacity(n_axes);
    for j in 0..n_axes {
        let sl = raw[off[j]..off[j + 1]].to_vec();
        for &x in &sl {
            if !x.is_finite() {
                return Err(make_js_error("NanInf", &format!("{ctx}: axis {j} contains NaN/Inf")));
            }
        }
        slices.push(sl);
    }
    Ok(slices)
}

/// Extract `(lo, hi)` domain from two `Float64Array` length `n_axes`.
fn extract_domain(
    lo: &Float64Array,
    hi: &Float64Array,
    n_axes: usize,
) -> Result<Vec<(f64, f64)>, JsValue> {
    if lo.length() as usize != n_axes || hi.length() as usize != n_axes {
        return Err(make_js_error("GridMismatch", "VarCoefTtEvolver: domLo/domHi length must equal n_axes"));
    }
    let mut lo_v = vec![0.0f64; n_axes];
    let mut hi_v = vec![0.0f64; n_axes];
    lo.copy_to(&mut lo_v);
    hi.copy_to(&mut hi_v);
    lo_v.iter()
        .zip(hi_v.iter())
        .enumerate()
        .map(|(j, (&l, &h))| {
            if !l.is_finite() || !h.is_finite() {
                Err(make_js_error("NanInf", &format!("VarCoefTtEvolver: domain[{j}] NaN/Inf")))
            } else if l >= h {
                Err(make_js_error("OutOfDomain", &format!("VarCoefTtEvolver: domain[{j}].lo >= hi")))
            } else {
                Ok((l, h))
            }
        })
        .collect()
}
