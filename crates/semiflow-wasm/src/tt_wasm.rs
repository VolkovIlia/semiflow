//! v9.0.0-beta WASM bindings for `TtState` and `TtEvolver` (Shift C, ADR-0171).
//!
//! Exposes two JS classes:
//! - `TtState`   — tensor-train carrier (rank-1 separable construction).
//! - `TtEvolver` — separable diagonal-A TT-Chernoff evolver.
//!
//! ## JS API (minimal example)
//!
//! ```js
//! const data    = new Float64Array([1,2,3, 0,1,0]);  // two axes
//! const offsets = new Uint32Array([0, 3, 6]);
//! const state   = new TtState(data, offsets);
//! const evolver = new TtEvolver(
//!   new Float64Array([0.5, 0.5]),  // a
//!   new Float64Array([0.0, 0.0]),  // b
//!   0.0,                           // c
//!   new Float64Array([-1.0, -1.0]),// dom_min
//!   new Float64Array([1.0, 1.0]),  // dom_max
//!   1e-8,                          // eps_round
//! );
//! evolver.evolve(state, 0.1, 4);
//! const inner = state.innerSeparable(data, offsets);
//! ```
//!
//! ## Ragged-array convention (mirrors C-2 from contract)
//!
//! `data: Float64Array` + `offsets: Uint32Array` of length `n_axes + 1`.
//! Axis `j` occupies `data[offsets[j] .. offsets[j+1]]`.
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

use semiflow::{TtChernoff, TtState as CoreTtState};

use crate::error::make_js_error;

// ---------------------------------------------------------------------------
// Ragged-array helpers (mirrors FFI tt_ffi.rs C-2 convention)
// ---------------------------------------------------------------------------

/// Validate and decode a ragged-array offsets `Uint32Array`.
fn decode_offsets(offsets: &js_sys::Uint32Array) -> Result<Vec<usize>, JsValue> {
    let len = offsets.length() as usize;
    if len < 2 {
        return Err(make_js_error("GridMismatch", "TT: offsets must have length >= 2"));
    }
    let v: Vec<usize> = (0..len).map(|i| offsets.get_index(i as u32) as usize).collect();
    if v[0] != 0 {
        return Err(make_js_error("GridMismatch", "TT: offsets[0] must be 0"));
    }
    for win in v.windows(2) {
        if win[1] <= win[0] {
            return Err(make_js_error("GridMismatch", "TT: offsets must be strictly increasing"));
        }
    }
    Ok(v)
}

/// Extract per-axis slices from flat `Float64Array` + offset vec.
fn extract_slices(
    data: &js_sys::Float64Array,
    offsets: &[usize],
    n_axes: usize,
) -> Result<Vec<Vec<f64>>, JsValue> {
    let total = offsets[n_axes];
    if data.length() as usize != total {
        return Err(make_js_error("GridMismatch", "TT: data.length must equal offsets[n_axes]"));
    }
    let mut raw = vec![0.0f64; total];
    data.copy_to(&mut raw);
    let mut slices = Vec::with_capacity(n_axes);
    for j in 0..n_axes {
        let s = offsets[j];
        let e = offsets[j + 1];
        let sl = raw[s..e].to_vec();
        for &v in &sl {
            if !v.is_finite() {
                return Err(make_js_error("NanInf", "TT: data contains NaN or Inf"));
            }
        }
        slices.push(sl);
    }
    Ok(slices)
}

// ---------------------------------------------------------------------------
// TtState JS class
// ---------------------------------------------------------------------------

/// Tensor-train carrier state (rank-1 separable construction, v9.0.0-beta).
///
/// Built from per-axis slices passed as a flat `Float64Array` with a
/// `Uint32Array` prefix-sum offsets array (C-2 ragged convention).
///
/// ## Error model (`.kind`)
///
/// - `"GridMismatch"` — invalid offsets or empty axes.
/// - `"NanInf"`       — non-finite data.
/// - `"OutOfDomain"`  — axis out of range.
#[wasm_bindgen]
pub struct TtState {
    inner: CoreTtState<f64>,
}

#[wasm_bindgen]
impl TtState {
    /// Construct a rank-1 separable `TtState` from per-axis slices.
    ///
    /// `data`    — `Float64Array`: concatenated axis slices (axis 0 first).
    /// `offsets` — `Uint32Array`: prefix-sum offsets, length `n_axes + 1`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        data: &js_sys::Float64Array,
        offsets: &js_sys::Uint32Array,
    ) -> Result<TtState, JsValue> {
        let off = decode_offsets(offsets)?;
        let n_axes = off.len() - 1;
        if n_axes == 0 {
            return Err(make_js_error("GridMismatch", "TtState: n_axes must be >= 1"));
        }
        let slices = extract_slices(data, &off, n_axes)?;
        Ok(TtState { inner: CoreTtState::<f64>::rank1_separable(slices) })
    }

    /// Number of tensor modes (d).
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Mode size `n_j` for `axis`. Throws `OutOfDomain` if out of range.
    #[wasm_bindgen(js_name = "nJ")]
    pub fn n_j(&self, axis: usize) -> Result<usize, JsValue> {
        if axis >= self.inner.ndim() {
            return Err(make_js_error("OutOfDomain", "TtState.nJ: axis out of range"));
        }
        Ok(self.inner.n_j(axis))
    }

    /// Peak bond rank (curse-escape diagnostic: poly-in-d).
    #[wasm_bindgen(js_name = "peakRank")]
    #[must_use]
    pub fn peak_rank(&self) -> usize {
        self.inner.peak_rank()
    }

    /// Total stored scalar count (O(d·n·r²) storage).
    #[wasm_bindgen(js_name = "storageSize")]
    #[must_use]
    pub fn storage_size(&self) -> usize {
        self.inner.storage_size()
    }

    /// Scalar projection `⟨f, u⟩` for a separable functional.
    ///
    /// `data` / `offsets` — same ragged convention as the constructor.
    /// `n_axes` must equal `ndim()`; each functional length must equal `n_j(axis)`.
    #[wasm_bindgen(js_name = "innerSeparable")]
    pub fn inner_separable(
        &self,
        data: &js_sys::Float64Array,
        offsets: &js_sys::Uint32Array,
    ) -> Result<f64, JsValue> {
        let off = decode_offsets(offsets)?;
        let n_axes = off.len() - 1;
        if n_axes != self.inner.ndim() {
            return Err(make_js_error(
                "GridMismatch",
                "TtState.innerSeparable: n_axes must equal ndim",
            ));
        }
        let funcs = extract_slices(data, &off, n_axes)?;
        for (j, fj) in funcs.iter().enumerate() {
            if fj.len() != self.inner.n_j(j) {
                return Err(make_js_error(
                    "GridMismatch",
                    "TtState.innerSeparable: functional length must equal n_j(axis)",
                ));
            }
        }
        Ok(self.inner.inner_separable(&funcs))
    }
}

// ---------------------------------------------------------------------------
// TtEvolver JS class
// ---------------------------------------------------------------------------

/// Separable diagonal-A TT-Chernoff evolver (v9.0.0-beta, math §52).
///
/// Advances a `TtState` in-place via `evolve(state, t_final, n_steps)`.
/// The curse-escape guarantee: d-dimensional state stored as O(d·n·r²).
///
/// ## Error model (`.kind`)
///
/// - `"GridMismatch"` — `n_axes == 0` or array length mismatch.
/// - `"NanInf"`       — non-finite coefficients or domain bounds.
/// - `"OutOfDomain"`  — `n_steps == 0`, `t_final < 0`, or ndim mismatch.
#[wasm_bindgen]
pub struct TtEvolver {
    inner: TtChernoff<f64>,
}

#[wasm_bindgen]
impl TtEvolver {
    /// Construct a separable `TtEvolver`.
    ///
    /// `a`        — `Float64Array`: per-axis diffusion (length `n_axes`).
    /// `b`        — `Float64Array`: per-axis drift (length `n_axes`).
    /// `c`        — scalar reaction coefficient.
    /// `domMin`   — `Float64Array`: per-axis `x_min` (length `n_axes`).
    /// `domMax`   — `Float64Array`: per-axis `x_max` (length `n_axes`).
    /// `epsRound` — TT-rounding tolerance (>= 0, finite).
    #[wasm_bindgen(constructor)]
    pub fn new(
        a: &js_sys::Float64Array,
        b: &js_sys::Float64Array,
        c: f64,
        dom_min: &js_sys::Float64Array,
        dom_max: &js_sys::Float64Array,
        eps_round: f64,
    ) -> Result<TtEvolver, JsValue> {
        let n = a.length() as usize;
        if n == 0 {
            return Err(make_js_error("GridMismatch", "TtEvolver: n_axes must be >= 1"));
        }
        if b.length() as usize != n
            || dom_min.length() as usize != n
            || dom_max.length() as usize != n
        {
            return Err(make_js_error(
                "GridMismatch",
                "TtEvolver: a, b, domMin, domMax must all have the same length",
            ));
        }
        if !c.is_finite() || !eps_round.is_finite() {
            return Err(make_js_error("NanInf", "TtEvolver: c and epsRound must be finite"));
        }
        let ev = build_tt_evolver_wasm(a, b, c, dom_min, dom_max, n, eps_round)?;
        Ok(TtEvolver { inner: ev })
    }

    /// Evolve `state` in-place for time `t_final` with `n_steps` Chernoff steps.
    ///
    /// `state` is mutated behind its JS handle; chain calls to continue.
    pub fn evolve(
        &self,
        state: &mut TtState,
        t_final: f64,
        n_steps: usize,
    ) -> Result<(), JsValue> {
        if n_steps == 0 {
            return Err(make_js_error("OutOfDomain", "TtEvolver.evolve: n_steps must be >= 1"));
        }
        if !t_final.is_finite() || t_final < 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "TtEvolver.evolve: t_final must be finite and >= 0",
            ));
        }
        if self.inner.ndim() != state.inner.ndim() {
            return Err(make_js_error(
                "OutOfDomain",
                "TtEvolver.evolve: evolver ndim must match state ndim",
            ));
        }
        self.inner.evolve(t_final, n_steps, &mut state.inner);
        Ok(())
    }

    /// Return the number of tensor modes this evolver was built for.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }
}

// ---------------------------------------------------------------------------
// Crate-internal accessor (used by TtCoupledEvolver)
// ---------------------------------------------------------------------------

impl TtState {
    /// Mutable access to the core `TtState<f64>` (crate-internal).
    pub(crate) fn inner_mut(&mut self) -> &mut CoreTtState<f64> {
        &mut self.inner
    }
}

// ---------------------------------------------------------------------------
// Private build helper
// ---------------------------------------------------------------------------

fn build_tt_evolver_wasm(
    a: &js_sys::Float64Array,
    b: &js_sys::Float64Array,
    c: f64,
    dom_min: &js_sys::Float64Array,
    dom_max: &js_sys::Float64Array,
    n: usize,
    eps_round: f64,
) -> Result<TtChernoff<f64>, JsValue> {
    let mut a_v = vec![0.0f64; n];
    let mut b_v = vec![0.0f64; n];
    let mut lo_v = vec![0.0f64; n];
    let mut hi_v = vec![0.0f64; n];
    a.copy_to(&mut a_v);
    b.copy_to(&mut b_v);
    dom_min.copy_to(&mut lo_v);
    dom_max.copy_to(&mut hi_v);
    for &v in a_v.iter().chain(b_v.iter()).chain(lo_v.iter()).chain(hi_v.iter()) {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", "TtEvolver: coefficient or domain is NaN/Inf"));
        }
    }
    for &ai in &a_v {
        if ai < 0.0 {
            return Err(make_js_error("NanInf", "TtEvolver: diffusion a_j must be >= 0"));
        }
    }
    let domain: Vec<(f64, f64)> = lo_v
        .iter()
        .zip(hi_v.iter())
        .map(|(&lo, &hi)| {
            if lo >= hi {
                Err(make_js_error("NanInf", "TtEvolver: dom_min must be < dom_max"))
            } else {
                Ok((lo, hi))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TtChernoff::new(a_v, b_v, c, domain, eps_round))
}
