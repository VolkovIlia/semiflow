//! v8.1.0 WASM binding for `AdjointFokkerPlanckChernoff` (C2, ADR-0138, ADR-0107 Amdt 1).
//!
//! Implements `AdjointFokkerPlanckV8` — a stateless-per-call JS class that applies
//! adjoint Fokker-Planck Chernoff steps on M(ℝ) via two `Float64Array` buffers
//! (positions, weights).
//!
//! ## NARROW scope (§38.3, ADR-0107 AMENDMENT 1 NORMATIVE)
//!
//! Adjoint (weak-*) Fokker-Planck on M(ℝ). D=1 constant-coefficient 4-Dirac
//! pushforward (Lemma A.1, §38.3). Dirac count grows ×4 per step. Forward
//! kernel = `DiffusionChernoff` (Brownian benchmark).
//!
//! ## ABI-safety invariant (ADR-0138 hard constraint)
//!
//! `MeasureState`<f64,1> never crosses the WASM boundary. The caller passes and
//! receives two `Float64Array`s (positions, weights). The JS return value is a
//! plain object `{ positions: Float64Array, weights: Float64Array }`.
//!
//! Profile: `[profile.release]` (`panic = "abort"`) per ADR-0028 Amendment 1.
//! All error paths return `Err(JsValue)`.
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util with semiflow-ffi/py.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_arguments
)]

use js_sys::{Object, Reflect};
use semiflow::{
    AdjointFokkerPlanckChernoff, ChernoffFunction, DiffusionChernoff, Grid1D, MeasureState,
    ScratchPool, State,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// AdjointFokkerPlanckV8 WASM class
// ---------------------------------------------------------------------------

/// v8.1.0 Adjoint Fokker-Planck Chernoff on M(ℝ) — flat two-buffer interface (ADR-0138).
///
/// Applies adjoint (weak-*) Fokker-Planck Chernoff steps: each Dirac `δ_x`
/// is pushed to four children (Lemma A.1, §38.3):
///
///   S*(τ) `δ_x` = ¼δ_{x+h} + ¼δ_{x-h} + ½δ_{x+k} + `τc·δ_x`
///
/// where ``h = 2√(aτ)`` and ``k = 2bτ``. The measure is passed as two parallel
/// `Float64Array`s; `MeasureState` never crosses the boundary.
///
/// **NARROW scope**: D=1 constant-coefficient 4-Dirac pushforward (§38.3).
/// Dirac count grows ×4 per step. Forward kernel = `DiffusionChernoff` (Brownian).
///
/// ## JS Example
///
/// ```js
/// import init, { AdjointFokkerPlanckV8 } from "@semiflow/wasm";
/// await init();
/// const adj = new AdjointFokkerPlanckV8(0.5, 0.0, 0.0);
/// const positions = new Float64Array([0.0]);
/// const weights   = new Float64Array([1.0]);
/// const result = adj.step(0.1, positions, weights, 1);
/// // result.positions: Float64Array of length 4
/// // result.weights:   Float64Array of length 4
/// ```
///
/// ## Error model (`.kind` discriminator)
///
/// - `"OutOfDomain"` — `tau <= 0`, non-finite, `nSteps == 0`.
/// - `"GridMismatch"` — `positions.length != weights.length`.
#[wasm_bindgen]
pub struct AdjointFokkerPlanckV8 {
    a: f64,
    b: f64,
    c: f64,
}

#[wasm_bindgen]
impl AdjointFokkerPlanckV8 {
    /// Construct an adjoint Fokker-Planck handle.
    ///
    /// ## Parameters
    /// - `a` — diffusion coefficient (`h = 2√(aτ)`). Must be finite.
    /// - `b` — drift coefficient (`k = 2bτ`). Must be finite.
    /// - `c` — reaction coefficient (mass factor `1 + τc`). Must be finite.
    #[wasm_bindgen(constructor)]
    pub fn new(a: f64, b: f64, c: f64) -> Result<AdjointFokkerPlanckV8, JsValue> {
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return Err(make_js_error("OutOfDomain", "a, b, c must be finite"));
        }
        Ok(AdjointFokkerPlanckV8 { a, b, c })
    }

    /// Apply `nSteps` adjoint Fokker-Planck steps; return `{ positions, weights }`.
    ///
    /// `MeasureState` never crosses the boundary (ADR-0138).
    ///
    /// ## Parameters
    /// - `tau`       — step size (`> 0`, finite).
    /// - `positions` — `Float64Array` of input Dirac positions.
    /// - `weights`   — `Float64Array` of input Dirac weights.
    /// - `nSteps`    — number of steps (`>= 1`).
    ///
    /// ## Returns
    /// JS object `{ positions: Float64Array, weights: Float64Array }`.
    ///
    /// ## Errors
    /// - `.kind = "OutOfDomain"` — `tau <= 0`, non-finite, or `nSteps == 0`.
    /// - `.kind = "GridMismatch"` — `positions.length != weights.length`.
    #[wasm_bindgen(js_name = "step")]
    pub fn step(
        &self,
        tau: f64,
        positions: &js_sys::Float64Array,
        weights: &js_sys::Float64Array,
        n_steps: usize,
    ) -> Result<JsValue, JsValue> {
        if !tau.is_finite() || tau <= 0.0 {
            return Err(make_js_error("OutOfDomain", "tau must be finite and > 0"));
        }
        if n_steps == 0 {
            return Err(make_js_error("OutOfDomain", "nSteps must be >= 1"));
        }
        let n_in = positions.length() as usize;
        if weights.length() as usize != n_in {
            return Err(make_js_error(
                "GridMismatch",
                "positions.length must equal weights.length",
            ));
        }
        let mut pos_buf = vec![0.0f64; n_in];
        let mut wts_buf = vec![0.0f64; n_in];
        positions.copy_to(&mut pos_buf);
        weights.copy_to(&mut wts_buf);
        let (out_pos, out_wts) =
            run_steps(self.a, self.b, self.c, tau, &pos_buf, &wts_buf, n_steps)
                .map_err(|e| err_to_js(&e))?;
        let n_out = out_pos.len();
        let js_pos = js_sys::Float64Array::new_with_length(n_out as u32);
        let js_wts = js_sys::Float64Array::new_with_length(n_out as u32);
        js_pos.copy_from(&out_pos);
        js_wts.copy_from(&out_wts);
        let obj = Object::new();
        Reflect::set(&obj, &"positions".into(), &js_pos).unwrap();
        Reflect::set(&obj, &"weights".into(), &js_wts).unwrap();
        Ok(obj.into())
    }

    /// Return total variation ``‖ρ‖_TV = Σ|w_i|``.
    #[wasm_bindgen(js_name = "totalVariation")]
    #[must_use]
    pub fn total_variation(&self, weights: &js_sys::Float64Array) -> f64 {
        let mut wts_buf = vec![0.0f64; weights.length() as usize];
        weights.copy_to(&mut wts_buf);
        wts_buf.iter().map(|w| w.abs()).sum()
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust multi-step push
// ---------------------------------------------------------------------------

fn run_steps(
    a: f64,
    b: f64,
    c: f64,
    tau: f64,
    pos_in: &[f64],
    wts_in: &[f64],
    n_steps: usize,
) -> Result<(Vec<f64>, Vec<f64>), semiflow::SemiflowError> {
    let grid = Grid1D::new(-4.0_f64, 4.0, 32)?;
    let fwd = DiffusionChernoff::new_const_a(a, a, grid);
    let kernel = AdjointFokkerPlanckChernoff::new(fwd, a, b, c);
    let mut rho = build_measure(pos_in, wts_in);
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        let mut rho_next = MeasureState::<f64, 1>::dirac([0.0_f64], 0.0);
        kernel.apply_into(tau, &rho, &mut rho_next, &mut pool)?;
        rho = rho_next;
    }
    Ok(rho.to_flat_buffers_d1())
}

// ---------------------------------------------------------------------------
// Builder helper
// ---------------------------------------------------------------------------

fn build_measure(positions: &[f64], weights: &[f64]) -> MeasureState<f64, 1> {
    let mut m = MeasureState::<f64, 1>::dirac([0.0_f64], 0.0);
    m.zero_into();
    for (&p, &w) in positions.iter().zip(weights.iter()) {
        let atom = MeasureState::<f64, 1>::dirac([p], w);
        m.axpy_into(1.0, &atom);
    }
    m
}
