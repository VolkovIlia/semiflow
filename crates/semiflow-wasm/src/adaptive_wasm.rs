//! Round-9 adaptive-step engine for WebAssembly (`full` feature).
//!
//! | JS class     | Core type                                  | Python mirror |
//! |--------------|--------------------------------------------|---------------|
//! | `AdaptivePI1D` | `AdaptivePI<C, f64, ClassicalPI<f64>>` | `AdaptivePI`  |
//!
//! ## Design
//!
//! Mirrors `semiflow-py` `AdaptivePI` (`adaptive.rs`), 5 kernel variants.
//! `kernel` string selector: `"heat2"` (default), `"heat4"`, `"heat6"`,
//! `"drift"`, `"shift"`. The Python `evolve(t)` takes one argument; mirrored
//! exactly (no `n_steps`).
//!
//! The `AdaptivePI::evolve_adaptive` method takes `&mut self`, so the WASM
//! class stores it as mutable state. Returns the final `Float64Array` (copy)
//! plus diagnostics via `steps_accepted()` and `steps_rejected()` on the last call.
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow::{
    AdaptivePI, Diffusion4thChernoff, Diffusion6thChernoff, DiffusionChernoff,
    DriftReactionChernoff, Grid1D, GridFn1D, ShiftChernoff1D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Coefficient fn-pointers (unit / zero)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_adpi(_: f64) -> f64 { 1.0 }
extern "Rust" fn zero_adpi(_: f64) -> f64 { 0.0 }

// ---------------------------------------------------------------------------
// 5-kernel enum (avoids Box<dyn ChernoffFunction>)
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
enum AdpiKernel {
    Diff2(AdaptivePI<DiffusionChernoff<f64>>),
    Diff4(AdaptivePI<Diffusion4thChernoff<f64>>),
    Diff6(AdaptivePI<Diffusion6thChernoff<f64>>),
    DriftReaction(AdaptivePI<DriftReactionChernoff<f64>>),
    Shift(AdaptivePI<ShiftChernoff1D<f64>>),
}

impl AdpiKernel {
    fn set_tolerance(&mut self, abs: f64, rel: f64) {
        match self {
            Self::Diff2(k) => { k.tol_abs = abs; k.tol_rel = rel; }
            Self::Diff4(k) => { k.tol_abs = abs; k.tol_rel = rel; }
            Self::Diff6(k) => { k.tol_abs = abs; k.tol_rel = rel; }
            Self::DriftReaction(k) => { k.tol_abs = abs; k.tol_rel = rel; }
            Self::Shift(k) => { k.tol_abs = abs; k.tol_rel = rel; }
        }
    }

    fn evolve_adaptive(
        &mut self,
        t: f64,
        u0: &GridFn1D<f64>,
    ) -> Result<(Vec<f64>, usize, usize), semiflow::SemiflowError> {
        let outcome = match self {
            Self::Diff2(k) => k.evolve_adaptive(t, u0)?,
            Self::Diff4(k) => k.evolve_adaptive(t, u0)?,
            Self::Diff6(k) => k.evolve_adaptive(t, u0)?,
            Self::DriftReaction(k) => k.evolve_adaptive(t, u0)?,
            Self::Shift(k) => k.evolve_adaptive(t, u0)?,
        };
        Ok((outcome.final_state.values, outcome.steps_accepted, outcome.steps_rejected))
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn extract_u0_adpi(u0: &Float64Array, n: usize) -> Result<Vec<f64>, JsValue> {
    if u0.length() as usize != n {
        return Err(make_js_error("GridMismatch", "u0.length() must equal n"));
    }
    let mut buf = vec![0.0f64; n];
    u0.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", "u0 contains NaN or Inf"));
        }
    }
    Ok(buf)
}

fn fn_to_js_adpi(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

// ---------------------------------------------------------------------------
// Kernel builders (each ≤50 lines)
// ---------------------------------------------------------------------------

fn build_adpi_kernel(
    xmin: f64,
    xmax: f64,
    n: usize,
    kernel: &str,
) -> Result<AdpiKernel, JsValue> {
    let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
    match kernel {
        "heat2" => {
            let inner = DiffusionChernoff::new(unit_a_adpi, zero_adpi, zero_adpi, 1.0, grid);
            Ok(AdpiKernel::Diff2(AdaptivePI::new(inner)))
        }
        "heat4" => {
            let inner = Diffusion4thChernoff::new(unit_a_adpi, zero_adpi, zero_adpi, 1.0, grid);
            Ok(AdpiKernel::Diff4(AdaptivePI::new(inner)))
        }
        "heat6" => {
            let inner = Diffusion6thChernoff::new(unit_a_adpi, zero_adpi, zero_adpi, 1.0, grid);
            Ok(AdpiKernel::Diff6(AdaptivePI::new(inner)))
        }
        "drift" => {
            let inner = DriftReactionChernoff::with_closure(|_| 0.5_f64, |_| 0.0, 0.0, grid);
            Ok(AdpiKernel::DriftReaction(AdaptivePI::new(inner)))
        }
        "shift" => {
            let inner = ShiftChernoff1D::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
            Ok(AdpiKernel::Shift(AdaptivePI::new(inner)))
        }
        other => Err(make_js_error(
            "OutOfDomain",
            &format!("unknown kernel '{other}'; expected heat2|heat4|heat6|drift|shift"),
        )),
    }
}

// ---------------------------------------------------------------------------
// AdaptivePI1D — JS class
// ---------------------------------------------------------------------------

/// PI-controller adaptive-step integrator for any supported 1-D kernel.
///
/// Mirrors `AdaptivePI` (Python, `adaptive.rs`).  NOT a fixed-step Chernoff
/// product — substep sizes are chosen automatically to meet
/// `tol_abs + tol_rel * ‖u‖` (ADR-0044, math.md §11.1.bis).
///
/// Kernel selector: `"heat2"` (default), `"heat4"`, `"heat6"`, `"drift"`,
/// `"shift"`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct AdaptivePI1D {
    integrator: AdpiKernel,
    current: GridFn1D<f64>,
    last_accepted: usize,
    last_rejected: usize,
}

#[wasm_bindgen]
impl AdaptivePI1D {
    /// Construct `AdaptivePI1D`.
    ///
    /// - `xmin`, `xmax` — domain bounds (finite, `xmin < xmax`).
    /// - `n` — grid nodes (≥ 4).
    /// - `u0` — `Float64Array` of length `n`, all finite.
    /// - `kernel` — `"heat2"` (default), `"heat4"`, `"heat6"`, `"drift"`, `"shift"`.
    /// - `tol_abs` — absolute tolerance (default `1e-6`).
    /// - `tol_rel` — relative tolerance (default `1e-4`).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        kernel: &str,
        tol_abs: f64,
        tol_rel: f64,
    ) -> Result<AdaptivePI1D, JsValue> {
        if !tol_abs.is_finite() || tol_abs <= 0.0 {
            return Err(make_js_error("OutOfDomain", "tol_abs must be finite and > 0"));
        }
        if !tol_rel.is_finite() || tol_rel <= 0.0 {
            return Err(make_js_error("OutOfDomain", "tol_rel must be finite and > 0"));
        }
        let buf = extract_u0_adpi(u0, n)?;
        let mut iv = build_adpi_kernel(xmin, xmax, n, kernel)?;
        iv.set_tolerance(tol_abs, tol_rel);
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(AdaptivePI1D { integrator: iv, current, last_accepted: 0, last_rejected: 0 })
    }

    /// Evolve by time `t` using adaptive PI substeps.
    ///
    /// Returns updated `Float64Array` of length `n` (copy).
    /// Substep diagnostics available via `stepsAccepted()` / `stepsRejected()`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64) -> Result<Float64Array, JsValue> {
        if !t.is_finite() || t <= 0.0 {
            return Err(make_js_error("OutOfDomain", "t must be finite and > 0"));
        }
        let input = self.current.clone();
        let (vals, accepted, rejected) = self.integrator
            .evolve_adaptive(t, &input)
            .map_err(|e| err_to_js(&e))?;
        let grid = self.current.grid;
        let gfn = GridFn1D::new(grid, vals.clone()).map_err(|e| err_to_js(&e))?;
        self.current = gfn;
        self.last_accepted = accepted;
        self.last_rejected = rejected;
        Ok(fn_to_js_adpi(&vals))
    }

    /// Return current grid values as `Float64Array` of length `n` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        fn_to_js_adpi(&self.current.values)
    }

    /// Number of accepted substeps in the most recent `evolve` call.
    #[must_use]
    pub fn steps_accepted(&self) -> usize {
        self.last_accepted
    }

    /// Number of rejected substeps in the most recent `evolve` call.
    #[must_use]
    pub fn steps_rejected(&self) -> usize {
        self.last_rejected
    }

    /// Number of grid nodes.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}
