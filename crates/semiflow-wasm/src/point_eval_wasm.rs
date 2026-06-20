//! WASM binding for `PointEval` — 1-D pointwise evaluation (`full` feature, Round 11).
//!
//! | JS class    | Core backend             | Python mirror |
//! |-------------|--------------------------|---------------|
//! | `PointEval` | `DiffusionChernoff<f64>` | `PointEval`   |
//!
//! Exposes Backend A (1-D variable-coefficient heat, ADR-0080, math §31.2).
//! `evalAt(tau, u0, x, n_steps)` returns the scalar
//! `(F(τ))^{n_steps} u0` evaluated at the single query point `x`.
//!
//! Byte-identity contract (Proposition 31.1): the returned scalar is
//! bit-identical to sampling the result of `n_steps` full `apply_into`
//! calls at `x`.
//!
//! ## Error model
//!
//! `.kind`-tagged JS `Error`. `panic = "abort"` (ADR-0028 Amendment 1).

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use semiflow_core::{
    diffusion::DiffusionChernoff,
    point_eval::PointEval as _,
    Grid1D, GridFn1D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// PointEval
// ---------------------------------------------------------------------------

/// Pointwise evaluation via `DiffusionChernoff<f64>` Backend A (ADR-0080).
///
/// For a 1-D diffusion kernel with unit diffusion coefficient (a ≡ 1),
/// `evalAt(tau, u0, x, n_steps)` returns the scalar
/// `(F(τ))^{n_steps} u0` sampled at `x`.
///
/// Mirrors Python `PointEval`.
///
/// ## Parameters (constructor)
/// - `xmin`, `xmax` — grid boundaries (finite, `xmin < xmax`).
/// - `n`            — number of grid nodes (`>= 4`).
///
/// # Errors
/// - `.kind = "GridMismatch"` — invalid grid parameters.
/// - `.kind = "GridMismatch"` — `u0.length != n` in `evalAt`.
/// - `.kind = "NanInf"` — `u0` contains NaN or Inf.
/// - `.kind = "OutOfDomain"` — `n_steps == 0`, `tau < 0`, or non-finite.
#[wasm_bindgen(js_name = "PointEval")]
pub struct PointEvalWasm {
    xmin: f64,
    xmax: f64,
    n: usize,
}

#[wasm_bindgen(js_class = "PointEval")]
impl PointEvalWasm {
    /// Construct a `PointEval` for a 1-D unit-diffusion grid.
    ///
    /// # Errors
    /// - `.kind = "GridMismatch"` — grid params invalid.
    #[wasm_bindgen(constructor)]
    pub fn new(xmin: f64, xmax: f64, n: usize) -> Result<PointEvalWasm, JsValue> {
        // Validate grid eagerly.
        Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        Ok(Self { xmin, xmax, n })
    }

    /// Evaluate `(F(τ))^{n_steps} u0` at point `x`.
    ///
    /// ## Parameters
    /// - `tau`     — Chernoff step size (`>= 0`, finite).
    /// - `u0`      — `Float64Array` of length `n`; initial condition.
    /// - `x`       — query point (any finite `f64`; clamped to `[xmin,xmax]`).
    /// - `n_steps` — number of Chernoff iterations (`>= 1`; default 1).
    ///
    /// Returns a `number` (scalar `f64`).
    ///
    /// # Errors
    /// See struct-level error table.
    #[wasm_bindgen(js_name = "evalAt")]
    pub fn eval_at(
        &self,
        tau: f64,
        u0: &[f64],
        x: f64,
        n_steps: u32,
    ) -> Result<f64, JsValue> {
        if n_steps == 0 {
            return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
        }
        if !tau.is_finite() || tau < 0.0 {
            return Err(make_js_error("OutOfDomain", "tau must be finite and >= 0"));
        }
        if u0.len() != self.n {
            return Err(make_js_error(
                "GridMismatch",
                "u0.length must equal n",
            ));
        }
        for &v in u0 {
            if !v.is_finite() {
                return Err(make_js_error("NanInf", "u0 contains NaN or Inf"));
            }
        }
        let grid = Grid1D::new(self.xmin, self.xmax, self.n)
            .map_err(|e| err_to_js(&e))?;
        let kernel = DiffusionChernoff::new(
            |_: f64| 1.0_f64,
            |_: f64| 0.0_f64,
            |_: f64| 0.0_f64,
            1.0_f64,
            grid,
        );
        let src = GridFn1D { values: u0.to_vec(), grid };
        kernel.eval_at(tau, &src, &[x], n_steps).map_err(|e| err_to_js(&e))
    }

    /// Grid node count `n`.
    #[must_use]
    pub fn n(&self) -> u32 {
        self.n as u32
    }

    /// Grid left boundary `xmin`.
    #[must_use]
    pub fn xmin(&self) -> f64 {
        self.xmin
    }

    /// Grid right boundary `xmax`.
    #[must_use]
    pub fn xmax(&self) -> f64 {
        self.xmax
    }
}
