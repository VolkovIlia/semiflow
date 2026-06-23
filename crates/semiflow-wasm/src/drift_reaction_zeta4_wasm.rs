//! `DriftReaction4th1D` — order-4 palindromic Strang drift-reaction JS class (`full` feature).
//!
//! Mirrors `semiflow-py`'s `DriftReaction4th1D` and `semiflow-ffi`'s
//! `smf_drift_reaction_zeta4_*`.
//!
//! Default coefficients: `b(x) = 0.5`, `b'(x) = 0.0`, `c(x) = 0.0`.
//! Variable-coefficient closure API deferred (separate architect task).

use js_sys::Float64Array;
use semiflow_core::{
    ChernoffSemigroup, Diffusion4thChernoff, DriftReactionZeta4Chernoff, Grid1D, GridFn1D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Static coefficient fn-pointers
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_dr4_wasm(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_dr4_wasm(_: f64) -> f64 {
    0.0
}
extern "Rust" fn half_b_dr4_wasm(_: f64) -> f64 {
    0.5
}

// ---------------------------------------------------------------------------
// Helpers (mirrors diffusion_hi_wasm.rs)
// ---------------------------------------------------------------------------

fn extract_u0(u0: &Float64Array, n: usize) -> Result<Vec<f64>, JsValue> {
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

fn validate_evolve(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn fn_to_js(f: &GridFn1D<f64>) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(f.values.len() as u32);
    arr.copy_from(&f.values);
    arr
}

// ---------------------------------------------------------------------------
// DriftReaction4th1D
// ---------------------------------------------------------------------------

/// 1-D drift-reaction with order-4 palindromic Strang kernel (ADR-0127).
///
/// Solves `∂_t u = 0.5·∂_x u + 0.0·u` (default; variable-coefficient closure deferred).
/// Unit diffusion backbone `a = 1`.
///
/// Same JS API as `Heat1D4th`: `new(xmin, xmax, n, u0)`, `evolve(t, n_steps)`,
/// `values()`, `len()`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct DriftReaction4th1D {
    sg: ChernoffSemigroup<DriftReactionZeta4Chernoff, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl DriftReaction4th1D {
    /// Create a new `DriftReaction4th1D` state.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — domain bounds (`xmin < xmax`, both finite).
    /// - `n` — grid nodes (≥ 4).
    /// - `u0` — `Float64Array` of length `n`, all finite.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
    ) -> Result<DriftReaction4th1D, JsValue> {
        let buf = extract_u0(u0, n)?;
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let d4 = Diffusion4thChernoff::new(
            unit_a_dr4_wasm,
            zero_dr4_wasm,
            zero_dr4_wasm,
            1.0,
            grid,
        );
        let kernel = DriftReactionZeta4Chernoff::new(
            d4,
            half_b_dr4_wasm,
            zero_dr4_wasm,
            zero_dr4_wasm,
            0.5,
            grid,
        );
        let sg = ChernoffSemigroup::new(kernel, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(DriftReaction4th1D { sg, current })
    }

    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve(t, n_steps)?;
        let func = self.sg.func.clone();
        let sg = ChernoffSemigroup::new(func, n_steps).map_err(|e| err_to_js(&e))?;
        self.current = sg.evolve(t, &self.current).map_err(|e| err_to_js(&e))?;
        self.sg = sg;
        Ok(())
    }

    /// Return current grid values as a new `Float64Array` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        fn_to_js(&self.current)
    }

    /// Number of grid nodes.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}
