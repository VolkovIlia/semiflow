//! Boundary-condition WASM — Part 2: `DirichletHeat2nd1D` (M11, §21.9, ADR-0176).
//!
//! Extracted from `bc_wasm.rs` to stay within the 500-line suckless limit.
//!
//! Error model: same `.kind`-tagged JS `Error` as `Heat1D` — see crate docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`; validate first.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow::{
    killing_order2::DirichletHeat2ndChernoff, reflection::HalfSpaceRegion, ChernoffSemigroup,
    DiffusionChernoff, Grid1D, GridFn1D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Local fn-pointer stubs (each wasm module is standalone)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_bc2(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_bc2(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Shared helpers (local copies — avoid cross-module private fn access)
// ---------------------------------------------------------------------------

fn extract_u0_d2(u0: &Float64Array, n: usize) -> Result<Vec<f64>, JsValue> {
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

fn validate_evolve_d2(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn fn_to_js_d2(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

type DiffUnit = DiffusionChernoff<f64>;
type DirichletHeat2ndKernel = DirichletHeat2ndChernoff<DiffUnit, HalfSpaceRegion<f64, 1>>;

// ===========================================================================
// DirichletHeat2nd1D
// ===========================================================================

/// 1-D heat with Dirichlet BC via the odd image method (M11, §21.9, ADR-0176).
///
/// Solves `∂_t u = ∂²u` with absorbing `u = 0` at `origin`.
/// Backed by `DirichletHeat2ndChernoff<DiffusionChernoff, HalfSpaceRegion>` (order 2).
///
/// This is the order-2 companion of `Killing1D` (order 1) and the
/// Dirichlet mirror of `Reflected1D` (Neumann, order 2).
///
/// ## Parameters
/// - `xmin`, `xmax` — domain bounds.
/// - `n`      — grid nodes (≥ 4).
/// - `u0`     — `Float64Array` of length `n`, all finite.
/// - `origin` — absorbing boundary point (default = `xmin`).
///
/// ## Note
/// The odd ghost subtracts mass; solution does NOT preserve non-negativity.
/// This is correct: an absorbing wall removes mass.
#[wasm_bindgen]
pub struct DirichletHeat2nd1D {
    sg: ChernoffSemigroup<DirichletHeat2ndKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl DirichletHeat2nd1D {
    /// Create a new `DirichletHeat2nd1D` state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        origin: f64,
    ) -> Result<DirichletHeat2nd1D, JsValue> {
        let buf = extract_u0_d2(u0, n)?;
        let origin_eff = if origin.is_finite() { origin } else { xmin };
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let diff = DiffusionChernoff::new(unit_a_bc2, zero_bc2, zero_bc2, 1.0, grid);
        let region =
            HalfSpaceRegion::<f64, 1>::new([origin_eff], [1.0]).map_err(|e| err_to_js(&e))?;
        let kernel = DirichletHeat2ndChernoff::new(diff, region).map_err(|e| err_to_js(&e))?;
        let sg = ChernoffSemigroup::new(kernel, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(DirichletHeat2nd1D { sg, current })
    }

    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve_d2(t, n_steps)?;
        let func = self.sg.func.clone();
        let sg = ChernoffSemigroup::new(func, n_steps).map_err(|e| err_to_js(&e))?;
        self.current = sg.evolve(t, &self.current).map_err(|e| err_to_js(&e))?;
        self.sg = sg;
        Ok(())
    }

    /// Return current grid values as a new `Float64Array` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        fn_to_js_d2(&self.current.values)
    }

    /// Number of grid nodes.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}
