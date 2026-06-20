//! Higher-order 1-D diffusion engines for WebAssembly (`full` feature).
//!
//! Exposes five JS classes mirroring the Python binding:
//!
//! | JS class       | Core type                    | Temporal order |
//! |----------------|------------------------------|----------------|
//! | `Heat1D4th`    | `Diffusion4thChernoff`       | 4              |
//! | `Heat1D6th`    | `Diffusion6thChernoff`       | 6              |
//! | `Heat1DZeta4`  | `Diffusion4thZeta4Chernoff`  | 4 (ζ⁴)        |
//! | `Heat1DZeta6`  | `Diffusion6thZeta6Chernoff`  | 6 (ζ⁶)        |
//! | `Heat1DZeta8`  | `Diffusion8thZeta8Chernoff`  | 8 (ζ⁸)        |
//!
//! All classes share the same JS API as [`crate::state::Heat1D`]:
//! `new(xmin, xmax, n, u0)`, `evolve(t, n_steps)`, `values()`, `len()`.
//!
//! Error model: same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`; validate before calling core.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow_core::{
    ChernoffSemigroup, Diffusion4thChernoff, Diffusion4thZeta4Chernoff, Diffusion6thChernoff,
    Diffusion6thZeta6Chernoff, Diffusion8thZeta8Chernoff, Grid1D, GridFn1D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Unit coefficient statics
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_d(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Shared validation + construction helpers
// ---------------------------------------------------------------------------

/// Copy `Float64Array` to `Vec<f64>` and validate length and finiteness.
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

/// Validate evolve parameters before calling core.
fn validate_evolve(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

/// Build a `Grid1D` (unit: reflect boundary, matching wasm lite default).
fn make_grid(xmin: f64, xmax: f64, n: usize) -> Result<Grid1D<f64>, JsValue> {
    Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))
}

/// Emit the current values of a `GridFn1D` as a JS `Float64Array`.
fn fn_to_js(f: &GridFn1D<f64>) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(f.values.len() as u32);
    arr.copy_from(&f.values);
    arr
}

// ---------------------------------------------------------------------------
// Heat1D4th
// ---------------------------------------------------------------------------

/// 1-D diffusion with 4th-order Chernoff kernel (unit `a = 1`).
///
/// Same JS API as `Heat1D` but backed by `Diffusion4thChernoff` (spatial order 4).
///
/// # Errors
/// Throws JS `Error` with `.kind` property — see crate-level docs.
#[wasm_bindgen]
pub struct Heat1D4th {
    sg: ChernoffSemigroup<Diffusion4thChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Heat1D4th {
    /// Create a new `Heat1D4th` state.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — domain bounds (`xmin < xmax`, both finite).
    /// - `n` — grid nodes (≥ 4).
    /// - `u0` — `Float64Array` of length `n`, all finite.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(xmin: f64, xmax: f64, n: usize, u0: &Float64Array) -> Result<Heat1D4th, JsValue> {
        let buf = extract_u0(u0, n)?;
        let grid = make_grid(xmin, xmax, n)?;
        let chernoff = Diffusion4thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
        let sg = ChernoffSemigroup::new(chernoff, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Heat1D4th { sg, current })
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

// ---------------------------------------------------------------------------
// Heat1D6th
// ---------------------------------------------------------------------------

/// 1-D diffusion with 6th-order Chernoff kernel (unit `a = 1`).
///
/// Same JS API as `Heat1D4th` but backed by `Diffusion6thChernoff` (spatial order 6).
#[wasm_bindgen]
pub struct Heat1D6th {
    sg: ChernoffSemigroup<Diffusion6thChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Heat1D6th {
    /// Create a new `Heat1D6th` state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(xmin: f64, xmax: f64, n: usize, u0: &Float64Array) -> Result<Heat1D6th, JsValue> {
        let buf = extract_u0(u0, n)?;
        let grid = make_grid(xmin, xmax, n)?;
        let chernoff = Diffusion6thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
        let sg = ChernoffSemigroup::new(chernoff, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Heat1D6th { sg, current })
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

// ---------------------------------------------------------------------------
// Heat1DZeta4
// ---------------------------------------------------------------------------

/// 1-D diffusion with ζ⁴ Chernoff kernel — temporal order 4.
///
/// Chain: `Diffusion4thChernoff → Diffusion4thZeta4Chernoff`.
#[wasm_bindgen]
pub struct Heat1DZeta4 {
    sg: ChernoffSemigroup<Diffusion4thZeta4Chernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Heat1DZeta4 {
    /// Create a new `Heat1DZeta4` state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(xmin: f64, xmax: f64, n: usize, u0: &Float64Array) -> Result<Heat1DZeta4, JsValue> {
        let buf = extract_u0(u0, n)?;
        let grid = make_grid(xmin, xmax, n)?;
        let d4 = Diffusion4thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
        let zeta4 = Diffusion4thZeta4Chernoff::new(d4, Some(1.0_f64)).map_err(|e| err_to_js(&e))?;
        let sg = ChernoffSemigroup::new(zeta4, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Heat1DZeta4 { sg, current })
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

// ---------------------------------------------------------------------------
// Heat1DZeta6
// ---------------------------------------------------------------------------

/// 1-D diffusion with ζ⁶ Chernoff kernel — temporal order 6.
///
/// Chain: `Diffusion4thChernoff → Diffusion4thZeta4Chernoff → Diffusion6thZeta6Chernoff`.
#[wasm_bindgen]
pub struct Heat1DZeta6 {
    sg: ChernoffSemigroup<Diffusion6thZeta6Chernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Heat1DZeta6 {
    /// Create a new `Heat1DZeta6` state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(xmin: f64, xmax: f64, n: usize, u0: &Float64Array) -> Result<Heat1DZeta6, JsValue> {
        let buf = extract_u0(u0, n)?;
        let grid = make_grid(xmin, xmax, n)?;
        let d4 = Diffusion4thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
        let z4 = Diffusion4thZeta4Chernoff::new(d4, Some(1.0_f64)).map_err(|e| err_to_js(&e))?;
        let zeta6 = Diffusion6thZeta6Chernoff::new(z4, Some(1.0_f64)).map_err(|e| err_to_js(&e))?;
        let sg = ChernoffSemigroup::new(zeta6, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Heat1DZeta6 { sg, current })
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

// ---------------------------------------------------------------------------
// Heat1DZeta8
// ---------------------------------------------------------------------------

/// 1-D diffusion with ζ⁸ Chernoff kernel — temporal order 8.
///
/// Chain: `D4 → Zeta4 → Zeta6 → Diffusion8thZeta8Chernoff`.
#[wasm_bindgen]
pub struct Heat1DZeta8 {
    sg: ChernoffSemigroup<Diffusion8thZeta8Chernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Heat1DZeta8 {
    /// Create a new `Heat1DZeta8` state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(xmin: f64, xmax: f64, n: usize, u0: &Float64Array) -> Result<Heat1DZeta8, JsValue> {
        let buf = extract_u0(u0, n)?;
        let grid = make_grid(xmin, xmax, n)?;
        let d4 = Diffusion4thChernoff::new(unit_a, zero_d, zero_d, 1.0, grid);
        let z4 = Diffusion4thZeta4Chernoff::new(d4, Some(1.0_f64)).map_err(|e| err_to_js(&e))?;
        let z6 = Diffusion6thZeta6Chernoff::new(z4, Some(1.0_f64)).map_err(|e| err_to_js(&e))?;
        let zeta8 = Diffusion8thZeta8Chernoff::new(z6, Some(1.0_f64)).map_err(|e| err_to_js(&e))?;
        let sg = ChernoffSemigroup::new(zeta8, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Heat1DZeta8 { sg, current })
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
