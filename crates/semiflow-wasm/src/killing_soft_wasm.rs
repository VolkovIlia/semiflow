//! `Killing2nd1D` — order-2 soft-killing Feynman-Kac JS class (`full` feature).
//!
//! | JS class       | Core type                                    | Python mirror   |
//! |----------------|----------------------------------------------|-----------------|
//! | `Killing2nd1D` | `Killing2ndChernoff<DiffusionChernoff, ...>` | `Killing2nd1D`  |
//!
//! Constant `κ ≥ 0`; palindromic Strang `e^{-τκ/2} C(τ) e^{-τκ/2}`; order 2.
//! Error model: same `.kind`-tagged JS `Error` as `Heat1D`.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

use js_sys::Float64Array;
use semiflow::{
    diffusion::DiffusionChernoff,
    grid::Grid1D,
    grid_fn::GridFn1D,
    killing_soft::{Killing2ndChernoff, KillingRate},
    ChernoffSemigroup,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Constant killing rate — Clone + Copy, no closure
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ConstKappaWasm(f64);

impl KillingRate<f64> for ConstKappaWasm {
    fn kappa(&self, _x: f64) -> f64 {
        self.0
    }
}

type DiffUnit = DiffusionChernoff<f64>;
type Killing2ndUnit = Killing2ndChernoff<DiffUnit, ConstKappaWasm, f64>;

// ---------------------------------------------------------------------------
// Static fn-pointers
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_k2_wasm(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_k2_wasm(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Helpers
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
// Killing2nd1D JS class
// ---------------------------------------------------------------------------

/// Order-2 soft-killing 1-D Feynman-Kac Chernoff (ADR-0126, §21.8).
///
/// Solves `∂_t u = ∂²u − κ·u` for constant `κ ≥ 0`.
/// Palindromic Strang `e^{−τκ/2} C(τ) e^{−τκ/2}`; order 2.
///
/// ## Constructor
/// `new(xmin, xmax, n, u0, kappa)`
///
/// ## Methods
/// `evolve(t, n_steps)`, `values()`, `order()`, `len()`
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct Killing2nd1D {
    sg: ChernoffSemigroup<Killing2ndUnit, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Killing2nd1D {
    /// Create a new `Killing2nd1D` state.
    ///
    /// - `xmin`, `xmax` — domain endpoints.
    /// - `n` — grid nodes (>= 4).
    /// - `u0` — `Float64Array` of length `n`.
    /// - `kappa` — constant killing rate (must be >= 0).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` if parameters are invalid.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        kappa: f64,
    ) -> Result<Killing2nd1D, JsValue> {
        if !kappa.is_finite() || kappa < 0.0 {
            return Err(make_js_error(
                "OutOfDomain",
                "kappa must be finite and >= 0",
            ));
        }
        let buf = extract_u0(u0, n)?;
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let diff = DiffusionChernoff::new(unit_a_k2_wasm, zero_k2_wasm, zero_k2_wasm, 1.0, grid);
        let rate = ConstKappaWasm(kappa);
        let kernel = Killing2ndUnit::new(diff, rate, grid).map_err(|e| err_to_js(&e))?;
        let sg = ChernoffSemigroup::new(kernel, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Killing2nd1D { sg, current })
    }

    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve(t, n_steps)?;
        let func = self.sg.func.clone();
        let sg = ChernoffSemigroup::new(func, n_steps).map_err(|e| err_to_js(&e))?;
        self.current = sg.evolve(t, &self.current).map_err(|e| err_to_js(&e))?;
        self.sg = sg;
        Ok(())
    }

    /// Return current grid values as `Float64Array` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        fn_to_js(&self.current)
    }

    /// Approximation order (always 2 — palindromic Strang).
    #[must_use]
    pub fn order(&self) -> u32 {
        2
    }

    /// Number of grid nodes.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}
