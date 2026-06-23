//! Round-2 1-D grid engines for WebAssembly (`full` feature).
//!
//! Exposes five JS classes mirroring the Python binding:
//!
//! | JS class            | Core type                                               | Python mirror        |
//! |---------------------|---------------------------------------------------------|----------------------|
//! | `TruncatedExp1D`    | `TruncatedExpDiffusionChernoff`                         | `TruncatedExp1D`     |
//! | `TruncatedExp4th1D` | `TruncatedExp4thDiffusionChernoff`                      | `TruncatedExp4th1D`  |
//! | `DriftReaction1D`   | `DriftReactionChernoff` (closure path, constant `b`,`c`)| `DriftReaction1D`    |
//! | `Shift1D`           | `ShiftChernoff1D` (closure path, constant `a`,`b`,`c`)  | `Shift1D`            |
//! | `Strang1D`          | `StrangSplit<DiffusionChernoff, DriftReactionChernoff>`  | `Strang1D`           |
//!
//! Error model: same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`; validate before calling core.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow::{
    ChernoffSemigroup, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, ShiftChernoff1D,
    StrangSplit, TruncatedExp4thDiffusionChernoff, TruncatedExpDiffusionChernoff,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Coefficient fn-pointers (unit / zero) — Copy-friendly kernels
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_ex(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_ex(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Shared helpers (mirrors diffusion_hi_wasm)
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

fn make_grid(xmin: f64, xmax: f64, n: usize) -> Result<Grid1D<f64>, JsValue> {
    Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))
}

fn fn_to_js(f: &GridFn1D<f64>) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(f.values.len() as u32);
    arr.copy_from(&f.values);
    arr
}

// ---------------------------------------------------------------------------
// TruncatedExp1D
// ---------------------------------------------------------------------------

/// 1-D diffusion with K=4 truncated-exp Chernoff kernel (unit `a = 1`).
///
/// Mirrors `TruncatedExp1D` (Python). CFL condition `2·τ < dx²` is checked
/// on every `evolve` call and throws `kind = "CflViolated"` on violation.
#[wasm_bindgen]
pub struct TruncatedExp1D {
    sg: ChernoffSemigroup<TruncatedExpDiffusionChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl TruncatedExp1D {
    /// Create a new `TruncatedExp1D` state.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — domain bounds (`xmin < xmax`, both finite).
    /// - `n` — grid nodes (≥ 4).
    /// - `u0` — `Float64Array` of length `n`, all finite.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(xmin: f64, xmax: f64, n: usize, u0: &Float64Array) -> Result<TruncatedExp1D, JsValue> {
        let buf = extract_u0(u0, n)?;
        let grid = make_grid(xmin, xmax, n)?;
        let kernel = TruncatedExpDiffusionChernoff::new(unit_a_ex, zero_ex, zero_ex, 1.0, grid);
        let sg = ChernoffSemigroup::new(kernel, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(TruncatedExp1D { sg, current })
    }

    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve(t, n_steps)?;
        let func = self.sg.func;
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
// TruncatedExp4th1D
// ---------------------------------------------------------------------------

/// 1-D diffusion with 4th-order truncated-exp Chernoff kernel (unit `a = 1`).
///
/// Mirrors `TruncatedExp4th1D` (Python). CFL condition checked per evolve step.
#[wasm_bindgen]
pub struct TruncatedExp4th1D {
    sg: ChernoffSemigroup<TruncatedExp4thDiffusionChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl TruncatedExp4th1D {
    /// Create a new `TruncatedExp4th1D` state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
    ) -> Result<TruncatedExp4th1D, JsValue> {
        let buf = extract_u0(u0, n)?;
        let grid = make_grid(xmin, xmax, n)?;
        let kernel = TruncatedExp4thDiffusionChernoff::new(unit_a_ex, zero_ex, zero_ex, 1.0, grid);
        let sg = ChernoffSemigroup::new(kernel, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(TruncatedExp4th1D { sg, current })
    }

    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve(t, n_steps)?;
        let func = self.sg.func;
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
// DriftReaction1D
// ---------------------------------------------------------------------------

/// 1-D drift+reaction state (RK2 characteristic-flow Chernoff).
///
/// Solves `∂_t u = b·∂_x u + c·u` with constant scalar coefficients.
/// Mirrors `DriftReaction1D` (Python). Default values: `b = 0.5`, `c = 0.0`.
///
/// The closure path (`with_closure`) stores the scalar via `Arc` — clone is
/// cheap (Arc reference count increment).
#[wasm_bindgen]
pub struct DriftReaction1D {
    sg: ChernoffSemigroup<DriftReactionChernoff<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl DriftReaction1D {
    /// Create a new `DriftReaction1D` state.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — domain bounds.
    /// - `n` — grid nodes (≥ 4).
    /// - `u0` — initial condition `Float64Array` of length `n`.
    /// - `b` — constant drift coefficient (Python default: `0.5`).
    /// - `c` — constant reaction coefficient (Python default: `0.0`).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        b: f64,
        c: f64,
    ) -> Result<DriftReaction1D, JsValue> {
        if !b.is_finite() || !c.is_finite() {
            return Err(make_js_error("OutOfDomain", "b and c must be finite"));
        }
        let buf = extract_u0(u0, n)?;
        let grid = make_grid(xmin, xmax, n)?;
        let kernel = DriftReactionChernoff::with_closure(
            move |_| b,
            move |_| c,
            c.abs(),
            grid,
        );
        let sg = ChernoffSemigroup::new(kernel, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(DriftReaction1D { sg, current })
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
// Shift1D
// ---------------------------------------------------------------------------

/// 1-D CDR state via formula (6) — kernel `L = a(x)∂² + b(x)∂ + c(x)`.
///
/// Solves `∂_t u = a·∂²u + b·∂_x u + c·u` with constant scalar coefficients.
/// Mirrors `Shift1D` (Python). Default values: `a = 0.5`, `b = 0.0`, `c = 0.0`.
/// Requires `a > 0` (strict ellipticity).
#[wasm_bindgen]
pub struct Shift1D {
    sg: ChernoffSemigroup<ShiftChernoff1D<f64>, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Shift1D {
    /// Create a new `Shift1D` state.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — domain bounds.
    /// - `n` — grid nodes (≥ 4).
    /// - `u0` — initial condition `Float64Array` of length `n`.
    /// - `a` — diffusion coefficient (must be > 0; Python default: `0.5`).
    /// - `b` — drift coefficient (Python default: `0.0`).
    /// - `c` — reaction coefficient (Python default: `0.0`).
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
        a: f64,
        b: f64,
        c: f64,
    ) -> Result<Shift1D, JsValue> {
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return Err(make_js_error("OutOfDomain", "a, b, c must be finite"));
        }
        if a <= 0.0 {
            return Err(make_js_error("OutOfDomain", "a must be > 0 (strict ellipticity)"));
        }
        let buf = extract_u0(u0, n)?;
        let grid = make_grid(xmin, xmax, n)?;
        let norm = a.abs() + b.abs() + c.abs();
        let kernel = ShiftChernoff1D::with_closure(
            move |_| a,
            move |_| b,
            move |_| c,
            norm,
            grid,
        );
        let sg = ChernoffSemigroup::new(kernel, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Shift1D { sg, current })
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
// Strang1D
// ---------------------------------------------------------------------------

type StrangKernel = StrangSplit<DiffusionChernoff<f64>, DriftReactionChernoff<f64>>;

/// 1-D advection-diffusion via Strang splitting `D(τ/2) ∘ R(τ) ∘ D(τ/2)`.
///
/// Solves `∂_t u = ∂²u + b·∂_x u` (unit diffusion, constant drift `b`).
/// Mirrors `Strang1D` (Python). Default value: `b = 0.5`.
#[wasm_bindgen]
pub struct Strang1D {
    sg: ChernoffSemigroup<StrangKernel, GridFn1D<f64>>,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Strang1D {
    /// Create a new `Strang1D` state.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — domain bounds.
    /// - `n` — grid nodes (≥ 4).
    /// - `u0` — initial condition `Float64Array` of length `n`.
    /// - `b` — constant drift coefficient (Python default: `0.5`).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        b: f64,
    ) -> Result<Strang1D, JsValue> {
        if !b.is_finite() {
            return Err(make_js_error("OutOfDomain", "b must be finite"));
        }
        let buf = extract_u0(u0, n)?;
        let grid = make_grid(xmin, xmax, n)?;
        let diff = DiffusionChernoff::new(unit_a_ex, zero_ex, zero_ex, 1.0, grid);
        let drift = DriftReactionChernoff::with_closure(move |_| b, |_| 0.0, 0.0, grid);
        let split = StrangSplit::new(diff, drift);
        let sg = ChernoffSemigroup::new(split, 100).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(Strang1D { sg, current })
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
