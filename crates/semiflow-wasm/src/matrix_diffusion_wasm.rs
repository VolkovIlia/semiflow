//! Coupled 2-component 1D diffusion WASM binding (`full` feature).
//!
//! Exposes one JS class:
//!
//! | JS class            | Core type                        | Python mirror      |
//! |---------------------|----------------------------------|--------------------|
//! | `MatrixDiffusion1D` | `MatrixDiffusionChernoff<f64, 2>`| `MatrixDiffusion1D`|
//!
//! ## Layout
//!
//! State is a flat `Float64Array` of length `2*n` with component-inner layout:
//! index `k*2+i` holds component `i` at grid point `k` (row-major, same as Python).
//!
//! ## Physics
//!
//! Solves `∂_t u = a_diag · ∂_xx u + c_coupling · u` (M=2 coupled system,
//! M17, ADR-0082, math §33).  Palindromic Strang splitting; order 2.
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow::{
    matrix_system::{MatrixDiffusionChernoff, MatrixGridFn1D},
    ChernoffSemigroup, Grid1D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract flat `Float64Array` of length `2*n` into `Vec<f64>`, validate.
fn extract_u0(u0: &Float64Array, n: usize) -> Result<Vec<f64>, JsValue> {
    if u0.length() as usize != 2 * n {
        return Err(make_js_error(
            "GridMismatch",
            "u0.length() must equal 2*n (component-inner layout)",
        ));
    }
    let mut buf = vec![0.0f64; 2 * n];
    u0.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", "u0 contains NaN or Inf"));
        }
    }
    Ok(buf)
}

/// Validate time-step parameters: `t >= 0` and finite, `n_steps >= 1`.
fn validate_evolve_mat(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

/// Build the `MatrixDiffusionChernoff<f64, 2>` kernel from scalar parameters.
fn build_kernel(
    a_diag: f64,
    c_coupling: f64,
    grid: Grid1D<f64>,
) -> Result<MatrixDiffusionChernoff<f64, 2>, semiflow::SemiflowError> {
    let a_d = a_diag;
    let c_c = c_coupling;
    MatrixDiffusionChernoff::<f64, 2>::new(
        move |_x, mat| {
            mat[0][0] = a_d;
            mat[0][1] = 0.0;
            mat[1][0] = 0.0;
            mat[1][1] = a_d;
        },
        |_x, mat| {
            mat[0][0] = 0.0;
            mat[0][1] = 0.0;
            mat[1][0] = 0.0;
            mat[1][1] = 0.0;
        },
        move |_x, mat| {
            mat[0][0] = 0.0;
            mat[0][1] = c_c;
            mat[1][0] = c_c;
            mat[1][1] = 0.0;
        },
        grid,
    )
}

/// Evolve from `vals` (length `2*n`) and return updated flat `Vec<f64>`.
fn evolve_matrix(
    a_diag: f64,
    c_coupling: f64,
    grid: Grid1D<f64>,
    vals: &[f64],
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let kernel = build_kernel(a_diag, c_coupling, grid)?;
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let mut src = MatrixGridFn1D::<f64, 2>::new(grid);
    src.values.copy_from_slice(vals);
    let out = sg.evolve(t, &src)?;
    Ok(out.values)
}

// ---------------------------------------------------------------------------
// MatrixDiffusion1D
// ---------------------------------------------------------------------------

/// Coupled 2-component 1D diffusion state (M17, ADR-0082, math §33).
///
/// Solves `∂_t u = a_diag · ∂_xx u + c_coupling · u` for `u ∈ ℝ²`.
/// Palindromic Strang splitting; order 2.
///
/// ## Buffer layout
///
/// State is a flat `Float64Array` of length `2*n`.  Component `i` at grid
/// point `k` is at index `k*2+i` (row-major, component-inner, same as Python).
///
/// ## Parameters (constructor)
/// - `xmin`, `xmax` — domain bounds (finite, `xmin < xmax`).
/// - `n` — grid nodes (≥ 5, required by block-CN stencil per ADR-0082).
/// - `u0` — `Float64Array` of length `2*n`; all finite.
/// - `a_diag` — diagonal diffusion coefficient `a₀₀ = a₁₁` (default 1.0, must be > 0).
/// - `c_coupling` — off-diagonal reaction `c₀₁ = c₁₀` (default 0.0).
///
/// # Errors
/// Throws JS `Error` with `.kind` property — see crate-level docs.
#[wasm_bindgen]
pub struct MatrixDiffusion1D {
    grid: Grid1D<f64>,
    a_diag: f64,
    c_coupling: f64,
    /// Flat state vector; length = `2 * n`.
    vals: Vec<f64>,
    n: usize,
}

#[wasm_bindgen]
impl MatrixDiffusion1D {
    /// Construct `MatrixDiffusion1D`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        a_diag: f64,
        c_coupling: f64,
    ) -> Result<MatrixDiffusion1D, JsValue> {
        if n < 5 {
            return Err(make_js_error(
                "OutOfDomain",
                "n must be >= 5 (block-CN stencil)",
            ));
        }
        if !a_diag.is_finite() || a_diag <= 0.0 {
            return Err(make_js_error("OutOfDomain", "a_diag must be finite and > 0"));
        }
        let vals = extract_u0(u0, n)?;
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        // Validate kernel construction eagerly.
        build_kernel(a_diag, c_coupling, grid).map_err(|e| err_to_js(&e))?;
        Ok(MatrixDiffusion1D { grid, a_diag, c_coupling, vals, n })
    }

    /// Advance state by time `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve_mat(t, n_steps)?;
        let out = evolve_matrix(self.a_diag, self.c_coupling, self.grid, &self.vals, t, n_steps)
            .map_err(|e| err_to_js(&e))?;
        self.vals = out;
        Ok(())
    }

    /// Return current state as a flat `Float64Array` of length `2*n` (copy).
    ///
    /// Component `i` at grid point `k` is at index `k*2+i` (row-major).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        #[allow(clippy::cast_possible_truncation)]
        let arr = Float64Array::new_with_length(self.vals.len() as u32);
        arr.copy_from(&self.vals);
        arr
    }

    /// Return approximation order (always 2 — palindromic Strang).
    #[must_use]
    pub fn order(&self) -> u32 {
        2
    }

    /// Number of grid nodes `n`.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.n
    }
}
