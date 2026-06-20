//! Sparse-grid Smolyak Chernoff engine for WebAssembly (`full` feature).
//!
//! | JS class     | Core type                  | Python mirror |
//! |--------------|----------------------------|---------------|
//! | `SmolyakD6`  | `SmolyakGridND<f64, 6>`    | `SmolyakD6V8` |
//!
//! ## Narrow scope (mirrors Python ADR-0138)
//!
//! D = 6, unit diffusion (a = I, b = 0, c = 0) only.
//! Variable coefficients are NOT exposed (TIER-3).
//! Default Smolyak level ℓ = D + 3 = 9 (→ 533 nodes).
//!
//! ## ABI note
//!
//! `GridFnND<f64, 6>` does NOT cross the WASM boundary — only the flat
//! `Float64Array` of length `n_per_axis^6` is exchanged.
//!
//! Error model: `.kind`-tagged JS `Error` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amdt 1): no `catch_unwind`.

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use js_sys::Float64Array;
use semiflow_core::{
    grid_nd::{GridFnND, GridND},
    smolyak::SmolyakGridND,
    ChernoffFunction, Grid1D, ScratchPool, SquareMatrix,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const D: usize = 6;
const DEFAULT_LEVEL: usize = D + 3; // ℓ = 9 → 533 Smolyak nodes

// ---------------------------------------------------------------------------
// SmolyakD6
// ---------------------------------------------------------------------------

/// Sparse-grid Smolyak Chernoff step, D = 6, unit diffusion (v8.1.0, ADR-0138).
///
/// Applies one or more Chernoff steps of the unit-diffusion Smolyak operator
/// to a flat 6-D `Float64Array` of length `n_per_axis^6`.  The 6-D grid
/// function stays inside Rust; only the flat `f64` buffer crosses the boundary.
///
/// **Narrow scope**: unit a = I, b = 0, c = 0 only.  Default level ℓ = 9.
///
/// Parameters
/// ----------
/// `domain_lo` : f64 — lower bound (all axes; must be finite).
/// `domain_hi` : f64 — upper bound (`> domain_lo`).
/// `n_per_axis` : usize — nodes per axis (>= 4).
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct SmolyakD6 {
    domain_lo: f64,
    domain_hi: f64,
    n_per_axis: usize,
}

#[wasm_bindgen]
impl SmolyakD6 {
    /// Construct `SmolyakD6`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(domain_lo: f64, domain_hi: f64, n_per_axis: usize) -> Result<SmolyakD6, JsValue> {
        validate_domain(domain_lo, domain_hi, n_per_axis)?;
        Ok(SmolyakD6 { domain_lo, domain_hi, n_per_axis })
    }

    /// Apply `n_steps` Smolyak Chernoff steps to flat `u0` and return the result.
    ///
    /// `u0`: `Float64Array` of length `n_per_axis^6`.
    /// `tau`: step size (>= 0, finite).
    /// `n_steps`: number of steps (default 1, >= 1).
    ///
    /// Returns a new `Float64Array` of the same length.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn apply(&self, u0: &Float64Array, tau: f64, n_steps: usize) -> Result<Float64Array, JsValue> {
        validate_tau(tau, n_steps)?;
        let expected = self.n_per_axis.pow(D as u32);
        if u0.length() as usize != expected {
            return Err(make_js_error(
                "GridMismatch",
                &format!("u0 length {} != n_per_axis^6={expected}", u0.length()),
            ));
        }
        let mut buf = vec![0.0f64; expected];
        u0.copy_to(&mut buf);
        let lo = self.domain_lo;
        let hi = self.domain_hi;
        let n = self.n_per_axis;
        let result = run_smolyak(lo, hi, n, tau, &buf, n_steps)?;
        let out = Float64Array::new_with_length(result.len() as u32);
        out.copy_from(&result);
        Ok(out)
    }

    /// Smolyak node count (sparse grid size, computed at call time).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn n_nodes(&self) -> Result<usize, JsValue> {
        let kernel = build_kernel(self.domain_lo, self.domain_hi, self.n_per_axis)?;
        Ok(kernel.n_nodes())
    }

    /// Smolyak level ℓ (always `D + 3 = 9`).
    #[must_use]
    pub fn level(&self) -> usize { DEFAULT_LEVEL }

    /// Total number of grid points (`n_per_axis^6`).
    #[must_use]
    pub fn size(&self) -> usize { self.n_per_axis.pow(D as u32) }

    /// Nodes per axis.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn n_per_axis(&self) -> usize { self.n_per_axis }
}

// ---------------------------------------------------------------------------
// Pure-Rust compute
// ---------------------------------------------------------------------------

fn run_smolyak(
    lo: f64, hi: f64, n: usize,
    tau: f64, u0: &[f64], n_steps: usize,
) -> Result<Vec<f64>, JsValue> {
    let kernel = build_kernel(lo, hi, n)?;
    let grid = build_grid(lo, hi, n)?;
    let mut src = GridFnND::new(grid.clone(), u0.to_vec()).map_err(|e| err_to_js(&e))?;
    let mut dst = GridFnND::new(grid, vec![0.0f64; u0.len()]).map_err(|e| err_to_js(&e))?;
    let mut pool = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut pool).map_err(|e| err_to_js(&e))?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ---------------------------------------------------------------------------
// Builders (per-crate dup, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

fn build_grid(lo: f64, hi: f64, n: usize) -> Result<GridND<f64, D>, JsValue> {
    let ax = Grid1D::new(lo, hi, n).map_err(|e| err_to_js(&e))?;
    GridND::new([ax; D]).map_err(|e| err_to_js(&e))
}

fn build_kernel(lo: f64, hi: f64, n: usize) -> Result<SmolyakGridND<f64, D>, JsValue> {
    let grid = build_grid(lo, hi, n)?;
    SmolyakGridND::with_level(
        |_x: &[f64; D], a: &mut SquareMatrix<f64, D>| {
            for i in 0..D {
                a.set(i, i, 1.0);
            }
        },
        |_x: &[f64; D], b: &mut [f64; D]| {
            for v in b.iter_mut() { *v = 0.0; }
        },
        |_x: &[f64; D]| 0.0_f64,
        grid,
        DEFAULT_LEVEL,
    ).map_err(|e| err_to_js(&e))
}

// ---------------------------------------------------------------------------
// Validators
// ---------------------------------------------------------------------------

fn validate_domain(lo: f64, hi: f64, n: usize) -> Result<(), JsValue> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(make_js_error("OutOfDomain", "domain bounds must be finite"));
    }
    if lo >= hi {
        return Err(make_js_error("GridMismatch", "domain_lo must be < domain_hi"));
    }
    if n < 4 {
        return Err(make_js_error("GridMismatch", "n_per_axis must be >= 4"));
    }
    Ok(())
}

fn validate_tau(tau: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !tau.is_finite() || tau < 0.0 {
        return Err(make_js_error("OutOfDomain", "tau must be finite and >= 0"));
    }
    Ok(())
}
