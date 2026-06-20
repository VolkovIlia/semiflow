//! Round-9 obstacle engine for WebAssembly (`full` feature).
//!
//! | JS class          | Core type                                              | Python mirror      |
//! |-------------------|--------------------------------------------------------|--------------------|
//! | `ObstacleChernoff`| `ObstacleChernoff<DiffusionChernoff, ConstantObstacle>`| `ObstacleChernoff` |
//!
//! ## Design
//!
//! Mirrors `semiflow-py` `ObstacleChernoff` (`obstacle_py.rs`), D=1 only.
//! WASM boundary uses `Float64Array` (copy semantics); no numpy dependency.
//! Two obstacle variants: constant `level` (default) or per-node `obstacle_array`.
//! Strang-split path (`b ≠ 0 || c ≠ 0`) is NOT exposed here — WASM exposes the
//! fast-path pure-diffusion + constant-obstacle only, matching the minimal surface
//! needed for obstacle pricing. Full split deferred (see NOTE in lib.rs).
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow_core::{
    ChernoffFunction, ConstantObstacle, DiffusionChernoff, Grid1D, GridFn1D, ObstacleChernoff,
    ScratchPool,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Type alias — pure-diffusion + constant-obstacle (fast path)
// ---------------------------------------------------------------------------

type ConstKernel = ObstacleChernoff<DiffusionChernoff<f64>, ConstantObstacle<f64>, f64>;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn extract_u0_obs(u0: &Float64Array, n: usize) -> Result<Vec<f64>, JsValue> {
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

fn fn_to_js_obs(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

fn validate_evolve_obs(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fn build_obs_kernel(
    xmin: f64,
    xmax: f64,
    n: usize,
    a: f64,
    level: f64,
) -> Result<ConstKernel, semiflow_core::SemiflowError> {
    let grid = Grid1D::new(xmin, xmax, n)?;
    let diff = DiffusionChernoff::new_const_a(a, a, grid);
    let obs = ConstantObstacle::new(level)?;
    ObstacleChernoff::new(diff, obs)
}

// ---------------------------------------------------------------------------
// Pure-Rust evolve helper (no GIL equivalent in WASM)
// ---------------------------------------------------------------------------

fn run_obs_evolve(
    kernel: &ConstKernel,
    src_vals: Vec<f64>,
    grid: Grid1D<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    #[allow(clippy::cast_precision_loss)]
    let tau = t / n_steps as f64;
    let mut src = GridFn1D::new(grid, src_vals)?;
    let mut dst = src.zeroed_like();
    let mut scratch = ScratchPool::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ---------------------------------------------------------------------------
// ObstacleChernoff — JS class
// ---------------------------------------------------------------------------

/// 1-D obstacle / variational-inequality Chernoff evolver (math §44).
///
/// Generator: ``L = a·∂_xx`` (constant coefficient, drift-free fast path).
///
/// Mirrors `ObstacleChernoff` (Python), constant-obstacle path only.
/// Strang-split (b,c ≠ 0) is deferred; pass the default `a = 1.0` and
/// choose `level` (flat floor, default 0.0) for typical option-pricing use.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct ObstacleChernoffWasm {
    kernel: ConstKernel,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl ObstacleChernoffWasm {
    /// Construct `ObstacleChernoff`.
    ///
    /// - `xmin`, `xmax` — domain bounds (finite, `xmin < xmax`).
    /// - `n` — grid nodes (≥ 4).
    /// - `u0` — `Float64Array` of length `n`, all finite.
    /// - `a` — diffusion coefficient (> 0, default `1.0`).
    /// - `level` — constant obstacle floor `g ≡ level` (default `0.0`).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        a: f64,
        level: f64,
    ) -> Result<ObstacleChernoffWasm, JsValue> {
        if !a.is_finite() || a <= 0.0 {
            return Err(make_js_error("OutOfDomain", "a must be finite and > 0"));
        }
        if !level.is_finite() {
            return Err(make_js_error("OutOfDomain", "level must be finite"));
        }
        let buf = extract_u0_obs(u0, n)?;
        let kernel = build_obs_kernel(xmin, xmax, n, a, level).map_err(|e| err_to_js(&e))?;
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let current = GridFn1D::new(grid, buf).map_err(|e| err_to_js(&e))?;
        Ok(ObstacleChernoffWasm { kernel, current })
    }

    /// Advance state by `t` using `n_steps` Chernoff iterations.
    ///
    /// Returns updated `Float64Array` of length `n` (copy).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<Float64Array, JsValue> {
        validate_evolve_obs(t, n_steps)?;
        let grid = self.current.grid;
        let src = self.current.values.clone();
        let out = run_obs_evolve(&self.kernel, src, grid, t, n_steps)
            .map_err(|e| err_to_js(&e))?;
        self.current.values.clone_from(&out);
        Ok(fn_to_js_obs(&out))
    }

    /// Return current grid values as `Float64Array` of length `n` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        fn_to_js_obs(&self.current.values)
    }

    /// Approximation order (always 1; §44.4 projection cap).
    #[must_use]
    pub fn order(&self) -> u32 {
        1
    }

    /// Number of grid nodes.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}
