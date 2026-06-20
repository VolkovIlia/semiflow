//! Non-separable 2D diffusion engines for WebAssembly (`full` feature).
//!
//! | JS class               | Core type                                              | Python mirror          |
//! |------------------------|--------------------------------------------------------|------------------------|
//! | `NonSeparable2D`       | `NonSeparableMixedChernoff<Dc, Dc>` (scalar coupling)  | `NonSeparable2D`       |
//! | `NonSeparable2DAniso`  | `NonSeparableMixedChernoff<Dc, Dc>` (beta array)       | `NonSeparable2DAniso`  |
//!
//! Buffer layout: flat row-major, x-fastest, `values[j*nx + i] ≈ u(x_i, y_j)`.
//!
//! Error model: `.kind`-tagged JS `Error` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amdt 1): no `catch_unwind`; validate before calling core.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::cast_possible_wrap)]

use alloc::sync::Arc;

extern crate alloc;

use js_sys::Float64Array;
use semiflow_core::{
    nonseparable_mixed_closure, ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D, GridFn2D,
    ScratchPool,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Shared type alias
// ---------------------------------------------------------------------------

type Nsm = semiflow_core::NonSeparableMixedChernoff<DiffusionChernoff<f64>, DiffusionChernoff<f64>>;

// ---------------------------------------------------------------------------
// Shared helpers (mirrors strang_nd_wasm.rs)
// ---------------------------------------------------------------------------

fn validate_tau(tau: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(make_js_error("OutOfDomain", "tau must be finite and > 0"));
    }
    Ok(())
}

fn extract_flat_2d(u0: &Float64Array, nx: usize, ny: usize) -> Result<Vec<f64>, JsValue> {
    let expected = nx * ny;
    if u0.length() as usize != expected {
        return Err(make_js_error("GridMismatch", "u0 length must equal nx * ny"));
    }
    let mut buf = vec![0.0f64; expected];
    u0.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", "u0 contains NaN or Inf"));
        }
    }
    Ok(buf)
}

fn extract_finite_array(arr: &Float64Array, expected: usize, name: &str) -> Result<Vec<f64>, JsValue> {
    if arr.length() as usize != expected {
        return Err(make_js_error("GridMismatch", &format!("{name} length must equal {expected}")));
    }
    let mut buf = vec![0.0f64; expected];
    arr.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", &format!("{name} contains NaN or Inf")));
        }
    }
    Ok(buf)
}

fn vec_to_js(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

fn unit_diff(grid: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid)
}

fn build_grid_2d(
    xmin: f64, xmax: f64, nx: usize,
    ymin: f64, ymax: f64, ny: usize,
) -> Result<(Grid1D<f64>, Grid1D<f64>, Grid2D<f64>), JsValue> {
    let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
    let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| err_to_js(&e))?;
    Ok((gx, gy, Grid2D::new(gx, gy)))
}

fn evolve_nsm(
    kernel: &Nsm,
    grid: Grid2D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, JsValue> {
    let mut state = GridFn2D::new(grid, input).map_err(|e| err_to_js(&e))?;
    let mut dst = GridFn2D::new(grid, vec![0.0; state.values.len()]).map_err(|e| err_to_js(&e))?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &state, &mut dst, &mut scratch).map_err(|e| err_to_js(&e))?;
        core::mem::swap(&mut state, &mut dst);
    }
    Ok(state.values)
}

// ---------------------------------------------------------------------------
// NonSeparable2D — constant scalar coupling c
// ---------------------------------------------------------------------------

/// 2-D non-separable diffusion with constant scalar coupling `c`.
///
/// Solves `∂_t u = ∂_xx u + ∂_yy u + c·∂_xy u` on `[xmin,xmax]×[ymin,ymax]`.
/// Layout: flat row-major, x-fastest, `values[j*nx + i] ≈ u(x_i, y_j)`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct NonSeparable2D {
    kernel: Nsm,
    grid: Grid2D<f64>,
    nx: usize,
    ny: usize,
}

#[wasm_bindgen]
impl NonSeparable2D {
    /// Construct `NonSeparable2D` with constant coupling `c` (default `0.0`).
    ///
    /// Parameters: `xmin`, `xmax`, `nx` (x-axis, nx ≥ 4),
    ///             `ymin`, `ymax`, `ny` (y-axis, ny ≥ 4), `c` (scalar coupling).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64, xmax: f64, nx: usize,
        ymin: f64, ymax: f64, ny: usize,
        c: f64,
    ) -> Result<NonSeparable2D, JsValue> {
        let (gx, gy, grid) = build_grid_2d(xmin, xmax, nx, ymin, ymax, ny)?;
        let c_norm = c.abs();
        let c_val = c;
        let c_arc: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'static> =
            Arc::new(move |_x, _y| c_val);
        let kernel = nonseparable_mixed_closure::with_closure_c(
            unit_diff(gx), unit_diff(gy), c_arc, c_norm, grid,
        ).map_err(|e| err_to_js(&e))?;
        Ok(NonSeparable2D { kernel, grid, nx, ny })
    }

    /// Evolve flat row-major `u0` (length `nx * ny`) by `n_steps` of size `tau`.
    ///
    /// Returns a new `Float64Array`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&self, u0: &Float64Array, tau: f64, n_steps: usize) -> Result<Float64Array, JsValue> {
        validate_tau(tau, n_steps)?;
        let input = extract_flat_2d(u0, self.nx, self.ny)?;
        let result = evolve_nsm(&self.kernel, self.grid, input, tau, n_steps)?;
        Ok(vec_to_js(&result))
    }

    /// X-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> usize { self.nx }

    /// Y-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn ny(&self) -> usize { self.ny }

    /// Total number of grid nodes (`nx * ny`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize { self.nx * self.ny }
}

// ---------------------------------------------------------------------------
// NonSeparable2DAniso — spatially-varying beta array
// ---------------------------------------------------------------------------

/// 2-D non-separable diffusion with spatially-varying coupling `β(x,y)`.
///
/// Solves `∂_t u = ∂_xx u + ∂_yy u + β(x,y)·∂_xy u`.
/// `beta_values`: flat row-major `Float64Array` of length `nx * ny`, finite.
/// `beta_norm_bound`: sup-norm bound on `|β|`; pass `0.0` to auto-compute as
/// `1.1 × max|β|` (a zero bound will be interpreted as zero coupling).
///
/// Layout: flat row-major, x-fastest, `values[j*nx + i] ≈ u(x_i, y_j)`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct NonSeparable2DAniso {
    kernel: Nsm,
    grid: Grid2D<f64>,
    nx: usize,
    ny: usize,
}

#[wasm_bindgen]
impl NonSeparable2DAniso {
    /// Construct `NonSeparable2DAniso`.
    ///
    /// `beta_values`: `Float64Array` of length `nx * ny`, finite.
    /// `beta_norm_bound`: pass `0.0` to auto-compute (`1.1 × max|β|`).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64, xmax: f64, nx: usize,
        ymin: f64, ymax: f64, ny: usize,
        beta_values: &Float64Array,
        beta_norm_bound: f64,
    ) -> Result<NonSeparable2DAniso, JsValue> {
        let beta_raw = extract_finite_array(beta_values, nx * ny, "beta_values")?;
        let norm_bound = compute_norm_bound(&beta_raw, beta_norm_bound);
        let (gx, gy, grid) = build_grid_2d(xmin, xmax, nx, ymin, ymax, ny)?;
        let (beta_arc, axes, ns) = build_beta_closure_2d(beta_raw, xmin, xmax, nx, ymin, ymax, ny);
        let beta_cls: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'static> =
            Arc::new(move |x, y| {
                let i = clamp_idx(x, axes[0].0, axes[0].1, ns[0]);
                let j = clamp_idx(y, axes[1].0, axes[1].1, ns[1]);
                beta_arc[j * ns[0] + i]
            });
        let kernel = nonseparable_mixed_closure::with_closure_beta(
            unit_diff(gx), unit_diff(gy), beta_cls, norm_bound, grid,
        ).map_err(|e| err_to_js(&e))?;
        Ok(NonSeparable2DAniso { kernel, grid, nx, ny })
    }

    /// Evolve flat row-major `u0` (length `nx * ny`) by `n_steps` of size `tau`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&self, u0: &Float64Array, tau: f64, n_steps: usize) -> Result<Float64Array, JsValue> {
        validate_tau(tau, n_steps)?;
        let input = extract_flat_2d(u0, self.nx, self.ny)?;
        let result = evolve_nsm(&self.kernel, self.grid, input, tau, n_steps)?;
        Ok(vec_to_js(&result))
    }

    /// X-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> usize { self.nx }

    /// Y-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn ny(&self) -> usize { self.ny }

    /// Total number of grid nodes (`nx * ny`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize { self.nx * self.ny }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn compute_norm_bound(beta_raw: &[f64], hint: f64) -> f64 {
    if hint > 0.0 {
        return hint;
    }
    let m = beta_raw.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    if m == 0.0 { 0.0 } else { m * 1.1 }
}

type BetaClosureParts = (Arc<Vec<f64>>, [(f64, f64); 2], [usize; 2]);

fn build_beta_closure_2d(
    beta: Vec<f64>,
    xmin: f64, xmax: f64, nx: usize,
    ymin: f64, ymax: f64, ny: usize,
) -> BetaClosureParts {
    (Arc::new(beta), [(xmin, xmax), (ymin, ymax)], [nx, ny])
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
fn clamp_idx(x: f64, lo: f64, hi: f64, n: usize) -> usize {
    if n <= 1 { return 0; }
    let fi = (x - lo) / (hi - lo) * (n as f64 - 1.0);
    (fi.round() as isize).clamp(0, n as isize - 1) as usize
}
