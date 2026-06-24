//! 2-D and 3-D tensor-product diffusion engines for WebAssembly (`full` feature).
//!
//! Exposes four JS classes mirroring the Python binding:
//!
//! | JS class       | Core type                                          | Python mirror   |
//! |----------------|----------------------------------------------------|-----------------|
//! | `Heat2D`       | `Strang2D<DiffusionChernoff, DiffusionChernoff>`   | `Heat2D`        |
//! | `Heat3D`       | `Strang3D<Dc, Dc, Dc>` (unit `a = 1`)             | `Heat3D`        |
//! | `Heat2DVarA`   | `Strang2D<DiffusionChernoff, DiffusionChernoff>`   | `Heat2DVarA`    |
//! | `Heat3DVarA`   | `Strang3D<Dc, Dc, Dc>` (per-axis variable `a`)     | `Heat3DVarA`    |
//!
//! Buffer layout (row-major, x-fastest):
//! - 2D: `values[j * nx + i] ≈ u(x_i, y_j)`, length `nx * ny`.
//! - 3D: `values[k * nx * ny + j * nx + i] ≈ u(x_i, y_j, z_k)`, length `nx * ny * nz`.
//!
//! JS receives/returns a flat `Float64Array`. Dimensions exposed as `nx`, `ny`, `nz` getters.
//!
//! Error model: same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`; validate before calling core.

#![allow(unsafe_code)]
// Mirror the Python binding suppression list (too_many_arguments is inherent to 2D/3D APIs).
#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use js_sys::Float64Array;
use semiflow::{
    ChernoffFunction, DiffusionChernoff, Grid1D, Grid2D, Grid3D, GridFn2D, GridFn3D, ScratchPool,
    Strang2D, Strang3D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Validate that `tau > 0` and `n_steps >= 1` (mirrors Python `validate_tau_nsteps`).
fn validate_tau_nsteps(tau: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(make_js_error("OutOfDomain", "tau must be finite and > 0"));
    }
    Ok(())
}

/// Copy `Float64Array` to `Vec<f64>`, check length and finiteness.
fn extract_flat(u0: &Float64Array, expected: usize) -> Result<Vec<f64>, JsValue> {
    if u0.length() as usize != expected {
        return Err(make_js_error(
            "GridMismatch",
            "u0.length() must equal the total grid size",
        ));
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

/// Validate a positive finite coefficient vector (mirrors Python `extract_pos_coeff_vec`).
fn extract_pos_coeff(a: &Float64Array, n: usize, name: &str) -> Result<Vec<f64>, JsValue> {
    if a.length() as usize != n {
        return Err(make_js_error(
            "GridMismatch",
            &format!("{name}.length() must equal n"),
        ));
    }
    let mut buf = vec![0.0f64; n];
    a.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() || v <= 0.0 {
            return Err(make_js_error(
                "NanInf",
                &format!("{name} must be > 0 and finite"),
            ));
        }
    }
    Ok(buf)
}

/// Emit flat values as a JS `Float64Array` (copy).
fn vec_to_js(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

/// Build a unit-coefficient `DiffusionChernoff` on `grid` (`a ≡ 1`, no drift/reaction).
fn unit_diff(grid: Grid1D<f64>) -> DiffusionChernoff<f64> {
    DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid)
}

/// Build a variable-coefficient `DiffusionChernoff` from a sampled `a_vals` table.
fn var_diff(
    a_vals: Vec<f64>,
    amin: f64,
    amax: f64,
    n: usize,
    grid: Grid1D<f64>,
) -> DiffusionChernoff<f64> {
    let norm = a_vals.iter().copied().fold(0.0_f64, f64::max);
    let arc = Arc::new(a_vals);
    DiffusionChernoff::with_closure(
        move |t: f64| interp_1d(&arc, amin, amax, n, t),
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        norm,
        grid,
    )
}

/// Linear interpolation of `vals` at point `t` (mirrors Python `interp_1d`).
///
/// Casts are intentional: grid indices are much smaller than 2^52 in practice.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn interp_1d(vals: &[f64], amin: f64, amax: f64, n: usize, t: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let n_f = n as f64;
    let dx = (amax - amin) / (n_f - 1.0).max(1.0);
    let frac = (t - amin) / dx;
    let frac_clamped = frac.clamp(0.0, (n.saturating_sub(2)) as f64);
    let i = frac_clamped as usize;
    let lo = vals[i];
    let hi = vals[(i + 1).min(n - 1)];
    lo + (hi - lo) * (frac_clamped - i as f64)
}

// ---------------------------------------------------------------------------
// Evolve helpers (pure Rust, no WASM-specific code)
// ---------------------------------------------------------------------------

type Strang2Dc = Strang2D<DiffusionChernoff<f64>, DiffusionChernoff<f64>>;
type Strang3Dc = Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>;

fn evolve_2d(
    strang: &Strang2Dc,
    grid: Grid2D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, JsValue> {
    let mut state = GridFn2D::new(grid, input).map_err(|e| err_to_js(&e))?;
    let mut dst = GridFn2D::new(grid, vec![0.0; state.values.len()]).map_err(|e| err_to_js(&e))?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        strang
            .apply_into(tau, &state, &mut dst, &mut scratch)
            .map_err(|e| err_to_js(&e))?;
        core::mem::swap(&mut state, &mut dst);
    }
    Ok(state.values)
}

fn evolve_3d(
    strang: &Strang3Dc,
    grid: Grid3D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, JsValue> {
    let mut state = GridFn3D::new(grid, input).map_err(|e| err_to_js(&e))?;
    let mut dst = GridFn3D::new(grid, vec![0.0; state.values.len()]).map_err(|e| err_to_js(&e))?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        strang
            .apply_into(tau, &state, &mut dst, &mut scratch)
            .map_err(|e| err_to_js(&e))?;
        core::mem::swap(&mut state, &mut dst);
    }
    Ok(state.values)
}

// ---------------------------------------------------------------------------
// Heat2D — unit coefficient 2D heat
// ---------------------------------------------------------------------------

/// 2-D heat equation (`a = 1`, palindromic Strang splitting).
///
/// Solves `∂_t u = ∂_xx u + ∂_yy u` on `[xmin,xmax] × [ymin,ymax]`.
/// Buffer layout: flat row-major, `values[j*nx + i] ≈ u(x_i, y_j)`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct Heat2D {
    strang: Strang2Dc,
    grid: Grid2D<f64>,
    nx: usize,
    ny: usize,
}

#[wasm_bindgen]
impl Heat2D {
    /// Construct `Heat2D` on a `Grid2D` (unit `a = 1`).
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — X-axis bounds (finite, `xmin < xmax`).
    /// - `nx` — X-axis nodes (≥ 4).
    /// - `ymin`, `ymax` — Y-axis bounds (finite, `ymin < ymax`).
    /// - `ny` — Y-axis nodes (≥ 4).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
    ) -> Result<Heat2D, JsValue> {
        let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
        let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| err_to_js(&e))?;
        let grid = Grid2D::new(gx, gy);
        let strang = Strang2D::new(unit_diff(gx), unit_diff(gy));
        Ok(Heat2D {
            strang,
            grid,
            nx,
            ny,
        })
    }

    /// Evolve flat row-major `u0` (length `nx * ny`) by `n_steps` of size `tau`.
    ///
    /// Returns a new `Float64Array` with the evolved state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(
        &self,
        u0: &Float64Array,
        tau: f64,
        n_steps: usize,
    ) -> Result<Float64Array, JsValue> {
        validate_tau_nsteps(tau, n_steps)?;
        let input = extract_flat(u0, self.nx * self.ny)?;
        let result = evolve_2d(&self.strang, self.grid, input, tau, n_steps)?;
        Ok(vec_to_js(&result))
    }

    /// Return grid width (X-axis node count).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Return grid height (Y-axis node count).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Total number of grid nodes (`nx * ny`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.nx * self.ny
    }
}

// Heat3D and Heat3DVarA — extracted to keep file ≤500 lines.
include!("strang_nd_wasm_3d.rs");
