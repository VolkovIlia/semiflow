//! Anisotropic shift Chernoff engines for WebAssembly (`full` feature).
//!
//! | JS class               | Core type                                    | Python mirror          |
//! |------------------------|----------------------------------------------|------------------------|
//! | `AnisotropicShiftND2`  | `AnisotropicShiftChernoffND<f64, 2>`         | `AnisotropicShiftND2`  |
//! | `AnisotropicShiftND3`  | `AnisotropicShiftChernoffND<f64, 3>`         | `AnisotropicShiftND3`  |
//!
//! ## State model (mirrors Python M19)
//!
//! Unlike `Heat2D`, these engines hold mutable state (`current`).
//! `set_state(u0)` sets the initial condition; `evolve(t, n_steps)` advances
//! in-place; `values()` returns the current state as `Float64Array`.
//!
//! ## Coefficient layout (x-fastest / axis-0-fastest)
//!
//! - `a_values`: length `nx * ny * 4` (D=2) or `nx * ny * nz * 9` (D=3).
//!   Row-major, SPD diffusion tensor per point.
//! - `b_values`: length `nx * ny * 2` or `nx * ny * nz * 3`. `null` → zero.
//! - `c_values`: length `nx * ny` or `nx * ny * nz`. `null` → zero.
//!
//! Error model: `.kind`-tagged JS `Error` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amdt 1): no `catch_unwind`.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::needless_pass_by_value)]

extern crate alloc;

use alloc::sync::Arc;

use js_sys::Float64Array;
use semiflow::{
    grid_nd::{GridFnND, GridND},
    shift_nd::AnisotropicShiftChernoffND,
    ChernoffFunction, Grid1D, ScratchPool,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn validate_t(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn extract_finite(arr: &Float64Array, expected: usize, name: &str) -> Result<Vec<f64>, JsValue> {
    if arr.length() as usize != expected {
        return Err(make_js_error(
            "GridMismatch",
            &format!("{name} length {} != expected {expected}", arr.length()),
        ));
    }
    let mut buf = vec![0.0f64; expected];
    arr.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() {
            return Err(make_js_error(
                "NanInf",
                &format!("{name} contains NaN or Inf"),
            ));
        }
    }
    Ok(buf)
}

fn vec_to_js(values: &[f64]) -> Float64Array {
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

/// Nearest grid index for physical coordinate `x` in axis `[lo, hi]` of size `n`.
fn phys_to_idx(x: f64, lo: f64, hi: f64, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let fi = (x - lo) / (hi - lo) * (n as f64 - 1.0);
    (fi.round() as isize).clamp(0, n as isize - 1) as usize
}

// ---------------------------------------------------------------------------
// AnisotropicShiftND2
// ---------------------------------------------------------------------------

/// Anisotropic shift Chernoff kernel on a 2-D tensor-product grid (M19).
///
/// Solves `∂_t u = A(x)·∇²u + b(x)·∇u + c(x)·u` where `A` is 2×2 SPD.
/// Order 1 (ADR-0112). Use `set_state(u0)`, then `evolve(t, n_steps)`,
/// then `values()` to retrieve the current state.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct AnisotropicShiftND2 {
    kernel: Arc<AnisotropicShiftChernoffND<f64, 2>>,
    grid: GridND<f64, 2>,
    current: Vec<f64>,
    nx: usize,
    ny: usize,
}

#[wasm_bindgen]
impl AnisotropicShiftND2 {
    /// Construct `AnisotropicShiftND2`.
    ///
    /// `a_values`: `Float64Array` length `nx * ny * 4` (2×2 row-major SPD per point).
    /// `b_values`: `Float64Array` length `nx * ny * 2`, or `null` → zero.
    /// `c_values`: `Float64Array` length `nx * ny`, or `null` → zero.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        nx: usize,
        ny: usize,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        a_values: &Float64Array,
        b_values: Option<Float64Array>,
        c_values: Option<Float64Array>,
    ) -> Result<AnisotropicShiftND2, JsValue> {
        let n_pts = nx * ny;
        let a_raw = extract_finite(a_values, 4 * n_pts, "a_values")?;
        let b_raw = match b_values {
            Some(ref b) => extract_finite(b, 2 * n_pts, "b_values")?,
            None => vec![0.0f64; 2 * n_pts],
        };
        let c_raw = match c_values {
            Some(ref c) => extract_finite(c, n_pts, "c_values")?,
            None => vec![0.0f64; n_pts],
        };
        let (kernel, grid) = build_nd2_kernel(nx, ny, xmin, xmax, ymin, ymax, a_raw, b_raw, c_raw)?;
        Ok(AnisotropicShiftND2 {
            kernel: Arc::new(kernel),
            grid,
            current: vec![0.0f64; n_pts],
            nx,
            ny,
        })
    }

    /// Set initial condition from flat `Float64Array` of length `nx * ny`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn set_state(&mut self, u0: &Float64Array) -> Result<(), JsValue> {
        let vals = extract_finite(u0, self.nx * self.ny, "u0")?;
        self.current = vals;
        Ok(())
    }

    /// Advance state by time `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_t(t, n_steps)?;
        let tau = t / n_steps as f64;
        let kernel = Arc::clone(&self.kernel);
        let grid = self.grid.clone();
        let input = self.current.clone();
        self.current = evolve_nd2(kernel, grid, input, tau, n_steps)?;
        Ok(())
    }

    /// Return current state as `Float64Array` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        vec_to_js(&self.current)
    }

    /// Approximation order (always 1, ADR-0112).
    #[must_use]
    pub fn order(&self) -> u32 {
        1
    }

    /// X-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Y-axis node count.
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

fn evolve_nd2(
    kernel: Arc<AnisotropicShiftChernoffND<f64, 2>>,
    grid: GridND<f64, 2>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, JsValue> {
    let mut src = GridFnND::<f64, 2>::new(grid.clone(), input).map_err(|e| err_to_js(&e))?;
    let mut dst =
        GridFnND::<f64, 2>::new(grid, vec![0.0; src.values.len()]).map_err(|e| err_to_js(&e))?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel
            .apply_into(tau, &src, &mut dst, &mut scratch)
            .map_err(|e| err_to_js(&e))?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

fn build_nd2_kernel(
    nx: usize,
    ny: usize,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
) -> Result<(AnisotropicShiftChernoffND<f64, 2>, GridND<f64, 2>), JsValue> {
    let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
    let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| err_to_js(&e))?;
    let grid = GridND::<f64, 2>::new([gx, gy]).map_err(|e| err_to_js(&e))?;
    let ns = [nx, ny];
    let axes = [(xmin, xmax, nx), (ymin, ymax, ny)];
    let a_arc = Arc::new(a);
    let b_arc = Arc::new(b);
    let c_arc = Arc::new(c);
    // Each closure must own a distinct Arc clone (moves).
    let a_a = Arc::clone(&a_arc);
    let b_b = Arc::clone(&b_arc);
    let c_c = Arc::clone(&c_arc);
    drop((a_arc, b_arc, c_arc));
    let kernel = AnisotropicShiftChernoffND::<f64, 2>::new(
        move |x, mat| {
            let flat = nd_idx_2(x, &ns, &axes);
            let base = flat * 4;
            mat.set(0, 0, a_a[base]);
            mat.set(0, 1, a_a[base + 1]);
            mat.set(1, 0, a_a[base + 2]);
            mat.set(1, 1, a_a[base + 3]);
        },
        move |x, bv| {
            let flat = nd_idx_2(x, &ns, &axes);
            bv[0] = b_b[flat * 2];
            bv[1] = b_b[flat * 2 + 1];
        },
        move |x| c_c[nd_idx_2(x, &ns, &axes)],
        grid.clone(),
    )
    .map_err(|e| err_to_js(&e))?;
    Ok((kernel, grid))
}

fn nd_idx_2(x: &[f64; 2], ns: &[usize; 2], axes: &[(f64, f64, usize)]) -> usize {
    let k0 = phys_to_idx(x[0], axes[0].0, axes[0].1, ns[0]);
    let k1 = phys_to_idx(x[1], axes[1].0, axes[1].1, ns[1]);
    k0 + k1 * ns[0]
}

// ---------------------------------------------------------------------------
// AnisotropicShiftND3
// ---------------------------------------------------------------------------

/// Anisotropic shift Chernoff kernel on a 3-D tensor-product grid (M19, D=3).
///
/// Same contract as `AnisotropicShiftND2` for 3 spatial dimensions.
/// `a_values`: length `nx * ny * nz * 9`; `b_values`: `nx * ny * nz * 3`
/// (or `null`); `c_values`: `nx * ny * nz` (or `null`).
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct AnisotropicShiftND3 {
    kernel: Arc<AnisotropicShiftChernoffND<f64, 3>>,
    grid: GridND<f64, 3>,
    current: Vec<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
}

#[wasm_bindgen]
impl AnisotropicShiftND3 {
    /// Construct `AnisotropicShiftND3`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        zmin: f64,
        zmax: f64,
        a_values: &Float64Array,
        b_values: Option<Float64Array>,
        c_values: Option<Float64Array>,
    ) -> Result<AnisotropicShiftND3, JsValue> {
        let n_pts = nx * ny * nz;
        let a_raw = extract_finite(a_values, 9 * n_pts, "a_values")?;
        let b_raw = match b_values {
            Some(ref b) => extract_finite(b, 3 * n_pts, "b_values")?,
            None => vec![0.0f64; 3 * n_pts],
        };
        let c_raw = match c_values {
            Some(ref c) => extract_finite(c, n_pts, "c_values")?,
            None => vec![0.0f64; n_pts],
        };
        let (kernel, grid) = build_nd3_kernel(
            nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, a_raw, b_raw, c_raw,
        )?;
        Ok(AnisotropicShiftND3 {
            kernel: Arc::new(kernel),
            grid,
            current: vec![0.0f64; n_pts],
            nx,
            ny,
            nz,
        })
    }

    /// Set initial condition from flat `Float64Array` of length `nx * ny * nz`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn set_state(&mut self, u0: &Float64Array) -> Result<(), JsValue> {
        let vals = extract_finite(u0, self.nx * self.ny * self.nz, "u0")?;
        self.current = vals;
        Ok(())
    }

    /// Advance state by time `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_t(t, n_steps)?;
        let tau = t / n_steps as f64;
        let kernel = Arc::clone(&self.kernel);
        let grid = self.grid.clone();
        let input = self.current.clone();
        self.current = evolve_nd3(kernel, grid, input, tau, n_steps)?;
        Ok(())
    }

    /// Return current state as `Float64Array` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        vec_to_js(&self.current)
    }

    /// Approximation order (always 1, ADR-0112).
    #[must_use]
    pub fn order(&self) -> u32 {
        1
    }

    /// X-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Y-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Z-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nz(&self) -> usize {
        self.nz
    }

    /// Total number of grid nodes (`nx * ny * nz`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}

fn evolve_nd3(
    kernel: Arc<AnisotropicShiftChernoffND<f64, 3>>,
    grid: GridND<f64, 3>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, JsValue> {
    let mut src = GridFnND::<f64, 3>::new(grid.clone(), input).map_err(|e| err_to_js(&e))?;
    let mut dst =
        GridFnND::<f64, 3>::new(grid, vec![0.0; src.values.len()]).map_err(|e| err_to_js(&e))?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel
            .apply_into(tau, &src, &mut dst, &mut scratch)
            .map_err(|e| err_to_js(&e))?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

fn build_nd3_kernel(
    nx: usize,
    ny: usize,
    nz: usize,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: f64,
    zmax: f64,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
) -> Result<(AnisotropicShiftChernoffND<f64, 3>, GridND<f64, 3>), JsValue> {
    let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
    let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| err_to_js(&e))?;
    let gz = Grid1D::new(zmin, zmax, nz).map_err(|e| err_to_js(&e))?;
    let grid = GridND::<f64, 3>::new([gx, gy, gz]).map_err(|e| err_to_js(&e))?;
    let ns = [nx, ny, nz];
    let axes = [(xmin, xmax, nx), (ymin, ymax, ny), (zmin, zmax, nz)];
    let a_arc = Arc::new(a);
    let b_arc = Arc::new(b);
    let c_arc = Arc::new(c);
    // Each closure must own a distinct Arc clone (moves).
    let a_a = Arc::clone(&a_arc);
    let b_b = Arc::clone(&b_arc);
    let c_c = Arc::clone(&c_arc);
    drop((a_arc, b_arc, c_arc));
    let kernel = AnisotropicShiftChernoffND::<f64, 3>::new(
        move |x, mat| {
            let flat = nd_idx_3(x, &ns, &axes);
            let base = flat * 9;
            for r in 0..3 {
                for ci in 0..3 {
                    mat.set(r, ci, a_a[base + r * 3 + ci]);
                }
            }
        },
        move |x, bv| {
            let flat = nd_idx_3(x, &ns, &axes);
            bv[0] = b_b[flat * 3];
            bv[1] = b_b[flat * 3 + 1];
            bv[2] = b_b[flat * 3 + 2];
        },
        move |x| c_c[nd_idx_3(x, &ns, &axes)],
        grid.clone(),
    )
    .map_err(|e| err_to_js(&e))?;
    Ok((kernel, grid))
}

fn nd_idx_3(x: &[f64; 3], ns: &[usize; 3], axes: &[(f64, f64, usize)]) -> usize {
    let k0 = phys_to_idx(x[0], axes[0].0, axes[0].1, ns[0]);
    let k1 = phys_to_idx(x[1], axes[1].0, axes[1].1, ns[1]);
    let k2 = phys_to_idx(x[2], axes[2].0, axes[2].1, ns[2]);
    k0 + k1 * ns[0] + k2 * ns[0] * ns[1]
}
