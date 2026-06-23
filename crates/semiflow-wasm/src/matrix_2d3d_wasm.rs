//! `MatrixDiffusion2D` and `MatrixDiffusion3D` — coupled 2-component
//! 2D/3D diffusion JS classes (`full` feature). ADR-0124, math §33.2/33.3.
//!
//! ## Buffer layout
//!
//! 2D: flat `2*nx*ny`, index `(j*nx+i)*2+c` (j=y, i=x, c∈{0,1}).
//! 3D: flat `2*nx*ny*nz`, index `(k*nx*ny+j*nx+i)*2+c`.
//!
//! Error model: same `.kind`-tagged JS `Error` as `Heat1D`.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

use js_sys::Float64Array;
use semiflow_core::{
    matrix_2d3d::{MatrixDiffusionChernoff2D, MatrixDiffusionChernoff3D, MatrixGridFn2D, MatrixGridFn3D},
    matrix_system::MatrixDiffusionChernoff,
    ChernoffSemigroup, Grid1D, Grid2D, Grid3D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Shared kernel builder
// ---------------------------------------------------------------------------

fn build_axis_kernel(
    a_diag: f64,
    c_coupling: f64,
    axis: Grid1D<f64>,
) -> Result<MatrixDiffusionChernoff<f64, 2>, semiflow_core::SemiflowError> {
    let a = a_diag;
    let c = c_coupling;
    MatrixDiffusionChernoff::<f64, 2>::new(
        move |_x, mat| {
            mat[0][0] = a;
            mat[0][1] = 0.0;
            mat[1][0] = 0.0;
            mat[1][1] = a;
        },
        |_x, mat| {
            mat[0][0] = 0.0;
            mat[0][1] = 0.0;
            mat[1][0] = 0.0;
            mat[1][1] = 0.0;
        },
        move |_x, mat| {
            mat[0][0] = 0.0;
            mat[0][1] = c;
            mat[1][0] = c;
            mat[1][1] = 0.0;
        },
        axis,
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_2d_u0(u0: &Float64Array, nx: usize, ny: usize) -> Result<Vec<f64>, JsValue> {
    let expected = 2 * nx * ny;
    if u0.length() as usize != expected {
        return Err(make_js_error(
            "GridMismatch",
            "u0.length() must equal 2*nx*ny",
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

fn validate_3d_u0(u0: &Float64Array, nx: usize, ny: usize, nz: usize) -> Result<Vec<f64>, JsValue> {
    let expected = 2 * nx * ny * nz;
    if u0.length() as usize != expected {
        return Err(make_js_error(
            "GridMismatch",
            "u0.length() must equal 2*nx*ny*nz",
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

fn validate_evolve(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn vals_to_js(vals: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(vals.len() as u32);
    arr.copy_from(vals);
    arr
}

// ---------------------------------------------------------------------------
// 2D evolve helper
// ---------------------------------------------------------------------------

fn evolve_2d(
    a_diag: f64,
    c_coupling: f64,
    g: Grid2D<f64>,
    vals: &[f64],
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let kx = build_axis_kernel(a_diag, c_coupling, g.x)?;
    let ky = build_axis_kernel(a_diag, c_coupling, g.y)?;
    let kernel = MatrixDiffusionChernoff2D::new(kx, ky);
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let mut src = MatrixGridFn2D::<f64, 2>::new(g);
    src.values.copy_from_slice(vals);
    let out = sg.evolve(t, &src)?;
    Ok(out.values)
}

// ---------------------------------------------------------------------------
// 3D evolve helper
// ---------------------------------------------------------------------------

fn evolve_3d(
    a_diag: f64,
    c_coupling: f64,
    g: Grid3D<f64>,
    vals: &[f64],
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let kx = build_axis_kernel(a_diag, c_coupling, g.x)?;
    let ky = build_axis_kernel(a_diag, c_coupling, g.y)?;
    let kz = build_axis_kernel(a_diag, c_coupling, g.z)?;
    let kernel = MatrixDiffusionChernoff3D::new(kx, ky, kz);
    let sg = ChernoffSemigroup::new(kernel, n_steps)?;
    let mut src = MatrixGridFn3D::<f64, 2>::new(g);
    src.values.copy_from_slice(vals);
    let out = sg.evolve(t, &src)?;
    Ok(out.values)
}

// ===========================================================================
// MatrixDiffusion2D JS class
// ===========================================================================

/// Coupled 2-component 2D diffusion state (ADR-0124, §33.2).
///
/// Solves `∂_t u = a_diag·∂²u + c_coupling·u` on `[xmin,xmax]×[ymin,ymax]`.
/// Palindromic 3-leg Strang; order 2.
///
/// ## Buffer layout
/// `Float64Array` of length `2*nx*ny`. Index `(j*nx+i)*2+c`.
///
/// ## Constructor
/// `new(xmin, xmax, nx, ymin, ymax, ny, u0, a_diag, c_coupling)`
///
/// ## Methods
/// `evolve(t, n_steps)`, `values()`, `order()`, `size()`
#[wasm_bindgen]
pub struct MatrixDiffusion2D {
    a_diag: f64,
    c_coupling: f64,
    grid2d: Grid2D<f64>,
    current: Vec<f64>,
    #[allow(dead_code)]
    nx: usize,
    #[allow(dead_code)]
    ny: usize,
}

#[wasm_bindgen]
impl MatrixDiffusion2D {
    /// Construct `MatrixDiffusion2D`.
    ///
    /// - `a_diag` — diagonal diffusion (default 1.0, must be > 0).
    /// - `c_coupling` — off-diagonal reaction (default 0.0).
    /// - `u0` — flat `Float64Array` of length `2*nx*ny`.
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
        u0: &Float64Array,
        a_diag: f64,
        c_coupling: f64,
    ) -> Result<MatrixDiffusion2D, JsValue> {
        if nx < 5 {
            return Err(make_js_error("OutOfDomain", "nx must be >= 5"));
        }
        if ny < 5 {
            return Err(make_js_error("OutOfDomain", "ny must be >= 5"));
        }
        if !a_diag.is_finite() || a_diag <= 0.0 {
            return Err(make_js_error("OutOfDomain", "a_diag must be finite and > 0"));
        }
        let vals = validate_2d_u0(u0, nx, ny)?;
        let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
        let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| err_to_js(&e))?;
        let grid2d = Grid2D::new(gx, gy);
        // Validate kernel eagerly.
        build_axis_kernel(a_diag, c_coupling, gx).map_err(|e| err_to_js(&e))?;
        Ok(MatrixDiffusion2D { a_diag, c_coupling, grid2d, current: vals, nx, ny })
    }

    /// Advance by time `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve(t, n_steps)?;
        let out = evolve_2d(self.a_diag, self.c_coupling, self.grid2d, &self.current, t, n_steps)
            .map_err(|e| err_to_js(&e))?;
        self.current = out;
        Ok(())
    }

    /// Return flat `Float64Array` of length `2*nx*ny` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        vals_to_js(&self.current)
    }

    /// Approximation order (always 2).
    #[must_use]
    pub fn order(&self) -> u32 {
        2
    }

    /// Buffer size (`2 * nx * ny`).
    #[must_use]
    pub fn size(&self) -> usize {
        self.current.len()
    }
}

// ===========================================================================
// MatrixDiffusion3D JS class
// ===========================================================================

/// Coupled 2-component 3D diffusion state (ADR-0124, §33.3).
///
/// Solves `∂_t u = a_diag·∂²u + c_coupling·u` on a 3D cuboid.
/// Palindromic 5-leg Strang; order 2.
///
/// ## Buffer layout
/// `Float64Array` of length `2*nx*ny*nz`. Index `(k*nx*ny+j*nx+i)*2+c`.
///
/// ## Constructor
/// `new(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, u0, a_diag, c_coupling)`
#[wasm_bindgen]
pub struct MatrixDiffusion3D {
    a_diag: f64,
    c_coupling: f64,
    grid3d: Grid3D<f64>,
    current: Vec<f64>,
    #[allow(dead_code)]
    nx: usize,
    #[allow(dead_code)]
    ny: usize,
    #[allow(dead_code)]
    nz: usize,
}

#[wasm_bindgen]
impl MatrixDiffusion3D {
    /// Construct `MatrixDiffusion3D`.
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
        zmin: f64,
        zmax: f64,
        nz: usize,
        u0: &Float64Array,
        a_diag: f64,
        c_coupling: f64,
    ) -> Result<MatrixDiffusion3D, JsValue> {
        if nx < 5 || ny < 5 || nz < 5 {
            return Err(make_js_error("OutOfDomain", "nx, ny, nz must be >= 5"));
        }
        if !a_diag.is_finite() || a_diag <= 0.0 {
            return Err(make_js_error("OutOfDomain", "a_diag must be finite and > 0"));
        }
        let vals = validate_3d_u0(u0, nx, ny, nz)?;
        let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
        let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| err_to_js(&e))?;
        let gz = Grid1D::new(zmin, zmax, nz).map_err(|e| err_to_js(&e))?;
        let grid3d = Grid3D::new(gx, gy, gz).map_err(|e| err_to_js(&e))?;
        build_axis_kernel(a_diag, c_coupling, gx).map_err(|e| err_to_js(&e))?;
        Ok(MatrixDiffusion3D { a_diag, c_coupling, grid3d, current: vals, nx, ny, nz })
    }

    /// Advance by time `t` using `n_steps` Chernoff iterations.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve(t, n_steps)?;
        let out = evolve_3d(self.a_diag, self.c_coupling, self.grid3d, &self.current, t, n_steps)
            .map_err(|e| err_to_js(&e))?;
        self.current = out;
        Ok(())
    }

    /// Return flat `Float64Array` of length `2*nx*ny*nz` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        vals_to_js(&self.current)
    }

    /// Approximation order (always 2).
    #[must_use]
    pub fn order(&self) -> u32 {
        2
    }

    /// Buffer size (`2 * nx * ny * nz`).
    #[must_use]
    pub fn size(&self) -> usize {
        self.current.len()
    }
}
