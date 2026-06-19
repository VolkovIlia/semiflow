//! v8.3.0 WASM bindings for `ResolventJumpChernoff2D`/`3D` (F2 ND, ADR-0153, ADR-0148).
//!
//! Implements `ResolventJump2DV8` and `ResolventJump3DV8` — stateless-per-call
//! JS classes that evaluate `e^{tA}g` for the 2D/3D unit-diffusion heat kernel
//! via the TWS parabolic-contour inverse Laplace quadrature (math.md §47.8).
//!
//! ## NARROW scope (§47.8, ADR-0148 NORMATIVE)
//!
//! Self-adjoint / sectorial parabolic generators only.
//! Non-self-adjoint / advection-dominated generators are OUT of scope.
//! `mNodes >= 6` is enforced at construction.
//!
//! ## ND layout contract (§3.1, `V8_3_TIER3_BINDING_DESIGN.md` — NORMATIVE)
//!
//! The JS caller is responsible for providing a `Float64Array` whose elements
//! follow the **axis-0-fastest** (Fortran / column-major) layout:
//!   `idx(i,j) = j·nx + i`              (2D),
//!   `idx(i,j,k) = k·nx·ny + j·nx + i`  (3D).
//! The returned `Float64Array` uses the same layout.
//! Array length: `nx·ny` (2D) or `nx·ny·nz` (3D).
//!
//! ## Error model
//!
//! All errors return `Err(JsValue)` with a `.kind` string discriminator:
//! `"GridMismatch"`, `"OutOfDomain"`, `"Panic"`.
//!
//! ## Panic boundary
//!
//! Uses workspace `[profile.release]` (`panic = "abort"`, ADR-0028 Amendment 1).
//! No `catch_unwind`. Better diagnostics via `panic_hook_init()`.
//!
//! ## ADR-0028 Amendment 2
//!
//! Per-crate duplication required — no shared util with semiflow-ffi/py.

// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::missing_errors_doc,
    clippy::too_many_arguments
)]

use wasm_bindgen::prelude::*;

use semiflow_core::{
    Grid1D, Grid2D, Grid3D, GridFn2D, GridFn3D, ResolventJumpChernoff2D, ResolventJumpChernoff3D,
};

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// ResolventJump2DV8 WASM class
// ---------------------------------------------------------------------------

/// v8.3.0 Resolvent time-jump evaluator for 2D unit-diffusion heat (ADR-0153).
///
/// Evaluates `e^{tA}g` via the TWS parabolic-contour inverse Laplace
/// quadrature (math.md §47.8, ADR-0148). Suitable for large `t`.
///
/// **NARROW scope**: self-adjoint / sectorial parabolic generators only.
///
/// ## ND layout (NORMATIVE)
///
/// `g` must be a `Float64Array` with **axis-0-fastest** element order:
/// `g[j*nx + i]` is the value at grid point `(i, j)`.
/// The returned array uses the same layout.
///
/// ## JS Example
///
/// ```js
/// import init, { ResolventJump2DV8 } from "@semiflow/wasm";
/// await init();
/// const nx = 8, ny = 8;
/// const g = new Float64Array(nx * ny);
/// for (let j = 0; j < ny; j++) {
///   for (let i = 0; i < nx; i++) {
///     const x = -5 + 10*i/(nx-1), y = -5 + 10*j/(ny-1);
///     g[j*nx + i] = Math.exp(-x*x - y*y);  // axis-0-fastest
///   }
/// }
/// const rj = new ResolventJump2DV8(-5, 5, nx, -5, 5, ny, 8);
/// const result = rj.jump(1.0, g);  // Float64Array length nx*ny
/// ```
///
/// ## Error model (`.kind` discriminator)
///
/// - `"GridMismatch"` — invalid geometry or `g.length != nx*ny`.
/// - `"OutOfDomain"`  — `mNodes < 6` or `t <= 0`.
#[wasm_bindgen]
pub struct ResolventJump2DV8 {
    kernel: ResolventJumpChernoff2D<f64>,
}

#[wasm_bindgen]
impl ResolventJump2DV8 {
    /// Construct a 2D resolvent-jump evaluator for unit-diffusion heat.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — x-axis bounds (finite, xmin < xmax).
    /// - `nx` — x-axis grid nodes (>= 4).
    /// - `ymin`, `ymax` — y-axis bounds (finite, ymin < ymax).
    /// - `ny` — y-axis grid nodes (>= 4).
    /// - `mNodes` — TWS contour node count (>= 6).
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        m_nodes: usize,
    ) -> Result<ResolventJump2DV8, JsValue> {
        let kernel =
            build_kernel_2d(xmin, xmax, nx, ymin, ymax, ny, m_nodes).map_err(|e| err_to_js(&e))?;
        Ok(ResolventJump2DV8 { kernel })
    }

    /// Evaluate `e^{tA}g`; return result as `Float64Array` (axis-0-fastest).
    ///
    /// ## Parameters
    /// - `t` — time step (`> 0`, finite).
    /// - `g` — `Float64Array` of length `nx*ny`, axis-0-fastest.
    ///
    /// ## Errors
    /// - `.kind = "GridMismatch"` — `g.length != nx*ny`.
    /// - `.kind = "OutOfDomain"` — `t <= 0` or non-finite.
    pub fn jump(&self, t: f64, g: &js_sys::Float64Array) -> Result<js_sys::Float64Array, JsValue> {
        if !t.is_finite() || t <= 0.0 {
            return Err(make_js_error("OutOfDomain", "t must be finite and > 0"));
        }
        let n = self.kernel.grid.len();
        if g.length() as usize != n {
            return Err(make_js_error("GridMismatch", "g.length must equal nx*ny"));
        }
        let mut g_buf = vec![0.0f64; n];
        g.copy_to(&mut g_buf);
        let vals = run_jump_2d(self.kernel.grid, &g_buf, self.kernel.m_nodes, t)
            .map_err(|e| err_to_js(&e))?;
        let out = js_sys::Float64Array::new_with_length(n as u32);
        out.copy_from(&vals);
        Ok(out)
    }

    /// Return `[nx, ny]` shape array.
    #[must_use]
    pub fn shape(&self) -> js_sys::Array {
        let arr = js_sys::Array::new();
        arr.push(&JsValue::from(self.kernel.grid.x.n as u32));
        arr.push(&JsValue::from(self.kernel.grid.y.n as u32));
        arr
    }

    /// Return the number of TWS contour nodes.
    #[wasm_bindgen(js_name = "mNodes")]
    #[must_use]
    pub fn m_nodes(&self) -> usize {
        self.kernel.m_nodes
    }
}

// ---------------------------------------------------------------------------
// ResolventJump3DV8 WASM class
// ---------------------------------------------------------------------------

/// v8.3.0 Resolvent time-jump evaluator for 3D unit-diffusion heat (ADR-0153).
///
/// Evaluates `e^{tA}g` via the TWS parabolic-contour inverse Laplace
/// quadrature (math.md §47.8). Suitable for large `t`.
///
/// **NARROW scope**: self-adjoint / sectorial parabolic generators only.
///
/// ## ND layout (NORMATIVE)
///
/// `g` must be a `Float64Array` with **axis-0-fastest** element order:
/// `g[k*nx*ny + j*nx + i]` is the value at grid point `(i, j, k)`.
/// Array length: `nx*ny*nz`.
///
/// ## Error model (`.kind` discriminator)
///
/// - `"GridMismatch"` — invalid geometry or `g.length != nx*ny*nz`.
/// - `"OutOfDomain"`  — `mNodes < 6` or `t <= 0`.
#[wasm_bindgen]
pub struct ResolventJump3DV8 {
    kernel: ResolventJumpChernoff3D<f64>,
}

#[wasm_bindgen]
impl ResolventJump3DV8 {
    /// Construct a 3D resolvent-jump evaluator for unit-diffusion heat.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax`, `nx` — x-axis bounds + node count (>= 4).
    /// - `ymin`, `ymax`, `ny` — y-axis bounds + node count (>= 4).
    /// - `zmin`, `zmax`, `nz` — z-axis bounds + node count (>= 4).
    /// - `mNodes` — TWS contour node count (>= 6).
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
        m_nodes: usize,
    ) -> Result<ResolventJump3DV8, JsValue> {
        let kernel = build_kernel_3d(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, m_nodes)
            .map_err(|e| err_to_js(&e))?;
        Ok(ResolventJump3DV8 { kernel })
    }

    /// Evaluate `e^{tA}g`; return result as `Float64Array` (axis-0-fastest).
    ///
    /// ## Parameters
    /// - `t` — time step (`> 0`, finite).
    /// - `g` — `Float64Array` of length `nx*ny*nz`, axis-0-fastest.
    ///
    /// ## Errors
    /// - `.kind = "GridMismatch"` — `g.length != nx*ny*nz`.
    /// - `.kind = "OutOfDomain"` — `t <= 0` or non-finite.
    pub fn jump(&self, t: f64, g: &js_sys::Float64Array) -> Result<js_sys::Float64Array, JsValue> {
        if !t.is_finite() || t <= 0.0 {
            return Err(make_js_error("OutOfDomain", "t must be finite and > 0"));
        }
        let n = self.kernel.grid.len();
        if g.length() as usize != n {
            return Err(make_js_error(
                "GridMismatch",
                "g.length must equal nx*ny*nz",
            ));
        }
        let mut g_buf = vec![0.0f64; n];
        g.copy_to(&mut g_buf);
        let vals = run_jump_3d(self.kernel.grid, &g_buf, self.kernel.m_nodes, t)
            .map_err(|e| err_to_js(&e))?;
        let out = js_sys::Float64Array::new_with_length(n as u32);
        out.copy_from(&vals);
        Ok(out)
    }

    /// Return `[nx, ny, nz]` shape array.
    #[must_use]
    pub fn shape(&self) -> js_sys::Array {
        let arr = js_sys::Array::new();
        arr.push(&JsValue::from(self.kernel.grid.x.n as u32));
        arr.push(&JsValue::from(self.kernel.grid.y.n as u32));
        arr.push(&JsValue::from(self.kernel.grid.z.n as u32));
        arr
    }

    /// Return the number of TWS contour nodes.
    #[wasm_bindgen(js_name = "mNodes")]
    #[must_use]
    pub fn m_nodes(&self) -> usize {
        self.kernel.m_nodes
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust contour solves (per-crate dup, ADR-0028 Amdt 2)
// ---------------------------------------------------------------------------

fn run_jump_2d(
    grid: Grid2D<f64>,
    g_vals: &[f64],
    m_nodes: usize,
    t: f64,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let kernel = ResolventJumpChernoff2D::new(grid, m_nodes)?;
    let g = GridFn2D {
        values: g_vals.to_vec(),
        grid,
    };
    let result = kernel.jump(t, &g)?;
    Ok(result.values)
}

fn run_jump_3d(
    grid: Grid3D<f64>,
    g_vals: &[f64],
    m_nodes: usize,
    t: f64,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let kernel = ResolventJumpChernoff3D::new(grid, m_nodes)?;
    let g = GridFn3D {
        values: g_vals.to_vec(),
        grid,
    };
    let result = kernel.jump(t, &g)?;
    Ok(result.values)
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

fn build_kernel_2d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    m_nodes: usize,
) -> Result<ResolventJumpChernoff2D<f64>, semiflow_core::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    let grid = Grid2D::new(gx, gy);
    ResolventJumpChernoff2D::new(grid, m_nodes)
}

fn build_kernel_3d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    zmin: f64,
    zmax: f64,
    nz: usize,
    m_nodes: usize,
) -> Result<ResolventJumpChernoff3D<f64>, semiflow_core::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?;
    let gy = Grid1D::new(ymin, ymax, ny)?;
    let gz = Grid1D::new(zmin, zmax, nz)?;
    let grid = Grid3D::new(gx, gy, gz)?;
    ResolventJumpChernoff3D::new(grid, m_nodes)
}
