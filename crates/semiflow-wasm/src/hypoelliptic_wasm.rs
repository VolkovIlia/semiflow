//! Hypoelliptic / sub-Riemannian Chernoff engines — WebAssembly binding (`full` feature).
//!
//! Mirrors the Python binding in `semiflow-py` (Round 8).
//!
//! | JS class                          | Core type                           | Python mirror                    |
//! |-----------------------------------|-------------------------------------|----------------------------------|
//! | `HypoellipticChernoffHeisenberg`  | `HypoellipticChernoff<f64, 3, 2>`   | `HypoellipticChernoffHeisenberg` |
//! | `HypoellipticChernoffKolmogorov`  | `KolmogorovHypoelliptic<f64>`       | `HypoellipticChernoffKolmogorov` |
//! | `HypoellipticChernoffEngel`       | `HypoellipticChernoff<f64, 4, 2>`   | `HypoellipticChernoffEngel`      |
//!
//! ## Buffer layouts
//!
//! - Kolmogorov: flat row-major `Float64Array`, length `nx * nv`.
//!   `values[j*nx + i] ≈ u(x_i, v_j)`.
//! - Engel: flat axis-0-fastest `Float64Array`, length `n**4`.
//!   All 4 axes share `n` nodes over `[xmin, xmax]`.
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use js_sys::Float64Array;
use semiflow::{
    heisenberg_heat_kernel,
    hormander::{HypoellipticChernoff, KolmogorovPhaseSpace, VectorField},
    ChernoffFunction, Grid1D, Grid2D, GridFn2D, GridFnND, GridND, ScratchPool, SemiflowError,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Shared helpers (mirrors strang_nd_wasm.rs / nonsep_wasm.rs)
// ---------------------------------------------------------------------------

/// Validate `tau > 0` (finite) and `n_steps >= 1`.
fn validate_tau_steps(tau: f64, n_steps: usize) -> Result<(), JsValue> {
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
    let mut buf = vec![0.0_f64; expected];
    u0.copy_to(&mut buf);
    for &v in &buf {
        if !v.is_finite() {
            return Err(make_js_error("NanInf", "u0 contains NaN or Inf"));
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

// ---------------------------------------------------------------------------
// HypoellipticChernoffHeisenberg
// ---------------------------------------------------------------------------

/// Heisenberg group H₁ Chernoff approximation (palindromic Strang-Hörmander).
///
/// Mirrors `HypoellipticChernoffHeisenberg` from `semiflow-py`.
/// Wraps `HypoellipticChernoff<f64, 3, 2>` via `new_heisenberg()`.
///
/// Provides `kernel(h, x, y, tc)` — the Gaveau-Hulanicki heat kernel oracle
/// (32-pt Gauss-Legendre quadrature).  No `evolve` method: the Python binding
/// does not expose `evolve` for this class either.
///
/// # Errors
/// Constructor throws JS `Error` with `.kind = "OutOfDomain"` if the
/// Hörmander bracket condition fails at the origin (should never occur with
/// canonical HeisenbergX/HeisenbergY fields).
#[wasm_bindgen]
pub struct HypoellipticChernoffHeisenberg {
    inner: HypoellipticChernoff<f64, 3, 2>,
}

#[wasm_bindgen]
impl HypoellipticChernoffHeisenberg {
    /// Construct the Heisenberg Chernoff kernel.
    ///
    /// Verifies step-2 Carnot bracket `[X₁, X₂] = ∂_t` at the origin.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind = "OutOfDomain"` on bracket failure.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<HypoellipticChernoffHeisenberg, JsValue> {
        let inner = HypoellipticChernoff::<f64, 3, 2>::new_heisenberg()
            .map_err(|e| err_to_js(&e))?;
        Ok(HypoellipticChernoffHeisenberg { inner })
    }

    /// Approximation order (always 2 — palindromic Strang-Hörmander).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn order(&self) -> u32 {
        self.inner.order()
    }

    /// Heisenberg heat kernel oracle `p_h(x, y, tc)`.
    ///
    /// Delegates to core `heisenberg_heat_kernel(h, x, y, tc)`
    /// (Gaveau-Hulanicki, 32-pt Gauss-Legendre quadrature).
    /// Returns `0.0` for `h <= 0`.
    #[must_use]
    pub fn kernel(&self, h: f64, x: f64, y: f64, tc: f64) -> f64 {
        heisenberg_heat_kernel(h, x, y, tc)
    }
}

// ---------------------------------------------------------------------------
// HypoellipticChernoffKolmogorov
// ---------------------------------------------------------------------------

/// Kolmogorov phase-space hypoelliptic Chernoff approximation.
///
/// Mirrors `HypoellipticChernoffKolmogorov` from `semiflow-py`.
/// Wraps `KolmogorovHypoelliptic<f64>` (= `HypoellipticChernoff<f64, 2, 1>`)
/// with the palindromic Strang-Hörmander step:
/// `F(τ) = exp(τX₀/2) ∘ exp(τ/2·∂²_v) ∘ exp(τX₀/2)`.
///
/// Buffer layout: flat row-major `Float64Array`, length `nx * nv`.
/// `values[j*nx + i] ≈ u(x_i, v_j)`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct HypoellipticChernoffKolmogorov {
    grid: Grid2D<f64>,
    nx: usize,
    nv: usize,
}

#[wasm_bindgen]
impl HypoellipticChernoffKolmogorov {
    /// Construct Kolmogorov Chernoff on a 2D phase-space grid.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — x-axis bounds (finite, `xmin < xmax`).
    /// - `nx` — x-axis nodes (≥ 4).
    /// - `vmin`, `vmax` — velocity-axis bounds (finite, `vmin < vmax`).
    /// - `nv` — velocity-axis nodes (≥ 4).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        vmin: f64,
        vmax: f64,
        nv: usize,
    ) -> Result<HypoellipticChernoffKolmogorov, JsValue> {
        let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
        let gv = Grid1D::new(vmin, vmax, nv).map_err(|e| err_to_js(&e))?;
        let grid = Grid2D::new(gx, gv);
        // Verify Hörmander bracket condition at construction (mirrors Python).
        make_kolmogorov_kernel().map_err(|e| err_to_js(&e))?;
        Ok(HypoellipticChernoffKolmogorov { grid, nx, nv })
    }

    /// Approximation order (always 2 — palindromic Strang-Hörmander).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn order(&self) -> u32 {
        2
    }

    /// Evolve flat row-major `u0` (length `nx * nv`) by `n_steps` of size `tau`.
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
        validate_tau_steps(tau, n_steps)?;
        let input = extract_flat(u0, self.nx * self.nv)?;
        let result = run_kolmogorov(self.grid, input, tau, n_steps)
            .map_err(|e| err_to_js(&e))?;
        Ok(vec_to_js(&result))
    }

    /// X-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Velocity-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nv(&self) -> usize {
        self.nv
    }

    /// Total grid nodes (`nx * nv`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.nx * self.nv
    }
}

// ---------------------------------------------------------------------------
// HypoellipticChernoffEngel
// ---------------------------------------------------------------------------

/// Engel step-3 Carnot group hypoelliptic Chernoff approximation.
///
/// Mirrors `HypoellipticChernoffEngel` from `semiflow-py`.
/// Wraps `HypoellipticChernoff<f64, 4, 2>` via `new_engel()`.
///
/// Buffer layout: flat axis-0-fastest `Float64Array`, length `n**4`.
/// All 4 axes share `n` nodes over `[xmin, xmax]`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct HypoellipticChernoffEngel {
    grid: GridND<f64, 4>,
    n: usize,
}

#[wasm_bindgen]
impl HypoellipticChernoffEngel {
    /// Construct Engel Chernoff on a uniform 4D grid.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — common bounds for all 4 axes (finite, `xmin < xmax`).
    /// - `n` — per-axis node count (≥ 4). All 4 axes share `n`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
    ) -> Result<HypoellipticChernoffEngel, JsValue> {
        let ax = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let grid = GridND::<f64, 4>::new([ax, ax, ax, ax]).map_err(|e| err_to_js(&e))?;
        // Verify Engel bracket condition at construction (mirrors Python).
        HypoellipticChernoff::<f64, 4, 2>::new_engel().map_err(|e| err_to_js(&e))?;
        Ok(HypoellipticChernoffEngel { grid, n })
    }

    /// Approximation order (always 2 — palindromic Strang-Hörmander).
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn order(&self) -> u32 {
        2
    }

    /// Evolve flat axis-0-fastest `u0` (length `n**4`) by `n_steps` of size `tau`.
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
        validate_tau_steps(tau, n_steps)?;
        let total = self.n * self.n * self.n * self.n;
        let input = extract_flat(u0, total)?;
        let result = run_engel(self.grid.clone(), input, tau, n_steps)
            .map_err(|e| err_to_js(&e))?;
        Ok(vec_to_js(&result))
    }

    /// Per-axis node count `n`.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Total grid nodes (`n**4`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.n * self.n * self.n * self.n
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust compute helpers (no wasm-bindgen, easier to unit-test)
// ---------------------------------------------------------------------------

/// Build the canonical Kolmogorov kernel (cheap bracket check at origin).
fn make_kolmogorov_kernel(
) -> Result<HypoellipticChernoff<f64, 2, 1>, SemiflowError> {
    let x0: Box<dyn VectorField<f64, 2>> =
        Box::new(KolmogorovPhaseSpace::<f64>::x0_drift());
    let x1: Box<dyn VectorField<f64, 2>> =
        Box::new(KolmogorovPhaseSpace::<f64>::x1_diffusion());
    HypoellipticChernoff::<f64, 2, 1>::new(x0, [x1])
}

fn run_kolmogorov(
    grid: Grid2D<f64>,
    values: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, SemiflowError> {
    let kernel = make_kolmogorov_kernel()?;
    let mut src = GridFn2D::new(grid, values)?;
    let mut dst = GridFn2D::new(grid, vec![0.0_f64; src.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

fn run_engel(
    grid: GridND<f64, 4>,
    values: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, SemiflowError> {
    let kernel = HypoellipticChernoff::<f64, 4, 2>::new_engel()?;
    let total = values.len();
    let mut src = GridFnND::new(grid.clone(), values)?;
    let mut dst = GridFnND::new(grid, vec![0.0_f64; total])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}
