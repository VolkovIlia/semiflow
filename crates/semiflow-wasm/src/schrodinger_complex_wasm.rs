//! Native-complex Schrödinger WASM binding (`full` feature).
//!
//! Exposes one JS class:
//!
//! | JS class              | Core type                              | Python mirror        |
//! |-----------------------|----------------------------------------|----------------------|
//! | `SchrodingerComplex1D`| `SchrödingerChernoffComplex<C64>`      | `SchrodingerComplex1D`|
//!
//! ## Kernel
//!
//! `SchrödingerChernoffComplex` (ADR-0079 Option B, math §30.3): palindromic
//! Strang with Cayley–Crank-Nicolson kinetic step.  Globally order 2; exactly
//! unitary.  Operates natively in ℂ (no real/imaginary split overhead).
//!
//! ## Complex buffer convention
//!
//! Interleaved `Float64Array` of length `2*n`: `[re₀, im₀, re₁, im₁, …]`.
//! This matches `schrodinger_wasm.rs` and the FFI Round-4 convention.
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]

use std::sync::Arc;

use js_sys::Float64Array;
use num_complex::Complex;
use semiflow::{ChernoffSemigroup, Grid1D, GridFnComplex1D, SchrödingerChernoffComplex};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Type alias
// ---------------------------------------------------------------------------

type C64 = Complex<f64>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract interleaved `[re0, im0, …]` from a `Float64Array` of length `2*n`.
fn extract_psi_cx(psi: &Float64Array, n: usize) -> Result<Vec<C64>, JsValue> {
    if psi.length() as usize != 2 * n {
        return Err(make_js_error(
            "GridMismatch",
            "psi0.length() must equal 2*n (interleaved re/im)",
        ));
    }
    let mut buf = vec![0.0f64; 2 * n];
    psi.copy_to(&mut buf);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = buf[2 * i];
        let c = buf[2 * i + 1];
        if !r.is_finite() || !c.is_finite() {
            return Err(make_js_error("NanInf", "psi0 contains NaN or Inf"));
        }
        out.push(C64::new(r, c));
    }
    Ok(out)
}

/// Extract a `Float64Array` as `Vec<f64>`, validating length and finiteness.
fn extract_v(v: &Float64Array, n: usize) -> Result<Vec<f64>, JsValue> {
    if v.length() as usize != n {
        return Err(make_js_error("GridMismatch", "v.length() must equal n"));
    }
    let mut buf = vec![0.0f64; n];
    v.copy_to(&mut buf);
    for &vi in &buf {
        if !vi.is_finite() {
            return Err(make_js_error("NanInf", "v contains NaN or Inf"));
        }
    }
    Ok(buf)
}

/// Pack `Vec<C64>` → interleaved `Float64Array` of length `2*n`.
#[allow(clippy::cast_possible_truncation)]
fn pack_cx(vals: &[C64]) -> Float64Array {
    let n = vals.len();
    let arr = Float64Array::new_with_length((2 * n) as u32);
    let mut buf = vec![0.0f64; 2 * n];
    for (i, z) in vals.iter().enumerate() {
        buf[2 * i] = z.re;
        buf[2 * i + 1] = z.im;
    }
    arr.copy_from(&buf);
    arr
}

/// Validate evolve parameters: `n_steps >= 1`, `t >= 0` and finite.
fn validate_evolve_cx(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

/// Build kernel from stored potential (called each `evolve`).
fn build_cx_kernel(
    v_at_node: &[f64],
    grid: Grid1D<f64>,
) -> Result<SchrödingerChernoffComplex<C64>, semiflow::SemiflowError> {
    let v = Arc::new(v_at_node.to_vec());
    let v2 = v.clone();
    let dx = grid.dx();
    let xmin = grid.xmin;
    SchrödingerChernoffComplex::<C64>::new(grid, move |x: f64| {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let idx = ((x - xmin) / dx).round() as usize;
        v2[idx.min(v2.len().saturating_sub(1))]
    })
}

/// Compute dx = (xmax − xmin) / (n − 1).
#[allow(clippy::cast_precision_loss)]
fn compute_dx_cx(xmin: f64, xmax: f64, n: usize) -> f64 {
    if n > 1 {
        (xmax - xmin) / (n as f64 - 1.0)
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// SchrodingerComplex1D
// ---------------------------------------------------------------------------

/// 1-D Schrödinger equation with native complex state: `iψ_t = (−½Δ + V)ψ`.
///
/// Backed by `SchrödingerChernoffComplex` (ADR-0079 Option B, math §30.3):
/// palindromic Strang splitting with Cayley–Crank-Nicolson kinetic step.
/// Globally order 2; exactly unitary.
///
/// Unlike `Schrodinger1D` (real-pair split), this class stores `ψ` as a native
/// complex vector and reconstructs the kernel from the pre-sampled potential
/// on each `evolve` call.
///
/// ## Buffer convention
///
/// `psi0` and `values()` use an interleaved `Float64Array` of length `2*n`:
/// `[re₀, im₀, re₁, im₁, …, reₙ₋₁, imₙ₋₁]`.
///
/// # Errors
/// Throws JS `Error` with `.kind` property — see crate-level docs.
#[wasm_bindgen]
pub struct SchrodingerComplex1D {
    /// Pre-sampled potential values `V(x_0), …, V(x_{N-1})`.
    v_at_node: Vec<f64>,
    /// Current wavefunction `ψ ∈ ℂⁿ`.
    values_cx: Vec<C64>,
    /// Grid geometry.
    grid: Grid1D<f64>,
    /// Cached grid spacing for `norm_squared`.
    dx: f64,
    /// Number of grid nodes.
    n: usize,
}

#[wasm_bindgen]
impl SchrodingerComplex1D {
    /// Construct free-particle `SchrodingerComplex1D` (V = 0).
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — domain bounds (finite, `xmin < xmax`).
    /// - `n` — grid nodes (≥ 4).
    /// - `psi0` — `Float64Array` of length `2*n`; interleaved `[re₀, im₀, …]`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        psi0: &Float64Array,
    ) -> Result<SchrodingerComplex1D, JsValue> {
        let psi = extract_psi_cx(psi0, n)?;
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let v_at_node = vec![0.0_f64; n];
        // Validate kernel construction eagerly.
        build_cx_kernel(&v_at_node, grid).map_err(|e| err_to_js(&e))?;
        GridFnComplex1D::<C64>::new(grid, psi.clone()).map_err(|e| err_to_js(&e))?;
        let dx = compute_dx_cx(xmin, xmax, n);
        Ok(SchrodingerComplex1D {
            v_at_node,
            values_cx: psi,
            grid,
            dx,
            n,
        })
    }

    /// Construct with a pre-sampled real potential `V(x)`.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — domain bounds (finite, `xmin < xmax`).
    /// - `n` — grid nodes (≥ 4).
    /// - `v` — `Float64Array` of length `n`; sampled `V(x_i)`; all finite.
    /// - `psi0` — `Float64Array` of length `2*n`; interleaved `[re₀, im₀, …]`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(js_name = "withPotential")]
    pub fn with_potential(
        xmin: f64,
        xmax: f64,
        n: usize,
        v: &Float64Array,
        psi0: &Float64Array,
    ) -> Result<SchrodingerComplex1D, JsValue> {
        let v_at_node = extract_v(v, n)?;
        let psi = extract_psi_cx(psi0, n)?;
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        build_cx_kernel(&v_at_node, grid).map_err(|e| err_to_js(&e))?;
        GridFnComplex1D::<C64>::new(grid, psi.clone()).map_err(|e| err_to_js(&e))?;
        let dx = compute_dx_cx(xmin, xmax, n);
        Ok(SchrodingerComplex1D {
            v_at_node,
            values_cx: psi,
            grid,
            dx,
            n,
        })
    }

    /// Advance the wavefunction by time `t` using `n_steps` Chernoff steps.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve_cx(t, n_steps)?;
        let kernel = build_cx_kernel(&self.v_at_node, self.grid).map_err(|e| err_to_js(&e))?;
        let sg = ChernoffSemigroup::new(kernel, n_steps).map_err(|e| err_to_js(&e))?;
        let src = GridFnComplex1D::<C64>::new(self.grid, self.values_cx.clone())
            .map_err(|e| err_to_js(&e))?;
        let out = sg.evolve(t, &src).map_err(|e| err_to_js(&e))?;
        self.values_cx = out.values;
        Ok(())
    }

    /// Return current wavefunction as interleaved `Float64Array` of length `2*n`.
    ///
    /// Layout: `[re₀, im₀, re₁, im₁, …]`.
    #[must_use]
    pub fn values(&self) -> Float64Array {
        pack_cx(&self.values_cx)
    }

    /// Number of grid nodes `n`.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.n
    }

    /// Approximation order (always 2 — palindromic Strang).
    #[must_use]
    pub fn order(&self) -> u32 {
        2
    }

    /// Return `Σ |ψᵢ|² · dx` — grid-spacing-weighted squared L2 norm.
    ///
    /// Equals `1.0` for a normalised wavefunction; used to verify unitarity.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        let raw: f64 = self
            .values_cx
            .iter()
            .fold(0.0_f64, |acc, z| acc + z.re * z.re + z.im * z.im);
        raw * self.dx
    }
}
