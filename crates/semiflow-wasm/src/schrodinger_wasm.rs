//! 1-D Schrödinger equation WASM binding (`full` feature).
//!
//! Exposes one JS class:
//!
//! | JS class        | Core type                  | Python mirror   |
//! |-----------------|----------------------------|-----------------|
//! | `Schrodinger1D` | `SchrodingerChernoff<f64>` | `Schrodinger1D` |
//!
//! ## Complex buffer convention
//!
//! Wavefunction `ψ(x)` is complex.  Because `Float64Array` is real, we use an
//! **interleaved** layout: a `Float64Array` of length `2*n` whose entries are
//! `[re₀, im₀, re₁, im₁, …, reₙ₋₁, imₙ₋₁]`.  This matches the FFI Round-4
//! convention used by `semiflow-ffi`.
//!
//! ## Backward evolution (D2 — ADR-0113)
//!
//! `t` may be negative for backward (time-reversed) unitary evolution.
//! The palindromic Strang kernel satisfies `S(−τ) = S(τ)⁻¹` exactly.
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow_core::{
    ChernoffFunction, Diffusion4thChernoff, Grid1D, GridFn1D, SchrodingerChernoff,
    SchrodingerState, ScratchPool,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Coefficient statics (free-particle kinetic term: a ≡ 1, d ≡ 0)
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_s(_: f64) -> f64 {
    1.0
}
extern "Rust" fn zero_d_s(_: f64) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate `n_steps >= 1` and `t.is_finite()` (allows negative t for D2).
fn validate_evolve(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() {
        return Err(make_js_error(
            "OutOfDomain",
            "t must be finite (negative t allowed for backward evolution)",
        ));
    }
    Ok(())
}

/// Extract interleaved `[re0, im0, …]` → `(Vec<f64>, Vec<f64>)`, validate finiteness.
fn extract_psi(psi: &Float64Array, n: usize) -> Result<(Vec<f64>, Vec<f64>), JsValue> {
    if psi.length() as usize != 2 * n {
        return Err(make_js_error(
            "GridMismatch",
            "psi0.length() must equal 2*n (interleaved re/im)",
        ));
    }
    let mut buf = vec![0.0f64; 2 * n];
    psi.copy_to(&mut buf);
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for i in 0..n {
        let r = buf[2 * i];
        let c = buf[2 * i + 1];
        if !r.is_finite() || !c.is_finite() {
            return Err(make_js_error("NanInf", "psi0 contains NaN or Inf"));
        }
        re.push(r);
        im.push(c);
    }
    Ok((re, im))
}

/// Pack `(Vec<f64>, Vec<f64>)` → interleaved `Float64Array` of length `2*n`.
#[allow(clippy::cast_possible_truncation)]
fn pack_psi(re: &[f64], im: &[f64]) -> Float64Array {
    let n = re.len();
    let arr = Float64Array::new_with_length((2 * n) as u32);
    let mut buf = vec![0.0f64; 2 * n];
    for i in 0..n {
        buf[2 * i] = re[i];
        buf[2 * i + 1] = im[i];
    }
    arr.copy_from(&buf);
    arr
}

/// Compute dx = (xmax − xmin) / (n − 1).
#[allow(clippy::cast_precision_loss)]
fn compute_dx(xmin: f64, xmax: f64, n: usize) -> f64 {
    if n > 1 { (xmax - xmin) / (n as f64 - 1.0) } else { 1.0 }
}

// ---------------------------------------------------------------------------
// Schrodinger1D
// ---------------------------------------------------------------------------

/// 1-D Schrödinger equation state: `iψ_t = (−Δ + V(x))ψ` (free particle, V=0).
///
/// Backed by `SchrodingerChernoff<f64>` — palindromic Strang splitting,
/// order 2, unitary by construction (ADR-0057, math §17).
///
/// ## Buffer convention
/// `psi0` and the `values()` return are `Float64Array` of length `2*n`
/// with interleaved layout: `[re₀, im₀, re₁, im₁, …]`.
///
/// ## Negative time
/// `t < 0` performs backward evolution (ADR-0113 D2).
///
/// # Errors
/// Throws JS `Error` with `.kind` property — see crate-level docs.
#[wasm_bindgen]
pub struct Schrodinger1D {
    chernoff: SchrodingerChernoff<f64>,
    re: Vec<f64>,
    im: Vec<f64>,
    grid: Grid1D<f64>,
    dx: f64,
    n: usize,
}

#[wasm_bindgen]
impl Schrodinger1D {
    /// Construct free-particle `Schrodinger1D` (V = 0) from an interleaved complex array.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax` — domain bounds (finite, `xmin < xmax`).
    /// - `n` — grid nodes (≥ 4).
    /// - `psi0` — `Float64Array` of length `2*n`; interleaved `[re₀, im₀, …]`.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(xmin: f64, xmax: f64, n: usize, psi0: &Float64Array) -> Result<Schrodinger1D, JsValue> {
        let (re, im) = extract_psi(psi0, n)?;
        let grid = Grid1D::new(xmin, xmax, n).map_err(|e| err_to_js(&e))?;
        let kinetic = Diffusion4thChernoff::new(unit_a_s, zero_d_s, zero_d_s, 1.0, grid);
        let chernoff = SchrodingerChernoff::new(kinetic, |_| 0.0).map_err(|e| err_to_js(&e))?;
        let dx = compute_dx(xmin, xmax, n);
        Ok(Schrodinger1D { chernoff, re, im, grid, dx, n })
    }

    /// Advance the wavefunction by time `t` using `n_steps` Chernoff steps.
    ///
    /// Negative `t` performs backward (time-reversed) unitary evolution.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind` — see crate-level error table.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve(t, n_steps)?;
        #[allow(clippy::cast_precision_loss)]
        let tau = t / n_steps as f64;
        let psi_re = GridFn1D::new(self.grid, self.re.clone()).map_err(|e| err_to_js(&e))?;
        let psi_im = GridFn1D::new(self.grid, self.im.clone()).map_err(|e| err_to_js(&e))?;
        let mut state = SchrodingerState::new(psi_re, psi_im).map_err(|e| err_to_js(&e))?;
        let mut scratch = ScratchPool::new();
        for _ in 0..n_steps {
            let mut next = state.clone();
            self.chernoff
                .apply_into(tau, &state, &mut next, &mut scratch)
                .map_err(|e| err_to_js(&e))?;
            state = next;
        }
        self.re = state.psi_re.values;
        self.im = state.psi_im.values;
        Ok(())
    }

    /// Return current wavefunction as interleaved `Float64Array` of length `2*n`.
    ///
    /// Layout: `[re₀, im₀, re₁, im₁, …]`.
    #[must_use]
    pub fn values(&self) -> Float64Array {
        pack_psi(&self.re, &self.im)
    }

    /// Number of grid nodes `n`.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.n
    }

    /// Return `Σ |ψᵢ|² · dx` — grid-spacing-weighted squared L2 norm.
    ///
    /// Equals `1.0` for a normalised wavefunction.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        let raw: f64 = self
            .re
            .iter()
            .zip(self.im.iter())
            .fold(0.0_f64, |acc, (&r, &i)| acc + r * r + i * i);
        raw * self.dx
    }
}
