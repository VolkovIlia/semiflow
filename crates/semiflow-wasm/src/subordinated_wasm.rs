//! Subordinated heat semigroup — WebAssembly binding (`full` feature).
//!
//! | JS class         | Core type                                               | Python mirror    |
//! |------------------|---------------------------------------------------------|------------------|
//! | `Subordinated1D` | `SubordinatedChernoff<DiffusionChernoff<f64>, Sub, f64>` | `Subordinated1D` |
//!
//! ## Design
//!
//! Mirrors `semiflow-py` `Subordinated1D` (M12, `subordinated_py.rs`).
//! Lévy subordinator backend selected by a `u32` tag:
//! - `0` = `"stable"` (default)
//! - `1` = `"gamma"`
//! - `2` = `"inverse_gaussian"`
//!
//! ## Error model
//!
//! Same `.kind`-tagged JS `Error` as `Heat1D` — see crate-level docs.
//! `panic = "abort"` (ADR-0028 Amendment 1): no `catch_unwind`.

#![allow(unsafe_code)]

use js_sys::Float64Array;
use semiflow::{
    diffusion::DiffusionChernoff,
    subordinated::{
        GammaSubordinator, InverseGaussianSubordinator, LevySubordinator, StableSubordinator,
        SubordinatedChernoff,
    },
    ChernoffSemigroup, Grid1D, GridFn1D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// ---------------------------------------------------------------------------
// Coefficient stubs
// ---------------------------------------------------------------------------

extern "Rust" fn unit_a_sub(_: f64) -> f64 { 1.0 }
extern "Rust" fn zero_sub(_: f64) -> f64 { 0.0 }

// ---------------------------------------------------------------------------
// SubordinatorEnum — binding-side dispatch enum (mirrors semiflow-py)
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum SubordinatorEnum {
    Stable(StableSubordinator<f64>),
    Gamma(GammaSubordinator<f64>),
    InverseGaussian(InverseGaussianSubordinator<f64>),
}

impl LevySubordinator<f64> for SubordinatorEnum {
    fn laplace_exponent(&self, lambda: f64) -> f64 {
        match self {
            SubordinatorEnum::Stable(s) => s.laplace_exponent(lambda),
            SubordinatorEnum::Gamma(s) => s.laplace_exponent(lambda),
            SubordinatorEnum::InverseGaussian(s) => s.laplace_exponent(lambda),
        }
    }

    fn quadrature(&self, tau: f64, n_nodes: usize) -> (Vec<f64>, Vec<f64>) {
        match self {
            SubordinatorEnum::Stable(s) => s.quadrature(tau, n_nodes),
            SubordinatorEnum::Gamma(s) => s.quadrature(tau, n_nodes),
            SubordinatorEnum::InverseGaussian(s) => s.quadrature(tau, n_nodes),
        }
    }
}

// Type alias
type SubKernel = SubordinatedChernoff<DiffusionChernoff<f64>, SubordinatorEnum, f64>;

// ---------------------------------------------------------------------------
// Backend selection (u32 tag: 0=stable, 1=gamma, 2=inverse_gaussian)
// ---------------------------------------------------------------------------

fn parse_sub(tag: u32, alpha: f64, c: f64) -> Result<SubordinatorEnum, JsValue> {
    match tag {
        0 => StableSubordinator::new(alpha)
            .map(SubordinatorEnum::Stable)
            .map_err(|e| err_to_js(&e)),
        1 => GammaSubordinator::new(c)
            .map(SubordinatorEnum::Gamma)
            .map_err(|e| err_to_js(&e)),
        2 => InverseGaussianSubordinator::new(c)
            .map(SubordinatorEnum::InverseGaussian)
            .map_err(|e| err_to_js(&e)),
        _ => Err(make_js_error(
            "Unsupported",
            "subordinator tag must be 0 (stable), 1 (gamma), or 2 (inverse_gaussian)",
        )),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn extract_u0_sub(u0: &Float64Array, n: usize) -> Result<Vec<f64>, JsValue> {
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

fn validate_evolve_sub(t: f64, n_steps: usize) -> Result<(), JsValue> {
    if n_steps == 0 {
        return Err(make_js_error("OutOfDomain", "n_steps must be >= 1"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(make_js_error("OutOfDomain", "t must be finite and >= 0"));
    }
    Ok(())
}

fn vec_to_js_sub(values: &[f64]) -> Float64Array {
    #[allow(clippy::cast_possible_truncation)]
    let arr = Float64Array::new_with_length(values.len() as u32);
    arr.copy_from(values);
    arr
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fn build_subordinated_wasm(
    xmin: f64,
    xmax: f64,
    n: usize,
    u0: &[f64],
    sub: SubordinatorEnum,
    n_nodes: usize,
) -> Result<(SubKernel, GridFn1D<f64>), semiflow::SemiflowError> {
    let grid = Grid1D::new(xmin, xmax, n)?;
    let diff = DiffusionChernoff::new(unit_a_sub, zero_sub, zero_sub, 1.0, grid);
    let kernel = SubordinatedChernoff::with_n_nodes(diff, sub, n_nodes)?;
    let current = GridFn1D::new(grid, u0.to_vec())?;
    Ok((kernel, current))
}


// ---------------------------------------------------------------------------
// Subordinated1D — JS class
// ---------------------------------------------------------------------------

/// 1-D subordinated heat semigroup (M12).
///
/// Backed by `SubordinatedChernoff<DiffusionChernoff<f64>, Subordinator, f64>`
/// (Butko 2018 Thm 2.1, math §37, ADR-0103).
///
/// ## Subordinator tag
/// - `0` = stable α-stable (`alpha ∈ (0,1)`).
/// - `1` = gamma (`c > 0`).
/// - `2` = `inverse_gaussian` (`c > 0`).
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct Subordinated1D {
    kernel: SubKernel,
    current: GridFn1D<f64>,
}

#[wasm_bindgen]
impl Subordinated1D {
    /// Construct `Subordinated1D`.
    ///
    /// - `xmin`, `xmax` — domain bounds.
    /// - `n` — grid nodes (≥ 4).
    /// - `u0` — `Float64Array` of length `n`.
    /// - `subordinator_tag` — `0`=stable, `1`=gamma, `2`=`inverse_gaussian`.
    /// - `alpha` — stability index for stable (default `0.5`; ignored otherwise).
    /// - `c` — rate parameter for gamma/`inverse_gaussian` (default `1.0`).
    /// - `n_nodes` — GL quadrature nodes (1–32, default 32).
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        n: usize,
        u0: &Float64Array,
        subordinator_tag: u32,
        alpha: f64,
        c: f64,
        n_nodes: usize,
    ) -> Result<Subordinated1D, JsValue> {
        let buf = extract_u0_sub(u0, n)?;
        let sub = parse_sub(subordinator_tag, alpha, c)?;
        let (kernel, current) =
            build_subordinated_wasm(xmin, xmax, n, &buf, sub, n_nodes).map_err(|e| err_to_js(&e))?;
        Ok(Subordinated1D { kernel, current })
    }

    /// Advance state by time `t` using `n_steps` Chernoff steps.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(&mut self, t: f64, n_steps: usize) -> Result<(), JsValue> {
        validate_evolve_sub(t, n_steps)?;
        let sg = ChernoffSemigroup::new(self.kernel.clone(), n_steps).map_err(|e| err_to_js(&e))?;
        let next = sg.evolve(t, &self.current).map_err(|e| err_to_js(&e))?;
        self.current = next;
        Ok(())
    }

    /// Return current grid values as `Float64Array` (copy).
    #[must_use]
    pub fn values(&self) -> Float64Array {
        vec_to_js_sub(&self.current.values)
    }

    /// Approximation order (always 1).
    #[must_use]
    pub fn order(&self) -> u32 {
        1
    }

    /// Number of grid nodes `n`.
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.current.values.len()
    }
}
