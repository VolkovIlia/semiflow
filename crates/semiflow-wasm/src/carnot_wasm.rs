//! WASM binding for `ComplexTripleJump` (`full` feature, Round 11).
//!
//! | JS class              | Core type            | Python mirror          |
//! |-----------------------|----------------------|------------------------|
//! | `ComplexTripleJumpV8` | `ComplexTripleJump`  | `ComplexTripleJumpV8`  |
//!
//! Order-4 complex-time triple-jump over the filiform-N5 step-4 Carnot Strang.
//! D=5 ONLY. Complex substeps are internal; only `Re(Ψ(τ)f)` is exposed
//! (ABI-safety, ADR-0138).
//!
//! ## Buffer convention
//!
//! Flat `Float64Array` of length `n_per_axis^5` — real grid values.
//!
//! ## Error model
//!
//! `.kind`-tagged JS `Error`. `panic = "abort"` (ADR-0028 Amendment 1).

#![allow(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use semiflow_core::{
    grid_nd::{GridFnND, GridND},
    ComplexTripleJump, Grid1D,
};
use wasm_bindgen::prelude::*;

use crate::error::{err_to_js, make_js_error};

// D=5 (filiform-N5 fixed)
const D: usize = 5;

// ---------------------------------------------------------------------------
// ComplexTripleJumpV8
// ---------------------------------------------------------------------------

/// Order-4 complex triple-jump, filiform-N5 Carnot, D=5 only (ADR-0138).
///
/// `Ψ(τ) = K(γ⋆τ) ∘ K((1−2γ⋆)τ) ∘ K(γ⋆τ)` where K is the filiform-N5
/// palindromic Strang. Complex substeps are internal; only `Re(Ψ(τ)f)` is
/// exposed. Mirrors Python `ComplexTripleJumpV8`.
///
/// ## Parameters (constructor)
/// - `domain_lo` — lower bound of each axis (same for all 5 axes, finite).
/// - `domain_hi` — upper bound (`> domain_lo`).
/// - `n_per_axis` — grid nodes per axis (`>= 4`).
///
/// # Errors
/// - `.kind = "GridMismatch"` — invalid domain or `n_per_axis < 4`.
/// - `.kind = "OutOfDomain"` — non-finite domain bound or γ⋆ check fails.
/// - `.kind = "GridMismatch"` — `u0.length != n_per_axis^5` in `applyReal`.
/// - `.kind = "OutOfDomain"` — `tau < 0` or non-finite in `applyReal`.
#[wasm_bindgen(js_name = "ComplexTripleJumpV8")]
pub struct ComplexTripleJumpWasm {
    domain_lo: f64,
    domain_hi: f64,
    n_per_axis: usize,
}

#[wasm_bindgen(js_class = "ComplexTripleJumpV8")]
impl ComplexTripleJumpWasm {
    /// Construct the order-4 complex triple-jump state.
    ///
    /// # Errors
    /// See struct-level error table.
    #[wasm_bindgen(constructor)]
    pub fn new(
        domain_lo: f64,
        domain_hi: f64,
        n_per_axis: usize,
    ) -> Result<ComplexTripleJumpWasm, JsValue> {
        validate_domain(domain_lo, domain_hi, n_per_axis)?;
        // Eagerly construct to catch γ⋆ / inner kernel errors.
        ComplexTripleJump::new().map_err(|e| err_to_js(&e))?;
        Ok(Self { domain_lo, domain_hi, n_per_axis })
    }

    /// Apply one order-4 step; return the real projection `Re(Ψ(τ)f)`.
    ///
    /// ## Parameters
    /// - `tau` — step size (`>= 0`, finite).
    /// - `u0`  — flat 5-D grid function, length `n_per_axis^5`.
    ///
    /// Returns `Float64Array` of same length as `u0`.
    ///
    /// # Errors
    /// - `.kind = "OutOfDomain"` — `tau < 0` or non-finite.
    /// - `.kind = "GridMismatch"` — `u0.length != n_per_axis^5`.
    /// - `.kind = "NanInf"` — `u0` contains NaN or Inf.
    #[wasm_bindgen(js_name = "applyReal")]
    pub fn apply_real(&self, tau: f64, u0: &[f64]) -> Result<Vec<f64>, JsValue> {
        if !tau.is_finite() || tau < 0.0 {
            return Err(make_js_error("OutOfDomain", "tau must be finite and >= 0"));
        }
        let expected = self.n_per_axis.pow(D as u32);
        if u0.len() != expected {
            return Err(make_js_error(
                "GridMismatch",
                "u0.length must equal n_per_axis^5",
            ));
        }
        for &v in u0 {
            if !v.is_finite() {
                return Err(make_js_error("NanInf", "u0 contains NaN or Inf"));
            }
        }
        let grid = build_grid(self.domain_lo, self.domain_hi, self.n_per_axis)
            .map_err(|e| err_to_js(&e))?;
        let src = GridFnND::new(grid, u0.to_vec()).map_err(|e| err_to_js(&e))?;
        let kernel = ComplexTripleJump::new().map_err(|e| err_to_js(&e))?;
        let out = kernel.apply_real(tau, &src).map_err(|e| err_to_js(&e))?;
        Ok(out.values)
    }

    /// Verify γ⋆ satisfies `2γ³+(1−2γ)³=0` with Re > 0 and residual < 1e-12.
    ///
    /// Returns `true` iff the check passes.
    #[must_use]
    #[wasm_bindgen(js_name = "verifyGammaStar")]
    pub fn verify_gamma_star() -> bool {
        ComplexTripleJump::verify_gamma_star()
    }

    /// Total number of grid points (`n_per_axis^5`).
    #[must_use]
    pub fn size(&self) -> u32 {
        self.n_per_axis.pow(D as u32) as u32
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_grid(
    lo: f64,
    hi: f64,
    n: usize,
) -> Result<GridND<f64, D>, semiflow_core::SemiflowError> {
    let ax = Grid1D::new(lo, hi, n)?;
    GridND::new([ax; D])
}

fn validate_domain(lo: f64, hi: f64, n: usize) -> Result<(), JsValue> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(make_js_error("OutOfDomain", "domain bounds must be finite"));
    }
    if lo >= hi {
        return Err(make_js_error("GridMismatch", "domain_lo must be < domain_hi"));
    }
    if n < 4 {
        return Err(make_js_error("GridMismatch", "n_per_axis must be >= 4"));
    }
    Ok(())
}
