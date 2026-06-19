//! [`TruncatedExp4thDiffusionChernoff`] — truncated-exp K=4 + 4th-order stencil (v0.6.0, ADR-0013).
//!
//! 5-point divergence-form stencil G⁴ (Mickens-style; constant-a collapse is
//! the standard 4th-order central Laplacian, sympy gate M⁴_const-a-fast):
//!
//! ```text
//! G⁴ f|_i = [ −a_{i+3/2}·(f_{i+2}−f_{i+1})/12
//!            + 5·a_{i+1/2}·(f_{i+1}−f_i)/4
//!            − 5·a_{i-1/2}·(f_i−f_{i-1})/4
//!            + a_{i-3/2}·(f_{i-1}−f_{i-2})/12 ] / dx²
//! ```
//!
//! Half-node evaluations: `a_{i±½} = a(x_i ± dx/2)`, `a_{i±3/2} = a(x_i ± 3·dx/2)`.
//!
//! K=4 truncated power series (unchanged from v0.4.0 `TruncatedExpDiffusionChernoff`):
//!
//! ```text
//! M⁴(τ) f ≈ Σ_{k=0..4} (τ^k / k!) · (G⁴)^k f(x_mid)
//! ```
//!
//! CFL (25% tighter than v0.4.0): `τ < 3·dx² / (8·‖a‖_∞)`.
//!
//! Sympy gates (`verify_v0_6_0_magnus4.py`): `M⁴_τ⁰` ✓  `M⁴_τ¹` ✓  `M⁴_τ²` ✓
//!   M⁴_spatial-order ✓  M⁴_const-a-fast ✓.
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 1)
//!
//! `TruncatedExp4thDiffusionChernoff<F: SemiflowFloat = f64>` — the `= f64` default
//! keeps all existing call-sites compiling unchanged. `TruncatedExp4thDiffusionChernoff<f64>`
//! implements `ChernoffFunction`. Other `F` types use `apply_f` (scalar path).

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// Wave B2 cache types live in a sibling module (constitution Override #1, ≤700 lines).
pub use crate::truncated_exp4_cached::{HalfNodeCoeffCache, TruncatedExp4WithCache};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of terms in the truncated power series (K=4): k = 0, 1, 2, 3, 4.
pub(crate) const TRUNC_ORDER_USIZE: usize = 4;

/// Factorial inverses `[1/0!, 1/1!, 1/2!, 1/3!, 1/4!]` for k = 0..=K.
const FACTORIAL_INVERSE: [f64; TRUNC_ORDER_USIZE + 1] = [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0];

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Chernoff function for `A_self f = ∂_x(a(x)·∂_x f)` via truncated-exp K=4 + 4th-order stencil.
///
/// **What this computes**: the truncated Taylor series `Σ_{k=0}^{K} (τᵏ/k!) (G⁴)ᵏ` of the
/// 5-point divergence-form generator G⁴ (NOT the genuine Magnus expansion). Renamed from
/// `Magnus4thDiffusionChernoff` in v0.7.0 per audit finding D2.
///
/// Additive sibling of [`crate::TruncatedExpDiffusionChernoff`]: identical constructor
/// signature; G⁴ delivers O(dx⁴) spatial accuracy vs O(dx²) of the 3-point stencil.
///
/// ## Generic-over-Float (ADR-0025)
///
/// `TruncatedExp4thDiffusionChernoff<F: SemiflowFloat = f64>` — `= f64` keeps all existing
/// call-sites unchanged. `TruncatedExp4thDiffusionChernoff<f64>` implements
/// [`ChernoffFunction`]. Other `F` types use `apply_f`.
///
/// # Caller invariants
/// - `a(x) > 0` everywhere (strict ellipticity).
/// - `a_norm_bound` is a valid upper bound for `‖a‖_∞` (used for CFL check).
/// - CFL: `8·τ·a_norm_bound < 3·dx²`; violated → `Err(CflViolated)`.
///
/// # Example
///
/// ```rust
/// use semiflow_core::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, TruncatedExp4thDiffusionChernoff};
/// let grid = Grid1D::new(-4.0, 4.0, 64).unwrap();
/// // CFL (4th-order tighter): tau < 3*dx²/(8*a) ≈ 0.002 for a=1.0, dx≈0.126
/// let me4 = TruncatedExp4thDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// let u1 = me4.apply_chernoff(0.001, &u0).unwrap();
/// assert_eq!(u1.values.len(), 64);
/// ```
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy)]
pub struct TruncatedExp4thDiffusionChernoff<F: SemiflowFloat = f64> {
    /// Diffusion coefficient `a(x)`. Caller MUST guarantee `a(x) > 0`.
    pub a: fn(F) -> F,
    /// First derivative `a'(x)`. Accepted for API symmetry; NOT used in stencil.
    pub a_prime: fn(F) -> F,
    /// Second derivative `a''(x)`. Accepted for API symmetry; NOT used in stencil.
    pub a_double_prime: fn(F) -> F,
    /// Upper bound for `‖a‖_∞`. Used in the 4th-order CFL stability check.
    pub a_norm_bound: f64,
    /// Reference grid geometry (node iteration and output allocation).
    pub grid: Grid1D<F>,
    /// Optional IP-conjugation drift `b(x)`.
    pub b_for_conjugation: Option<fn(F) -> F>,
}

// ---------------------------------------------------------------------------
// impl TruncatedExp4thDiffusionChernoff<f64> — concrete f64 path
// ---------------------------------------------------------------------------

impl TruncatedExp4thDiffusionChernoff<f64> {
    /// Truncated-series order exposed for tests: `K = 4`.
    #[allow(clippy::cast_possible_truncation)]
    pub const TRUNC_ORDER: u32 = TRUNC_ORDER_USIZE as u32;

    /// CFL numerator: `τ < NUMER·dx² / (DENOM·‖a‖_∞)`.
    pub const CFL_NUMER: u64 = 3;

    /// CFL denominator (see `CFL_NUMER`).
    pub const CFL_DENOM: u64 = 8;

    /// Construct a `TruncatedExp4thDiffusionChernoff` (v0.6.0, 5-arg constructor).
    #[must_use]
    pub fn new(
        a: fn(f64) -> f64,
        a_prime: fn(f64) -> f64,
        a_double_prime: fn(f64) -> f64,
        a_norm_bound: f64,
        grid: Grid1D<f64>,
    ) -> Self {
        Self {
            a,
            a_prime,
            a_double_prime,
            a_norm_bound,
            grid,
            b_for_conjugation: None,
        }
    }

    /// Builder: enable IP-conjugation with drift `b` (f64).
    #[must_use]
    pub fn with_drift_conjugation(mut self, b: fn(f64) -> f64) -> Self {
        self.b_for_conjugation = Some(b);
        self
    }
}

// ---------------------------------------------------------------------------
// impl<F> TruncatedExp4thDiffusionChernoff<F> — generic path
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> TruncatedExp4thDiffusionChernoff<F> {
    /// Truncated-series order: `K = 4`.
    #[allow(clippy::cast_possible_truncation)]
    pub const TRUNC_ORDER_GENERIC: u32 = TRUNC_ORDER_USIZE as u32;

    /// Construct a `TruncatedExp4thDiffusionChernoff<F>` (generic, non-f64 floats).
    #[must_use]
    pub fn new_generic(
        a: fn(F) -> F,
        a_prime: fn(F) -> F,
        a_double_prime: fn(F) -> F,
        a_norm_bound: f64,
        grid: Grid1D<F>,
    ) -> Self {
        Self {
            a,
            a_prime,
            a_double_prime,
            a_norm_bound,
            grid,
            b_for_conjugation: None,
        }
    }

    /// Builder: enable IP-conjugation with drift `b` (generic).
    #[must_use]
    pub fn with_drift_conjugation_generic(mut self, b: fn(F) -> F) -> Self {
        self.b_for_conjugation = Some(b);
        self
    }

    /// Apply `M⁴(τ)` — generic scalar path for non-f64 types.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0`, non-finite, or `a_norm_bound ≤ 0`.
    /// - [`SemiflowError::CflViolated`] if `8·tau·a_norm_bound ≥ 3·dx²`.
    /// - [`SemiflowError::Unsupported`] propagated from `f.sample_generic()`.
    pub fn apply_f(&self, tau: F, f: &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError> {
        validate_tau_generic(tau)?;
        validate_a_norm_bound_generic(self.a_norm_bound)?;
        let dx = self.grid.dx();
        validate_cfl_4th_generic(tau, self.a_norm_bound, dx)?;

        let g_grids = precompute_g4_grids_generic(self, f)?;

        let mut out = f.zeroed_like();
        for i in 0..f.values.len() {
            let x_i = self.grid.x_at(i);
            let x_mid = compute_x_mid_generic(self, x_i, tau);
            out.values[i] = apply_power_series_generic(tau, &g_grids, x_mid)?;
        }
        Ok(out)
    }

    /// Consistency order: 2.
    pub fn order_val(&self) -> u32 {
        2
    }

    /// Growth bound `(M, ω) = (1.0, 0.0)`.
    pub fn growth_val(&self) -> (f64, f64) {
        (1.0, 0.0)
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl for TruncatedExp4thDiffusionChernoff<f64>
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for TruncatedExp4thDiffusionChernoff<f64> {
    type S = GridFn1D<f64>;

    /// Consistency order **2**.
    fn order(&self) -> u32 {
        2
    }

    /// Growth `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction.
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }

    /// Allocation-free override (ADR-0041): scratch-pool g-grids, output into `dst`.
    /// Bit-identical to `apply`; uses `take_vec`/`return_vec` for simultaneous buffers.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        validate_tau_f64(tau)?;
        validate_a_norm_bound_f64(self.a_norm_bound)?;
        // Use src.grid for dx so that bc_index/interp uses the same n as src.values.
        // self.grid.n may differ from src.values.len() when run_matrix applies the
        // operator across multiple grid sizes (ADR-0036 bit-equal gate).
        let src_grid = src.grid;
        let dx = src_grid.dx();
        validate_cfl_4th_f64(tau, self.a_norm_bound, dx)?;
        let n = src.values.len();
        let mut g1 = scratch.take_vec(n);
        let mut g2 = scratch.take_vec(n);
        let mut g3 = scratch.take_vec(n);
        let mut g4 = scratch.take_vec(n);
        apply_g4_stencil_into_slice(self, src_grid, &src.values, &mut g1, n, dx)?;
        apply_g4_stencil_into_slice(self, src_grid, &g1, &mut g2, n, dx)?;
        apply_g4_stencil_into_slice(self, src_grid, &g2, &mut g3, n, dx)?;
        apply_g4_stencil_into_slice(self, src_grid, &g3, &mut g4, n, dx)?;
        let g_slices: [&[f64]; TRUNC_ORDER_USIZE + 1] = [&src.values, &g1, &g2, &g3, &g4];
        dst.values.resize(n, 0.0);
        for i in 0..n {
            dst.values[i] = if self.b_for_conjugation.is_none() {
                apply_power_series_slices_at_node(tau, &g_slices, i)
            } else {
                let x_mid = compute_x_mid_f64(self, src_grid.x_at(i), tau);
                apply_power_series_slices(tau, &g_slices, src_grid, x_mid)?
            };
        }
        scratch.return_vec(g1);
        scratch.return_vec(g2);
        scratch.return_vec(g3);
        scratch.return_vec(g4);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private validation helpers — f64
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn validate_tau_f64(tau: f64) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau,
        });
    }
    Ok(())
}

#[inline]
pub(crate) fn validate_a_norm_bound_f64(a_norm_bound: f64) -> Result<(), SemiflowError> {
    if !a_norm_bound.is_finite() || a_norm_bound <= 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "a_norm_bound must be finite and > 0",
            value: a_norm_bound,
        });
    }
    Ok(())
}

/// 4th-order CFL check: `Err(CflViolated)` when `8·tau·a_norm_bound ≥ 3·dx²`.
#[inline]
pub(crate) fn validate_cfl_4th(tau: f64, a_norm_bound: f64, dx: f64) -> Result<(), SemiflowError> {
    let dx_squared = dx * dx;
    if 8.0 * tau * a_norm_bound >= 3.0 * dx_squared {
        return Err(SemiflowError::CflViolated {
            tau,
            dx_squared,
            a_norm_bound,
        });
    }
    Ok(())
}

#[inline]
fn validate_cfl_4th_f64(tau: f64, a_norm_bound: f64, dx: f64) -> Result<(), SemiflowError> {
    validate_cfl_4th(tau, a_norm_bound, dx)
}

// ---------------------------------------------------------------------------
// Private validation helpers — generic
// ---------------------------------------------------------------------------

#[inline]
fn validate_tau_generic<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

#[inline]
fn validate_a_norm_bound_generic(a_norm_bound: f64) -> Result<(), SemiflowError> {
    if !a_norm_bound.is_finite() || a_norm_bound <= 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "a_norm_bound must be finite and > 0",
            value: a_norm_bound,
        });
    }
    Ok(())
}

#[inline]
fn validate_cfl_4th_generic<F: SemiflowFloat>(
    tau: F,
    a_norm_bound: f64,
    dx: F,
) -> Result<(), SemiflowError> {
    let tau_f64 = tau.to_f64().unwrap_or(f64::NAN);
    let dx_f64 = dx.to_f64().unwrap_or(f64::NAN);
    validate_cfl_4th(tau_f64, a_norm_bound, dx_f64)
}

// ---------------------------------------------------------------------------
// Private computation helpers — f64
// ---------------------------------------------------------------------------

#[inline]
fn compute_x_mid_f64(mc: &TruncatedExp4thDiffusionChernoff<f64>, x_i: f64, tau: f64) -> f64 {
    match mc.b_for_conjugation {
        Some(b) => x_i - 0.5 * tau * b(x_i),
        None => x_i,
    }
}

/// No-drift bypass: `x_mid == x_i`, so `values[i]` is exact (no interpolation needed).
///
/// Used when `b_for_conjugation.is_none()` (see ADR-0019 Amendment 2, v0.13.0 Wave B1).
/// Direct index avoids a cubic-Hermite call per node. Bit-equal to the interpolation
/// path because `catmull_rom`(*, `values[i]`, *, *, 0.0) == `values[i]` exactly in IEEE 754.
#[inline]
pub(crate) fn apply_power_series_f64_at_node(
    tau: f64,
    g_grids: &[GridFn1D<f64>; TRUNC_ORDER_USIZE + 1],
    i: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut tau_pow = 1.0;
    for k in 0..=TRUNC_ORDER_USIZE {
        sum += FACTORIAL_INVERSE[k] * tau_pow * g_grids[k].values[i];
        tau_pow *= tau;
    }
    sum
}

// ---------------------------------------------------------------------------
// Private helpers — extracted to child module to keep file under 500 lines
// ---------------------------------------------------------------------------

#[path = "truncated_exp4_compute.rs"]
mod compute;
use compute::{
    apply_g4_stencil_into_slice, apply_power_series_generic, apply_power_series_slices,
    apply_power_series_slices_at_node, compute_x_mid_generic, precompute_g4_grids_generic,
};
