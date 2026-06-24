//! [`TruncatedExpDiffusionChernoff`] — truncated-exp K=4 Chernoff for `A_self = ∂_x(a(x)·∂_x)` (v0.4.0, ADR-0011).
//!
//! Truncated operator power series:
//!
//! ```text
//! M(τ) f(x) ≈ Σ_{k=0..4} (τ^k / k!) · G^k f(x_mid)
//! ```
//!
//! where `G` is the 3-point divergence-form stencil:
//!
//! ```text
//! G f|_i = (a_{i+½}·(f_{i+1} − f_i) − a_{i-½}·(f_i − f_{i-1})) / dx²
//! a_{i±½} = a((x_i + x_{i±1}) / 2)
//! ```
//!
//! Optional IP conjugation: `x_mid = x − (τ/2)·b(x)` (enabled via `with_drift_conjugation`).
//!
//! CFL stability: requires `2·τ·‖a‖_∞ < dx²`; violated → `SemiflowError::CflViolated`.
//!
//! Sympy gates (`verify_magnus_sympy.py`): `M_τ⁰` ✓  `M_τ¹` ✓  `M_τ²` ✓.
//!
//! **API contract**: `new` takes 5 args `(a, a_prime, a_double_prime, a_norm_bound, grid)`.
//! `a_prime` and `a_double_prime` are accepted for API symmetry with `DiffusionChernoff`
//! but are **NOT used** in the divergence-form stencil — the stencil differentiates `a`
//! implicitly via `a_{i±½}` evaluations.
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D Wave 1)
//!
//! `TruncatedExpDiffusionChernoff<F: SemiflowFloat = f64>` — the `= f64` default keeps
//! all existing call-sites compiling unchanged. `TruncatedExpDiffusionChernoff<f64>`
//! implements `ChernoffFunction`. Other `F` types use `apply_f` (scalar path).

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::SemiflowFloat,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of terms in the truncated power series (K=4): k = 0, 1, 2, 3, 4.
const TRUNC_ORDER_USIZE: usize = 4;

/// Factorial inverses `[1/0!, 1/1!, 1/2!, 1/3!, 1/4!]` for k = 0..=K.
///
/// Table-driven: no runtime factorial computation.
const FACTORIAL_INVERSE: [f64; TRUNC_ORDER_USIZE + 1] = [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0];

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Chernoff function for `A_self f = ∂_x(a(x)·∂_x f)` via truncated-exp K=4 series.
///
/// **What this computes**: the truncated Taylor series `Σ_{k=0}^{K} (τᵏ/k!) Gᵏ` of the
/// 3-point divergence-form generator G (NOT the genuine Magnus expansion).
/// Renamed from `MagnusDiffusionChernoff` in v0.7.0 per audit finding D2.
///
/// Preserves positivity when the CFL condition holds; useful for stiff problems
/// where the diffusion time-scale is much shorter than `t`.
///
/// ## Generic-over-Float (ADR-0025)
///
/// `TruncatedExpDiffusionChernoff<F: SemiflowFloat = f64>` — `= f64` keeps all existing
/// call-sites unchanged. `TruncatedExpDiffusionChernoff<f64>` implements
/// [`ChernoffFunction`]. Other `F` types use `apply_f`.
///
/// # Caller invariants
/// - `a(x) > 0` everywhere (strict ellipticity; required by the 3-point stencil).
/// - `a_norm_bound` is a valid upper bound for `‖a‖_∞` (used for CFL check).
/// - CFL: `2·τ·a_norm_bound < dx²`; violated → `Err(CflViolated)`.
///
/// # Example
///
/// ```rust
/// use semiflow::{chernoff::ApplyChernoffExt, Grid1D, GridFn1D, TruncatedExpDiffusionChernoff};
/// let grid = Grid1D::new(-4.0, 4.0, 64).unwrap();
/// // CFL: tau=0.001 < dx²/(2*a) ≈ 0.003 for a=1.0, dx≈0.126
/// let me = TruncatedExpDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// let u1 = me.apply_chernoff(0.001, &u0).unwrap();
/// assert_eq!(u1.values.len(), 64);
/// ```
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy)]
pub struct TruncatedExpDiffusionChernoff<F: SemiflowFloat = f64> {
    /// Diffusion coefficient `a(x)`. Caller MUST guarantee `a(x) > 0`.
    pub a: fn(F) -> F,
    /// First derivative `a'(x)`. Accepted for API symmetry; NOT used in stencil.
    pub a_prime: fn(F) -> F,
    /// Second derivative `a''(x)`. Accepted for API symmetry; NOT used in stencil.
    pub a_double_prime: fn(F) -> F,
    /// Upper bound for `‖a‖_∞`. Used in the CFL stability check.
    pub a_norm_bound: f64,
    /// Reference grid geometry (node iteration and output allocation).
    pub grid: Grid1D<F>,
    /// Optional IP-conjugation drift `b(x)`. When `Some(b)`, foot-point is
    /// `x_mid = x − (τ/2)·b(x)`; when `None`, `x_mid = x`.
    pub b_for_conjugation: Option<fn(F) -> F>,
}

// ---------------------------------------------------------------------------
// impl TruncatedExpDiffusionChernoff<f64> — concrete f64 path
// ---------------------------------------------------------------------------

impl TruncatedExpDiffusionChernoff<f64> {
    /// Truncated-series order exposed for tests: `K = 4`.
    #[allow(clippy::cast_possible_truncation)]
    pub const TRUNC_ORDER: u32 = TRUNC_ORDER_USIZE as u32;

    /// Construct a `TruncatedExpDiffusionChernoff` (v0.4.0, 5-arg constructor).
    ///
    /// Mirror of [`crate::DiffusionChernoff::new`]. `b_for_conjugation` defaults to `None`.
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

    /// Builder: enable interaction-picture (IP) conjugation with drift `b` (f64).
    #[must_use]
    pub fn with_drift_conjugation(mut self, b: fn(f64) -> f64) -> Self {
        self.b_for_conjugation = Some(b);
        self
    }
}

// ---------------------------------------------------------------------------
// impl<F> TruncatedExpDiffusionChernoff<F> — generic path for non-f64 types
// ---------------------------------------------------------------------------

impl<F: SemiflowFloat> TruncatedExpDiffusionChernoff<F> {
    /// Truncated-series order: `K = 4`.
    #[allow(clippy::cast_possible_truncation)]
    pub const TRUNC_ORDER_GENERIC: u32 = TRUNC_ORDER_USIZE as u32;

    /// Construct a `TruncatedExpDiffusionChernoff<F>` (generic version for non-f64 floats).
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

    /// Apply `M(τ)` pointwise — generic scalar path for non-f64 types.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `tau < 0`, non-finite, or
    ///   `a_norm_bound ≤ 0`/non-finite.
    /// - [`SemiflowError::CflViolated`] if CFL violated.
    /// - [`SemiflowError::Unsupported`] propagated from `f.sample_generic()`.
    pub fn apply_f(&self, tau: F, f: &GridFn1D<F>) -> Result<GridFn1D<F>, SemiflowError> {
        validate_tau_generic(tau)?;
        validate_a_norm_bound_generic(self.a_norm_bound)?;
        let dx = self.grid.dx();
        validate_cfl_generic(tau, self.a_norm_bound, dx)?;

        let g_grids = precompute_g_grids_generic(self, f)?;

        let mut out = f.zeroed_like();
        for i in 0..f.values.len() {
            let x_i = self.grid.x_at(i);
            let x_mid = compute_x_mid_generic(self, x_i, tau);
            out.values[i] = apply_at_node_generic(tau, &g_grids, x_mid)?;
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
// ChernoffFunction impl for TruncatedExpDiffusionChernoff<f64>
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for TruncatedExpDiffusionChernoff<f64> {
    type S = GridFn1D<f64>;

    /// Consistency order 2 (global O(τ²) for variable `a ∈ C³`, math.md §9.2.3.C).
    fn order(&self) -> u32 {
        2
    }

    /// Growth `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction.
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }

    /// Allocation-free override: g-grid buffers come from `scratch`; output reuses `dst`.
    ///
    /// Bit-identical to [`Self::apply_f`] by construction: same stencil + series arithmetic,
    /// only the backing `Vec` storage is taken from the pool instead of the heap.
    ///
    /// Uses `take_vec`/`return_vec` (not `borrow_vec`) so that all four g-grid
    /// intermediate buffers can be live simultaneously without conflicting borrows.
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
        validate_cfl_f64(tau, self.a_norm_bound, dx)?;
        let n = src.values.len();
        // Take 4 owned scratch buffers for G-power intermediates (g1..g4).
        // g0 is src itself (no copy).
        let mut g1 = scratch.take_vec(n);
        let mut g2 = scratch.take_vec(n);
        let mut g3 = scratch.take_vec(n);
        let mut g4 = scratch.take_vec(n);
        apply_g_once_slice(self, src_grid, &src.values, &mut g1, n, dx)?;
        apply_g_once_slice(self, src_grid, &g1, &mut g2, n, dx)?;
        apply_g_once_slice(self, src_grid, &g2, &mut g3, n, dx)?;
        apply_g_once_slice(self, src_grid, &g3, &mut g4, n, dx)?;
        let g_slices: [&[f64]; TRUNC_ORDER_USIZE + 1] = [&src.values, &g1, &g2, &g3, &g4];
        dst.values.resize(n, 0.0);
        for i in 0..n {
            let x_i = src_grid.x_at(i);
            let x_mid = compute_x_mid_f64(self, x_i, tau);
            dst.values[i] = apply_at_node_slices(tau, &g_slices, src_grid, x_mid)?;
        }
        // Return buffers to pool (capacity preserved).
        scratch.return_vec(g1);
        scratch.return_vec(g2);
        scratch.return_vec(g3);
        scratch.return_vec(g4);
        Ok(())
    }
}

// Phase 5a: additive impl — delegates to generic scalar apply_f path.
impl ChernoffFunction<f32> for TruncatedExpDiffusionChernoff<f32> {
    type S = GridFn1D<f32>;

    /// Consistency order 2 (mirrors f64 impl; math.md §9.2.3.C).
    fn order(&self) -> u32 {
        2
    }

    /// Growth `(M, ω) = (1.0, 0.0)` — positivity-preserving contraction.
    fn growth(&self) -> Growth<f32> {
        Growth::contraction()
    }

    /// Scalar apply: delegates to `apply_f` (generic scalar path; no SIMD in 5a).
    fn apply_into(
        &self,
        tau: f32,
        src: &GridFn1D<f32>,
        dst: &mut GridFn1D<f32>,
        _scratch: &mut ScratchPool<f32>,
    ) -> Result<(), SemiflowError> {
        let result = self.apply_f(tau, src)?;
        dst.values.resize(result.values.len(), 0.0);
        dst.values.copy_from_slice(&result.values);
        dst.grid = result.grid;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private validation helpers — f64
// ---------------------------------------------------------------------------

#[inline]
fn validate_tau_f64(tau: f64) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "tau must be finite and >= 0",
            value: tau,
        });
    }
    Ok(())
}

#[inline]
fn validate_a_norm_bound_f64(a_norm_bound: f64) -> Result<(), SemiflowError> {
    if !a_norm_bound.is_finite() || a_norm_bound <= 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "a_norm_bound must be finite and > 0",
            value: a_norm_bound,
        });
    }
    Ok(())
}

/// CFL stability check: returns `Err(CflViolated)` when `2·tau·a_norm_bound ≥ dx²`.
#[inline]
pub(crate) fn validate_cfl(tau: f64, a_norm_bound: f64, dx: f64) -> Result<(), SemiflowError> {
    let dx_squared = dx * dx;
    if 2.0 * tau * a_norm_bound >= dx_squared {
        return Err(SemiflowError::CflViolated {
            tau,
            dx_squared,
            a_norm_bound,
        });
    }
    Ok(())
}

#[inline]
fn validate_cfl_f64(tau: f64, a_norm_bound: f64, dx: f64) -> Result<(), SemiflowError> {
    validate_cfl(tau, a_norm_bound, dx)
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

/// CFL check for generic `F` — converts `tau` and `dx` to f64 for the check.
#[inline]
fn validate_cfl_generic<F: SemiflowFloat>(
    tau: F,
    a_norm_bound: f64,
    dx: F,
) -> Result<(), SemiflowError> {
    let tau_f64 = tau.to_f64().unwrap_or(f64::NAN);
    let dx_f64 = dx.to_f64().unwrap_or(f64::NAN);
    validate_cfl(tau_f64, a_norm_bound, dx_f64)
}

// ---------------------------------------------------------------------------
// Private computation helpers — f64
// ---------------------------------------------------------------------------

#[inline]
fn compute_x_mid_f64(mc: &TruncatedExpDiffusionChernoff<f64>, x_i: f64, tau: f64) -> f64 {
    match mc.b_for_conjugation {
        Some(b) => x_i - 0.5 * tau * b(x_i),
        None => x_i,
    }
}

/// Apply the G stencil once to a plain slice, writing result into `out`.
///
/// Used by the scratch-based `apply_into` override to avoid allocating `GridFn1D`.
/// Boundary sampling delegates to `grid.interp(prev, x)` without allocation.
///
/// `grid` MUST have the same `n` and `dx` as `prev.len()` (i.e. it should be
/// `src.grid`, not `mc.grid`, when the two may differ across grid-size sweeps).
fn apply_g_once_slice(
    mc: &TruncatedExpDiffusionChernoff<f64>,
    grid: crate::grid::Grid1D<f64>,
    prev: &[f64],
    out: &mut [f64],
    n: usize,
    dx: f64,
) -> Result<(), SemiflowError> {
    let dx_sq = dx * dx;
    for i in 0..n {
        let x_i = grid.x_at(i);
        let h_right = if i + 1 < n {
            prev[i + 1]
        } else {
            grid.interp(prev, x_i + dx)?
        };
        let h_left = if i > 0 {
            prev[i - 1]
        } else {
            grid.interp(prev, x_i - dx)?
        };
        let h_i = prev[i];
        let a_half_right = (mc.a)((x_i + x_i + dx) / 2.0);
        let a_half_left = (mc.a)((x_i + x_i - dx) / 2.0);
        out[i] = (a_half_right * (h_right - h_i) - a_half_left * (h_i - h_left)) / dx_sq;
    }
    Ok(())
}

/// Evaluate the truncated power series at `x_mid` from raw slices.
///
/// Equivalent to `apply_at_node_f64` but samples via `grid.interp` on plain slices
/// without allocating `GridFn1D` objects.
fn apply_at_node_slices(
    tau: f64,
    g_slices: &[&[f64]; TRUNC_ORDER_USIZE + 1],
    grid: crate::grid::Grid1D<f64>,
    x_mid: f64,
) -> Result<f64, SemiflowError> {
    let mut sum = 0.0;
    let mut tau_pow = 1.0;
    for k in 0..=TRUNC_ORDER_USIZE {
        let gk_val = grid.interp(g_slices[k], x_mid)?;
        sum += FACTORIAL_INVERSE[k] * tau_pow * gk_val;
        tau_pow *= tau;
    }
    Ok(sum)
}

// ---------------------------------------------------------------------------
// Private computation helpers — generic (extracted to sibling file per ≤500-line cap)
// ---------------------------------------------------------------------------

#[path = "truncated_exp_generic.rs"]
mod generic_helpers;
use generic_helpers::{apply_at_node_generic, compute_x_mid_generic, precompute_g_grids_generic};
