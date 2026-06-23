//! [`DriftReactionZeta4Chernoff`] — order-4-temporal ζ⁴ sibling for
//! `L = a(x)∂ₓₓ + b(x)∂ₓ + c(x)` (ADR-0131, math.md §27.7-bis, v7.0.0).
//!
//! ## Mathematical foundation (ADR-0131, math.md §27.7-bis, NORMATIVE)
//!
//! Path β = Richardson extrapolation over the **symmetric** (palindromic) Strang base:
//!
//! ```text
//! S(τ) = R_sym(τ/2) ∘ K5(τ) ∘ R_sym(τ/2)
//! ```
//!
//! where `K5(τ)` is [`Diffusion4thChernoff`] (palindromic) and `R_sym(τ)` is
//! a **symmetric** (palindromic) drift-reaction step using the Newton-step
//! implicit-midpoint characteristic solver.
//!
//! ## Palindromic requirement (ADR-0131 §"Order-4 necessity")
//!
//! Richardson gives order 4 iff the base S has only **odd** local errors.
//! This requires `S(τ)·S(−τ) = I + O(τ⁵)` (palindromic to order 4).
//!
//! The existing [`DriftReactionChernoff`] uses an asymmetric RK2 foot with
//! round-trip error O(τ⁴), which limits Richardson to global order 3.
//! `DriftReactionZeta4Chernoff` instead uses a one-step Newton iteration
//! of the implicit midpoint (palindromic to O(τ⁵)):
//!
//! ```text
//! x_euler  = x + τ·b(x)
//! x_mid    = (x + x_euler) / 2
//! b_prime  = b'(x_mid)                    // caller-supplied
//! denom    = 1 − (τ/2)·b_prime
//! residual = x_euler − x − τ·b(x_mid)    // g(x_euler)
//! x_foot   = x_euler − residual / denom   // Newton step
//! ```
//!
//! For **linear** b(x) = αx + β: one step is EXACT (closed-form implicit midpoint),
//! achieving `R_sym(τ)·R_sym(−τ) = I` exactly. For nonlinear b: round-trip
//! O(τ⁵), sufficient for order-4 Richardson.
//!
//! ## Gate
//!
//! **G_DR_ZETA4_TRUTHFUL_ORDER** (RELEASE_BLOCKING per ADR-0131):
//! OLS slope ≤ −3.5 on a `[R, D] ≠ 0` datum. Feature `slow-tests`.
//!
//! ## Caller invariants
//!
//! 1. `f ∈ D(L⁴)`: smooth initial data.
//! 2. `a, b, c ∈ C⁴_b`; `b'` bounded and correct.
//! 3. `a(x) > 0` everywhere.

// Mathematical LaTeX symbols in doc comments; not code identifiers.
#![allow(clippy::doc_markdown)]

use crate::{
    chernoff::{ChernoffFunction, Growth},
    diffusion4::Diffusion4thChernoff,
    error::SemiflowError,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Order-4-temporal Chernoff kernel for `∂_t u = a(x)∂ₓₓu + b(x)∂ₓu + c(x)u`
/// (ADR-0131, math.md §27.7-bis).
///
/// Uses a palindromic Strang base `S(τ) = R_sym(τ/2) ∘ K5(τ) ∘ R_sym(τ/2)` with
/// Richardson extrapolation. `R_sym` uses the Newton-step implicit midpoint for the
/// characteristic foot — palindromic to O(τ⁵), enabling genuine order-4 convergence.
///
/// # Constructor
///
/// ```rust
/// use semiflow::{Diffusion4thChernoff, DriftReactionZeta4Chernoff, Grid1D, ChernoffFunction};
/// let grid = Grid1D::new(-5.0, 5.0, 256).unwrap();
/// let diff = Diffusion4thChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
/// // b(x) = -0.3x, c = -0.3, b'(x) = -0.3
/// let k = DriftReactionZeta4Chernoff::new(diff, |x: f64| -0.3*x, |_| -0.3, |_| -0.3_f64, 0.3, grid);
/// assert_eq!(k.order(), 4);
/// ```
///
/// # Caller invariants
///
/// 1. `f ∈ D(L⁴)`: smooth initial data.
/// 2. `a, b, c ∈ C⁴_b`; `b_prime` must equal `b'(x)`.
/// 3. `a(x) > 0` everywhere.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct DriftReactionZeta4Chernoff {
    base: PalindromicStrang,
    grid: Grid1D<f64>,
}

/// Internal palindromic Strang base: `R_sym(τ/2) ∘ K5(τ) ∘ R_sym(τ/2)`.
struct PalindromicStrang {
    diffusion: Diffusion4thChernoff<f64>,
    b: fn(f64) -> f64,
    b_prime: fn(f64) -> f64,
    c: fn(f64) -> f64,
    c_norm_bound: f64,
    grid: Grid1D<f64>,
}

impl Clone for PalindromicStrang {
    fn clone(&self) -> Self {
        Self {
            diffusion: self.diffusion.clone(),
            b: self.b,
            b_prime: self.b_prime,
            c: self.c,
            c_norm_bound: self.c_norm_bound,
            grid: self.grid,
        }
    }
}

impl PalindromicStrang {
    /// Apply `R_sym(τ)` at a single grid node using Newton-step implicit midpoint.
    ///
    /// Symmetric (palindromic to O(τ⁵)) for any b(x) with bounded b'(x).
    /// Exact for linear b(x) (one Newton step = exact implicit midpoint).
    #[inline]
    fn apply_r_sym_at(&self, tau: f64, f: &GridFn1D<f64>, i: usize) -> Result<f64, SemiflowError> {
        let x = self.grid.x_at(i);
        // Newton-step implicit midpoint for characteristic foot.
        let b_x = (self.b)(x);
        let x_euler = x + tau * b_x;
        let x_mid = 0.5 * (x + x_euler); // = x + (tau/2)*b(x)
        let b_mid = (self.b)(x_mid);
        let b_prime_mid = (self.b_prime)(x_mid);
        // g(x_euler) = x_euler - x - tau*b(x_mid)
        let residual = x_euler - x - tau * b_mid;
        // g'(x_euler) = 1 - (tau/2)*b'(x_mid)
        let g_prime = 1.0 - 0.5 * tau * b_prime_mid;
        let x_foot = if g_prime.abs() > 1e-15 {
            x_euler - residual / g_prime
        } else {
            x_euler // fallback: near-degenerate
        };
        // Trapezoidal reaction factor (exactly palindromic).
        let c_x = (self.c)(x);
        let c_foot = (self.c)(x_foot);
        let factor = libm::exp(0.5 * tau * (c_x + c_foot));
        let val = f.sample(x_foot)?;
        Ok(factor * val)
    }

    /// Apply `R_sym(τ/2) ∘ K5(τ) ∘ R_sym(τ/2)` into `dst`.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        let half_tau = tau / 2.0;
        let n = src.values.len();

        // R_sym(τ/2) · src → tmp1
        let mut tmp1 = GridFn1D {
            grid: src.grid,
            values: scratch.take_vec(n),
        };
        tmp1.values.resize(n, 0.0);
        for i in 0..n {
            tmp1.values[i] = self.apply_r_sym_at(half_tau, src, i)?;
        }
        tmp1.grid = src.grid;

        // K5(τ) · tmp1 → tmp2
        let mut tmp2 = GridFn1D {
            grid: src.grid,
            values: scratch.take_vec(n),
        };
        self.diffusion.apply_into(tau, &tmp1, &mut tmp2, scratch)?;

        // R_sym(τ/2) · tmp2 → dst
        dst.values.resize(n, 0.0);
        for i in 0..n {
            dst.values[i] = self.apply_r_sym_at(half_tau, &tmp2, i)?;
        }
        dst.grid = src.grid;

        scratch.return_vec(tmp1.values);
        scratch.return_vec(tmp2.values);
        Ok(())
    }

    fn growth(&self) -> Growth<f64> {
        let gd = self.diffusion.growth();
        Growth {
            multiplier: gd.multiplier,
            omega: gd.omega + self.c_norm_bound,
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl DriftReactionZeta4Chernoff {
    /// Construct from a K5 diffusion kernel and palindromic drift-reaction parameters.
    ///
    /// - `diffusion`: [`Diffusion4thChernoff`] (K5, palindromic) for the diffusion part.
    /// - `b`: drift coefficient `b(x)`.
    /// - `b_prime`: derivative `b'(x)` — required for the Newton-step implicit midpoint.
    /// - `c`: reaction coefficient `c(x)`.
    /// - `c_norm_bound`: upper bound for `‖c‖_∞`.
    /// - `grid`: spatial grid.
    ///
    /// For **linear** b(x) = αx + β: pass `b_prime = |_| α` (constant).
    /// One Newton step is then exact — the implicit midpoint is computed analytically.
    #[must_use]
    pub fn new(
        diffusion: Diffusion4thChernoff<f64>,
        b: fn(f64) -> f64,
        b_prime: fn(f64) -> f64,
        c: fn(f64) -> f64,
        c_norm_bound: f64,
        grid: Grid1D<f64>,
    ) -> Self {
        Self {
            base: PalindromicStrang {
                diffusion,
                b,
                b_prime,
                c,
                c_norm_bound,
                grid,
            },
            grid,
        }
    }
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for DriftReactionZeta4Chernoff {
    type S = GridFn1D<f64>;

    /// Consistency order **4** (Path β: Richardson over palindromic Strang-K5 base).
    fn order(&self) -> u32 {
        4
    }

    /// Growth bound: inherited from the palindromic Strang base.
    fn growth(&self) -> Growth<f64> {
        self.base.growth()
    }

    /// Path β: double Richardson extrapolation of the Strang-K5 base.
    ///
    /// Seven inner S applications per outer step (1+2+4):
    ///
    /// ```text
    /// u1 = S(τ)·src                         (1 step at τ)
    /// u2 = S(τ/2)²·src                      (2 steps at τ/2)
    /// u4 = S(τ/4)⁴·src                      (4 steps at τ/4)
    /// v1 = (4·u2 − u1) / 3                  (order-3 estimate at τ)
    /// v2 = (4·u4 − u2) / 3                  (order-3 estimate at τ/2)
    /// dst = (8·v2 − v1) / 7                 (order-4: eliminates τ³ term)
    /// ```
    ///
    /// Derivation: a symmetric order-2 base S has global error
    /// `u_n = u_exact + A·τ² + B·τ³ + …`.  Richardson-2 (factor 4) eliminates the
    /// τ² term: `v1 = u_exact − B·τ³/6 + …`, `v2 = u_exact − B·τ³/48 + …`.
    /// Richardson-3 (factor 8): `(8·v2 − v1)/7 = u_exact + O(τ⁴)`.
    /// This does **not** require the palindromic (time-symmetric) property of S —
    /// it eliminates both τ² and τ³ errors directly.
    ///
    /// # Errors
    /// Returns [`SemiflowError::DomainViolation`] on invalid τ or inner failures.
    fn apply_into(
        &self,
        tau: f64,
        src: &GridFn1D<f64>,
        dst: &mut GridFn1D<f64>,
        scratch: &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError> {
        if !tau.is_finite() || tau < 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "tau must be finite and >= 0",
                value: tau,
            });
        }
        let n = src.values.len();
        let tau2 = tau / 2.0;
        let tau4 = tau / 4.0;
        let (mut u1, mut u2, mut u4, mut tmp) = alloc_four_gridfns(self.grid, n, scratch);

        self.base.apply_into(tau, src, &mut u1, scratch)?; // u1 = S(τ)·src
        self.base.apply_into(tau2, src, &mut tmp, scratch)?; // u2 = S(τ/2)²·src
        self.base.apply_into(tau2, &tmp, &mut u2, scratch)?;
        apply_four_steps(
            |t, s, d, sc| self.base.apply_into(t, s, d, sc),
            src,
            &mut u4,
            &mut tmp,
            tau4,
            n,
            scratch,
        )?;

        // Richardson: dst = (32·u4 − 12·u2 + u1) / 21
        // Derivation: v1=(4u2−u1)/3, v2=(4u4−u2)/3; (8v2−v1)/7 = (32u4−12u2+u1)/21
        dst.values.resize(n, 0.0);
        for i in 0..n {
            dst.values[i] = (32.0 * u4.values[i] - 12.0 * u2.values[i] + u1.values[i]) / 21.0;
        }
        scratch.return_vec(u1.values);
        scratch.return_vec(u2.values);
        scratch.return_vec(u4.values);
        scratch.return_vec(tmp.values);
        Ok(())
    }
}

/// Allocate 4 scratch `GridFn1D<f64>` buffers (all zero-initialized).
#[inline]
fn alloc_four_gridfns(
    grid: Grid1D<f64>,
    n: usize,
    scratch: &mut ScratchPool<f64>,
) -> (GridFn1D<f64>, GridFn1D<f64>, GridFn1D<f64>, GridFn1D<f64>) {
    let mk = |sc: &mut ScratchPool<f64>| {
        let mut v = sc.take_vec(n);
        v.resize(n, 0.0);
        GridFn1D { grid, values: v }
    };
    (mk(scratch), mk(scratch), mk(scratch), mk(scratch))
}

/// Apply base semigroup `S` (via `step` closure) four times at step `tau4`.
///
/// `tmp` is the carry buffer; ends at step 3/4.  `u4` receives step 4/4.
#[allow(clippy::too_many_arguments)]
fn apply_four_steps<F>(
    mut step: F,
    src: &GridFn1D<f64>,
    u4: &mut GridFn1D<f64>,
    tmp: &mut GridFn1D<f64>,
    tau4: f64,
    n: usize,
    scratch: &mut ScratchPool<f64>,
) -> Result<(), SemiflowError>
where
    F: FnMut(
        f64,
        &GridFn1D<f64>,
        &mut GridFn1D<f64>,
        &mut ScratchPool<f64>,
    ) -> Result<(), SemiflowError>,
{
    step(tau4, src, tmp, scratch)?; // step 1/4
    let mut tmp2 = GridFn1D {
        grid: tmp.grid,
        values: scratch.take_vec(n),
    };
    step(tau4, tmp, &mut tmp2, scratch)?; // step 2/4
    step(tau4, &tmp2, tmp, scratch)?; // step 3/4
    scratch.return_vec(tmp2.values);
    step(tau4, tmp, u4, scratch)?; // step 4/4
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

// Tests live in drift_reaction_zeta4_tests.rs (included below via include!).
// Using include! preserves super::* access to private types (PalindromicStrang etc.)
// while keeping this file within the 500-line suckless limit.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Diffusion4thChernoff, Grid1D, GridFn1D, ScratchPool};
    include!("drift_reaction_zeta4_tests.rs");
}
