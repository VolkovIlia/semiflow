//! [`DiffusionExpmvChernoff`] — tolerance-driven `e^{τA}·v` via Al-Mohy & Higham (2011)
//! `expmv` action kernel (ADR-0121, math.md §45).
//!
//! ## Mathematical foundation
//!
//! Computes `e^{τA}·v` without forming, squaring, or inverting any matrix.
//! A truncated Taylor polynomial `T_m(τA/s)` is applied to the VECTOR, `s` times:
//!
//! ```text
//! e^{τA} v ≈ (T_m(τA/s))^s v,  T_m(z) = Σ_{k=0}^{m} z^k / k!
//! ```
//!
//! Realised as Horner-on-vector (one `apply_div_form` call per inner term):
//!
//! ```text
//! y ← v
//! for i in 1..=s:
//!     w ← y
//!     for k in 1..=m:
//!         w ← (τ/s) · (A·w) / k       # one apply_div_form
//!         y ← y + w
//! ```
//!
//! Scaling: `s = ⌈τ‖A‖ / θ_m⌉` bounds `(τ/s)‖A‖ ≤ θ_m` — the `τ‖A‖≈62` blow-up
//! regime that defeated the Padé kernel is exactly what `s`-scaling tames.
//!
//! ## ADR-0121 status
//!
//! **ADDITIVE** — does not modify any existing kernel. ADR-0101 Padé terminal closure
//! is UNCHANGED; this is a different kernel class.
//!
//! ## References
//!
//! - A. H. Al-Mohy, N. J. Higham (2011), SIAM J. Sci. Comput. 33(2):488–511,
//!   DOI 10.1137/100788860 — primary (`expmv`, Table 3.1 θ_m, Algorithm 3.2).
//! - ADR-0121 (PRE-FLIGHT GO; engineer spec).
//! - math.md §45 (NORMATIVE algorithm).
//! - `scripts/verify_expmv_preflight.py` (PRE-FLIGHT harness; executed 2026-06-05).

// Mathematical LaTeX symbols and doc-markdown intentional.
#![allow(clippy::doc_markdown)]

extern crate alloc;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    diffusion4::Diffusion4thChernoff,
    diffusion4_zeta4::apply_div_form,
    error::SemiflowError,
    grid::Grid1D,
    grid_fn::GridFn1D,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// θ_m table — Al-Mohy & Higham (2011) Table 3.1, double-precision (tol = 2^-53).
//
// CORRECTED in ADR-0197: the shipped table paired each degree with a radius from
// two to three rows further down the published table (m=18 carried θ≈8.84, the
// radius of m≈51, where its own is 1.09), so `select_s_m` chose too few substeps
// and the answer was silently wrong — 1.6e-4 where double precision was claimed.
// ---------------------------------------------------------------------------

/// `(degree, θ_m)` — the largest argument at which a degree-`m` truncated Taylor
/// exponential meets double-precision BACKWARD error.
///
/// Recomputed from the definition rather than re-copied (ADR-0197): expand
/// `log(e^{-x}·T_m(x))` in exact rational arithmetic, form
/// `h_{m+1}(x) = Σ_{k>m}|c_k|x^k`, and solve `h_{m+1}(θ)/θ = 2^{-53}`. The result
/// reproduces Table 3.1 to three significant figures at every degree.
///
/// Denser than the old sparse subset because `select_s_m` minimises `s·m` over
/// the table: more entries is strictly cheaper as well as correct.
///
/// Re-used by [`crate::phi_action`] for the augmented-matrix φ-action.
pub(crate) const THETA_M: &[(u32, f64)] = &[
    (1, 2.220e-16),
    (2, 2.581e-8),
    (3, 1.386e-5),
    (4, 3.397e-4),
    (5, 2.401e-3),
    (6, 9.066e-3),
    (7, 2.384e-2),
    (8, 4.991e-2),
    (9, 8.958e-2),
    (10, 1.442e-1),
    (11, 2.142e-1),
    (12, 2.996e-1),
    (13, 3.998e-1),
    (14, 5.139e-1),
    (15, 6.411e-1),
    (16, 7.803e-1),
    (17, 9.305e-1),
    (18, 1.091),
    (19, 1.260),
    (20, 1.438),
    (25, 2.429),
    (30, 3.540),
];

/// Maximum Taylor degree.
///
/// Raised 18 → 30 in ADR-0197. The old cap cited "above arg ≈ 9 a plain monomial
/// Horner loses precision" — a limit on the *argument*, which the mis-transcribed
/// table was violating (it fed arg up to 8.84 into `m = 18`). With correct radii
/// the per-substep argument is at most `θ_30 = 3.54`, well inside that limit,
/// and capping at 18 would cost `s = 7` substeps where `s = 2` suffices.
pub(crate) const M_MAX: u32 = 30;

// ---------------------------------------------------------------------------
// (s, m) selector — Al-Mohy & Higham Algorithm 3.2 (conservative norm bound).
// ---------------------------------------------------------------------------

/// Select `(s, m)` minimising `s·m` s.t. `(τ/s)·norm_a ≤ θ_m` and `m ≤ M_MAX`.
///
/// `norm_a` is an upper bound on `‖A‖`. Returns the cheapest valid pair.
/// Falls back to `(s_min, M_MAX)` if no entry in the table suffices at cost 1.
///
/// Re-used by [`crate::phi_action`] for the augmented operator norm.
#[allow(clippy::many_single_char_names)] // s, m are standard Al-Mohy–Higham notation
pub(crate) fn select_s_m(norm_a: f64, tau: f64) -> (u32, u32) {
    let arg = tau * norm_a;
    // Store (s, m, cost) so the comparison uses the actual cost, not m.
    let mut best: Option<(u32, u32, u64)> = None;
    for &(m, theta) in THETA_M {
        if m > M_MAX {
            break;
        }
        // s = ceil(arg / theta), minimum 1.
        let s_raw = (arg / theta).ceil();
        // Skip entries where s is astronomically large (tiny theta, large arg).
        if s_raw > 1.0e14_f64 {
            continue;
        }
        let s = if s_raw < 1.0 {
            1u32
        } else {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            {
                s_raw as u32
            }
        };
        // Use u64 for cost to avoid u32 overflow in the comparison.
        let cost = u64::from(s) * u64::from(m);
        let better = best.map_or(true, |(_, _, prev_cost)| cost < prev_cost);
        if better {
            best = Some((s, m, cost));
        }
    }
    // If no entry was feasible, fall back to large s at M_MAX.
    best.map_or((1, M_MAX), |(s, m, _)| (s, m))
}

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

/// Tolerance-driven `e^{τA}·v` evolver via Al-Mohy & Higham (2011) `expmv`
/// (ADR-0121, math.md §45, PRE-FLIGHT GO 2026-06-05).
///
/// Wraps a [`Diffusion4thChernoff`] (carrier of `apply_div_form` + grid + `a(x)`)
/// and realises the action `e^{τA}·v` by applying a scaled truncated Taylor
/// polynomial to the VECTOR — no Padé denominator, no matrix squaring.
///
/// **ADDITIVE**: does not replace any existing kernel. ADR-0101 Padé terminal
/// closure is UNCHANGED.
///
/// # Constructor
///
/// ```rust,no_run
/// use semiflow::{DiffusionExpmvChernoff, Diffusion4thChernoff, Grid1D};
/// let grid = Grid1D::new(0.0_f64, 20.0, 64).unwrap();
/// let inner = Diffusion4thChernoff::new(
///     |x: f64| 1.0 + 0.3 * (2.0 * core::f64::consts::PI * x / 20.0).sin(),
///     |_| 0.0,
///     |_| 0.0,
///     1.3,   // ‖a‖_∞ bound
///     grid,
/// );
/// let kernel = DiffusionExpmvChernoff::new(inner);
/// ```
#[derive(Clone)]
pub struct DiffusionExpmvChernoff {
    /// Inner divergence-form kernel — carrier of `apply_div_form` + grid.
    inner: Diffusion4thChernoff<f64>,
    /// Grid geometry (copy for direct access).
    grid: Grid1D<f64>,
    /// Conservative analytic ‖A‖ estimate: `4 · a_norm_bound / dx²`.
    ///
    /// Over-estimation only raises `s` (more, cheaper steps) without harming
    /// correctness. No Higham–Tisseur estimator needed (ADR-0121 rationale).
    norm_a_est: f64,
}

impl DiffusionExpmvChernoff {
    /// Construct from an inner `Diffusion4thChernoff`.
    ///
    /// `‖A‖` is estimated conservatively as `4 · a_norm_bound / dx²`.
    #[must_use]
    pub fn new(inner: Diffusion4thChernoff<f64>) -> Self {
        let grid = inner.grid;
        let dx = grid.dx();
        let norm_a_est = 4.0 * inner.a_norm_bound / (dx * dx);
        Self {
            inner,
            grid,
            norm_a_est,
        }
    }

    /// Override the default tolerance (reserved for future use).
    ///
    /// Currently has no effect on the `(s, m)` selection, which is determined
    /// by the baked Al-Mohy–Higham table at `tol = 2^-53`.
    #[must_use]
    pub const fn with_tolerance(self, _tol: f64) -> Self {
        // Table is fixed at double precision; tol override is a no-op placeholder.
        self
    }
}

// ---------------------------------------------------------------------------
// Core expmv action (separated for testability)
// ---------------------------------------------------------------------------

/// Apply `T_m(τ_s · A)` to `y` in place: one outer step of the Horner loop.
///
/// `τ_s = τ / s` is the per-step time. `w` is a scratch buffer (same shape as `y`).
/// Returns error if `apply_div_form` fails.
#[allow(clippy::many_single_char_names)] // k, m are standard Taylor-series indices
fn horner_step(
    inner: &Diffusion4thChernoff<f64>,
    y: &mut GridFn1D<f64>,
    w: &mut GridFn1D<f64>,
    tau_s: f64,
    m: u32,
    av_scratch: &mut GridFn1D<f64>,
) -> Result<(), SemiflowError> {
    // w ← y (start of Horner: w accumulates the k-th term)
    w.values.clone_from(&y.values);
    for k in 1..=m {
        // av_scratch = A · w
        apply_div_form(inner, w, av_scratch)?;
        // w ← τ_s · (A·w) / k
        let factor = tau_s / f64::from(k);
        for (wi, &avi) in w.values.iter_mut().zip(av_scratch.values.iter()) {
            *wi = factor * avi;
        }
        // y ← y + w
        for (yi, &wi) in y.values.iter_mut().zip(w.values.iter()) {
            *yi += wi;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ChernoffFunction impl
// ---------------------------------------------------------------------------

impl ChernoffFunction<f64> for DiffusionExpmvChernoff {
    type S = GridFn1D<f64>;

    /// `order()` returns `u32::MAX` — tolerance-driven, NOT fixed-order.
    ///
    /// `expmv` is an accuracy-controlled algorithm; comparing its order to the
    /// fixed-order slope gates (§27/§40) is INAPPLICABLE (ADR-0121 §Consequences).
    /// Callers must NOT interpret `u32::MAX` as a convergence order.
    fn order(&self) -> u32 {
        u32::MAX
    }

    /// Growth bound: contraction `(1, 0)` — same as inner K5 step.
    fn growth(&self) -> Growth<f64> {
        Growth::contraction()
    }

    /// Compute `dst = e^{τA} src` via scaled truncated-Taylor Horner-on-vector.
    ///
    /// Algorithm (ADR-0121 / math.md §45.1):
    /// 1. `(s, m) = select_s_m(‖A‖_est, τ)` — minimise `s·m` s.t. arg ≤ θ_m.
    /// 2. `y ← src`
    /// 3. for `i in 1..=s`: apply `T_m(τ/s · A)` to `y` in place (Horner loop).
    /// 4. `dst ← y`
    ///
    /// Uses `O(1)` extra work vectors (`y`, `w`, `av_scratch`); no LU, no matrix.
    ///
    /// # Errors
    ///
    /// - [`SemiflowError::DomainViolation`] for invalid `tau` or `n < 3`.
    #[allow(clippy::many_single_char_names)] // n, s, m are standard mathematical names here
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
        if n < 3 {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "expmv requires >= 3 grid points",
                value: n as f64,
            });
        }
        if tau == 0.0 {
            dst.values.clone_from(&src.values);
            return Ok(());
        }

        let (s, m) = select_s_m(self.norm_a_est, tau);
        let tau_s = tau / f64::from(s);

        // Scratch buffers: y (accumulator), w (Horner term), av_scratch (A·w).
        let (mut y, mut w, mut av_scratch) =
            take_three_gridfn1d(self.grid, n, &src.values, scratch);

        // s outer steps, each applying T_m(τ_s · A) in place.
        for _ in 0..s {
            horner_step(&self.inner, &mut y, &mut w, tau_s, m, &mut av_scratch)?;
        }

        dst.values.clone_from(&y.values);
        scratch.return_vec(y.values);
        scratch.return_vec(w.values);
        scratch.return_vec(av_scratch.values);
        Ok(())
    }
}

/// Allocate three `GridFn1D<f64>` scratch buffers from `pool`:
/// `y` (copy of `init`), `w` (zero), `av` (zero).
fn take_three_gridfn1d(
    grid: Grid1D<f64>,
    n: usize,
    init: &[f64],
    pool: &mut ScratchPool<f64>,
) -> (GridFn1D<f64>, GridFn1D<f64>, GridFn1D<f64>) {
    let mut y_buf = pool.take_vec(n);
    y_buf.clone_from(&init.to_vec());
    let mut w_buf = pool.take_vec(n);
    w_buf.resize(n, 0.0);
    let mut av_buf = pool.take_vec(n);
    av_buf.resize(n, 0.0);
    (
        GridFn1D {
            grid,
            values: y_buf,
        },
        GridFn1D {
            grid,
            values: w_buf,
        },
        GridFn1D {
            grid,
            values: av_buf,
        },
    )
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    include!("expmv_tests_mod.rs");
}
