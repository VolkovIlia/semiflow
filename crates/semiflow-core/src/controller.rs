//! Step-size controllers for [`crate::AdaptivePI`] (Wave 4, ADR-0044).
//!
//! ## Controllers
//!
//! - [`ClassicalPI<F>`] — Söderlind 2002 "PI.4.7" law. **NORMATIVE default** per
//!   math.md §11.1.bis. Bit-identical accepted-τ trajectory to v1.0.0.
//! - [`H211bFilter<F>`] — Söderlind 2003 digital low-pass filter (ADVISORY,
//!   opt-in only via `.with_controller`). NOT in math.md.
//!
//! Both are zero-alloc (`core` + `libm` only) and safe (`deny(unsafe_code)`).

use crate::float::SemiflowFloat;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Step-size law for [`crate::AdaptivePI`].
///
/// The controller owns its own mutable state (e.g. previous error norm) and
/// returns the next-τ *multiplier* per call.  Clamping to `[min_ratio,
/// max_ratio]` is performed by [`crate::AdaptivePI`], not here.
///
/// `propose_accept` and `propose_reject` are separate so that digital-filter
/// controllers update their history only on accepted steps (Söderlind 2003 §3).
pub trait StepController<F: SemiflowFloat> {
    /// Called after a substep is **accepted**.
    ///
    /// Returns the multiplier `r` such that `next_τ = clamp(prev_τ · r)`.
    /// The controller MUST update its internal state on this call.
    ///
    /// # Arguments
    /// - `err_norm` — Richardson error norm of the accepted step.
    /// - `tol`      — mixed abs/rel tolerance at the current state.
    /// - `safety`   — safety factor from `AdaptivePI` (e.g. `0.9`).
    /// - `p_order`  — consistency order of the inner Chernoff function.
    fn propose_accept(&mut self, err_norm: F, tol: F, safety: F, p_order: u32) -> F;

    /// Called after a substep is **rejected**.
    ///
    /// Returns the shrink multiplier.  Controllers MAY or MAY NOT update their
    /// internal state on reject — `ClassicalPI` and `H211bFilter` do NOT.
    fn propose_reject(&mut self, err_norm: F, tol: F, safety: F, p_order: u32) -> F;

    /// Reset internal state to initial conditions.
    ///
    /// Called at the top of `evolve_adaptive` so consecutive evolutions on the
    /// same integrator are independent.  Must be idempotent.
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// ClassicalPI<F>
// ---------------------------------------------------------------------------

/// Söderlind 2002 "PI.4.7" step-size law (NORMATIVE default).
///
/// Gains: `α = 0.7/p`, `β = 0.4/p` (math.md §11.1.bis). Bit-identical to the
/// v1.0.0 inlined `pi_step_factor` / `reject_step_factor` helpers (ADR-0044).
///
/// The FP association `safety * e * e_prev` (left-to-right) is NORMATIVE and
/// enforced by `tests/adaptive_classical_bit_equal.rs`.
#[derive(Clone, Debug)]
pub struct ClassicalPI<F: SemiflowFloat> {
    /// Söderlind PI.4.7 P-gain exponent `α = 0.7 / p`.
    pub alpha: F,
    /// Söderlind PI.4.7 I-gain exponent `β = 0.4 / p`.
    pub beta: F,
    /// Previous-step error norm (I-term memory). Seeded to `F::one()`.
    err_prev: F,
}

impl<F: SemiflowFloat> ClassicalPI<F> {
    /// Construct with §11.1.bis gains for the given inner order `p`.
    #[must_use]
    pub fn with_order(p: u32) -> Self {
        let pf = F::from(f64::from(p)).unwrap_or(F::one());
        Self {
            alpha: F::from(0.7).unwrap_or(F::one()) / pf,
            beta: F::from(0.4).unwrap_or(F::one()) / pf,
            err_prev: F::one(),
        }
    }

    /// Direct constructor for tests and advanced users.
    #[must_use]
    pub fn new(alpha: F, beta: F) -> Self {
        Self {
            alpha,
            beta,
            err_prev: F::one(),
        }
    }

    /// Return the current alpha (P-gain exponent).
    #[inline]
    pub fn alpha(&self) -> F {
        self.alpha
    }

    /// Return the current beta (I-gain exponent).
    #[inline]
    pub fn beta(&self) -> F {
        self.beta
    }
}

impl<F: SemiflowFloat> Default for ClassicalPI<F> {
    /// Defaults assume `p = 2` (matches `DiffusionChernoff::order()` post-D1).
    fn default() -> Self {
        Self::with_order(2)
    }
}

impl<F: SemiflowFloat> StepController<F> for ClassicalPI<F> {
    fn propose_accept(&mut self, err_norm: F, tol: F, safety: F, _p: u32) -> F {
        // NORMATIVE FP association: `(safety * e) * e_prev` — left-to-right.
        // Changing this order breaks bit-equality with v1.0.0 (ADR-0044 §Risk 1).
        let floor = F::from(1e-300).unwrap_or(F::min_positive_value());
        let safe_err = err_norm.max(floor);
        let e = (tol / safe_err).powf(self.alpha);
        let e_prev = (self.err_prev / safe_err).powf(self.beta);
        let factor = safety * e * e_prev; // NORMATIVE: left-to-right
        self.err_prev = err_norm; // I-term update on accept
        factor
    }

    fn propose_reject(&mut self, err_norm: F, tol: F, safety: F, _p: u32) -> F {
        // I-term only; err_prev NOT updated on reject (matches v1.0.0).
        let floor = F::from(1e-300).unwrap_or(F::min_positive_value());
        let safe_err = err_norm.max(floor);
        safety * (tol / safe_err).powf(self.alpha)
    }

    fn reset(&mut self) {
        self.err_prev = F::one();
    }
}

// ---------------------------------------------------------------------------
// H211bFilter<F>
// ---------------------------------------------------------------------------

/// Söderlind 2003 H211b digital low-pass filter (ADVISORY, opt-in only).
///
/// Reduces step-size variance on stiff/oscillatory error norms at negligible
/// accuracy cost.  **NOT NORMATIVE** — documented at ADR-0044 + contract scope
/// only.  NOT in math.md §11.  Use only via:
/// `AdaptivePI::new(func).with_controller(H211bFilter::default())`.
///
/// Parameters: `b = 4`, `c = 1` (H211b convention, Söderlind 2003 Table 1).
/// Per-step exponent: `1/(b·p) = 1/(4p)`. Multiplier feedback: `−c/b = −1/4`.
#[derive(Clone, Debug)]
pub struct H211bFilter<F: SemiflowFloat> {
    /// Previous error norm `err_{n-1}`. Seeded to `F::one()`.
    err_prev: F,
    /// Previous accepted multiplier `r_{n-1}`. Seeded to `F::one()`.
    r_prev: F,
}

impl<F: SemiflowFloat> H211bFilter<F> {
    /// Construct with neutral initial state (all seeds = 1).
    #[must_use]
    pub fn new() -> Self {
        Self {
            err_prev: F::one(),
            r_prev: F::one(),
        }
    }
}

impl<F: SemiflowFloat> Default for H211bFilter<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SemiflowFloat> StepController<F> for H211bFilter<F> {
    fn propose_accept(&mut self, err_norm: F, tol: F, safety: F, p_order: u32) -> F {
        // H211b: ρ_n = (tol/e_n)^{1/(4p)} × (tol/e_{n-1})^{1/(4p)} × r_{n-1}^{-1/4}
        // r_n = safety · ρ_n
        let p = F::from(f64::from(p_order)).unwrap_or(F::one());
        let b4p = F::from(4.0).unwrap_or(F::one()) * p; // b·p with b=4
        let exp_e = F::one() / b4p; // 1/(4p)
        let exp_r = F::from(-0.25).unwrap_or(-F::one() / F::from(4.0).unwrap_or(F::one()));
        let floor = F::from(1e-300).unwrap_or(F::min_positive_value());
        let safe_e = err_norm.max(floor);
        let safe_ep = self.err_prev.max(floor);
        let term_e = (tol / safe_e).powf(exp_e);
        let term_ep = (tol / safe_ep).powf(exp_e);
        let term_r = self.r_prev.powf(exp_r);
        let factor = safety * term_e * term_ep * term_r; // left-to-right
                                                         // Update state on accept only:
        self.err_prev = err_norm;
        self.r_prev = factor;
        factor
    }

    fn propose_reject(&mut self, err_norm: F, tol: F, safety: F, p_order: u32) -> F {
        // On reject: fall back to I-term shrink (classical-style).
        // H211b literature is silent on the reject branch; I-term keeps shrink predictable.
        // State (err_prev, r_prev) NOT updated on reject.
        let p = F::from(f64::from(p_order)).unwrap_or(F::one());
        let alpha = F::from(0.7).unwrap_or(F::one()) / p;
        let floor = F::from(1e-300).unwrap_or(F::min_positive_value());
        let safe_e = err_norm.max(floor);
        safety * (tol / safe_e).powf(alpha)
    }

    fn reset(&mut self) {
        self.err_prev = F::one();
        self.r_prev = F::one();
    }
}
