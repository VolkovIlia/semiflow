//! [`AdaptivePI`] — generic adaptive PI integrator (Wave 4, ADR-0044).
//!
//! # NORMATIVE: Semigroup-splitting interpretation
//!
//! **`AdaptivePI<C, F, K>` is NOT a `ChernoffFunction`.**
//!
//! `AdaptivePI` wraps any `C: ChernoffFunction<F>` and integrates over a target
//! time `T` by composing adaptive substeps:
//!
//! ```text
//! u(T) = S(τ_k) ∘ S(τ_{k-1}) ∘ … ∘ S(τ_1) u₀,   Σ τᵢ = T
//! ```
//!
//! Each substep applies `S(τᵢ) ≈ e^{τᵢ A}` and the substeps are glued via the
//! semigroup property.  Convergence is governed by per-step local truncation
//! `O(τ^{p+1})`, accumulated via Lady Windermere's fan (HLW §II.3) — NOT by
//! Theorem 6's `O(1/n)` Chernoff-product bound. Do NOT wrap `AdaptivePI<C,...>`
//! in a `ChernoffSemigroup`.
//!
//! ## Step controllers (Wave 4, ADR-0044)
//!
//! The step-size law is pluggable via the [`crate::controller::StepController<F>`]
//! trait:
//!
//! - **`ClassicalPI<F>`** — Söderlind 2002 "PI.4.7", gains `α = 0.7/p, β = 0.4/p`.
//!   NORMATIVE default per math.md §11.1.bis. Bit-identical to v1.0.0 behaviour.
//! - **`H211bFilter<F>`** — advisory opt-in. Use
//!   `.with_controller(H211bFilter::default())` to swap in.
//!
//! See `docs/adr/0044-stepcontroller-trait-h211b-advisory.md` for design rationale.
//!
//! ## v2.0 breaking changes
//!
//! - `evolve_adaptive` is now `&mut self` (controllers carry mutable state).
//! - `pi.alpha` / `pi.beta` fields replaced by `pi.alpha()` / `pi.beta()` methods.
//!
//! See migration table in `contracts/v2/wave4-stepcontroller.md §7`.

use crate::{
    chernoff::ChernoffFunction,
    controller::{ClassicalPI, StepController},
    error::SemiflowError,
    float::SemiflowFloat,
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// Public structs
// ---------------------------------------------------------------------------

/// Adaptive PI semigroup-splitting integrator (Wave 4, ADR-0044).
///
/// Generic over:
/// - `C: ChernoffFunction<F>` — inner Chernoff function.
/// - `F: SemiflowFloat = f64` — scalar type.
/// - `K: StepController<F> = ClassicalPI<F>` — step-size law.
///
/// The default `K = ClassicalPI<F>` reproduces the v1.0.0 accepted-τ
/// trajectory byte-for-byte on f64 (math.md §11.1.bis NORMATIVE).
///
/// **NOT a `ChernoffFunction`** — see module-level doc. Do NOT wrap in
/// [`crate::ChernoffSemigroup`].
///
/// # When to use
///
/// Use `AdaptivePI` when:
/// - `t` is large or the solution has stiff transient regions.
/// - You cannot predict a safe fixed substep count `n` in advance.
/// - You need a tolerance target rather than a fixed error budget.
///
/// Use [`crate::ChernoffSemigroup`] when the step count `n` is known a priori.
///
/// # Examples
///
/// Default (`ClassicalPI`, f64 — v1.0.0 equivalent):
/// ```rust
/// use semiflow::{Grid1D, GridFn1D, DiffusionChernoff, AdaptivePI};
/// let grid = Grid1D::new(-4.0, 4.0, 64).unwrap();
/// let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
/// let mut ctrl = AdaptivePI::new(diff).with_tolerance(1e-6, 1e-4);
/// let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
/// let outcome = ctrl.evolve_adaptive(0.5, &u0).unwrap();
/// assert!(outcome.steps_accepted > 0);
/// ```
///
/// See: `docs/adr/0044-stepcontroller-trait-h211b-advisory.md`.
// reason: preserved across versions for API stability.
#[allow(clippy::module_name_repetitions)]
pub struct AdaptivePI<
    C: ChernoffFunction<F>,
    F: SemiflowFloat = f64,
    K: StepController<F> = ClassicalPI<F>,
> {
    /// Inner Chernoff function applied at each substep.
    pub func: C,
    /// Absolute tolerance for the mixed error norm.
    pub tol_abs: F,
    /// Relative tolerance for the mixed error norm.
    pub tol_rel: F,
    /// Safety factor for the step-size controller.
    pub safety: F,
    /// Minimum allowed step-size ratio (`next_τ` / `prev_τ`).
    pub min_ratio: F,
    /// Maximum allowed step-size ratio.
    pub max_ratio: F,
    /// Hard cap on total substeps (accepted + rejected) — runaway protection.
    pub max_substeps: usize,
    /// Step-size law (default: `ClassicalPI<F>`).
    controller: K,
    /// Lazily-allocated state scratch slots for zero-alloc Richardson.
    /// None until first `evolve_adaptive` call.
    state_scratch: Option<[C::S; 3]>,
    /// Vec-level scratch pool for `apply_into` calls.
    scratch: ScratchPool<F>,
    // Phantom: tie C to F without owning F directly
    _marker: core::marker::PhantomData<F>,
}

/// Observable outcome of a completed [`AdaptivePI::evolve_adaptive`] call.
#[derive(Clone, Debug)]
// reason: preserved across versions for API stability.
#[allow(clippy::module_name_repetitions)]
pub struct AdaptiveOutcome<S, F: SemiflowFloat = f64> {
    /// Evolved state `u(t)`.
    pub final_state: S,
    /// Number of substeps whose Richardson error estimate was ≤ tolerance.
    pub steps_accepted: usize,
    /// Number of substeps that were rejected and retried with a smaller `τ`.
    pub steps_rejected: usize,
    /// The substep size `τ` used on the last accepted step.
    ///
    /// Useful for warm-starting a subsequent call.
    pub last_tau: F,
}

// ---------------------------------------------------------------------------
// Constructor + builder (ClassicalPI default path)
// ---------------------------------------------------------------------------

impl<C, F> AdaptivePI<C, F, ClassicalPI<F>>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
{
    /// Construct with Söderlind PI.4.7 defaults for the given inner function.
    ///
    /// Sets `alpha = 0.7/p`, `beta = 0.4/p` where `p = func.order()`.
    /// Defaults: `tol_abs = 1e-8`, `tol_rel = 1e-6`, `safety = 0.9`,
    /// `min_ratio = 0.2`, `max_ratio = 5.0`, `max_substeps = 100_000`.
    #[must_use]
    pub fn new(func: C) -> Self {
        let controller = ClassicalPI::<F>::with_order(func.order());
        Self {
            tol_abs: F::from(1e-8).unwrap_or(F::zero()),
            tol_rel: F::from(1e-6).unwrap_or(F::zero()),
            safety: F::from(0.9).unwrap_or(F::one()),
            min_ratio: F::from(0.2).unwrap_or(F::zero()),
            max_ratio: F::from(5.0).unwrap_or(F::one()),
            max_substeps: 100_000,
            func,
            controller,
            state_scratch: None,
            scratch: ScratchPool::new(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Set absolute and relative tolerances (builder pattern).
    #[must_use]
    pub fn with_tolerance(mut self, abs: F, rel: F) -> Self {
        self.tol_abs = abs;
        self.tol_rel = rel;
        self
    }
}

// ---------------------------------------------------------------------------
// Generic builder methods (all K)
// ---------------------------------------------------------------------------

impl<C, F, K> AdaptivePI<C, F, K>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
    K: StepController<F>,
{
    /// Swap the step-size controller (type-changing builder).
    ///
    /// Consumes `self` and returns an `AdaptivePI` with a different `K`.
    /// Example: `.with_controller(H211bFilter::default())`.
    #[must_use]
    pub fn with_controller<K2: StepController<F>>(self, ctrl: K2) -> AdaptivePI<C, F, K2> {
        AdaptivePI {
            func: self.func,
            tol_abs: self.tol_abs,
            tol_rel: self.tol_rel,
            safety: self.safety,
            min_ratio: self.min_ratio,
            max_ratio: self.max_ratio,
            max_substeps: self.max_substeps,
            controller: ctrl,
            state_scratch: None, // new K may have different state; re-init on first evolve
            scratch: ScratchPool::new(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Borrow the current controller (read-only).
    #[inline]
    pub fn controller(&self) -> &K {
        &self.controller
    }

    /// Borrow the current controller (mutable).
    #[inline]
    pub fn controller_mut(&mut self) -> &mut K {
        &mut self.controller
    }
}

// ---------------------------------------------------------------------------
// Source-compat accessors for ClassicalPI users
// ---------------------------------------------------------------------------

impl<C, F> AdaptivePI<C, F, ClassicalPI<F>>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
{
    /// Return the P-gain exponent `α = 0.7/p` (math.md §11.1.bis).
    #[inline]
    pub fn alpha(&self) -> F {
        self.controller.alpha()
    }

    /// Return the I-gain exponent `β = 0.4/p` (math.md §11.1.bis).
    #[inline]
    pub fn beta(&self) -> F {
        self.controller.beta()
    }
}

// ---------------------------------------------------------------------------
// Main algorithm
// ---------------------------------------------------------------------------

impl<C, F, K> AdaptivePI<C, F, K>
where
    C: ChernoffFunction<F>,
    F: SemiflowFloat,
    K: StepController<F>,
    C::S: State<F> + Clone,
{
    /// Integrate `u0` forward by time `t` using adaptive substeps.
    ///
    /// Returns [`AdaptiveOutcome`] with final state and diagnostics.
    /// The initial substep size is `t * 0.01` (100-step heuristic).
    ///
    /// **v2.0 change**: `&mut self` (controller carries mutable state).
    ///
    /// # Errors
    ///
    /// - [`SemiflowError::DomainViolation`] if `t <= 0` or non-finite.
    /// - Any error from `func.apply_into` propagates unchanged.
    /// - [`SemiflowError::AdaptiveStepRejected`] if total substeps ≥ `max_substeps`.
    ///
    /// # Panics
    ///
    /// Does not panic in practice. The internal `.unwrap()` on `state_scratch` is
    /// an invariant upheld by the lazy initialisation that runs before the loop.
    pub fn evolve_adaptive(
        &mut self,
        t: F,
        u0: &C::S,
    ) -> Result<AdaptiveOutcome<C::S, F>, SemiflowError> {
        let t_f64 = t.to_f64().unwrap_or(0.0);
        if !t_f64.is_finite() || t_f64 <= 0.0 {
            return Err(SemiflowError::DomainViolation {
                what: "AdaptivePI: t must be finite and > 0",
                value: t_f64,
            });
        }
        self.controller.reset();
        if self.state_scratch.is_none() {
            self.state_scratch = Some([u0.clone(), u0.clone(), u0.clone()]);
        }
        let p = self.func.order();
        let two = F::from(2.0).unwrap_or(F::one() + F::one());
        self.adaptive_loop(t, p, two, u0)
    }

    /// Inner loop: advance `u0` by total time `t` using adaptive substeps.
    ///
    /// Assumes `state_scratch` is already initialised and `controller` is reset.
    fn adaptive_loop(
        &mut self,
        t: F,
        p: u32,
        two: F,
        u0: &C::S,
    ) -> Result<AdaptiveOutcome<C::S, F>, SemiflowError> {
        let mut u_curr = u0.clone();
        let mut t_curr = F::zero();
        let mut tau = t / F::from(100.0).unwrap_or(F::one());
        let mut last_err = F::one();
        let mut steps_accepted = 0_usize;
        let mut steps_rejected = 0_usize;
        loop {
            let total = steps_accepted + steps_rejected;
            if total >= self.max_substeps {
                return Err(SemiflowError::AdaptiveStepRejected {
                    last_tau: tau.to_f64().unwrap_or(0.0),
                    last_err: last_err.to_f64().unwrap_or(0.0),
                    steps_attempted: total,
                });
            }
            let tau_step = if tau < t - t_curr { tau } else { t - t_curr };
            let (accepted, new_tau, err_norm) = self.substep(tau_step, &u_curr, p, two)?;
            last_err = err_norm;
            tau = new_tau;
            if accepted {
                u_curr.copy_from(&self.state_scratch.as_ref().unwrap()[2]);
                t_curr += tau_step;
                steps_accepted += 1;
                if t - t_curr <= F::epsilon() * t {
                    break;
                }
            } else {
                steps_rejected += 1;
            }
        }
        Ok(AdaptiveOutcome {
            final_state: u_curr,
            steps_accepted,
            steps_rejected,
            last_tau: tau,
        })
    }

    /// Execute one adaptive substep.
    ///
    /// Returns `(accepted: bool, new_tau: F, err_norm: F)`.
    /// Reads/writes `self.state_scratch` in-place (zero-alloc after first call).
    fn substep(
        &mut self,
        tau_step: F,
        u_curr: &C::S,
        p: u32,
        two: F,
    ) -> Result<(bool, F, F), SemiflowError>
    where
        C::S: State<F> + Clone,
    {
        // Safety: state_scratch is always Some (ensured by caller).
        let [s_full, s_half_a, s_half] = self.state_scratch.as_mut().unwrap();

        // Full step: s_full = S(τ)·u_curr
        self.func
            .apply_into(tau_step, u_curr, s_full, &mut self.scratch)?;
        // Two half steps: s_half = S(τ/2)²·u_curr (s_half_a is intermediate carry)
        self.func
            .apply_into(tau_step / two, u_curr, s_half_a, &mut self.scratch)?;
        self.func
            .apply_into(tau_step / two, s_half_a, s_half, &mut self.scratch)?;

        // Richardson error (zero-alloc sup-norm; reuse s_half_a as diff scratch).
        // SAFEGUARD path §5.4: sup-norm preserves accepted-τ byte-identity.
        s_half_a.copy_from(s_half);
        s_half_a.axpy_into(F::zero() - F::one(), s_full);
        let err_norm = compute_richardson_err(s_half_a, p);

        // Mixed tolerance.
        let tol = self.tol_abs + self.tol_rel * u_curr.norm_sup().max(s_full.norm_sup());
        let (accepted, factor) = if err_norm <= tol {
            (
                true,
                self.controller
                    .propose_accept(err_norm, tol, self.safety, p),
            )
        } else {
            (
                false,
                self.controller
                    .propose_reject(err_norm, tol, self.safety, p),
            )
        };
        let new_tau = clamp_step(tau_step * factor, tau_step, self.min_ratio, self.max_ratio);
        Ok((accepted, new_tau, err_norm))
    }
}

/// Compute Richardson error norm: `sup-norm(diff) / (2^p - 1)`.
///
/// `p ≤ 6` so `(1u64<<p)-1 ≤ 63`, exact in f64/f32.
#[allow(clippy::cast_precision_loss)]
#[inline]
fn compute_richardson_err<F: SemiflowFloat, S: State<F>>(diff: &S, p: u32) -> F {
    let divisor_f64 = ((1u64 << p) - 1) as f64;
    let divisor = F::from(divisor_f64).unwrap_or(F::one());
    diff.norm_sup() / divisor
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Clamp the new step size to `[min_ratio * prev_tau, max_ratio * prev_tau]`.
#[inline]
fn clamp_step<F: SemiflowFloat>(new_tau: F, prev_tau: F, min_ratio: F, max_ratio: F) -> F {
    new_tau.clamp(min_ratio * prev_tau, max_ratio * prev_tau)
}
