//! B2 — Howland Nonautonomous Lift (math.md §23, ADR-0070).
//!
//! Converts time-dependent L(s) to autonomous L̂ := -∂_s + L(s) on L²([0,T], X).
//! Chernoff approximation (left-endpoint shift): `F̂(τ) f̂(s) := F(τ, s−τ) f̂(s−τ)`.
//! Order = `min(C::order(), 1)`. Cite: Howland 1974 *Trans. AMS* **207** Theorem 1.
//!
//! L²([0,T], X) is discretized as `Vec<C::S>` of `n_t` uniform samples (`Δs = T/(n_t−1)`).
//! `apply_into` enforces `τ = Δs` (§23.4); mismatches return `Err(DomainViolation)`.
//! Wrapper-type blanket impls deferred to v3.x.

// Time-step count n_t (usize) cast to f64 for coordinate/index computations; ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

extern crate alloc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// TimedChernoffFunction<F> — super-trait
// ---------------------------------------------------------------------------

/// Time-aware Chernoff function super-trait (math §23.7, ADR-0070).
///
/// Extends [`ChernoffFunction<F>`] with `apply_at(t, tau, …)` that samples the
/// generator at absolute time `t`. Default impl ignores `t` and delegates to
/// `apply_into` (autonomous bridge). Autonomous leaf types satisfy this trait for
/// free via a one-liner marker impl. Time-dependent types override `apply_at`.
pub trait TimedChernoffFunction<F: SemiflowFloat = f64>: ChernoffFunction<F> {
    /// Apply the Chernoff step of duration `tau`, sampling the generator at
    /// absolute time `t` (left endpoint of the interval [t, t + tau]).
    ///
    /// Default: ignores `t` and delegates to `apply_into` (autonomous bridge).
    ///
    /// # Errors
    /// Returns [`SemiflowError`] if the underlying `apply_into` fails.
    fn apply_at(
        &self,
        t: F,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let _ = t; // autonomous bridge: t ignored
        self.apply_into(tau, src, dst, scratch)
    }
}

// ---------------------------------------------------------------------------
// HowlandState<S, F> — discretized L²([0,T], X)
// ---------------------------------------------------------------------------

/// Discretized L²([0,T], X): `n_t` uniform time samples (math §23.3).
///
/// `samples[i]` ≈ `f̂(s_i)` where `s_i` = i · Δs, Δs = T / (`n_t` − 1).
///
/// Implements [`State<F>`] by applying every `State<F>` primitive
/// component-wise over `samples`. Shapes must match (`n_t` must agree).
#[derive(Debug, Clone)]
pub struct HowlandState<S, F: SemiflowFloat = f64>
where
    S: State<F>,
{
    /// Time-slice samples: `samples[i]` ≈ f̂(i · Δs).
    pub samples: Vec<S>,
    /// Number of time samples (cached from `samples.len()`).
    pub n_t: usize,
    _f: PhantomData<F>,
}

impl<S, F> HowlandState<S, F>
where
    S: State<F>,
    F: SemiflowFloat,
{
    /// Construct from a non-empty Vec of samples.
    ///
    /// # Errors
    ///
    /// Returns [`SemiflowError::DomainViolation`] if `samples` is empty.
    pub fn new(samples: Vec<S>) -> Result<Self, SemiflowError> {
        if samples.is_empty() {
            return Err(SemiflowError::DomainViolation {
                what: "HowlandState: samples must be non-empty",
                value: 0.0,
            });
        }
        let n_t = samples.len();
        Ok(Self {
            samples,
            n_t,
            _f: PhantomData,
        })
    }
}

impl<S, F> State<F> for HowlandState<S, F>
where
    S: State<F>,
    F: SemiflowFloat,
{
    /// Total degrees of freedom = `n_t` × samples[0].`len()`.
    fn len(&self) -> usize {
        self.samples.iter().map(super::state::State::len).sum()
    }

    /// Component-wise axpy: `self.samples[i] += alpha * other.samples[i]`.
    ///
    /// # Panics (debug)
    /// `debug_assert_eq!(self.n_t, other.n_t)` — shape mismatch.
    fn axpy_into(&mut self, alpha: F, other: &Self) {
        debug_assert_eq!(self.n_t, other.n_t, "axpy_into: n_t mismatch");
        for (d, src) in self.samples.iter_mut().zip(&other.samples) {
            d.axpy_into(alpha, src);
        }
    }

    /// Component-wise copy: `self.samples[i] ← other.samples[i]`.
    ///
    /// # Panics (debug)
    /// `debug_assert_eq!(self.n_t, other.n_t)` — shape mismatch.
    fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.n_t, other.n_t, "copy_from: n_t mismatch");
        for (d, src) in self.samples.iter_mut().zip(&other.samples) {
            d.copy_from(src);
        }
    }

    /// Zero all time slices.
    fn zero_into(&mut self) {
        for s in &mut self.samples {
            s.zero_into();
        }
    }

    /// Sup-norm over all time slices: `max_i` ‖samples[i]‖_∞.
    fn norm_sup(&self) -> F {
        self.samples.iter().fold(F::zero(), |acc, s| {
            let n = s.norm_sup();
            if n > acc {
                n
            } else {
                acc
            }
        })
    }

    /// Component-wise scale: `self.samples[i] *= k`.
    fn scale_into(&mut self, k: F) {
        for s in &mut self.samples {
            s.scale_into(k);
        }
    }
}

// ---------------------------------------------------------------------------
// HowlandLift<C, F> — ChernoffFunction wrapper
// ---------------------------------------------------------------------------

/// Howland lifted Chernoff function for nonautonomous generators (math §23.8).
///
/// Wraps a [`TimedChernoffFunction<F>`] `C`; implements [`ChernoffFunction<F>`] on
/// `HowlandState<C::S, F>` via the left-endpoint shift (math 23.3):
/// `dst[i] = C.apply_at(s_{i-1}, Δs, src[i-1])` for `i ≥ 1`; `dst[0] = 0`.
/// Validates `n_t >= 2` and finite positive `t_horizon`. `apply_into` enforces
/// `|τ − Δs| ≤ ε·Δs` (§23.4).
#[derive(Debug, Clone)]
pub struct HowlandLift<C, F = f64>
where
    C: TimedChernoffFunction<F>,
    F: SemiflowFloat,
{
    inner: C,
    t_horizon: F,
    n_t: usize,
    delta_s: F,
    _f: PhantomData<F>,
}

impl<C, F> HowlandLift<C, F>
where
    C: TimedChernoffFunction<F>,
    F: SemiflowFloat,
{
    /// Construct with validated parameters.
    ///
    /// # Errors
    ///
    /// - [`SemiflowError::DomainViolation`] if `n_t < 2`.
    /// - [`SemiflowError::DomainViolation`] if `t_horizon` is not finite or ≤ 0.
    pub fn new(inner: C, t_horizon: F, n_t: usize) -> Result<Self, SemiflowError> {
        if n_t < 2 {
            return Err(SemiflowError::DomainViolation {
                what: "HowlandLift: n_t must be >= 2",
                value: n_t as f64,
            });
        }
        if !t_horizon.is_finite() || t_horizon <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "HowlandLift: t_horizon must be finite and > 0",
                value: t_horizon.to_f64().unwrap_or(f64::NAN),
            });
        }
        let denom = from_f64::<F>((n_t - 1) as f64);
        let delta_s = t_horizon / denom;
        Ok(Self {
            inner,
            t_horizon,
            n_t,
            delta_s,
            _f: PhantomData,
        })
    }

    /// Time grid spacing Δs = T / (`n_t` − 1).
    pub fn delta_s(&self) -> F {
        self.delta_s
    }

    /// Number of time samples.
    pub fn n_t(&self) -> usize {
        self.n_t
    }
}

/// Validate τ vs Δs; return Err if |τ − Δs| > ε·Δs (math §23.4).
fn check_matched_step<F: SemiflowFloat>(tau: F, delta_s: F) -> Result<(), SemiflowError> {
    let tau_err = (tau - delta_s).abs();
    let tol = F::epsilon() * delta_s;
    if tau_err > tol {
        return Err(SemiflowError::DomainViolation {
            what: "HowlandLift: tau must equal delta_s (matched-step requirement §23.4)",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

/// Validate src and dst `n_t` match `self.n_t`.
fn check_shape(src_n_t: usize, dst_n_t: usize, expected: usize) -> Result<(), SemiflowError> {
    if src_n_t != expected || dst_n_t != expected {
        return Err(SemiflowError::DomainViolation {
            what: "HowlandLift: src/dst n_t must match HowlandLift n_t",
            value: src_n_t as f64,
        });
    }
    Ok(())
}

impl<C, F> ChernoffFunction<F> for HowlandLift<C, F>
where
    C: TimedChernoffFunction<F>,
    C::S: Clone,
    F: SemiflowFloat,
{
    type S = HowlandState<C::S, F>;

    /// Single Howland-Chernoff step with τ = Δs (matched-step, §23.4).
    ///
    /// Implements the discretized shift formula (23.4):
    ///   dst[0] := 0       (boundary convention: f̂(s<0) ≡ 0)
    ///   dst[i] := `C.apply_at(t`_{i-1}, Δs, src[i-1])   for i ≥ 1
    ///
    /// where t_{i-1} = (i-1) · Δs is the left endpoint of step i.
    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        check_matched_step(tau, self.delta_s)?;
        check_shape(src.n_t, dst.n_t, self.n_t)?;

        // Boundary: slot 0 is always zero (f̂(s<0) ≡ 0, §23.5).
        dst.samples[0].zero_into();

        // Shift: dst[i] = C.apply_at(t_{i-1}, Δs, src[i-1]) for i >= 1.
        for i in 1..self.n_t {
            let t_prev = from_f64::<F>((i - 1) as f64) * self.delta_s;
            self.inner.apply_at(
                t_prev,
                self.delta_s,
                &src.samples[i - 1],
                &mut dst.samples[i],
                scratch,
            )?;
        }
        Ok(())
    }

    /// Order = `min(inner.order()`, 1) — left-endpoint shift is order-1 globally.
    fn order(&self) -> u32 {
        core::cmp::min(self.inner.order(), 1)
    }

    /// Growth bound on Ŷ = L²([0,T], X): `M_c` · exp(T · |`ω_c`|) (math §23.5).
    ///
    /// The time-shift is unitary on L²([0,T]); inner growth integrates over T.
    fn growth(&self) -> Growth<F> {
        let gc = self.inner.growth();
        let t_h = self.t_horizon;
        let m_c = gc.multiplier;
        let omega_c = gc.omega;
        Growth {
            multiplier: m_c * (t_h * omega_c.abs()).exp(),
            omega: F::zero(),
        }
    }
}

// ---------------------------------------------------------------------------
// Marker impls — autonomous leaves get TimedChernoffFunction for free
// ---------------------------------------------------------------------------

// f64-monomorphic leaves
impl TimedChernoffFunction<f64> for crate::shift1d::ShiftChernoff1D<f64> {}
impl TimedChernoffFunction<f64> for crate::diffusion::DiffusionChernoff<f64> {}
impl TimedChernoffFunction<f64> for crate::diffusion4::Diffusion4thChernoff<f64> {}
impl TimedChernoffFunction<f64> for crate::diffusion6::Diffusion6thChernoff<f64> {}
impl TimedChernoffFunction<f64> for crate::drift_reaction::DriftReactionChernoff<f64> {}
impl TimedChernoffFunction<f64> for crate::truncated_exp::TruncatedExpDiffusionChernoff<f64> {}
impl TimedChernoffFunction<f64> for crate::truncated_exp4::TruncatedExp4thDiffusionChernoff<f64> {}

// Fully-generic leaves
impl<F: SemiflowFloat> TimedChernoffFunction<F> for crate::graph_heat::GraphHeatChernoff<F> {}
impl<F: SemiflowFloat> TimedChernoffFunction<F> for crate::graph_heat4::GraphHeat4thChernoff<F> {}
impl<F: SemiflowFloat> TimedChernoffFunction<F> for crate::graph_heat6::GraphHeat6thChernoff<F> {}
#[rustfmt::skip]
impl<F: SemiflowFloat> TimedChernoffFunction<F> for crate::graph_var_coef::VarCoefGraphHeatChernoff<F> {}
impl<F: SemiflowFloat> TimedChernoffFunction<F> for crate::schrodinger::SchrodingerChernoff<F> {}
impl<C, F> TimedChernoffFunction<F> for crate::adjoint::AdjointChernoff<C, F>
where
    F: SemiflowFloat,
    C: TimedChernoffFunction<F>,
    C::S: crate::state::HilbertState<F>,
{
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
// Exact float comparisons in tests verify round-trip identity or sentinel values.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::{
        diffusion::DiffusionChernoff, grid::Grid1D, grid_fn::GridFn1D, scratch::ScratchPool,
    };

    // Helper: build a simple DiffusionChernoff on [-1, 1] with n=16.
    fn make_diffusion() -> DiffusionChernoff<f64> {
        let grid = Grid1D::new(-1.0, 1.0, 16).unwrap();
        DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid)
    }

    // Helper: build an initial HowlandState with sin(πx) on each slice.
    fn make_state(n_t: usize, grid: Grid1D<f64>) -> HowlandState<GridFn1D<f64>, f64> {
        let samples: Vec<_> = (0..n_t)
            .map(|_| GridFn1D::from_fn(grid, |x| (core::f64::consts::PI * x).sin()))
            .collect();
        HowlandState::new(samples).unwrap()
    }

    // --- HowlandState construction ---

    #[test]
    fn howland_state_empty_vec_err() {
        let samples: Vec<GridFn1D<f64>> = Vec::new();
        let result = HowlandState::new(samples);
        assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
    }

    #[test]
    fn howland_state_nonempty_ok() {
        let grid = Grid1D::new(-1.0, 1.0, 8).unwrap();
        let s = GridFn1D::from_fn(grid, |x| x);
        let result = HowlandState::new(vec![s.clone(), s]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().n_t, 2);
    }

    // --- HowlandLift construction ---

    #[test]
    fn howland_lift_n_t_too_small_err() {
        let diff = make_diffusion();
        let result = HowlandLift::new(diff, 1.0_f64, 1_usize);
        assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
    }

    #[test]
    fn howland_lift_n_t_zero_err() {
        let diff = make_diffusion();
        let result = HowlandLift::new(diff, 1.0_f64, 0_usize);
        assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
    }

    #[test]
    fn howland_lift_nonfinite_t_horizon_err() {
        let diff = make_diffusion();
        let result = HowlandLift::new(diff, f64::INFINITY, 11_usize);
        assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
    }

    #[test]
    fn howland_lift_zero_t_horizon_err() {
        let diff = make_diffusion();
        let result = HowlandLift::new(diff, 0.0_f64, 11_usize);
        assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
    }

    #[test]
    fn howland_lift_negative_t_horizon_err() {
        let diff = make_diffusion();
        let result = HowlandLift::new(diff, -1.0_f64, 11_usize);
        assert!(matches!(result, Err(SemiflowError::DomainViolation { .. })));
    }

    #[test]
    fn howland_lift_delta_s_correct() {
        let diff = make_diffusion();
        let lift = HowlandLift::new(diff, 1.0_f64, 11_usize).unwrap();
        // Δs = 1.0 / (11 - 1) = 0.1
        let expected = 0.1_f64;
        assert!(
            (lift.delta_s() - expected).abs() < 1e-14,
            "delta_s = {}, expected {}",
            lift.delta_s(),
            expected
        );
    }

    // --- apply_into: matched-step enforcement ---

    #[test]
    fn howland_lift_wrong_tau_err() {
        let diff = make_diffusion();
        let grid = Grid1D::new(-1.0, 1.0, 16).unwrap();
        let lift = HowlandLift::new(diff, 1.0_f64, 11_usize).unwrap();
        let src = make_state(11, grid);
        let mut dst = src.clone();
        let mut scratch = ScratchPool::new();
        // delta_s = 0.1; pass tau = 0.2 (wrong)
        let result = lift.apply_into(0.2_f64, &src, &mut dst, &mut scratch);
        assert!(
            matches!(result, Err(SemiflowError::DomainViolation { .. })),
            "expected DomainViolation for mismatched tau"
        );
    }

    // --- apply_into: smoke test with autonomous DiffusionChernoff ---

    #[test]
    fn howland_lift_apply_smoke() {
        let diff = make_diffusion();
        let grid = Grid1D::new(-1.0, 1.0, 16).unwrap();
        // n_t=11, t_horizon=1.0 → delta_s=0.1
        let lift = HowlandLift::new(diff, 1.0_f64, 11_usize).unwrap();
        let src = make_state(11, grid);
        let mut dst = src.clone();
        let mut scratch = ScratchPool::new();
        let tau = lift.delta_s();
        lift.apply_into(tau, &src, &mut dst, &mut scratch).unwrap();

        // dst[0] must be zero (boundary convention)
        assert_eq!(
            dst.samples[0].norm_sup(),
            0.0,
            "slot 0 must be zero after Howland step"
        );
        // dst[1] is the result of applying DiffusionChernoff to src[0] —
        // must be non-trivial (sin(πx) has non-zero norm under diffusion).
        assert!(
            dst.samples[1].norm_sup() > 0.0,
            "slot 1 should be non-zero after Howland step"
        );
        // Check order() returns 1 (min(2, 1) = 1 for DiffusionChernoff order-2 inner)
        assert_eq!(lift.order(), 1);
    }
}
