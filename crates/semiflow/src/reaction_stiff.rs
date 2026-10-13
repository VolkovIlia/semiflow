//! Stiff kinetics: adaptive two-stage Rosenbrock (ROS2) reaction flow (ADR-0212,
//! math §69).
//!
//! ROS2 (Verwer, Spee, Blom & Hundsdorfer, SIAM J. Sci. Comput. 20 (1999)),
//! `γ = 1 + 1/√2`, per node `W = I − γhJ`:
//! `W k₁ = f(t, u)`, `W k₂ = f(t + h, u + hk₁) − 2k₁`, `u⁺ = u + h(3k₁ + k₂)/2`.
//! L-stable (`R(∞) = 0`) and of order 2 for ANY matrix `J` (a W-method), so a
//! forward-difference Jacobian or the omitted `∂f/∂t` keep order 2.
//! Local error `u⁺ − (u + hk₁) = h(k₁ + k₂)/2` in the mixed norm
//! `max_{k,i} |e|/(atol + rtol·max(|u|, |u⁺|))`; a step is accepted at `≤ 1`, the
//! next one is `h·clamp(0.9/√err, 0.2, 5)`. One step size for the whole field, so
//! every stage is ONE batched kinetics call; the controller is pure arithmetic and
//! the result reproducible bit for bit. A singular `W` or a non-finite stage is a
//! rejection (`×0.2`); a non-finite `f(t, u)` or `J` at an accepted state, or a step
//! below `16ε` of the half-step clock, is a `DomainViolation` (no step can fix it).

use alloc::{vec, vec::Vec};

use crate::{
    dense_la::{lu_in_place, lu_solve},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    reaction::Kinetics,
    scratch::ScratchPool,
};

/// Which integrator the reaction half-steps of [`crate::ReactionDiffusion`] use
/// when the kinetics has no exact flow.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum ReactionSolver {
    /// Classical RK4 with `with_reaction_substeps` fixed substeps (default).
    #[default]
    Rk4,
    /// Adaptive ROS2 for stiff kinetics (module docs).
    Rosenbrock(RosenbrockOptions),
}

/// Tolerances of [`ReactionSolver::Rosenbrock`]; build with `default()` and `with_*`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RosenbrockOptions {
    /// Relative tolerance `∈ (0, 1)`. Default `1e−6`.
    pub rtol: f64,
    /// Absolute tolerance `≥ 0` (`0`: pure relative control; steps near a
    /// vanishing species may then shrink). Default `1e−9`.
    pub atol: f64,
    /// Step cap per reaction half-step (accepted plus rejected). Default 100 000.
    pub max_steps: usize,
}

impl Default for RosenbrockOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-9,
            max_steps: 100_000,
        }
    }
}

impl RosenbrockOptions {
    /// Set `rtol`.
    #[must_use]
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }
    /// Set `atol`.
    #[must_use]
    pub fn with_atol(mut self, atol: f64) -> Self {
        self.atol = atol;
        self
    }
    /// Set the step cap.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// `DomainViolation` unless `rtol ∈ (0, 1)`, `atol ≥ 0` finite, `max_steps ≥ 1`.
    pub(crate) fn validate(&self) -> Result<(), SemiflowError> {
        let bad = |what, value| Err(SemiflowError::DomainViolation { what, value });
        if !(self.rtol > 0.0 && self.rtol < 1.0) {
            return bad("Rosenbrock: rtol must be in (0, 1)", self.rtol);
        }
        if !(self.atol >= 0.0 && self.atol.is_finite()) {
            return bad("Rosenbrock: atol must be finite and >= 0", self.atol);
        }
        if self.max_steps == 0 {
            return bad("Rosenbrock: max_steps must be >= 1", 0.0);
        }
        Ok(())
    }
}

/// Stage buffers of one evolution (scratch-pool vectors; `piv` is a plain `Vec`).
pub(crate) struct RosBuffers<F> {
    k: usize,
    f0: Vec<F>,
    k1: Vec<F>,
    k2: Vec<F>,
    stage: Vec<F>,
    next: Vec<F>,
    jac: Vec<F>,
    lu: Vec<F>,
    work: Vec<F>,
    piv: Vec<usize>,
}

impl<F: SemiflowFloat> RosBuffers<F> {
    pub(crate) fn new(k: usize, n: usize, scratch: &mut ScratchPool<F>) -> Self {
        Self {
            k,
            f0: scratch.take_vec(k * n),
            k1: scratch.take_vec(k * n),
            k2: scratch.take_vec(k * n),
            stage: scratch.take_vec(k * n),
            next: scratch.take_vec(k * n),
            jac: scratch.take_vec(k * k * n),
            lu: scratch.take_vec(k * k * n),
            work: scratch.take_vec(2 * k),
            piv: vec![0; k * n],
        }
    }

    pub(crate) fn release(self, scratch: &mut ScratchPool<F>) {
        for b in [
            self.f0, self.k1, self.k2, self.stage, self.next, self.jac, self.lu, self.work,
        ] {
            scratch.return_vec(b);
        }
    }
}

/// `γ = 1 + 1/√2`.
fn gamma<F: SemiflowFloat>() -> F {
    from_f64(1.0 + core::f64::consts::FRAC_1_SQRT_2)
}

/// `u ← Φ_h(u)` of `u′ = f(t, x, u)` by adaptive ROS2 (module docs); returns the
/// number of attempted steps.
///
/// # Errors
/// Any kinetics error; `DomainViolation` for a non-finite `f(t, u)` or `J` at an
/// accepted state (no step size can repair it) or a step size that underflows the
/// half-step clock; `ConvergenceFailed` after `max_steps` attempts.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)] // Kinetics signature
pub(crate) fn ros2_flow<F: SemiflowFloat>(
    kin: &dyn Kinetics<F>,
    opts: &RosenbrockOptions,
    t: F,
    h: F,
    n: usize,
    x: &[F],
    u: &mut [F],
    b: &mut RosBuffers<F>,
) -> Result<usize, SemiflowError> {
    let (mut hs, mut elapsed, mut steps) = (h, F::zero(), 0_usize);
    while elapsed < h {
        let tc = t + elapsed;
        kin.eval_batch(tc, n, x, u, &mut b.f0)?;
        jacobian(kin, tc, n, x, u, b)?;
        check_finite(&b.f0, &b.jac)?;
        let mut last = hs >= h - elapsed;
        loop {
            if steps >= opts.max_steps {
                return Err(SemiflowError::ConvergenceFailed {
                    last_residual: hs.to_f64().unwrap_or(f64::NAN),
                    max_iter: opts.max_steps,
                });
            }
            if last {
                hs = h - elapsed;
            } else if stalled(hs, elapsed) {
                return Err(SemiflowError::DomainViolation {
                    what: "Rosenbrock: step size underflow (kinetics non-finite near the state, or tolerances unattainable)",
                    value: hs.to_f64().unwrap_or(f64::NAN),
                });
            }
            steps += 1;
            let err = attempt(kin, opts, tc, hs, n, x, u, b)?;
            if err <= 1.0 {
                u.copy_from_slice(&b.next);
                elapsed = if last { h } else { elapsed + hs };
                hs *= from_f64::<F>(next_factor(err));
                break;
            }
            hs *= from_f64::<F>(next_factor(err).min(1.0));
            last = false;
        }
    }
    Ok(steps)
}

/// `f(t, u)` and `J` at an accepted state are finite (they do not depend on the
/// step size, so a rejection could never repair them).
fn check_finite<F: SemiflowFloat>(f0: &[F], jac: &[F]) -> Result<(), SemiflowError> {
    match f0.iter().chain(jac).find(|v| !v.is_finite()) {
        Some(bad) => Err(SemiflowError::DomainViolation {
            what: "ReactionDiffusion: non-finite reaction result or Jacobian at an accepted state (Rosenbrock; kinetics left its domain?)",
            value: bad.to_f64().unwrap_or(f64::NAN),
        }),
        None => Ok(()),
    }
}

/// No usable progress on the half-step's LOCAL clock: `hs ≤ 16ε·elapsed` or
/// `elapsed + hs == elapsed` (the final step, `h − elapsed`, is exempt).
fn stalled<F: SemiflowFloat>(hs: F, elapsed: F) -> bool {
    hs <= from_f64::<F>(16.0) * F::epsilon() * elapsed || elapsed + hs == elapsed
}

/// `clamp(0.9/√err, 0.2, 5)`; a non-finite `err` (a failed stage) gives `0.2`.
fn next_factor(err: f64) -> f64 {
    if err.is_finite() {
        (0.9 / libm::sqrt(err.max(1e-300))).clamp(0.2, 5.0)
    } else {
        0.2
    }
}

/// One trial step of size `hs` from `u` (with `f0`, `jac` at `u`): fills `next`,
/// returns the local error in the mixed norm (`∞` when `W` is singular or a stage
/// is not finite — the caller then shrinks the step).
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn attempt<F: SemiflowFloat>(
    kin: &dyn Kinetics<F>,
    opts: &RosenbrockOptions,
    tc: F,
    hs: F,
    n: usize,
    x: &[F],
    u: &[F],
    b: &mut RosBuffers<F>,
) -> Result<f64, SemiflowError> {
    if factor_w(hs, n, b).is_err() {
        return Ok(f64::INFINITY);
    }
    b.k1.copy_from_slice(&b.f0);
    solve_nodes(n, &b.lu, &b.piv, &mut b.k1, &mut b.work, b.k);
    for ((s, &ui), &k1) in b.stage.iter_mut().zip(u).zip(&b.k1) {
        *s = ui + hs * k1;
    }
    kin.eval_batch(tc + hs, n, x, &b.stage, &mut b.k2)?;
    let two = from_f64::<F>(2.0);
    for (k2, &k1) in b.k2.iter_mut().zip(&b.k1) {
        *k2 -= two * k1;
    }
    solve_nodes(n, &b.lu, &b.piv, &mut b.k2, &mut b.work, b.k);
    let (w1, w2, half) = (from_f64::<F>(1.5), from_f64::<F>(0.5), from_f64::<F>(0.5));
    let mut err = 0.0_f64;
    for (i, nx) in b.next.iter_mut().enumerate() {
        let (k1, k2) = (b.k1[i], b.k2[i]);
        *nx = u[i] + hs * (w1 * k1 + w2 * k2);
        let e = (hs * half * (k1 + k2)).abs().to_f64().unwrap_or(f64::NAN);
        let scale = opts.atol + opts.rtol * u[i].abs().max(nx.abs()).to_f64().unwrap_or(f64::NAN);
        // A zero error contributes 0 even at a zero scale (`atol = 0` and a species
        // that is exactly 0 before and after the step).
        let ratio = if e == 0.0 { 0.0 } else { e / scale };
        err = if ratio.is_nan() || !nx.is_finite() {
            f64::INFINITY
        } else {
            err.max(ratio)
        };
    }
    Ok(err)
}

/// `W_i = I − γ·hs·J_i`, factored in place per node.
fn factor_w<F: SemiflowFloat>(hs: F, n: usize, b: &mut RosBuffers<F>) -> Result<(), SemiflowError> {
    let k = b.k;
    let kk = k * k;
    let gh = gamma::<F>() * hs;
    for i in 0..n {
        let w = &mut b.lu[i * kk..(i + 1) * kk];
        for (e, wv) in w.iter_mut().enumerate() {
            let id = if e / k == e % k { F::one() } else { F::zero() };
            *wv = id - gh * b.jac[e * n + i];
        }
        lu_in_place(k, w, &mut b.piv[i * k..(i + 1) * k])?;
    }
    Ok(())
}

/// `v_i ← W_i⁻¹ v_i` for every node (`v` species-major).
fn solve_nodes<F: SemiflowFloat>(
    n: usize,
    lu: &[F],
    piv: &[usize],
    v: &mut [F],
    work: &mut [F],
    k: usize,
) {
    let (node, scratch) = work.split_at_mut(k);
    for i in 0..n {
        for (s, nv) in node.iter_mut().enumerate() {
            *nv = v[s * n + i];
        }
        lu_solve(
            k,
            &lu[i * k * k..(i + 1) * k * k],
            &piv[i * k..(i + 1) * k],
            node,
            scratch,
        );
        for (s, &nv) in node.iter().enumerate() {
            v[s * n + i] = nv;
        }
    }
}

/// `b.jac` at `(tc, u)`: the kinetics' own Jacobian, or forward differences
/// `(f(u + δe_l) − f(u))/δ`, `δ = √ε·max(1, |u_l|)` (exactly representable), one
/// species column per batched call. Needs `b.f0 = f(tc, u)`.
#[allow(clippy::many_single_char_names)] // the Kinetics signature
fn jacobian<F: SemiflowFloat>(
    kin: &dyn Kinetics<F>,
    tc: F,
    n: usize,
    x: &[F],
    u: &[F],
    b: &mut RosBuffers<F>,
) -> Result<(), SemiflowError> {
    if let Some(r) = kin.jacobian_batch(tc, n, x, u, &mut b.jac) {
        return r;
    }
    let k = b.k;
    let sq = F::epsilon().sqrt();
    b.stage.copy_from_slice(u);
    for l in 0..k {
        for i in 0..n {
            let base = u[l * n + i];
            b.stage[l * n + i] = base + sq * base.abs().max(F::one());
        }
        kin.eval_batch(tc, n, x, &b.stage, &mut b.k2)?;
        for i in 0..n {
            let delta = b.stage[l * n + i] - u[l * n + i];
            for row in 0..k {
                b.jac[(row * k + l) * n + i] = (b.k2[row * n + i] - b.f0[row * n + i]) / delta;
            }
            b.stage[l * n + i] = u[l * n + i];
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "reaction_stiff_tests.rs"]
mod tests;
