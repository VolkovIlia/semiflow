//! Semilinear reaction–diffusion systems on any Chernoff engine (ADR-0208).
//!
//! ```text
//! ∂ₜu_k = L_k u_k + f_k(t, x, u_1, …, u_K),   k = 1..K
//! ```
//!
//! Each species `k` diffuses with its own engine `C_k ≈ e^{τL_k}` — any
//! [`ChernoffFunction`] on a [`NodalField`] state: 1-D/2-D/3-D grids with any
//! boundary policy and interpolant, Strang/ADI compositions, graph Laplacians —
//! and all species react through one [`Kinetics`]. A step is the Strang splitting
//!
//! ```text
//! U(t + τ) = R(t + τ/2, τ/2) ∘ L(τ) ∘ R(t, τ/2) U(t),
//! ```
//!
//! `R` the pointwise reaction flow (exact when the reaction provides it, otherwise
//! classical RK4 with `r` substeps or, for stiff kinetics, adaptive ROS2 —
//! [`ReactionSolver`], ADR-0212), `L` the per-species Chernoff step.
//! [`ReactionDiffusion::evolve_into`] merges the adjacent half reaction steps
//! (`R(τ/2)LR(τ)L…LR(τ/2)`). Order: `min(2, min_k order(C_k))`; Richardson
//! extrapolation (`crate::richardson`) raises it.
//!
//! **Cross-diffusion** (ADR-0213): with [`ReactionDiffusion::with_cross_diffusion`]
//! the diffusion is `DΔ` for a constant `K×K` matrix `D = PΛP⁻¹`
//! ([`CrossDiffusion`]); the engines then diffuse the MODES (`engines[k] ≈
//! e^{τλ_k L}`), and each diffusion step mixes the species into modes `P⁻¹u` and
//! back.

use alloc::{boxed::Box, vec::Vec};

use crate::{
    chernoff::ChernoffFunction,
    cross_diffusion::CrossDiffusion,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    reaction::Kinetics,
    reaction_stiff::{ros2_flow, ReactionSolver, RosBuffers},
    scratch::ScratchPool,
    state::State,
    step_plan::PencilPlan,
};

/// A state with node values and node coordinates (the reaction's view of it).
pub trait NodalField<F: SemiflowFloat>: State<F> {
    /// Node values.
    fn node_values(&self) -> &[F];
    /// Mutable node values.
    fn node_values_mut(&mut self) -> &mut [F];
    /// Node coordinates, axis-major (`x[d·N + i]`); empty on a graph.
    fn node_coordinates(&self) -> Vec<F>;
    /// `true` when a pointwise linear combination of this field's and `other`'s
    /// node values, carried on this field, evolves under any linear engine as the
    /// same combination of their evolutions: the same nodes and interpolant and
    /// the same homogeneous linear boundary policy (cross-diffusion, ADR-0213).
    /// Default `false`: fields that cannot be verified are rejected.
    fn mixes_with(&self, other: &Self) -> bool {
        let _ = other;
        false
    }
}

#[path = "reaction_diffusion_fields.rs"]
mod fields;

/// `K` fields, one per species, on the same nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct Species<S> {
    /// The species' fields, in reaction order.
    pub fields: Vec<S>,
}

impl<S> Species<S> {
    /// Wrap `fields` (one per species).
    #[must_use]
    pub fn new(fields: Vec<S>) -> Self {
        Self { fields }
    }
}

impl<F: SemiflowFloat, S: State<F>> State<F> for Species<S> {
    fn len(&self) -> usize {
        self.fields.iter().map(State::len).sum()
    }
    fn axpy_into(&mut self, alpha: F, src: &Self) {
        for (a, b) in self.fields.iter_mut().zip(&src.fields) {
            a.axpy_into(alpha, b);
        }
    }
    fn copy_from(&mut self, src: &Self) {
        for (a, b) in self.fields.iter_mut().zip(&src.fields) {
            a.copy_from(b);
        }
    }
    fn zero_into(&mut self) {
        self.fields.iter_mut().for_each(State::zero_into);
    }
    fn norm_sup(&self) -> F {
        self.fields
            .iter()
            .map(State::norm_sup)
            .fold(F::zero(), |a, b| {
                if a.is_nan() || b.is_nan() {
                    F::nan()
                } else if b > a {
                    b
                } else {
                    a
                }
            })
    }
    fn scale_into(&mut self, k: F) {
        for f in &mut self.fields {
            f.scale_into(k);
        }
    }
}

/// Boxed per-species engine.
pub type SpeciesEngine<'a, F, S> = Box<dyn ChernoffFunction<F, S = S> + Send + Sync + 'a>;

/// Strang-split reaction–diffusion system (see the module notes).
pub struct ReactionDiffusion<'a, F: SemiflowFloat, S: NodalField<F>> {
    engines: Vec<SpeciesEngine<'a, F, S>>,
    reaction: Box<dyn Kinetics<F> + 'a>,
    substeps: u32,
    solver: ReactionSolver,
    cross: Option<CrossDiffusion<F>>,
}

impl<'a, F: SemiflowFloat, S: NodalField<F> + Clone> ReactionDiffusion<'a, F, S> {
    /// One engine per species, in the reaction's species order.
    ///
    /// # Errors
    /// `DomainViolation` if there are no engines or their count differs from
    /// `reaction.species()`.
    pub fn new(
        engines: Vec<SpeciesEngine<'a, F, S>>,
        reaction: Box<dyn Kinetics<F> + 'a>,
    ) -> Result<Self, SemiflowError> {
        if engines.is_empty() || engines.len() != reaction.species() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "ReactionDiffusion: need one engine per reaction species (K >= 1)",
                value: engines.len() as f64,
            });
        }
        Ok(Self {
            engines,
            reaction,
            substeps: 1,
            solver: ReactionSolver::Rk4,
            cross: None,
        })
    }

    /// Cross-diffusion `DΔ` with the decomposition `cd` of `D` (ADR-0213): the
    /// engines must be the MODE engines, `engines[k] ≈ e^{τλ_k L}` with
    /// `λ_k = cd.eigenvalues()[k]` (an identity engine for `λ_k = 0`). Every species
    /// field must [`NodalField::mixes_with`] the others (checked at each run).
    ///
    /// # Errors
    /// `DomainViolation` if `cd` has a different species count.
    pub fn with_cross_diffusion(mut self, cd: CrossDiffusion<F>) -> Result<Self, SemiflowError> {
        if cd.species() != self.species() {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "ReactionDiffusion::with_cross_diffusion: D must be K×K for K species",
                value: cd.species() as f64,
            });
        }
        self.cross = Some(cd);
        Ok(self)
    }

    /// Reaction integrator when the kinetics has no exact flow (default
    /// [`ReactionSolver::Rk4`]); [`ReactionSolver::Rosenbrock`] for stiff kinetics
    /// (adaptive, L-stable; options validated at the first step).
    #[must_use]
    pub fn with_reaction_solver(mut self, solver: ReactionSolver) -> Self {
        self.solver = solver;
        self
    }

    /// RK4 substeps per reaction half-step (default 1, min 1). Ignored when the
    /// reaction has an exact flow or the solver is [`ReactionSolver::Rosenbrock`].
    #[must_use]
    pub fn with_reaction_substeps(mut self, substeps: u32) -> Self {
        self.substeps = substeps.max(1);
        self
    }

    /// Number of species `K`.
    #[must_use]
    pub fn species(&self) -> usize {
        self.engines.len()
    }

    /// Global order in `τ`: `min(2, min_k order(C_k))`.
    #[must_use]
    pub fn order(&self) -> u32 {
        self.engines.iter().map(|e| e.order()).fold(2, u32::min)
    }

    /// One Strang step `U ← R(t+τ/2, τ/2)∘L(τ)∘R(t, τ/2) U`, in place.
    ///
    /// # Errors
    /// Shape mismatch, a non-finite reaction result, or any engine error.
    pub fn step_into(
        &self,
        t: F,
        tau: F,
        u: &mut Species<S>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let half = tau * from_f64::<F>(0.5);
        self.run(u, scratch, |ws, u, scratch| {
            ws.react(self, t, half, u)?;
            ws.diffuse(self, tau, u, scratch)?;
            ws.react(self, t + half, half, u)
        })
    }

    /// `n` Strang steps from `src` at time `t0` over `[t0, t0 + t]` into `dst`,
    /// adjacent reaction half-steps merged.
    ///
    /// # Errors
    /// `t` negative or non-finite, `n == 0`, shape mismatch, a non-finite
    /// reaction result, or any engine error.
    #[allow(clippy::too_many_arguments)] // (t0, t, n) + (src, dst, scratch)
    pub fn evolve_into(
        &self,
        t0: F,
        t: F,
        n: usize,
        src: &Species<S>,
        dst: &mut Species<S>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        validate_horizon(t, n)?;
        #[allow(clippy::cast_precision_loss)]
        let tau = t / from_f64::<F>(n as f64);
        let half = tau * from_f64::<F>(0.5);
        let mut u = src.clone();
        self.run(&mut u, scratch, |ws, u, scratch| {
            ws.react(self, t0, half, u)?;
            for j in 0..n {
                ws.diffuse(self, tau, u, scratch)?;
                #[allow(clippy::cast_precision_loss)]
                let tj = t0 + tau * from_f64::<F>(j as f64) + half;
                let h = if j + 1 == n { half } else { tau };
                ws.react(self, tj, h, u)?;
            }
            Ok(())
        })?;
        dst.copy_from(&u);
        Ok(())
    }

    /// Validate shapes, build the workspace, run `body`, return the buffers.
    fn run(
        &self,
        u: &mut Species<S>,
        scratch: &mut ScratchPool<F>,
        body: impl FnOnce(
            &mut Workspace<F, S>,
            &mut Species<S>,
            &mut ScratchPool<F>,
        ) -> Result<(), SemiflowError>,
    ) -> Result<(), SemiflowError> {
        let n = check_shapes(self.species(), u)?;
        if self.cross.is_some() && u.fields.iter().any(|f| !u.fields[0].mixes_with(f)) {
            return Err(SemiflowError::DomainViolation {
                what: "ReactionDiffusion: cross-diffusion needs every species on the same nodes with one homogeneous linear boundary (no non-zero Dirichlet value)",
                value: 0.0,
            });
        }
        if let ReactionSolver::Rosenbrock(opts) = &self.solver {
            opts.validate()?;
        }
        let mut ws = Workspace::new(u, n, &self.solver, scratch);
        let result = body(&mut ws, u, scratch);
        ws.release(scratch);
        result
    }
}

/// `t ≥ 0` finite and `n ≥ 1`.
fn validate_horizon<F: SemiflowFloat>(t: F, n: usize) -> Result<(), SemiflowError> {
    if !t.is_finite() || t < F::zero() || n == 0 {
        return Err(SemiflowError::DomainViolation {
            what: "ReactionDiffusion::evolve_into: need finite t >= 0 and n >= 1",
            value: t.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

/// Node count shared by all `k` species fields.
fn check_shapes<F: SemiflowFloat, S: NodalField<F>>(
    k: usize,
    u: &Species<S>,
) -> Result<usize, SemiflowError> {
    let n = u.fields.first().map_or(0, |f| f.node_values().len());
    if u.fields.len() != k || n == 0 || u.fields.iter().any(|f| f.node_values().len() != n) {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "ReactionDiffusion: need K non-empty species fields with equal node counts",
            value: u.fields.len() as f64,
        });
    }
    Ok(n)
}

/// Buffers of one evolution: flat species values, RK4 stages, coordinates, and
/// a spare field for the diffusion ping-pong.
struct Workspace<F: SemiflowFloat, S> {
    n: usize,
    x: Vec<F>,
    flat: Vec<F>,
    acc: Vec<F>,
    stage: Vec<F>,
    k: Vec<F>,
    spare: S,
    ros: Option<RosBuffers<F>>,
    /// One τ-keyed plan per species engine (ADR-0215).
    plans: Vec<PencilPlan<F>>,
}

impl<F: SemiflowFloat, S: NodalField<F> + Clone> Workspace<F, S> {
    fn new(
        u: &Species<S>,
        n: usize,
        solver: &ReactionSolver,
        scratch: &mut ScratchPool<F>,
    ) -> Self {
        let kn = u.fields.len() * n;
        let ros = matches!(solver, ReactionSolver::Rosenbrock(_))
            .then(|| RosBuffers::new(u.fields.len(), n, scratch));
        Self {
            n,
            x: u.fields[0].node_coordinates(),
            flat: scratch.take_vec(kn),
            acc: scratch.take_vec(kn),
            stage: scratch.take_vec(kn),
            k: scratch.take_vec(kn),
            spare: u.fields[0].clone(),
            ros,
            plans: u.fields.iter().map(|_| PencilPlan::new()).collect(),
        }
    }

    fn release(self, scratch: &mut ScratchPool<F>) {
        for b in [self.flat, self.acc, self.stage, self.k] {
            scratch.return_vec(b);
        }
        if let Some(r) = self.ros {
            r.release(scratch);
        }
        for p in self.plans {
            p.release(scratch);
        }
    }

    /// `u_k ← C_k(τ) u_k` for every species; with cross-diffusion the species are
    /// mixed to modes `P⁻¹u` before and back `Pm` after.
    fn diffuse(
        &mut self,
        rd: &ReactionDiffusion<'_, F, S>,
        tau: F,
        u: &mut Species<S>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if let Some(cd) = &rd.cross {
            gather(&mut self.flat, u, self.n);
            cd.to_modes(&self.flat, &mut self.acc, self.n);
            scatter(&self.acc, u, self.n);
        }
        let pairs = rd.engines.iter().zip(u.fields.iter_mut());
        for ((engine, field), plan) in pairs.zip(self.plans.iter_mut()) {
            plan.apply(engine.as_ref(), tau, field, &mut self.spare, scratch)?;
            core::mem::swap(field, &mut self.spare);
        }
        if let Some(cd) = &rd.cross {
            gather(&mut self.flat, u, self.n);
            cd.to_species(&self.flat, &mut self.acc, self.n);
            scatter(&self.acc, u, self.n);
        }
        Ok(())
    }

    /// `u ← R(t, h) u`: gather, exact flow or RK4 substeps, check, scatter.
    fn react(
        &mut self,
        rd: &ReactionDiffusion<'_, F, S>,
        t: F,
        h: F,
        u: &mut Species<S>,
    ) -> Result<(), SemiflowError> {
        let n = self.n;
        gather(&mut self.flat, u, n);
        if let Some(r) = rd.reaction.exact_flow(t, h, n, &self.x, &mut self.flat) {
            r?;
        } else if let (ReactionSolver::Rosenbrock(opts), Some(buf)) = (&rd.solver, &mut self.ros) {
            let kin = rd.reaction.as_ref();
            ros2_flow(kin, opts, t, h, n, &self.x, &mut self.flat, buf)?;
        } else {
            let sub = h / from_f64::<F>(f64::from(rd.substeps));
            for j in 0..rd.substeps {
                self.rk4(
                    rd.reaction.as_ref(),
                    t + sub * from_f64::<F>(f64::from(j)),
                    sub,
                )?;
            }
        }
        if let Some(bad) = self.flat.iter().find(|v| !v.is_finite()) {
            return Err(SemiflowError::DomainViolation {
                what: "ReactionDiffusion: non-finite reaction result (stiff? use the Rosenbrock reaction solver or more substeps)",
                value: bad.to_f64().unwrap_or(f64::NAN),
            });
        }
        scatter(&self.flat, u, n);
        Ok(())
    }

    /// One classical RK4 step of `u′ = f(t, x, u)` on `self.flat`.
    fn rk4(&mut self, reaction: &dyn Kinetics<F>, t: F, h: F) -> Result<(), SemiflowError> {
        let (n, two) = (self.n, from_f64::<F>(2.0));
        let half = h * from_f64::<F>(0.5);
        reaction.eval_batch(t, n, &self.x, &self.flat, &mut self.k)?;
        self.acc.copy_from_slice(&self.k);
        for (stage_h, weight, stage_t) in [
            (half, two, t + half),
            (half, two, t + half),
            (h, F::one(), t + h),
        ] {
            for ((s, &u), &k) in self.stage.iter_mut().zip(&self.flat).zip(&self.k) {
                *s = u + stage_h * k;
            }
            reaction.eval_batch(stage_t, n, &self.x, &self.stage, &mut self.k)?;
            for (a, &k) in self.acc.iter_mut().zip(&self.k) {
                *a += weight * k;
            }
        }
        let sixth = h / from_f64::<F>(6.0);
        for (u, &a) in self.flat.iter_mut().zip(&self.acc) {
            *u += sixth * a;
        }
        Ok(())
    }
}

/// Species values into the flat species-major buffer (`flat[k·n + i]`).
fn gather<F: SemiflowFloat, S: NodalField<F>>(flat: &mut [F], u: &Species<S>, n: usize) {
    for (k, field) in u.fields.iter().enumerate() {
        flat[k * n..(k + 1) * n].copy_from_slice(field.node_values());
    }
}

/// The flat species-major buffer back into the species fields.
fn scatter<F: SemiflowFloat, S: NodalField<F>>(flat: &[F], u: &mut Species<S>, n: usize) {
    for (k, field) in u.fields.iter_mut().enumerate() {
        field
            .node_values_mut()
            .copy_from_slice(&flat[k * n..(k + 1) * n]);
    }
}

#[cfg(test)]
#[path = "reaction_diffusion_tests.rs"]
mod tests;
