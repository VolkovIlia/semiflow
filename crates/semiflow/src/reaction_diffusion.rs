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
//! `R` the pointwise reaction flow (exact when the reaction provides it,
//! classical RK4 with `r` substeps otherwise), `L` the per-species Chernoff step.
//! [`ReactionDiffusion::evolve_into`] merges the adjacent half reaction steps
//! (`R(τ/2)LR(τ)L…LR(τ/2)`). Order: `min(2, min_k order(C_k))`; Richardson
//! extrapolation (`crate::richardson`) raises it.

use alloc::{boxed::Box, vec::Vec};

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph_signal::GraphSignal,
    grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
    grid_fn3d::GridFn3D,
    reaction::Kinetics,
    scratch::ScratchPool,
    state::State,
};

/// A state with node values and node coordinates (the reaction's view of it).
pub trait NodalField<F: SemiflowFloat>: State<F> {
    /// Node values.
    fn node_values(&self) -> &[F];
    /// Mutable node values.
    fn node_values_mut(&mut self) -> &mut [F];
    /// Node coordinates, axis-major (`x[d·N + i]`); empty on a graph.
    fn node_coordinates(&self) -> Vec<F>;
}

impl<F: SemiflowFloat> NodalField<F> for GridFn1D<F> {
    fn node_values(&self) -> &[F] {
        &self.values
    }
    fn node_values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }
    fn node_coordinates(&self) -> Vec<F> {
        (0..self.values.len()).map(|i| self.grid.x_at(i)).collect()
    }
}

impl<F: SemiflowFloat> NodalField<F> for GridFn2D<F> {
    fn node_values(&self) -> &[F] {
        &self.values
    }
    fn node_values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }
    /// Row-major, `x` fastest: node `iy·nx + ix`.
    fn node_coordinates(&self) -> Vec<F> {
        let (gx, gy) = (self.grid.x, self.grid.y);
        let (xs, ys): (Vec<F>, Vec<F>) = (0..gy.n)
            .flat_map(|iy| (0..gx.n).map(move |ix| (gx.x_at(ix), gy.x_at(iy))))
            .unzip();
        [xs, ys].concat()
    }
}

impl<F: SemiflowFloat> NodalField<F> for GridFn3D<F> {
    fn node_values(&self) -> &[F] {
        &self.values
    }
    fn node_values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }
    /// `x` fastest, then `y`, then `z`: node `(iz·ny + iy)·nx + ix`.
    fn node_coordinates(&self) -> Vec<F> {
        let (gx, gy, gz) = (self.grid.x, self.grid.y, self.grid.z);
        let n = gx.n * gy.n * gz.n;
        let mut out = Vec::with_capacity(3 * n);
        let node = |i: usize| (i % gx.n, (i / gx.n) % gy.n, i / (gx.n * gy.n));
        out.extend((0..n).map(|i| gx.x_at(node(i).0)));
        out.extend((0..n).map(|i| gy.x_at(node(i).1)));
        out.extend((0..n).map(|i| gz.x_at(node(i).2)));
        out
    }
}

impl<F: SemiflowFloat> NodalField<F> for GraphSignal<F> {
    fn node_values(&self) -> &[F] {
        &self.values
    }
    fn node_values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }
    fn node_coordinates(&self) -> Vec<F> {
        Vec::new()
    }
}

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
        })
    }

    /// RK4 substeps per reaction half-step (stiff reactions; default 1, min 1).
    /// Ignored when the reaction has an exact flow.
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
        let mut ws = Workspace::new(u, n, scratch);
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
}

impl<F: SemiflowFloat, S: NodalField<F> + Clone> Workspace<F, S> {
    fn new(u: &Species<S>, n: usize, scratch: &mut ScratchPool<F>) -> Self {
        let kn = u.fields.len() * n;
        Self {
            n,
            x: u.fields[0].node_coordinates(),
            flat: scratch.take_vec(kn),
            acc: scratch.take_vec(kn),
            stage: scratch.take_vec(kn),
            k: scratch.take_vec(kn),
            spare: u.fields[0].clone(),
        }
    }

    fn release(self, scratch: &mut ScratchPool<F>) {
        for b in [self.flat, self.acc, self.stage, self.k] {
            scratch.return_vec(b);
        }
    }

    /// `u_k ← C_k(τ) u_k` for every species.
    fn diffuse(
        &mut self,
        rd: &ReactionDiffusion<'_, F, S>,
        tau: F,
        u: &mut Species<S>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        for (engine, field) in rd.engines.iter().zip(u.fields.iter_mut()) {
            engine.apply_into(tau, field, &mut self.spare, scratch)?;
            core::mem::swap(field, &mut self.spare);
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
        for (k, field) in u.fields.iter().enumerate() {
            self.flat[k * n..(k + 1) * n].copy_from_slice(field.node_values());
        }
        if let Some(r) = rd.reaction.exact_flow(t, h, n, &self.x, &mut self.flat) {
            r?;
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
                what: "ReactionDiffusion: non-finite reaction result (stiff? raise with_reaction_substeps)",
                value: bad.to_f64().unwrap_or(f64::NAN),
            });
        }
        for (k, field) in u.fields.iter_mut().enumerate() {
            field
                .node_values_mut()
                .copy_from_slice(&self.flat[k * n..(k + 1) * n]);
        }
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

#[cfg(test)]
#[path = "reaction_diffusion_tests.rs"]
mod tests;
