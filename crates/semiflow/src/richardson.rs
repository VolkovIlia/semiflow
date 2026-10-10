//! Richardson extrapolation in the step count (ADR-0207).
//!
//! A fixed-step method whose global error expands in powers of the step,
//! `U_n = u + c_p n^{−p} + c_{p+1} n^{−(p+1)} + …`, is run with `n_j = (j+1)·n`
//! steps, `j < L`, and combined as `Σ w_j U_{n_j}` with
//!
//! ```text
//! Σ_j w_j = 1,    Σ_j w_j (j+1)^{−q} = 0   for q = p, …, p + L − 2,
//! ```
//!
//! which removes the `L − 1` leading error terms: order `p` becomes `p + L − 1`
//! at a cost of `L(L+1)/2` times the base run. Chernoff products have such an
//! expansion for data in the domain of a sufficiently high power of the
//! generator (Galkin & Remizov 2025); for rough data the gain is smaller, and
//! the combination amplifies rounding by `Σ|w_j|` ([`richardson_weights`]
//! exposes it). Above the spatial error floor of the discretisation the gain is
//! real; at the floor extrapolation cannot help (gate `G_RICHARDSON_*`).

use alloc::vec::Vec;

use crate::{
    chernoff::{ChernoffFunction, Evolver},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    reaction_diffusion::{NodalField, ReactionDiffusion, Species},
    scratch::ScratchPool,
    state::State,
};

/// Largest supported number of levels (`Σ|w|` grows quickly beyond).
pub const MAX_RICHARDSON_LEVELS: usize = 6;

/// A fixed-step integrator with a known leading error exponent.
pub trait StepIntegrator<F: SemiflowFloat> {
    /// The evolved state.
    type S: State<F> + Clone;

    /// Leading global error exponent `p` (`error = O(n^{−p})`).
    fn error_order(&self) -> u32;

    /// `dst ←` `n` steps over `[t0, t0 + t]` from `src`.
    ///
    /// # Errors
    /// Implementation-defined.
    #[allow(clippy::too_many_arguments)] // (t0, t, n) + (src, dst, scratch)
    fn integrate_into(
        &self,
        t0: F,
        t: F,
        n: usize,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;
}

/// Gaussian elimination with partial pivoting on an `l × (l+1)` augmented system.
#[allow(clippy::many_single_char_names)] // l, a, w, r, c, f, x, s: textbook elimination
fn solve_augmented(l: usize, a: &mut [f64]) -> Vec<f64> {
    let w = l + 1;
    for col in 0..l {
        let piv = (col..l)
            .max_by(|&i, &j| libm::fabs(a[i * w + col]).total_cmp(&libm::fabs(a[j * w + col])))
            .unwrap_or(col);
        for c in 0..w {
            a.swap(col * w + c, piv * w + c);
        }
        for r in col + 1..l {
            let f = a[r * w + col] / a[col * w + col];
            for c in col..w {
                a[r * w + c] -= f * a[col * w + c];
            }
        }
    }
    let mut x = alloc::vec![0.0; l];
    for r in (0..l).rev() {
        let s: f64 = (r + 1..l).map(|c| a[r * w + c] * x[c]).sum();
        x[r] = (a[r * w + l] - s) / a[r * w + r];
    }
    x
}

/// Richardson weights `w_j` for step counts `(j+1)·n`, `j < levels`, leading
/// exponent `order` (see the module notes). `levels = 1` gives `[1]`.
///
/// # Errors
/// `DomainViolation` if `order == 0`, `order == u32::MAX` (tolerance-driven
/// methods have no error expansion), or `levels ∉ 1..=MAX_RICHARDSON_LEVELS`.
pub fn richardson_weights(order: u32, levels: usize) -> Result<Vec<f64>, SemiflowError> {
    if order == 0 || order == u32::MAX || !(1..=MAX_RICHARDSON_LEVELS).contains(&levels) {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "richardson_weights: need 1 <= order < u32::MAX and 1 <= levels <= 6",
            value: if order == 0 || order == u32::MAX {
                f64::from(order)
            } else {
                levels as f64
            },
        });
    }
    // Row 0: Σ w = 1; row r: Σ w_j (j+1)^{-(order + r - 1)} = 0.
    let l = levels;
    let mut a: Vec<f64> = Vec::with_capacity(l * (l + 1));
    for r in 0..l {
        #[allow(clippy::cast_possible_truncation)] // r < MAX_RICHARDSON_LEVELS
        let q = if r == 0 { 0 } else { order + r as u32 - 1 };
        for j in 0..l {
            #[allow(clippy::cast_precision_loss)] // j < MAX_RICHARDSON_LEVELS
            let h = 1.0 / (j + 1) as f64;
            a.push(libm::pow(h, f64::from(q)));
        }
        a.push(if r == 0 { 1.0 } else { 0.0 });
    }
    Ok(solve_augmented(l, &mut a))
}

/// `dst ← Σ_j w_j U_{(j+1)n}` over `[t0, t0 + t]` (see the module notes).
///
/// # Errors
/// Invalid `levels` / order (see [`richardson_weights`]), `(j+1)·n` overflow,
/// or any error of the integrator.
#[allow(clippy::too_many_arguments)] // (t0, t, n, levels) + (src, dst, scratch)
pub fn extrapolate_into<F: SemiflowFloat, I: StepIntegrator<F>>(
    integ: &I,
    t0: F,
    t: F,
    n: usize,
    levels: usize,
    src: &I::S,
    dst: &mut I::S,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let weights = richardson_weights(integ.error_order(), levels)?;
    let mut run = src.clone();
    let mut acc = src.clone();
    acc.zero_into();
    for (j, &w) in weights.iter().enumerate() {
        let steps =
            n.checked_mul(j + 1)
                .filter(|&s| s > 0)
                .ok_or(SemiflowError::DomainViolation {
                    what: "extrapolate_into: need n >= 1 and (levels·n) without overflow",
                    #[allow(clippy::cast_precision_loss)]
                    value: n as f64,
                })?;
        integ.integrate_into(t0, t, steps, src, &mut run, scratch)?;
        acc.axpy_into(from_f64::<F>(w), &run);
    }
    dst.copy_from(&acc);
    Ok(())
}

impl<C, F> StepIntegrator<F> for Evolver<C, F>
where
    C: ChernoffFunction<F>,
    C::S: Clone,
    F: SemiflowFloat,
{
    type S = C::S;

    fn error_order(&self) -> u32 {
        self.func.order()
    }

    /// `(C(t/n))^n src` (as [`Evolver::evolve_into`] with `n` steps); `t0` is
    /// ignored (autonomous).
    #[allow(clippy::too_many_arguments)]
    fn integrate_into(
        &self,
        _t0: F,
        t: F,
        n: usize,
        src: &C::S,
        dst: &mut C::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if n == 0 || !t.is_finite() || t < F::zero() {
            return Err(SemiflowError::DomainViolation {
                what: "Evolver::integrate_into: need n >= 1 and finite t >= 0",
                value: t.to_f64().unwrap_or(f64::NAN),
            });
        }
        #[allow(clippy::cast_precision_loss)]
        let tau = t / from_f64::<F>(n as f64);
        let mut other = src.clone();
        dst.copy_from(src);
        for _ in 0..n {
            self.func.apply_into(tau, dst, &mut other, scratch)?;
            core::mem::swap(dst, &mut other);
        }
        Ok(())
    }
}

impl<C, F> Evolver<C, F>
where
    C: ChernoffFunction<F>,
    C::S: Clone,
    F: SemiflowFloat,
{
    /// Richardson-extrapolated `(C(t/n_j))^{n_j} src`, `n_j = (j+1)·self.n`,
    /// `j < levels` (ADR-0207): order `p` becomes `p + levels − 1`.
    ///
    /// # Errors
    /// See [`extrapolate_into`].
    pub fn evolve_extrapolated_into(
        &self,
        t: F,
        levels: usize,
        src: &C::S,
        dst: &mut C::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        extrapolate_into(self, F::zero(), t, self.n, levels, src, dst, scratch)
    }
}

impl<F, S> StepIntegrator<F> for ReactionDiffusion<'_, F, S>
where
    F: SemiflowFloat,
    S: NodalField<F> + Clone,
{
    type S = Species<S>;

    fn error_order(&self) -> u32 {
        self.order()
    }

    #[allow(clippy::too_many_arguments)]
    fn integrate_into(
        &self,
        t0: F,
        t: F,
        n: usize,
        src: &Species<S>,
        dst: &mut Species<S>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        self.evolve_into(t0, t, n, src, dst, scratch)
    }
}
