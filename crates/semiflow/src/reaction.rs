//! Pointwise reaction terms `∂ₜu = f(t, x, u)` for [`ReactionDiffusion`] (ADR-0208).
//!
//! A [`Kinetics`] maps the `K` species values at every node to their rates, in
//! ONE batched call over all nodes, so a host-language callback (Python) is
//! invoked once per Runge–Kutta stage, never once per node. Buffers are flat and
//! species-major: `u[k·N + i]` is species `k` at node `i`; coordinates are
//! axis-major, `x[d·N + i]`.
//!
//! The catalogue covers the standard models; [`FnReaction`] wraps any pointwise
//! closure. A reaction with a closed-form flow overrides [`Kinetics::exact_flow`]
//! (logistic, Allen–Cahn, linear), and the splitter then uses it instead of RK4.
//!
//! [`ReactionDiffusion`]: crate::ReactionDiffusion

use alloc::{vec, vec::Vec};

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
};

/// Rates `du = f(t, x, u)` of `K` species, evaluated for all nodes at once.
pub trait Kinetics<F: SemiflowFloat>: Send + Sync {
    /// Number of species `K`.
    fn species(&self) -> usize;

    /// `du[k·N + i] = f_k(t, x_i, u_i)` for every node `i < n`.
    ///
    /// `x` holds `dim·n` coordinates (axis-major; empty on a graph), `u` and
    /// `du` hold `K·n` values (species-major).
    ///
    /// # Errors
    /// Implementation-defined (a host callback that fails, a non-finite rate).
    fn eval_batch(
        &self,
        t: F,
        n: usize,
        x: &[F],
        u: &[F],
        du: &mut [F],
    ) -> Result<(), SemiflowError>;

    /// Exact flow `u ← Φ_h(u)` of `u′ = f(t, x, u)` over `[t, t + h]`, if known.
    ///
    /// `None` (the default) makes the splitter integrate with classical RK4.
    fn exact_flow(
        &self,
        _t: F,
        _h: F,
        _n: usize,
        _x: &[F],
        _u: &mut [F],
    ) -> Option<Result<(), SemiflowError>> {
        None
    }
}

/// Logistic (Fisher–KPP) growth `u′ = r·u·(1 − u/cap)`, exact flow.
#[derive(Clone, Copy, Debug)]
pub struct FisherKpp<F: SemiflowFloat = f64> {
    /// Growth rate `r`.
    pub rate: F,
    /// Carrying capacity (`> 0`).
    pub capacity: F,
}

impl<F: SemiflowFloat> Kinetics<F> for FisherKpp<F> {
    fn species(&self) -> usize {
        1
    }
    fn eval_batch(
        &self,
        _t: F,
        n: usize,
        _x: &[F],
        u: &[F],
        du: &mut [F],
    ) -> Result<(), SemiflowError> {
        for (d, &v) in du[..n].iter_mut().zip(&u[..n]) {
            *d = self.rate * v * (F::one() - v / self.capacity);
        }
        Ok(())
    }
    /// `u(h) = cap·u / (u + (cap − u)·e^{−rh})`.
    fn exact_flow(
        &self,
        _t: F,
        h: F,
        n: usize,
        _x: &[F],
        u: &mut [F],
    ) -> Option<Result<(), SemiflowError>> {
        let decay = (F::zero() - self.rate * h).libm_exp();
        for v in &mut u[..n] {
            *v = self.capacity * *v / (*v + (self.capacity - *v) * decay);
        }
        Some(Ok(()))
    }
}

/// Allen–Cahn bistable reaction `u′ = κ·(u − u³)`, exact flow.
#[derive(Clone, Copy, Debug)]
pub struct AllenCahnReaction<F: SemiflowFloat = f64> {
    /// Rate `κ` (`1/ε²` in the usual scaling).
    pub kappa: F,
}

impl<F: SemiflowFloat> Kinetics<F> for AllenCahnReaction<F> {
    fn species(&self) -> usize {
        1
    }
    fn eval_batch(
        &self,
        _t: F,
        n: usize,
        _x: &[F],
        u: &[F],
        du: &mut [F],
    ) -> Result<(), SemiflowError> {
        for (d, &v) in du[..n].iter_mut().zip(&u[..n]) {
            *d = self.kappa * (v - v * v * v);
        }
        Ok(())
    }
    /// `u(h) = u / √(u² + (1 − u²)·e^{−2κh})`.
    fn exact_flow(
        &self,
        _t: F,
        h: F,
        n: usize,
        _x: &[F],
        u: &mut [F],
    ) -> Option<Result<(), SemiflowError>> {
        let decay = (F::zero() - from_f64::<F>(2.0) * self.kappa * h).libm_exp();
        for v in &mut u[..n] {
            let sq = *v * *v;
            *v /= (sq + (F::one() - sq) * decay).sqrt();
        }
        Some(Ok(()))
    }
}

/// Nagumo (bistable, threshold `a`) reaction `u′ = u(1 − u)(u − a)`.
#[derive(Clone, Copy, Debug)]
pub struct Nagumo<F: SemiflowFloat = f64> {
    /// Threshold `a ∈ (0, 1)`.
    pub a: F,
}

impl<F: SemiflowFloat> Kinetics<F> for Nagumo<F> {
    fn species(&self) -> usize {
        1
    }
    fn eval_batch(
        &self,
        _t: F,
        n: usize,
        _x: &[F],
        u: &[F],
        du: &mut [F],
    ) -> Result<(), SemiflowError> {
        for (d, &v) in du[..n].iter_mut().zip(&u[..n]) {
            *d = v * (F::one() - v) * (v - self.a);
        }
        Ok(())
    }
}

/// Gray–Scott `u′ = −uv² + f(1 − u)`, `v′ = uv² − (f + k)v`.
#[derive(Clone, Copy, Debug)]
pub struct GrayScott<F: SemiflowFloat = f64> {
    /// Feed rate `f`.
    pub feed: F,
    /// Kill rate `k`.
    pub kill: F,
}

impl<F: SemiflowFloat> Kinetics<F> for GrayScott<F> {
    fn species(&self) -> usize {
        2
    }
    fn eval_batch(
        &self,
        _t: F,
        n: usize,
        _x: &[F],
        u: &[F],
        du: &mut [F],
    ) -> Result<(), SemiflowError> {
        let (uu, vv) = u.split_at(n);
        let (du_u, du_v) = du.split_at_mut(n);
        for i in 0..n {
            let uv2 = uu[i] * vv[i] * vv[i];
            du_u[i] = self.feed * (F::one() - uu[i]) - uv2;
            du_v[i] = uv2 - (self.feed + self.kill) * vv[i];
        }
        Ok(())
    }
}

/// FitzHugh–Nagumo `v′ = v − v³/3 − w + I`, `w′ = ε(v + a − b·w)`.
#[derive(Clone, Copy, Debug)]
pub struct FitzHughNagumo<F: SemiflowFloat = f64> {
    /// Recovery offset `a`.
    pub a: F,
    /// Recovery gain `b`.
    pub b: F,
    /// Time-scale ratio `ε`.
    pub eps: F,
    /// Applied current `I`.
    pub current: F,
}

impl<F: SemiflowFloat> Kinetics<F> for FitzHughNagumo<F> {
    fn species(&self) -> usize {
        2
    }
    fn eval_batch(
        &self,
        _t: F,
        n: usize,
        _x: &[F],
        u: &[F],
        du: &mut [F],
    ) -> Result<(), SemiflowError> {
        let (v, w) = u.split_at(n);
        let (dv, dw) = du.split_at_mut(n);
        let third = from_f64::<F>(1.0 / 3.0);
        for i in 0..n {
            dv[i] = v[i] - third * v[i] * v[i] * v[i] - w[i] + self.current;
            dw[i] = self.eps * (v[i] + self.a - self.b * w[i]);
        }
        Ok(())
    }
}

/// Brusselator `u′ = a − (b + 1)u + u²v`, `v′ = bu − u²v`.
#[derive(Clone, Copy, Debug)]
pub struct Brusselator<F: SemiflowFloat = f64> {
    /// Parameter `a`.
    pub a: F,
    /// Parameter `b`.
    pub b: F,
}

impl<F: SemiflowFloat> Kinetics<F> for Brusselator<F> {
    fn species(&self) -> usize {
        2
    }
    fn eval_batch(
        &self,
        _t: F,
        n: usize,
        _x: &[F],
        u: &[F],
        du: &mut [F],
    ) -> Result<(), SemiflowError> {
        let (uu, vv) = u.split_at(n);
        let (du_u, du_v) = du.split_at_mut(n);
        for i in 0..n {
            let u2v = uu[i] * uu[i] * vv[i];
            du_u[i] = self.a - (self.b + F::one()) * uu[i] + u2v;
            du_v[i] = self.b * uu[i] - u2v;
        }
        Ok(())
    }
}

/// Constant linear coupling `u′ = C·u` (`C` row-major `K×K`), exact flow.
#[derive(Clone, Debug)]
pub struct LinearReaction<F: SemiflowFloat = f64> {
    k: usize,
    matrix: Vec<F>,
}

impl<F: SemiflowFloat> LinearReaction<F> {
    /// Build from a row-major `K×K` matrix.
    ///
    /// # Errors
    /// `DomainViolation` if `matrix.len()` is not a positive square or an entry
    /// is not finite.
    pub fn new(matrix: Vec<F>) -> Result<Self, SemiflowError> {
        let k = (1..=64)
            .find(|k| k * k == matrix.len())
            .ok_or(SemiflowError::DomainViolation {
                what: "LinearReaction: matrix must be K×K with 1 <= K <= 64",
                #[allow(clippy::cast_precision_loss)]
                value: matrix.len() as f64,
            })?;
        if let Some(bad) = matrix.iter().find(|v| !v.is_finite()) {
            return Err(SemiflowError::DomainViolation {
                what: "LinearReaction: non-finite matrix entry",
                value: bad.to_f64().unwrap_or(f64::NAN),
            });
        }
        Ok(Self { k, matrix })
    }
}

impl<F: SemiflowFloat> Kinetics<F> for LinearReaction<F> {
    fn species(&self) -> usize {
        self.k
    }
    fn eval_batch(
        &self,
        _t: F,
        n: usize,
        _x: &[F],
        u: &[F],
        du: &mut [F],
    ) -> Result<(), SemiflowError> {
        let k = self.k;
        du[..k * n].fill(F::zero());
        for (r, row) in self.matrix.chunks_exact(k).enumerate() {
            for (c, &coef) in row.iter().enumerate() {
                for i in 0..n {
                    du[r * n + i] += coef * u[c * n + i];
                }
            }
        }
        Ok(())
    }
    /// `u ← e^{hC}u` per node, `e^{hC}` by scaling and squaring of a degree-18
    /// Taylor polynomial (`‖hC/2ˢ‖₁ ≤ ½`, truncation `< 1e−22`).
    #[allow(clippy::many_single_char_names)] // h, n, u, k, e: the formula's names
    fn exact_flow(
        &self,
        _t: F,
        h: F,
        n: usize,
        _x: &[F],
        u: &mut [F],
    ) -> Option<Result<(), SemiflowError>> {
        let k = self.k;
        let e = dense_exp(k, &self.matrix, h);
        let mut node = vec![F::zero(); k];
        for i in 0..n {
            for (r, slot) in node.iter_mut().enumerate() {
                *slot = (0..k).fold(F::zero(), |acc, c| acc + e[r * k + c] * u[c * n + i]);
            }
            for (r, &val) in node.iter().enumerate() {
                u[r * n + i] = val;
            }
        }
        Some(Ok(()))
    }
}

/// `e^{hC}` for a small dense row-major `K×K` matrix.
fn dense_exp<F: SemiflowFloat>(k: usize, c: &[F], h: F) -> Vec<F> {
    let norm = (0..k)
        .map(|col| (0..k).fold(F::zero(), |acc, row| acc + (c[row * k + col] * h).abs()))
        .fold(F::zero(), |a, b| if b > a { b } else { a });
    let mut squarings = 0_u32;
    let mut scale = F::one();
    while norm * scale > from_f64::<F>(0.5) && squarings < 1100 {
        scale *= from_f64::<F>(0.5);
        squarings += 1;
    }
    let a: Vec<F> = c.iter().map(|&v| v * h * scale).collect();
    let mut result = identity(k);
    let mut term = identity(k);
    for d in 1..=18_u32 {
        term = matmul(k, &term, &a);
        let inv = F::one() / from_f64::<F>(f64::from(d));
        for v in &mut term {
            *v *= inv;
        }
        for (r, &t) in result.iter_mut().zip(&term) {
            *r += t;
        }
    }
    for _ in 0..squarings {
        result = matmul(k, &result, &result);
    }
    result
}

fn identity<F: SemiflowFloat>(k: usize) -> Vec<F> {
    let mut m = vec![F::zero(); k * k];
    for i in 0..k {
        m[i * k + i] = F::one();
    }
    m
}

fn matmul<F: SemiflowFloat>(k: usize, a: &[F], b: &[F]) -> Vec<F> {
    let mut out = vec![F::zero(); k * k];
    for r in 0..k {
        for m in 0..k {
            let arm = a[r * k + m];
            for c in 0..k {
                out[r * k + c] += arm * b[m * k + c];
            }
        }
    }
    out
}

/// Any pointwise reaction `f(t, x, u, du)` from a closure (`x`: the node's
/// coordinates, `u`, `du`: its `K` species values).
pub struct FnReaction<G> {
    k: usize,
    f: G,
}

impl<G> FnReaction<G> {
    /// Wrap `f` for `k` species.
    pub fn new(k: usize, f: G) -> Self {
        Self { k, f }
    }
}

impl<F, G> Kinetics<F> for FnReaction<G>
where
    F: SemiflowFloat,
    G: Fn(F, &[F], &[F], &mut [F]) + Send + Sync,
{
    fn species(&self) -> usize {
        self.k
    }
    #[allow(clippy::many_single_char_names)] // t, n, x, u, k: the trait's names
    fn eval_batch(
        &self,
        t: F,
        n: usize,
        x: &[F],
        u: &[F],
        du: &mut [F],
    ) -> Result<(), SemiflowError> {
        let k = self.k;
        let dim = x.len().checked_div(n).unwrap_or(0);
        let (mut xi, mut ui, mut di) =
            (vec![F::zero(); dim], vec![F::zero(); k], vec![F::zero(); k]);
        for i in 0..n {
            for (d, slot) in xi.iter_mut().enumerate() {
                *slot = x[d * n + i];
            }
            for (s, slot) in ui.iter_mut().enumerate() {
                *slot = u[s * n + i];
            }
            (self.f)(t, &xi, &ui, &mut di);
            for (s, &v) in di.iter().enumerate() {
                du[s * n + i] = v;
            }
        }
        Ok(())
    }
}
