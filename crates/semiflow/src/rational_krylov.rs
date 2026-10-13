//! Rational (shift-and-invert) Krylov action `e^{−t·M⁻¹A}·v` for SPD sparse
//! operators (ADR-0206, math §66).
//!
//! Polynomial Krylov methods (Chebyshev, Lanczos) need `O(√(λ_max t))` mat-vecs:
//! 47 188 at `λ_max t = 1e8`. Shift-and-invert Lanczos (van den Eshof &
//! Hochbruck, SIAM J. Sci. Comput. 27 (2006)) runs Lanczos on
//! `Z = (M/γ + A)⁻¹M` — self-adjoint in `⟨x, y⟩_M = xᵀMy`, spectrum
//! `θ = 1/(1/γ + λ) ∈ (0, γ]` — and needs a number of solves that is independent
//! of `λ_max`: the stiff modes all land near `θ = 0`, where
//! `f(θ) = e^{t/γ − t/θ}` is flat. One factor of `M/γ + A` serves every step and
//! every vector (`O(N)` per solve on a tridiagonal pattern, cancellation-free for
//! M-matrices; PCG otherwise).
//!
//! Stopping rule (a priori): `m` solves with `m` the smallest dimension whose
//! Chebyshev bound `2·E_{m−1}(f; [0, γ]) ≤ tol` (`m = 27` at `tol = 1e−10`,
//! `γ = t/10`), or fewer on an invariant subspace. An a-posteriori difference test
//! would stop early when every Ritz value is stiff and silently drop a small slow
//! component that has not yet entered the Krylov space.

use alloc::vec::Vec;

use factor::Inner;
use lanczos::Lanczos;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph_signal::GraphSignal,
    pow2::pow2_scale,
    scratch::ScratchPool,
    spd_resolvent::SpdSolver,
    state::State,
    symmetric_operator::SymmetricOperator,
};

/// Options for [`RationalKrylov`]. Build with [`RationalOptions::default`] and the
/// `with_*` setters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RationalOptions {
    /// Target accuracy, relative to `‖v‖_M`, `∈ (0, 1)`. Default `1e−10`.
    pub tol: f64,
    /// Shift `γ > 0`; `None` → `t/10` (van den Eshof & Hochbruck's choice range).
    pub gamma: Option<f64>,
    /// Maximum Krylov dimension (number of solves). Default 64, clamped to `[3, n]`.
    /// Below the a-priori dimension for `tol` (27 at the defaults) the action
    /// reports `ConvergenceFailed` unless `n` is reached first.
    pub m_max: usize,
    /// Solver of `M/γ + A`. `Auto` / `Tridiagonal`: the cancellation-free
    /// tridiagonal factor when `A` is a tridiagonal M-matrix pattern, else
    /// [`crate::SpdResolvent`] with this choice (`LDLᵀ` or PCG). Default `Auto`.
    pub solver: SpdSolver,
}

impl Default for RationalOptions {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            gamma: None,
            m_max: 64,
            solver: SpdSolver::Auto,
        }
    }
}

impl RationalOptions {
    /// Set the target accuracy.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
    /// Set the shift `γ`.
    #[must_use]
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }
    /// Set the maximum Krylov dimension.
    #[must_use]
    pub fn with_m_max(mut self, m_max: usize) -> Self {
        self.m_max = m_max;
        self
    }
    /// Set the inner solver.
    #[must_use]
    pub fn with_solver(mut self, solver: SpdSolver) -> Self {
        self.solver = solver;
        self
    }
}

/// Diagnostics of one [`RationalKrylov::apply_into`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RationalReport {
    /// Resolvent solves (= Krylov steps).
    pub solves: usize,
    /// Final Krylov dimension.
    pub krylov_dim: usize,
    /// `‖y_m − y_{m−1}‖_M/‖v‖_M` at the last step: an a-posteriori diagnostic (the
    /// stop is the a-priori bound, module docs). After an early breakdown (an
    /// invariant subspace, or the whole space: the projection is exact) it still
    /// reports that last difference, which then overstates the error.
    pub est_error: f64,
    /// Total CG iterations of the inner solves (0 on the tridiagonal path).
    pub inner_iterations: usize,
}

/// `v ↦ e^{−t·M⁻¹A}·v` by shift-and-invert Lanczos, factor built once.
#[derive(Clone)]
pub struct RationalKrylov<F: SemiflowFloat = f64> {
    inner: Inner<F>,
    n: usize,
    mass: Option<Vec<F>>,
    /// `mass·2^{−e}` with `max mass·2^{−e} ∈ [1, 2)`: the weights of `⟨·,·⟩_M` (any
    /// positive multiple of `M` gives the same Lanczos), free of under/overflow.
    weights: Option<Vec<F>>,
    t: F,
    gamma: F,
    m_max: usize,
    /// A-priori stopping dimension; `None`: not reachable within `m_max`.
    m_min: Option<usize>,
}

fn domain(what: &'static str, value: f64) -> SemiflowError {
    SemiflowError::DomainViolation { what, value }
}

/// Scalar and mass checks of [`RationalKrylov::new`].
fn validate<F: SemiflowFloat>(
    n: usize,
    t64: f64,
    opts: &RationalOptions,
    mass: Option<&[F]>,
) -> Result<(), SemiflowError> {
    if n == 0 {
        return Err(domain(
            "rational_krylov: operator dimension n must be >= 1",
            0.0,
        ));
    }
    // The projected coefficients and the Chebyshev bound are f64 scalars: a
    // forward-mode `Dual` (of f32 or f64) would silently lose its tangent.
    if !crate::float::is_plain_ieee::<F>() {
        return Err(SemiflowError::Unsupported {
            feature: "rational_krylov: forward-mode Dual (only f32 / f64)",
        });
    }
    if !t64.is_finite() || t64 < 0.0 {
        return Err(domain("rational_krylov: t must be finite and >= 0", t64));
    }
    if !(opts.tol > 0.0 && opts.tol < 1.0) {
        return Err(domain("rational_krylov: tol must be in (0, 1)", opts.tol));
    }
    if opts.m_max == 0 {
        return Err(domain("rational_krylov: m_max must be >= 1", 0.0));
    }
    let Some(m) = mass else { return Ok(()) };
    if m.len() != n {
        #[allow(clippy::cast_precision_loss)]
        return Err(domain("rational_krylov: mass.len() != n", m.len() as f64));
    }
    if let Some(&bad) = m.iter().find(|&&x| !x.is_finite() || x <= F::zero()) {
        return Err(domain(
            "rational_krylov: mass must be finite and > 0",
            bad.to_f64().unwrap_or(f64::NAN),
        ));
    }
    Ok(())
}

/// `(max mᵢ / min mᵢ, ‖M⁻¹A‖ ≤ λ_max_bound(A) / min mᵢ)` in `f64` (`(1, ‖A‖)`
/// without a mass).
fn mass_stats<F: SemiflowFloat>(op: &SymmetricOperator<F>, mass: Option<&[F]>) -> (f64, f64) {
    let (lo, hi) = mass.map_or((1.0, 1.0), |m| {
        m.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), x| {
            let x = x.to_f64().unwrap_or(f64::NAN);
            (lo.min(x), hi.max(x))
        })
    });
    (
        hi / lo,
        op.lambda_max_bound().to_f64().unwrap_or(f64::NAN) / lo,
    )
}

/// The solver and the a-priori dimension, or the identity when `t·‖M⁻¹A‖ < ε`.
///
/// PCG stops on `‖b − Sx‖₂ ≤ tol_in·‖b‖₂` with `b = M·q`, `‖q‖_M = 1`, which bounds
/// the `M`-norm error of `Zq` only by `γ·tol_in·√(max m / min m)`: the inner
/// tolerance is divided by that square root (floor `1e−15`).
///
/// # Errors
/// `1/γ` not finite (`DomainViolation`); as [`Inner::build`].
fn build_inner<F: SemiflowFloat>(
    op: &SymmetricOperator<F>,
    mass: Option<&[F]>,
    (t64, gamma): (f64, f64),
    opts: &RationalOptions,
    m_max: usize,
) -> Result<(Inner<F>, Option<usize>), SemiflowError> {
    let (contrast, bound) = mass_stats(op, mass);
    if t64 * bound < f64::EPSILON {
        return Ok((Inner::Identity, None));
    }
    if !(1.0 / gamma).is_finite() {
        return Err(domain("rational_krylov: 1/gamma must be finite", gamma));
    }
    let inner_tol = from_f64::<F>((0.1 * opts.tol / libm::sqrt(contrast)).max(1e-15));
    let inner = Inner::build(op, from_f64(1.0 / gamma), mass, opts.solver, inner_tol)?;
    Ok((inner, factor::a_priori_dim(t64 / gamma, opts.tol, m_max)))
}

impl<F: SemiflowFloat> RationalKrylov<F> {
    /// Factor `M/γ + A` for `A = op`, `M = diag(mass)` (default `I`).
    ///
    /// When `t·‖M⁻¹A‖ < ε` (including `t = 0` and a positive `t` too small for
    /// `1/γ` to be finite) the action is the identity to rounding and no factor
    /// is built.
    ///
    /// # Errors
    /// `Unsupported`: `F` is not `f32` / `f64` (forward-mode `Dual`: the
    /// coefficients `f(T)e₁` are `f64` scalars, so tangents would be dropped).
    /// `DomainViolation`: `n = 0`; `t < 0` or non-finite; `γ ≤ 0`, non-finite, or
    /// (unless the action is the identity) with `1/γ` non-finite; `tol ∉ (0, 1)`; `m_max == 0`; `mass.len() != n`, a
    /// mass `≤ 0` or non-finite; and every error of [`crate::SpdResolvent::new`]
    /// on the non-tridiagonal / forced-PCG path (a non-SPD pattern).
    pub fn new(
        op: &SymmetricOperator<F>,
        mass: Option<&[F]>,
        t: F,
        opts: RationalOptions,
    ) -> Result<Self, SemiflowError> {
        let n = op.n();
        let t64 = t.to_f64().unwrap_or(f64::NAN);
        validate(n, t64, &opts, mass)?;
        let gamma = opts
            .gamma
            .unwrap_or(if t64 > 0.0 { t64 / 10.0 } else { 1.0 });
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(domain(
                "rational_krylov: gamma must be finite and > 0",
                gamma,
            ));
        }
        let m_max = opts.m_max.max(3).min(n);
        let (inner, m_min) = build_inner(op, mass, (t64, gamma), &opts, m_max)?;
        let weights = mass.map(|m| {
            let (down, _) = pow2_scale(m).unwrap_or((F::one(), F::one()));
            m.iter().map(|&x| x * down).collect()
        });
        Ok(Self {
            inner,
            n,
            mass: mass.map(<[F]>::to_vec),
            weights,
            t,
            gamma: from_f64(gamma),
            m_max,
            m_min,
        })
    }

    /// Operator dimension.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Time `t` the factor was built for.
    #[must_use]
    pub fn t(&self) -> F {
        self.t
    }

    /// Shift `γ` in use.
    #[must_use]
    pub fn gamma(&self) -> F {
        self.gamma
    }

    /// `out ← e^{−t·M⁻¹A}·v`.
    ///
    /// `v` is prescaled by an exact power of two (`‖v‖∞ ∈ [1, 2)`), so the result is
    /// homogeneous bit for bit under `v ↦ 2ʲv` and free of under/overflow in `‖v‖_M`.
    ///
    /// # Errors
    /// `DomainViolation`: `v.len()` / `out.len() != n`, non-finite `v`, a non-finite
    /// inner solution. `ConvergenceFailed { last_residual, max_iter }`: the
    /// a-priori dimension exceeds `m_max` (and `n`) — `last_residual` is the last
    /// a-posteriori estimate — or an inner PCG solve failed.
    pub fn apply_into(
        &self,
        v: &[F],
        out: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<RationalReport, SemiflowError> {
        let n = self.n();
        if v.len() != n || out.len() != n {
            #[allow(clippy::cast_precision_loss)]
            return Err(domain(
                "rational_krylov: v.len() or out.len() != n",
                v.len().min(out.len()) as f64,
            ));
        }
        if let Some(&bad) = v.iter().find(|x| !x.is_finite()) {
            return Err(domain(
                "rational_krylov: non-finite v",
                bad.to_f64().unwrap_or(f64::NAN),
            ));
        }
        let identity = matches!(self.inner, Inner::Identity);
        let Some((down, up)) = pow2_scale(v).filter(|_| !identity) else {
            // v ≡ 0, or t·‖M⁻¹A‖ < ε.
            out.copy_from_slice(v);
            return Ok(RationalReport {
                solves: 0,
                krylov_dim: 0,
                est_error: 0.0,
                inner_iterations: 0,
            });
        };
        let mut lanczos = Lanczos::new(n);
        let norm_v = lanczos.start(self, v, down, scratch);
        let result = lanczos.run(self, scratch);
        if let Ok((ref coeffs, _)) = result {
            lanczos.combine(coeffs, norm_v, up, out);
        }
        lanczos.release(scratch);
        result.map(|(_, report)| report)
    }

    /// Weight of `⟨·,·⟩_M` (the power-of-two-scaled mass).
    fn weight(&self, i: usize) -> F {
        self.weights.as_ref().map_or(F::one(), |m| m[i])
    }

    /// The mass `mᵢ` itself (right-hand sides `M·q`).
    fn mass_at(&self, i: usize) -> F {
        self.mass.as_ref().map_or(F::one(), |m| m[i])
    }

    /// `⟨x, y⟩_M`.
    fn dot_m(&self, x: &[F], y: &[F]) -> F {
        x.iter()
            .zip(y)
            .enumerate()
            .fold(F::zero(), |acc, (i, (&a, &b))| acc + self.weight(i) * a * b)
    }

    fn norm_m(&self, x: &[F]) -> F {
        self.dot_m(x, x).sqrt()
    }

    /// `f(θ) = e^{−tλ}`, `λ = 1/θ − 1/γ`; `θ ≤ 0` (a mode beyond rounding) → 0.
    fn f(&self, theta: f64) -> f64 {
        let (t, gamma) = (
            self.t.to_f64().unwrap_or(0.0),
            self.gamma.to_f64().unwrap_or(1.0),
        );
        if theta > 0.0 {
            libm::exp(t / gamma - t / theta)
        } else {
            0.0
        }
    }
}

/// One-shot `out ← e^{−t·M⁻¹A}·v` (builds the factor, applies once).
///
/// # Errors
/// As [`RationalKrylov::new`] and [`RationalKrylov::apply_into`].
#[allow(clippy::too_many_arguments)] // mirrors `graph_expmv_krylov`
pub fn rational_krylov_expmv<F: SemiflowFloat>(
    op: &SymmetricOperator<F>,
    mass: Option<&[F]>,
    t: F,
    v: &[F],
    out: &mut [F],
    opts: RationalOptions,
    scratch: &mut ScratchPool<F>,
) -> Result<RationalReport, SemiflowError> {
    RationalKrylov::new(op, mass, t, opts)?.apply_into(v, out, scratch)
}

// ---------------------------------------------------------------------------
// RationalKrylovChernoff — the action as a ChernoffFunction on graph signals
// ---------------------------------------------------------------------------

/// [`RationalKrylov`] as a [`ChernoffFunction`] on [`GraphSignal`]s, for
/// [`crate::graph_batched::evolve_batched`] and the generic drivers.
///
/// The factor is built once for the step `τ` given at construction; an
/// `apply_into` with another `τ` builds a temporary factor (allocating, correct).
#[derive(Clone)]
pub struct RationalKrylovChernoff<F: SemiflowFloat = f64> {
    op: SymmetricOperator<F>,
    opts: RationalOptions,
    kernel: RationalKrylov<F>,
}

impl<F: SemiflowFloat> RationalKrylovChernoff<F> {
    /// Factor for step `tau`.
    ///
    /// # Errors
    /// As [`RationalKrylov::new`].
    pub fn new(
        op: &SymmetricOperator<F>,
        mass: Option<&[F]>,
        tau: F,
        opts: RationalOptions,
    ) -> Result<Self, SemiflowError> {
        Ok(Self {
            op: op.clone(),
            opts,
            kernel: RationalKrylov::new(op, mass, tau, opts)?,
        })
    }
}

impl<F: SemiflowFloat> ChernoffFunction<F> for RationalKrylovChernoff<F> {
    type S = GraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let fresh;
        let kernel = if tau == self.kernel.t {
            &self.kernel
        } else {
            fresh = RationalKrylov::new(&self.op, self.kernel.mass.as_deref(), tau, self.opts)?;
            &fresh
        };
        let mut out = scratch.take_vec(src.values().len());
        let r = kernel.apply_into(src.values(), &mut out, scratch);
        if r.is_ok() {
            dst.zero_into();
            dst.axpy_into_slice(F::one(), &out);
        }
        scratch.return_vec(out);
        r.map(|_| ())
    }

    /// `u32::MAX`: tolerance-driven, like the polynomial Krylov paths.
    fn order(&self) -> u32 {
        u32::MAX
    }

    /// `e^{−τM⁻¹A}` is a contraction in the `M`-norm; in the Euclidean norm its
    /// bound is `√(max m / min m)` (1 without a mass).
    fn growth(&self) -> Growth<F> {
        let Some(m) = self.kernel.mass.as_ref() else {
            return Growth::contraction();
        };
        let (lo, hi) = m.iter().fold((F::infinity(), F::zero()), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
        Growth::new((hi / lo).sqrt(), F::zero())
    }
}

#[path = "rational_krylov_factor.rs"]
mod factor;
#[path = "rational_krylov_lanczos.rs"]
mod lanczos;

#[cfg(test)]
#[path = "rational_krylov_tests.rs"]
mod tests;
