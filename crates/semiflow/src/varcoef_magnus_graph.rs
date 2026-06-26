//! Variable-coefficient × time-dependent Magnus K=4 on graphs (ADR-0063).
//!
//! Composes the variable-coefficient operator `L_a(t) = A(t)^{1/2} L_G(t) A(t)^{1/2}`
//! from `graph_var_coef.rs` with the order-4 Magnus expansion from
//! `magnus_graph.rs`. At each GL₂ quadrature point `c_i · τ` the library samples
//! BOTH `a(t)` and `L_G(t)` and assembles `Ω₄(τ)` using the operator-form
//! `L_a(t)` instead of a raw Laplacian.
//!
//! See math.md §20 (NORMATIVE) and ADR-0063 §"Decision".
//!
//! ## Per-step cost (NORMATIVE — ADR-0063)
//! - 2 weight-function calls (`a_at_t(c_i τ)`)
//! - 2 Laplacian-function calls (`lap_at_t(c_i τ)`)
//! - 4 sqrt-vector computations (size N each)
//! - Ω₄·v: 4 `apply_la_on_slice` calls = 4 × (2·N diag + 1 `SpMV`)
//! - exp(Ω₄)·v Taylor degree-4: 4 × `apply_omega4_la` = 16 `apply_la_on_slice` calls
//!   total per step (vs. 16 raw `SpMV` for `MagnusGraphHeatChernoff`).
//!
//! ## Zero-alloc steady state (R4 invariant)
//! `apply_into` acquires 9 scratch buffers from `ScratchPool` (2 `sqrt_a` + 5 Ω₄
//! intermediates + 2 `apply_la_on_slice` intermediates) and returns them all
//! before returning. No heap allocations after warm-up.

// Grid/index/count values (usize) cast to f64 for coordinate and coefficient computations;
// all values are grid sizes or step counts ≪ 2^52, so precision loss is impossible in practice.
#![allow(clippy::cast_precision_loss)]

use alloc::{boxed::Box, vec::Vec};

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph::Laplacian,
    graph_signal::GraphSignal,
    graph_var_coef::apply_la_on_slice,
    magnus_graph::LaplacianAtTime,
    scratch::ScratchPool,
    state::State,
};

/// GL₂ abscissa `c₁ = (3 − √3) / 6` ≈ 0.2113.
const GL4_C1_F64: f64 = 0.211_324_865_405_187_13;
/// GL₂ abscissa `c₂ = (3 + √3) / 6` ≈ 0.7887.
const GL4_C2_F64: f64 = 0.788_675_134_594_812_9;
/// `√3 / 12` ≈ 0.1443 (Ω₄ commutator coefficient).
const SQRT3_OVER_12_F64: f64 = 0.144_337_567_297_406_45;

/// Caller-supplied closure mapping absolute time `t` to a node-weight vector
/// `a(t) ∈ (0, ∞)^N`. All entries MUST be strictly positive and finite for
/// every `t` in the integration interval; `apply_into` returns
/// `DomainViolation` if any entry violates the constraint at a sampled
/// quadrature point.
pub type WeightAtTime<F> = Box<dyn Fn(F) -> Vec<F> + Send + Sync>;

/// Order-4 Magnus Chernoff for `∂_t u = −L_a(t) u` with
/// `L_a(t) = A(t)^{1/2} L_G(t) A(t)^{1/2}` and time-varying node weights
/// `a(t)` AND edge weights (via `L_G(t)`).
///
/// Implements [`ChernoffFunction<F, S = GraphSignal<F>>`] with `order() == 4`.
/// See math.md §20 and ADR-0063.
///
/// # Convergence
///
/// Global error `‖(S₄(t/n))^n f − u_exact(t)‖₂ = O(1/n⁴)` by Iserles+ 2000 §5
/// Theorem 5.2 + Chernoff product formula. The variable-coefficient
/// composition does not degrade the order; only round-off scales by
/// `a_sup_max²` per step.
///
/// # Convergence radius
///
/// Each `apply_into` validates `ρ̄_max · a_sup_max² · τ < π/2`. On violation
/// returns [`SemiflowError::OutOfMagnusRadius`].
pub struct VarCoefMagnusGraphHeatChernoff<F: SemiflowFloat = f64> {
    pub(crate) n_nodes: usize,
    pub(crate) lap_at_t: LaplacianAtTime<F>,
    pub(crate) a_at_t: WeightAtTime<F>,
    pub(crate) rho_bar_max: F,
    pub(crate) a_sup_max: F,
    pub(crate) convergence_radius_check: bool,
}

impl<F: SemiflowFloat> VarCoefMagnusGraphHeatChernoff<F> {
    /// Construct from time-to-Laplacian + time-to-weights closures.
    ///
    /// # Parameters
    /// - `n_nodes`: graph node count; must match output length of both
    ///   `lap_at_t(t).n_nodes()` and `a_at_t(t).len()` for every sampled `t`.
    /// - `lap_at_t`: closure `t ↦ Arc<Laplacian<F>>`. Topology
    ///   (`row_ptr`, `col_idx`) MUST be invariant in `t`.
    /// - `a_at_t`: closure `t ↦ Vec<F>` returning length-`N` node weights.
    ///   Each entry MUST be strictly positive and finite at every sampled `t`.
    /// - `rho_bar_max`: caller-supplied upper bound for `max_t ρ̄(L_G(t))`.
    /// - `a_sup_max`: caller-supplied upper bound for `max_t sqrt(max_i a_i(t))`.
    ///   (Note: `sqrt(max a)`, not `max sqrt(a)` — same value, but the
    ///   former is the typical caller computation.)
    ///
    /// # Errors
    /// `DomainViolation` if `n_nodes == 0`, `rho_bar_max <= 0`, `a_sup_max <= 0`,
    /// or either is non-finite.
    pub fn new(
        n_nodes: usize,
        lap_at_t: LaplacianAtTime<F>,
        a_at_t: WeightAtTime<F>,
        rho_bar_max: F,
        a_sup_max: F,
    ) -> Result<Self, SemiflowError> {
        validate_positive(rho_bar_max, "rho_bar_max")?;
        validate_positive(a_sup_max, "a_sup_max")?;
        if n_nodes == 0 {
            return Err(SemiflowError::DomainViolation {
                what: "VarCoefMagnusGraphHeatChernoff: n_nodes must be >= 1",
                value: 0.0,
            });
        }
        Ok(Self {
            n_nodes,
            lap_at_t,
            a_at_t,
            rho_bar_max,
            a_sup_max,
            convergence_radius_check: true,
        })
    }

    /// Toggle the Magnus convergence-radius check (default: `true`).
    #[must_use]
    pub fn with_radius_check(mut self, enabled: bool) -> Self {
        self.convergence_radius_check = enabled;
        self
    }

    /// Number of nodes the operator acts on.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Apply one Magnus K=4 step starting at absolute time `t_start`.
    ///
    /// # Errors
    /// Same as [`ChernoffFunction::apply_into`].
    pub fn apply_into_at(
        &self,
        t_start: F,
        tau: F,
        src: &GraphSignal<F>,
        dst: &mut GraphSignal<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        apply_varcoef_magnus_k4_at(self, t_start, tau, src, dst, scratch)
    }
}

impl<F: SemiflowFloat> ChernoffFunction<F> for VarCoefMagnusGraphHeatChernoff<F> {
    type S = GraphSignal<F>;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        apply_varcoef_magnus_k4_at(self, F::zero(), tau, src, dst, scratch)
    }

    /// Returns `4`.
    fn order(&self) -> u32 {
        4
    }

    /// Returns `Growth { multiplier: 1, omega: ρ̄_max · a_sup_max² }` (quasi-contractivity bound
    /// `‖exp(Ω₄(τ))‖₂ ≤ exp(τ · ρ̄_max · a_sup_max²)`).
    fn growth(&self) -> Growth<F> {
        Growth {
            multiplier: F::one(),
            omega: self.rho_bar_max * self.a_sup_max * self.a_sup_max,
        }
    }
}

/// Estimate `(rho_bar_max, a_sup_max)` over `[interval.0, interval.1]` via
/// `n_samples` equispaced sample points.
///
/// For each sample `t_k`, this calls `lap_at_t(t_k).spectral_radius_bound()`
/// (Gershgorin-tight bound; see `graph::Laplacian::spectral_radius_bound`)
/// and `a_at_t(t_k).iter().max()`, then returns the per-axis maxima as
/// `(rho_bar_max, sqrt(max_t max_i a_i(t)))`.
///
/// Use this when the caller does not have closed-form bounds. The
/// recommended default is `n_samples = 32`.
///
/// # Panics
/// Panics if `n_samples == 0`.
pub fn compute_rho_bar<F: SemiflowFloat>(
    lap_at_t: &LaplacianAtTime<F>,
    a_at_t: &WeightAtTime<F>,
    interval: (F, F),
    n_samples: usize,
) -> (F, F) {
    assert!(n_samples >= 1, "compute_rho_bar: n_samples must be >= 1");
    let (t0, t1) = interval;
    let mut rho_max = F::zero();
    let mut a_max = F::zero();
    let denom = F::from(n_samples.max(1)).unwrap_or(F::one());
    for k in 0..n_samples {
        let frac = F::from(k).unwrap_or(F::zero()) / denom;
        let t = t0 + (t1 - t0) * frac;
        let lap = lap_at_t(t);
        let rho = lap.spectral_radius_bound();
        if rho > rho_max {
            rho_max = rho;
        }
        let a = a_at_t(t);
        for &ai in &a {
            if ai > a_max {
                a_max = ai;
            }
        }
    }
    (rho_max, a_max.sqrt())
}

fn validate_positive<F: SemiflowFloat>(val: F, name: &'static str) -> Result<(), SemiflowError> {
    if !val.is_finite() || val <= F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: name,
            value: val.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

pub(crate) fn validate_tau<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    if !tau.is_finite() || tau < F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "VarCoefMagnusGraphHeatChernoff: tau must be finite and >= 0",
            value: tau.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

pub(crate) fn validate_magnus_radius<F: SemiflowFloat>(
    rho_bar_max: F,
    a_sup_max: F,
    tau: F,
) -> Result<(), SemiflowError> {
    let radius = rho_bar_max * a_sup_max * a_sup_max * tau;
    let half_pi = from_f64::<F>(core::f64::consts::FRAC_PI_2);
    if radius >= half_pi {
        return Err(SemiflowError::OutOfMagnusRadius {
            tau: tau.to_f64().unwrap_or(f64::NAN),
            rho_estimate: (rho_bar_max * a_sup_max * a_sup_max)
                .to_f64()
                .unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

fn apply_varcoef_magnus_k4_at<F: SemiflowFloat>(
    mc: &VarCoefMagnusGraphHeatChernoff<F>,
    t_start: F,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    validate_tau(tau)?;
    if mc.convergence_radius_check {
        validate_magnus_radius(mc.rho_bar_max, mc.a_sup_max, tau)?;
    }
    let n = mc.n_nodes;
    debug_assert_eq!(src.len(), n, "varcoef_magnus: src len mismatch");
    debug_assert_eq!(dst.len(), n, "varcoef_magnus: dst len mismatch");
    let c1 = from_f64::<F>(GL4_C1_F64);
    let c2 = from_f64::<F>(GL4_C2_F64);
    let lap1 = (mc.lap_at_t)(t_start + c1 * tau);
    let lap2 = (mc.lap_at_t)(t_start + c2 * tau);
    let a1 = (mc.a_at_t)(t_start + c1 * tau);
    let a2 = (mc.a_at_t)(t_start + c2 * tau);
    validate_a_weights(n, &a1, &a2)?;
    let mut sqrt_a1 = scratch.take_vec(n);
    let mut sqrt_a2 = scratch.take_vec(n);
    for i in 0..n {
        sqrt_a1[i] = a1[i].sqrt();
        sqrt_a2[i] = a2[i].sqrt();
    }
    apply_exp_omega4_la_kernel(&lap1, &sqrt_a1, &lap2, &sqrt_a2, tau, src, dst, scratch);
    scratch.return_vec(sqrt_a2);
    scratch.return_vec(sqrt_a1);
    Ok(())
}

/// Validate that sampled weight vectors have correct length and positive finite entries.
pub(crate) fn validate_a_weights<F: SemiflowFloat>(
    n: usize,
    a1: &[F],
    a2: &[F],
) -> Result<(), SemiflowError> {
    if a1.len() != n || a2.len() != n {
        return Err(SemiflowError::DomainViolation {
            what: "VarCoefMagnusGraphHeatChernoff: a_at_t(t).len() != n_nodes",
            value: a1.len().max(a2.len()) as f64,
        });
    }
    for &ai in a1.iter().chain(a2.iter()) {
        if !ai.is_finite() || ai <= F::zero() {
            return Err(SemiflowError::DomainViolation {
                what:
                    "VarCoefMagnusGraphHeatChernoff: a_at_t(t) must be strictly positive and finite",
                value: ai.to_f64().unwrap_or(f64::NAN),
            });
        }
    }
    Ok(())
}

/// Degree-4 Taylor truncation of `exp(Ω₄) · src → dst` using the operator-form
/// `L_a(t)`. Mirrors `magnus_graph::apply_exp_omega4_kernel` but with
/// `apply_la_on_slice` instead of raw `Laplacian::apply_into_slice`.
///
/// R4 zero-alloc: 7 scratch buffers acquired and returned.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_exp_omega4_la_kernel<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    sqrt_a1: &[F],
    lap2: &Laplacian<F>,
    sqrt_a2: &[F],
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) {
    let n = src.len();
    let mut omega_v = scratch.take_vec(n);
    let mut omega_pow = scratch.take_vec(n);
    let mut tmp_a = scratch.take_vec(n);
    let mut tmp_b = scratch.take_vec(n);
    let mut tmp_c = scratch.take_vec(n);
    let mut vt1 = scratch.take_vec(n);
    let mut vt2 = scratch.take_vec(n);
    accumulate_la_taylor4(
        lap1,
        sqrt_a1,
        lap2,
        sqrt_a2,
        tau,
        src,
        dst,
        &mut omega_v,
        &mut omega_pow,
        &mut tmp_a,
        &mut tmp_b,
        &mut tmp_c,
        &mut vt1,
        &mut vt2,
    );
    scratch.return_vec(vt2);
    scratch.return_vec(vt1);
    scratch.return_vec(tmp_c);
    scratch.return_vec(tmp_b);
    scratch.return_vec(tmp_a);
    scratch.return_vec(omega_pow);
    scratch.return_vec(omega_v);
}

/// Degree-4 Taylor accumulation for LA forward map (`comm_sign = +1`).
/// Verbatim extraction from `apply_exp_omega4_la_kernel` — float op order preserved.
#[allow(clippy::too_many_arguments)]
fn accumulate_la_taylor4<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    sqrt_a1: &[F],
    lap2: &Laplacian<F>,
    sqrt_a2: &[F],
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    omega_v: &mut [F],
    omega_pow: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
    tmp_c: &mut [F],
    vt1: &mut [F],
    vt2: &mut [F],
) {
    let one = F::one();
    let two = one + one;
    let six = two + two + two;
    let twenty_four = (two + two) * (two + one) * two;
    let src_v = src.values(); // local borrow to keep call-sites compact
                              // k=1: Ω·src → omega_v; dst ← src + omega_v
    apply_omega4_la_step(
        lap1, sqrt_a1, lap2, sqrt_a2, tau, one, src_v, omega_v, tmp_a, tmp_b, tmp_c, vt1, vt2,
    );
    dst.copy_from(src);
    dst.axpy_into_slice(one, omega_v);
    // k=2: Ω²·src → omega_v; dst += (1/2)·omega_v
    omega_pow.copy_from_slice(omega_v);
    apply_omega4_la_step(
        lap1, sqrt_a1, lap2, sqrt_a2, tau, one, omega_pow, omega_v, tmp_a, tmp_b, tmp_c, vt1, vt2,
    );
    dst.axpy_into_slice(one / two, omega_v);
    // k=3: Ω³·src → omega_v; dst += (1/6)·omega_v
    omega_pow.copy_from_slice(omega_v);
    apply_omega4_la_step(
        lap1, sqrt_a1, lap2, sqrt_a2, tau, one, omega_pow, omega_v, tmp_a, tmp_b, tmp_c, vt1, vt2,
    );
    dst.axpy_into_slice(one / six, omega_v);
    // k=4: Ω⁴·src → omega_v; dst += (1/24)·omega_v
    omega_pow.copy_from_slice(omega_v);
    apply_omega4_la_step(
        lap1, sqrt_a1, lap2, sqrt_a2, tau, one, omega_pow, omega_v, tmp_a, tmp_b, tmp_c, vt1, vt2,
    );
    dst.axpy_into_slice(one / twenty_four, omega_v);
}

/// One `Ω₄·v` application step; thin alias to avoid repetitive call-site unrolling.
#[allow(clippy::too_many_arguments)]
fn apply_omega4_la_step<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    sqrt_a1: &[F],
    lap2: &Laplacian<F>,
    sqrt_a2: &[F],
    tau: F,
    comm_sign: F,
    v: &[F],
    out: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
    tmp_c: &mut [F],
    vt1: &mut [F],
    vt2: &mut [F],
) {
    apply_omega4_la(
        lap1, sqrt_a1, lap2, sqrt_a2, tau, comm_sign, v, out, tmp_a, tmp_b, tmp_c, vt1, vt2,
    );
}

/// Apply `Ω₄ · v → out` (forward) or `Ω₄ᵀ · v → out` (adjoint) for `L_a(t)`.
///
/// ```text
/// Ω₄·v = −(τ/2)·(L_a1+L_a2)·v + comm_sign·(√3τ²/12)·[L_a2,L_a1]·v
/// ```
///
/// `comm_sign = +1` for the forward map; `comm_sign = −1` for the adjoint
/// (math.md §42, Theorem 42.1).
///
/// Mirrors `magnus_graph::apply_omega4` but uses `apply_la_on_slice` for
/// every operator application. Two extra scratch slices (`var_tmp1`,
/// `var_tmp2`) are reused across all four LA-applications.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_omega4_la<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    sqrt_a1: &[F],
    lap2: &Laplacian<F>,
    sqrt_a2: &[F],
    tau: F,
    comm_sign: F,
    v: &[F],
    out: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
    tmp_c: &mut [F],
    var_tmp1: &mut [F],
    var_tmp2: &mut [F],
) {
    let n = v.len();
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(tmp_a.len(), n);
    debug_assert_eq!(tmp_b.len(), n);
    debug_assert_eq!(tmp_c.len(), n);

    let half = from_f64::<F>(0.5);
    let sqrt3_over_12 = from_f64::<F>(SQRT3_OVER_12_F64);

    apply_la_on_slice(lap1, sqrt_a1, v, tmp_a, var_tmp1, var_tmp2);
    apply_la_on_slice(lap2, sqrt_a2, v, tmp_b, var_tmp1, var_tmp2);
    apply_la_on_slice(lap2, sqrt_a2, tmp_a, tmp_c, var_tmp1, var_tmp2);

    let scale = -half * tau;
    for i in 0..n {
        out[i] = scale * (tmp_a[i] + tmp_b[i]);
    }

    apply_la_on_slice(lap1, sqrt_a1, tmp_b, tmp_a, var_tmp1, var_tmp2);

    let comm_scale = comm_sign * sqrt3_over_12 * tau * tau;
    for i in 0..n {
        out[i] += comm_scale * (tmp_c[i] - tmp_a[i]);
    }
}

// Unit tests live in `crates/semiflow-core/tests/varcoef_magnus_unit.rs`
// (extracted to keep the library file under the 500-LoC cap per
// `.dev-docs/constitution.md` Override #1 budget).
