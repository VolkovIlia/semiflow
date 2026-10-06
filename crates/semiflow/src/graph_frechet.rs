//! `graph_expmv_frechet` — VJP gradient via the exact Duhamel integral (ADR-0185, ADR-0203).
//!
//! Computes `∂J/∂θ_k` for `J = Σ_c ⟨dj_c, e^{−tL} u0_c⟩`:
//!
//! ```text
//! ∂J/∂θ_k = ∫₀ᵗ ⟨ a(σ), (∂A/∂θ_k) b(σ) ⟩ dσ,   A = −L,
//!           a(σ) = e^{−(t−σ)L} dj,   b(σ) = e^{−σL} u0.
//! ```
//!
//! Exact for ALL directions (including non-commuting `[L, ∂L/∂θ_k] ≠ 0`). The
//! integral is evaluated on the two-sided graded mesh of math §63.3
//! (`graph_frechet_mesh`): split at `t/2`, geometric panels towards both
//! endpoints, 8-point Gauss-Legendre per panel. This resolves the boundary
//! layers of width `1/λ_max` that the pre-ADR-0203 single panel could not; the
//! quadrature error is `≤ 1.1e-14` of the absolute magnitude of the pair terms
//! for EVERY `λ_max·t` (§63.4).
//!
//! # Evaluation order (§63.5)
//!
//! Per half, one of `a`, `b` is *near* its source (recomputed per panel into
//! 8 buffers) and the other is *far* (advanced monotonically, outer panel to
//! inner). Memory is `≤ 15·N` floats plus the operator, independent of
//! `λ_max·t`, of the node count and of `n_params`. Step lengths are
//! differences of node DISTANCES; `σ = t − r` is never formed. Each node ends
//! in one call of `GeneratorSensitivity::accumulate_bilinear`, so the number of
//! propagator actions does not depend on `n_params` (§63.6).

use alloc::vec::Vec;

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph_frechet_mesh::{half_of, half_panels, panel_nodes, GL8},
    graph_krylov::GraphKrylovChernoff,
    graph_sensitivity::GeneratorSensitivity,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// Argument validation helper
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn validate_args<F: SemiflowFloat>(
    t_final: F,
    u0_len: usize,
    dj_len: usize,
    grad_len: usize,
    n_cols: usize,
    n: usize,
    n_p: usize,
) -> Result<(), SemiflowError> {
    if !t_final.is_finite() || t_final <= F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "graph_expmv_frechet: t_final must be finite and positive",
            value: t_final.to_f64().unwrap_or(f64::NAN),
        });
    }
    if u0_len != n_cols * n {
        return Err(SemiflowError::DomainViolation {
            what: "graph_expmv_frechet: u0_cols length != n_cols * n_nodes",
            #[allow(clippy::cast_precision_loss)]
            value: u0_len as f64,
        });
    }
    if dj_len != n_cols * n {
        return Err(SemiflowError::DomainViolation {
            what: "graph_expmv_frechet: dj_cols length != n_cols * n_nodes",
            #[allow(clippy::cast_precision_loss)]
            value: dj_len as f64,
        });
    }
    if grad_len != n_p {
        return Err(SemiflowError::DomainViolation {
            what: "graph_expmv_frechet: grad_w.len() != n_params",
            #[allow(clippy::cast_precision_loss)]
            value: grad_len as f64,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Propagator abstraction (ADR-0203 Amendment 2)
// ---------------------------------------------------------------------------

/// The action `dst ← e^{−τL} src` the §63.5 sweep is generic over.
///
/// Production uses [`GraphKrylovChernoff`]; the in-crate gate
/// `g_frechet_sweep_exact_prop` (§63.7.b) drives the SAME sweep with an exact
/// eigen-propagator so the propagator error does not mask sweep defects.
pub(crate) trait FrechetPropagator<F: SemiflowFloat> {
    /// `dst ← e^{−τL} src` (`dst.len() == src.len()`).
    fn propagate(
        &self,
        tau: F,
        src: &[F],
        dst: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;
}

impl<F: SemiflowFloat> FrechetPropagator<F> for GraphKrylovChernoff<F> {
    fn propagate(
        &self,
        tau: F,
        src: &[F],
        dst: &mut [F],
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        self.expmv_slice(tau, src, dst, scratch)
    }
}

/// Scalar inputs of one sweep.
#[derive(Clone, Copy)]
pub(crate) struct SweepParams<F> {
    /// Operator dimension.
    pub n: usize,
    /// Number of channels.
    pub n_cols: usize,
    /// Evolution time `t`.
    pub t: F,
    /// Spectral bound `ρ̄` the mesh of §63.3 is graded with.
    pub rho: F,
}

// ---------------------------------------------------------------------------
// §63.5 near/far sweep
// ---------------------------------------------------------------------------

/// Work vectors of one channel: 8 near buffers, the far vector, one temporary.
struct Work<F> {
    near: [Vec<F>; 8],
    far: Vec<F>,
    tmp: Vec<F>,
    /// Distance at which `far` currently sits.
    r_far: F,
}

impl<F: SemiflowFloat> Work<F> {
    fn take(n: usize, scratch: &mut ScratchPool<F>) -> Self {
        Self {
            near: core::array::from_fn(|_| scratch.take_vec(n)),
            far: scratch.take_vec(n),
            tmp: scratch.take_vec(n),
            r_far: F::zero(),
        }
    }

    fn give_back(self, scratch: &mut ScratchPool<F>) {
        scratch.return_vec(self.tmp);
        scratch.return_vec(self.far);
        for buf in self.near {
            scratch.return_vec(buf);
        }
    }
}

/// Fixed inputs of one half-sweep.
struct HalfRun<'a, F: SemiflowFloat, Q, P, S> {
    prop: &'a Q,
    sens: &'a P,
    /// Decay-skip predicate `(‖far‖, ‖far_src‖) → stop?` of §63.5.b.
    skip: &'a S,
    par: SweepParams<F>,
    /// `true` on the left half (`r = σ`: near vector is `b`, far vector is `a`).
    near_is_b: bool,
}

/// `‖x‖₂` without intermediate overflow/underflow (scaled by `max |x_i|`).
///
/// A plain sum of squares overflows for entries `≳ 1e154` and underflows for
/// entries `≲ 1e-162`; either would corrupt the decay-skip comparison of §63.5.b.
/// NaN anywhere gives NaN; a true norm above `F::MAX` gives `+∞`.
fn norm2<F: SemiflowFloat>(x: &[F]) -> F {
    let scale = x.iter().fold(F::zero(), |m, &v| {
        let a = v.abs();
        if m.is_nan() || a.is_nan() {
            F::nan()
        } else if a > m {
            a
        } else {
            m
        }
    });
    if scale == F::zero() || !scale.is_finite() {
        return scale;
    }
    let sum = x.iter().fold(F::zero(), |acc, &v| {
        let r = v / scale;
        acc + r * r
    });
    scale * sum.sqrt()
}

/// Near buffers of one panel: `near[0] = e^{−r₀L} src`, `near[q] = e^{−(r_q−r_{q−1})L} near[q−1]`.
fn fill_near_panel<F: SemiflowFloat, Q: FrechetPropagator<F>>(
    prop: &Q,
    near_src: &[F],
    r: &[F; 8],
    near: &mut [Vec<F>; 8],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    prop.propagate(r[0], near_src, &mut near[0], scratch)?;
    for q in 1..8 {
        let (done, rest) = near.split_at_mut(q);
        prop.propagate(r[q] - r[q - 1], &done[q - 1], &mut rest[0], scratch)?;
    }
    Ok(())
}

/// Nodes of one panel in DESCENDING distance: advance `far`, contract.
fn sweep_panel_nodes<F, Q, P, S>(
    run: &HalfRun<'_, F, Q, P, S>,
    dists: &[F; 8],
    panel_len: F,
    work: &mut Work<F>,
    grad: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    Q: FrechetPropagator<F>,
    P: GeneratorSensitivity<F>,
    S: Fn(F, F) -> bool,
{
    for node in (0..8).rev() {
        let step = work.r_far - dists[node];
        if step > F::zero() {
            run.prop
                .propagate(step, &work.far, &mut work.tmp, scratch)?;
            core::mem::swap(&mut work.far, &mut work.tmp);
        }
        work.r_far = dists[node];
        let weight = panel_len * from_f64::<F>(GL8[node].1);
        let (vec_a, vec_b) = if run.near_is_b {
            (&work.far[..], &work.near[node][..])
        } else {
            (&work.near[node][..], &work.far[..])
        };
        run.sens
            .accumulate_bilinear(run.par.t, weight, vec_a, vec_b, grad, scratch)?;
    }
    Ok(())
}

/// One half of `[0, t]` (§63.5): panels outer to inner, decay skip after each.
fn half_sweep<F, Q, P, S>(
    run: &HalfRun<'_, F, Q, P, S>,
    near_src: &[F],
    far_src: &[F],
    w: &mut Work<F>,
    grad: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    Q: FrechetPropagator<F>,
    P: GeneratorSensitivity<F>,
    S: Fn(F, F) -> bool,
{
    let half = half_of(run.par.t);
    run.prop.propagate(half, far_src, &mut w.far, scratch)?;
    w.r_far = half;
    // §63.5.b: once the predicate holds (production: ‖far‖ ≤ tol·‖far_src‖) the rest of
    // this half is below `tol·N_k/2`.
    let src_norm = norm2(far_src);
    for &(lo, h) in half_panels(half, run.par.rho).iter().rev() {
        let r = panel_nodes(lo, h);
        fill_near_panel(run.prop, near_src, &r, &mut w.near, scratch)?;
        sweep_panel_nodes(run, &r, h, w, grad, scratch)?;
        if (run.skip)(norm2(&w.far), src_norm) {
            break;
        }
    }
    Ok(())
}

/// Exact VJP for one channel: accumulates into `grad` (not zeroed here).
///
/// Left half first (`near = b`), then right half (`near = a`), §63.5.
fn frechet_channel<F, Q, P, S>(
    run: &HalfRun<'_, F, Q, P, S>,
    u0_c: &[F],
    dj_c: &[F],
    grad: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    Q: FrechetPropagator<F>,
    P: GeneratorSensitivity<F>,
    S: Fn(F, F) -> bool,
{
    let mut work = Work::take(u0_c.len(), scratch);
    let left = HalfRun {
        near_is_b: true,
        ..*run
    };
    let mut status = half_sweep(&left, u0_c, dj_c, &mut work, grad, scratch);
    if status.is_ok() {
        let right = HalfRun {
            near_is_b: false,
            ..*run
        };
        status = half_sweep(&right, dj_c, u0_c, &mut work, grad, scratch);
    }
    work.give_back(scratch);
    status
}

/// The decay-skip predicate of §63.5.b: stop a half once `‖far‖ ≤ tol·‖far_src‖`.
///
/// A non-finite level `tol·‖far_src‖` never skips. Production and the part-B gate of
/// `G_FRECHET_SWEEP_EXACT_PROP` (§63.7.c) share this one definition.
pub(crate) fn decay_skip<F: SemiflowFloat>(tol: F) -> impl Fn(F, F) -> bool {
    move |far, far_src| {
        let level = tol * far_src;
        level.is_finite() && far <= level
    }
}

/// The §63.5 sweep for all channels, generic over the propagator (inputs already validated).
///
/// Zeroes `grad`, then accumulates channels in ascending order (ADR-0184 D4).
#[allow(clippy::too_many_arguments)] // propagator, sensitivity, skip predicate, params, columns, grad, scratch
pub(crate) fn frechet_sweep<F, Q, P, S>(
    prop: &Q,
    sens: &P,
    skip: &S,
    par: SweepParams<F>,
    (u0_cols, dj_cols): (&[F], &[F]),
    grad: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    Q: FrechetPropagator<F>,
    P: GeneratorSensitivity<F>,
    S: Fn(F, F) -> bool,
{
    for g in grad.iter_mut() {
        *g = F::zero();
    }
    let run = HalfRun {
        prop,
        sens,
        skip,
        par,
        near_is_b: true,
    };
    for c in 0..par.n_cols {
        let (lo, hi) = (c * par.n, (c + 1) * par.n);
        frechet_channel(&run, &u0_cols[lo..hi], &dj_cols[lo..hi], grad, scratch)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// VJP gradient `∂J/∂θ` for `J = Σ_c ⟨dj_c, e^{−t L} u0_c⟩` (§54.5, ADR-0185; §63, ADR-0203).
///
/// Evaluates the exact Duhamel integral
/// `∂J/∂θ_k = ∫₀ᵗ ⟨e^{−(t−σ)L}dj_c, (∂A/∂θ_k) e^{−σL}u0_c⟩ dσ` on the two-sided
/// graded Gauss-Legendre mesh of §63.3, with quadrature error independent of `λ_max·t` (the
/// pre-ADR-0203 single 8-point panel lost all digits beyond `λ_max·t ≈ 30`).
/// Exact for ALL graph topologies, including non-commuting directions.
/// Cost: `2 + 32(K+1)` propagator calls and `≤ m_Z·(3ρ̄t/Z_SAFE + 2 + 32(K+1))`
/// `SpMV`s (§63.6.c; `→ 6×` one action as `ρ̄t → ∞`), plus `O(nnz)` per
/// contraction; independent of `n_params` for the in-tree sensitivities.
/// The f64 conditioning floor `≈ (r+n)·u·ρ̄t` of any `SpMV`-based method remains.
/// `L` must be PSD (§63.1.d).
/// With the `ImplicitEuler` path the result is the gradient of the backward-Euler
/// propagator (`O(Δt)` bias, outside the §63.7 bound).
///
/// # Arguments
///
/// * `gk` — Krylov solver owning the fixed Laplacian `L` (A1 primitive).
/// * `u0_cols` — flat row-major initial states; slice `c·n … (c+1)·n` = `u0_c`.
/// * `dj_cols` — flat row-major loss-gradient vectors; same layout as `u0_cols`.
/// * `n_cols` — number of channels (D4 ascending sweep).
/// * `t_final` — evolution time `t > 0`.
/// * `param_deriv` — `(∂A/∂θ_k) v` provider (`A = −L`, generator sign).
/// * `grad_w` — output slice of length `n_params`; zeroed then accumulated.
/// * `scratch` — reusable buffer pool.
///
/// # Errors
///
/// Returns `DomainViolation` if `t_final ≤ 0`, any length mismatch, or the
/// inner Krylov solve fails.
#[allow(clippy::too_many_arguments)]
pub fn graph_expmv_frechet<F, P>(
    gk: &GraphKrylovChernoff<F>,
    u0_cols: &[F],
    dj_cols: &[F],
    n_cols: usize,
    t_final: F,
    param_deriv: &P,
    grad_w: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    let n = gk.n_nodes();
    let n_p = param_deriv.n_params();
    validate_args(
        t_final,
        u0_cols.len(),
        dj_cols.len(),
        grad_w.len(),
        n_cols,
        n,
        n_p,
    )?;
    let par = SweepParams {
        n,
        n_cols,
        t: t_final,
        rho: gk.lambda_max_bound(),
    };
    let skip = decay_skip(gk.tol());
    frechet_sweep(
        gk,
        param_deriv,
        &skip,
        par,
        (u0_cols, dj_cols),
        grad_w,
        scratch,
    )
}
