//! Private free-function helpers for [`crate::magnus_graph`].
//!
//! Validation, trajectory utilities, and the Taylor-truncation kernel.
//! Re-exported from `magnus_graph` via `pub(crate) use`, so external
//! callers keep the canonical `crate::magnus_graph::*` paths.

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph::Laplacian,
    graph_signal::GraphSignal,
    graph_traj::{GraphTraj, SegmentWeightFn},
    magnus_graph::{MagnusGraphHeatChernoff, GL4_C1_F64, GL4_C2_F64, SQRT3_OVER_12_F64},
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that `rho_bar_max` is finite and strictly positive.
#[inline]
pub(crate) fn validate_rho<F: SemiflowFloat>(rho: F) -> Result<(), SemiflowError> {
    let v = rho.to_f64().unwrap_or(f64::NAN);
    if !v.is_finite() || v <= 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "MagnusGraphHeatChernoff: rho_bar_max must be finite and > 0",
            value: v,
        });
    }
    Ok(())
}

/// Validate that `tau` is finite and non-negative.
#[inline]
pub(crate) fn validate_tau<F: SemiflowFloat>(tau: F) -> Result<(), SemiflowError> {
    let v = tau.to_f64().unwrap_or(f64::NAN);
    if !v.is_finite() || v < 0.0 {
        return Err(SemiflowError::DomainViolation {
            what: "MagnusGraphHeatChernoff: tau must be finite and >= 0",
            value: v,
        });
    }
    Ok(())
}

/// Check Magnus convergence radius: `ρ̄_max · τ < π/2` (50% safety margin).
///
/// Theoretical radius: `∫₀^τ ‖A(s)‖₂ ds < π` (Blanes+ 2009 Theorem 1).
/// Library policy: 50% safety margin → `ρ̄_max · τ < π/2`.
#[inline]
pub(crate) fn validate_magnus_radius<F: SemiflowFloat>(
    rho_bar_max: F,
    tau: F,
) -> Result<(), SemiflowError> {
    let radius = rho_bar_max * tau;
    let half_pi = from_f64::<F>(core::f64::consts::FRAC_PI_2);
    if radius >= half_pi {
        return Err(SemiflowError::OutOfMagnusRadius {
            tau: tau.to_f64().unwrap_or(f64::NAN),
            rho_estimate: rho_bar_max.to_f64().unwrap_or(f64::NAN),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Trajectory helpers
// ---------------------------------------------------------------------------

/// Validate inputs for `evolve_with_traj_into`.
#[allow(clippy::cast_precision_loss)] // usize→f64 for error reporting only; not on hot path
pub(crate) fn validate_traj_inputs<F: SemiflowFloat>(
    traj: &GraphTraj<F>,
    n_steps_per_segment: usize,
    f0: &GraphSignal<F>,
) -> Result<(), SemiflowError> {
    if n_steps_per_segment == 0 {
        return Err(SemiflowError::DomainViolation {
            what: "evolve_with_traj: n_steps_per_segment must be >= 1",
            value: 0.0,
        });
    }
    let first_bp = traj.breakpoints().first().copied().unwrap_or(F::zero());
    if first_bp != F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "evolve_with_traj: traj.breakpoints()[0] must be 0",
            value: first_bp.to_f64().unwrap_or(f64::NAN),
        });
    }
    let snap0_n = traj.snapshot(0).map_or(0, |g| g.n_nodes());
    if f0.len() != snap0_n {
        return Err(SemiflowError::DomainViolation {
            what: "evolve_with_traj: f0.n_nodes() != traj.snapshot(0).n_nodes()",
            value: f0.len() as f64,
        });
    }
    for k in 1..traj.n_segments() {
        let nk = traj.snapshot(k).map_or(0, |g| g.n_nodes());
        if nk != snap0_n {
            return Err(SemiflowError::DomainViolation {
                what: "evolve_with_traj: all segments must have the same vertex count",
                value: nk as f64,
            });
        }
    }
    Ok(())
}

/// Outer segment loop for `evolve_with_traj_into`.
///
/// Iterates over all `n_seg` segments; for each, validates the Magnus radius,
/// resets `cur` to the current `dst`, then calls `run_segment_steps`.
pub(crate) fn run_all_segments<F: SemiflowFloat>(
    mc: &MagnusGraphHeatChernoff<F>,
    traj: &GraphTraj<F>,
    n_steps_per_segment: usize,
    dst: &mut GraphSignal<F>,
    cur: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    for k in 0..traj.n_segments() {
        let seg_start = traj.breakpoints()[k];
        let seg_end = traj.breakpoints()[k + 1];
        let n_steps_f = F::from(n_steps_per_segment).unwrap_or(F::one());
        let tau = (seg_end - seg_start) / n_steps_f;

        if mc.convergence_radius_check {
            validate_magnus_radius(mc.rho_bar_max, tau)?;
        }
        cur.copy_from(dst);

        let weight_fn = traj.weight_fns_segment(k);
        run_segment_steps(
            weight_fn,
            seg_start,
            tau,
            n_steps_per_segment,
            cur,
            dst,
            scratch,
        )?;

        dst.copy_from(cur);
    }
    Ok(())
}

/// Inner step loop for a single trajectory segment.
///
/// Runs `n_steps` Magnus K=4 steps using `weight_fn`, ping-ponging between
/// `cur` (input) and `dst` (output). After the loop `cur` holds the result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_segment_steps<F: SemiflowFloat>(
    weight_fn: &SegmentWeightFn<F>,
    seg_start: F,
    tau: F,
    n_steps: usize,
    cur: &mut GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    for step in 0..n_steps {
        let step_f = F::from(step).unwrap_or(F::zero());
        let t_start = seg_start + step_f * tau;
        apply_magnus_k4_with_fn(weight_fn, t_start, tau, cur, dst, scratch)?;
        core::mem::swap(cur, dst);
    }
    Ok(())
}

/// Apply one Magnus K=4 step using a borrowed `SegmentWeightFn`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_magnus_k4_with_fn<F: SemiflowFloat>(
    weight_fn: &SegmentWeightFn<F>,
    t_start: F,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    GraphSignal<F>: Clone,
{
    validate_tau(tau)?;
    let n = src.len();
    debug_assert_eq!(dst.len(), n, "apply_magnus_k4_with_fn: dst len mismatch");

    let c1 = from_f64::<F>(GL4_C1_F64);
    let c2 = from_f64::<F>(GL4_C2_F64);
    let lap1 = weight_fn(t_start + c1 * tau);
    let lap2 = weight_fn(t_start + c2 * tau);

    apply_exp_omega4_kernel(&lap1, &lap2, tau, src, dst, scratch);
    Ok(())
}

// ---------------------------------------------------------------------------
// Core kernel
// ---------------------------------------------------------------------------

/// Apply one Magnus K=4 step: `dst ← exp(Ω₄(t_start, τ)) · src`.
pub(crate) fn apply_magnus_k4_into_at<F: SemiflowFloat>(
    mc: &MagnusGraphHeatChernoff<F>,
    t_start: F,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    validate_tau(tau)?;
    if mc.convergence_radius_check {
        validate_magnus_radius(mc.rho_bar_max, tau)?;
    }
    let n = src.len();
    debug_assert_eq!(dst.len(), n, "apply_magnus_k4_into_at: dst len mismatch");

    let c1 = from_f64::<F>(GL4_C1_F64);
    let c2 = from_f64::<F>(GL4_C2_F64);
    let lap1 = (mc.lap_at_t)(t_start + c1 * tau);
    let lap2 = (mc.lap_at_t)(t_start + c2 * tau);

    debug_assert_laplacian_topology(&lap1, &lap2, mc.graph.n_nodes());
    apply_exp_omega4_kernel(&lap1, &lap2, tau, src, dst, scratch);
    Ok(())
}

/// Debug-only topology-drift check (contract §6.4).
///
/// Compares two sampled Laplacians for topology equality.
/// NOTE: `Laplacian::row_ptr()` includes diagonal entries (invariant L1).
#[inline]
pub(crate) fn debug_assert_laplacian_topology<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    expected_n_nodes: usize,
) {
    #[cfg(debug_assertions)]
    {
        debug_assert_eq!(
            lap1.row_ptr(),
            lap2.row_ptr(),
            "MagnusGraphHeatChernoff: topology drift — \
             lap_at_t(t0+c1·τ).row_ptr() ≠ lap_at_t(t0+c2·τ).row_ptr()"
        );
        debug_assert_eq!(
            lap1.col_idx(),
            lap2.col_idx(),
            "MagnusGraphHeatChernoff: topology drift — \
             lap_at_t(t0+c1·τ).col_idx() ≠ lap_at_t(t0+c2·τ).col_idx()"
        );
        debug_assert_eq!(
            lap1.n_nodes(),
            expected_n_nodes,
            "MagnusGraphHeatChernoff: topology drift — \
             sampled Laplacian n_nodes ≠ topology graph n_nodes"
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (lap1, lap2, expected_n_nodes);
    }
}

// ---------------------------------------------------------------------------
// Taylor-truncation kernel: exp(Ω₄)·src via degree-4 expansion
// ---------------------------------------------------------------------------

/// Acquire scratch buffers, compute `Σ_{k=0..4} Ω₄^k/k! · src`, write to
/// `dst`, return buffers (R4 zero-alloc wrapper).
pub(crate) fn apply_exp_omega4_kernel<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) {
    let n = src.len();
    let mut ov = scratch.take_vec(n);
    let mut op = scratch.take_vec(n);
    let mut ta = scratch.take_vec(n);
    let mut tb = scratch.take_vec(n);
    let mut tc = scratch.take_vec(n);
    compute_taylor4(
        lap1, lap2, tau, src, dst, &mut ov, &mut op, &mut ta, &mut tb, &mut tc,
    );
    scratch.return_vec(ov);
    scratch.return_vec(op);
    scratch.return_vec(ta);
    scratch.return_vec(tb);
    scratch.return_vec(tc);
}

/// Compute degree-4 Taylor: `dst ← Σ_{k=0..4} Ω₄^k/k! · src` using
/// pre-acquired scratch slices.  Called by `apply_exp_omega4_kernel`.
#[allow(clippy::too_many_arguments)]
fn compute_taylor4<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    omega_v: &mut [F],
    omega_pow: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
    tmp_c: &mut [F],
) {
    let one = F::one();
    let (two, six, twenty_four) = taylor_denominators(one);
    // k=1: Ω₄·src → omega_v; dst ← src + (1/1!)·omega_v.
    apply_omega4(
        lap1,
        lap2,
        tau,
        one,
        src.values(),
        omega_v,
        tmp_a,
        tmp_b,
        tmp_c,
    );
    dst.copy_from(src);
    dst.axpy_into_slice(one, omega_v);
    // k=2,3,4.
    accumulate_taylor_terms(
        lap1,
        lap2,
        tau,
        one,
        two,
        six,
        twenty_four,
        dst,
        omega_v,
        omega_pow,
        tmp_a,
        tmp_b,
        tmp_c,
    );
}

/// Compute factorial denominators for the degree-4 Taylor expansion.
///
/// Returns `(2!, 3!, 4!)` as `F` values built without integer literals
/// (generic-over-float constraint requires arithmetic on `F::one()`).
#[inline]
fn taylor_denominators<F: SemiflowFloat>(one: F) -> (F, F, F) {
    let two = one + one;
    let six = two + two + two;
    let twenty_four = (two + two) * (two + one) * two;
    (two, six, twenty_four)
}

/// Accumulate Taylor terms k=2, 3, 4 into `dst`.
///
/// On entry `omega_v` holds `Ω₄^1 · src`. For k=2,3,4 applies `Ω₄`
/// to build `Ω₄^k · src` and adds `Ω₄^k·src / k!` to `dst`.
/// Operation order is IDENTICAL to the original unrolled code.
#[allow(clippy::too_many_arguments)]
pub(crate) fn accumulate_taylor_terms<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    tau: F,
    one: F,
    two: F,
    six: F,
    twenty_four: F,
    dst: &mut GraphSignal<F>,
    omega_v: &mut [F],
    omega_pow: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
    tmp_c: &mut [F],
) {
    // k=2: omega_pow ← omega_v (= Ω₄¹·src); omega_v ← Ω₄²·src
    omega_pow.copy_from_slice(omega_v);
    apply_omega4(
        lap1, lap2, tau, one, omega_pow, omega_v, tmp_a, tmp_b, tmp_c,
    );
    dst.axpy_into_slice(one / two, omega_v);

    // k=3: omega_pow ← omega_v (= Ω₄²·src); omega_v ← Ω₄³·src
    omega_pow.copy_from_slice(omega_v);
    apply_omega4(
        lap1, lap2, tau, one, omega_pow, omega_v, tmp_a, tmp_b, tmp_c,
    );
    dst.axpy_into_slice(one / six, omega_v);

    // k=4: omega_pow ← omega_v (= Ω₄³·src); omega_v ← Ω₄⁴·src
    omega_pow.copy_from_slice(omega_v);
    apply_omega4(
        lap1, lap2, tau, one, omega_pow, omega_v, tmp_a, tmp_b, tmp_c,
    );
    dst.axpy_into_slice(one / twenty_four, omega_v);
}

// ---------------------------------------------------------------------------
// Ω₄·v helper
// ---------------------------------------------------------------------------

/// Apply `Ω₄ · v → out` (forward) or `Ω₄ᵀ · v → out` (adjoint / sign-flipped).
///
/// ```text
/// Ω₄·v  = −(τ/2)·(L₁+L₂)·v + comm_sign·(√3·τ²/12)·[A₂,A₁]·v
/// ```
///
/// - Forward (`comm_sign = +1`): `Ω₄·v = (τ/2)(A₁+A₂)v + (√3τ²/12)[A₂,A₁]v`
/// - Adjoint (`comm_sign = −1`): sign-flip on commutator term
///   (math.md §42, Theorem 42.1; `[A₂,A₁]ᵀ = −[A₂,A₁]` for symmetric `L`).
///
/// # Scratch buffer protocol (no aliasing)
///
/// 1. `tmp_a ← L₁·v`
/// 2. `tmp_b ← L₂·v`
/// 3. `tmp_c ← L₂·(L₁·v)` (commutator first product)
/// 4. `out   ← −(τ/2)·(tmp_a + tmp_b)` (leading term)
/// 5. `tmp_a ← L₁·(L₂·v)` (commutator second product)
/// 6. `out   += comm_sign·(√3·τ²/12)·(tmp_c − tmp_a)`
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_omega4<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    tau: F,
    comm_sign: F,
    v: &[F],
    out: &mut [F],
    tmp_a: &mut [F],
    tmp_b: &mut [F],
    tmp_c: &mut [F],
) {
    let n = v.len();
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(tmp_a.len(), n);
    debug_assert_eq!(tmp_b.len(), n);
    debug_assert_eq!(tmp_c.len(), n);

    let half = from_f64::<F>(0.5);
    let sqrt3_over_12 = from_f64::<F>(SQRT3_OVER_12_F64);

    // Steps 1-2: L₁·v and L₂·v
    lap1.apply_into_slice(v, tmp_a);
    lap2.apply_into_slice(v, tmp_b);

    // Step 3: L₂·(L₁·v)
    lap2.apply_into_slice(tmp_a, tmp_c);

    // Step 4: leading term out = −(τ/2)·(L₁v + L₂v) = (τ/2)·(A₁v + A₂v)
    let scale = -half * tau;
    for i in 0..n {
        out[i] = scale * (tmp_a[i] + tmp_b[i]);
    }

    // Step 5: L₁·(L₂·v); tmp_b is no longer needed after this.
    lap1.apply_into_slice(tmp_b, tmp_a);

    // Step 6: commutator term; comm_sign=+1 forward, -1 adjoint.
    let comm_scale = comm_sign * sqrt3_over_12 * tau * tau;
    for i in 0..n {
        out[i] += comm_scale * (tmp_c[i] - tmp_a[i]);
    }
}
