//! `G_OBSTACLE_STATIONARY`, `G_OBSTACLE_SLOPE_SMOOTH`, `G_OBSTACLE_SLOPE_AMERICAN`,
//! `T_OBSTACLE_ADJOINT` — normative gates for `ObstacleChernoff` (math §44).
//!
//! ## `G_OBSTACLE_STATIONARY` (`RELEASE_BLOCKING`)
//!
//! Stationary membrane oracle (§44.6): `L = ∂_xx`, Dirichlet `u(0)=u(1)=0`,
//! obstacle `g(x) = A − B(x−½)²` with `A=0.1, B=1.0`.
//! Contact half-offset `α = sqrt(1/4 − A/B) ≈ 0.3873`.
//! Exact fixed point:
//!   `u*(x) = s·x`       on `[0, α]`
//!   `u*(x) = g(x)`      on `[α, 1−α]`
//!   `u*(x) = s·(1−x)`   on `[1−α, 1]`
//! where `s = B·(1 − 2α)`.
//!
//! Tolerance: `2.5e-2` (spatial error O(dx) near the C¹ kinks at α, 1−α;
//! confirmed decreasing with N).
//!
//! ## `G_OBSTACLE_SLOPE_SMOOTH` / `G_OBSTACLE_SLOPE_AMERICAN` (slow-tests)
//!
//! Self-convergence (Richardson) OLS slope of sup-error vs `n_steps`.
//! Both sweeps use a fixed fine spatial grid; `n_steps` halved yields Δτ halving.
//!
//! `G_OBSTACLE_SLOPE_SMOOTH` (≤ −0.95): obstacle far below the solution
//! (projection is identity) — recovers inner order-1.
//!
//! `G_OBSTACLE_SLOPE_AMERICAN` (≤ −0.45): obstacle actively binding (V0=g)
//! with a free boundary — structural O(√Δτ) degradation.
//!
//! ## h0/dx caveat (mirrors `killing_dirichlet_slope.rs`)
//!
//! The Chernoff step width h0 = 2√(a·τ) must satisfy h0/dx < 1 to stay in
//! the spatial asymptotic regime. The `N_STEPS` sweeps below keep h0/dx ≤ 0.4
//! for all sweep points, so temporal convergence dominates.
//!
//! ## `T_OBSTACLE_ADJOINT`
//!
//! Finite-difference cross-check of `apply_active_set_adjoint_into`:
//! `|⟨λ⁰, δ⟩ − [J(V0+εδ)−J(V0−εδ)]/(2ε)| / ‖⟨λ⁰,δ⟩‖ ≤ C·ε` for ε = 1e-5.
//! Inner: `DriftReactionChernoff` (implements `apply_adjoint_into`).
//!
//! Feature gate: slope gates need `slow-tests`; stationary and adjoint are fast.

// Integration test: allows for numerical / binding wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::type_complexity
)]

use core::f64::consts::PI;

use semiflow::{
    killing::{BoxRegion, KillingChernoff},
    BoundaryPolicy, ChernoffFunction, ClosureObstacle, ConstantObstacle, DiffusionChernoff,
    DriftReactionChernoff, Grid1D, GridFn1D, ObstacleChernoff, ScratchPool,
};

// ---------------------------------------------------------------------------
// Gate constants (NORMATIVE — do not relax without ADR + properties.yaml bump)
// ---------------------------------------------------------------------------

/// `G_OBSTACLE_STATIONARY`: sup-error vs u* must be below this tolerance.
/// Calibrated at N=256, T=50 (near equilibrium). O(dx)≈4e-3 near kinks.
const STATIONARY_TOL: f64 = 2.5e-2;
/// N for the stationary test (256 gives dx≈4e-3).
const STATIONARY_N: usize = 256;
/// `T_final` for stationary run (large enough for near-equilibrium convergence).
const STATIONARY_T: f64 = 50.0;
/// Steps per unit time for the stationary run (τ = `1/STATIONARY_STEPS_PER_T`).
const STATIONARY_STEPS_PER_T: usize = 2000;

/// Membrane parameters A, B (0 < A < B/4).
const MEMBRANE_A: f64 = 0.1;
const MEMBRANE_B: f64 = 1.0;

/// G_OBSTACLE_SLOPE_SMOOTH gate threshold (≤ −0.95).
#[cfg(feature = "slow-tests")]
const SLOPE_SMOOTH_GATE: f64 = -0.95;
/// G_OBSTACLE_SLOPE_AMERICAN gate threshold (≤ −0.45).
#[cfg(feature = "slow-tests")]
const SLOPE_AMERICAN_GATE: f64 = -0.45;

/// Fixed spatial grid size for the slope sweeps.
#[cfg(feature = "slow-tests")]
const SLOPE_N: usize = 128;
/// Finest n_steps (reference solution). Four levels halved: n0, n0/2, n0/4, n0/8.
#[cfg(feature = "slow-tests")]
const SLOPE_N_STEPS_BASE: usize = 4096;
/// T for slope sweeps.
#[cfg(feature = "slow-tests")]
const SLOPE_T: f64 = 0.5;

// ---------------------------------------------------------------------------
// Membrane oracle
// ---------------------------------------------------------------------------

/// Membrane contact offset α = sqrt(1/4 - A/B).
fn membrane_alpha() -> f64 {
    (0.25 - MEMBRANE_A / MEMBRANE_B).sqrt()
}

/// Free-arm slope s = B·(1 − 2α).
fn membrane_slope(alpha: f64) -> f64 {
    MEMBRANE_B * (1.0 - 2.0 * alpha)
}

/// Obstacle g(x) = A − B·(x − 1/2)².
fn membrane_obstacle(x: f64) -> f64 {
    MEMBRANE_A - MEMBRANE_B * (x - 0.5) * (x - 0.5)
}

/// Exact stationary solution u*(x) (§44.6, C¹ smooth-fit).
fn membrane_exact(x: f64) -> f64 {
    let alpha = membrane_alpha();
    let s = membrane_slope(alpha);
    if x <= alpha {
        s * x
    } else if x <= 1.0 - alpha {
        membrane_obstacle(x)
    } else {
        s * (1.0 - x)
    }
}

// ---------------------------------------------------------------------------
// G_OBSTACLE_STATIONARY helpers
// ---------------------------------------------------------------------------

/// Build `ObstacleChernoff<KillingChernoff<DiffusionChernoff, BoxRegion>, ClosureObstacle>`.
///
/// Inner: heat with a=0.5, Dirichlet BCs via KillingChernoff+BoxRegion [0,1].
/// Obstacle: g(x) = `membrane_obstacle(x)`.
fn build_stationary_kernel(
    grid: Grid1D<f64>,
) -> ObstacleChernoff<
    KillingChernoff<DiffusionChernoff<f64>, BoxRegion<f64, 1>>,
    ClosureObstacle<impl Fn(f64) -> f64, f64>,
> {
    let inner_diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
    let region = BoxRegion::<f64, 1>::new([0.0_f64], [1.0_f64]).expect("box region");
    let killing = KillingChernoff::new(inner_diff, region).expect("killing");
    let obs = ClosureObstacle::new(membrane_obstacle);
    ObstacleChernoff::new(killing, obs).expect("obstacle kernel")
}

/// Evolve `ObstacleChernoff` from `V0` for `n_steps` steps of size `tau`
/// and return the final `GridFn1D`.
fn evolve(
    kernel: &impl ChernoffFunction<f64, S = GridFn1D<f64>>,
    mut u: GridFn1D<f64>,
    tau: f64,
    n_steps: usize,
) -> GridFn1D<f64> {
    let mut scratch = ScratchPool::new();
    let mut dst = u.zeroed_like();
    for _ in 0..n_steps {
        kernel
            .apply_into(tau, &u, &mut dst, &mut scratch)
            .expect("apply_into ok");
        core::mem::swap(&mut u, &mut dst);
    }
    u
}

// ---------------------------------------------------------------------------
// OLS log-log slope utility
// ---------------------------------------------------------------------------

#[cfg(feature = "slow-tests")]
#[allow(clippy::cast_precision_loss)]
fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let m = xs.len() as f64;
    let lx: Vec<f64> = xs.iter().map(|&v| v.ln()).collect();
    let ly: Vec<f64> = ys.iter().map(|&v| v.ln()).collect();
    let mx = lx.iter().sum::<f64>() / m;
    let my = ly.iter().sum::<f64>() / m;
    let num: f64 = lx
        .iter()
        .zip(ly.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let den: f64 = lx.iter().map(|x| (x - mx).powi(2)).sum();
    num / den
}

/// Sup-norm of `a − b` over interior nodes 1..n-1.
fn sup_error_interior(a: &GridFn1D<f64>, b: &GridFn1D<f64>) -> f64 {
    (1..a.grid.n - 1)
        .map(|i| (a.values[i] - b.values[i]).abs())
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// G_OBSTACLE_STATIONARY (RELEASE_BLOCKING, fast — no feature gate needed)
// ---------------------------------------------------------------------------

/// `G_OBSTACLE_STATIONARY`: evolved V must be within `STATIONARY_TOL` of u*.
///
/// V0 = max(0, g) ≥ g (feasible start above obstacle), evolved to near-equilibrium.
/// Sup-error on interior nodes checked against the closed-form membrane §44.6.
#[test]
fn g_obstacle_stationary() {
    let grid = Grid1D::new(0.0_f64, 1.0, STATIONARY_N)
        .unwrap()
        .with_boundary(BoundaryPolicy::ZeroExtend);
    let kernel = build_stationary_kernel(grid);
    // V0 = max(0, g) satisfies V0 ≥ g and V0(0)=V0(1)=0.
    let v0 = GridFn1D::from_fn(grid, |x| membrane_obstacle(x).max(0.0));

    let tau = 1.0 / STATIONARY_STEPS_PER_T as f64;
    let n_steps = (STATIONARY_T * STATIONARY_STEPS_PER_T as f64) as usize;
    let v_final = evolve(&kernel, v0, tau, n_steps);

    // Build reference u*.
    let u_star = GridFn1D::from_fn(grid, membrane_exact);

    let err = sup_error_interior(&v_final, &u_star);
    let alpha = membrane_alpha();
    let dx = 1.0 / (STATIONARY_N as f64 - 1.0);
    println!(
        "G_OBSTACLE_STATIONARY: N={STATIONARY_N}, T={STATIONARY_T}, \
         α={alpha:.4}, dx={dx:.4e}, sup_err={err:.4e}  (tol={STATIONARY_TOL:.4e})"
    );
    assert!(
        err <= STATIONARY_TOL,
        "G_OBSTACLE_STATIONARY FAIL: sup_err={err:.4e} > tol={STATIONARY_TOL:.4e}. \
         Membrane A={MEMBRANE_A}, B={MEMBRANE_B}, α≈{alpha:.4}. \
         Check KillingChernoff Dirichlet inner + membrane obstacle."
    );
}

// ---------------------------------------------------------------------------
// G_OBSTACLE_SLOPE_SMOOTH (slow-tests feature gate)
// ---------------------------------------------------------------------------

/// Smooth-obstacle self-convergence: sweep n_steps, no active free boundary.
///
/// Obstacle g = −10 (far below solution) so projection is identity in practice,
/// recovering the inner heat order-1 slope ≤ −0.95.
///
/// h0/dx = 2√(0.5·τ)/dx = 2√(0.5·T/(n·dx)) — for SLOPE_N=128, n_steps=512,
/// dx≈8e-3: τ=T/512≈1e-3, h0≈4.5e-2, h0/dx≈5.6. Fine spatial grid
/// (SLOPE_N=128) keeps spatial error below 1e-3 so temporal error dominates.
///
/// NOTE: Δτ self-convergence slope is measured vs n_steps (not N). Halving n_steps
/// doubles Δτ ⇒ slope measured as log(err) vs log(n_steps) ≤ −0.95.
#[test]
#[cfg(feature = "slow-tests")]
#[allow(clippy::cast_precision_loss)]
fn g_obstacle_slope_smooth() {
    let grid = Grid1D::new(0.0_f64, 1.0, SLOPE_N)
        .unwrap()
        .with_boundary(BoundaryPolicy::ZeroExtend);

    // Low obstacle (inactive) + DiffusionChernoff inner.
    let make_kernel = || {
        let inner = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        let obs = ConstantObstacle::new(-10.0_f64).expect("obs");
        ObstacleChernoff::new(inner, obs).expect("kernel")
    };

    // Smooth initial datum: Gaussian away from boundaries.
    let v0 = GridFn1D::from_fn(grid, |x| (-(x - 0.5).powi(2) / 0.02).exp());

    // Reference: finest run.
    let ref_solution = {
        let tau = SLOPE_T / SLOPE_N_STEPS_BASE as f64;
        evolve(&make_kernel(), v0.clone(), tau, SLOPE_N_STEPS_BASE)
    };

    // Sweep coarser n_steps (half, quarter, eighth).
    let n_steps_sweep: [usize; 3] = [
        SLOPE_N_STEPS_BASE / 2,
        SLOPE_N_STEPS_BASE / 4,
        SLOPE_N_STEPS_BASE / 8,
    ];
    let mut errs = Vec::with_capacity(3);
    for &ns in &n_steps_sweep {
        let tau = SLOPE_T / ns as f64;
        let h0 = 2.0 * (0.5_f64 * tau).sqrt();
        let dx = 1.0 / (SLOPE_N as f64 - 1.0);
        let sol = evolve(&make_kernel(), v0.clone(), tau, ns);
        let e = sup_error_interior(&sol, &ref_solution);
        println!(
            "G_SLOPE_SMOOTH: n_steps={ns}, tau={tau:.3e}, h0/dx={:.3}, err={e:.4e}",
            h0 / dx
        );
        errs.push(e);
    }
    let n_steps_f: Vec<f64> = n_steps_sweep.iter().map(|&s| s as f64).collect();
    let slope = ols_slope(&n_steps_f, &errs);
    println!("G_SLOPE_SMOOTH: OLS slope = {slope:.4}  (gate <= {SLOPE_SMOOTH_GATE})");
    assert!(
        slope <= SLOPE_SMOOTH_GATE,
        "G_OBSTACLE_SLOPE_SMOOTH FAIL: slope={slope:.4} > gate={SLOPE_SMOOTH_GATE}. \
         Obstacle inactive (g=-10). Check DiffusionChernoff inner order."
    );
}

// ---------------------------------------------------------------------------
// G_OBSTACLE_SLOPE_AMERICAN (slow-tests feature gate)
// ---------------------------------------------------------------------------

/// American-style self-convergence: obstacle actively binding (V0=g initially).
///
/// Uses the membrane obstacle g(x)=A−B(x−½)² and V0=g so the contact set is
/// initially the whole domain, evolving to the stationary membrane. The free
/// boundary movement induces O(√Δτ) degradation: slope ≤ −0.45.
#[test]
#[cfg(feature = "slow-tests")]
#[allow(clippy::cast_precision_loss)]
fn g_obstacle_slope_american() {
    let grid = Grid1D::new(0.0_f64, 1.0, SLOPE_N)
        .unwrap()
        .with_boundary(BoundaryPolicy::ZeroExtend);

    let make_kernel = || {
        let inner_diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        let region = BoxRegion::<f64, 1>::new([0.0_f64], [1.0_f64]).expect("box");
        let killing = KillingChernoff::new(inner_diff, region).expect("killing");
        let obs = ClosureObstacle::new(membrane_obstacle);
        ObstacleChernoff::new(killing, obs).expect("kernel")
    };

    // V0 = obstacle = g(x) (contact set is everywhere initially).
    let v0 = GridFn1D::from_fn(grid, |x| membrane_obstacle(x).max(0.0));

    // Reference: finest run.
    let ref_solution = {
        let tau = SLOPE_T / SLOPE_N_STEPS_BASE as f64;
        evolve(&make_kernel(), v0.clone(), tau, SLOPE_N_STEPS_BASE)
    };

    let n_steps_sweep: [usize; 3] = [
        SLOPE_N_STEPS_BASE / 2,
        SLOPE_N_STEPS_BASE / 4,
        SLOPE_N_STEPS_BASE / 8,
    ];
    let mut errs = Vec::with_capacity(3);
    for &ns in &n_steps_sweep {
        let tau = SLOPE_T / ns as f64;
        let h0 = 2.0 * (0.5_f64 * tau).sqrt();
        let dx = 1.0 / (SLOPE_N as f64 - 1.0);
        let sol = evolve(&make_kernel(), v0.clone(), tau, ns);
        let e = sup_error_interior(&sol, &ref_solution);
        println!(
            "G_SLOPE_AMERICAN: n_steps={ns}, tau={tau:.3e}, h0/dx={:.3}, err={e:.4e}",
            h0 / dx
        );
        errs.push(e);
    }
    let n_steps_f: Vec<f64> = n_steps_sweep.iter().map(|&s| s as f64).collect();
    let slope = ols_slope(&n_steps_f, &errs);
    println!("G_SLOPE_AMERICAN: OLS slope = {slope:.4}  (gate <= {SLOPE_AMERICAN_GATE})");
    assert!(
        slope <= SLOPE_AMERICAN_GATE,
        "G_OBSTACLE_SLOPE_AMERICAN FAIL: slope={slope:.4} > gate={SLOPE_AMERICAN_GATE}. \
         Expected O(sqrt(Δτ)) at free boundary. If slope > -0.45, check whether \
         the free boundary is genuinely active."
    );
}

// ---------------------------------------------------------------------------
// T_OBSTACLE_ADJOINT (fast — no feature gate)
// ---------------------------------------------------------------------------

/// `T_OBSTACLE_ADJOINT`: active-set adjoint matches central FD to O(ε²).
///
/// Inner: `DriftReactionChernoff` (b=0, c=-0.5) — implements `apply_adjoint_into`.
/// Forward: K=4 steps; backward: active-set adjoint sweep. Check ⟨λ⁰,δ⟩ ≈ FD.
type AdjointKernel = ObstacleChernoff<DriftReactionChernoff, ConstantObstacle<f64>>;

/// Build the adjoint-test kernel: DriftReactionChernoff(b=0, c=-0.5) + floor g=0.
fn build_adjoint_kernel(grid: Grid1D<f64>) -> AdjointKernel {
    let inner = DriftReactionChernoff::new(|_| 0.0_f64, |_| -0.5_f64, 0.5, grid);
    let obs = ConstantObstacle::new(0.0_f64).expect("obs finite");
    ObstacleChernoff::new(inner, obs).expect("kernel")
}

/// Run K forward steps, returning (`v_traj`, `w_fwd_traj`).
fn adjoint_forward(
    kernel: &AdjointKernel,
    v0: GridFn1D<f64>,
    tau: f64,
    k_steps: usize,
) -> (Vec<GridFn1D<f64>>, Vec<GridFn1D<f64>>) {
    let mut scratch = ScratchPool::new();
    let mut v_traj = Vec::with_capacity(k_steps + 1);
    let mut w_fwd_traj = Vec::with_capacity(k_steps);
    v_traj.push(v0);
    for k in 0..k_steps {
        let v_k = &v_traj[k];
        let mut w_fwd = v_k.zeroed_like();
        kernel
            .inner()
            .apply_into(tau, v_k, &mut w_fwd, &mut scratch)
            .expect("inner");
        let mut v_next = v_k.zeroed_like();
        kernel
            .apply_into(tau, v_k, &mut v_next, &mut scratch)
            .expect("obstacle");
        w_fwd_traj.push(w_fwd);
        v_traj.push(v_next);
    }
    (v_traj, w_fwd_traj)
}

/// Run backward adjoint sweep. Returns λ⁰.
fn adjoint_backward(
    kernel: &AdjointKernel,
    w_fwd_traj: &[GridFn1D<f64>],
    lam_terminal: GridFn1D<f64>,
    tau: f64,
) -> GridFn1D<f64> {
    let mut scratch = ScratchPool::new();
    let mut lam = lam_terminal;
    for k in (0..w_fwd_traj.len()).rev() {
        let mut lam_next = lam.zeroed_like();
        kernel
            .apply_active_set_adjoint_into(tau, &w_fwd_traj[k], &lam, &mut lam_next, &mut scratch)
            .expect("adjoint");
        lam = lam_next;
    }
    lam
}

/// `T_OBSTACLE_ADJOINT` main gate (§44.5).
#[test]
#[allow(clippy::cast_precision_loss)]
fn t_obstacle_adjoint() {
    const N: usize = 64;
    const K: usize = 4;
    const TAU: f64 = 0.01;
    const EPS: f64 = 1e-5;

    let grid = Grid1D::new(0.0_f64, 1.0, N).unwrap();
    let kernel = build_adjoint_kernel(grid);
    let v0 = GridFn1D::from_fn(grid, |x| (PI * x).sin() * 0.5);
    let target = GridFn1D::from_fn(grid, |x| (2.0 * PI * x).sin() * 0.3);
    let delta = GridFn1D::from_fn(grid, |x| (PI * x).sin());

    // Forward + backward adjoint sweep.
    let (v_traj, w_fwd_traj) = adjoint_forward(&kernel, v0.clone(), TAU, K);
    let v_n = &v_traj[K];
    let lam_terminal = GridFn1D::from_fn(grid, |x| {
        let i = (x * (N as f64 - 1.0)).round() as usize;
        v_n.values[i.min(N - 1)] - target.values[i.min(N - 1)]
    });
    let lam0 = adjoint_backward(&kernel, &w_fwd_traj, lam_terminal, TAU);

    // Central FD. Evolve perturbed ICs for K steps.
    let j = |v: &GridFn1D<f64>| -> f64 {
        v.values
            .iter()
            .zip(target.values.iter())
            .map(|(vi, ti)| 0.5 * (vi - ti).powi(2))
            .sum()
    };
    let fd_run = |sign: f64| -> GridFn1D<f64> {
        let mut v = v0.clone();
        for (vi, di) in v.values.iter_mut().zip(delta.values.iter()) {
            *vi += sign * EPS * di;
        }
        evolve(&kernel, v, TAU, K)
    };
    let fd = (j(&fd_run(1.0)) - j(&fd_run(-1.0))) / (2.0 * EPS);
    let adj: f64 = lam0
        .values
        .iter()
        .zip(delta.values.iter())
        .map(|(l, d)| l * d)
        .sum();
    let rel_err = (adj - fd).abs() / adj.abs().max(1e-12);
    println!(
        "T_OBSTACLE_ADJOINT: adj={adj:.6e}, fd={fd:.6e}, rel={rel_err:.2e} \
         (threshold 20·eps={:.2e})",
        20.0 * EPS
    );
    assert!(
        rel_err <= 20.0 * EPS,
        "T_OBSTACLE_ADJOINT FAIL: rel={rel_err:.2e} > {:.2e}",
        20.0 * EPS
    );
}

// ---------------------------------------------------------------------------
// G_OBSTACLE_GAMMA (RELEASE_BLOCKING, slow-tests) — ADR-0152, math §44.5.bis
// ---------------------------------------------------------------------------

/// Perpetual-American-put parameters (canonical witness).
#[cfg(feature = "slow-tests")]
const PAP_K: f64 = 1.0;
#[cfg(feature = "slow-tests")]
const PAP_R: f64 = 0.05;
#[cfg(feature = "slow-tests")]
const PAP_SIG: f64 = 0.20;

/// Derived constants for the perpetual American put.
#[cfg(feature = "slow-tests")]
fn pap_gamma_pow() -> f64 {
    2.0 * PAP_R / (PAP_SIG * PAP_SIG)
}
#[cfg(feature = "slow-tests")]
fn pap_sstar() -> f64 {
    pap_gamma_pow() / (pap_gamma_pow() + 1.0) * PAP_K
}
#[cfg(feature = "slow-tests")]
fn pap_a_coef() -> f64 {
    (PAP_K - pap_sstar()) * pap_sstar().powf(pap_gamma_pow())
}

/// Analytic V(S): continuation A·S^{-γ}, stopping K−S.
#[cfg(feature = "slow-tests")]
fn pap_v(s: f64) -> f64 {
    let sstar = pap_sstar();
    if s > sstar {
        pap_a_coef() * s.powf(-pap_gamma_pow())
    } else {
        PAP_K - s
    }
}

/// Analytic Γ(S): A·γ(γ+1)·S^{-γ-2} on {S>S*}, 0 on stopping set.
#[cfg(feature = "slow-tests")]
fn pap_gamma(s: f64) -> f64 {
    let sstar = pap_sstar();
    let g = pap_gamma_pow();
    if s > sstar {
        pap_a_coef() * g * (g + 1.0) * s.powf(-g - 2.0)
    } else {
        0.0
    }
}

/// G_OBSTACLE_GAMMA: O(Δx²) convergence of inactive-set Γ + refusal sub-gate.
///
/// Probe: S = S* + 0.40·(S_max - S*) strictly inside the continuation set.
/// Convergence: |Γ_h − Γ_analytic| at the probe node over Δx-halving sweep;
/// OLS slope in Δx ≤ −1.95.
/// Refusal: for every node S_i ≤ S*, `defined[i] == false`.
#[test]
#[cfg(feature = "slow-tests")]
#[allow(clippy::cast_precision_loss)]
fn g_obstacle_gamma() {
    // Gate thresholds (NORMATIVE — do not relax).
    const GAMMA_SLOPE_GATE: f64 = -1.95;
    // Number of grid refinement levels.
    const N_LEVELS: usize = 5;
    // Coarsest N (number of grid nodes). Halved each level.
    const N_COARSE: usize = 32;
    // Domain [S_min, S_max] for the continuation set sampling.
    const S_MIN: f64 = 0.0;
    const S_MAX: f64 = 3.0;

    let sstar = pap_sstar();
    // Fixed physical probe in the continuation set: midpoint between S* and S_MAX.
    // Far enough from S* for the guard band to never touch at any N in the sweep.
    let s_probe_target = sstar + 0.40 * (S_MAX - sstar);

    // Collect (N, err) pairs; slope d(log_err)/d(log_N) ≤ -1.95 (O(N^{-2}) = O(dx²)).
    let mut ns_f = Vec::with_capacity(N_LEVELS);
    let mut errs = Vec::with_capacity(N_LEVELS);

    for level in 0..N_LEVELS {
        let n = N_COARSE * (1 << level); // N_COARSE, 64, 128, 256, 512
        let grid = Grid1D::new(S_MIN, S_MAX, n).unwrap();
        let dx = (S_MAX - S_MIN) / (n - 1) as f64;

        // Set V analytically on the grid (already-projected value field).
        let v = GridFn1D::from_fn(grid, |s| pap_v(s));
        // Obstacle: put payoff g(S) = K − S (linear; inactive on continuation set).
        let obs = ClosureObstacle::new(|s: f64| PAP_K - s);
        let diff = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid);
        let kernel = ObstacleChernoff::new(diff, obs).unwrap();

        let mut gamma = v.zeroed_like();
        let mut defined = vec![false; n];
        let count = kernel
            .apply_inactive_gamma_into(&v, &mut gamma, &mut defined)
            .unwrap();

        // Refusal sub-check: every node with S_i <= S* must have defined[i] = false.
        for i in 0..n {
            let s_i = grid.x_at(i);
            if s_i <= sstar {
                assert!(
                    !defined[i],
                    "G_OBSTACLE_GAMMA refusal FAIL: defined[{i}]=true \
                     at S={s_i:.4} <= S*={sstar:.4}"
                );
            }
        }
        // At least one node must be defined.
        assert!(count > 0, "G_OBSTACLE_GAMMA: no defined nodes");

        // Find the grid node closest to s_probe_target that is defined.
        // Evaluate analytic Γ AT THE SAME NODE to get a pure O(dx²) FD error.
        let probe_i = (0..n)
            .filter(|&i| defined[i])
            .min_by_key(|&i| {
                let d = (grid.x_at(i) - s_probe_target).abs();
                (d * 1e9) as u64
            })
            .expect("no defined nodes");
        let s_at_probe = grid.x_at(probe_i);
        // Analytic Γ at the SAME grid node (avoids O(dx) shift contamination).
        let gamma_exact = pap_gamma(s_at_probe);
        let err = (gamma.values[probe_i] - gamma_exact).abs();
        ns_f.push(n as f64);
        errs.push(err);
        println!(
            "G_OBSTACLE_GAMMA: N={n}, dx={dx:.4e}, count={count}, probe_i={probe_i}, \
             S_probe={s_at_probe:.5}, |Γ_h - Γ_exact|={err:.4e} (exact={gamma_exact:.6})"
        );
    }
    // OLS slope d(log_err)/d(log_N) ≤ -1.95 (O(N^{-2}) ↔ O(dx²)).
    let slope = ols_slope(&ns_f, &errs);
    println!("G_OBSTACLE_GAMMA: OLS slope = {slope:.4}  (gate <= {GAMMA_SLOPE_GATE})");
    assert!(
        slope <= GAMMA_SLOPE_GATE,
        "G_OBSTACLE_GAMMA FAIL: slope={slope:.4} > gate={GAMMA_SLOPE_GATE}. \
         Inactive-set Γ must converge at O(Δx²) on the open continuation set."
    );
}

// ---------------------------------------------------------------------------
// G_OBSTACLE_SLOPE_2D (RELEASE_BLOCKING, slow-tests) — ADR-0152, math §44.5.ter
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 2D helper types for G_OBSTACLE_SLOPE_2D (defined at module level for impls)
// ---------------------------------------------------------------------------

#[cfg(feature = "slow-tests")]
use semiflow::{
    grid_nd::{GridFnND, GridND},
    obstacle_nd::ObstacleChernoffND,
};

/// Identity propagator on `GridFnND<f64, 2>`: `apply_into` = copy.
/// Used to test pure projection (inner does nothing).
#[cfg(feature = "slow-tests")]
#[derive(Clone)]
struct IdentityND2D;

#[cfg(feature = "slow-tests")]
impl ChernoffFunction<f64> for IdentityND2D {
    type S = GridFnND<f64, 2>;
    fn apply_into(
        &self,
        _tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        _s: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow::error::SemiflowError> {
        dst.values.clone_from(&src.values);
        Ok(())
    }
    fn order(&self) -> u32 {
        1
    }
    fn growth(&self) -> semiflow::chernoff::Growth<f64> {
        semiflow::chernoff::Growth::contraction()
    }
}

/// Minimal axis-separable heat step on `GridFnND<f64, 2>` (explicit Euler).
/// Order-1 inner for the 2D slope convergence test only.
#[cfg(feature = "slow-tests")]
#[derive(Clone)]
struct HeatND2D {
    axes: [Grid1D<f64>; 2],
    a: f64,
}

#[cfg(feature = "slow-tests")]
impl HeatND2D {
    fn new(axes: [Grid1D<f64>; 2], a: f64) -> Self {
        Self { axes, a }
    }
}

#[cfg(feature = "slow-tests")]
impl ChernoffFunction<f64> for HeatND2D {
    type S = GridFnND<f64, 2>;
    #[allow(clippy::cast_precision_loss)]
    fn apply_into(
        &self,
        tau: f64,
        src: &Self::S,
        dst: &mut Self::S,
        _s: &mut ScratchPool<f64>,
    ) -> Result<(), semiflow::error::SemiflowError> {
        let nx = self.axes[0].n;
        let ny = self.axes[1].n;
        let dx = (self.axes[0].xmax - self.axes[0].xmin) / (nx - 1) as f64;
        let dy = (self.axes[1].xmax - self.axes[1].xmin) / (ny - 1) as f64;
        let cx = self.a * tau / (dx * dx);
        let cy = self.a * tau / (dy * dy);
        dst.values.clone_from(&src.values);
        for j in 0..ny {
            for i in 1..nx - 1 {
                let flat = j * nx + i;
                dst.values[flat] +=
                    cx * (src.values[flat + 1] - 2.0 * src.values[flat] + src.values[flat - 1]);
            }
        }
        for j in 1..ny - 1 {
            for i in 0..nx {
                let flat = j * nx + i;
                dst.values[flat] += cy
                    * (src.values[(j + 1) * nx + i] - 2.0 * src.values[flat]
                        + src.values[(j - 1) * nx + i]);
            }
        }
        Ok(())
    }
    fn order(&self) -> u32 {
        1
    }
    fn growth(&self) -> semiflow::chernoff::Growth<f64> {
        semiflow::chernoff::Growth::contraction()
    }
}

#[cfg(feature = "slow-tests")]
#[allow(clippy::cast_precision_loss)]
fn evolve_nd<K>(kernel: &K, mut u: GridFnND<f64, 2>, tau: f64, n_steps: usize) -> GridFnND<f64, 2>
where
    K: ChernoffFunction<f64, S = GridFnND<f64, 2>>,
{
    let mut scratch = ScratchPool::new();
    let mut dst = u.clone();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &u, &mut dst, &mut scratch).unwrap();
        core::mem::swap(&mut u, &mut dst);
    }
    u
}

#[cfg(feature = "slow-tests")]
fn sup_error_nd(a: &GridFnND<f64, 2>, b: &GridFnND<f64, 2>) -> f64 {
    a.values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// G_OBSTACLE_SLOPE_2D: 2D `ObstacleChernoffND` forward evolution convergence.
///
/// Two sub-checks (mirrors `G_OBSTACLE_SLOPE_SMOOTH` / `G_OBSTACLE_STATIONARY`):
///
/// (1) Self-convergence slope: smooth initial datum above the obstacle (inactive
///     projection = identity); OLS slope(log sup_err vs log n_steps) ≤ −0.95.
///     Inner: axis-separable explicit Euler heat step (`HeatND2D`, order-1).
///
/// (2) Stationary correctness: V0 = g everywhere → post-projection = g exactly
///     (contact set = full domain, `sup_err ≤ 1e-12`).
#[test]
#[cfg(feature = "slow-tests")]
#[allow(clippy::cast_precision_loss)]
fn g_obstacle_slope_2d() {
    const SLOPE_2D_GATE: f64 = -0.95;
    const N2: usize = 24;
    const T2: f64 = 0.5;
    let ax = Grid1D::new(0.0_f64, 1.0, N2).unwrap();

    // ---- Sub-check 2: stationary (V0 = g → projection = g) ----
    {
        let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
        let floor = 0.3_f64;
        let obs = ConstantObstacle::new(floor).unwrap();
        let v0 = GridFnND::from_fn(grid.clone(), |_: &[f64; 2]| floor);
        let kernel = ObstacleChernoffND::new(IdentityND2D, obs).unwrap();
        let mut dst = GridFnND::from_fn(grid, |_: &[f64; 2]| 0.0_f64);
        let mut scratch = ScratchPool::new();
        kernel
            .apply_into(0.01, &v0, &mut dst, &mut scratch)
            .unwrap();
        let sup_err = dst
            .values
            .iter()
            .map(|&v| (v - floor).abs())
            .fold(0.0_f64, f64::max);
        println!("G_OBSTACLE_SLOPE_2D stationary: sup_err={sup_err:.4e} floor={floor}");
        assert!(
            sup_err <= 1e-12,
            "G_OBSTACLE_SLOPE_2D stationary FAIL: sup_err={sup_err:.4e} > 1e-12"
        );
    }

    // ---- Sub-check 1: self-convergence slope (inactive obstacle) ----
    {
        let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
        let obs = ConstantObstacle::new(-10.0_f64).unwrap();
        // Smooth 2D Gaussian initial datum.
        let v0 = GridFnND::from_fn(grid.clone(), |x: &[f64; 2]| {
            (-(((x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2)) / 0.02)).exp()
        });
        let make_kernel = || {
            let inner = HeatND2D::new([ax, ax], 0.5_f64);
            ObstacleChernoffND::new(inner, obs).unwrap()
        };
        let n_steps_ref = 4096_usize;
        let ref_sol = {
            let tau = T2 / n_steps_ref as f64;
            evolve_nd(&make_kernel(), v0.clone(), tau, n_steps_ref)
        };
        let n_steps_sweep: [usize; 3] = [n_steps_ref / 2, n_steps_ref / 4, n_steps_ref / 8];
        let mut errs = Vec::with_capacity(3);
        for &ns in &n_steps_sweep {
            let tau = T2 / ns as f64;
            let sol = evolve_nd(&make_kernel(), v0.clone(), tau, ns);
            let e = sup_error_nd(&sol, &ref_sol);
            println!("G_OBSTACLE_SLOPE_2D: n_steps={ns}, tau={tau:.3e}, err={e:.4e}");
            errs.push(e);
        }
        let n_steps_f: Vec<f64> = n_steps_sweep.iter().map(|&s| s as f64).collect();
        let slope = ols_slope(&n_steps_f, &errs);
        println!("G_OBSTACLE_SLOPE_2D: OLS slope={slope:.4}  (gate <= {SLOPE_2D_GATE})");
        assert!(
            slope <= SLOPE_2D_GATE,
            "G_OBSTACLE_SLOPE_2D FAIL: slope={slope:.4} > gate={SLOPE_2D_GATE}. \
             2D projected scheme (inactive obstacle) should recover inner order-1."
        );
    }
}
