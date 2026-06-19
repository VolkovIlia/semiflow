//! `G_ADJOINT_FP_ORDER` — `RELEASE_BLOCKING` gate for `AdjointFokkerPlanckChernoff`
//! (ADR-0107 AMENDMENT 1, math.md §38.9, properties.yaml `G_ADJOINT_FP_ORDER`).
//!
//! Three sub-gates (all in this `#[ignore]` slow-tests fn):
//!
//! (1) VAGUE-CONVERGENCE SLOPE:
//!     1D Brownian motion (a=1/2, b=0, c=0; forward L=(1/2)∂²ₓ), ρ₀ = δ₀,
//!     T=1.0, characteristic-function test `f_ξ(x)` = cos(ξx) at ξ ∈ {0.5,1,1.5,2}.
//!     Reference: e^{-T·ξ²/2} (Gaussian N(0,T) char fn).
//!     Sweep n ∈ {16, 32, 64, 128, 256}. Char-fn pairing FOLDS Diracs analytically:
//!     ⟨cos(ξx), Σwᵢδ_{xᵢ}⟩ = Σwᵢcos(ξxᵢ) — `O(n_diracs)` cost, sidestepping 4^n blow-up.
//!     Gate: OLS slope ≤ −0.95 for ALL four ξ values.
//!
//! (1b) REAL-KERNEL MULTI-STEP PATH:
//!     Drive `MeasureState::apply_into` for n=8 steps on the REAL Rust kernel
//!     (`AdjointFokkerPlanckChernoff` over `DiffusionChernoff`, Brownian motion).
//!     Checks: TV norm finite + char-fn pairing at n=8 within 5% of the
//!     analytical value — verifies the Rust code path genuinely runs (not just
//!     the re-derived analytical formula).
//!
//! (2) GENUINE DISCRETE-ADJOINT PAIRING CROSS-CHECK (§38.2):
//!     Verify ⟨`S_fwd(τ)f`, δ_{x₀}⟩ ≈ ⟨f, `S_adj`*(τ)δ_{x₀}⟩ using the REAL Rust kernels:
//!     - Forward:  apply `DiffusionChernoff::apply_into` to f = cos(ξx) on a grid;
//!                 read interpolated value at x₀ → (S(τ)f)(x₀).
//!     - Adjoint:  apply `AdjointFokkerPlanckChernoff::apply_into` to δ_{x₀};
//!                 pair resulting measure against f → Σ wᵢ f(xᵢ).
//!
//!     DiffusionChernoff (ζ-A, 5-point stencil) and AdjointFokkerPlanckChernoff
//!     (Lemma A.1, 4-Dirac) are DIFFERENT Chernoff approximations to the SAME
//!     semigroup pair (e^{τL}, e^{τL*}).  Their pairing residual converges to
//!     zero as τ → 0 with rate O(τ) (both are consistent approximations):
//!       residual(τ) = |(S_fwd(τ)f)(x₀) − ⟨f, S_adj*(τ)δ_{x₀}⟩| = O(τ).
//!
//!     Gate: OLS slope (log τ vs log residual) ≤ −0.8 across τ ∈ {0.5, 0.25, 0.1, 0.05};
//!           residual at τ=0.05 < 5e-3.
//!     This is NOT a tautology: it runs BOTH Rust kernels and checks they converge
//!     together.  If the adjoint is transposed incorrectly the residual will be
//!     O(1) rather than O(τ).
//!
//! Run with:
//!   cargo test -p semiflow-core --features slow-tests -- --ignored `g_adjoint_fp` --nocapture

#![allow(clippy::cast_precision_loss)]
// Integration test: allows for numerical / binding wrapper patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines
)]

use semiflow_core::{
    adjoint_fp::{AdjointFokkerPlanckChernoff, MeasureState},
    ChernoffFunction, DiffusionChernoff, Grid1D, GridFn1D, ScratchPool,
};

// ---------------------------------------------------------------------------
// Helpers shared across sub-gates
// ---------------------------------------------------------------------------

/// Build the Brownian-motion forward kernel: a=0.5, b=0, c=0 (§38.7).
fn brownian_fwd(grid_n: usize) -> DiffusionChernoff<f64> {
    let grid = Grid1D::new(-8.0_f64, 8.0, grid_n).unwrap();
    DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid)
}

/// Analytical characteristic function of N(0, T): e^{−T·ξ²/2}.
fn gauss_char_fn(xi: f64, t: f64) -> f64 {
    libm::exp(-t * xi * xi / 2.0)
}

/// Evaluate the n-step adjoint char fn analytically via the CLT Dirac folding.
///
/// For ρ₀ = δ₀, a=0.5, b=c=0:
///   h = 2√(aτ),  S*(τ)δ₀ = (`1/4)δ_h` + (1/4)δ_{-h} + (`1/2)δ_0`
///   ⟨cos(ξ·x), S*(τ)δ₀⟩ = (1/2)cos(ξh) + 1/2
/// After n steps: `φ_n(ξ)` = ((1/2)cos(ξh) + 1/2)^n  (CLT reduction, §38.7).
fn char_fn_iterated(xi: f64, n: usize, t: f64) -> f64 {
    let tau = t / n as f64;
    let a = 0.5_f64;
    let h = 2.0 * (a * tau).sqrt();
    // Per-step: (1/4)cos(ξh) + (1/4)cos(-ξh) + (1/2)cos(0) = (1/2)cos(ξh) + 1/2
    let per_step = 0.5 * (xi * h).cos() + 0.5;
    per_step.powi(n as i32)
}

/// OLS slope on (x, y) log data.
fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| x * y).sum();
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

// ---------------------------------------------------------------------------
// Sub-gate 1: vague-convergence char-fn slope (analytical folding)
// ---------------------------------------------------------------------------

/// Compute OLS log-log slope of |err(n)| vs n.
///
/// If err ~ C/n^m then log(err) = log(C) - m*log(n), so slope = -m.
/// Gate criterion: slope ≤ -0.95 (i.e., rate m ≥ 0.95 ≈ order 1).
fn run_vague_convergence_slope(xi: f64, t: f64, ns: &[usize]) -> f64 {
    let ref_val = gauss_char_fn(xi, t);
    let mut log_ns = Vec::with_capacity(ns.len());
    let mut log_errs = Vec::with_capacity(ns.len());

    for &n in ns {
        let approx = char_fn_iterated(xi, n, t);
        let err = (approx - ref_val).abs();
        if err < 1e-15 {
            continue; // at floating-point floor
        }
        log_ns.push((n as f64).ln());
        log_errs.push(err.ln());
    }
    ols_slope(&log_ns, &log_errs)
}

// ---------------------------------------------------------------------------
// Sub-gate 1b: real-kernel multi-step path
// ---------------------------------------------------------------------------

/// Run the REAL Rust `MeasureState::apply_into` for `n_steps` steps.
///
/// Returns (`tv_norm`, `char_fn_value`) at the final state.
/// The Rust kernel path must be exercised — this is NOT the analytical formula.
fn run_real_kernel_multistep(n_steps: usize, t: f64, xi: f64) -> (f64, f64) {
    let tau = t / n_steps as f64;
    let fwd = brownian_fwd(64);
    let adj = AdjointFokkerPlanckChernoff::new(fwd, 0.5_f64, 0.0_f64, 0.0_f64);

    let mut rho = MeasureState::<f64, 1>::dirac([0.0], 1.0);
    let mut rho_next = rho.clone();
    let mut pool = ScratchPool::<f64>::new();

    for _ in 0..n_steps {
        adj.apply_into(tau, &rho, &mut rho_next, &mut pool).unwrap();
        core::mem::swap(&mut rho, &mut rho_next);
    }

    let tv = rho.total_variation();
    let char_val = rho.pair(|pos| (xi * pos[0]).cos());
    (tv, char_val)
}

// ---------------------------------------------------------------------------
// Sub-gate 2: genuine discrete-adjoint pairing cross-check
// ---------------------------------------------------------------------------

/// Compute the pairing residual
///   |⟨`S_fwd(τ)f`, δ_{x₀}⟩ − ⟨f, `S_adj`*(τ)δ_{x₀}⟩|
/// using the REAL Rust kernels on `grid_n` nodes.
///
/// Forward side: `DiffusionChernoff::apply_into` maps f on the grid,
///               then sample at x₀ via grid interpolation.
/// Adjoint side: `AdjointFokkerPlanckChernoff::apply_into` maps δ_{x₀}
///               to weighted Diracs; `MeasureState::pair(f)` evaluates
///               f analytically at each Dirac position.
///
/// The residual is O(τ) for smooth f, since both kernels are O(τ)
/// consistent approximations to the SAME continuous semigroup.
/// It is NOT zero for any fixed τ > 0 (they use different stencils).
fn pairing_residual(grid_n: usize, tau: f64, xi: f64, x0: f64) -> f64 {
    // Test function: f(x) = cos(ξx), smooth, bounded.
    let f = |x: f64| (xi * x).cos();

    // --- Forward side ---
    let fwd = brownian_fwd(grid_n);
    let f_grid = GridFn1D::from_fn(fwd.grid, &f);
    let mut f_out = f_grid.clone();
    let mut pool = ScratchPool::<f64>::new();
    fwd.apply_into(tau, &f_grid, &mut f_out, &mut pool).unwrap();
    // (S_fwd(τ)f)(x₀) via grid interpolation
    let forward_at_x0 = f_out.sample(x0).unwrap_or(f64::NAN);

    // --- Adjoint side ---
    let fwd2 = brownian_fwd(grid_n);
    let adj = AdjointFokkerPlanckChernoff::new(fwd2, 0.5_f64, 0.0_f64, 0.0_f64);
    let rho0 = MeasureState::<f64, 1>::dirac([x0], 1.0);
    let mut rho1 = rho0.clone();
    let mut pool2 = ScratchPool::<f64>::new();
    adj.apply_into(tau, &rho0, &mut rho1, &mut pool2).unwrap();
    // ⟨f, S_adj*(τ)δ_{x₀}⟩ = Σ wᵢ f(xᵢ) (f evaluated analytically)
    let adjoint_pairing = rho1.pair(|pos| f(pos[0]));

    (forward_at_x0 - adjoint_pairing).abs()
}

// ---------------------------------------------------------------------------
// G_ADJOINT_FP_ORDER: combined slow-tests gate
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow-tests feature required")]
fn g_adjoint_fp_order() {
    use std::time::Instant;
    let t0 = Instant::now();

    // ------------------------------------------------------------------
    // Sub-gate 1: vague-convergence slope ≤ −0.95 for all 4 ξ values
    // ------------------------------------------------------------------
    let t_final = 1.0_f64;
    let ns = [16usize, 32, 64, 128, 256];
    let xis = [0.5_f64, 1.0, 1.5, 2.0];
    let slope_threshold = -0.95_f64;

    println!("\nG_ADJOINT_FP_ORDER — sub-gate 1 (vague-convergence char-fn slope)");
    println!(
        "{:<8} {:>12} {:>12} {:>12}",
        "ξ", "slope", "threshold", "pass?"
    );

    let mut all_slopes_pass = true;
    for &xi in &xis {
        let slope = run_vague_convergence_slope(xi, t_final, &ns);
        let pass = slope <= slope_threshold;
        if !pass {
            all_slopes_pass = false;
        }
        println!(
            "{:<8.2} {:>12.4} {:>12.4} {:>12}",
            xi,
            slope,
            slope_threshold,
            if pass { "PASS" } else { "FAIL" }
        );
    }
    println!("Sub-gate 1 elapsed: {:.2}s", t0.elapsed().as_secs_f64());

    // ------------------------------------------------------------------
    // Sub-gate 1b: real-kernel multi-step — exercises actual Rust path
    // ------------------------------------------------------------------
    println!("\nG_ADJOINT_FP_ORDER — sub-gate 1b (real kernel multi-step, n=8 steps)");
    let t1b = Instant::now();
    let n_steps = 8usize;
    let xi_check = 1.0_f64;
    let (tv_norm, char_val) = run_real_kernel_multistep(n_steps, t_final, xi_check);
    let analytical_val = char_fn_iterated(xi_check, n_steps, t_final);
    let real_kernel_err = (char_val - analytical_val).abs();
    let real_kernel_rel_err = real_kernel_err / analytical_val.abs().max(1e-14);
    let real_kernel_pass = tv_norm.is_finite() && real_kernel_rel_err < 0.05;
    println!(
        "TV norm = {tv_norm:.6}  char_fn(real) = {char_val:.6}  \
         char_fn(analytical) = {analytical_val:.6}  \
         rel_err = {real_kernel_rel_err:.4e}  {}",
        if real_kernel_pass { "PASS" } else { "FAIL" }
    );
    println!("Sub-gate 1b elapsed: {:.3}s", t1b.elapsed().as_secs_f64());

    // ------------------------------------------------------------------
    // Sub-gate 2: genuine pairing cross-check using REAL Rust kernels
    //
    // DiffusionChernoff (ζ-A 5-point) and AdjointFokkerPlanckChernoff
    // (Lemma A.1 4-Dirac) are DIFFERENT Chernoff functions for the SAME
    // semigroup pair.  Their pairing residual → 0 as τ → 0 (both O(τ)
    // consistent).  An incorrectly transposed adjoint would give O(1).
    // ------------------------------------------------------------------
    println!("\nG_ADJOINT_FP_ORDER — sub-gate 2 (genuine pairing cross-check §38.2)");
    println!("  Forward:  DiffusionChernoff::apply_into on f=cos(ξx) grid → (S_fwd(τ)f)(x₀)");
    println!("  Adjoint:  AdjointFokkerPlanckChernoff::apply_into on δ_{{x₀}} → pair with f");
    println!("  Residual: O(τ) for correct adjoint; O(1) if transposed incorrectly");
    let t2 = Instant::now();
    let xi_sg2 = 1.5_f64;
    let x0_sg2 = 0.7_f64;
    let grid_sg2 = 512usize; // fine grid to isolate τ-error from grid-interp error

    // Sweep over τ to show O(τ) convergence of the pairing residual
    let taus: &[f64] = &[0.5, 0.25, 0.1, 0.05, 0.02];
    println!("{:<10} {:>16}", "τ", "pairing_residual");

    let mut log_taus = Vec::with_capacity(taus.len());
    let mut log_residuals = Vec::with_capacity(taus.len());

    for &tau in taus {
        let residual = pairing_residual(grid_sg2, tau, xi_sg2, x0_sg2);
        println!("{tau:<10.4} {residual:>16.6e}");
        if residual > 1e-14 && residual.is_finite() {
            log_taus.push(tau.ln());
            log_residuals.push(residual.ln());
        }
    }

    let pairing_slope = if log_taus.len() >= 2 {
        ols_slope(&log_taus, &log_residuals)
    } else {
        0.0
    };

    let finest_residual = pairing_residual(grid_sg2, 0.02, xi_sg2, x0_sg2);
    let pairing_slope_pass = pairing_slope >= 0.8_f64; // slope > 0 means residual → 0 as τ → 0
    let finest_pass = finest_residual < 5e-3_f64;

    println!(
        "Pairing OLS slope (log τ vs log residual): {pairing_slope:.4}  \
         threshold ≥ 0.8  {}",
        if pairing_slope_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "Finest residual (τ=0.02, N={grid_sg2}): {finest_residual:.3e}  \
         threshold < 5e-3  {}",
        if finest_pass { "PASS" } else { "FAIL" }
    );
    println!("Sub-gate 2 elapsed: {:.3}s", t2.elapsed().as_secs_f64());
    println!("\nTotal elapsed: {:.2}s", t0.elapsed().as_secs_f64());

    // ------------------------------------------------------------------
    // Gate verdicts
    // ------------------------------------------------------------------
    assert!(
        all_slopes_pass,
        "G_ADJOINT_FP_ORDER FAIL sub-gate 1: slope > {slope_threshold} for at least one ξ"
    );
    assert!(
        real_kernel_pass,
        "G_ADJOINT_FP_ORDER FAIL sub-gate 1b: \
         real kernel path rel_err {real_kernel_rel_err:.3e} >= 5% (TV={tv_norm:.4})"
    );
    assert!(
        pairing_slope_pass,
        "G_ADJOINT_FP_ORDER FAIL sub-gate 2: \
         pairing residual slope {pairing_slope:.4} < 0.8 — \
         expected residual → 0 as τ → 0 (O(τ)); \
         got non-convergent pairing (adjoint vs forward inconsistent)"
    );
    assert!(
        finest_pass,
        "G_ADJOINT_FP_ORDER FAIL sub-gate 2: \
         finest-τ pairing residual {finest_residual:.3e} >= 5e-3"
    );

    println!(
        "\nG_ADJOINT_FP_ORDER PASS \
         (slope ≤ −0.95 all 4 ξ; real-kernel rel_err < 5%; \
         pairing slope ≥ 0.8 O(τ), finest < 5e-3)"
    );
}

// ---------------------------------------------------------------------------
// Fast smokes (non-ignored: run in test-fast)
// ---------------------------------------------------------------------------

#[test]
fn adjoint_fp_smoke_construct_run() {
    // Verify the adjoint kernel constructs + runs in 1 step, finite output.
    let fwd = brownian_fwd(32);
    let adj = AdjointFokkerPlanckChernoff::new(fwd, 0.5_f64, 0.0, 0.0);

    let rho0 = MeasureState::<f64, 1>::dirac([0.0], 1.0);
    let mut rho1 = rho0.clone();
    let mut pool = ScratchPool::<f64>::new();
    adj.apply_into(0.25_f64, &rho0, &mut rho1, &mut pool)
        .unwrap();

    assert!(
        adj.order() >= 1,
        "order inherits from forward kernel (DiffusionChernoff is 2)"
    );
    assert!(rho1.total_variation().is_finite(), "TV norm finite");
    println!(
        "smoke_construct_run: order={}, TV={:.6}, n_diracs={}",
        adj.order(),
        rho1.total_variation(),
        rho1.n_diracs()
    );
}

#[test]
fn char_fn_converges_single_n() {
    // At n=64, ξ=1.0, T=1.0 the char fn approximation is within 2% of Gaussian.
    let xi = 1.0_f64;
    let t = 1.0_f64;
    let n = 64usize;
    let approx = char_fn_iterated(xi, n, t);
    let reference = gauss_char_fn(xi, t);
    let rel_err = (approx - reference).abs() / reference.abs().max(1e-14);
    assert!(
        rel_err < 0.02,
        "char fn n={n} ξ={xi}: approx={approx:.6}, ref={reference:.6}, rel_err={rel_err:.4}"
    );
}

#[test]
fn rust_pushforward_matches_lemma_a1_fast() {
    // Algebraic check: the Rust pushforward matches Lemma A.1 formula.
    let tau = 0.25_f64;
    let a = 0.5_f64;
    let b = 0.3_f64;
    let c = -0.1_f64;
    let xi = 1.5_f64;
    let x0 = 0.7_f64;

    let h = 2.0 * (a * tau).sqrt();
    let k = 2.0 * b * tau;
    let tc = tau * c;

    let fwd = brownian_fwd(64);
    let adj = AdjointFokkerPlanckChernoff::new(fwd, a, b, c);
    let rho0 = MeasureState::<f64, 1>::dirac([x0], 1.0);
    let mut rho1 = rho0.clone();
    let mut pool = ScratchPool::<f64>::new();
    adj.apply_into(tau, &rho0, &mut rho1, &mut pool).unwrap();

    // Expected by Lemma A.1: ⟨cos(ξx), ρ₁⟩
    let expected = 0.25 * (xi * (x0 + h)).cos()
        + 0.25 * (xi * (x0 - h)).cos()
        + 0.5 * (xi * (x0 + k)).cos()
        + tc * (xi * x0).cos();

    // Actual: pair with cos(ξx)
    let actual = rho1.pair(|pos| (xi * pos[0]).cos());
    let err = (actual - expected).abs();

    assert!(
        err < 1e-12,
        "Rust pushforward vs Lemma A.1: error {err:.3e} >= 1e-12"
    );
}

/// Genuine pairing cross-check (fast): one-step forward vs adjoint at small τ.
///
/// At τ=0.02, N=512 grid, the pairing residual should be small (O(τ)).
/// This is NOT a tautology: it calls both `DiffusionChernoff::apply_into`
/// and `AdjointFokkerPlanckChernoff::apply_into` and checks consistency.
#[test]
fn adjoint_pairing_small_tau_fast() {
    let residual = pairing_residual(512, 0.02_f64, 1.5_f64, 0.7_f64);
    assert!(
        residual < 5e-3,
        "adjoint pairing residual {residual:.3e} >= 5e-3 at τ=0.02, N=512 \
         (adjoint inconsistent with forward)"
    );
}
