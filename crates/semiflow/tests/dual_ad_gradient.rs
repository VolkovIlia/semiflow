//! `G_DUAL_AD_GRADIENT` — forward-mode dual-number gradient correctness gate.
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0133, math.md §46.7):
//!   |`forward_mode_grad` − `richardson_fd_grad`| ≤ 1e-10
//!   for {`DiffusionChernoff`, `Diffusion4thChernoff`, `StrangSplit`}.
//!
//! # Design: why Richardson extrapolation (not raw central diff)?
//!
//! The dual forward mode yields the exact derivative (up to `ε_mach` rounding).
//! Raw central difference at h=1e-5 has truncation error O(h²·f''') ~ O(1e-10)
//! and roundoff ~ `ε_mach/h` ~ 1e-11, so its *own* error floor is ~1e-10, making
//! a 1e-10 forward-vs-central comparison marginal by construction.
//! Richardson extrapolation (4-point stencil) cancels the O(h²) term,
//! achieving O(h⁴) truncation ~ 1e-20 at h=1e-3, leaving only ~1e-11 roundoff.
//! The 1e-10 gate is then a genuine test of the FORWARD mode, not of two
//! approximate methods agreeing. The architect's contract is satisfied:
//! sub-check (d) in `dual_ad_kit.py` already validates h=1e-5 central-diff at
//! the scalar level; the full-kernel gate uses Richardson for honesty.
//!
//! # Note on `ChernoffFunction<Dual<f64>>`
//!
//! `DiffusionChernoff<Dual<f64>>` does NOT implement `ChernoffFunction<Dual<f64>>`
//! (the trait impl is f64-concrete for SIMD; ADR-0018). The generic path is
//! `DiffusionChernoff::apply_f(tau, &u)`, which the existing unit test
//! `dual_kernels_compile_and_run` demonstrates. We iterate `apply_f` in a
//! manual loop to compute the full Chernoff product.
//!
//! # Parameter
//!   θ = diffusivity constant; seed tangent 1.0 at θ₀ = 0.5.
//!   Initial condition: u₀(x) = exp(-x²) (θ-independent → data tangent = 0).
//!   Observable: scalar L² norm of the evolved state.
//!   t = 1.0, n = 32 Chernoff steps, N = 128 grid nodes.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)]

use semiflow::{
    chernoff::ApplyChernoffExt, Diffusion4thChernoff, DiffusionChernoff, DriftReactionChernoff,
    Dual, Grid1D, GridFn1D, InterpKind,
};

// ---------------------------------------------------------------------------
// Gate constants (NON-NEGOTIABLE per ADR-0133)
// ---------------------------------------------------------------------------

const GRAD_GATE: f64 = 1e-10;
const THETA0: f64 = 0.5;
const T_FINAL: f64 = 1.0;
const N_STEPS: usize = 32;
const N_GRID: usize = 128;
const X_MIN: f64 = -10.0;
const X_MAX: f64 = 10.0;
/// h for 4-point Richardson extrapolation: O(h⁴) truncation after extrapolation.
const FD_H: f64 = 1e-3;

// ---------------------------------------------------------------------------
// fn-ptrs for Dual<f64> kernels (closures cannot coerce to fn(Dual) -> Dual).
// ---------------------------------------------------------------------------

fn a_seeded_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::variable(THETA0)
}
fn zero_dual(_: Dual<f64>) -> Dual<f64> {
    Dual::constant(0.0)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Discrete L² norm of a f64 grid function.
fn l2_f64(u: &GridFn1D<f64>) -> f64 {
    let dx = (X_MAX - X_MIN) / (N_GRID - 1) as f64;
    u.values.iter().map(|&v| v * v * dx).sum::<f64>().sqrt()
}

/// d/dθ of ‖u(θ)‖₂ via the tangent field: (u · ∂u/∂θ · dx) / ‖u‖₂.
fn grad_from_dual(u: &GridFn1D<Dual<f64>>) -> f64 {
    let dx = (X_MAX - X_MIN) / (N_GRID - 1) as f64;
    let norm_sq: f64 = u.values.iter().map(|d| d.value * d.value * dx).sum();
    let dot: f64 = u.values.iter().map(|d| d.value * d.tangent * dx).sum();
    dot / norm_sq.sqrt()
}

/// `Grid1D<f64>` with `CubicHermite` interp for the FD reference path of the
/// existing sub-tests (Diffusion, Diffusion4th, Strang). These were written
/// when `new_generic` defaulted to `CubicHermite`; they pin explicitly so
/// their reference path matches the Dual<f64> forward path.
///
/// The new `g_dual_ad_gradient_diffusion_septic_default` sub-test uses
/// `Grid1D::<f64>::new_generic` (now `SepticHermite`) for its own reference.
fn f64_grid() -> Grid1D<f64> {
    Grid1D::new(X_MIN, X_MAX, N_GRID)
        .expect("grid valid")
        .with_interp(InterpKind::CubicHermite)
}

/// 4-point Richardson central-difference of scalar function f(θ):
/// [−f(θ+2h) + 8f(θ+h) − 8f(θ−h) + f(θ−2h)] / (12h) = O(h⁴).
fn richardson(f: impl Fn(f64) -> f64) -> f64 {
    let h = FD_H;
    let fp2 = f(THETA0 + 2.0 * h);
    let fp1 = f(THETA0 + h);
    let fn1 = f(THETA0 - h);
    let fn2 = f(THETA0 - 2.0 * h);
    (-fp2 + 8.0 * fp1 - 8.0 * fn1 + fn2) / (12.0 * h)
}

/// Manual n-step `apply_f` loop (mirrors `Evolver::evolve` for generic `F`).
fn chernoff_product_dual<K>(
    kernel: &K,
    tau: Dual<f64>,
    u0: &GridFn1D<Dual<f64>>,
) -> GridFn1D<Dual<f64>>
where
    K: Fn(
        Dual<f64>,
        &GridFn1D<Dual<f64>>,
    ) -> Result<GridFn1D<Dual<f64>>, semiflow::error::SemiflowError>,
{
    let mut u = u0.clone();
    for _ in 0..N_STEPS {
        u = kernel(tau, &u).expect("apply_f step");
    }
    u
}

// ---------------------------------------------------------------------------
// Sub-test: DiffusionChernoff
// ---------------------------------------------------------------------------

/// Forward-mode gradient of L²(θ) at θ₀ via DiffusionChernoff<Dual<f64>>.
/// Pinned to CubicHermite to match `f64_grid()` reference (pre-v8 cross-check).
fn forward_grad_diffusion() -> f64 {
    let grid =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .expect("grid valid")
            .with_interp(InterpKind::CubicHermite);
    let diff = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        THETA0,
        grid,
    );
    let u0 = GridFn1D::from_fn_generic(grid, |x| Dual::constant((-x.value * x.value).exp()));
    let tau = Dual::constant(T_FINAL / N_STEPS as f64);
    let u_t = chernoff_product_dual(&|t, u| diff.apply_f(t, u), tau, &u0);
    grad_from_dual(&u_t)
}

/// f64 L² norm of DiffusionChernoff at diffusivity θ — uses `apply_f` (generic
/// scalar path) to match EXACTLY the code path exercised by the Dual<f64> mode.
/// Uses CubicHermite interp to match `new_generic` default (see `f64_grid`).
fn l2_diffusion(theta: f64) -> f64 {
    let grid = f64_grid();
    let diff =
        DiffusionChernoff::with_closure(move |_| theta, |_| 0.0_f64, |_| 0.0_f64, theta, grid);
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let tau = T_FINAL / N_STEPS as f64;
    let mut u = u0.clone();
    for _ in 0..N_STEPS {
        u = diff.apply_f(tau, &u).expect("step ok");
    }
    l2_f64(&u)
}

#[test]
#[ignore = "G_DUAL_AD_GRADIENT: run with --features slow-tests --release -- --ignored"]
fn g_dual_ad_gradient_diffusion() {
    let fwd = forward_grad_diffusion();
    let ref_grad = richardson(l2_diffusion);
    let err = (fwd - ref_grad).abs();
    println!(
        "G_DUAL_AD_GRADIENT DiffusionChernoff: \
         forward={fwd:.12e}, richardson={ref_grad:.12e}, |diff|={err:.3e}  \
         (gate <= {GRAD_GATE:.0e})"
    );
    assert!(
        err <= GRAD_GATE,
        "G_DUAL_AD_GRADIENT DiffusionChernoff FAIL: \
         |forward − reference| = {err:.3e} > {GRAD_GATE:.0e}"
    );
}

// ---------------------------------------------------------------------------
// Sub-test: Diffusion4thChernoff
// ---------------------------------------------------------------------------

/// Pinned to CubicHermite to match `f64_grid()` reference (pre-v8 cross-check).
fn forward_grad_diffusion4() -> f64 {
    let grid =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .expect("grid valid")
            .with_interp(InterpKind::CubicHermite);
    let diff4 = Diffusion4thChernoff::<Dual<f64>>::new_generic(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        THETA0,
        grid,
    );
    let u0 = GridFn1D::from_fn_generic(grid, |x| Dual::constant((-x.value * x.value).exp()));
    let tau = Dual::constant(T_FINAL / N_STEPS as f64);
    let u_t = chernoff_product_dual(&|t, u| diff4.apply_f(t, u), tau, &u0);
    grad_from_dual(&u_t)
}

fn l2_diffusion4(theta: f64) -> f64 {
    let grid = f64_grid();
    let diff4 =
        Diffusion4thChernoff::with_closure(move |_| theta, |_| 0.0_f64, |_| 0.0_f64, theta, grid);
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let tau = T_FINAL / N_STEPS as f64;
    let mut u = u0.clone();
    for _ in 0..N_STEPS {
        u = diff4.apply_chernoff(tau, &u).expect("step ok");
    }
    l2_f64(&u)
}

#[test]
#[ignore = "G_DUAL_AD_GRADIENT: run with --features slow-tests --release -- --ignored"]
fn g_dual_ad_gradient_diffusion4() {
    let fwd = forward_grad_diffusion4();
    let ref_grad = richardson(l2_diffusion4);
    let err = (fwd - ref_grad).abs();
    println!(
        "G_DUAL_AD_GRADIENT Diffusion4thChernoff: \
         forward={fwd:.12e}, richardson={ref_grad:.12e}, |diff|={err:.3e}  \
         (gate <= {GRAD_GATE:.0e})"
    );
    assert!(
        err <= GRAD_GATE,
        "G_DUAL_AD_GRADIENT Diffusion4thChernoff FAIL: \
         |forward − reference| = {err:.3e} > {GRAD_GATE:.0e}"
    );
}

// ---------------------------------------------------------------------------
// Sub-test: StrangSplit (θ enters the diffusion leg)
// ---------------------------------------------------------------------------

/// StrangSplit step via the generic apply_into path:
/// D(τ/2) ∘ R(τ) ∘ D(τ/2) using `apply_f` on each leg.
fn strang_step_dual(
    diff: &DiffusionChernoff<Dual<f64>>,
    drift: &DriftReactionChernoff<Dual<f64>>,
    tau: Dual<f64>,
    u: &GridFn1D<Dual<f64>>,
) -> GridFn1D<Dual<f64>> {
    let two = Dual::constant(2.0);
    let half_tau = tau / two;
    let u1 = diff.apply_f(half_tau, u).expect("D(τ/2)");
    let u2 = drift.apply_f(tau, &u1).expect("R(τ)");
    diff.apply_f(half_tau, &u2).expect("D(τ/2)")
}

/// Pinned to CubicHermite to match `f64_grid()` reference (pre-v8 cross-check).
fn forward_grad_strang() -> f64 {
    let grid =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .expect("grid valid")
            .with_interp(InterpKind::CubicHermite);
    let diff = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        THETA0,
        grid,
    );
    let drift = DriftReactionChernoff::<Dual<f64>>::new_generic(
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        0.0,
        grid,
    );
    let u0 = GridFn1D::from_fn_generic(grid, |x| Dual::constant((-x.value * x.value).exp()));
    let tau = Dual::constant(T_FINAL / N_STEPS as f64);
    let mut u = u0.clone();
    for _ in 0..N_STEPS {
        u = strang_step_dual(&diff, &drift, tau, &u);
    }
    grad_from_dual(&u)
}

/// f64 L² norm of StrangSplit at diffusivity θ — uses apply_f on each leg
/// to match the generic path exercised by Dual<f64> forward mode.
fn l2_strang(theta: f64) -> f64 {
    let grid = f64_grid();
    let diff =
        DiffusionChernoff::with_closure(move |_| theta, |_| 0.0_f64, |_| 0.0_f64, theta, grid);
    let drift = DriftReactionChernoff::new(|_| 0.0_f64, |_| 0.0_f64, 0.0, grid);
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let tau = T_FINAL / N_STEPS as f64;
    let half_tau = tau / 2.0;
    let mut u = u0.clone();
    for _ in 0..N_STEPS {
        let u1 = diff.apply_f(half_tau, &u).expect("D(τ/2)");
        let u2 = drift.apply_f(tau, &u1).expect("R(τ)");
        u = diff.apply_f(half_tau, &u2).expect("D(τ/2)");
    }
    l2_f64(&u)
}

#[test]
#[ignore = "G_DUAL_AD_GRADIENT: run with --features slow-tests --release -- --ignored"]
fn g_dual_ad_gradient_strang() {
    let fwd = forward_grad_strang();
    let ref_grad = richardson(l2_strang);
    let err = (fwd - ref_grad).abs();
    println!(
        "G_DUAL_AD_GRADIENT StrangSplit: \
         forward={fwd:.12e}, richardson={ref_grad:.12e}, |diff|={err:.3e}  \
         (gate <= {GRAD_GATE:.0e})"
    );
    assert!(
        err <= GRAD_GATE,
        "G_DUAL_AD_GRADIENT StrangSplit FAIL: \
         |forward − reference| = {err:.3e} > {GRAD_GATE:.0e}"
    );
}

// ---------------------------------------------------------------------------
// Sub-test: DiffusionChernoff on the DEFAULT new_generic grid (SepticHermite)
// §46.5.bis NORMATIVE: this sub-test FAILS if the SepticHermite arm in
// interp_generic regresses to Unsupported.  NOT pinned to CubicHermite.
// ---------------------------------------------------------------------------

/// Forward-mode gradient via the default `Grid1D::<Dual<f64>>::new_generic`
/// (SepticHermite since v8.0 §46.5.bis).  No `.with_interp` call.
fn forward_grad_diffusion_septic() -> f64 {
    let grid =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .expect("grid valid");
    // Grid::interp field is now SepticHermite (the new_generic default).
    // Do NOT override — that is the point of this sub-test.
    let diff = DiffusionChernoff::<Dual<f64>>::new(
        a_seeded_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        zero_dual as fn(Dual<f64>) -> Dual<f64>,
        THETA0,
        grid,
    );
    let u0 = GridFn1D::from_fn_generic(grid, |x| Dual::constant((-x.value * x.value).exp()));
    let tau = Dual::constant(T_FINAL / N_STEPS as f64);
    let u_t = chernoff_product_dual(&|t, u| diff.apply_f(t, u), tau, &u0);
    grad_from_dual(&u_t)
}

/// f64 L² norm for the SepticHermite reference path.
/// Uses `Grid1D::<f64>::new_generic` (default SepticHermite) + `apply_f`
/// to match the generic scalar path taken by the Dual<f64> forward mode.
fn l2_diffusion_septic(theta: f64) -> f64 {
    let grid = Grid1D::<f64>::new_generic(X_MIN, X_MAX, N_GRID).expect("grid valid");
    let diff =
        DiffusionChernoff::with_closure(move |_| theta, |_| 0.0_f64, |_| 0.0_f64, theta, grid);
    let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    let tau = T_FINAL / N_STEPS as f64;
    let mut u = u0.clone();
    for _ in 0..N_STEPS {
        u = diff.apply_f(tau, &u).expect("step ok");
    }
    l2_f64(&u)
}

/// §46.5.bis gate: forward-mode AD on the DEFAULT grid (SepticHermite)
/// matches Richardson FD to ≤ 1e-10.
///
/// This sub-test MUST be in the gate set per ADR-0133 Amendment 1: "at least
/// one kernel MUST run on a grid built by the default Grid1D::new constructor
/// (i.e. InterpKind::SepticHermite, §46.5.bis) — NOT pinned to CubicHermite".
#[test]
#[ignore = "G_DUAL_AD_GRADIENT: run with --features slow-tests --release -- --ignored"]
fn g_dual_ad_gradient_diffusion_septic_default() {
    let fwd = forward_grad_diffusion_septic();
    let ref_grad = richardson(l2_diffusion_septic);
    let err = (fwd - ref_grad).abs();
    println!(
        "G_DUAL_AD_GRADIENT DiffusionChernoff(SepticHermite default): \
         forward={fwd:.12e}, richardson={ref_grad:.12e}, |diff|={err:.3e}  \
         (gate <= {GRAD_GATE:.0e})"
    );
    assert!(
        err <= GRAD_GATE,
        "G_DUAL_AD_GRADIENT DiffusionChernoff(SepticHermite default) FAIL: \
         |forward − reference| = {err:.3e} > {GRAD_GATE:.0e}"
    );
}
