//! `G_BINDING_REVERSE_AD_PARITY` — sub-test 1 (core golden, `RELEASE_BLOCKING`).
//!
//! Gate spec (v9.0.0 Shift B, math §51.5, ADR-0156):
//!
//! Canonical smoke: constant-a `DiffusionChernoff` K=1 `ReverseChernoff`,
//!   θ = 0.4, `n_grid` = 24, domain [−4, 4], `n_steps` = 8, τ = 0.05,
//!   u0 = exp(−x²), target = 0.
//!
//! This file produces the golden `(value, grad)` pair that all binding
//! sub-tests (`PyO3`, WASM) must match byte-for-byte (0 ULP).
//!
//! Sub-tests 2 (`PyO3`) and 3 (WASM) run the SAME Rust arithmetic inline
//! (per-crate dup, ADR-0028 Amdt 2) and verify 0-ULP vs this golden.
//! Any divergence between a binding and this core golden would indicate
//! a marshalling bug in that binding layer.
//!
//! # Why GENUINE (not a tautology)
//!
//! The binding sub-tests independently reconstruct `ReverseChernoff` from
//! flat scalar parameters received from the host language (Python/JS).
//! A wrong parameter conversion, wrong array layout, or wrong Dual seed
//! would produce a different bit pattern from the golden defined here.
//! The cross-binding 0-ULP assertion ensures that the host-side marshalling
//! path is byte-identical to the direct Rust call.

#![allow(clippy::cast_precision_loss)]
// Integration test: allows for numerical / binding wrapper patterns.
#![allow(
    clippy::cast_possible_wrap,
    clippy::missing_panics_doc,
    clippy::similar_names
)]

use semiflow_core::{
    CheckpointSchedule, DiffusionChernoff, Dual, Grid1D, GridFn1D, InterpKind, ReverseChernoff,
};

// ---------------------------------------------------------------------------
// Canonical smoke parameters
// ---------------------------------------------------------------------------

/// Diffusivity parameter θ.
pub const THETA: f64 = 0.4;
/// Grid node count.
pub const N_GRID: usize = 24;
/// Left domain boundary.
pub const X_MIN: f64 = -4.0;
/// Right domain boundary.
pub const X_MAX: f64 = 4.0;
/// Chernoff steps.
pub const N_STEPS: usize = 8;
/// Per-step time increment.
pub const TAU: f64 = 0.05;

// ---------------------------------------------------------------------------
// Core reference computation
// ---------------------------------------------------------------------------

/// Build the canonical `ReverseChernoff<f64>` for the smoke parameters.
///
/// Both kernels use `CubicHermite` (matching `reverse_ad.rs` unit tests and
/// the §46 Dual grid convention).
#[must_use]
pub fn build_canonical_rc() -> ReverseChernoff<f64> {
    let grid_f64 = Grid1D::<f64>::new(X_MIN, X_MAX, N_GRID)
        .expect("f64 grid valid")
        .with_interp(InterpKind::CubicHermite);

    let grid_dual =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .expect("dual grid valid")
            .with_interp(InterpKind::CubicHermite);

    let kernel_f64 = DiffusionChernoff::with_closure(
        |_: f64| THETA,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        THETA,
        grid_f64,
    );

    let kernel_dual = DiffusionChernoff::<Dual<f64>>::with_closure(
        |_: Dual<f64>| Dual::variable(THETA),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        THETA,
        grid_dual,
    );

    let schedule = CheckpointSchedule::sqrt_n(N_STEPS);
    ReverseChernoff::new(kernel_f64, kernel_dual, schedule)
}

/// Canonical initial condition: u0[i] = `exp(−x_i²)`.
#[must_use]
pub fn make_u0() -> Vec<f64> {
    let dx = (X_MAX - X_MIN) / (N_GRID - 1) as f64;
    (0..N_GRID)
        .map(|i| {
            let x = X_MIN + i as f64 * dx;
            (-x * x).exp()
        })
        .collect()
}

/// Target: all zeros (convenient; loss = ‖`u_n‖²`).
#[must_use]
pub fn make_target() -> Vec<f64> {
    vec![0.0_f64; N_GRID]
}

/// Run the canonical `value_and_grad_k1` and return `(value, grad)`.
///
/// This is the golden pair that all binding sub-tests must match byte-for-byte.
#[must_use]
pub fn canonical_reverse_ad_core() -> (f64, f64) {
    let rc = build_canonical_rc();
    let grid = Grid1D::<f64>::new(X_MIN, X_MAX, N_GRID)
        .expect("grid valid")
        .with_interp(InterpKind::CubicHermite);
    let u0_fn = GridFn1D::new(grid, make_u0()).expect("u0 valid");
    let target_fn = GridFn1D::new(grid, make_target()).expect("target valid");

    rc.value_and_grad_k1(TAU, N_STEPS, &u0_fn, &target_fn)
        .expect("value_and_grad_k1 ok")
}

// ---------------------------------------------------------------------------
// K-vector canonical reference (Phase 3 — K>1 gradient, 0-ULP parity)
// ---------------------------------------------------------------------------

/// K for the K-vector smoke (must match across all binding sub-tests).
pub const K_VEC: usize = 4;

/// Run `value_and_grad` with a K-vector `theta = [THETA; K_VEC]` and return
/// `(value, grad_vec)`. This is the golden pair for sub-test 4 (K-vector parity).
#[must_use]
pub fn canonical_reverse_ad_kvec() -> (f64, Vec<f64>) {
    let rc = build_canonical_rc();
    let grid = Grid1D::<f64>::new(X_MIN, X_MAX, N_GRID)
        .expect("grid valid")
        .with_interp(InterpKind::CubicHermite);
    let u0_fn = GridFn1D::new(grid, make_u0()).expect("u0 valid");
    let target_fn = GridFn1D::new(grid, make_target()).expect("target valid");
    let theta = vec![THETA; K_VEC];
    rc.value_and_grad(TAU, N_STEPS, &u0_fn, &target_fn, &theta)
        .expect("value_and_grad K-vec ok")
}

// ---------------------------------------------------------------------------
// Test: core golden (RELEASE_BLOCKING)
// ---------------------------------------------------------------------------

#[test]
fn g_binding_reverse_ad_parity_core_golden() {
    let (value, grad) = canonical_reverse_ad_core();

    // Sanity: value is the L² norm of the evolved state — must be positive.
    assert!(
        value.is_finite() && value > 0.0,
        "G_BINDING_REVERSE_AD_PARITY: value = {value:.6e} must be finite and > 0"
    );
    // Sanity: grad must be finite.
    assert!(
        grad.is_finite(),
        "G_BINDING_REVERSE_AD_PARITY: grad = {grad:.6e} must be finite"
    );

    println!(
        "G_BINDING_REVERSE_AD_PARITY (core golden):\n\
         config: θ={THETA}, N={N_GRID}, n_steps={N_STEPS}, τ={TAU}, \
         domain=[{X_MIN},{X_MAX}], u0=exp(-x²), target=0\n\
         value = {value:.16e}\n\
         grad  = {grad:.16e}\n\
         NOTE: sub-tests 2 (PyO3) and 3 (WASM) must match these bit-exactly (0 ULP).",
    );
}

// ---------------------------------------------------------------------------
// Test: determinism — two identical runs must be bit-exact (0 ULP)
// ---------------------------------------------------------------------------

#[test]
fn g_binding_reverse_ad_parity_determinism_0ulp() {
    let (va, ga) = canonical_reverse_ad_core();
    let (vb, gb) = canonical_reverse_ad_core();

    let value_ulp = (va.to_bits() as i64 - vb.to_bits() as i64).unsigned_abs();
    let grad_ulp = (ga.to_bits() as i64 - gb.to_bits() as i64).unsigned_abs();

    println!(
        "G_BINDING_REVERSE_AD_PARITY (determinism):\n\
         run_a: value={va:.16e}  grad={ga:.16e}\n\
         run_b: value={vb:.16e}  grad={gb:.16e}\n\
         ULP diff: value={value_ulp}  grad={grad_ulp}  (both must be 0)",
    );

    assert_eq!(
        value_ulp, 0,
        "G_BINDING_REVERSE_AD_PARITY: value not bit-identical across two runs \
         (ULP={value_ulp})"
    );
    assert_eq!(
        grad_ulp, 0,
        "G_BINDING_REVERSE_AD_PARITY: grad not bit-identical across two runs \
         (ULP={grad_ulp})"
    );
}

// ---------------------------------------------------------------------------
// Test: numerical anchor — grad matches central-FD within loose tolerance
// ---------------------------------------------------------------------------

/// Evaluate J(θ) = ‖(`F_θ(τ))ⁿ` u₀‖² at a given θ (independent f64 path).
fn eval_loss_at_theta(theta: f64) -> f64 {
    let grid = Grid1D::<f64>::new(X_MIN, X_MAX, N_GRID)
        .expect("grid ok")
        .with_interp(InterpKind::CubicHermite);
    let k = DiffusionChernoff::with_closure(
        move |_: f64| theta,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        theta,
        grid,
    );
    let u0_vec = make_u0();
    let mut u = GridFn1D::new(grid, u0_vec).expect("u0 ok");
    for _ in 0..N_STEPS {
        u = k.apply_f(TAU, &u).expect("step");
    }
    // Loss = ‖u_n − target‖² = ‖u_n‖² (target = 0).
    u.values.iter().map(|&v| v * v).sum()
}

#[test]
fn g_binding_reverse_ad_parity_grad_fd_anchor() {
    let (_, grad_reverse) = canonical_reverse_ad_core();

    // Richardson 4-point FD at h = 1e-4.
    let h = 1e-4_f64;
    let fp2 = eval_loss_at_theta(THETA + 2.0 * h);
    let fp1 = eval_loss_at_theta(THETA + h);
    let fn1 = eval_loss_at_theta(THETA - h);
    let fn2 = eval_loss_at_theta(THETA - 2.0 * h);
    let grad_rich = (-fp2 + 8.0 * fp1 - 8.0 * fn1 + fn2) / (12.0 * h);

    let rel = (grad_reverse - grad_rich).abs() / (grad_rich.abs() + 1e-30);

    println!(
        "G_BINDING_REVERSE_AD_PARITY (FD anchor, h={h:.0e}):\n\
         grad_reverse = {grad_reverse:.16e}\n\
         grad_rich    = {grad_rich:.16e}\n\
         rel diff     = {rel:.3e}  (gate < 1e-5)",
    );

    assert!(
        rel < 1e-5,
        "G_BINDING_REVERSE_AD_PARITY: grad {grad_reverse:.6e} vs FD {grad_rich:.6e}, \
         rel={rel:.3e} > 1e-5"
    );
}

// ---------------------------------------------------------------------------
// Sub-test 4 (K-vector) — golden + determinism (RELEASE_BLOCKING, Phase 3)
// ---------------------------------------------------------------------------

/// Core golden for K-vector case: two identical runs must be 0-ULP.
#[test]
fn g_binding_reverse_ad_parity_kvec_core_golden() {
    let (value, grad_vec) = canonical_reverse_ad_kvec();

    assert!(
        value.is_finite() && value > 0.0,
        "G_BINDING_REVERSE_AD_PARITY (kvec): value = {value:.6e} must be finite and > 0"
    );
    assert_eq!(
        grad_vec.len(),
        K_VEC,
        "G_BINDING_REVERSE_AD_PARITY (kvec): grad_vec len {}, expected {K_VEC}",
        grad_vec.len()
    );
    for (p, &g) in grad_vec.iter().enumerate() {
        assert!(
            g.is_finite(),
            "G_BINDING_REVERSE_AD_PARITY (kvec): grad[{p}] = {g:.6e} must be finite"
        );
    }

    println!(
        "G_BINDING_REVERSE_AD_PARITY (kvec golden, K={K_VEC}):\n\
         config: θ={THETA}×{K_VEC}, N={N_GRID}, n_steps={N_STEPS}, τ={TAU}\n\
         value      = {value:.16e}\n\
         grad_vec   = {grad_vec:?}\n\
         NOTE: sub-tests 2 (PyO3) and 3 (WASM) K-vec must match bit-exactly (0 ULP)."
    );
}

/// K-vector determinism: two identical runs must be 0 ULP.
#[test]
fn g_binding_reverse_ad_parity_kvec_determinism_0ulp() {
    let (va, ga) = canonical_reverse_ad_kvec();
    let (vb, gb) = canonical_reverse_ad_kvec();

    let v_ulp = (va.to_bits() as i64 - vb.to_bits() as i64).unsigned_abs();
    for (p, (a, b)) in ga.iter().zip(gb.iter()).enumerate() {
        let ulp = (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs();
        assert_eq!(
            ulp, 0,
            "G_BINDING_REVERSE_AD_PARITY (kvec determinism): grad[{p}] ULP={ulp}"
        );
    }
    assert_eq!(
        v_ulp, 0,
        "G_BINDING_REVERSE_AD_PARITY (kvec determinism): value ULP={v_ulp}"
    );
    println!("G_BINDING_REVERSE_AD_PARITY (kvec determinism): value ULP=0 grad ULP=0 ✓");
}

/// K-vector vs K=1: all K grad components must equal the K=1 scalar gradient
/// (same θ seed, same backward sweep — just accumulated K times).
#[test]
fn g_binding_reverse_ad_parity_kvec_vs_k1_0ulp() {
    let (_, grad_k1) = canonical_reverse_ad_core();
    let (_, grad_kv) = canonical_reverse_ad_kvec();

    for (p, &g) in grad_kv.iter().enumerate() {
        let ulp = (g.to_bits() as i64 - grad_k1.to_bits() as i64).unsigned_abs();
        assert_eq!(
            ulp, 0,
            "G_BINDING_REVERSE_AD_PARITY (kvec vs k1): grad_vec[{p}]={g:.16e} \
             vs k1={grad_k1:.16e} ULP={ulp} (expected 0)"
        );
    }
    println!(
        "G_BINDING_REVERSE_AD_PARITY (kvec vs k1, K={K_VEC}): all {K_VEC} \
         components 0-ULP vs K=1 scalar ✓"
    );
}
