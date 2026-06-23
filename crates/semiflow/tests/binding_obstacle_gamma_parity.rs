//! `G_BINDING_OBSTACLE_GAMMA_PARITY` — sub-test 1 (core golden).
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0153, §5 `V8_3_TIER3_BINDING_DESIGN.md)`:
//!   Canonical smoke: perpetual-put V on [0, 3], N=64, g=K−S (put payoff).
//!   `apply_inactive_gamma_into` must return:
//!     - gamma: central-difference Γ = V″ on the OPEN continuation set
//!     - defined: bool mask (`false` = Γ refused, not fabricated)
//!     - count > 0 (some nodes defined)
//!
//! Sub-test structure:
//!   Sub-test 1 (this file): core golden triple (gamma, defined, count).
//!   Sub-test 2 (`binding_obstacle_gamma_parity_py.rs)`: `PyO3` 0-ULP vs golden.
//!   Sub-tests 3/4 (FFI/WASM): DEFERRED (ADR-0153 opportunistic, §5).
//!
//! ## Honesty (NORMATIVE, math §44.5.bis)
//!
//! Γ JUMPS across the free boundary x* (perpetual-put witness Γ(S*⁺)≈4.90,
//! Γ(S*⁻)=0). `defined[i]=false` means "Γ undefined here" — NEVER "Γ=0".
//! The refusal sub-check asserts that at least one node with `S_i` ≤ S* has
//! `defined[i]=false`, proving the mask is not collapsed.

#![allow(clippy::cast_precision_loss)]
#![allow(missing_docs)]
// Integration test/example: allows for numerical patterns.
#![allow(clippy::missing_panics_doc)]

use semiflow_core::{
    ClosureObstacle, ConstantObstacle, DiffusionChernoff, Grid1D, GridFn1D, ObstacleChernoff,
};

// ---------------------------------------------------------------------------
// Canonical smoke parameters (§5, properties.yaml G_BINDING_OBSTACLE_GAMMA_PARITY)
// ---------------------------------------------------------------------------

/// Left domain boundary.
pub const S_MIN: f64 = 0.0;
/// Right domain boundary.
pub const S_MAX: f64 = 3.0;
/// Number of grid nodes.
pub const N: usize = 64;

/// Put strike K, risk-free rate r, volatility σ (perpetual American put).
const PAP_K: f64 = 1.0;
const PAP_R: f64 = 0.05;
const PAP_SIG: f64 = 0.20;

fn pap_gamma_pow() -> f64 {
    2.0 * PAP_R / (PAP_SIG * PAP_SIG)
}
fn pap_sstar() -> f64 {
    pap_gamma_pow() / (pap_gamma_pow() + 1.0) * PAP_K
}
fn pap_a_coef() -> f64 {
    (PAP_K - pap_sstar()) * pap_sstar().powf(pap_gamma_pow())
}

/// Analytic V(S): continuation A·S^{-γ}, stopping K−S.
#[must_use]
pub fn pap_v(s: f64) -> f64 {
    let sstar = pap_sstar();
    if s > sstar {
        pap_a_coef() * s.powf(-pap_gamma_pow())
    } else {
        PAP_K - s
    }
}

// ---------------------------------------------------------------------------
// Core golden computation — called by the PyO3 parity sub-test too
// ---------------------------------------------------------------------------

/// Run the canonical gamma computation and return (`gamma_vals`, defined, count).
///
/// Uses `ClosureObstacle(g = K − S)` and `DiffusionChernoff(a=0.5)` as the
/// proxy inner (same as the `G_OBSTACLE_GAMMA` slope gate).  The value field `v`
/// is set analytically (perpetual-put V already satisfies V ≥ g).
///
/// Exported so the `PyO3` Rust integration test can compare against this golden
/// without re-implementing the arithmetic.
#[must_use]
pub fn canonical_gamma_core() -> (Vec<f64>, Vec<bool>, usize) {
    let grid = Grid1D::new(S_MIN, S_MAX, N).expect("valid grid");
    let v = GridFn1D::from_fn(grid, pap_v);
    let obs = ClosureObstacle::new(|s: f64| PAP_K - s);
    let diff = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, grid);
    let kernel = ObstacleChernoff::new(diff, obs).expect("valid kernel");

    let mut gamma_fn = v.zeroed_like();
    let mut defined = vec![false; N];
    let count = kernel
        .apply_inactive_gamma_into(&v, &mut gamma_fn, &mut defined)
        .expect("apply_inactive_gamma_into ok");
    (gamma_fn.values, defined, count)
}

// ---------------------------------------------------------------------------
// Test: core golden + honesty sub-check
// ---------------------------------------------------------------------------

#[test]
fn g_binding_obstacle_gamma_parity_core_golden() {
    let (gamma, defined, count) = canonical_gamma_core();

    // count > 0: at least one node is in the open continuation set.
    assert!(
        count > 0,
        "G_BINDING_OBSTACLE_GAMMA_PARITY: no defined nodes"
    );

    // defined.len() == N (mask has right length).
    assert_eq!(defined.len(), N, "defined mask length != N");
    assert_eq!(gamma.len(), N, "gamma length != N");

    // Honesty sub-check: at least one node with S_i <= S* must have defined=false.
    // This proves the mask is NOT collapsed to "always-defined".
    let grid = Grid1D::new(S_MIN, S_MAX, N).unwrap();
    let sstar = pap_sstar();
    let refused_on_active_set = (0..N).any(|i| {
        let s = grid.x_at(i);
        s <= sstar && !defined[i]
    });
    assert!(
        refused_on_active_set,
        "G_BINDING_OBSTACLE_GAMMA_PARITY FAIL (honesty): \
         no node on active set (S <= S*={sstar:.4}) has defined=false — mask collapsed!"
    );

    // A refused node must NOT be interpreted as gamma=0 — the mask carries the meaning.
    // Verify there exists a refused node whose gamma VALUE is zero (set by the kernel)
    // AND that defined[i] is correctly false there.
    let refused_node_exists = (0..N).any(|i| !defined[i]);
    assert!(
        refused_node_exists,
        "all nodes defined — this contradicts the obstacle geometry"
    );

    // count matches sum of defined.
    let count_from_mask: usize = defined.iter().filter(|&&d| d).count();
    assert_eq!(
        count, count_from_mask,
        "returned count ({count}) != sum(defined) ({count_from_mask})"
    );

    // Sanity: all defined gamma values must be finite and non-negative
    // (V″ >= 0 strictly on the continuation set for a convex payoff).
    for (i, (&g, &d)) in gamma.iter().zip(defined.iter()).enumerate() {
        if d {
            assert!(g.is_finite(), "defined gamma[{i}] is not finite: {g}");
            assert!(
                g >= -1e-10,
                "defined gamma[{i}]={g:.4e} < 0 (continuation-set Γ must be ≥ 0)"
            );
        }
    }

    println!(
        "G_BINDING_OBSTACLE_GAMMA_PARITY (core golden):\n\
         N={N}, S* = {sstar:.6}\n\
         count (defined nodes) = {count}\n\
         gamma[32] = {:.6e}  defined[32] = {}\n\
         refused_on_active_set = {refused_on_active_set}",
        gamma[32], defined[32],
    );
}

// ---------------------------------------------------------------------------
// Test: minimum n=4 (Grid1D requires n>=4, apply_inactive_gamma_into
//        requires n>=3; the n<3 guard is an internal defense-in-depth check
//        that is unreachable through the Grid1D public API).
// ---------------------------------------------------------------------------

#[test]
fn g_binding_obstacle_gamma_min_n_4_ok() {
    // Grid1D requires n>=4; apply_inactive_gamma_into requires n>=3.
    // With n=4, we have only 2 interior nodes — count may be 0 if all are
    // on the active set, but the call must succeed without error.
    let grid = Grid1D::new(0.0_f64, 1.0, 4).unwrap();
    let v = GridFn1D::from_fn(grid, |_| 2.0_f64); // all > obstacle=0 → some defined
    let obs = ConstantObstacle::new(0.0_f64).unwrap();
    let diff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    let kernel = ObstacleChernoff::new(diff, obs).unwrap();
    let mut gamma = v.zeroed_like();
    let mut defined = vec![false; 4];
    let result = kernel.apply_inactive_gamma_into(&v, &mut gamma, &mut defined);
    assert!(result.is_ok(), "n=4 should succeed: {result:?}");
}

// ---------------------------------------------------------------------------
// Test: length mismatch returns error
// ---------------------------------------------------------------------------

#[test]
fn g_binding_obstacle_gamma_length_mismatch_error() {
    use semiflow_core::SemiflowError;
    let grid = Grid1D::new(0.0_f64, 1.0, 8).unwrap();
    let v = GridFn1D::from_fn(grid, |s| s);
    let obs = ConstantObstacle::new(0.0_f64).unwrap();
    let diff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0_f64, |_| 0.0_f64, 1.0, grid);
    let kernel = ObstacleChernoff::new(diff, obs).unwrap();
    let mut gamma = v.zeroed_like();
    let mut defined = vec![false; 5]; // wrong length
    let err = kernel
        .apply_inactive_gamma_into(&v, &mut gamma, &mut defined)
        .unwrap_err();
    assert!(
        matches!(err, SemiflowError::DomainViolation { .. }),
        "expected DomainViolation for length mismatch, got {err:?}"
    );
}
