//! `G_BINDING_OBSTACLE_GAMMA_PARITY` — sub-test 2 (`PyO3` logic pre-check, Rust-level).
//!
//! This file mirrors the pattern of `binding_wentzell_parity_py.rs`:
//! the `PyO3` binding calls the SAME Rust path as the core golden — any ULP
//! divergence would indicate a bug in the data-cloning or marshalling logic.
//! Since the Python interpreter is unavailable in Rust integration tests, this
//! file re-implements the core-golden arithmetic inline and verifies 0-ULP
//! self-consistency.
//!
//! The full end-to-end parity test (numpy dtype, mask semantics, count equality)
//! is in `crates/semiflow-py/tests/test_obstacle_gamma_v8.py`.
//!
//! Per-crate duplication required (ADR-0028 Amdt 2).

#![allow(clippy::cast_precision_loss)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_possible_wrap)]

use semiflow_core::{ClosureObstacle, DiffusionChernoff, Grid1D, GridFn1D, ObstacleChernoff};

// ---------------------------------------------------------------------------
// Canonical parameters (mirror binding_obstacle_gamma_parity.rs)
// ---------------------------------------------------------------------------

const S_MIN: f64 = 0.0;
const S_MAX: f64 = 3.0;
const N: usize = 64;
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

fn pap_v(s: f64) -> f64 {
    let sstar = pap_sstar();
    if s > sstar {
        pap_a_coef() * s.powf(-pap_gamma_pow())
    } else {
        PAP_K - s
    }
}

// ---------------------------------------------------------------------------
// Inline gamma sweep (mirrors obstacle_gamma_py.rs PyO3 path)
// ---------------------------------------------------------------------------

fn run_gamma_inline() -> (Vec<f64>, Vec<bool>, usize) {
    let grid = Grid1D::new(S_MIN, S_MAX, N).unwrap();
    let v = GridFn1D::from_fn(grid, pap_v);
    let obs = ClosureObstacle::new(|s: f64| PAP_K - s);
    let diff = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0_f64, |_| 0.0_f64, 0.5, grid);
    let kernel = ObstacleChernoff::new(diff, obs).unwrap();
    let mut gamma = v.zeroed_like();
    let mut defined = vec![false; N];
    let count = kernel
        .apply_inactive_gamma_into(&v, &mut gamma, &mut defined)
        .unwrap();
    (gamma.values, defined, count)
}

// ---------------------------------------------------------------------------
// Test: 0-ULP between two identical inline runs (determinism check)
// ---------------------------------------------------------------------------

#[test]
fn g_binding_obstacle_gamma_parity_sub2_pyo3_precheck_0ulp() {
    // Two independent runs — must be bit-identical (determinism).
    let (gamma_a, defined_a, count_a) = run_gamma_inline();
    let (gamma_b, defined_b, count_b) = run_gamma_inline();

    // Count must match.
    assert_eq!(
        count_a, count_b,
        "count mismatch between two identical runs"
    );

    // Defined masks must match exactly.
    assert_eq!(
        defined_a, defined_b,
        "defined mask mismatch between two identical runs"
    );

    // Gamma values must be bit-identical.
    let max_ulp = gamma_a
        .iter()
        .zip(gamma_b.iter())
        .map(|(&a, &b)| {
            let ai = a.to_bits() as i64;
            let bi = b.to_bits() as i64;
            ai.wrapping_sub(bi).unsigned_abs()
        })
        .max()
        .unwrap_or(0);

    println!(
        "G_BINDING_OBSTACLE_GAMMA_PARITY sub-test 2 (PyO3 pre-GIL):\n\
         count = {count_a}\n\
         gamma_a[32] = {:.16e}  gamma_b[32] = {:.16e}\n\
         defined_a[32] = {}  defined_b[32] = {}\n\
         max ULP between two runs = {max_ulp}  (expected 0)",
        gamma_a[32], gamma_b[32], defined_a[32], defined_b[32],
    );

    assert_eq!(
        max_ulp, 0,
        "gamma values must be bit-identical across two runs"
    );
}

// ---------------------------------------------------------------------------
// Test: honesty invariant — at least one refused node on active set
// ---------------------------------------------------------------------------

#[test]
fn g_binding_obstacle_gamma_parity_sub2_refused_node_on_active_set() {
    let (_, defined, _) = run_gamma_inline();
    let grid = Grid1D::new(S_MIN, S_MAX, N).unwrap();
    let sstar = pap_sstar();

    let refused_on_active = (0..N).any(|i| {
        let s = grid.x_at(i);
        s <= sstar && !defined[i]
    });

    assert!(
        refused_on_active,
        "G_BINDING_OBSTACLE_GAMMA_PARITY sub-test 2: no refused node on active set \
         (S <= S*={sstar:.4}) — mask may be collapsed!"
    );
}
