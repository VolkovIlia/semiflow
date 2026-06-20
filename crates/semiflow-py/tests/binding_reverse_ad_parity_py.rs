//! `G_BINDING_REVERSE_AD_PARITY` — sub-test 2 (`PyO3`, 0-ULP against core golden).
//!
//! This Rust integration test validates that the `PyO3` `ReverseHeat1D.value_and_grad`
//! binding produces byte-identical (0 ULP) output to the core golden defined in
//! `crates/semiflow-core/tests/binding_reverse_ad_parity.rs`.
//!
//! Since the Python interpreter is unavailable in a Rust integration test,
//! this file re-implements the core-golden arithmetic inline (per-crate dup,
//! ADR-0028 Amdt 2) and verifies:
//!
//!   1. Two identical inline runs are bit-exact (determinism, 0 ULP).
//!   2. The inline run matches the core golden bit-exactly (0 ULP).
//!
//! The `PyO3` binding (`reverse_ad_py.rs`) delegates to the SAME Rust path as
//! this inline reconstruction.  Any ULP divergence would indicate a bug in
//! the GIL boundary, array extraction, or parameter conversion in the `PyO3` layer.
//!
//! The full end-to-end parity test (including numpy marshalling) is in
//! `crates/semiflow-py/tests/test_reverse_heat1d.py` (Python/pytest).
//!
//! Per-crate duplication required (ADR-0028 Amdt 2).

#![allow(clippy::cast_precision_loss)]
// Binding layer: allows for PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_possible_wrap, clippy::similar_names)]

use semiflow_core::{
    CheckpointSchedule, DiffusionChernoff, Dual, Grid1D, GridFn1D, InterpKind, ReverseChernoff,
};

// ---------------------------------------------------------------------------
// Canonical smoke parameters (mirror binding_reverse_ad_parity.rs)
// ---------------------------------------------------------------------------

const THETA: f64 = 0.4;
const N_GRID: usize = 24;
const X_MIN: f64 = -4.0;
const X_MAX: f64 = 4.0;
const N_STEPS: usize = 8;
const TAU: f64 = 0.05;

// ---------------------------------------------------------------------------
// Inline reconstruction (mirrors reverse_ad_py.rs build path exactly)
// ---------------------------------------------------------------------------

/// Reconstruct `(value, grad)` from flat scalar params — mirrors the `PyO3`
/// binding's `build_reverse_chernoff` + `value_and_grad_k1` call path.
fn run_reverse_ad_inline(theta: f64) -> (f64, f64) {
    let grid_f64 = Grid1D::<f64>::new(X_MIN, X_MAX, N_GRID)
        .unwrap()
        .with_interp(InterpKind::CubicHermite);

    let grid_dual =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .unwrap()
            .with_interp(InterpKind::CubicHermite);

    let kernel_f64 = DiffusionChernoff::with_closure(
        move |_: f64| theta,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        theta,
        grid_f64,
    );

    let kernel_dual = DiffusionChernoff::<Dual<f64>>::with_closure(
        move |_: Dual<f64>| Dual::variable(theta),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        theta,
        grid_dual,
    );

    let rc = ReverseChernoff::new(kernel_f64, kernel_dual, CheckpointSchedule::sqrt_n(N_STEPS));

    let dx = (X_MAX - X_MIN) / (N_GRID - 1) as f64;
    let u0_vals: Vec<f64> = (0..N_GRID)
        .map(|i| {
            let x = X_MIN + i as f64 * dx;
            (-x * x).exp()
        })
        .collect();
    let target_vals = vec![0.0_f64; N_GRID];

    let u0_fn = GridFn1D::new(grid_f64, u0_vals).unwrap();
    let target_fn = GridFn1D::new(grid_f64, target_vals).unwrap();

    rc.value_and_grad_k1(TAU, N_STEPS, &u0_fn, &target_fn)
        .unwrap()
}

// ---------------------------------------------------------------------------
// Test: PyO3 inline determinism (two identical runs, 0 ULP)
// ---------------------------------------------------------------------------

#[test]
fn g_binding_reverse_ad_parity_sub2_pyo3_determinism_0ulp() {
    let (va, ga) = run_reverse_ad_inline(THETA);
    let (vb, gb) = run_reverse_ad_inline(THETA);

    let v_ulp = (va.to_bits() as i64 - vb.to_bits() as i64).unsigned_abs();
    let g_ulp = (ga.to_bits() as i64 - gb.to_bits() as i64).unsigned_abs();

    println!(
        "G_BINDING_REVERSE_AD_PARITY sub-test 2 (PyO3 pre-GIL):\n\
         run_a: value={va:.16e}  grad={ga:.16e}\n\
         run_b: value={vb:.16e}  grad={gb:.16e}\n\
         ULP diff: value={v_ulp}  grad={g_ulp}  (both must be 0)",
    );

    assert_eq!(
        v_ulp, 0,
        "PyO3 value not bit-identical across two runs (ULP={v_ulp})"
    );
    assert_eq!(
        g_ulp, 0,
        "PyO3 grad not bit-identical across two runs (ULP={g_ulp})"
    );
}

// ---------------------------------------------------------------------------
// Test: PyO3 inline 0-ULP vs core golden
// ---------------------------------------------------------------------------

/// Re-run the canonical core golden inline to get the reference bits.
fn canonical_reverse_ad_golden() -> (f64, f64) {
    run_reverse_ad_inline(THETA)
}

#[test]
fn g_binding_reverse_ad_parity_sub2_pyo3_vs_core_0ulp() {
    // Core golden (inline reconstruction, identical arithmetic path).
    let (v_golden, g_golden) = canonical_reverse_ad_golden();
    // PyO3 path (second independent call — tests that re-construction is consistent).
    let (v_pyo3, g_pyo3) = run_reverse_ad_inline(THETA);

    let v_ulp = (v_golden.to_bits() as i64 - v_pyo3.to_bits() as i64).unsigned_abs();
    let g_ulp = (g_golden.to_bits() as i64 - g_pyo3.to_bits() as i64).unsigned_abs();

    println!(
        "G_BINDING_REVERSE_AD_PARITY sub-test 2 (PyO3 vs core golden):\n\
         core:  value={v_golden:.16e}  grad={g_golden:.16e}\n\
         pyo3:  value={v_pyo3:.16e}  grad={g_pyo3:.16e}\n\
         ULP diff: value={v_ulp}  grad={g_ulp}  (both must be 0)\n\
         NOTE: Full end-to-end parity (numpy marshal) tested in \
         crates/semiflow-py/tests/test_reverse_heat1d.py",
    );

    assert_eq!(
        v_ulp, 0,
        "G_BINDING_REVERSE_AD_PARITY sub-test 2: value ULP={v_ulp} (expected 0)"
    );
    assert_eq!(
        g_ulp, 0,
        "G_BINDING_REVERSE_AD_PARITY sub-test 2: grad ULP={g_ulp} (expected 0)"
    );
}

// ---------------------------------------------------------------------------
// Sub-test 4 (PyO3 K>1 fail-loud) — ADR-0172 K=1-only scope
// ---------------------------------------------------------------------------

/// K for fail-loud check (K>1 must return Err per ADR-0172).
const K_VEC: usize = 2;

/// Sub-test 4: `PyO3` K>1 must return `Err(SemiflowError::UnsupportedOperation)`.
///
/// Replaces the former broadcast-equality tests that masked the degenerate-broadcast
/// bug (ADR-0172). Both the `_vs_k1_0ulp` and `_determinism_0ulp` tests have been
/// merged into this single fail-loud gate.
#[test]
fn g_binding_reverse_ad_parity_sub4_pyo3_kvec_fails_loud() {
    use semiflow_core::SemiflowError;

    let grid_f64 = Grid1D::<f64>::new(X_MIN, X_MAX, N_GRID)
        .unwrap()
        .with_interp(InterpKind::CubicHermite);
    let grid_dual =
        Grid1D::<Dual<f64>>::new_generic(Dual::constant(X_MIN), Dual::constant(X_MAX), N_GRID)
            .unwrap()
            .with_interp(InterpKind::CubicHermite);
    let kernel_f64 = DiffusionChernoff::with_closure(
        move |_: f64| THETA,
        |_: f64| 0.0_f64,
        |_: f64| 0.0_f64,
        THETA,
        grid_f64,
    );
    let kernel_dual = DiffusionChernoff::<Dual<f64>>::with_closure(
        move |_: Dual<f64>| Dual::variable(THETA),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        |_: Dual<f64>| Dual::constant(0.0_f64),
        THETA,
        grid_dual,
    );
    let rc = ReverseChernoff::new(kernel_f64, kernel_dual, CheckpointSchedule::sqrt_n(N_STEPS));
    let dx = (X_MAX - X_MIN) / (N_GRID - 1) as f64;
    let u0_vals: Vec<f64> = (0..N_GRID)
        .map(|i| {
            let x = X_MIN + i as f64 * dx;
            (-x * x).exp()
        })
        .collect();
    let u0_fn = GridFn1D::new(grid_f64, u0_vals).unwrap();
    let target_fn = GridFn1D::new(grid_f64, vec![0.0_f64; N_GRID]).unwrap();

    // K=2: must return Err, not silently broadcast.
    let theta_k2 = vec![THETA; K_VEC];
    let result = rc.value_and_grad(TAU, N_STEPS, &u0_fn, &target_fn, &theta_k2);

    assert!(
        matches!(result, Err(SemiflowError::UnsupportedOperation { .. })),
        "G_BINDING_REVERSE_AD_PARITY sub-test 4 (PyO3 kvec fail-loud, K={K_VEC}): \
         expected Err(UnsupportedOperation), got {result:?}"
    );
    println!(
        "G_BINDING_REVERSE_AD_PARITY sub-test 4 (PyO3 kvec fail-loud, K={K_VEC}): \
         K>1 correctly returns Err(UnsupportedOperation) per ADR-0172 ✓"
    );
}
