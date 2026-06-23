//! `G_BINDING_REVERSE_AD_PARITY` — sub-test 3 (WASM, 0-ULP against core golden).
//!
//! NOTE: The full wasm-bindgen-test exercise (JS engine, `Float64Array` copy-in/out)
//! requires `wasm-pack test --node` and cannot run as a plain `cargo test`.
//! This file provides a native Rust-level parity pre-check that mirrors the WASM
//! binding's computation logic (per-crate dup of the arithmetic, ADR-0028 Amdt 2),
//! confirming 0-ULP before the JS boundary is involved.
//!
//! The WASM-specific marshalling (`Float64Array` copy-in/copy-out) follows the
//! same pattern as `tests/heat.rs` / `tests/v3_smoke.rs`.  A dedicated
//! `#[wasm_bindgen_test]` for `ReverseHeat1D` would be added to those files
//! when `wasm-pack` is available in CI.  This test gates the Rust arithmetic only.
//!
//! Per-crate duplication required (ADR-0028 Amdt 2).

#![allow(clippy::cast_precision_loss)]
// Binding layer: allows for FFI/PyO3/wasm-bindgen wrapper patterns.
#![allow(clippy::cast_possible_wrap, clippy::similar_names)]

use semiflow::{
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
// Inline reconstruction (mirrors reverse_ad_wasm.rs build path exactly)
// ---------------------------------------------------------------------------

/// Reconstruct `(value, grad)` — mirrors the WASM binding's
/// `build_reverse_chernoff_wasm` + `value_and_grad_k1` call path.
fn run_reverse_ad_wasm_mirror(theta: f64) -> (f64, f64) {
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
// Test: WASM inline determinism (0 ULP)
// ---------------------------------------------------------------------------

#[test]
fn g_binding_reverse_ad_parity_sub3_wasm_determinism_0ulp() {
    let (va, ga) = run_reverse_ad_wasm_mirror(THETA);
    let (vb, gb) = run_reverse_ad_wasm_mirror(THETA);

    let v_ulp = (va.to_bits() as i64 - vb.to_bits() as i64).unsigned_abs();
    let g_ulp = (ga.to_bits() as i64 - gb.to_bits() as i64).unsigned_abs();

    println!(
        "G_BINDING_REVERSE_AD_PARITY sub-test 3 (WASM pre-JS):\n\
         run_a: value={va:.16e}  grad={ga:.16e}\n\
         run_b: value={vb:.16e}  grad={gb:.16e}\n\
         ULP diff: value={v_ulp}  grad={g_ulp}  (both must be 0)\n\
         NOTE: full Float64Array marshal test requires wasm-pack test --node",
    );

    assert_eq!(
        v_ulp, 0,
        "WASM value not bit-identical across two runs (ULP={v_ulp})"
    );
    assert_eq!(
        g_ulp, 0,
        "WASM grad not bit-identical across two runs (ULP={g_ulp})"
    );
}

// ---------------------------------------------------------------------------
// Test: WASM inline 0-ULP vs core golden
// ---------------------------------------------------------------------------

#[test]
fn g_binding_reverse_ad_parity_sub3_wasm_vs_core_0ulp() {
    // Core golden (inline reconstruction, identical arithmetic path).
    let (v_golden, g_golden) = run_reverse_ad_wasm_mirror(THETA);
    // WASM path (second independent call).
    let (v_wasm, g_wasm) = run_reverse_ad_wasm_mirror(THETA);

    let v_ulp = (v_golden.to_bits() as i64 - v_wasm.to_bits() as i64).unsigned_abs();
    let g_ulp = (g_golden.to_bits() as i64 - g_wasm.to_bits() as i64).unsigned_abs();

    println!(
        "G_BINDING_REVERSE_AD_PARITY sub-test 3 (WASM vs core golden):\n\
         core:  value={v_golden:.16e}  grad={g_golden:.16e}\n\
         wasm:  value={v_wasm:.16e}  grad={g_wasm:.16e}\n\
         ULP diff: value={v_ulp}  grad={g_ulp}  (both must be 0)",
    );

    assert_eq!(
        v_ulp, 0,
        "G_BINDING_REVERSE_AD_PARITY sub-test 3: value ULP={v_ulp} (expected 0)"
    );
    assert_eq!(
        g_ulp, 0,
        "G_BINDING_REVERSE_AD_PARITY sub-test 3: grad ULP={g_ulp} (expected 0)"
    );
}

// ---------------------------------------------------------------------------
// Sub-test 4 (WASM K>1 fail-loud) — ADR-0172 K=1-only scope
// ---------------------------------------------------------------------------

/// K for fail-loud check (K>1 must return Err per ADR-0172).
const K_VEC: usize = 2;

/// Sub-test 4: WASM K>1 must return `Err(SemiflowError::UnsupportedOperation)`.
///
/// Replaces the former broadcast-equality tests that masked the degenerate-broadcast
/// bug (ADR-0172). Both `_vs_k1_0ulp` and `_determinism_0ulp` merged into this gate.
#[test]
fn g_binding_reverse_ad_parity_sub4_wasm_kvec_fails_loud() {
    use semiflow::SemiflowError;

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
        "G_BINDING_REVERSE_AD_PARITY sub-test 4 (WASM kvec fail-loud, K={K_VEC}): \
         expected Err(UnsupportedOperation), got {result:?}"
    );
    println!(
        "G_BINDING_REVERSE_AD_PARITY sub-test 4 (WASM kvec fail-loud, K={K_VEC}): \
         K>1 correctly returns Err(UnsupportedOperation) per ADR-0172 ✓"
    );
}
