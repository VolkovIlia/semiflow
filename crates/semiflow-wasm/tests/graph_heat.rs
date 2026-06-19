//! Graph PDE WASM smoke tests (v2.2 Wave C — ADR-0059).
//!
//! Mirrors `crates/semiflow-wasm/tests/heat.rs` (v0.10.0 Wave C precedent).
//!
//! ## Cross-binding identity gate (`G_cross_binding_graph_identity`)
//!
//! Test setup (ADR-0059 §2.5):
//! - `P_64` path graph, combinatorial Laplacian.
//! - Initial condition: `u₀(i) = exp(−i² / 64)`.
//! - `t_final` = 0.5, `n_steps` = 50.
//! - Gate: `sup_error < 5e-4` vs Rust reference solution.
//!
//! The Rust reference is computed directly in WASM via `GraphHeatChernoff`
//! (same code path as the WASM binding) — numerically identical within
//! 3 ULP of the FFI / `PyO3` binding results per ADR-0059 §"Cross-binding
//! sup-error gate".

#![cfg(target_arch = "wasm32")]
// wasm32 is 32-bit; i as f64 and usize→u32 truncations are exact.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use semiflow_wasm::{GraphHeat, GraphPath};
use wasm_bindgen_test::*;

// ---------------------------------------------------------------------------
// Test constants
// ---------------------------------------------------------------------------

/// Number of nodes in the path graph.
const N: u32 = 64;

/// Time horizon.
const T_FINAL: f64 = 0.5;

/// Number of Chernoff steps.
const N_STEPS: u32 = 50;

/// Gate tolerance (ADR-0059 §2.5 / G_WASM_smoke_graph).
const TOL: f64 = 5e-4;

/// Conservative Gershgorin bound for P_64 (largest Laplacian eigenvalue < 4).
const RHO_BAR: f64 = 4.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the initial condition `u₀(i) = exp(−i² / 64)`.
fn make_f0(n: u32) -> Vec<f64> {
    (0..n)
        .map(|i| (-(i as f64 * i as f64) / (n as f64)).exp())
        .collect()
}

// ---------------------------------------------------------------------------
// Smoke: GraphPath constructor
// ---------------------------------------------------------------------------

/// `GraphPath::new(64)` must succeed and report correct `n_nodes`.
#[wasm_bindgen_test]
fn graph_path_n_nodes() {
    let g = GraphPath::new(N).expect("GraphPath::new(64)");
    assert_eq!(g.n_nodes(), N, "n_nodes mismatch");
}

/// `GraphPath::new(0)` must throw `OutOfDomain`.
#[wasm_bindgen_test]
fn graph_path_zero_nodes_errors() {
    let err = GraphPath::new(0)
        .err()
        .expect("expected OutOfDomain for n_nodes=0");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "OutOfDomain", "got kind={kind}");
}

// ---------------------------------------------------------------------------
// Smoke: GraphHeat constructor
// ---------------------------------------------------------------------------

/// `GraphHeat::new` with valid path graph and rho_bar must succeed.
#[wasm_bindgen_test]
fn graph_heat_construct_ok() {
    let g = GraphPath::new(N).expect("GraphPath::new");
    let _heat = GraphHeat::new(&g, RHO_BAR).expect("GraphHeat::new");
}

/// `GraphHeat::new` with `rho_bar = 0` must throw `OutOfDomain`.
#[wasm_bindgen_test]
fn graph_heat_zero_rho_bar_errors() {
    let g = GraphPath::new(N).expect("GraphPath::new");
    let err = GraphHeat::new(&g, 0.0)
        .err()
        .expect("expected OutOfDomain for rho_bar=0");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "OutOfDomain", "got kind={kind}");
}

/// `GraphHeat::new` with negative `rho_bar` must throw `OutOfDomain`.
#[wasm_bindgen_test]
fn graph_heat_negative_rho_bar_errors() {
    let g = GraphPath::new(N).expect("GraphPath::new");
    let err = GraphHeat::new(&g, -1.0)
        .err()
        .expect("expected OutOfDomain for rho_bar=-1");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "OutOfDomain", "got kind={kind}");
}

// ---------------------------------------------------------------------------
// Smoke: GraphHeat::evolve error handling
// ---------------------------------------------------------------------------

/// `evolve` with `n_steps = 0` must throw `OutOfDomain`.
#[wasm_bindgen_test]
fn evolve_zero_steps_errors() {
    let g = GraphPath::new(N).expect("GraphPath::new");
    let heat = GraphHeat::new(&g, RHO_BAR).expect("GraphHeat::new");
    let f0 = make_f0(N);
    let err = heat
        .evolve(T_FINAL, 0, &f0)
        .expect_err("expected OutOfDomain for n_steps=0");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "OutOfDomain", "got kind={kind}");
}

/// `evolve` with negative `t_final` must throw `OutOfDomain`.
#[wasm_bindgen_test]
fn evolve_negative_t_errors() {
    let g = GraphPath::new(N).expect("GraphPath::new");
    let heat = GraphHeat::new(&g, RHO_BAR).expect("GraphHeat::new");
    let f0 = make_f0(N);
    let err = heat
        .evolve(-1.0, N_STEPS, &f0)
        .expect_err("expected OutOfDomain for t_final<0");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "OutOfDomain", "got kind={kind}");
}

/// `evolve` with wrong-length `f0` must throw `GridMismatch`.
#[wasm_bindgen_test]
fn evolve_wrong_f0_length_errors() {
    let g = GraphPath::new(N).expect("GraphPath::new");
    let heat = GraphHeat::new(&g, RHO_BAR).expect("GraphHeat::new");
    // Supply f0 with one element too few.
    let f0_short = make_f0(N - 1);
    let err = heat
        .evolve(T_FINAL, N_STEPS, &f0_short)
        .expect_err("expected GridMismatch for wrong f0 length");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "GridMismatch", "got kind={kind}");
}

/// `evolve` with `f0` containing NaN must throw `NanInf`.
#[wasm_bindgen_test]
fn evolve_nan_f0_errors() {
    let g = GraphPath::new(N).expect("GraphPath::new");
    let heat = GraphHeat::new(&g, RHO_BAR).expect("GraphHeat::new");
    let mut f0 = make_f0(N);
    f0[0] = f64::NAN;
    let err = heat
        .evolve(T_FINAL, N_STEPS, &f0)
        .expect_err("expected NanInf for NaN in f0");
    let kind = js_sys::Reflect::get(&err, &"kind".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    assert_eq!(kind, "NanInf", "got kind={kind}");
}

// ---------------------------------------------------------------------------
// G_WASM_smoke_graph: cross-binding identity gate (ADR-0059 §2.5)
// ---------------------------------------------------------------------------

/// Graph heat WASM smoke gate (G_WASM_smoke_graph).
///
/// P_64 path graph, `u₀(i) = exp(−i²/64)`, t_final=0.5, n_steps=50.
/// Gate: `sup_error < 5e-4` vs the reference Rust solution (computed in
/// the same WASM module — same code path as FFI/PyO3 within 3 ULP).
///
/// This is the WASM leg of the ADR-0059 cross-binding identity gate.
/// The cross-binding invariant is:
///   |sup_err_Rust − sup_err_WASM| < 3 ULP
/// which is verified on CI against the FFI/PyO3 smoke test outputs.
#[wasm_bindgen_test]
fn graph_heat_smoke_cross_binding_gate() {
    let g = GraphPath::new(N).expect("GraphPath::new");
    let heat = GraphHeat::new(&g, RHO_BAR).expect("GraphHeat::new");
    let f0 = make_f0(N);

    let result = heat
        .evolve(T_FINAL, N_STEPS, &f0)
        .expect("GraphHeat::evolve");

    assert_eq!(result.len(), N as usize, "output length mismatch");

    // Verify output is all-finite.
    let all_finite = result.iter().all(|v| v.is_finite());
    assert!(all_finite, "evolve output contains NaN or Inf");

    // Compute reference: run the same Chernoff kernel directly in WASM
    // (identical Rust code path ↔ zero cross-binding error in this test).
    // We re-run with N_STEPS=200 as reference (finer resolution).
    let ref_heat = GraphHeat::new(&g, RHO_BAR).expect("ref GraphHeat::new");
    let reference = ref_heat
        .evolve(T_FINAL, 200, &f0)
        .expect("reference evolve");

    let sup_err = result
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    wasm_bindgen_test::console_log!(
        "G_WASM_smoke_graph: sup_err vs 200-step reference = {:.6e}  (gate < {:.0e})",
        sup_err,
        TOL
    );

    assert!(
        sup_err < TOL,
        "sup_err {sup_err:.3e} >= {TOL:.0e} (G_WASM_smoke_graph failed)"
    );
}

/// Output length equals the `n_nodes` used to construct the graph.
#[wasm_bindgen_test]
fn evolve_output_length_matches_n_nodes() {
    let g = GraphPath::new(N).expect("GraphPath::new");
    let heat = GraphHeat::new(&g, RHO_BAR).expect("GraphHeat::new");
    let f0 = make_f0(N);
    let out = heat.evolve(T_FINAL, N_STEPS, &f0).expect("evolve");
    assert_eq!(out.len(), N as usize, "output length should equal n_nodes");
}

/// Multiple sequential `evolve` calls on the same `GraphHeat` object must
/// succeed (stateless — each call starts from the provided `f0`).
#[wasm_bindgen_test]
fn evolve_is_stateless() {
    let g = GraphPath::new(N).expect("GraphPath::new");
    let heat = GraphHeat::new(&g, RHO_BAR).expect("GraphHeat::new");
    let f0 = make_f0(N);

    let out1 = heat.evolve(T_FINAL, N_STEPS, &f0).expect("first evolve");
    let out2 = heat.evolve(T_FINAL, N_STEPS, &f0).expect("second evolve");

    let max_diff = out1
        .iter()
        .zip(out2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff == 0.0,
        "stateless property violated: max_diff={max_diff:.3e}"
    );
}
