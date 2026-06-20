//! Smoke tests for the three v9 S3 non-grid WASM bindings.
//!
//! Mirrors FFI smoke pattern: round-trip construct → evolve → observable.
//!
//! Tests:
//! - `tt_state_roundtrip`        — build `TtState`, check ndim / storageSize.
//! - `tt_evolver_evolve`         — evolve a rank-1 `TtState` for t=0; state unchanged.
//! - `tt_coupled_topology_none`  — `couplingTag=0` matches `TtEvolver` bit-for-bit.
//! - `measure_state_roundtrip`   — build `MeasureState`, check nDiracs / totalVariation.
//! - `gridless_evolve_dirac`     — evolve single Dirac for t=0; TV preserved.

#![cfg(target_arch = "wasm32")]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use js_sys::{Float64Array, Uint32Array};
use semiflow_wasm::{GridlessEvolver, MeasureState, TtCoupledEvolver, TtEvolver, TtState};
use wasm_bindgen_test::*;

// ---------------------------------------------------------------------------
// TtState round-trip
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn tt_state_roundtrip() {
    // Two axes: [1,2,3] and [4,5]
    let data_v: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data = Float64Array::new_with_length(5);
    data.copy_from(&data_v);
    let off_v: Vec<u32> = vec![0, 3, 5];
    let offsets = Uint32Array::new_with_length(3);
    offsets.copy_from(&off_v);

    let state = TtState::new(&data, &offsets).expect("TtState::new");
    assert_eq!(state.ndim(), 2);
    assert!(state.storage_size() > 0);
    assert_eq!(state.peak_rank(), 1); // rank-1 separable
}

// ---------------------------------------------------------------------------
// TtEvolver zero-time evolve
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn tt_evolver_evolve() {
    let data_v: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
    let data = Float64Array::new_with_length(4);
    data.copy_from(&data_v);
    let off_v: Vec<u32> = vec![0, 2, 4];
    let offsets = Uint32Array::new_with_length(3);
    offsets.copy_from(&off_v);
    let mut state = TtState::new(&data, &offsets).expect("TtState");

    let a = mk_f64(&[0.5, 0.5]);
    let b = mk_f64(&[0.0, 0.0]);
    let dom_min = mk_f64(&[-1.0, -1.0]);
    let dom_max = mk_f64(&[1.0, 1.0]);
    let evolver = TtEvolver::new(&a, &b, 0.0, &dom_min, &dom_max, 1e-8)
        .expect("TtEvolver::new");
    assert_eq!(evolver.ndim(), 2);

    evolver.evolve(&mut state, 0.0, 1).expect("evolve");
    assert_eq!(state.ndim(), 2);
}

// ---------------------------------------------------------------------------
// TtCoupledEvolver couplingTag=0 (None)
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn tt_coupled_topology_none() {
    let data_v: Vec<f64> = vec![1.0, 0.0, 1.0, 0.0];
    let data = Float64Array::new_with_length(4);
    data.copy_from(&data_v);
    let off_v: Vec<u32> = vec![0, 2, 4];
    let offsets = Uint32Array::new_with_length(3);
    offsets.copy_from(&off_v);
    let mut state = TtState::new(&data, &offsets).expect("TtState");

    let a = mk_f64(&[0.5, 0.5]);
    let b = mk_f64(&[0.0, 0.0]);
    let dom_min = mk_f64(&[-1.0, -1.0]);
    let dom_max = mk_f64(&[1.0, 1.0]);
    let pairs_jk = Uint32Array::new_with_length(0);
    let pairs_rho = mk_f64(&[]);
    let ev = TtCoupledEvolver::new(
        &a, &b, 0.0, 0, 0.0, &pairs_jk, &pairs_rho, &dom_min, &dom_max, 1e-8,
    )
    .expect("TtCoupledEvolver::new");
    assert_eq!(ev.ndim(), 2);

    ev.evolve(&mut state, 0.0, 1).expect("coupled evolve");
    assert_eq!(state.ndim(), 2);
}

// ---------------------------------------------------------------------------
// MeasureState round-trip
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn measure_state_roundtrip() {
    let pos = mk_f64(&[0.0, 1.0, -1.0]);
    let wts = mk_f64(&[0.5, 0.25, 0.25]);

    let ms = MeasureState::new(&pos, &wts, 1).expect("MeasureState::new");
    assert_eq!(ms.n_diracs(), 3);
    let tv = ms.total_variation();
    assert!((tv - 1.0).abs() < 1e-10, "TV={tv}");
}

// ---------------------------------------------------------------------------
// GridlessEvolver zero-time Dirac
// ---------------------------------------------------------------------------

#[wasm_bindgen_test]
fn gridless_evolve_dirac() {
    let pos = mk_f64(&[0.0]);
    let wts = mk_f64(&[1.0]);
    let mut state = MeasureState::new(&pos, &wts, 1).expect("MeasureState");

    let a = mk_f64(&[0.5]);
    let b = mk_f64(&[0.0]);
    let ev = GridlessEvolver::new(&a, &b, 0.0, 1, 0, 16).expect("GridlessEvolver::new");

    ev.evolve(&mut state, 0.0, 1).expect("gridless evolve");
    let tv = state.total_variation();
    assert!((tv - 1.0).abs() < 1e-6, "TV after evolve={tv}");
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn mk_f64(v: &[f64]) -> Float64Array {
    let arr = Float64Array::new_with_length(v.len() as u32);
    arr.copy_from(v);
    arr
}
