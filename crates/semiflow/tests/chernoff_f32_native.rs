//! Smoke tests for the Phase 5a native `impl ChernoffFunction<f32>` on the 7 leaf kernels.
//!
//! Each test constructs the kernel directly as `ChernoffFunction<f32>` (no shim),
//! applies one step, and verifies: output length matches input, all values are finite.
//!
//! ADR-0175 (Phase 5a): additive impls delegate to the generic scalar `apply_f` path.

use semiflow_core::{
    chernoff::ApplyChernoffExt,
    diffusion::DiffusionChernoff,
    diffusion4::Diffusion4thChernoff,
    diffusion6::Diffusion6thChernoff,
    drift_reaction::DriftReactionChernoff,
    shift1d::ShiftChernoff1D,
    truncated_exp::TruncatedExpDiffusionChernoff,
    truncated_exp4::TruncatedExp4thDiffusionChernoff,
    Grid1D, GridFn1D,
};

const N: usize = 32;
const TAU: f32 = 0.01_f32;
const A_NORM: f64 = 0.25;

fn grid_f32() -> Grid1D<f32> {
    Grid1D::<f32>::new_generic(-3.0_f32, 3.0_f32, N).expect("Grid1D f32")
}

fn gaussian_f32(g: Grid1D<f32>) -> GridFn1D<f32> {
    GridFn1D::<f32>::from_fn_generic(g, |x| (-x * x).exp())
}

fn assert_finite_f32(values: &[f32], label: &str) {
    for (i, &v) in values.iter().enumerate() {
        assert!(v.is_finite(), "{label}: value[{i}] = {v} is not finite");
    }
}

/// Native `ChernoffFunction<f32>` impl on `DiffusionChernoff<f32>`.
#[test]
fn diffusion_chernoff_f32_native() {
    let g = grid_f32();
    let k = DiffusionChernoff::<f32>::new(
        |_| 0.5_f32, |_| 0.0_f32, |_| 0.0_f32, A_NORM, g,
    );
    let f = gaussian_f32(g);
    let out = k.apply_chernoff(TAU, &f).expect("DiffusionChernoff<f32> apply");
    assert_eq!(out.values.len(), N, "output length mismatch");
    assert_finite_f32(&out.values, "DiffusionChernoff<f32>");
}

/// Native `ChernoffFunction<f32>` impl on `Diffusion4thChernoff<f32>`.
#[test]
fn diffusion4_chernoff_f32_native() {
    let g = grid_f32();
    let k = Diffusion4thChernoff::<f32>::new_generic(
        |_| 0.5_f32, |_| 0.0_f32, |_| 0.0_f32, A_NORM, g,
    );
    let f = gaussian_f32(g);
    let out = k.apply_chernoff(TAU, &f).expect("Diffusion4thChernoff<f32> apply");
    assert_eq!(out.values.len(), N, "output length mismatch");
    assert_finite_f32(&out.values, "Diffusion4thChernoff<f32>");
}

/// Native `ChernoffFunction<f32>` impl on `Diffusion6thChernoff<f32>`.
#[test]
fn diffusion6_chernoff_f32_native() {
    let g = grid_f32();
    let k = Diffusion6thChernoff::<f32>::new_generic(
        |_| 0.5_f32, |_| 0.0_f32, |_| 0.0_f32, A_NORM, g,
    );
    let f = gaussian_f32(g);
    let out = k.apply_chernoff(TAU, &f).expect("Diffusion6thChernoff<f32> apply");
    assert_eq!(out.values.len(), N, "output length mismatch");
    assert_finite_f32(&out.values, "Diffusion6thChernoff<f32>");
}

/// Native `ChernoffFunction<f32>` impl on `DriftReactionChernoff<f32>`.
#[test]
fn drift_reaction_chernoff_f32_native() {
    let g = grid_f32();
    let k = DriftReactionChernoff::<f32>::new_generic(
        |_| 0.1_f32, |_| 0.0_f32, 0.0, g,
    );
    let f = gaussian_f32(g);
    let out = k.apply_chernoff(TAU, &f).expect("DriftReactionChernoff<f32> apply");
    assert_eq!(out.values.len(), N, "output length mismatch");
    assert_finite_f32(&out.values, "DriftReactionChernoff<f32>");
}

/// Native `ChernoffFunction<f32>` impl on `TruncatedExpDiffusionChernoff<f32>`.
#[test]
fn truncated_exp_chernoff_f32_native() {
    let g = grid_f32();
    let k = TruncatedExpDiffusionChernoff::<f32>::new_generic(
        |_| 0.25_f32, |_| 0.0_f32, |_| 0.0_f32, A_NORM, g,
    );
    let f = gaussian_f32(g);
    let out = k.apply_chernoff(TAU, &f).expect("TruncatedExpDiffusionChernoff<f32> apply");
    assert_eq!(out.values.len(), N, "output length mismatch");
    assert_finite_f32(&out.values, "TruncatedExpDiffusionChernoff<f32>");
}

/// Native `ChernoffFunction<f32>` impl on `TruncatedExp4thDiffusionChernoff<f32>`.
#[test]
fn truncated_exp4_chernoff_f32_native() {
    let g = grid_f32();
    let k = TruncatedExp4thDiffusionChernoff::<f32>::new_generic(
        |_| 0.25_f32, |_| 0.0_f32, |_| 0.0_f32, A_NORM, g,
    );
    let f = gaussian_f32(g);
    let out = k.apply_chernoff(TAU, &f).expect("TruncatedExp4thDiffusionChernoff<f32> apply");
    assert_eq!(out.values.len(), N, "output length mismatch");
    assert_finite_f32(&out.values, "TruncatedExp4thDiffusionChernoff<f32>");
}

/// Native `ChernoffFunction<f32>` impl on `ShiftChernoff1D<f32>`.
#[test]
fn shift_chernoff1d_f32_native() {
    let g = grid_f32();
    let k = ShiftChernoff1D::<f32>::new_generic(
        |_| 0.5_f32, |_| 0.1_f32, |_| 0.0_f32, 0.0, g,
    );
    let f = gaussian_f32(g);
    let out = k.apply_chernoff(TAU, &f).expect("ShiftChernoff1D<f32> apply");
    assert_eq!(out.values.len(), N, "output length mismatch");
    assert_finite_f32(&out.values, "ShiftChernoff1D<f32>");
}
