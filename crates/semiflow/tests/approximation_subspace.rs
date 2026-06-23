//! `G_AS_K` — gate tests for `ApproximationSubspace<const K, F>` impls (v3.0, ADR-0073).
//!
//! Gate name: `G_AS_K`.
//! Classification: **RELEASE-BLOCKING** (feature = slow-tests).
//!
//! ## What is tested
//!
//! Three impls shipped in `src/approximation.rs`:
//!
//! - K=2 for `DiffusionChernoff<f64>` (ζ-A kernel)
//! - K=4 for `Diffusion4thChernoff<f64>` (ζ⁴ kernel, ADR-0013)
//! - K=6 for `TruncatedExp4thDiffusionChernoff<f64>` (truncated-exp kernel,
//!   ADR-0011 naming; K=6 subspace per math.md §27)
//!
//! Each K gate checks:
//! 1. `in_subspace` returns `true` for a smooth Gaussian datum on a sufficiently
//!    large grid.
//! 2. `in_subspace` returns `false` when the grid is too coarse.
//! 3. `jet` returns a slice of length K+1 with finite values.
//! 4. `jet` returns `DomainViolation` when `out.len() != K+1`.
//! 5. `assert_in_subspace` helper round-trips the predicate correctly.
//!
//! ## Guards
//!
//! K=6 uses a large (N=64) grid to satisfy the ≥13-point precondition for all
//! jet iterations. All tests run only under `--features slow-tests` to keep
//! the default test profile snappy.

#![cfg(feature = "slow-tests")]

use semiflow_core::{
    approximation::{assert_in_subspace, ApproximationSubspace},
    Diffusion4thChernoff, DiffusionChernoff, Grid1D, GridFn1D, TruncatedExp4thDiffusionChernoff,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Gaussian u0 on [-4, 4] with N grid points.
fn gaussian(n: usize) -> (Grid1D<f64>, GridFn1D<f64>) {
    let grid = Grid1D::new(-4.0, 4.0, n).expect("grid");
    let f = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    (grid, f)
}

/// Zero-allocation output slice of K+1 clones of `proto`.
fn alloc_jet_out(proto: &GridFn1D<f64>, len: usize) -> Vec<GridFn1D<f64>> {
    vec![proto.clone(); len]
}

// ---------------------------------------------------------------------------
// K=2: DiffusionChernoff
// ---------------------------------------------------------------------------

#[test]
fn g_as_k_diffusion_k2_in_subspace_true() {
    let (grid, f) = gaussian(64);
    let dc = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    assert!(
        dc.in_subspace(&f),
        "K=2: Gaussian on N=64 must be in D(A^2)"
    );
}

#[test]
fn g_as_k_diffusion_k2_in_subspace_false_too_coarse() {
    let (grid, f) = gaussian(4);
    let dc = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    // Minimum is 5 points; 4 is below threshold.
    assert!(
        !dc.in_subspace(&f),
        "K=2: N=4 grid must NOT be in D(A^2) (below 5-point minimum)"
    );
}

#[test]
fn g_as_k_diffusion_k2_jet_finite() {
    let (grid, f) = gaussian(64);
    let dc = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let mut out = alloc_jet_out(&f, 3); // K+1 = 3
    dc.jet(&f, &mut out).expect("jet K=2 must succeed");
    for (k, slice) in out.iter().enumerate() {
        assert!(
            slice.values.iter().all(|v| v.is_finite()),
            "K=2 jet: out[{k}] contains non-finite values"
        );
    }
}

#[test]
fn g_as_k_diffusion_k2_jet_wrong_len_errors() {
    let (grid, f) = gaussian(64);
    let dc = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    // Wrong length: 4 instead of 3.
    let mut out = alloc_jet_out(&f, 4);
    let err = dc.jet(&f, &mut out);
    assert!(
        err.is_err(),
        "K=2 jet must return error for out.len()=4 (expected 3)"
    );
}

#[test]
fn g_as_k_diffusion_k2_assert_in_subspace_ok() {
    let (grid, f) = gaussian(64);
    let dc = DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    assert_in_subspace::<_, f64, 2>(&dc, &f).expect("assert_in_subspace K=2 must be Ok");
}

// ---------------------------------------------------------------------------
// K=4: Diffusion4thChernoff
// ---------------------------------------------------------------------------

#[test]
fn g_as_k_diffusion4_k4_in_subspace_true() {
    let (grid, f) = gaussian(64);
    let d4 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    assert!(
        <Diffusion4thChernoff as ApproximationSubspace<4>>::in_subspace(&d4, &f),
        "K=4: Gaussian on N=64 must be in D(A^4)"
    );
}

#[test]
fn g_as_k_diffusion4_k4_in_subspace_false_too_coarse() {
    let (grid, f) = gaussian(7);
    let d4 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    // Minimum is 9 points; 7 is below threshold.
    assert!(
        !<Diffusion4thChernoff as ApproximationSubspace<4>>::in_subspace(&d4, &f),
        "K=4: N=7 grid must NOT be in D(A^4) (below 9-point minimum)"
    );
}

#[test]
fn g_as_k_diffusion4_k4_jet_finite() {
    let (grid, f) = gaussian(64);
    let d4 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let mut out = alloc_jet_out(&f, 5); // K+1 = 5
    <Diffusion4thChernoff as ApproximationSubspace<4>>::jet(&d4, &f, &mut out)
        .expect("jet K=4 must succeed");
    for (k, slice) in out.iter().enumerate() {
        assert!(
            slice.values.iter().all(|v| v.is_finite()),
            "K=4 jet: out[{k}] contains non-finite values"
        );
    }
}

#[test]
fn g_as_k_diffusion4_k4_jet_wrong_len_errors() {
    let (grid, f) = gaussian(64);
    let d4 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    // Wrong length: 6 instead of 5.
    let mut out = alloc_jet_out(&f, 6);
    let err = <Diffusion4thChernoff as ApproximationSubspace<4>>::jet(&d4, &f, &mut out);
    assert!(
        err.is_err(),
        "K=4 jet must return error for out.len()=6 (expected 5)"
    );
}

#[test]
fn g_as_k_diffusion4_k4_assert_in_subspace_ok() {
    let (grid, f) = gaussian(64);
    let d4 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    assert_in_subspace::<_, f64, 4>(&d4, &f).expect("assert_in_subspace K=4 must be Ok");
}

// ---------------------------------------------------------------------------
// K=6: TruncatedExp4thDiffusionChernoff
// ---------------------------------------------------------------------------

#[test]
fn g_as_k_truncated_exp4_k6_in_subspace_true() {
    let (grid, f) = gaussian(64);
    let te4 = TruncatedExp4thDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    assert!(
        te4.in_subspace(&f),
        "K=6: Gaussian on N=64 must be in D(A^6)"
    );
}

#[test]
fn g_as_k_truncated_exp4_k6_in_subspace_false_too_coarse() {
    let (grid, f) = gaussian(11);
    let te4 = TruncatedExp4thDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    // Minimum is 13 points; 11 is below threshold.
    assert!(
        !te4.in_subspace(&f),
        "K=6: N=11 grid must NOT be in D(A^6) (below 13-point minimum)"
    );
}

#[test]
fn g_as_k_truncated_exp4_k6_jet_finite() {
    let (grid, f) = gaussian(64);
    let te4 = TruncatedExp4thDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let mut out = alloc_jet_out(&f, 7); // K+1 = 7
    te4.jet(&f, &mut out).expect("jet K=6 must succeed");
    for (k, slice) in out.iter().enumerate() {
        assert!(
            slice.values.iter().all(|v| v.is_finite()),
            "K=6 jet: out[{k}] contains non-finite values"
        );
    }
}

#[test]
fn g_as_k_truncated_exp4_k6_jet_wrong_len_errors() {
    let (grid, f) = gaussian(64);
    let te4 = TruncatedExp4thDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    // Wrong length: 8 instead of 7.
    let mut out = alloc_jet_out(&f, 8);
    let err = te4.jet(&f, &mut out);
    assert!(
        err.is_err(),
        "K=6 jet must return error for out.len()=8 (expected 7)"
    );
}

#[test]
fn g_as_k_truncated_exp4_k6_assert_in_subspace_ok() {
    let (grid, f) = gaussian(64);
    let te4 = TruncatedExp4thDiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    assert_in_subspace::<_, f64, 6>(&te4, &f).expect("assert_in_subspace K=6 must be Ok");
}
