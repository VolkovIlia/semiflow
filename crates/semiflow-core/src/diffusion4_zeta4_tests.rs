// Unit tests for [`Diffusion4thZeta4Chernoff`] (extracted per suckless ≤500-line cap).
#![allow(clippy::doc_markdown)]

use super::*;
use crate::{Diffusion4thChernoff, Grid1D, GridFn1D};

fn make_kernel(n: usize) -> Diffusion4thZeta4Chernoff<f64> {
    let grid = Grid1D::new(-4.0, 4.0, n).expect("grid");
    let inner = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    Diffusion4thZeta4Chernoff::new(inner, Some(1.0_f64)).expect("kernel")
}

#[test]
fn constructor_validates_bound() {
    let grid = Grid1D::new(-4.0, 4.0, 32).expect("grid");
    let inner = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    assert!(Diffusion4thZeta4Chernoff::new(inner.clone(), Some(-1.0_f64)).is_err());
    assert!(Diffusion4thZeta4Chernoff::new(inner, Some(f64::NAN)).is_err());
}

/// v4.1: order() restored to 4 per ADR-0086 Path β (Path β achieves what v3.0 promised).
#[test]
fn order_is_4() {
    let k = make_kernel(32);
    assert_eq!(k.order(), 4);
}

/// v4.1: growth multiplier = 1.0 (no correction overhead; Path β bounded by ‖f‖_{D(A³)}).
#[test]
fn growth_multiplier_is_1p0() {
    let k = make_kernel(32);
    let g = k.growth();
    assert!((g.multiplier - 1.0).abs() < 1e-12);
    assert!(g.omega.abs() < f64::EPSILON, "omega must be zero");
}

#[test]
fn apply_into_produces_finite_output() {
    let k = make_kernel(64);
    let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
    let mut dst = f.zeroed_like();
    let mut scratch = ScratchPool::new();
    k.apply_into(0.01, &f, &mut dst, &mut scratch)
        .expect("apply_into should succeed");
    assert!(
        dst.values.iter().all(|v| v.is_finite()),
        "output must be finite"
    );
}

#[test]
fn apply_into_rejects_negative_tau() {
    let k = make_kernel(32);
    let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
    let mut dst = f.zeroed_like();
    let mut scratch = ScratchPool::new();
    assert!(k.apply_into(-0.01, &f, &mut dst, &mut scratch).is_err());
}

/// Path β: tau=0 should return f unchanged (all correction terms vanish).
#[test]
fn apply_into_tau_zero_returns_src() {
    let k = make_kernel(64);
    let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
    let mut dst = f.zeroed_like();
    let mut scratch = ScratchPool::new();
    k.apply_into(0.0, &f, &mut dst, &mut scratch)
        .expect("tau=0 must succeed");
    let max_diff = f
        .values
        .iter()
        .zip(dst.values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-14,
        "tau=0 must return src unchanged; max_diff={max_diff:.2e}"
    );
}

#[test]
fn k4_in_subspace_true_for_large_grid() {
    let k = make_kernel(64);
    let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
    assert!(<Diffusion4thZeta4Chernoff<f64> as ApproximationSubspace<
        4,
        f64,
    >>::in_subspace(&k, &f));
}

#[test]
fn k4_in_subspace_false_without_bound() {
    let grid = Grid1D::new(-4.0, 4.0, 64).expect("grid");
    let inner = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid);
    let k = Diffusion4thZeta4Chernoff::new(inner, None).expect("kernel");
    let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
    assert!(!<Diffusion4thZeta4Chernoff<f64> as ApproximationSubspace<
        4,
        f64,
    >>::in_subspace(&k, &f));
}

#[test]
fn k4_jet_finite_on_gaussian() {
    let k = make_kernel(64);
    let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
    let mut out: [GridFn1D<f64>; 5] = core::array::from_fn(|_| f.zeroed_like());
    <Diffusion4thZeta4Chernoff<f64> as ApproximationSubspace<4, f64>>::jet(&k, &f, &mut out)
        .expect("jet K=4 must succeed");
    for (j, slice) in out.iter().enumerate() {
        assert!(
            slice.values.iter().all(|v| v.is_finite()),
            "jet[{j}] has non-finite values"
        );
    }
}

#[test]
fn k4_jet_wrong_len_errors() {
    let k = make_kernel(64);
    let f = GridFn1D::from_fn(k.grid, |x| (-x * x).exp());
    let mut out: [GridFn1D<f64>; 7] = core::array::from_fn(|_| f.zeroed_like());
    assert!(
        <Diffusion4thZeta4Chernoff<f64> as ApproximationSubspace<4, f64>>::jet(&k, &f, &mut out,)
            .is_err()
    );
}
