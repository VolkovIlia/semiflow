//! Unit tests of the rational Krylov path (ADR-0206). The gates live in
//! `tests/g_rational_krylov.rs`.

use alloc::{sync::Arc, vec, vec::Vec};

use super::*;
use crate::{
    graph::Graph,
    graph_batched,
    symmetric_operator::{dense_csr_expmv_ref, SymmetricOperator},
};

/// Dirichlet-like SPD tridiagonal `n×n` (diag 2.5, off −1).
fn spd(n: usize) -> SymmetricOperator<f64> {
    let (mut rp, mut ci, mut va) = (vec![0], Vec::new(), Vec::new());
    for i in 0..n {
        if i > 0 {
            ci.push(u32::try_from(i - 1).unwrap());
            va.push(-1.0);
        }
        ci.push(u32::try_from(i).unwrap());
        va.push(2.5);
        if i + 1 < n {
            ci.push(u32::try_from(i + 1).unwrap());
            va.push(-1.0);
        }
        rp.push(ci.len());
    }
    SymmetricOperator::from_csr(n, &rp, &ci, &va, 0.0).unwrap()
}

#[allow(clippy::cast_precision_loss)]
fn vector(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 + (0.7 * i as f64).sin()).collect()
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn matches_dense_reference_without_mass() {
    let n = 10;
    let op = spd(n);
    let v = vector(n);
    let mut scratch = ScratchPool::new();
    for t in [1e-3, 0.5, 7.0, 1e3] {
        let mut want = vec![0.0; n];
        dense_csr_expmv_ref(&op, t, &v, &mut want).unwrap();
        let mut got = vec![0.0; n];
        let opts = RationalOptions::default().with_tol(1e-12);
        let rep = rational_krylov_expmv(&op, None, t, &v, &mut got, opts, &mut scratch).unwrap();
        assert!(
            max_err(&got, &want) <= 1e-11,
            "t={t}: {}",
            max_err(&got, &want)
        );
        assert!(rep.solves >= 1 && rep.solves <= n, "t={t}: {rep:?}");
        assert_eq!(rep.inner_iterations, 0, "tridiagonal path: direct solves");
    }
}

#[test]
fn identity_cases_and_errors() {
    let op = spd(6);
    let v = vector(6);
    let mut out = vec![9.0; 6];
    let mut scratch = ScratchPool::new();
    let opts = RationalOptions::default();
    let rep = rational_krylov_expmv(&op, None, 0.0, &v, &mut out, opts, &mut scratch).unwrap();
    assert_eq!(
        (out.clone(), rep.solves),
        (v.clone(), 0),
        "t = 0 is the identity"
    );
    let zero = vec![0.0; 6];
    rational_krylov_expmv(&op, None, 1.0, &zero, &mut out, opts, &mut scratch).unwrap();
    assert!(out.iter().all(|&x| x == 0.0));
    let bad = |o: RationalOptions, t: f64| RationalKrylov::new(&op, None, t, o).is_err();
    assert!(bad(opts.with_tol(0.0), 1.0) && bad(opts.with_tol(1.0), 1.0));
    assert!(bad(opts.with_gamma(0.0), 1.0) && bad(opts.with_gamma(f64::NAN), 1.0));
    assert!(bad(opts.with_m_max(0), 1.0) && bad(opts, -1.0) && bad(opts, f64::INFINITY));
    let rk = RationalKrylov::new(&op, None, 1.0, opts).unwrap();
    assert!(rk.apply_into(&v[..5], &mut out, &mut scratch).is_err());
    let mut nan = v.clone();
    nan[2] = f64::NAN;
    assert!(rk.apply_into(&nan, &mut out, &mut scratch).is_err());
}

#[test]
fn too_small_krylov_dimension_is_reported() {
    let n = 200;
    let op = spd(n);
    let v = vector(n);
    let mut out = vec![0.0; n];
    let opts = RationalOptions::default().with_tol(1e-14).with_m_max(3);
    let err = rational_krylov_expmv(&op, None, 50.0, &v, &mut out, opts, &mut ScratchPool::new());
    assert!(
        matches!(
            err,
            Err(SemiflowError::ConvergenceFailed { max_iter: 3, .. })
        ),
        "{err:?}"
    );
}

#[test]
fn chernoff_wrapper_matches_direct_action_and_other_tau() {
    let n = 12;
    let op = spd(n);
    let graph = Arc::new(Graph::<f64>::path(n));
    let v = vector(n);
    let opts = RationalOptions::default().with_tol(1e-12);
    let mut want = vec![0.0; n];
    let mut scratch = ScratchPool::new();
    // Two steps of τ = 0.25 against one action at t = 0.5.
    rational_krylov_expmv(&op, None, 0.5, &v, &mut want, opts, &mut scratch).unwrap();
    let chernoff = RationalKrylovChernoff::new(&op, None, 0.25, opts).unwrap();
    let mut got = vec![0.0; n];
    graph_batched::evolve_batched(&chernoff, &graph, 0.5, 2, &v, &mut got).unwrap();
    assert!(max_err(&got, &want) <= 1e-11, "{}", max_err(&got, &want));
    // A τ the factor was not built for still gives the right action.
    graph_batched::evolve_batched(&chernoff, &graph, 0.5, 1, &v, &mut got).unwrap();
    assert!(max_err(&got, &want) <= 1e-11, "{}", max_err(&got, &want));
    assert_eq!(chernoff.order(), u32::MAX);
}

#[test]
fn a_priori_dimension_matches_the_chebyshev_tail() {
    // r = t/γ = 10 (the default γ): the tail table of ADR-0206.
    let dims: Vec<_> = [1e-6, 1e-8, 1e-10, 1e-12]
        .iter()
        .map(|&tol| factor::a_priori_dim(10.0, tol, 64))
        .collect();
    assert_eq!(dims, [Some(16), Some(21), Some(27), Some(33)]);
    assert_eq!(factor::a_priori_dim(10.0, 1e-10, 20), None, "beyond m_cap");
}

#[test]
fn excess_factor_matches_the_resolvent_and_falls_back() {
    let n = 30;
    let op = spd(n);
    let b = vector(n);
    let mut scratch = ScratchPool::new();
    let lambda = 0.37;
    let excess = factor::Inner::build(&op, lambda, None, SpdSolver::Auto, 1e-12).unwrap();
    assert!(matches!(excess, factor::Inner::Excess(_)));
    let reference = op.resolvent(lambda, None, SpdSolver::Auto, 1e-12).unwrap();
    let (mut x, mut y) = (vec![0.0; n], vec![0.0; n]);
    excess.solve(&b, &mut x, &mut scratch).unwrap();
    reference.solve_into(&b, &mut y, &mut scratch).unwrap();
    assert!(max_err(&x, &y) <= 1e-14, "{}", max_err(&x, &y));
    // A positive off-diagonal (not an M-matrix pattern) takes the resolvent.
    let (rp, ci, va) = op.csr();
    let flipped: Vec<f64> = va.iter().map(|&v| if v < 0.0 { 1.0 } else { v }).collect();
    let pos = SymmetricOperator::from_csr(n, rp, ci, &flipped, 0.0).unwrap();
    let inner = factor::Inner::build(&pos, lambda, None, SpdSolver::Auto, 1e-12).unwrap();
    assert!(matches!(inner, factor::Inner::Resolvent(_)));
    let forced = SpdSolver::Pcg {
        precond: crate::spd_resolvent::Precond::Ic0,
        max_iter: None,
    };
    let pcg = factor::Inner::build(&op, lambda, None, forced, 1e-12).unwrap();
    assert!(matches!(pcg, factor::Inner::Resolvent(_)), "forced PCG");
}

#[test]
fn power_of_two_homogeneity_and_tiny_t_identity() {
    let n = 40;
    let op = spd(n);
    let v = vector(n);
    let mut scratch = ScratchPool::new();
    let rk = RationalKrylov::new(&op, None, 3.0, RationalOptions::default()).unwrap();
    let mut base = vec![0.0; n];
    rk.apply_into(&v, &mut base, &mut scratch).unwrap();
    for k in [-900_i32, -20, 7, 900] {
        let s = 2.0_f64.powi(k);
        let vs: Vec<f64> = v.iter().map(|x| x * s).collect();
        let mut got = vec![0.0; n];
        rk.apply_into(&vs, &mut got, &mut scratch).unwrap();
        assert!(
            got.iter()
                .zip(&base)
                .all(|(g, b)| g.to_bits() == (b * s).to_bits()),
            "2^{k}: not bitwise homogeneous"
        );
    }
    // t·‖A‖ < ε: the identity, no factor; 1/γ must be finite.
    let tiny = RationalKrylov::new(&op, None, 1e-20, RationalOptions::default()).unwrap();
    let mut out = vec![0.0; n];
    let rep = tiny.apply_into(&v, &mut out, &mut scratch).unwrap();
    assert_eq!((out, rep.solves), (v.clone(), 0));
    let opts = RationalOptions::default().with_gamma(1e-310);
    assert!(RationalKrylov::new(&op, None, 1.0, opts).is_err());
    let empty = SymmetricOperator::<f64>::from_csr(0, &[0], &[], &[], 0.0);
    if let Ok(e) = empty {
        assert!(RationalKrylov::new(&e, None, 1.0, RationalOptions::default()).is_err());
    }
}

#[test]
fn dual_numbers_are_rejected_not_silently_wrong() {
    use crate::dual::Dual;
    let op = spd(5);
    let (rp, ci, va) = op.csr();
    let dual_vals: Vec<Dual<f64>> = va.iter().map(|&v| Dual::constant(v)).collect();
    let dop = SymmetricOperator::from_csr(5, rp, ci, &dual_vals, Dual::constant(0.0)).unwrap();
    let r = RationalKrylov::new(&dop, None, Dual::constant(1.0), RationalOptions::default());
    assert!(matches!(r, Err(SemiflowError::Unsupported { .. })));
}
