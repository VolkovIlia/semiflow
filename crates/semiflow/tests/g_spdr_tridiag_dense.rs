//! `G_SPDR_TRIDIAG_DENSE` (`RELEASE_BLOCKING`, ADR-0202 D1, math §62.2.a).
//!
//! Tridiagonal LDLᵀ path vs dense Gaussian elimination (partial pivoting), n = 12,
//! conservative operator with `k(x) = 1 + x²`, three (λ, c, mass) cases; plus
//! factor reuse across right-hand sides.
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::useless_vec,
    clippy::similar_names
)]

use semiflow::{scratch::ScratchPool, ResolventMethod, SpdSolver, SymmetricOperator};

mod spdr_common;
use spdr_common::{
    dense_rel_residual, dense_shifted, half_cell_mass, lcg, op_1d, rel_sup_err, solve_dense,
};

const N: usize = 12;

fn run_case(name: &str, op: &SymmetricOperator<f64>, lam: f64, mass: Option<&[f64]>) {
    let ones = vec![1.0; N];
    let m = mass.unwrap_or(&ones);
    let r = op
        .resolvent(lam, mass, SpdSolver::Auto, 1e-12)
        .unwrap_or_else(|e| panic!("{name}: new failed: {e:?}"));
    assert_eq!(r.method(), ResolventMethod::Tridiagonal, "{name}: method");
    assert_eq!(r.n(), N);
    let s = dense_shifted(op, lam, m);
    let mut sc = ScratchPool::new();
    // Two different right-hand sides reuse the same factor.
    for seed in [7_u64, 99] {
        let b = lcg(seed, N);
        let mut x = vec![0.0; N];
        let rep = r.solve_into(&b, &mut x, &mut sc).expect("solve_into");
        let want = solve_dense(&s, &b);
        let err = rel_sup_err(&x, &want);
        let res = dense_rel_residual(&s, &x, &b);
        eprintln!(
            "{name} seed={seed}: rel_err={err:.3e} residual={res:.3e} reported={:.3e}",
            rep.rel_residual
        );
        assert!(err <= 1e-12, "{name}: rel_err {err:.3e} > 1e-12");
        assert!(res <= 1e-14, "{name}: residual {res:.3e} > 1e-14");
        assert_eq!(
            rep.iterations, 0,
            "{name}: direct path reports 0 iterations"
        );
        assert!(rep.rel_residual <= 1e-14, "{name}: reported residual");
    }
}

#[test]
fn g_spdr_tridiag_dense() {
    let base = op_1d(N);
    let c_pos: Vec<f64> = (0..N).map(|i| 0.5 + (i as f64) * 0.25).collect();
    let with_c = base.with_diagonal(&c_pos).unwrap();
    let half = half_cell_mass(N);
    run_case("lam0_c>0", &with_c, 0.0, None);
    run_case("lam0.5_c0", &base, 0.5, None);
    run_case("lam1e3_c>0_mass", &with_c, 1e3, Some(&half));
    // b = 0 gives x = 0 exactly.
    let r = base.resolvent(1.0, None, SpdSolver::Auto, 1e-12).unwrap();
    let mut x = vec![1.0; N];
    let rep = r
        .solve_into(&vec![0.0; N], &mut x, &mut ScratchPool::new())
        .unwrap();
    assert!(x.iter().all(|&v| v == 0.0) && rep.rel_residual == 0.0);
}
