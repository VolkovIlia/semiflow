//! `G_SPDR_PCG_DENSE` (`RELEASE_BLOCKING`, ADR-0202 D1, math §62.2.b).
//!
//! PCG (IC(0) and Jacobi) on a non-tridiagonal 2-D conservative operator + reaction,
//! λ ∈ {0 (c > 0), 1}, vs dense Gaussian elimination.
//! Grid is 4×4 (n = 16): `GridND::new` requires n ≥ 4 per axis (ADR-0202 Amendment 1).
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::useless_vec,
    clippy::similar_names
)]

use semiflow::{
    assemble_conservative_csr_nd, boundary::BoundaryPolicy, grid::Grid1D, grid_nd::GridND,
    scratch::ScratchPool, Precond, ResolventMethod, SpdSolver, SymmetricOperator,
};

mod spdr_common;
use spdr_common::{dense_shifted, lcg, rel_sup_err, solve_dense};

const N: usize = 16;

fn op_2d() -> SymmetricOperator<f64> {
    let ax0 = Grid1D::new(0.0, 1.0, 4).unwrap();
    let ax1 = Grid1D::new(0.0, 1.0, 4).unwrap();
    let grid = GridND::<f64, 2>::new([ax0, ax1]).unwrap();
    let k0: Vec<f64> = (0..4_u8).map(|i| 1.0 + 0.5 * f64::from(i)).collect();
    let k1: Vec<f64> = (0..4_u8).map(|i| 2.0 - 0.25 * f64::from(i)).collect();
    assemble_conservative_csr_nd(&grid, &[&k0, &k1], BoundaryPolicy::Neumann).unwrap()
}

fn solve(
    op: &SymmetricOperator<f64>,
    lam: f64,
    solver: SpdSolver,
    b: &[f64],
) -> (Vec<f64>, usize, ResolventMethod) {
    let r = op.resolvent(lam, None, solver, 1e-12).expect("new");
    let mut x = vec![0.0; N];
    let rep = r
        .solve_into(b, &mut x, &mut ScratchPool::new())
        .expect("solve");
    (x, rep.iterations, r.method())
}

#[test]
fn g_spdr_pcg_dense() {
    let c: Vec<f64> = (0..N).map(|i| 0.3 + 0.1 * i as f64).collect();
    let op = op_2d().with_diagonal(&c).unwrap();
    let ones = vec![1.0; N];
    let b = lcg(5, N);
    for lam in [0.0, 1.0] {
        let want = solve_dense(&dense_shifted(&op, lam, &ones), &b);
        let ic0 = SpdSolver::Pcg {
            precond: Precond::Ic0,
            max_iter: None,
        };
        let jac = SpdSolver::Pcg {
            precond: Precond::Jacobi,
            max_iter: None,
        };
        let (x1, it_ic0, m1) = solve(&op, lam, ic0, &b);
        let (x2, it_jac, m2) = solve(&op, lam, jac, &b);
        let (x3, _, m3) = solve(&op, lam, SpdSolver::Auto, &b);
        let (e1, e2, e3) = (
            rel_sup_err(&x1, &want),
            rel_sup_err(&x2, &want),
            rel_sup_err(&x3, &want),
        );
        eprintln!("lam={lam}: ic0 err={e1:.3e} it={it_ic0}; jacobi err={e2:.3e} it={it_jac}; auto err={e3:.3e}");
        assert!(e1 <= 1e-10 && e2 <= 1e-10 && e3 <= 1e-10, "rel_err > 1e-10");
        assert_eq!(m1, ResolventMethod::PcgIc0);
        assert_eq!(m2, ResolventMethod::PcgJacobi);
        assert_eq!(
            m3,
            ResolventMethod::PcgIc0,
            "Auto on a 2-D pattern dispatches to PCG+IC(0)"
        );
        assert!(
            it_ic0 <= it_jac,
            "iters_ic0 {it_ic0} > iters_jacobi {it_jac}"
        );
    }
}
