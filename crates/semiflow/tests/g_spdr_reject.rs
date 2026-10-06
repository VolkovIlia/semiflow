//! `G_SPDR_REJECT` (`RELEASE_BLOCKING` TEETH gate, ADR-0202 D1, math §62.1 singular case).
//!
//! Every misuse returns the exact error variant; never a garbage solution, never a panic.
//! NOTE: the 2-D operator is 4×4 (`GridND` needs n ≥ 4 per axis).
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::useless_vec,
    clippy::similar_names
)]

use semiflow::{
    assemble_conservative_csr_nd, boundary::BoundaryPolicy, generator_action::CsrGenerator,
    grid::Grid1D, grid_nd::GridND, phi_combination, scratch::ScratchPool, Precond, SemiflowError,
    SpdSolver, SymmetricOperator,
};

mod spdr_common;
use spdr_common::{is_domain, lcg, op_1d};

const N: usize = 12;

fn new_err(
    op: &SymmetricOperator<f64>,
    lam: f64,
    mass: Option<&[f64]>,
    solver: SpdSolver,
    tol: f64,
) -> SemiflowError {
    match op.resolvent(lam, mass, solver, tol) {
        Err(e) => e,
        Ok(_) => panic!("expected an error (lam={lam}, tol={tol})"),
    }
}

/// Two decoupled 4-node path blocks (integer entries, exact LDLᵀ arithmetic).
fn two_blocks(c: &[f64]) -> SymmetricOperator<f64> {
    let (mut rp, mut ci, mut va) = (vec![0_usize], Vec::new(), Vec::new());
    for blk in 0..2_u32 {
        for j in 0..4_u32 {
            let i = 4 * blk + j;
            if j > 0 {
                ci.push(i - 1);
                va.push(-1.0);
            }
            ci.push(i);
            va.push(if j == 0 || j == 3 { 1.0 } else { 2.0 });
            if j < 3 {
                ci.push(i + 1);
                va.push(-1.0);
            }
            rp.push(ci.len());
        }
    }
    let a = SymmetricOperator::from_csr(8, &rp, &ci, &va, 0.0).unwrap();
    a.with_diagonal(c).unwrap()
}

fn op_2d() -> SymmetricOperator<f64> {
    let ax = Grid1D::new(0.0, 1.0, 4).unwrap();
    let grid = GridND::<f64, 2>::new([ax, ax]).unwrap();
    let k = [1.0, 1.5, 2.0, 2.5];
    assemble_conservative_csr_nd(&grid, &[&k, &k], BoundaryPolicy::Neumann).unwrap()
}

#[test]
fn g_spdr_reject() {
    let neumann = op_1d(N);
    let c1 = vec![0.4; N];
    let reactive = neumann.with_diagonal(&c1).unwrap();
    // (1) pure Neumann, lam = 0, Auto: a-priori null-space rejection.
    assert!(is_domain(&new_err(
        &neumann,
        0.0,
        None,
        SpdSolver::Auto,
        1e-12
    )));
    // (2) decoupled blocks, one without reaction, forced Tridiagonal: pivot certificate.
    let blocks = two_blocks(&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(is_domain(&new_err(
        &blocks,
        0.0,
        None,
        SpdSolver::Tridiagonal,
        1e-12
    )));
    // (3) bad lambda.
    for lam in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(is_domain(&new_err(
            &reactive,
            lam,
            None,
            SpdSolver::Auto,
            1e-12
        )));
    }
    // (4) bad mass.
    let short = vec![1.0; N - 1];
    let mut zero = vec![1.0; N];
    zero[3] = 0.0;
    let mut nan = vec![1.0; N];
    nan[5] = f64::NAN;
    for m in [&short, &zero, &nan] {
        assert!(is_domain(&new_err(
            &reactive,
            1.0,
            Some(m),
            SpdSolver::Auto,
            1e-12
        )));
    }
    // (5) bad tol.
    for tol in [0.0, 1.0, f64::NAN] {
        assert!(is_domain(&new_err(
            &reactive,
            1.0,
            None,
            SpdSolver::Auto,
            tol
        )));
    }
    // (6) Tridiagonal forced on a 2-D operator.
    let two_d = op_2d().with_diagonal(&[0.5; 16]).unwrap();
    let e = new_err(&two_d, 1.0, None, SpdSolver::Tridiagonal, 1e-12);
    assert!(matches!(e, SemiflowError::Unsupported { .. }), "{e:?}");
    // (7) iteration cap of 1 on a system that needs more.
    let capped = SpdSolver::Pcg {
        precond: Precond::Jacobi,
        max_iter: Some(1),
    };
    let r = two_d.resolvent(0.0, None, capped, 1e-12).unwrap();
    let mut x = vec![0.0; 16];
    let e = r
        .solve_into(&lcg(1, 16), &mut x, &mut ScratchPool::new())
        .unwrap_err();
    assert!(
        matches!(e, SemiflowError::ConvergenceFailed { max_iter: 1, .. }),
        "{e:?}"
    );
    // (8) bad right-hand sides.
    let r = reactive
        .resolvent(1.0, None, SpdSolver::Auto, 1e-12)
        .unwrap();
    let mut sc = ScratchPool::new();
    let mut x = vec![0.0; N];
    assert!(is_domain(
        &r.solve_into(&vec![1.0; N - 1], &mut x, &mut sc)
            .unwrap_err()
    ));
    let mut bad = vec![1.0; N];
    bad[2] = f64::NAN;
    assert!(is_domain(&r.solve_into(&bad, &mut x, &mut sc).unwrap_err()));
    assert!(is_domain(
        &r.solve_into(&vec![1.0; N], &mut vec![0.0; N + 1], &mut sc)
            .unwrap_err()
    ));
    // (9) with_diagonal.
    assert!(is_domain(
        &neumann.with_diagonal(&vec![-1e-3; N]).err().unwrap()
    ));
    assert!(is_domain(
        &neumann.with_diagonal(&vec![0.0; N + 1]).err().unwrap()
    ));
    // (10) phi_combination.
    let g = CsrGenerator::from_symmetric(&neumann, None).unwrap();
    let w = lcg(2, N);
    let mut out = vec![0.0; N];
    let none: [&[f64]; 0] = [];
    assert!(is_domain(
        &phi_combination(&g, 0.1, &none, &mut out, &mut sc).unwrap_err()
    ));
    let five: Vec<&[f64]> = vec![&w; 5];
    assert!(is_domain(
        &phi_combination(&g, 0.1, &five, &mut out, &mut sc).unwrap_err()
    ));
    let short_w = &w[..N - 1];
    assert!(is_domain(
        &phi_combination(&g, 0.1, &[short_w], &mut out, &mut sc).unwrap_err()
    ));
    assert!(is_domain(
        &phi_combination(&g, -0.1, &[&w], &mut out, &mut sc).unwrap_err()
    ));
    eprintln!("G_SPDR_REJECT: all 10 case groups return the listed variants");
}
