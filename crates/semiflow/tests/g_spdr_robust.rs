//! ADR-0202 Wave 2 review regressions (not a gate): duplicate CSR entries, n = 0,
//! scale invariance of the PCG path and of the reported residual, congruence-scaled
//! Neumann at λ = 0.
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::useless_vec
)]

use semiflow::{scratch::ScratchPool, Precond, SemiflowError, SpdSolver, SymmetricOperator};

mod spdr_common;
use spdr_common::{dense_shifted, is_domain, op_1d, rel_sup_err, solve_dense};

/// Tridiagonal operator whose off-diagonals are each stored twice (-0.5 + -0.5).
fn duplicate_entries(n: usize) -> SymmetricOperator<f64> {
    let (mut rp, mut ci, mut va) = (vec![0_usize], Vec::new(), Vec::new());
    for i in 0..n {
        let i32_ = u32::try_from(i).unwrap();
        if i > 0 {
            ci.extend([i32_ - 1, i32_ - 1]);
            va.extend([-0.5, -0.5]);
        }
        ci.push(i32_);
        va.push(3.0);
        if i + 1 < n {
            ci.extend([i32_ + 1, i32_ + 1]);
            va.extend([-0.5, -0.5]);
        }
        rp.push(ci.len());
    }
    SymmetricOperator::from_csr(n, &rp, &ci, &va, 0.0).unwrap()
}

#[test]
fn duplicate_offdiagonal_entries_are_summed() {
    let n = 6;
    let op = duplicate_entries(n);
    let b = vec![1.0; n];
    let want = solve_dense(&dense_shifted(&op, 1.0, &vec![1.0; n]), &b);
    let pcg = SpdSolver::Pcg {
        precond: Precond::Ic0,
        max_iter: None,
    };
    for solver in [SpdSolver::Auto, SpdSolver::Tridiagonal, pcg] {
        let r = op.resolvent(1.0, None, solver, 1e-12).unwrap();
        let mut x = vec![0.0; n];
        let rep = r.solve_into(&b, &mut x, &mut ScratchPool::new()).unwrap();
        assert!(
            rel_sup_err(&x, &want) <= 1e-10,
            "{solver:?}: wrong solution"
        );
        assert!(
            rep.rel_residual <= 1e-10,
            "{solver:?}: residual {}",
            rep.rel_residual
        );
    }
}

#[test]
fn empty_operator_is_rejected_without_panic() {
    let op = SymmetricOperator::<f64>::from_csr(0, &[0], &[], &[], 0.0).unwrap();
    for lam in [0.0, 1.0] {
        let e = op
            .resolvent(lam, None, SpdSolver::Auto, 1e-12)
            .err()
            .expect("n=0");
        assert!(is_domain(&e), "{e:?}");
    }
}

#[test]
fn scale_invariance_and_honest_residual() {
    let n = 8;
    let op = op_1d(n).with_diagonal(&vec![0.5; n]).unwrap();
    let unit: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
    let solvers = [
        SpdSolver::Auto,
        SpdSolver::Pcg {
            precond: Precond::Jacobi,
            max_iter: None,
        },
        SpdSolver::Pcg {
            precond: Precond::Ic0,
            max_iter: None,
        },
    ];
    for solver in solvers {
        let r = op.resolvent(0.0, None, solver, 1e-12).unwrap();
        let mut x1 = vec![0.0; n];
        r.solve_into(&unit, &mut x1, &mut ScratchPool::new())
            .unwrap();
        for scale in [1e-155, 1e-200, 1e150] {
            let b: Vec<f64> = unit.iter().map(|v| v * scale).collect();
            let mut x = vec![0.0; n];
            let rep = r.solve_into(&b, &mut x, &mut ScratchPool::new()).unwrap();
            let back: Vec<f64> = x.iter().map(|v| v / scale).collect();
            let err = rel_sup_err(&back, &x1);
            assert!(err <= 1e-9, "{solver:?} scale {scale:e}: err {err:.3e}");
            assert!(
                rep.rel_residual <= 1e-10,
                "{solver:?} scale {scale:e}: residual {}",
                rep.rel_residual
            );
        }
    }
}

#[test]
fn congruence_scaled_neumann_at_lambda_zero_errors() {
    let n = 12;
    let masses: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
    let op = op_1d(n).lumped_congruence(&masses).unwrap();
    let pcg = SpdSolver::Pcg {
        precond: Precond::Ic0,
        max_iter: None,
    };
    let b: Vec<f64> = (0..n).map(|i| (i as f64).sin() + 2.0).collect();
    // Auto -> tridiagonal: rejected at build time by the pivot certificate.
    let e = op
        .resolvent(0.0, None, SpdSolver::Auto, 1e-12)
        .err()
        .expect("Auto must reject");
    assert!(is_domain(&e), "Auto: {e:?}");
    // Pcg: built, but the solve must end in ConvergenceFailed (gross-residual guard).
    let r = op.resolvent(0.0, None, pcg, 1e-12).expect("Pcg builds");
    let mut x = vec![0.0; n];
    let e = r
        .solve_into(&b, &mut x, &mut ScratchPool::new())
        .unwrap_err();
    assert!(
        matches!(e, SemiflowError::ConvergenceFailed { .. }),
        "Pcg: {e:?}"
    );
}

/// Dirichlet-like path (diag 2, off-diagonals -1), `λ = 0`.
fn dirichlet_path(n: usize) -> SymmetricOperator<f64> {
    let (mut rp, mut ci, mut va) = (vec![0_usize], Vec::new(), Vec::new());
    for i in 0..n {
        let c = u32::try_from(i).unwrap();
        if i > 0 {
            ci.push(c - 1);
            va.push(-1.0);
        }
        ci.push(c);
        va.push(2.0);
        if i + 1 < n {
            ci.push(c + 1);
            va.push(-1.0);
        }
        rp.push(ci.len());
    }
    SymmetricOperator::from_csr(n, &rp, &ci, &va, 0.0).unwrap()
}

#[test]
fn unrepresentable_solution_is_an_error() {
    let n = 8;
    let op = dirichlet_path(n);
    let jac = SpdSolver::Pcg {
        precond: Precond::Jacobi,
        max_iter: None,
    };
    for solver in [SpdSolver::Auto, SpdSolver::Tridiagonal, jac] {
        let r = op.resolvent(0.0, None, solver, 1e-12).unwrap();
        let mut x = vec![0.0; n];
        let e = r
            .solve_into(&vec![1e308; n], &mut x, &mut ScratchPool::new())
            .unwrap_err();
        assert!(is_domain(&e), "{solver:?}: {e:?}");
    }
}
