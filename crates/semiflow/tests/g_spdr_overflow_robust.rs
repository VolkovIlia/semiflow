//! `G_SPDR_OVERFLOW_ROBUST` (ADR-0202 PR-review regressions): finite inputs whose
//! intermediates overflow / underflow must either give the exact finite answer or a
//! typed `DomainViolation`; never `inf`/`NaN`/silent zero.
#![allow(clippy::doc_markdown, clippy::float_cmp)]

use semiflow::{
    general_operator::GeneralOperator,
    generator_action::{CsrGenerator, GeneratorAction},
    Precond, ScratchPool, SemiflowError, SpdSolver, SymmetricOperator,
};

fn one_by_one(a: f64) -> SymmetricOperator<f64> {
    SymmetricOperator::from_csr(1, &[0, 1], &[0], &[a], 0.0).unwrap()
}

fn two_by_two(v: f64) -> SymmetricOperator<f64> {
    SymmetricOperator::from_csr(2, &[0, 2, 4], &[0, 1, 0, 1], &[v, v, v, v], 1e-12).unwrap()
}

fn is_domain(e: &SemiflowError) -> bool {
    matches!(e, SemiflowError::DomainViolation { .. })
}

// ── item 3: with_diagonal overflow ──────────────────────────────────────────────

#[test]
fn with_diagonal_rejects_overflowing_sum() {
    let r = one_by_one(1e308).with_diagonal(&[1e308]);
    assert!(
        matches!(&r, Err(e) if is_domain(e)),
        "got ok/other: {:?}",
        r.is_ok()
    );
}

#[test]
fn with_diagonal_rejects_max_plus_max() {
    let r = one_by_one(f64::MAX).with_diagonal(&[f64::MAX]);
    assert!(matches!(&r, Err(e) if is_domain(e)));
}

#[test]
fn with_diagonal_finite_sum_still_ok() {
    let op = one_by_one(1.0).with_diagonal(&[2.0]).unwrap();
    assert_eq!(op.csr().2, &[3.0]);
}

// ── item 4: norm bound with mass ────────────────────────────────────────────────

#[test]
fn norm_bound_no_row_sum_overflow() {
    let a = two_by_two(1e308);
    let g = CsrGenerator::from_symmetric(&a, Some(&[1e308, 1e308])).unwrap();
    let nb = g.norm_bound();
    assert!(nb.is_finite() && (nb - 2.0).abs() < 1e-12, "bound = {nb}");
}

#[test]
fn unrepresentable_bound_rejected_symmetric() {
    let r = CsrGenerator::from_symmetric(&one_by_one(4.0), Some(&[1e-308]));
    assert!(matches!(&r, Err(e) if is_domain(e)));
}

#[test]
fn unrepresentable_bound_rejected_general() {
    let op = GeneralOperator::from_csr(1, &[0, 1], &[0], &[4.0]).unwrap();
    let r = CsrGenerator::from_general(&op, Some(&[1e-308]));
    assert!(matches!(&r, Err(e) if is_domain(e)));
}

// ── item 5: mass apply / transpose without intermediate over/underflow ─────────

#[test]
fn mass_forward_no_overflow() {
    let g = CsrGenerator::from_symmetric(&one_by_one(1e200), Some(&[1e200])).unwrap();
    let mut dst = [0.0];
    g.apply_generator(&[1e200], &mut dst);
    assert_eq!(dst[0], -1e200);
}

#[test]
fn mass_transpose_no_underflow() {
    let g = CsrGenerator::from_symmetric(&one_by_one(1e200), Some(&[1e200])).unwrap();
    let mut dst = [0.0];
    g.apply_generator_transpose(&[1e-200], &mut dst);
    assert_eq!(dst[0], -1e-200);
}

// ── item 6: subnormal RHS ───────────────────────────────────────────────────────

fn solve(solver: SpdSolver, b: f64) -> f64 {
    let op = one_by_one(1.0);
    let r = op.resolvent(0.0, None, solver, 1e-12).unwrap();
    let mut x = [0.0];
    r.solve_into(&[b], &mut x, &mut ScratchPool::new()).unwrap();
    x[0]
}

#[test]
fn subnormal_rhs_all_solvers() {
    for b in [1e-310, f64::from_bits(1), 5e-324 * 1000.0] {
        for s in [
            SpdSolver::Auto,
            SpdSolver::Tridiagonal,
            SpdSolver::Pcg {
                precond: Precond::Jacobi,
                max_iter: Some(100),
            },
        ] {
            let x = solve(s, b);
            assert!(x != 0.0 && ((x - b) / b).abs() < 1e-8, "b={b:e} x={x:e}");
        }
    }
}

#[test]
fn subnormal_rhs_f32_all_solvers() {
    let op = SymmetricOperator::<f32>::from_csr(1, &[0, 1], &[0], &[1.0], 0.0).unwrap();
    for b in [1e-40_f32, f32::from_bits(1)] {
        for s in [
            SpdSolver::Auto,
            SpdSolver::Tridiagonal,
            SpdSolver::Pcg {
                precond: Precond::Jacobi,
                max_iter: Some(100),
            },
        ] {
            let r = op.resolvent(0.0, None, s, 1e-5).unwrap();
            let mut x = [0.0_f32];
            r.solve_into(&[b], &mut x, &mut ScratchPool::new()).unwrap();
            assert!(
                x[0] != 0.0 && ((x[0] - b) / b).abs() < 1e-3,
                "b={b:e} x={:e}",
                x[0]
            );
        }
    }
}
