//! Unit tests of the adaptive Krylov φ path (ADR-0211). The gates live in
//! `tests/g_phi_krylov.rs`.

use alloc::{vec, vec::Vec};

use super::*;
use crate::{phi_action::phi_combination, phi_krylov_arnoldi::hess_exp_e1};

/// Upwind-drift tridiagonal `n×n` CSR operator `A` (non-symmetric, accretive).
fn drift(n: usize, d: f64, c: f64) -> GeneralOperator<f64> {
    let (mut rp, mut ci, mut va) = (vec![0], Vec::new(), Vec::new());
    for i in 0..n {
        if i > 0 {
            ci.push(u32::try_from(i - 1).unwrap());
            va.push(-d - c);
        }
        ci.push(u32::try_from(i).unwrap());
        va.push(2.0 * d + c);
        if i + 1 < n {
            ci.push(u32::try_from(i + 1).unwrap());
            va.push(-d);
        }
        rp.push(ci.len());
    }
    GeneralOperator::from_csr(n, &rp, &ci, &va).unwrap()
}

#[allow(clippy::cast_precision_loss)]
fn vector(n: usize) -> Vec<f64> {
    (0..n).map(|i| (0.37 * i as f64).sin() + 0.5).collect()
}

fn max_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn hessenberg_exponential_matches_a_long_series() {
    // 5×5 upper Hessenberg stored in a 6×6 array (the augmented layout).
    let dim = 6;
    let mut h = vec![0.0; dim * dim];
    for r in 0..5_usize {
        for c in r.saturating_sub(1)..5 {
            #[allow(clippy::cast_precision_loss)]
            let x = 0.3 * ((r * 7 + c * 3) % 5) as f64 - 0.6;
            h[r * dim + c] = x;
        }
    }
    let t = 2.5;
    let got = hess_exp_e1(&h, dim, 5, t).unwrap();
    // Reference: 200-term series of e^{tH}e₁ in 64 sub-steps.
    let mut y = vec![0.0; 5];
    y[0] = 1.0;
    for _ in 0..64 {
        let (mut term, mut acc) = (y.clone(), y.clone());
        for k in 1..=40 {
            let next: Vec<f64> = (0..5)
                .map(|r| {
                    (0..5).map(|c| h[r * dim + c] * term[c]).sum::<f64>() * t / 64.0 / f64::from(k)
                })
                .collect();
            term = next;
            acc.iter_mut().zip(&term).for_each(|(a, b)| *a += b);
        }
        y = acc;
    }
    assert!(max_err(&got, &y) <= 1e-13, "{}", max_err(&got, &y));
}

#[test]
fn krylov_combination_matches_taylor_and_reports_work() {
    let n = 200;
    let op = drift(n, 40.0, 12.0);
    let gen = CsrGenerator::from_general(&op, None).unwrap();
    let (w0, w1) = (vector(n), vec![0.25; n]);
    let tau = 0.5;
    let mut want = vec![0.0; n];
    let mut scratch = ScratchPool::new();
    phi_combination(&gen, tau, &[&w0, &w1], &mut want, &mut scratch).unwrap();
    let mut got = vec![0.0; n];
    let opts = KrylovPhiOptions::default().with_tol(1e-12);
    let rep = phi_combination_krylov(&gen, tau, &[&w0, &w1], &mut got, opts, &mut scratch).unwrap();
    assert!(max_err(&got, &want) <= 1e-10, "{}", max_err(&got, &want));
    assert!(rep.substeps >= 1 && rep.matvecs >= rep.substeps, "{rep:?}");
}

#[test]
fn zero_input_options_and_caps() {
    let n = 50;
    let op = drift(n, 40.0, 12.0);
    let mut scratch = ScratchPool::new();
    let mut out = vec![1.0; n];
    let zero = vec![0.0; n];
    let rep = expmv_krylov(
        &op,
        1.0,
        &zero,
        &mut out,
        KrylovPhiOptions::default(),
        &mut scratch,
    )
    .unwrap();
    assert!(out.iter().all(|&x| x == 0.0) && rep.matvecs == 0);
    let v = vector(n);
    for tol in [0.0, 1.0, f64::NAN] {
        let opts = KrylovPhiOptions::default().with_tol(tol);
        assert!(expmv_krylov(&op, 1.0, &v, &mut out, opts, &mut scratch).is_err());
    }
    let capped = KrylovPhiOptions::default().with_m(4).with_max_substeps(1);
    let r = expmv_krylov(&op, 5.0, &v, &mut out, capped, &mut scratch);
    assert!(
        matches!(r, Err(SemiflowError::ConvergenceFailed { max_iter: 1, .. })),
        "{r:?}"
    );
    // m ≥ n + p: one substep, exact projection (happy breakdown).
    let small = drift(6, 3.0, 1.0);
    let mut y = vec![0.0; 6];
    let rep = expmv_krylov(
        &small,
        0.7,
        &v[..6],
        &mut y,
        KrylovPhiOptions::default(),
        &mut scratch,
    )
    .unwrap();
    assert_eq!(rep.substeps, 1, "{rep:?}");
}

#[test]
fn dual_numbers_are_rejected() {
    use crate::{dual::Dual, generator_action::GeneratorAction};
    struct DualId;
    impl GeneratorAction<Dual<f64>> for DualId {
        fn dim(&self) -> usize {
            2
        }
        fn apply_generator(&self, src: &[Dual<f64>], dst: &mut [Dual<f64>]) {
            dst.copy_from_slice(src);
        }
        fn norm_bound(&self) -> Dual<f64> {
            Dual::constant(1.0)
        }
    }
    let w = [Dual::constant(1.0), Dual::variable(2.0)];
    let mut out = [Dual::constant(0.0); 2];
    let r = phi_combination_krylov(
        &DualId,
        Dual::constant(0.5),
        &[&w],
        &mut out,
        KrylovPhiOptions::default(),
        &mut ScratchPool::new(),
    );
    assert!(matches!(r, Err(SemiflowError::Unsupported { .. })), "{r:?}");
}
