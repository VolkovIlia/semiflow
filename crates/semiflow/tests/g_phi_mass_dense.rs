//! `G_PHI_MASS_DENSE` (`RELEASE_BLOCKING`, ADR-0202 D3, math §62.3).
//!
//! φ-functions with a diagonal mass: `G = −M⁻¹A`, n = 10 conservative operator
//! plus reaction, mass contrast 50.
//! (1) `CsrGenerator::from_symmetric(op, Some(m))` `φ_k`, k = 0..=3, vs a dense
//!     augmented exponential (≤ 1e-12).
//! (2) vs the §55.3 congruence route `M^{-½} φ_k(−τÂ) M^{½} v` (≤ 1e-13).
//! (3) `mass = None` is bitwise identical to `NegLaplacianGenerator`.
//! (4) `apply_generator_transpose` vs dense `(−M⁻¹A)ᵀ v` (≤ 1e-15 relative).

// Small index -> f64 conversions on tiny test operators.
#![allow(clippy::cast_precision_loss)]

use semiflow::{
    generator_action::{CsrGenerator, GeneratorAction, NegLaplacianGenerator},
    phi_action::{phi_action, phi_action_batched, PHI_MAX},
    scratch::ScratchPool,
    SemiflowError, SymmetricOperator,
};

mod phi_dense;
use phi_dense::{dense_phi_k, lcg_vec, sup, sup_diff};

const N: usize = 10;
const TAU: f64 = 0.3;

/// Dense SPD tridiagonal `A` (Neumann ends, conductances `t_i`, reaction `c_i`) and mass.
fn build_dense() -> (Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..N - 1)
        .map(|i| 0.8 + 0.4 * (i as f64 * 0.9).sin())
        .collect();
    let mut a = vec![0.0_f64; N * N];
    for i in 0..N {
        let left = if i > 0 { t[i - 1] } else { 0.0 };
        let right = if i + 1 < N { t[i] } else { 0.0 };
        a[i * N + i] = left + right + 0.2 * (1.0 + i as f64 / 3.0);
        if i > 0 {
            a[i * N + i - 1] = -left;
        }
        if i + 1 < N {
            a[i * N + i + 1] = -right;
        }
    }
    let mass: Vec<f64> = (0..N)
        .map(|i| 1.0 + 49.0 * (i as f64 / (N - 1) as f64).powi(2))
        .collect();
    (a, mass)
}

fn to_operator(a: &[f64]) -> SymmetricOperator<f64> {
    let (mut rp, mut ci, mut va) = (vec![0_usize], Vec::new(), Vec::new());
    for i in 0..N {
        for j in 0..N {
            if a[i * N + j] != 0.0 {
                ci.push(u32::try_from(j).unwrap());
                va.push(a[i * N + j]);
            }
        }
        rp.push(va.len());
    }
    SymmetricOperator::from_csr(N, &rp, &ci, &va, 1e-14).unwrap()
}

/// Dense `G = −M⁻¹A`.
fn dense_g(a: &[f64], mass: &[f64]) -> Vec<f64> {
    (0..N * N).map(|ij| -a[ij] / mass[ij / N]).collect()
}

fn check_vs_dense(gen: &CsrGenerator<f64>, g: &[f64], v: &[f64], sc: &mut ScratchPool<f64>) {
    for k in 0..=PHI_MAX {
        let mut got = vec![0.0_f64; N];
        phi_action(gen, k, TAU, v, &mut got, sc).unwrap();
        let want = dense_phi_k(g, N, TAU, k, v);
        let err = sup_diff(&got, &want) / sup(&want).max(1.0);
        eprintln!("(1) phi_{k} vs dense: {err:.3e}");
        assert!(err <= 1e-12, "(1) phi_{k}: {err:.3e} > 1e-12");
    }
}

fn check_vs_congruence(
    op: &SymmetricOperator<f64>,
    mass: &[f64],
    gen: &CsrGenerator<f64>,
    v: &[f64],
    sc: &mut ScratchPool<f64>,
) {
    let hat = NegLaplacianGenerator::new(op.lumped_congruence(mass).unwrap());
    let vh: Vec<f64> = v.iter().zip(mass).map(|(x, m)| x * m.sqrt()).collect();
    for k in 0..=PHI_MAX {
        let (mut got, mut via) = (vec![0.0_f64; N], vec![0.0_f64; N]);
        phi_action(gen, k, TAU, v, &mut got, sc).unwrap();
        phi_action(&hat, k, TAU, &vh, &mut via, sc).unwrap();
        for (x, m) in via.iter_mut().zip(mass) {
            *x /= m.sqrt();
        }
        let err = sup_diff(&got, &via) / sup(&via).max(1.0);
        eprintln!("(2) phi_{k} vs congruence: {err:.3e}");
        assert!(err <= 1e-13, "(2) phi_{k}: {err:.3e} > 1e-13");
    }
}

fn bits_equal(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn check_none_is_neg_laplacian(op: &SymmetricOperator<f64>, v: &[f64], sc: &mut ScratchPool<f64>) {
    let csr = CsrGenerator::from_symmetric(op, None).unwrap();
    let neg = NegLaplacianGenerator::new(op.clone());
    assert_eq!(csr.norm_bound().to_bits(), neg.norm_bound().to_bits());
    for k in 0..=PHI_MAX {
        let (mut a, mut b) = (vec![0.0_f64; N], vec![0.0_f64; N]);
        phi_action(&csr, k, TAU, v, &mut a, sc).unwrap();
        phi_action(&neg, k, TAU, v, &mut b, sc).unwrap();
        assert!(bits_equal(&a, &b), "(3) phi_{k} differs bitwise");
    }
    let (mut a, mut b) = (vec![0.0_f64; 4 * N], vec![0.0_f64; 4 * N]);
    phi_action_batched(&csr, PHI_MAX, TAU, v, &mut a, sc).unwrap();
    phi_action_batched(&neg, PHI_MAX, TAU, v, &mut b, sc).unwrap();
    assert!(bits_equal(&a, &b), "(3) phi_action_batched differs bitwise");
}

fn check_transpose(gen: &CsrGenerator<f64>, g: &[f64], v: &[f64]) {
    let mut got = vec![0.0_f64; N];
    gen.apply_generator_transpose(v, &mut got);
    let want: Vec<f64> = (0..N)
        .map(|i| (0..N).map(|j| g[j * N + i] * v[j]).sum())
        .collect();
    let err = sup_diff(&got, &want) / sup(&want);
    eprintln!("(4) transpose: {err:.3e}");
    assert!(err <= 1e-15, "(4) transpose {err:.3e} > 1e-15");
}

#[test]
fn g_phi_mass_dense() {
    let (a, mass) = build_dense();
    let op = to_operator(&a);
    let g = dense_g(&a, &mass);
    let gen = CsrGenerator::from_symmetric(&op, Some(&mass)).unwrap();
    let v = lcg_vec(21, N);
    let mut sc = ScratchPool::new();
    check_vs_dense(&gen, &g, &v, &mut sc);
    check_vs_congruence(&op, &mass, &gen, &v, &mut sc);
    check_none_is_neg_laplacian(&op, &v, &mut sc);
    check_transpose(&gen, &g, &lcg_vec(22, N));
}

#[test]
fn csr_generator_rejects_bad_mass() {
    let (a, mass) = build_dense();
    let op = to_operator(&a);
    let mut neg = mass.clone();
    neg[3] = -1.0;
    let mut nan = mass.clone();
    nan[0] = f64::NAN;
    for bad in [&mass[..N - 1], &neg[..], &nan[..]] {
        let r = CsrGenerator::from_symmetric(&op, Some(bad));
        assert!(matches!(r, Err(SemiflowError::DomainViolation { .. })));
    }
}
