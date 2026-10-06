//! `G_PHI_GENERAL_DENSE` (`RELEASE_BLOCKING`, ADR-0202 D3, closes MODEL gap G9).
//!
//! φ-functions on the non-symmetric `GeneralOperator` via
//! `CsrGenerator::from_general`. Operator: the `G_GENOP_DENSE` upwinded drifted
//! Fokker–Planck CSR (n = 12), with and without mass. Reference: a dense
//! augmented exponential. Tolerance 1e-10 mirrors `G_GENOP_DENSE` (non-normal).

// Small index -> f64 conversions on tiny test operators.
#![allow(clippy::cast_precision_loss)]

use semiflow::{
    general_operator::GeneralOperator,
    generator_action::{CsrGenerator, GeneratorAction},
    phi_action::{phi_action, phi_combination, PHI_MAX},
    scratch::ScratchPool,
};

mod phi_dense;
use phi_dense::{dense_phi_comb, dense_phi_k, lcg_vec, sup, sup_diff};

const N: usize = 12;
const TAU: f64 = 0.1;

/// Upwinded drifted Fokker–Planck `A` (non-symmetric by construction), as in `G_GENOP_DENSE`.
fn drifted_fokker_planck() -> Vec<f64> {
    let dx = 1.0 / (N as f64 - 1.0);
    let mut a = vec![0.0_f64; N * N];
    for i in 1..N - 1 {
        let diff = 0.05 / (dx * dx);
        let adv = 0.4 / dx;
        a[i * N + i - 1] = -diff - adv;
        a[i * N + i] = 2.0 * diff + adv;
        a[i * N + i + 1] = -diff;
    }
    a[0] = 1.0;
    a[(N - 1) * N + N - 1] = 1.0;
    a
}

fn to_operator(a: &[f64]) -> GeneralOperator<f64> {
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
    GeneralOperator::from_csr(N, &rp, &ci, &va).unwrap()
}

fn dense_g(a: &[f64], mass: Option<&[f64]>) -> Vec<f64> {
    (0..N * N)
        .map(|ij| -a[ij] / mass.map_or(1.0, |m| m[ij / N]))
        .collect()
}

fn run_case(a: &[f64], mass: Option<&[f64]>, label: &str) {
    let gen = CsrGenerator::from_general(&to_operator(a), mass).unwrap();
    let g = dense_g(a, mass);
    let mut sc = ScratchPool::new();
    let v = lcg_vec(31, N);
    for k in 0..=PHI_MAX {
        let mut got = vec![0.0_f64; N];
        phi_action(&gen, k, TAU, &v, &mut got, &mut sc).unwrap();
        let want = dense_phi_k(&g, N, TAU, k, &v);
        let err = sup_diff(&got, &want) / sup(&want).max(1.0);
        eprintln!("{label}: phi_{k} {err:.3e}");
        assert!(err <= 1e-10, "{label}: phi_{k} {err:.3e} > 1e-10");
    }
    check_combination(&gen, &g, label, &mut sc);
    check_transpose(&gen, &g, label);
}

fn check_combination(gen: &CsrGenerator<f64>, g: &[f64], label: &str, sc: &mut ScratchPool<f64>) {
    let w: Vec<Vec<f64>> = (0..=PHI_MAX).map(|k| lcg_vec(40 + k as u64, N)).collect();
    let refs: Vec<&[f64]> = w.iter().map(Vec::as_slice).collect();
    let mut got = vec![0.0_f64; N];
    phi_combination(gen, TAU, &refs, &mut got, sc).unwrap();
    let wt: Vec<Vec<f64>> = (1..=PHI_MAX)
        .map(|k| {
            w[k].iter()
                .map(|x| x * TAU.powi(i32::try_from(k).unwrap()))
                .collect()
        })
        .collect();
    let want = dense_phi_comb(g, N, TAU, &w[0], &wt);
    let err = sup_diff(&got, &want) / sup(&want).max(1.0);
    eprintln!("{label}: phi_combination {err:.3e}");
    assert!(err <= 1e-10, "{label}: phi_combination {err:.3e} > 1e-10");
}

fn check_transpose(gen: &CsrGenerator<f64>, g: &[f64], label: &str) {
    let v = lcg_vec(50, N);
    let mut got = vec![0.0_f64; N];
    gen.apply_generator_transpose(&v, &mut got);
    let want: Vec<f64> = (0..N)
        .map(|i| (0..N).map(|j| g[j * N + i] * v[j]).sum())
        .collect();
    let err = sup_diff(&got, &want) / sup(&want);
    eprintln!("{label}: transpose {err:.3e}");
    assert!(err <= 1e-14, "{label}: transpose {err:.3e} > 1e-14");
    // Non-vacuity: the operator really is non-symmetric (transpose != forward).
    let mut fwd = vec![0.0_f64; N];
    gen.apply_generator(&v, &mut fwd);
    assert!(
        sup_diff(&fwd, &got) > 1e-3,
        "{label}: operator looks symmetric"
    );
}

#[test]
fn g_phi_general_dense() {
    let a = drifted_fokker_planck();
    run_case(&a, None, "no-mass");
    let mass: Vec<f64> = (0..N).map(|i| 1.0 + i as f64 / 3.0).collect();
    run_case(&a, Some(&mass), "mass");
}
