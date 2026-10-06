//! `G_PHI_COMBINATION_DENSE` (`RELEASE_BLOCKING`, ADR-0202 D3, math §62.4.a).
//!
//! `phi_combination(τ, [w₀..w_p])` equals `Σ τ^k φ_k(τG) w_k`, checked against
//! the eigen-exact DST-I oracle of `G_PHI_AUG_DENSE` (6-node `tridiag(1,−2,1)`,
//! τ = 0.5, p ∈ {0,1,2,3}) and against the sum of separate `phi_action` calls.

// Small index -> f64 conversions on tiny test operators.
#![allow(clippy::cast_precision_loss)]

use semiflow::{
    generator_action::GeneratorAction,
    phi_action::{phi_action, phi_combination, PHI_MAX},
    scratch::ScratchPool,
    SemiflowError,
};

mod phi_dense;
use phi_dense::{eigen_exact_phi_k, lcg_vec, sup, sup_diff};

const N: usize = 6;
const TAU: f64 = 0.5;

struct TriDiagGen;

impl GeneratorAction<f64> for TriDiagGen {
    fn dim(&self) -> usize {
        N
    }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        for i in 0..N {
            let left = if i > 0 { src[i - 1] } else { 0.0 };
            let right = if i + 1 < N { src[i + 1] } else { 0.0 };
            dst[i] = left - 2.0 * src[i] + right;
        }
    }
    fn norm_bound(&self) -> f64 {
        4.0
    }
}

/// `τ^k` weighted coefficients and the maximum weighted magnitude.
fn weights(w: &[Vec<f64>]) -> f64 {
    w.iter()
        .enumerate()
        .map(|(k, wk)| sup(wk) * TAU.powi(i32::try_from(k).unwrap()))
        .fold(0.0_f64, f64::max)
}

fn eigen_oracle(w: &[Vec<f64>]) -> Vec<f64> {
    let mut acc = vec![0.0_f64; N];
    for (k, wk) in w.iter().enumerate() {
        let phi = eigen_exact_phi_k(k, TAU, 1.0, wk);
        let tk = TAU.powi(i32::try_from(k).unwrap());
        for (a, p) in acc.iter_mut().zip(&phi) {
            *a += tk * p;
        }
    }
    acc
}

fn separate_calls(w: &[Vec<f64>], scratch: &mut ScratchPool<f64>) -> Vec<f64> {
    let mut acc = vec![0.0_f64; N];
    for (k, wk) in w.iter().enumerate() {
        let mut out = vec![0.0_f64; N];
        phi_action(&TriDiagGen, k, TAU, wk, &mut out, scratch).unwrap();
        let tk = TAU.powi(i32::try_from(k).unwrap());
        for (a, o) in acc.iter_mut().zip(&out) {
            *a += tk * o;
        }
    }
    acc
}

fn check_order(p: usize, scratch: &mut ScratchPool<f64>) {
    let w: Vec<Vec<f64>> = (0..=p).map(|k| lcg_vec(100 + k as u64, N)).collect();
    let refs: Vec<&[f64]> = w.iter().map(Vec::as_slice).collect();
    let mut got = vec![0.0_f64; N];
    phi_combination(&TriDiagGen, TAU, &refs, &mut got, scratch).unwrap();
    let scale = weights(&w);
    let want = eigen_oracle(&w);
    let err_exact = sup_diff(&got, &want) / scale;
    let err_sep = sup_diff(&got, &separate_calls(&w, scratch)) / scale;
    eprintln!("p={p}: vs eigen {err_exact:.3e}   vs separate {err_sep:.3e}");
    assert!(sup(&got) > 0.01, "p={p}: vacuous output");
    assert!(
        err_exact <= 1e-12,
        "p={p}: vs eigen oracle {err_exact:.3e} > 1e-12"
    );
    assert!(
        err_sep <= 1e-13,
        "p={p}: vs separate calls {err_sep:.3e} > 1e-13"
    );
}

#[test]
fn g_phi_combination_dense() {
    let mut scratch = ScratchPool::new();
    for p in 0..=PHI_MAX {
        check_order(p, &mut scratch);
    }
}

#[test]
fn phi_combination_rejects_bad_input() {
    let mut scratch = ScratchPool::new();
    let ok = lcg_vec(1, N);
    let short = lcg_vec(2, N - 1);
    let mut out = vec![0.0_f64; N];
    let mut nan = ok.clone();
    nan[2] = f64::NAN;
    let five: Vec<&[f64]> = vec![&ok; PHI_MAX + 2];
    let cases: [(&[&[f64]], f64); 5] = [
        (&[], TAU),
        (&five, TAU),
        (&[&ok, &short], TAU),
        (&[&ok, &nan], TAU),
        (&[&ok], -1.0),
    ];
    for (i, (w, tau)) in cases.iter().enumerate() {
        let r = phi_combination(&TriDiagGen, *tau, w, &mut out, &mut scratch);
        assert!(
            matches!(r, Err(SemiflowError::DomainViolation { .. })),
            "case {i}: expected DomainViolation, got {r:?}"
        );
    }
    let mut short_out = vec![0.0_f64; N - 1];
    let r = phi_combination(&TriDiagGen, TAU, &[&ok], &mut short_out, &mut scratch);
    assert!(matches!(r, Err(SemiflowError::DomainViolation { .. })));
}
