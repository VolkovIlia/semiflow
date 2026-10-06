//! `G_PHI_COST_V_INVARIANT` (`RELEASE_BLOCKING`, ADR-0202 D4, math §62.4).
//!
//! The φ-action cost must not depend on the input vector. The pre-ADR bound
//! `τ‖A‖ + ‖v‖∞ + 1` made the substep count grow with `‖v‖∞`.
//!
//! 1. A counting `GeneratorAction` shows `phi_action(k=1)`, `phi_action_batched(p=3)`
//!    and `phi_combination` perform the same number of matvecs for
//!    `‖v‖∞ ∈ {1e-6, 1, 1e6}`, equal to `s·m` of `phi_cost_probe` (×(p+1) batched).
//! 2. Bitwise homogeneity `φ(2⁴⁰ v) = 2⁴⁰ φ(v)` (powers of two are exact).

// Small index -> f64 conversions on tiny test operators.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use semiflow::{
    generator_action::GeneratorAction,
    phi_action::{phi_action, phi_action_batched, phi_combination, phi_cost_probe, PHI_MAX},
    scratch::ScratchPool,
};
use std::sync::atomic::{AtomicUsize, Ordering};

mod phi_dense;
use phi_dense::lcg_vec;

const N: usize = 6;
const TAU: f64 = 0.5;
const SCALES: [f64; 3] = [1e-6, 1.0, 1e6];

/// `tridiag(1,−2,1)` that counts `apply_generator` calls.
struct CountingGen {
    calls: AtomicUsize,
}

impl GeneratorAction<f64> for CountingGen {
    fn dim(&self) -> usize {
        N
    }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
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

fn counted<R>(gen: &CountingGen, run: impl FnOnce() -> R) -> usize {
    gen.calls.store(0, Ordering::Relaxed);
    let _ = run();
    gen.calls.load(Ordering::Relaxed)
}

fn scaled(v: &[f64], c: f64) -> Vec<f64> {
    v.iter().map(|x| x * c).collect()
}

fn check_matvec_counts() {
    let gen = CountingGen {
        calls: AtomicUsize::new(0),
    };
    let (s, m) = phi_cost_probe(gen.norm_bound(), TAU);
    let per_sweep = (s * m) as usize;
    let base = lcg_vec(7, N);
    let mut scratch = ScratchPool::new();
    for &c in &SCALES {
        let v = scaled(&base, c);
        let mut out = vec![0.0_f64; N];
        let single = counted(&gen, || {
            phi_action(&gen, 1, TAU, &v, &mut out, &mut scratch).unwrap();
        });
        let mut outb = vec![0.0_f64; (PHI_MAX + 1) * N];
        let batched = counted(&gen, || {
            phi_action_batched(&gen, PHI_MAX, TAU, &v, &mut outb, &mut scratch).unwrap();
        });
        let zero = vec![0.0_f64; N];
        let (w1, w2, w3) = (scaled(&base, c), scaled(&base, 2.0 * c), scaled(&base, -c));
        let w: [&[f64]; 4] = [&zero, &w1, &w2, &w3];
        let comb = counted(&gen, || {
            phi_combination(&gen, TAU, &w, &mut out, &mut scratch).unwrap();
        });
        assert_eq!(single, per_sweep, "phi_action matvecs at scale {c:e}");
        assert_eq!(batched, (PHI_MAX + 1) * per_sweep, "batched at scale {c:e}");
        assert_eq!(comb, per_sweep, "phi_combination at scale {c:e}");
    }
}

fn bits_equal(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn check_bitwise_homogeneity() {
    let gen = CountingGen {
        calls: AtomicUsize::new(0),
    };
    let big = 2.0_f64.powi(40);
    let v = lcg_vec(11, N);
    let vb = scaled(&v, big);
    let mut scratch = ScratchPool::new();
    for k in 0..=PHI_MAX {
        let (mut a, mut b) = (vec![0.0; N], vec![0.0; N]);
        phi_action(&gen, k, TAU, &v, &mut a, &mut scratch).unwrap();
        phi_action(&gen, k, TAU, &vb, &mut b, &mut scratch).unwrap();
        assert!(
            bits_equal(&scaled(&a, big), &b),
            "phi_{k} not bitwise homogeneous"
        );
    }
    let (w1, w2) = (lcg_vec(12, N), lcg_vec(13, N));
    let (w1b, w2b) = (scaled(&w1, big), scaled(&w2, big));
    let zero = vec![0.0_f64; N];
    let (mut a, mut b) = (vec![0.0; N], vec![0.0; N]);
    phi_combination(&gen, TAU, &[&zero, &w1, &w2], &mut a, &mut scratch).unwrap();
    phi_combination(&gen, TAU, &[&zero, &w1b, &w2b], &mut b, &mut scratch).unwrap();
    assert!(
        bits_equal(&scaled(&a, big), &b),
        "phi_combination not bitwise homogeneous"
    );
}

/// Scaling EVERY `w_k` (including `w_0`) by `2^40` scales the whole output bitwise.
fn check_homogeneity_with_w0() {
    let gen = CountingGen {
        calls: AtomicUsize::new(0),
    };
    let big = 2.0_f64.powi(40);
    let ws: Vec<Vec<f64>> = (0..4).map(|k| lcg_vec(60 + k, N)).collect();
    let wb: Vec<Vec<f64>> = ws.iter().map(|w| scaled(w, big)).collect();
    let refs: Vec<&[f64]> = ws.iter().map(Vec::as_slice).collect();
    let refs_b: Vec<&[f64]> = wb.iter().map(Vec::as_slice).collect();
    let mut scratch = ScratchPool::new();
    let (mut a, mut b) = (vec![0.0; N], vec![0.0; N]);
    phi_combination(&gen, TAU, &refs, &mut a, &mut scratch).unwrap();
    phi_combination(&gen, TAU, &refs_b, &mut b, &mut scratch).unwrap();
    assert!(bits_equal(&scaled(&a, big), &b), "w0 != 0 not homogeneous");
}

/// A scale near the η clamp stays finite and linear to rounding.
fn check_near_clamp() {
    let gen = CountingGen {
        calls: AtomicUsize::new(0),
    };
    let huge = 1e300_f64;
    let v = lcg_vec(70, N);
    let mut scratch = ScratchPool::new();
    let (mut a, mut b) = (vec![0.0; N], vec![0.0; N]);
    phi_action(&gen, 2, TAU, &v, &mut a, &mut scratch).unwrap();
    phi_action(&gen, 2, TAU, &scaled(&v, huge), &mut b, &mut scratch).unwrap();
    let err = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x * huge - y).abs() / y.abs().max(1e-300))
        .fold(0.0_f64, f64::max);
    assert!(
        b.iter().all(|x| x.is_finite()) && err < 1e-12,
        "clamp case err={err:e}"
    );
}

/// `tridiag(1,−2,1)` in `f32`.
struct GenF32;

impl GeneratorAction<f32> for GenF32 {
    fn dim(&self) -> usize {
        N
    }
    fn apply_generator(&self, src: &[f32], dst: &mut [f32]) {
        for i in 0..N {
            let left = if i > 0 { src[i - 1] } else { 0.0 };
            let right = if i + 1 < N { src[i + 1] } else { 0.0 };
            dst[i] = left - 2.0 * src[i] + right;
        }
    }
    fn norm_bound(&self) -> f32 {
        4.0
    }
}

/// `f32` smoke: finite, close to the `f64` result, at two very different scales.
fn check_f32_smoke() {
    let base = lcg_vec(80, N);
    let mut scratch = ScratchPool::<f32>::new();
    let mut scratch64 = ScratchPool::<f64>::new();
    for &c in &[1e-3_f64, 1e3] {
        let v64 = scaled(&base, c);
        let v32: Vec<f32> = v64.iter().map(|&x| x as f32).collect();
        let (mut o32, mut o64) = (vec![0.0_f32; N], vec![0.0_f64; N]);
        phi_action(&GenF32, 1, 0.5_f32, &v32, &mut o32, &mut scratch).unwrap();
        let gen = CountingGen {
            calls: AtomicUsize::new(0),
        };
        phi_action(&gen, 1, TAU, &v64, &mut o64, &mut scratch64).unwrap();
        for (a, b) in o32.iter().zip(&o64) {
            assert!(
                a.is_finite() && (f64::from(*a) - b).abs() <= 1e-4 * c,
                "f32 scale {c:e}"
            );
        }
    }
}

#[test]
fn g_phi_cost_v_invariant() {
    check_matvec_counts();
    check_bitwise_homogeneity();
    check_homogeneity_with_w0();
    check_near_clamp();
    check_f32_smoke();
}
