//! Trace-shift gates for the augmented φ operator (`RELEASE_BLOCKING`, ADR-0210, math §67).
//!
//! * `G_PHI_SHIFT_COST`: a counting wrapper (forwarding `diagonal_shift`) around
//!   `NegLaplacianGenerator`, `CsrGenerator` (with and without a mass) and
//!   `DivFormGenerator` on 64-node diffusion operators: one `phi_combination` (p = 3)
//!   applies the generator exactly `s·m` times of `phi_sweep_probe`, the schedule is
//!   pinned (`(13, 30)` at `τ‖G‖ = 40`, `(117, 30)` — div-form `(115, 30)` — at 400),
//!   and that is ≤ 0.6× the unshifted `s·m` of `phi_cost_probe`. The reported
//!   `(ḡ, ‖G − ḡI‖_∞)` of every built-in generator equals the Gershgorin midpoint and
//!   norm of its dense matrix (to `1e−12` relative), and a NaN or negative reported
//!   norm falls back to the unshifted schedule.
//! * `G_PHI_SHIFT_DENSE`: on the shifted path (asserted active) `φ_k`, `k = 0..=3`,
//!   and a full `phi_combination` against the dense augmented exponential, n = 10
//!   SPD operator with mass contrast 50 (symmetric, general CSR, no mass), at
//!   `τ‖G‖ ∈ {2, 20, 200}`: `≤ 1e−12·max(1, ‖want‖∞)` — the `G_PHI_MASS_DENSE` bar.
//!   `CsrGenerator::from_symmetric(op, None)` and `NegLaplacianGenerator` agree bit
//!   for bit on the shifted path (the `csr_view` route of ADR-0210).
//! * `G_PHI_SHIFT_HOMOGENEITY`: the bitwise `φ(2⁴⁰v) = 2⁴⁰φ(v)` of
//!   `G_PHI_COST_V_INVARIANT`, on the shifted path (which that gate never reaches).

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)] // small indices; math names

use std::sync::atomic::{AtomicUsize, Ordering};

use semiflow::{
    general_operator::GeneralOperator,
    generator_action::{CsrGenerator, DivFormGenerator, GeneratorAction, NegLaplacianGenerator},
    phi_action::{phi_action, phi_action_batched, phi_combination, phi_cost_probe, PHI_MAX},
    phi_sweep_probe,
    scratch::ScratchPool,
    Diffusion4thChernoff, Grid1D, SymmetricOperator,
};

mod phi_dense;
use phi_dense::{dense_phi_k, lcg_vec, sup, sup_diff};

/// Forwards everything, counting generator applications.
struct Counted<G> {
    inner: G,
    calls: AtomicUsize,
}

impl<G: GeneratorAction<f64>> GeneratorAction<f64> for Counted<G> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.apply_generator(src, dst);
    }
    fn norm_bound(&self) -> f64 {
        self.inner.norm_bound()
    }
    fn diagonal_shift(&self) -> Option<(f64, f64)> {
        self.inner.diagonal_shift()
    }
}

/// Dense `n×n` SPD tridiagonal `A` (Neumann ends, conductances, reaction) and a
/// mass of contrast 50 — the operator family of `G_PHI_MASS_DENSE`.
fn build_dense(n: usize) -> (Vec<f64>, Vec<f64>) {
    let t: Vec<f64> = (0..n - 1)
        .map(|i| 0.8 + 0.4 * (i as f64 * 0.9).sin())
        .collect();
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        let left = if i > 0 { t[i - 1] } else { 0.0 };
        let right = if i + 1 < n { t[i] } else { 0.0 };
        a[i * n + i] = left + right + 0.2 * (1.0 + i as f64 / 3.0);
        if i > 0 {
            a[i * n + i - 1] = -left;
        }
        if i + 1 < n {
            a[i * n + i + 1] = -right;
        }
    }
    let mass = (0..n)
        .map(|i| 1.0 + 49.0 * (i as f64 / (n - 1) as f64).powi(2))
        .collect();
    (a, mass)
}

/// CSR triple of a dense matrix.
fn csr(a: &[f64], n: usize) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let (mut rp, mut ci, mut va) = (vec![0_usize], Vec::new(), Vec::new());
    for i in 0..n {
        for j in 0..n {
            if a[i * n + j] != 0.0 {
                ci.push(u32::try_from(j).unwrap());
                va.push(a[i * n + j]);
            }
        }
        rp.push(va.len());
    }
    (rp, ci, va)
}

fn symmetric(a: &[f64], n: usize) -> SymmetricOperator<f64> {
    let (rp, ci, va) = csr(a, n);
    SymmetricOperator::from_csr(n, &rp, &ci, &va, 1e-14).unwrap()
}

/// Generator applications of one `phi_combination` (p = 3) and the two probes;
/// `expected` pins the shifted schedule.
fn cost<G: GeneratorAction<f64>>(name: &str, gen: G, z: f64, expected: (u32, u32)) {
    let tau = z / gen.norm_bound();
    let n = gen.dim();
    let w: Vec<Vec<f64>> = (0..=PHI_MAX).map(|k| lcg_vec(7 + k as u64, n)).collect();
    let refs: Vec<&[f64]> = w.iter().map(Vec::as_slice).collect();
    let counted = Counted {
        inner: gen,
        calls: AtomicUsize::new(0),
    };
    let mut out = vec![0.0; n];
    phi_combination(&counted, tau, &refs, &mut out, &mut ScratchPool::new()).unwrap();
    let calls = counted.calls.load(Ordering::Relaxed) as u64;
    let (s, m) = phi_sweep_probe(&counted, tau);
    let (s0, m0) = phi_cost_probe(counted.norm_bound(), tau);
    let unshifted = u64::from(s0) * u64::from(m0);
    eprintln!("G_PHI_SHIFT_COST {name} z={z}: calls={calls} (s,m)=({s},{m}) unshifted={unshifted}");
    assert_eq!(
        calls,
        u64::from(s) * u64::from(m),
        "{name} z={z}: counter vs probe"
    );
    assert_eq!((s, m), expected, "{name} z={z}: schedule");
    assert!(
        10 * calls <= 6 * unshifted,
        "{name} z={z}: {calls} > 0.6 x {unshifted}"
    );
}

fn a_var(x: f64) -> f64 {
    0.5 + 0.2 * x.sin()
}
fn a_var_p(x: f64) -> f64 {
    0.2 * x.cos()
}
fn a_var_pp(x: f64) -> f64 {
    -0.2 * x.sin()
}

#[test]
fn g_phi_shift_cost() {
    let n = 64;
    let (a, mass) = build_dense(n);
    let op = symmetric(&a, n);
    let grid = Grid1D::new(-3.0, 3.0, n).unwrap();
    for (z, sym, div) in [(40.0, (13, 30), (13, 30)), (400.0, (117, 30), (115, 30))] {
        let neg = NegLaplacianGenerator::new(op.clone());
        bound_matches_dense("neg_laplacian", &neg);
        cost("neg_laplacian", neg, z, sym);
        let plain = CsrGenerator::from_symmetric(&op, None).unwrap();
        bound_matches_dense("csr", &plain);
        cost("csr", plain, z, sym);
        let massed = CsrGenerator::from_symmetric(&op, Some(&mass)).unwrap();
        bound_matches_dense("csr+mass", &massed);
        cost("csr+mass", massed, z, sym);
        let d4 = Diffusion4thChernoff::new(a_var, a_var_p, a_var_pp, 0.7, grid);
        let df = DivFormGenerator::new(d4);
        bound_matches_dense("div_form", &df);
        cost("div_form", df, z, div);
    }
    for bad in [f64::NAN, -1.0, f64::INFINITY] {
        let gen = BadShift(bad);
        let tau = 400.0 / gen.norm_bound();
        assert_eq!(
            phi_sweep_probe(&gen, tau),
            phi_cost_probe(gen.norm_bound(), tau),
            "norm {bad}: must fall back to the unshifted schedule"
        );
    }
}

/// The dense matrix of `gen` (columns `G·e_j`).
fn dense_of<G: GeneratorAction<f64>>(gen: &G) -> Vec<f64> {
    let n = gen.dim();
    let mut g = vec![0.0; n * n];
    let (mut e, mut col) = (vec![0.0; n], vec![0.0; n]);
    for j in 0..n {
        e[j] = 1.0;
        gen.apply_generator(&e, &mut col);
        for i in 0..n {
            g[i * n + j] = col[i];
        }
        e[j] = 0.0;
    }
    g
}

/// The reported `(ḡ, ‖G − ḡI‖_∞)` is the Gershgorin midpoint and norm of `G`.
fn bound_matches_dense<G: GeneratorAction<f64>>(name: &str, gen: &G) {
    let n = gen.dim();
    let g = dense_of(gen);
    let row = |i: usize| {
        let r: f64 = (0..n).filter(|&j| j != i).map(|j| g[i * n + j].abs()).sum();
        (g[i * n + i], r)
    };
    let hi = (0..n)
        .map(|i| row(i).0 + row(i).1)
        .fold(f64::NEG_INFINITY, f64::max);
    let lo = (0..n)
        .map(|i| row(i).0 - row(i).1)
        .fold(f64::INFINITY, f64::min);
    let mid = 0.5 * hi + 0.5 * lo;
    let norm = (0..n)
        .map(|i| (row(i).0 - mid).abs() + row(i).1)
        .fold(0.0, f64::max);
    let (gbar, ns) = gen
        .diagonal_shift()
        .expect("built-in generators report a shift");
    let scale = hi.abs().max(lo.abs());
    eprintln!("G_PHI_SHIFT_COST {name}: gbar {gbar:.6e} vs {mid:.6e}, norm {ns:.6e} vs {norm:.6e}");
    assert!(
        (gbar - mid).abs() <= 1e-12 * scale,
        "{name}: centre {gbar} vs {mid}"
    );
    assert!(
        (ns - norm).abs() <= 1e-12 * scale,
        "{name}: norm {ns} vs {norm}"
    );
}

/// A generator whose `diagonal_shift` reports an invalid norm.
struct BadShift(f64);

impl GeneratorAction<f64> for BadShift {
    fn dim(&self) -> usize {
        4
    }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        for i in 0..4 {
            dst[i] = -2.0 * src[i];
        }
    }
    fn norm_bound(&self) -> f64 {
        2.0
    }
    fn diagonal_shift(&self) -> Option<(f64, f64)> {
        Some((-0.1, self.0))
    }
}

/// Dense `G = −M⁻¹A`.
fn dense_g(a: &[f64], mass: &[f64], n: usize) -> Vec<f64> {
    (0..n * n).map(|ij| -a[ij] / mass[ij / n]).collect()
}

/// `φ_k` and one full combination on the shifted path vs the dense reference.
fn dense_check<G: GeneratorAction<f64>>(name: &str, gen: &G, g: &[f64], z: f64) {
    let n = gen.dim();
    let tau = z / gen.norm_bound();
    let shifted = phi_sweep_probe(gen, tau) != phi_cost_probe(gen.norm_bound(), tau);
    assert!(shifted, "{name} z={z}: shift not taken");
    let mut sc = ScratchPool::new();
    let v = lcg_vec(11, n);
    for k in 0..=PHI_MAX {
        let mut got = vec![0.0; n];
        phi_action(gen, k, tau, &v, &mut got, &mut sc).unwrap();
        let want = dense_phi_k(g, n, tau, k, &v);
        let err = sup_diff(&got, &want) / sup(&want).max(1.0);
        eprintln!("G_PHI_SHIFT_DENSE {name} z={z} phi_{k}: {err:.3e}");
        assert!(err <= 1e-12, "{name} z={z} phi_{k}: {err:e}");
    }
    let w: Vec<Vec<f64>> = (0..=PHI_MAX).map(|k| lcg_vec(20 + k as u64, n)).collect();
    let refs: Vec<&[f64]> = w.iter().map(Vec::as_slice).collect();
    let mut got = vec![0.0; n];
    phi_combination(gen, tau, &refs, &mut got, &mut sc).unwrap();
    // phi_combination(w) = e^{τG}w₀ + Σ_k τᵏ φ_k(τG) w_k, the reference summed from
    // the dense φ_k (each with unit coupling: the dense augmented exponential has no
    // η-scaling, so τᵏ ≈ 5e5 inside it would cost it digits).
    let mut want = dense_phi_k(g, n, tau, 0, &w[0]);
    for (k, wk) in w.iter().enumerate().skip(1) {
        let term = dense_phi_k(g, n, tau, k, wk);
        let tk = tau.powi(i32::try_from(k).unwrap());
        for (o, t) in want.iter_mut().zip(&term) {
            *o += tk * t;
        }
    }
    let err = sup_diff(&got, &want) / sup(&want).max(1.0);
    eprintln!("G_PHI_SHIFT_DENSE {name} z={z} combination: {err:.3e}");
    assert!(err <= 1e-12, "{name} z={z} combination: {err:e}");
}

#[test]
fn g_phi_shift_dense() {
    let n = 10;
    let (a, mass) = build_dense(n);
    let op = symmetric(&a, n);
    let ones = vec![1.0; n];
    let (rp, ci, va) = csr(&a, n);
    let general = GeneralOperator::from_csr(n, &rp, &ci, &va).unwrap();
    for z in [2.0, 20.0, 200.0] {
        dense_check(
            "csr+mass",
            &CsrGenerator::from_symmetric(&op, Some(&mass)).unwrap(),
            &dense_g(&a, &mass, n),
            z,
        );
        dense_check(
            "csr",
            &CsrGenerator::from_symmetric(&op, None).unwrap(),
            &dense_g(&a, &ones, n),
            z,
        );
        dense_check(
            "general+mass",
            &CsrGenerator::from_general(&general, Some(&mass)).unwrap(),
            &dense_g(&a, &mass, n),
            z,
        );
        dense_check(
            "neg_laplacian",
            &NegLaplacianGenerator::new(op.clone()),
            &dense_g(&a, &ones, n),
            z,
        );
        none_is_neg_laplacian(&op, z);
    }
}

/// `CsrGenerator::from_symmetric(op, None)` ≡ `NegLaplacianGenerator`, bit for bit,
/// ON the shifted path (`G_PHI_MASS_DENSE` (3) runs where the shift is not cheaper).
fn none_is_neg_laplacian(op: &SymmetricOperator<f64>, z: f64) {
    let (csr, neg) = (
        CsrGenerator::from_symmetric(op, None).unwrap(),
        NegLaplacianGenerator::new(op.clone()),
    );
    let tau = z / csr.norm_bound();
    let (s0, m0) = phi_cost_probe(csr.norm_bound(), tau);
    let shifted = phi_sweep_probe(&neg, tau);
    assert!(shifted.0 * shifted.1 < s0 * m0, "z={z}: shift not taken");
    let n = op.n();
    let v = lcg_vec(23, n);
    let mut sc = ScratchPool::new();
    let (mut a, mut b) = (vec![0.0; (PHI_MAX + 1) * n], vec![0.0; (PHI_MAX + 1) * n]);
    phi_action_batched(&csr, PHI_MAX, tau, &v, &mut a, &mut sc).unwrap();
    phi_action_batched(&neg, PHI_MAX, tau, &v, &mut b, &mut sc).unwrap();
    let same = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(
        same,
        "z={z}: CsrGenerator vs NegLaplacianGenerator bits differ"
    );
}

#[test]
fn g_phi_shift_homogeneity() {
    let n = 10;
    let (a, mass) = build_dense(n);
    let op = symmetric(&a, n);
    let gen = CsrGenerator::from_symmetric(&op, Some(&mass)).unwrap();
    let tau = 20.0 / gen.norm_bound();
    assert_ne!(
        phi_sweep_probe(&gen, tau),
        phi_cost_probe(gen.norm_bound(), tau),
        "shift not taken"
    );
    let mut sc = ScratchPool::new();
    let c = 2.0_f64.powi(40);
    let v = lcg_vec(5, n);
    let vc: Vec<f64> = v.iter().map(|x| x * c).collect();
    for k in 0..=PHI_MAX {
        let (mut y, mut yc) = (vec![0.0; n], vec![0.0; n]);
        phi_action(&gen, k, tau, &v, &mut y, &mut sc).unwrap();
        phi_action(&gen, k, tau, &vc, &mut yc, &mut sc).unwrap();
        for (a, b) in y.iter().zip(&yc) {
            assert_eq!(
                (a * c).to_bits(),
                b.to_bits(),
                "phi_{k}: not bitwise homogeneous"
            );
        }
    }
    let w: Vec<Vec<f64>> = (0..=PHI_MAX).map(|k| lcg_vec(30 + k as u64, n)).collect();
    let wc: Vec<Vec<f64>> = w
        .iter()
        .map(|x| x.iter().map(|y| y * c).collect())
        .collect();
    let (r, rc): (Vec<&[f64]>, Vec<&[f64]>) = (
        w.iter().map(Vec::as_slice).collect(),
        wc.iter().map(Vec::as_slice).collect(),
    );
    let (mut y, mut yc) = (vec![0.0; n], vec![0.0; n]);
    phi_combination(&gen, tau, &r, &mut y, &mut sc).unwrap();
    phi_combination(&gen, tau, &rc, &mut yc, &mut sc).unwrap();
    for (a, b) in y.iter().zip(&yc) {
        assert_eq!(
            (a * c).to_bits(),
            b.to_bits(),
            "combination: not bitwise homogeneous"
        );
    }
}
