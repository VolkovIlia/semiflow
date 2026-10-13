//! Adaptive Krylov φ gates (`RELEASE_BLOCKING`, ADR-0211, math §68).
//!
//! * `G_PHI_KRYLOV_DENSE`: `φ_k` (`k ≤ 3`) and a full combination for non-normal
//!   generators (drifted Fokker–Planck, birth–death inventory ladder, n = 12, with and
//!   without a lumped mass) against the dense augmented exponential, `tol = 1e−12`,
//!   `τ‖G‖ ∈ {1, 30, 300}`, Krylov dimension 30 (the whole augmented space: exact
//!   projection) and 6 (adaptive substeps, asserted > 1 at `τ‖G‖ ≥ 30`):
//!   `≤ 1e−10·max(1, ‖want‖∞)`.
//! * `G_PHI_KRYLOV_COST`: drifted Fokker–Planck, n = 1000, `τ‖G‖ = 4·10³`: the Krylov
//!   path uses ≥ 20× fewer generator applications than the Taylor path actually
//!   used (shift-aware `phi_sweep_probe`), counted exactly, and agrees with it to
//!   `1e−8·‖ref‖∞`.
//! * `G_PHI_KRYLOV_TOL`: same operator at `τ‖G‖ = 4·10²`, `tol ∈ {1e−6, 1e−9, 1e−12}`:
//!   error against the Taylor result `≤ 10·tol·‖w‖₂` (the tolerance is relative to the
//!   augmented start vector), and a tighter `tol` never costs fewer applications.
//! * `G_ETDRK4_KRYLOV_ORDER`: ETDRK4 with `PhiSolver::Krylov` keeps order 4 on the
//!   Allen–Cahn problem of `G_ETDRK4_ORDER` (slope ∈ [3.7, 4.3]) and matches the
//!   Taylor driver to `1e−12`.

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)] // small indices; math names

use std::sync::atomic::{AtomicUsize, Ordering};

use semiflow::{
    general_operator::GeneralOperator,
    generator_action::{CsrGenerator, GeneratorAction},
    phi_action::{phi_combination, PHI_MAX},
    phi_combination_krylov, phi_sweep_probe,
    scratch::ScratchPool,
    AllenCahn, Etdrk4, KrylovPhiOptions, PhiSolver,
};

mod phi_dense;
use phi_dense::{dense_phi_k, lcg_vec, sup, sup_diff};

/// Dense row-major `A` of the drifted Fokker–Planck generator `G = −A`
/// (conservative upwind drift `b(x) = 1.5 − x`, diffusion `d`) on `[−3, 3]`.
fn fokker_planck(n: usize, d: f64) -> Vec<f64> {
    let dx = 6.0 / (n - 1) as f64;
    let mut g = vec![0.0; n * n];
    for i in 0..n - 1 {
        // Flux across the face i+½: diffusion + upwind drift, into i+1 from i.
        let x = -3.0 + (i as f64 + 0.5) * dx;
        let b = 1.5 - x;
        let (bp, bm) = (b.max(0.0), (-b).max(0.0));
        let (from_i, from_ip) = (d / (dx * dx) + bp / dx, d / (dx * dx) + bm / dx);
        g[(i + 1) * n + i] += from_i;
        g[i * n + i] -= from_i;
        g[i * n + i + 1] += from_ip;
        g[(i + 1) * n + i + 1] -= from_ip;
    }
    g.iter().map(|v| -v).collect()
}

/// Dense `A` of a birth–death ladder (orders arrive at rate λ, are served at μ·min(i, 3)).
fn inventory_ladder(n: usize) -> Vec<f64> {
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        if i + 1 < n {
            g[(i + 1) * n + i] += 2.0;
            g[i * n + i] -= 2.0;
        }
        if i > 0 {
            let mu = 1.3 * (i.min(3) as f64);
            g[(i - 1) * n + i] += mu;
            g[i * n + i] -= mu;
        }
    }
    g.iter().map(|v| -v).collect()
}

fn general(a: &[f64], n: usize) -> GeneralOperator<f64> {
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
    GeneralOperator::from_csr(n, &rp, &ci, &va).unwrap()
}

fn dense_g(a: &[f64], mass: &[f64], n: usize) -> Vec<f64> {
    (0..n * n).map(|ij| -a[ij] / mass[ij / n]).collect()
}

/// `Σ_k τᵏ φ_k(τG) w_k` from the dense `φ_k`.
fn dense_combination(g: &[f64], n: usize, tau: f64, w: &[Vec<f64>]) -> Vec<f64> {
    let mut want = dense_phi_k(g, n, tau, 0, &w[0]);
    for (k, wk) in w.iter().enumerate().skip(1) {
        let term = dense_phi_k(g, n, tau, k, wk);
        let tk = tau.powi(i32::try_from(k).unwrap());
        for (o, t) in want.iter_mut().zip(&term) {
            *o += tk * t;
        }
    }
    want
}

fn check_dense(name: &str, gen: &CsrGenerator<f64>, g: &[f64], z: f64, m: usize) {
    let n = gen.dim();
    let tau = z / gen.norm_bound();
    let opts = KrylovPhiOptions::default().with_tol(1e-12).with_m(m);
    let mut sc = ScratchPool::new();
    let w: Vec<Vec<f64>> = (0..=PHI_MAX).map(|k| lcg_vec(40 + k as u64, n)).collect();
    for k in 0..=PHI_MAX {
        // φ_k(τG)v alone = combination with v in slot k and τᵏ divided out.
        let mut slots = vec![vec![0.0; n]; k + 1];
        slots[k].clone_from(&w[k]);
        let refs: Vec<&[f64]> = slots.iter().map(Vec::as_slice).collect();
        let mut got = vec![0.0; n];
        phi_combination_krylov(gen, tau, &refs, &mut got, opts, &mut sc).unwrap();
        let tk = tau.powi(i32::try_from(k).unwrap());
        let want: Vec<f64> = dense_phi_k(g, n, tau, k, &w[k])
            .iter()
            .map(|x| x * tk)
            .collect();
        let err = sup_diff(&got, &want) / sup(&want).max(1.0);
        eprintln!("G_PHI_KRYLOV_DENSE {name} z={z} phi_{k}: {err:.3e}");
        assert!(err <= 1e-10, "{name} z={z} phi_{k}: {err:e}");
    }
    let refs: Vec<&[f64]> = w.iter().map(Vec::as_slice).collect();
    let mut got = vec![0.0; n];
    let rep = phi_combination_krylov(gen, tau, &refs, &mut got, opts, &mut sc).unwrap();
    let want = dense_combination(g, n, tau, &w);
    let err = sup_diff(&got, &want) / sup(&want).max(1.0);
    eprintln!("G_PHI_KRYLOV_DENSE {name} z={z} m={m} combination: {err:.3e} {rep:?}");
    assert!(err <= 1e-10, "{name} z={z} combination: {err:e}");
    // m = 6 < n + p: a genuine adaptive run (several substeps at the larger z).
    if m < n && z >= 30.0 {
        assert!(rep.substeps > 1, "{name} z={z}: {rep:?}");
    }
}

#[test]
fn g_phi_krylov_dense() {
    let n = 12;
    let mass: Vec<f64> = (0..n).map(|i| 0.5 + 0.2 * i as f64).collect();
    let ones = vec![1.0; n];
    for (name, a) in [
        ("fokker_planck", fokker_planck(n, 0.4)),
        ("ladder", inventory_ladder(n)),
    ] {
        let op = general(&a, n);
        for z in [1.0, 30.0, 300.0] {
            for m in [6, 30] {
                let plain = CsrGenerator::from_general(&op, None).unwrap();
                check_dense(name, &plain, &dense_g(&a, &ones, n), z, m);
                let massed = CsrGenerator::from_general(&op, Some(&mass)).unwrap();
                check_dense(
                    &format!("{name}+mass"),
                    &massed,
                    &dense_g(&a, &mass, n),
                    z,
                    m,
                );
            }
        }
    }
}

/// Forwards everything, counting generator applications.
struct Counted<'a> {
    inner: &'a CsrGenerator<f64>,
    calls: AtomicUsize,
}

impl GeneratorAction<f64> for Counted<'_> {
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

/// `(krylov result, its exact count, report matvecs, taylor result, taylor s·m)`.
fn both_paths(
    gen: &CsrGenerator<f64>,
    z: f64,
    tol: f64,
) -> (Vec<f64>, usize, usize, Vec<f64>, u64, f64) {
    let n = gen.dim();
    let tau = z / gen.norm_bound();
    let w: Vec<Vec<f64>> = (0..=1).map(|k| lcg_vec(60 + k, n)).collect();
    let refs: Vec<&[f64]> = w.iter().map(Vec::as_slice).collect();
    let counted = Counted {
        inner: gen,
        calls: AtomicUsize::new(0),
    };
    let mut sc = ScratchPool::new();
    let mut kry = vec![0.0; n];
    let opts = KrylovPhiOptions::default().with_tol(tol);
    let rep = phi_combination_krylov(&counted, tau, &refs, &mut kry, opts, &mut sc).unwrap();
    let mut tay = vec![0.0; n];
    phi_combination(gen, tau, &refs, &mut tay, &mut sc).unwrap();
    let (s, m) = phi_sweep_probe(gen, tau);
    let w_norm = refs
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f64>())
        .sum::<f64>()
        .sqrt();
    (
        kry,
        counted.calls.load(Ordering::Relaxed),
        rep.matvecs,
        tay,
        u64::from(s) * u64::from(m),
        w_norm,
    )
}

#[test]
fn g_phi_krylov_cost() {
    let n = 1000;
    let op = general(&fokker_planck(n, 1.0), n);
    let gen = CsrGenerator::from_general(&op, None).unwrap();
    let (kry, calls, matvecs, tay, taylor, _) = both_paths(&gen, 4e3, 1e-10);
    let err = sup_diff(&kry, &tay) / sup(&tay);
    eprintln!(
        "G_PHI_KRYLOV_COST: krylov {calls} vs taylor {taylor} (x{:.1}), rel diff {err:.2e}",
        taylor as f64 / calls as f64
    );
    assert_eq!(calls, matvecs, "report vs counter");
    assert!(
        20 * calls as u64 <= taylor,
        "krylov {calls} vs taylor {taylor}"
    );
    assert!(err <= 1e-8, "krylov vs taylor: {err:e}");
}

#[test]
fn g_phi_krylov_tol() {
    let n = 1000;
    let op = general(&fokker_planck(n, 1.0), n);
    let gen = CsrGenerator::from_general(&op, None).unwrap();
    let mut prev_calls = 0;
    for tol in [1e-6, 1e-9, 1e-12] {
        let (kry, calls, _, tay, _, w_norm) = both_paths(&gen, 4e2, tol);
        let err = sup_diff(&kry, &tay);
        eprintln!(
            "G_PHI_KRYLOV_TOL tol={tol:e}: err={err:.2e} (bound {:.2e}) calls={calls}",
            10.0 * tol * w_norm
        );
        assert!(err <= 10.0 * tol * w_norm, "tol={tol:e}: {err:e}");
        // The tolerance drives the work: a tighter one never costs less.
        assert!(calls >= prev_calls, "tol={tol:e}: {calls} < {prev_calls}");
        prev_calls = calls;
    }
}

struct PeriodLaplacian(f64);

impl GeneratorAction<f64> for PeriodLaplacian {
    fn dim(&self) -> usize {
        8
    }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        for i in 0..8 {
            let (im, ip) = ((i + 7) % 8, (i + 1) % 8);
            dst[i] = self.0 * (src[im] - 2.0 * src[i] + src[ip]);
        }
    }
    fn norm_bound(&self) -> f64 {
        4.0 * self.0
    }
}

fn integrate_ac(h: f64, n_steps: usize, u0: &[f64], solver: PhiSolver) -> Vec<f64> {
    let driver = Etdrk4::new(PeriodLaplacian(0.01 * 64.0), AllenCahn::<f64>::new(), h)
        .unwrap()
        .with_phi_solver(solver);
    let mut out = u0.to_vec();
    driver
        .integrate(u0, n_steps, &mut out, &mut ScratchPool::new())
        .unwrap();
    out
}

#[test]
fn g_etdrk4_krylov_order() {
    let u0: Vec<f64> = (0..8)
        .map(|i| 0.5 * (2.0 * std::f64::consts::PI * f64::from(i) / 8.0).sin())
        .collect();
    let kry = PhiSolver::Krylov(KrylovPhiOptions::default().with_tol(1e-14));
    let h = 0.05;
    let u_h = integrate_ac(h, 4, &u0, kry);
    let u_h2 = integrate_ac(h / 2.0, 8, &u0, kry);
    let u_ref = integrate_ac(h / 4.0, 16, &u0, kry);
    let slope = (sup_diff(&u_h, &u_ref) / sup_diff(&u_h2, &u_ref)).log2();
    eprintln!("G_ETDRK4_KRYLOV_ORDER: slope {slope:.3}");
    assert!((3.7..=4.3).contains(&slope), "slope {slope}");
    let taylor = integrate_ac(h, 4, &u0, PhiSolver::Taylor);
    assert!(
        sup_diff(&u_h, &taylor) <= 1e-12,
        "krylov vs taylor driver: {:e}",
        sup_diff(&u_h, &taylor)
    );
}
