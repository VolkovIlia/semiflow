//! `G_GRIDLESS_CV` — 4-arm CV-controlled decisive experiment (Deliverable 2, §3, v9.0.0)
//!
//! **Purpose:** Measure whether deterministic branching + exact-moment CV achieves
//! ≥ 2× lower MSE than MC + the *identical* CV (arm C vs arm B).
//!
//! ## Anti-gaming design (§7 guard #1)
//!
//! The FOUR arms (A: MC, B: MC+CV, C: branching+CV, D: branching) differ in
//! **exactly two axes** at most: how the P sample points are produced, and
//! whether the CV is applied. The CV machinery (g, E[g], β̂) is **identical**
//! for arms B and C. The comparison is **C vs B ONLY** — never C vs A.
//!
//! ## Pre-registered constants (HARD-CODED, not tuned)
//!
//! T=1.0, P=4096, R=64, γ=0.5, DIMS={2,4,8}
//! `n_steps`: d=2→8, d=4→4, d=8→2
//!
//! ## Pre-registered verdict (§3.3)
//!
//! Variance sub-claim SALVAGED iff MSE(B)/MSE(C) ≥ 2.0 at BOTH d=4 AND d=8.
//! Variance sub-claim REFUTED (sharper) iff < 2.0 at d=4 OR d=8.
//!
//! ## CV construction (§2.4, guard #4)
//!
//! Control: g(x) = ∏_j `cos(η_j` `x_j`), `η_j` = `γ·ξ_j` (γ=0.5)
//! E[g]   = ∏_j `exp(−η_j²` `a_j` T)   (closed-form from §38.7, computed before sampling)
//! Estimator: `Ŷ_CV` = mean(f) − β̂·(mean(g) − E[g])
//!            β̂ = Cov(f,g)/Var(g) estimated per-arm from the SAME P samples.
//!
//! Run:
//!   cargo test -p semiflow-core --features slow-tests \
//!     --test `g_gridless_cv` -- --ignored --nocapture

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)]

use semiflow::{
    ChernoffFunction, GridlessChernoff, MeasureState, ParticleReduction, ScratchPool,
};

extern crate alloc;

// ═══════════════════════════════════════════════════════════════════════════════
// §A — LCG PRNG (copy-inline from g_gridless_rqmc.rs)
// ═══════════════════════════════════════════════════════════════════════════════

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Box-Muller standard normal.
    fn next_std_normal(&mut self) -> f64 {
        let u1 = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0 + 1e-15;
        let u2 = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        let r = libm::sqrt(-2.0 * libm::log(u1));
        r * libm::cos(core::f64::consts::TAU * u2)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §B — Per-axis anisotropic model (same as g_gridless.rs)
// ═══════════════════════════════════════════════════════════════════════════════

fn a_j(j: usize) -> f64 {
    0.5 * (1.0 + 0.1 * j as f64)
}
fn xi_j(j: usize) -> f64 {
    1.0 / (1.0 + 0.05 * j as f64)
}

/// Closed-form E[f] = ∏_j exp(−T ξ_j² a_j)  (truth for functional f).
fn truth_d(d: usize, t: f64) -> f64 {
    (0..d)
        .map(|j| libm::exp(-t * xi_j(j) * xi_j(j) * a_j(j)))
        .product()
}

/// Product functional f(x) = ∏_j cos(ξ_j x_j).
fn functional(pos: &[f64], d: usize) -> f64 {
    (0..d).map(|j| libm::cos(xi_j(j) * pos[j])).product()
}

// ═══════════════════════════════════════════════════════════════════════════════
// §C — CV kit (§2.4, guard #4)
//
// Control: g(x) = ∏_j cos(η_j x_j), η_j = GAMMA * ξ_j
// E[g] = ∏_j exp(−η_j² a_j T)  — closed-form, computed before sampling
// β̂ = Cov(f,g)/Var(g) per-arm from the arm's own P samples (Bessel-corrected)
// Ŷ_CV = mean(f) − β̂ · (mean(g) − E[g])
// ═══════════════════════════════════════════════════════════════════════════════

/// Pre-registered control frequency factor γ (§2.4, NOT tuned post-hoc).
const GAMMA: f64 = 0.5;

/// E[g] = ∏_j exp(−(γ ξ_j)² a_j T) — closed-form from §38.7, guard #4.
/// This is computed BEFORE sampling; it is a pure function of (d, T, γ, a_j, ξ_j).
fn e_g(d: usize, t: f64) -> f64 {
    (0..d)
        .map(|j| {
            let eta_j = GAMMA * xi_j(j);
            libm::exp(-eta_j * eta_j * a_j(j) * t)
        })
        .product()
}

/// Control value g(x) = ∏_j cos(η_j x_j), η_j = γ·ξ_j.
fn control(pos: &[f64], d: usize) -> f64 {
    (0..d)
        .map(|j| libm::cos(GAMMA * xi_j(j) * pos[j]))
        .product()
}

/// CV estimator. Takes arm's own (fs, gs) samples + pre-registered e_g constant.
/// Returns (cv_estimate, beta_hat, rho_fg).
/// Guard #4: e_g is a constant, never estimated; beta_hat from within-arm only.
fn cv_estimate(fs: &[f64], gs: &[f64], eg: f64) -> (f64, f64, f64) {
    assert_eq!(fs.len(), gs.len(), "fs and gs must have same length");
    let n = fs.len() as f64;
    if n < 2.0 {
        return (fs.iter().sum::<f64>() / n, 0.0, 0.0);
    }
    let mean_f = fs.iter().sum::<f64>() / n;
    let mean_g = gs.iter().sum::<f64>() / n;
    // Bessel-corrected Cov and Var
    let cov_fg: f64 = fs
        .iter()
        .zip(gs.iter())
        .map(|(&f, &g)| (f - mean_f) * (g - mean_g))
        .sum::<f64>()
        / (n - 1.0);
    let var_g: f64 = gs.iter().map(|&g| (g - mean_g) * (g - mean_g)).sum::<f64>() / (n - 1.0);
    // Ridge guard: if Var(g) < 1e-300, CV is no-op (guard #4 invariant)
    let beta_hat = if var_g < 1e-300 { 0.0 } else { cov_fg / var_g };
    let cv_est = mean_f - beta_hat * (mean_g - eg);
    // Pearson correlation (informational, printed per arm/d)
    let var_f: f64 = fs.iter().map(|&f| (f - mean_f) * (f - mean_f)).sum::<f64>() / (n - 1.0);
    let rho = if var_f < 1e-300 || var_g < 1e-300 {
        0.0
    } else {
        cov_fg / libm::sqrt(var_f * var_g)
    };
    (cv_est, beta_hat, rho)
}

// ═══════════════════════════════════════════════════════════════════════════════
// §D — Pre-registered experiment constants (§3.1, HARD-CODED)
// ═══════════════════════════════════════════════════════════════════════════════

const T: f64 = 1.0;
const P: usize = 4096; // points per replication per arm
const R: usize = 64; // replications
const DIMS: [usize; 3] = [2, 4, 8];
const EPS_IC: f64 = 1e-3; // δ_0 quantization spread

/// Per-d sub-step count: d=2→8, d=4→4, d=8→2 (§3.1).
fn n_steps(d: usize) -> usize {
    match d {
        2 => 8,
        4 => 4,
        8 => 2,
        _ => 4,
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §E — Arm A + B: MC random-walk endpoints (shared CRN δ₀ input)
//
// Arms A and B are byte-identical in their P point set per replication.
// They differ ONLY in whether the CV is applied (§3.2 NORMATIVE).
// CRN: the {z_i} cloud is drawn from lcg(seed(r)); continued stream drives paths.
// ═══════════════════════════════════════════════════════════════════════════════

/// Build the shared CRN z-cloud (P standard-normal d-vectors) from lcg.
/// Returns the z-cloud; also advances lcg state (paths consumed from same stream).
fn build_crn_cloud(lcg: &mut Lcg64, d: usize) -> alloc::vec::Vec<[f64; 10]> {
    // d ≤ 10 (max DIMS=8 handled here with fixed-size array, padded with 0.0)
    (0..P)
        .map(|_| {
            let mut z = [0.0f64; 10];
            for k in 0..d {
                z[k] = lcg.next_std_normal();
            }
            z
        })
        .collect()
}

/// MC arm: P random walks with Euler-Maruyama.
/// Starting point: eps * z_i (§3.2 NORMATIVE shared IC).
/// Returns (f-samples, g-samples) at terminal positions.
fn run_arm_mc(
    cloud: &[[f64; 10]],
    lcg: &mut Lcg64,
    d: usize,
    n_steps_d: usize,
) -> (alloc::vec::Vec<f64>, alloc::vec::Vec<f64>) {
    let tau = T / n_steps_d as f64;
    let mut fs = alloc::vec::Vec::with_capacity(P);
    let mut gs = alloc::vec::Vec::with_capacity(P);
    for z in cloud {
        let mut pos = [0.0f64; 10];
        for k in 0..d {
            pos[k] = EPS_IC * z[k];
        }
        for _ in 0..n_steps_d {
            for k in 0..d {
                let sigma = libm::sqrt(2.0 * a_j(k) * tau);
                pos[k] += sigma * lcg.next_std_normal();
            }
        }
        fs.push(functional(&pos[..d], d));
        gs.push(control(&pos[..d], d));
    }
    (fs, gs)
}

// ═══════════════════════════════════════════════════════════════════════════════
// §F — Arms C + D: branching evolver (const-D dispatch)
//
// The branching arms evolve the shared δ_0 ensemble from the CRN cloud.
// Arms C and D are byte-identical in their leaf set per replication.
// They differ ONLY in whether the CV is applied.
// Weighted pairing: Σ w_i f(x_i) over leaf (position, weight) pairs.
// ═══════════════════════════════════════════════════════════════════════════════

/// Run the branching arm for compile-time D, returns (f-samples-weighted, g-samples-weighted,
/// leaf_weights, leaf_f_g_pairs) for weighted CV. Returns (est_f, est_g, beta, rho).
/// The "samples" for weighted Cov/Var are the per-leaf (f_i, g_i) values,
/// with weights w_i. β̂ uses weighted covariance (§3.2 NORMATIVE).
fn run_arm_branch_cv<const D: usize>(
    cloud: &[[f64; 10]],
    n_steps_d: usize,
    eg_const: f64,
) -> (f64, f64, f64) {
    // Build δ_0 ensemble from the CRN cloud (same z_i as MC arms)
    let w0 = 1.0 / P as f64;
    let particles: alloc::vec::Vec<([f64; D], f64)> = cloud
        .iter()
        .map(|z| {
            let mut pos = [0.0f64; D];
            for k in 0..D {
                pos[k] = EPS_IC * z[k];
            }
            (pos, w0)
        })
        .collect();
    let ic = MeasureState::<f64, D>::from_particles(&particles);
    // Evolve
    let mut a_arr = [0.0f64; D];
    let b_arr = [0.0f64; D];
    for j in 0..D {
        a_arr[j] = a_j(j);
    }
    let evolver = GridlessChernoff::<f64, D>::new(
        a_arr,
        b_arr,
        0.0,
        ParticleReduction::WeightedVoronoi { cap: P },
    );
    let tau = T / n_steps_d as f64;
    let mut rho = ic;
    let mut rho_next = rho.clone();
    let mut pool = ScratchPool::new();
    for _ in 0..n_steps_d {
        evolver
            .apply_into(tau, &rho, &mut rho_next, &mut pool)
            .unwrap();
        core::mem::swap(&mut rho, &mut rho_next);
    }
    // Collect leaf (f_i, g_i, w_i) — we need weighted Cov/Var for arms C/D
    // pair gives weighted sum directly; for Cov we need two passes
    // First pass: get mean_f and mean_g
    let mean_f_w = rho.pair(|pos: &[f64; D]| functional(pos, D));
    let mean_g_w = rho.pair(|pos: &[f64; D]| control(pos, D));
    // Second pass via pair for variance components
    // We use the identity: Var_w(f) = E_w[f²] - E_w[f]²
    let mean_f2_w = rho.pair(|pos: &[f64; D]| {
        let f = functional(pos, D);
        f * f
    });
    let mean_g2_w = rho.pair(|pos: &[f64; D]| {
        let g = control(pos, D);
        g * g
    });
    let mean_fg_w = rho.pair(|pos: &[f64; D]| functional(pos, D) * control(pos, D));
    // Weighted population (co)variances (no Bessel — matching per-leaf weighted stats)
    // For the CV estimator we compute β̂ = Cov_w(f,g) / Var_w(g)
    let var_g = mean_g2_w - mean_g_w * mean_g_w;
    let cov_fg = mean_fg_w - mean_f_w * mean_g_w;
    let beta_hat = if var_g.abs() < 1e-300 {
        0.0
    } else {
        cov_fg / var_g
    };
    let cv_est = mean_f_w - beta_hat * (mean_g_w - eg_const);
    // Pearson ρ for diagnostics
    let var_f = mean_f2_w - mean_f_w * mean_f_w;
    let rho_fg = if var_f < 1e-300 || var_g < 1e-300 {
        0.0
    } else {
        cov_fg / libm::sqrt(var_f * var_g)
    };
    (cv_est, beta_hat, rho_fg)
}

/// D=2 branching arm wrapper.
fn arm_branch_d2(cloud: &[[f64; 10]], n: usize, eg: f64) -> (f64, f64, f64) {
    run_arm_branch_cv::<2>(cloud, n, eg)
}
/// D=4 branching arm wrapper.
fn arm_branch_d4(cloud: &[[f64; 10]], n: usize, eg: f64) -> (f64, f64, f64) {
    run_arm_branch_cv::<4>(cloud, n, eg)
}
/// D=8 branching arm wrapper.
fn arm_branch_d8(cloud: &[[f64; 10]], n: usize, eg: f64) -> (f64, f64, f64) {
    run_arm_branch_cv::<8>(cloud, n, eg)
}

/// D=2 branching no-CV arm (identical leaves, no CV applied).
fn arm_branch_no_cv_d2(cloud: &[[f64; 10]], n: usize) -> f64 {
    let w0 = 1.0 / P as f64;
    let particles: alloc::vec::Vec<([f64; 2], f64)> = cloud
        .iter()
        .map(|z| ([EPS_IC * z[0], EPS_IC * z[1]], w0))
        .collect();
    let ic = MeasureState::<f64, 2>::from_particles(&particles);
    let a_arr = [a_j(0), a_j(1)];
    let evolver = GridlessChernoff::<f64, 2>::new(
        a_arr,
        [0.0; 2],
        0.0,
        ParticleReduction::WeightedVoronoi { cap: P },
    );
    let tau = T / n as f64;
    let mut rho = ic;
    let mut rho_next = rho.clone();
    let mut pool = ScratchPool::new();
    for _ in 0..n {
        evolver
            .apply_into(tau, &rho, &mut rho_next, &mut pool)
            .unwrap();
        core::mem::swap(&mut rho, &mut rho_next);
    }
    rho.pair(|pos: &[f64; 2]| functional(pos, 2))
}

fn arm_branch_no_cv_d4(cloud: &[[f64; 10]], n: usize) -> f64 {
    let w0 = 1.0 / P as f64;
    let particles: alloc::vec::Vec<([f64; 4], f64)> = cloud
        .iter()
        .map(|z| {
            (
                [EPS_IC * z[0], EPS_IC * z[1], EPS_IC * z[2], EPS_IC * z[3]],
                w0,
            )
        })
        .collect();
    let ic = MeasureState::<f64, 4>::from_particles(&particles);
    let a_arr = [a_j(0), a_j(1), a_j(2), a_j(3)];
    let evolver = GridlessChernoff::<f64, 4>::new(
        a_arr,
        [0.0; 4],
        0.0,
        ParticleReduction::WeightedVoronoi { cap: P },
    );
    let tau = T / n as f64;
    let mut rho = ic;
    let mut rho_next = rho.clone();
    let mut pool = ScratchPool::new();
    for _ in 0..n {
        evolver
            .apply_into(tau, &rho, &mut rho_next, &mut pool)
            .unwrap();
        core::mem::swap(&mut rho, &mut rho_next);
    }
    rho.pair(|pos: &[f64; 4]| functional(pos, 4))
}

fn arm_branch_no_cv_d8(cloud: &[[f64; 10]], n: usize) -> f64 {
    let w0 = 1.0 / P as f64;
    let particles: alloc::vec::Vec<([f64; 8], f64)> = cloud
        .iter()
        .map(|z| {
            (
                [
                    EPS_IC * z[0],
                    EPS_IC * z[1],
                    EPS_IC * z[2],
                    EPS_IC * z[3],
                    EPS_IC * z[4],
                    EPS_IC * z[5],
                    EPS_IC * z[6],
                    EPS_IC * z[7],
                ],
                w0,
            )
        })
        .collect();
    let ic = MeasureState::<f64, 8>::from_particles(&particles);
    let a_arr = {
        let mut a = [0.0f64; 8];
        for j in 0..8 {
            a[j] = a_j(j);
        }
        a
    };
    let evolver = GridlessChernoff::<f64, 8>::new(
        a_arr,
        [0.0; 8],
        0.0,
        ParticleReduction::WeightedVoronoi { cap: P },
    );
    let tau = T / n as f64;
    let mut rho = ic;
    let mut rho_next = rho.clone();
    let mut pool = ScratchPool::new();
    for _ in 0..n {
        evolver
            .apply_into(tau, &rho, &mut rho_next, &mut pool)
            .unwrap();
        core::mem::swap(&mut rho, &mut rho_next);
    }
    rho.pair(|pos: &[f64; 8]| functional(pos, 8))
}

// ═══════════════════════════════════════════════════════════════════════════════
// §G — MSE over R replications
// ═══════════════════════════════════════════════════════════════════════════════

fn mse_over(estimates: &[f64], truth: f64) -> f64 {
    estimates.iter().map(|&e| (e - truth).powi(2)).sum::<f64>() / estimates.len() as f64
}

// ═══════════════════════════════════════════════════════════════════════════════
// §H — Run one dimension's 4-arm experiment; returns (mse_A,mse_B,mse_C,mse_D,rho_B,rho_C)
// ═══════════════════════════════════════════════════════════════════════════════

macro_rules! run_dim {
    ($d:expr, $n:expr, $arm_mc:ident, $arm_branch:ident, $arm_branch_noCV:ident) => {{
        let truth = truth_d($d, T);
        let eg_const = e_g($d, T);
        let mut ests_a: alloc::vec::Vec<f64> = alloc::vec::Vec::with_capacity(R);
        let mut ests_b: alloc::vec::Vec<f64> = alloc::vec::Vec::with_capacity(R);
        let mut ests_c: alloc::vec::Vec<f64> = alloc::vec::Vec::with_capacity(R);
        let mut ests_d: alloc::vec::Vec<f64> = alloc::vec::Vec::with_capacity(R);
        let mut rho_b_sum = 0.0f64;
        let mut rho_c_sum = 0.0f64;
        for r in 0..R {
            // CRN seed: per-replication, deterministic
            let seed = 0xDEAD_BEEF_C0FF_EE00_u64
                .wrapping_add((r as u64).wrapping_mul(1_000_003))
                .wrapping_add(($d as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let mut lcg = Lcg64::new(seed);
            // Build shared CRN z-cloud from the primary LCG stream
            let cloud = build_crn_cloud(&mut lcg, $d);
            // Arms A+B: MC paths driven by continued LCG stream (same cloud)
            let (fs_ab, gs_ab) = $arm_mc(&cloud, &mut lcg, $d, $n);
            // Arm A: MC, no CV
            let mean_a = fs_ab.iter().sum::<f64>() / fs_ab.len() as f64;
            ests_a.push(mean_a);
            // Arm B: MC + CV (same fs, gs — identical point set, only CV differs)
            let (est_b, _beta_b, rho_b) = cv_estimate(&fs_ab, &gs_ab, eg_const);
            ests_b.push(est_b);
            rho_b_sum += rho_b;
            // Arms C+D: branching (same CRN cloud, new evolver)
            let (est_c, _beta_c, rho_c) = $arm_branch(&cloud, $n, eg_const);
            ests_c.push(est_c);
            rho_c_sum += rho_c;
            let est_d = $arm_branch_noCV(&cloud, $n);
            ests_d.push(est_d);
        }
        let mse_a = mse_over(&ests_a, truth);
        let mse_b = mse_over(&ests_b, truth);
        let mse_c = mse_over(&ests_c, truth);
        let mse_d = mse_over(&ests_d, truth);
        // Guard #1: each arm must be a genuine (non-degenerate) estimator
        assert!(
            mse_a > 0.0,
            "G_GRIDLESS_CV: MSE_A=0 at d={} — arm A is degenerate (§3.3 anti-gaming)",
            $d
        );
        assert!(
            mse_b > 0.0,
            "G_GRIDLESS_CV: MSE_B=0 at d={} — arm B is degenerate (§3.3 anti-gaming)",
            $d
        );
        assert!(
            mse_c > 0.0,
            "G_GRIDLESS_CV: MSE_C=0 at d={} — arm C is degenerate (§3.3 anti-gaming)",
            $d
        );
        assert!(
            mse_d > 0.0,
            "G_GRIDLESS_CV: MSE_D=0 at d={} — arm D is degenerate (§3.3 anti-gaming)",
            $d
        );
        let rho_b_avg = rho_b_sum / R as f64;
        let rho_c_avg = rho_c_sum / R as f64;
        (mse_a, mse_b, mse_c, mse_d, rho_b_avg, rho_c_avg)
    }};
}

// ═══════════════════════════════════════════════════════════════════════════════
// §I — Main test: G_GRIDLESS_CV (§3, EVIDENTIARY, slow-tests gate)
// ═══════════════════════════════════════════════════════════════════════════════

/// G_GRIDLESS_CV — 4-arm CV-controlled experiment.
///
/// ## Anti-gaming
/// - Binding variance comparison is C vs B (guard #1): identical CV both arms.
/// - E[g] is a pre-registered closed-form constant (guard #4).
/// - β̂ estimated within each arm's own P samples only.
/// - Each arm's MSE > 0 (anti-degeneracy, §3.3).
///
/// ## This gate is EVIDENTIARY (not RELEASE_BLOCKING).
/// It can refute the variance sub-claim but does not gate v9.0.0 shipping.
/// The binding gate is G_GRIDLESS_MEMORY (g_gridless_memory.rs).
#[test]
#[ignore]
fn g_gridless_cv() {
    println!("\n{}", "═".repeat(72));
    println!("G_GRIDLESS_CV — 4-arm CV-controlled decisive experiment (§3, v9.0.0)");
    println!("{}", "═".repeat(72));
    println!("T={T}, P={P}, R={R}, γ={GAMMA}, DIMS={DIMS:?}");
    println!(
        "n_steps: d=2→{}, d=4→{}, d=8→{}",
        n_steps(2),
        n_steps(4),
        n_steps(8)
    );
    println!("E[g] is a pre-registered closed-form constant (guard #4).");
    println!("Binding comparison: C vs B (guard #1) — identical CV both arms.");
    println!();
    println!("Arms:");
    println!("  A: MC, no CV    — baseline");
    println!("  B: MC + CV      — CV's own contribution (same points as A)");
    println!("  C: branching+CV — does branching add ≥2× over MC+CV?");
    println!("  D: branching,no CV — re-confirms the 1.4×/collapse baseline");
    println!();

    // Run all three dimensions
    let (mse_a2, mse_b2, mse_c2, mse_d2, rho_b2, rho_c2) = run_dim!(
        2,
        n_steps(2),
        run_arm_mc,
        arm_branch_d2,
        arm_branch_no_cv_d2
    );
    let (mse_a4, mse_b4, mse_c4, mse_d4, rho_b4, rho_c4) = run_dim!(
        4,
        n_steps(4),
        run_arm_mc,
        arm_branch_d4,
        arm_branch_no_cv_d4
    );
    let (mse_a8, mse_b8, mse_c8, mse_d8, rho_b8, rho_c8) = run_dim!(
        8,
        n_steps(8),
        run_arm_mc,
        arm_branch_d8,
        arm_branch_no_cv_d8
    );

    // Pre-registered quantities
    let ratio_cb2 = mse_b2 / mse_c2.max(1e-300); // MSE(B)/MSE(C): C vs B
    let ratio_cb4 = mse_b4 / mse_c4.max(1e-300);
    let ratio_cb8 = mse_b8 / mse_c8.max(1e-300);
    // Informational (NOT the verdict)
    let ratio_da2 = mse_a2 / mse_d2.max(1e-300); // D vs A: no-CV comparison
    let ratio_da4 = mse_a4 / mse_d4.max(1e-300);
    let ratio_da8 = mse_a8 / mse_d8.max(1e-300);
    let ratio_ba2 = mse_a2 / mse_b2.max(1e-300); // CV's own gain
    let ratio_ba4 = mse_a4 / mse_b4.max(1e-300);
    let ratio_ba8 = mse_a8 / mse_b8.max(1e-300);

    println!("RESULTS TABLE:");
    println!(
        "{:<4} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "d", "MSE_A", "MSE_B", "MSE_C", "MSE_D", "B/C ratio", "A/C ratio"
    );
    println!("{}", "-".repeat(74));
    println!(
        "{:<4} | {:>10.4e} | {:>10.4e} | {:>10.4e} | {:>10.4e} | {:>10.3} | {:>10.3}",
        2,
        mse_a2,
        mse_b2,
        mse_c2,
        mse_d2,
        ratio_cb2,
        mse_a2 / mse_c2.max(1e-300)
    );
    println!(
        "{:<4} | {:>10.4e} | {:>10.4e} | {:>10.4e} | {:>10.4e} | {:>10.3} | {:>10.3}",
        4,
        mse_a4,
        mse_b4,
        mse_c4,
        mse_d4,
        ratio_cb4,
        mse_a4 / mse_c4.max(1e-300)
    );
    println!(
        "{:<4} | {:>10.4e} | {:>10.4e} | {:>10.4e} | {:>10.4e} | {:>10.3} | {:>10.3}",
        8,
        mse_a8,
        mse_b8,
        mse_c8,
        mse_d8,
        ratio_cb8,
        mse_a8 / mse_c8.max(1e-300)
    );

    println!();
    println!("CORRELATION ρ(f,g) PER ARM (informational):");
    println!("  d=2: ρ_B={rho_b2:.4}  ρ_C={rho_c2:.4}");
    println!("  d=4: ρ_B={rho_b4:.4}  ρ_C={rho_c4:.4}");
    println!("  d=8: ρ_B={rho_b8:.4}  ρ_C={rho_c8:.4}");

    println!();
    println!("INFORMATIONAL RATIOS (not the verdict):");
    println!("  A/B (CV own gain): d=2:{ratio_ba2:.3}×  d=4:{ratio_ba4:.3}×  d=8:{ratio_ba8:.3}×");
    println!(
        "  A/D (branching no-CV): d=2:{ratio_da2:.3}×  d=4:{ratio_da4:.3}×  d=8:{ratio_da8:.3}×"
    );
    println!("  Note: A/D re-confirms existing spatial-merge baseline (1.4×/collapse)");

    println!();
    println!("BINDING VERDICT (guard #1): ratio = MSE(B)/MSE(C) — C vs B ONLY");
    println!("  Pre-registered SALVAGE condition: MSE(B)/MSE(C) ≥ 2.0 at BOTH d=4 AND d=8");
    println!("  d=2: MSE(B)/MSE(C) = {ratio_cb2:.4}×  (context only)");
    println!("  d=4: MSE(B)/MSE(C) = {ratio_cb4:.4}×  (binding)");
    println!("  d=8: MSE(B)/MSE(C) = {ratio_cb8:.4}×  (binding)");

    // Owen bound note (guard #6)
    println!();
    println!("Owen-bound note: scrambled-net variance ≤ ≈3× MC (no impossibility claim).");
    println!("Variance verdicts are FALSIFICATION-by-measurement, not impossibility proofs.");

    let salvaged = ratio_cb4 >= 2.0 && ratio_cb8 >= 2.0;
    println!();
    if salvaged {
        println!("VERDICT: Variance sub-claim SALVAGED ✓");
        println!("  MSE(B)/MSE(C) ≥ 2.0 at BOTH d=4 ({ratio_cb4:.3}×) AND d=8 ({ratio_cb8:.3}×).");
        println!(
            "  Branching+CV beats MC+CV ≥ 2× — §50.5 reworded to 'branching+exact-moment CV'."
        );
        println!("  CV may be promoted to src/ library surface (subject to architect routing).");
    } else {
        println!("VERDICT: Variance sub-claim REFUTED (sharper) — FALSIFIED-by-measurement");
        if ratio_cb4 < 2.0 && ratio_cb8 < 2.0 {
            println!("  MSE(B)/MSE(C) < 2.0 at d=4 ({ratio_cb4:.3}×) AND d=8 ({ratio_cb8:.3}×).");
        } else if ratio_cb4 < 2.0 {
            println!("  MSE(B)/MSE(C) < 2.0 at d=4 ({ratio_cb4:.3}×) — criterion not met.");
        } else {
            println!("  MSE(B)/MSE(C) < 2.0 at d=8 ({ratio_cb8:.3}×) — criterion not met.");
        }
        println!("  Even granting the SAME exact-moment CV to both arms,");
        println!("  deterministic branching adds no ≥2× over MC+CV.");
        println!("  The CV, not the branching, drives variance reduction.");
        println!("  Owen bound (scrambled-net variance ≤ ≈3× MC): no impossibility implied.");
    }
    println!();
    println!("NOTE: This gate is EVIDENTIARY (not RELEASE_BLOCKING).");
    println!("  The binding gate for v9.0.0 is G_GRIDLESS_MEMORY (g_gridless_memory.rs).");
    println!();
    println!("G_GRIDLESS_CV: all anti-degeneracy asserts PASSED ✓ (MSE>0 for all arms/dims)");
}
