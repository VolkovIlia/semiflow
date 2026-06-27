//! `G_PHI_AUG_DENSE` (`RELEASE_BLOCKING`): accuracy of `phi_action` /
//! `phi_action_batched` verified against an **eigen-exact reference** from the
//! analytic DST-I eigendecomposition of the symmetric tridiagonal generator.
//!
//! ## Why eigen-exact replaces the previous Padé-vs-Taylor cross-check
//!
//! The original gate compared `phi_action` (Taylor-13 augmented Horner) against
//! `dense_phi_aug_ref` (Padé-13) — two degree-13 approximants of the same exact
//! `φ_k`.  At `z ≈ 2.0` both methods agreed to ~3e-10, which was incorrectly
//! attributed to a "Taylor-vs-Padé method artefact."  This explanation was
//! INCOMPLETE: two degree-13 methods should agree to ~1e-13 if both are accurate.
//! The discrepancy required an independent exact reference to diagnose.
//!
//! ## Analytic eigendecomposition (exact oracle)
//!
//! The operator `A = offdiag · tridiag(1,−2,1)` is **symmetric** with a
//! closed-form DST-I spectrum:
//!   - eigenvalues  `λ_j = offdiag · (−2 + 2cos(jπ/(n+1)))`,  j = 1 … n
//!   - eigenvectors `q_j[i] = √(2/(n+1)) · sin(ijπ/(n+1))`,   i = 1 … n
//!
//! so `φ_k(τA)·v = Σ_j φ_k(τλ_j) · ⟨q_j, v⟩ · q_j` where each scalar
//! `φ_k(z)` is evaluated via its convergent Taylor series (no cancellation for
//! negative z).  This reference is exact to f64 rounding (~1e-16).
//!
//! ## Measured errors (i7-12700K, release, tau=0.5, n=6)
//!
//! | k | `phi_action` vs eigen-exact | Padé-13 vs eigen-exact |
//! |---|--------------------------|------------------------|
//! | 0 | 3.076e-10                | 4.996e-16              |
//! | 1 | 1.895e-10                | 4.996e-16              |
//! | 2 | 1.167e-10                | 2.776e-16              |
//! | 3 | 7.195e-11                | 9.714e-17              |
//!
//! ## Root cause: genuine Taylor-13 truncation floor
//!
//! **`phi_action` has a genuine ~3e-10 error**; the Padé-13 oracle is machine-exact.
//! The 3e-10 is NOT a method artefact — it is the forward error of Taylor-13 at the
//! effective spectral radius of the computation.
//!
//! Cause: the bump `v = exp(−(x−0.5)²·10)` is **symmetric**, so it projects only onto
//! odd-indexed DST-I modes (j=1,3,5; even modes j=2,4,6 have `⟨q_j, v⟩ = 0`).
//! The most negative odd-mode eigenvalue is `τλ_5 = 0.5·(−2+2cos(5π/7)) ≈ −1.624`.
//! Taylor-13 at z=−1.624 has truncation `|R₁₃(−1.624)| ≈ 9.1e-9`; the projection
//! weight `|⟨q_5,v⟩| ≈ 0.064` and mode sup-norm `≈ 0.52` give
//! `0.064 × 0.52 × 9.1e-9 ≈ 3.0e-10` — matching the measurement.
//!
//! ## Fix (`PHI_NORM_TIGHTEN` = 2.0 in `phi_action.rs`)
//!
//! Passing `norm_aug * 2` to `select_s_m` causes it to choose Taylor-18 (m=18)
//! instead of Taylor-13 (m=13) at the canonical z≈2 test point.  Truncation at
//! the dominant spectral argument z=1.624 drops from ~9e-9 (Taylor-13) to ~8e-14
//! (Taylor-18), giving measured `phi_action` errors ~2–3e-15 after projection.
//! The augmented φ-action uses tightened norm selection vs the plain expmv
//! backward-error calibration in `THETA_M`.
//!
//! ## Gate structure
//!
//! 1. **`RELEASE_BLOCKING`** (`check_phi_k_accuracy`): `phi_action` vs eigen-exact ≤ 1e-12.
//!    Also: `‖φ_k(τA)·v‖_∞ ≥ 0.01` (non-vacuity); Padé vs exact ≤ 1e-12 (oracle sanity).
//!
//! 2. **Batched consistency** (`check_batched_vs_single`): `phi_action_batched` vs
//!    `phi_action` ≤ 1e-14 (unchanged from original).

use semiflow::{
    dense_phi_aug_ref,
    generator_action::GeneratorAction,
    phi_action::{phi_action, phi_action_batched, PHI_MAX},
    scratch::ScratchPool,
};

const N: usize = 6;

// ---------------------------------------------------------------------------
// Minimal tridiagonal generator: A = offdiag * tridiagonal(1, -2, 1)
// ---------------------------------------------------------------------------

struct TriDiagGen {
    n: usize,
    offdiag: f64,
}

impl GeneratorAction<f64> for TriDiagGen {
    fn dim(&self) -> usize { self.n }

    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        let od = self.offdiag;
        for i in 0..self.n {
            let left = if i > 0 { src[i - 1] } else { 0.0 };
            let right = if i + 1 < self.n { src[i + 1] } else { 0.0 };
            dst[i] = od * (left - 2.0 * src[i] + right);
        }
    }

    fn norm_bound(&self) -> f64 { 4.0 * self.offdiag.abs() }
}

// ---------------------------------------------------------------------------
// Scalar φ_k(z) via Taylor series — numerically stable for all z.
// φ_k(z) = Σ_{n≥0} z^n / (n+k)!   (converges; no cancellation for z < 0)
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn scalar_phi_k_exact(k: usize, z: f64) -> f64 {
    // First term: 1/k!
    let mut fact_k = 1.0_f64;
    for i in 1..=k {
        fact_k *= i as f64; // usize → f64 (max k=3, exact)
    }
    let mut term = 1.0 / fact_k;
    let mut sum = term;
    for n in 1_usize..120 {
        term *= z / (n + k) as f64;
        sum += term;
        if term.abs() < 1e-18 * (sum.abs() + 1e-300) {
            break;
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// Eigen-exact φ_k(τA)·v via analytic DST-I eigendecomposition.
// A = offdiag · tridiag(1,−2,1), size n×n.
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn eigen_exact_phi_k(k: usize, tau: f64, offdiag: f64, v: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let n = v.len();
    let norm = (2.0 / (n + 1) as f64).sqrt();
    let mut out = vec![0.0_f64; n];

    for j in 1..=n {
        let eig_ang = j as f64 * PI / (n + 1) as f64;
        let lam_j = offdiag * (-2.0 + 2.0 * eig_ang.cos());
        let tau_lam = tau * lam_j;
        // ⟨q_j, v⟩  (inline sine — avoids naming a similar-to-eig_ang binding)
        let inner: f64 = (1..=n)
            .map(|i| norm * (i as f64 * j as f64 * PI / (n + 1) as f64).sin() * v[i - 1])
            .sum();
        let phi_val = scalar_phi_k_exact(k, tau_lam);
        // Accumulate φ_k(τλ_j) · ⟨q_j, v⟩ · q_j
        for i in 1..=n {
            let q_ji = norm * (i as f64 * j as f64 * PI / (n + 1) as f64).sin();
            out[i - 1] += phi_val * inner * q_ji;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Per-k accuracy checks: phi_action vs eigen-exact (primary gate) and
// Padé oracle vs eigen-exact (diagnostic).
// ---------------------------------------------------------------------------

fn check_phi_k_accuracy(
    gen: &TriDiagGen,
    tau: f64,
    v: &[f64],
    eigen_refs: &[Vec<f64>],
    pade_ref: &[Vec<f64>],
    scratch: &mut ScratchPool<f64>,
) {
    eprintln!("  {:>5}  {:>18}  {:>18}", "k", "action_vs_exact", "pade_vs_exact");
    for k in 0..=PHI_MAX {
        let mut phi_k_out = [0.0_f64; N];
        phi_action(gen, k, tau, v, &mut phi_k_out, scratch).expect("phi_action failed");

        let action_err = phi_k_out.iter().zip(&eigen_refs[k]).map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let pade_err = pade_ref[k].iter().zip(&eigen_refs[k]).map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let out_sup = phi_k_out.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

        eprintln!("  phi_{k}: {action_err:>18.3e}  {pade_err:>18.3e}  (out_sup={out_sup:.3e})");

        // Primary gate: augmented φ-action uses tightened scaling (PHI_NORM_TIGHTEN=2)
        // vs the plain expmv backward-error selection, so select_s_m picks Taylor-18
        // instead of Taylor-13.  Truncation at z≈1.624 drops from ~1e-8 to ~1e-13.
        // Threshold 1e-12 = measured floor (~2e-15) × large headroom.
        assert!(
            action_err <= 1e-12,
            "G_PHI_AUG_DENSE phi_{k}: phi_action vs eigen-exact = {action_err:.3e} > 1e-12 \
             (augmented φ-action tightened scaling; Padé oracle is {pade_err:.3e})"
        );
        // Non-vacuity: phi_k_out must not be the zero vector.
        assert!(out_sup >= 0.01,
            "G_PHI_AUG_DENSE phi_{k}: output sup-norm {out_sup:.3e} < 0.01 — vacuous");
        // Oracle sanity: Padé-13 must be machine-accurate vs eigen-exact.
        assert!(
            pade_err <= 1e-12,
            "G_PHI_AUG_DENSE phi_{k}: Padé oracle vs eigen-exact = {pade_err:.3e} > 1e-12 \
             — eigen-exact reference may be broken"
        );
    }
    eprintln!("  (phi_action ~1e-15 floor = Taylor-18 augmented tightened scaling; Padé is machine-exact)");
}

// ---------------------------------------------------------------------------
// Batched consistency: phi_action_batched ≡ phi_action for all k.
// ---------------------------------------------------------------------------

fn check_batched_vs_single(
    gen: &TriDiagGen,
    tau: f64,
    v: &[f64],
    scratch: &mut ScratchPool<f64>,
) {
    let mut batched_out = vec![0.0_f64; (PHI_MAX + 1) * N];
    phi_action_batched(gen, PHI_MAX, tau, v, &mut batched_out, scratch)
        .expect("phi_action_batched failed");

    for (k, chunk) in batched_out.chunks(N).enumerate() {
        let mut single_out = [0.0_f64; N];
        phi_action(gen, k, tau, v, &mut single_out, scratch)
            .expect("phi_action single (recheck) failed");
        let sup_diff = chunk.iter().zip(&single_out)
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        eprintln!("  batched vs single phi_{k}: sup_diff = {sup_diff:.3e}");
        assert!(sup_diff <= 1e-14,
            "G_PHI_AUG_DENSE batched != single phi_{k}: sup_diff={sup_diff:.3e}");
    }
}

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_phi_aug_dense() {
    let gen = TriDiagGen { n: N, offdiag: 1.0 };
    let tau = 0.5_f64;

    let z = tau * gen.norm_bound(); // z = tau * 4.0 = 2.0
    assert!((0.5..=5.0).contains(&z),
        "G_PHI_AUG_DENSE: z={z:.3} not in [0.5, 5]");
    eprintln!("G_PHI_AUG_DENSE: n={N}  tau={tau}  z=tau*||A||={z:.4}");

    #[allow(clippy::cast_precision_loss)]
    let v: [f64; N] = core::array::from_fn(|i| {
        let x = i as f64 / (N as f64 - 1.0);
        (-(x - 0.5).powi(2) * 10.0).exp()
    });

    let eigen_refs: Vec<Vec<f64>> = (0..=PHI_MAX)
        .map(|k| eigen_exact_phi_k(k, tau, gen.offdiag, &v))
        .collect();

    let pade_ref = dense_phi_aug_ref(&gen, tau, &v).expect("dense oracle failed");
    assert_eq!(pade_ref.len(), PHI_MAX + 1, "oracle wrong length");

    let mut scratch = ScratchPool::new();
    check_phi_k_accuracy(&gen, tau, &v, &eigen_refs, &pade_ref, &mut scratch);
    check_batched_vs_single(&gen, tau, &v, &mut scratch);

    eprintln!("G_PHI_AUG_DENSE PASS");
}
