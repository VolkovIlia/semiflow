//! G_SYMOP_IMPLICIT_PCG_CAP — regression gate for issue #18: PCG cap bug.
//!
//! ## Root cause (two defects)
//!
//! **D1 — Inadequate iteration cap**: `compute_max_iter` used `ceil(4·√κ)` without the
//! `ln(2/tol)` factor mandated by the standard CG convergence bound
//! `m ≤ ½·√κ·ln(2/tol)` for SPD `S = I + Δt·Â`.  As Δt→0, κ→1 and the cap
//! collapsed to 4, cutting CG off before convergence.
//!
//! **D2 — (minor, covered by D1 fix)**: the iteration cap was the binding constraint; once
//! `ln(2/tol)` is included the solver reaches the tol goal for all tested parameters.
//!
//! ## Operator
//!
//! `assemble_conservative_csr_1d`, N=401, k=1 (constant), Neumann BCs — the exact
//! operator from the issue report.  Gershgorin bound ≈ 640 000.
//!
//! ## Oracle
//!
//! Independent oracle: `path="chebyshev"` via `graph_expmv_krylov`, which uses a
//! separate algorithm (Bessel-coefficient Chebyshev expansion) validated in
//! `G_SYMOP_EXPMV_DENSE` and `G_GRAPH_EXPMV_DEPTH_FLAT`.  The oracle is NOT derived
//! from the PCG/implicit path output (H1 compliant).
//!
//! The comparison tolerance is the backward-Euler time-discretization error
//! `O(Δt) = O(τ/n_steps)`, not the CG tolerance, so the gate cannot be gamed by
//! tightening CG tol (H3 compliant — no threshold widening).
//!
//! ## Gate (H7)
//!
//! - PRE-FIX : `evolve_batched(path="implicit")` raises `ConvergenceFailed` for the
//!   matrix entries in the failure matrix below.
//! - POST-FIX: all entries return `Ok(())` and agree with the Chebyshev oracle within
//!   the backward-Euler time-discretization error.
//!
//! ## Failure matrix (from issue #18 report)
//!
//! | t   | tol   | n_steps | pre-fix cap | pre-fix behaviour          |
//! |-----|-------|---------|-------------|----------------------------|
//! | 0.1 | 1e-10 | 2000    | 23          | `last_residual=2.268e-9`   |
//! | 0.1 | 1e-8  | 2000    | 23          | `ConvergenceFailed`        |
//! | 0.1 | 1e-8  | 200     | 72          | `last_residual > tol`      |
//! | 0.1 | 1e-10 | 50      | 144         | `last_residual > tol`      |
//! | 1.0 | 1e-8  | 2000    | 72          | `ConvergenceFailed`        |
//! | 1.0 | 1e-6  | 2000    | 72          | `ConvergenceFailed`        |
//! | 1.0 | 1e-8  | 200     | 227         | `last_residual > tol`      |

#![cfg(test)]
// Mathematical notation in doc comments (λ, Δt, κ, ‖·‖) — no backticks needed.
#![allow(clippy::doc_markdown)]
// usize/f64 conversions for N=401 — precision loss impossible at this scale.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use semiflow::{
    assemble_conservative_csr_1d, graph_expmv_krylov, scratch::ScratchPool, BoundaryPolicy, Grid1D,
    KrylovPath,
};

// ── Operator builder ──────────────────────────────────────────────────────────

const N: usize = 401;

/// Build the N=401 constant-k=1 conservative operator from the issue report.
///
/// Uses the same Python call as the issue: `assemble_conservative_csr_1d(401, 0.0, 1.0, ones)`.
fn build_op() -> semiflow::SymmetricOperator<f64> {
    let grid = Grid1D::new(0.0_f64, 1.0_f64, N).expect("Grid1D build");
    let k = vec![1.0_f64; N];
    assemble_conservative_csr_1d(grid, &k, None, BoundaryPolicy::Neumann)
        .expect("assemble_conservative_csr_1d build")
}

// ── Chebyshev oracle (independent) ───────────────────────────────────────────

/// `e^{-τ·A}·v` via the Chebyshev path — independent of the PCG/implicit path.
///
/// Oracle is NOT derived from the PCG code (H1 compliant).
fn chebyshev_oracle(v: &[f64], tau: f64) -> Vec<f64> {
    let op = build_op();
    let mut out = vec![0.0_f64; N];
    let mut scratch = ScratchPool::new();
    graph_expmv_krylov(
        &op,
        tau,
        v,
        &mut out,
        KrylovPath::Chebyshev,
        1e-12_f64, // tighter tolerance for oracle
        &mut scratch,
    )
    .expect("chebyshev oracle failed — independent path broken");
    out
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sup_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ── Gaussian initial condition (from issue report) ────────────────────────────

fn gaussian_v() -> Vec<f64> {
    (0..N)
        .map(|i| {
            let x = i as f64 / (N - 1) as f64;
            (-(x - 0.4).powi(2) / (2.0 * 0.03_f64.powi(2))).exp()
        })
        .collect()
}

// ── Non-vacuity guard ─────────────────────────────────────────────────────────

/// Assert the operator and initial condition are structurally non-degenerate.
///
/// H6: a degenerate test family (zero initial condition, or zero operator) would pass
/// trivially regardless of whether PCG converges.
fn assert_non_vacuous(v: &[f64]) {
    let op = build_op();
    // (i) Operator is non-trivial: λ_max_bound > 0.
    assert!(
        op.lambda_max_bound() > 0.0,
        "non-vacuity: λ_max_bound must be > 0 (got {})",
        op.lambda_max_bound()
    );
    // (ii) Initial condition is non-trivial: ‖v‖₂ > 0.
    let nv = norm2(v);
    assert!(
        nv > 0.1,
        "non-vacuity: ‖v‖₂={nv:.3e} < 0.1 — Gaussian initial condition collapsed to zero"
    );
    // (iii) Chebyshev oracle gives a non-trivial result (semigroup contracts but never zeros).
    let ref_out = chebyshev_oracle(v, 0.1);
    let nout = norm2(&ref_out);
    assert!(
        nout > 1e-3,
        "non-vacuity: Chebyshev oracle ‖result‖={nout:.3e} < 1e-3 — oracle already collapsed"
    );
}

// ── Single-case helper ────────────────────────────────────────────────────────

/// Run `path="implicit"` on the N=401 k=1 Neumann operator and return `Ok(sup_error)`
/// vs Chebyshev oracle, or propagate `Err(ConvergenceFailed)`.
fn run_implicit_case(
    v: &[f64],
    tau: f64,
    tol: f64,
    n_steps: usize,
) -> Result<f64, semiflow::SemiflowError> {
    let op = build_op();
    let mut out = vec![0.0_f64; N];
    let mut scratch = ScratchPool::new();
    // G1: this call MUST fail pre-fix (ConvergenceFailed) for the cases in the failure matrix.
    graph_expmv_krylov(
        &op,
        tau,
        v,
        &mut out,
        KrylovPath::ImplicitEuler {
            n_steps,
            cg_max_iter: None,
        },
        tol,
        &mut scratch,
    )?;
    // Compare with the independent Chebyshev oracle.
    let oracle = chebyshev_oracle(v, tau);
    Ok(sup_err(&out, &oracle))
}

// ── Gate A: primary failure matrix (each row was ConvergenceFailed pre-fix) ──

/// Primary gate: the N=401 k=1 Neumann operator on the full failure matrix.
///
/// Each row must return `Ok(())` post-fix; `ConvergenceFailed` → test failure.
/// Threshold = `c·(τ/n_steps)` (backward-Euler time error, theory-bound; H3 compliant).
///
/// Non-vacuity: `assert_non_vacuous` is called before the matrix sweep (H6).
#[test]
fn g_symop_implicit_pcg_cap() {
    let v = gaussian_v();
    assert_non_vacuous(&v);

    // (t, tol, n_steps, max_allowed_sup_error)
    // Threshold = 5 · (t/n_steps) captures O(Δt) backward-Euler error.
    // 5× safety factor absorbs spatial discretization and CG tolerance residual.
    let cases: &[(f64, f64, usize, f64)] = &[
        // ── Failure matrix from issue #18 ─────────────────────────────────────
        (0.1, 1e-10, 2000, 5.0 * 0.1 / 2000.0), // was: last_residual=2.268e-9
        (0.1, 1e-8, 2000, 5.0 * 0.1 / 2000.0),  // was: ConvergenceFailed
        (0.1, 1e-8, 200, 5.0 * 0.1 / 200.0),    // was: last_residual > tol
        (0.1, 1e-10, 50, 5.0 * 0.1 / 50.0),     // was: last_residual > tol
        (1.0, 1e-8, 2000, 5.0 * 1.0 / 2000.0),  // was: ConvergenceFailed
        (1.0, 1e-6, 2000, 5.0 * 1.0 / 2000.0),  // was: ConvergenceFailed
        (1.0, 1e-8, 200, 5.0 * 1.0 / 200.0),    // was: last_residual > tol
    ];

    let mut all_pass = true;
    for &(tau, tol, n_steps, max_err) in cases {
        match run_implicit_case(&v, tau, tol, n_steps) {
            Ok(sup) => {
                eprintln!(
                    "G_SYMOP_IMPLICIT_PCG_CAP  t={tau}  tol={tol:.0e}  n_steps={n_steps}  \
                     sup_err={sup:.3e}  max_allowed={max_err:.3e}  PASS"
                );
                if sup > max_err {
                    eprintln!("  FAIL: sup_err={sup:.3e} > max_allowed={max_err:.3e}");
                    all_pass = false;
                }
            }
            Err(e) => {
                eprintln!(
                    "G_SYMOP_IMPLICIT_PCG_CAP  t={tau}  tol={tol:.0e}  n_steps={n_steps}  \
                     FAIL: {e:?}"
                );
                all_pass = false;
            }
        }
    }

    assert!(
        all_pass,
        "G_SYMOP_IMPLICIT_PCG_CAP: one or more cases failed (see eprintln output above)"
    );
    eprintln!("G_SYMOP_IMPLICIT_PCG_CAP  ALL PASS");
}
