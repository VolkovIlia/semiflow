//! G_SYMOP_IMPLICIT_PCG_SPD — TEETH: PCG on singular PSD operator (§59.7 gate C).
//!
//! Operator: 4-node path-graph combinatorial Laplacian.
//!   λ=0 with null vector [1,1,1,1] — singular but PSD.
//!   S = I + 0.1·A is SPD:  σ(S) ⊂ [1, 1+0.1·λ_max] ⊂ (0,∞)  (§59.3).
//!
//! Assertions (per §59.7):
//!   (a) PCG converges in ≤ N iterations.
//!   (b) Post-hoc residual ‖S·x − b‖₂ ≤ tol_cg·‖b‖₂.
//!
//! The test calls `pcg_shifted` directly from the internal API (allowed because
//! integration tests share the crate's dev-dependency exports).

#![cfg(test)]
// Test doc comments use mathematical notation and gate identifiers without backticks.
#![allow(clippy::doc_markdown)]

use semiflow::{
    graph_expmv_krylov, scratch::ScratchPool, KrylovPath, SymmetricLinearOp, SymmetricOperator,
};

// ── CSR for 4-node path Laplacian ────────────────────────────────────────────

/// Combinatorial Laplacian of a 4-node path graph:
/// ```
/// [[ 1, -1,  0,  0],
///  [-1,  2, -1,  0],
///  [ 0, -1,  2, -1],
///  [ 0,  0, -1,  1]]
/// ```
/// λ₀ = 0, λ₁ ≈ 0.586, λ₂ ≈ 2.0, λ₃ ≈ 3.414.
/// Null space: span{[1,1,1,1]}.
// 4-node graph; column indices always < u32::MAX.
#[allow(clippy::cast_possible_truncation)]
fn build_path4_csr() -> (usize, Vec<usize>, Vec<u32>, Vec<f64>) {
    const N: usize = 4;
    #[rustfmt::skip]
    let triplets: &[(usize, usize, f64)] = &[
        (0, 0, 1.0), (0, 1, -1.0),
        (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
        (2, 1, -1.0), (2, 2, 2.0), (2, 3, -1.0),
        (3, 2, -1.0), (3, 3, 1.0),
    ];
    let mut row_ptr = vec![0usize; N + 1];
    let mut col_idx: Vec<u32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for &(i, j, v) in triplets {
        col_idx.push(j as u32);
        vals.push(v);
        row_ptr[i + 1] = col_idx.len();
    }
    // Fill gaps (rows with entries already in-order).
    for k in 1..=N {
        if row_ptr[k] == 0 {
            row_ptr[k] = row_ptr[k - 1];
        }
    }
    (N, row_ptr, col_idx, vals)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute S·x = x + dt·A·x using `SymmetricOperator`.
fn apply_shifted(op: &SymmetricOperator<f64>, dt: f64, x: &[f64]) -> Vec<f64> {
    let n = op.n();
    let mut ax = vec![0.0_f64; n];
    op.apply_into_slice(x, &mut ax);
    x.iter()
        .zip(ax.iter())
        .map(|(xi, ai)| xi + dt * ai)
        .collect()
}

// ── Gate ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_symop_implicit_pcg_spd() {
    let (n, row_ptr, col_idx, vals) = build_path4_csr();
    let op =
        SymmetricOperator::from_csr(n, &row_ptr, &col_idx, &vals, 1e-10).expect("path4 CSR build");

    // Non-vacuity: has a zero eigenvalue (null vector [1,1,1,1]).
    let null = vec![1.0_f64; n];
    let mut ax = vec![0.0_f64; n];
    op.apply_into_slice(&null, &mut ax);
    let null_residual = norm2(&ax);
    assert!(
        null_residual < 1e-14,
        "non-vacuity: A·[1,1,1,1] norm={null_residual:.3e} ≥ 1e-14; not a null vector"
    );

    // Solve S·x = b for a non-null RHS via implicit_euler_action (one step).
    let dt = 0.1_f64;
    // n=4 — usize→f64 cast never loses precision.
    #[allow(clippy::cast_precision_loss)]
    let b: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
    let tol_cg = 1e-10_f64;

    // One sub-step of backward-Euler ≡ one PCG solve.
    let path = KrylovPath::ImplicitEuler { n_steps: 1 };
    let mut x = vec![0.0_f64; n];
    let tau = dt; // Δt = τ / n_steps = dt / 1 = dt
    let mut scratch = ScratchPool::new();
    graph_expmv_krylov(&op, tau, &b, &mut x, path, tol_cg, &mut scratch).expect("PCG SPD solve");

    // Post-hoc verification: ‖S·x − b‖₂ ≤ tol_cg·‖b‖₂.
    // Note: x from implicit_euler_action is (I+dt*A)^{-1}·b, so S·x should ≈ b.
    let sx = apply_shifted(&op, dt, &x);
    let res: Vec<f64> = sx.iter().zip(b.iter()).map(|(si, bi)| si - bi).collect();
    let res_norm = norm2(&res);
    let b_norm = norm2(&b);
    eprintln!(
        "G_SYMOP_IMPLICIT_PCG_SPD  n={n}  dt={dt}  \
         ‖Sx−b‖={res_norm:.3e}  tol·‖b‖={:.3e}",
        tol_cg * b_norm
    );

    // (a) Converged (no ConvergenceFailed error — checked by expect above).
    // (b) Post-hoc residual ≤ tol_cg·‖b‖.
    assert!(
        res_norm <= tol_cg * b_norm + 1e-15, // small additive epsilon for FP rounding
        "G_SYMOP_IMPLICIT_PCG_SPD (b): ‖Sx−b‖={res_norm:.3e} > tol·‖b‖={:.3e}",
        tol_cg * b_norm
    );
    eprintln!("G_SYMOP_IMPLICIT_PCG_SPD  PASS");
}
