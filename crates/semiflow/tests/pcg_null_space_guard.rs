//! Regression test: PCG returns Ok when the initial residual is already zero (§59.6).
//!
//! ## Root cause (pre-fix behaviour)
//!
//! `pcg_shifted` with a warm-start `x = b` that lies in `ker(S-I) = ker(Â)`:
//!   r = b − S·x = b − (I + Δt·Â)·b = −Δt·Â·b = 0.
//! The preconditioner yields z = 0, p = 0, ⟨p, S·p⟩ = 0, loop breaks
//! without updating `last_r_sq` (still ∞) → `ConvergenceFailed`.
//!
//! ## Fix
//!
//! Pre-loop guard after `compute_residual`: if `‖r‖² ≤ tol²·‖b‖²`, return `Ok(0)`
//! immediately — warm start already solves the system.
//!
//! ## Oracle
//!
//! `e^{-τÂ}·c·1 = c·1` since `Â·1 = 0` (1 is eigenvector with eigenvalue 0,
//! spectral theorem: `e^{-τ·0} = 1`).  Oracle is INDEPENDENT of the code-under-test.
//!
//! ## Gate (H7)
//!
//! - PRE-FIX : `graph_expmv_krylov` returns `Err(ConvergenceFailed)`.
//! - POST-FIX: `graph_expmv_krylov` returns `Ok(())`, output ≈ c·1, `sup_error` ≤ 1e-10.

#![cfg(test)]

use semiflow::{
    graph_expmv_krylov, scratch::ScratchPool, KrylovPath, SymmetricLinearOp, SymmetricOperator,
};

/// 4-node Neumann Laplacian (1-D 2nd-order FD, zero-flux BCs).
///
/// ```text
/// A = [[1, -1,  0,  0],
///      [-1,  2, -1,  0],
///      [ 0, -1,  2, -1],
///      [ 0,  0, -1,  1]]
/// ```
/// Row sums = 0 ⇒ constant vector [1,1,1,1] ∈ ker(A).
// Triplet col indices are matrix positions for a 4-node graph — always < u32::MAX.
#[allow(clippy::cast_possible_truncation)]
fn build_neumann4_csr() -> (usize, Vec<usize>, Vec<u32>, Vec<f64>) {
    const N: usize = 4;
    #[rustfmt::skip]
    let triplets: &[(usize, usize, f64)] = &[
        (0, 0,  1.0), (0, 1, -1.0),
        (1, 0, -1.0), (1, 1,  2.0), (1, 2, -1.0),
        (2, 1, -1.0), (2, 2,  2.0), (2, 3, -1.0),
        (3, 2, -1.0), (3, 3,  1.0),
    ];
    let mut row_ptr = vec![0usize; N + 1];
    let mut col_idx: Vec<u32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for &(i, j, v) in triplets {
        col_idx.push(j as u32);
        vals.push(v);
        row_ptr[i + 1] = col_idx.len();
    }
    for k in 1..=N {
        if row_ptr[k] == 0 {
            row_ptr[k] = row_ptr[k - 1];
        }
    }
    (N, row_ptr, col_idx, vals)
}

/// Regression: constant input on a Neumann Laplacian must succeed, not fail.
///
/// Before the fix the test returned `Err(ConvergenceFailed)`.
/// After the fix it returns `Ok(())` with the constant preserved (oracle: e^{-τA}·c·1 = c·1).
///
/// The `path=ImplicitEuler` warm-start sets `x₀ = b = c·1`.
/// Since `Â·c·1 = 0`, the residual `r = b − S·x₀ = 0`, which is already within tolerance.
/// The pre-loop guard must detect this and return `Ok(0)` before entering the CG loop.
#[test]
fn pcg_null_space_guard() {
    let (n, row_ptr, col_idx, vals) = build_neumann4_csr();
    let op = SymmetricOperator::from_csr(n, &row_ptr, &col_idx, &vals, 1e-10)
        .expect("Neumann4 CSR build");

    // ── H6 non-degeneracy guard: A·1 must be ≈ 0 ──────────────────────────
    let ones = vec![1.0_f64; n];
    let mut ax = vec![0.0_f64; n];
    op.apply_into_slice(&ones, &mut ax);
    let null_residual: f64 = ax.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        null_residual < 1e-14,
        "H6: A·[1,1,1,1] norm={null_residual:.3e} ≥ 1e-14; constant is not a null vector — \
         test would be vacuous"
    );

    // ── Evolve constant input via implicit Euler ───────────────────────────
    let c = 2.0_f64;
    let v_in: Vec<f64> = vec![c; n];
    let mut v_out = vec![0.0_f64; n];
    let mut scratch = ScratchPool::new();

    let result = graph_expmv_krylov(
        &op,
        0.1_f64,
        &v_in,
        &mut v_out,
        KrylovPath::ImplicitEuler { n_steps: 1 },
        1e-10_f64,
        &mut scratch,
    );

    // (a) Must NOT return ConvergenceFailed — that was the pre-fix bug.
    assert!(
        result.is_ok(),
        "pcg_null_space_guard (a): expected Ok(()), got {:?}.  \
         Pre-loop convergence guard is missing or broken (Bug #16).",
        result.err()
    );

    // (b) Oracle: e^{{-τA}}·c·1 = c·1 (eigenvalue 0 ⇒ e^{{0}} = 1).
    //     Independent oracle: spectral theorem, not derived from code output.
    let oracle = vec![c; n];
    let sup_err = v_out
        .iter()
        .zip(oracle.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "pcg_null_space_guard  n={n}  c={c}  sup_err={sup_err:.3e}  \
         oracle=constant {c}"
    );
    assert!(
        sup_err <= 1e-10,
        "pcg_null_space_guard (b): sup_error={sup_err:.3e} > 1e-10; \
         constant should be preserved exactly (oracle: e^{{-τA}}·c·1 = c·1)"
    );
    eprintln!("pcg_null_space_guard  PASS");
}
