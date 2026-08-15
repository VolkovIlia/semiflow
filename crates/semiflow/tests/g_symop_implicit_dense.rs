//! G_SYMOP_IMPLICIT_DENSE — backward-Euler order-1 convergence (§59.7 gate A).
//!
//! N=10 Robin-BC 1-D Laplacian (path graph + diagonal shift).
//! τ = 0.01, n_steps ∈ {50, 100, 200}.
//! Assertions:
//!   (a) sup_error(n_steps=200) ≤ 1e-6
//!   (b) ratio err(n)/err(2n) ∈ [1.7, 2.3]  (first-order backward-Euler)
//!
//! Oracle: `dense_csr_expmv_ref` from the test helper in sym_op_dense.rs pattern.

#![cfg(test)]
// Test-file doc comments use mathematical notation (λ, τ, subscripts) and gate
// identifiers without backticks.  Allow the doc_markdown lint file-wide.
#![allow(clippy::doc_markdown)]

use std::time::Instant;

use semiflow::{
    dense_csr_expmv_ref, graph_expmv_krylov, scratch::ScratchPool, KrylovPath, SymmetricOperator,
};

// ── CSR helpers ──────────────────────────────────────────────────────────────

/// N=10 path-graph Laplacian + 0.5·I  (Robin BC: off-diagonals = -1, diagonal = 2.5).
// Node indices for N=10 fit in u32 — cast is always exact.
#[allow(clippy::cast_possible_truncation)]
fn robin_n10_csr() -> (usize, Vec<usize>, Vec<u32>, Vec<f64>) {
    const N: usize = 10;
    let mut row_ptr = vec![0usize; N + 1];
    let mut col_idx: Vec<u32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for i in 0..N {
        if i > 0 {
            col_idx.push((i - 1) as u32);
            vals.push(-1.0);
        }
        col_idx.push(i as u32);
        vals.push(if i == 0 || i == N - 1 {
            1.5 + 0.5
        } else {
            2.0 + 0.5
        });
        if i + 1 < N {
            col_idx.push((i + 1) as u32);
            vals.push(-1.0);
        }
        row_ptr[i + 1] = col_idx.len();
    }
    (N, row_ptr, col_idx, vals)
}

/// Dense 10×10 matrix exponential via `dense_csr_expmv_ref`.
fn oracle_expmv(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[f64],
    tau: f64,
    v: &[f64],
) -> Vec<f64> {
    let op =
        SymmetricOperator::from_csr(n, row_ptr, col_idx, vals, 1e-10).expect("oracle: CSR build");
    let mut dst = vec![0.0_f64; n];
    dense_csr_expmv_ref(&op, tau, v, &mut dst).expect("oracle: dense expmv");
    dst
}

/// Backward-Euler approximation via `ImplicitEuler` path.
// 7 args by necessity — CSR triplet (n/row_ptr/col_idx/vals) + tau/n_steps/v.
#[allow(clippy::too_many_arguments)]
fn implicit_expmv(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[u32],
    vals: &[f64],
    tau: f64,
    n_steps: usize,
    v: &[f64],
) -> Vec<f64> {
    let op =
        SymmetricOperator::from_csr(n, row_ptr, col_idx, vals, 1e-10).expect("implicit: CSR build");
    let path = KrylovPath::ImplicitEuler {
        n_steps,
        cg_max_iter: None,
    };
    let tol = 1e-12_f64;
    let mut out = vec![0.0_f64; n];
    let mut scratch = ScratchPool::new();
    graph_expmv_krylov(&op, tau, v, &mut out, path, tol, &mut scratch).expect("implicit: expmv");
    out
}

fn sup_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ── Gate test ────────────────────────────────────────────────────────────────

#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_symop_implicit_dense() {
    let t0 = Instant::now();

    let (n, row_ptr, col_idx, vals) = robin_n10_csr();
    let tau = 0.01_f64;

    // Non-vacuity: operator is non-trivial (λ_max > 0).
    let op = SymmetricOperator::from_csr(n, &row_ptr, &col_idx, &vals, 1e-10).unwrap();
    assert!(
        op.lambda_max_bound() > 0.0,
        "non-vacuity: λ_max must be positive"
    );

    // Reference vector: smooth sinusoidal initial condition.
    // n=10 — precision loss from usize→f64 cast is impossible at this size.
    #[allow(clippy::cast_precision_loss)]
    let v: Vec<f64> = (0..n)
        .map(|i| ((i as f64 + 1.0) * std::f64::consts::PI / (n as f64 + 1.0)).sin())
        .collect();

    let exact = oracle_expmv(n, &row_ptr, &col_idx, &vals, tau, &v);

    let ns = [50usize, 100, 200];
    let mut errs = [0.0_f64; 3];
    for (k, &steps) in ns.iter().enumerate() {
        let approx = implicit_expmv(n, &row_ptr, &col_idx, &vals, tau, steps, &v);
        errs[k] = sup_err(&exact, &approx);
        eprintln!(
            "G_SYMOP_IMPLICIT_DENSE  tau={tau}  n_steps={steps}  sup_error={:.3e}",
            errs[k]
        );
    }

    // (a) Finest grid must reach tolerance ≤ 1e-6.
    assert!(
        errs[2] <= 1e-6,
        "G_SYMOP_IMPLICIT_DENSE (a): sup_error={:.3e} > 1e-6 at n_steps=200",
        errs[2]
    );

    // (b) Order-1 ratio checks: err(n_steps)/err(2*n_steps) ∈ [1.7, 2.3].
    for (k, (&steps_coarse, _steps_fine)) in ns[..2].iter().zip(ns[1..].iter()).enumerate() {
        let ratio = errs[k] / errs[k + 1];
        eprintln!(
            "G_SYMOP_IMPLICIT_DENSE  ratio err({steps_coarse})/err({})  = {ratio:.4}",
            ns[k + 1]
        );
        assert!(
            (1.7..=2.3).contains(&ratio),
            "G_SYMOP_IMPLICIT_DENSE (b): order-1 ratio={ratio:.4} not in [1.7, 2.3]"
        );
    }

    eprintln!(
        "G_SYMOP_IMPLICIT_DENSE  wallclock={:.2}s  PASS",
        t0.elapsed().as_secs_f64()
    );
}
