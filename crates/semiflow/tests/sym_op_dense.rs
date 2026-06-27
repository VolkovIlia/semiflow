//! `G_SYMOP_DENSE` (`RELEASE_BLOCKING`, §55.2): `graph_expmv_krylov` on a
//! [`SymmetricOperator`] assembled from raw CSR vs `dense_csr_expmv_ref`.
//!
//! N=10 path Laplacian **+ 0.5·I** (reaction/Robin term).  Every row sum = 0.5 ≠ 0,
//! exercising the non-zero-row-sum capability introduced in issue #13 (which the
//! previous A1 `GraphChernoff` could not represent).
//! Chebyshev at tol=1e-12.  `sup_error ≤ 1e-10`.
//!
//! Non-vacuity:
//! 1. An explicit assertion (before `from_csr`) verifies that at least one row
//!    sum is non-zero, so the gate cannot silently regress to a zero-sum operator.
//! 2. `sup_error` is printed; any value ≥ 1e-14 confirms the Chebyshev path
//!    is doing real work (not a trivially-zero action).

use semiflow::{
    dense_csr_expmv_ref, graph_expmv_krylov, scratch::ScratchPool, KrylovPath,
    SymmetricOperator,
};

/// N=10 path Laplacian + 0.5·I in CSR form (row-major, sorted columns).
///
/// Every diagonal is bumped by c = 0.5 so every row sums to 0.5 ≠ 0:
/// - endpoint rows (0, 9): `1.5 − 1.0 = 0.5`
/// - interior rows (1–8): `2.5 − 1.0 − 1.0 = 0.5`
///
/// The matrix is symmetric and PSD (path Laplacian is PSD; adding c·I with c > 0
/// shifts all eigenvalues up by c, preserving positive semi-definiteness).
fn path_n10_reaction_csr() -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    // c = 0.5 added to each diagonal entry of the path Laplacian.
    let row_ptr = vec![0_usize, 2, 5, 8, 11, 14, 17, 20, 23, 26, 28];
    let col_idx = vec![
        0u32, 1,       // row 0
        0, 1, 2,       // row 1
        1, 2, 3,       // row 2
        2, 3, 4,       // row 3
        3, 4, 5,       // row 4
        4, 5, 6,       // row 5
        5, 6, 7,       // row 6
        6, 7, 8,       // row 7
        7, 8, 9,       // row 8
        8, 9,          // row 9
    ];
    let vals = vec![
        1.5, -1.0,             // row 0: sum = 0.5
        -1.0, 2.5, -1.0,      // row 1: sum = 0.5
        -1.0, 2.5, -1.0,      // row 2: sum = 0.5
        -1.0, 2.5, -1.0,      // row 3: sum = 0.5
        -1.0, 2.5, -1.0,      // row 4: sum = 0.5
        -1.0, 2.5, -1.0,      // row 5: sum = 0.5
        -1.0, 2.5, -1.0,      // row 6: sum = 0.5
        -1.0, 2.5, -1.0,      // row 7: sum = 0.5
        -1.0, 2.5, -1.0,      // row 8: sum = 0.5
        -1.0, 1.5,             // row 9: sum = 0.5
    ];
    (row_ptr, col_idx, vals)
}

/// `G_SYMOP_DENSE`: `graph_expmv_krylov` on `SymmetricOperator` vs Padé-13 oracle.
///
/// Expected: `sup_error ≤ 1e-10`.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
#[allow(clippy::cast_precision_loss)] // i < n ≤ 12; no precision loss possible
fn g_symop_dense() {
    let n = 10_usize;
    let tau = 1.0_f64;
    let tol = 1e-12_f64;

    let (row_ptr, col_idx, vals) = path_n10_reaction_csr();

    // Non-vacuity: assert at least one row sum is non-zero (gates #13 headline capability).
    // This assertion fails if the operator silently regresses to a zero-row-sum Laplacian.
    let row_sums: Vec<f64> = (0..n)
        .map(|i| vals[row_ptr[i]..row_ptr[i + 1]].iter().sum::<f64>())
        .collect();
    assert!(
        row_sums.iter().any(|&s| s.abs() > 1e-15_f64),
        "G_SYMOP_DENSE: all row sums are zero — gate is vacuous (regressed to zero-sum case)"
    );

    let op = SymmetricOperator::from_csr(n, &row_ptr, &col_idx, &vals, 1e-12_f64)
        .expect("G_SYMOP_DENSE: from_csr failed");

    // Gaussian-like signal centred on node 5.
    let src: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 - 5.0;
            (-0.5_f64 * x * x).exp()
        })
        .collect();
    let mut dst_krylov = vec![0.0_f64; n];
    let mut dst_dense = vec![0.0_f64; n];
    let mut scratch = ScratchPool::new();

    graph_expmv_krylov(
        &op, tau, &src, &mut dst_krylov, KrylovPath::Chebyshev, tol, &mut scratch,
    )
    .expect("G_SYMOP_DENSE: krylov failed");

    dense_csr_expmv_ref(&op, tau, &src, &mut dst_dense)
        .expect("G_SYMOP_DENSE: dense ref failed");

    let sup_error = dst_krylov
        .iter()
        .zip(dst_dense.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    eprintln!(
        "G_SYMOP_DENSE  n={n}  tau={tau}  sup_error={sup_error:.3e}  \
         row_sum[0]={:.3e} (expected 0.5)",
        row_sums[0]
    );
    assert!(
        sup_error <= 1e-10_f64,
        "G_SYMOP_DENSE: sup_error={sup_error:.3e} > 1e-10 (threshold)"
    );
}
