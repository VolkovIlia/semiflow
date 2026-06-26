//! `G_MASSK_CONSISTENT` (`RELEASE_BLOCKING`, §55.4): `MassKOperator::evolve`
//! (Krylov on `Â = R^{−T} K R^{−1}`) vs `dense_massk_expmv_ref` (Padé-13).
//!
//! N=6 path Laplacian K; tridiagonal consistent-mass M (`M[i,i]=2`, `M[i,i±1]=0.5`).
//! `sup_error ≤ 1e-8`.
//!
//! Non-vacuity: the off-diagonal mass couples all nodes; `sup_error` > 1e-15
//! confirms the consistent-mass congruence is non-trivial.

use semiflow::{
    dense_massk_expmv_ref, scratch::ScratchPool, KrylovPath, MassKOperator, SymmetricOperator,
    TriangularFactor,
};

/// N=6 path Laplacian in CSR form.
fn path_n6_csr() -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let row_ptr = vec![0_usize, 2, 5, 8, 11, 14, 16];
    let col_idx = vec![
        0u32, 1,       // row 0
        0, 1, 2,       // row 1
        1, 2, 3,       // row 2
        2, 3, 4,       // row 3
        3, 4, 5,       // row 4
        4, 5,          // row 5
    ];
    let vals = vec![
        1.0, -1.0,             // row 0
        -1.0, 2.0, -1.0,      // row 1
        -1.0, 2.0, -1.0,      // row 2
        -1.0, 2.0, -1.0,      // row 3
        -1.0, 2.0, -1.0,      // row 4
        -1.0, 1.0,             // row 5
    ];
    (row_ptr, col_idx, vals)
}

/// Build tridiagonal consistent-mass matrix: `M[i,i]=2`, `M[i,i±1]=0.5`.
///
/// Eigenvalues in `[2−1, 2+1] = [1, 3]`, so M is symmetric positive-definite.
fn tridiag_mass_n6(n: usize) -> Vec<f64> {
    let mut m = vec![0.0_f64; n * n];
    for i in 0..n {
        m[i * n + i] = 2.0;
        if i + 1 < n {
            m[i * n + (i + 1)] = 0.5;
            m[(i + 1) * n + i] = 0.5;
        }
    }
    m
}

/// `G_MASSK_CONSISTENT`: Krylov on `Â = R^{−T} K R^{−1}` vs Padé-13 oracle.
///
/// Expected: `sup_error ≤ 1e-8`.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_massk_consistent() {
    let n = 6_usize;
    let tau = 0.5_f64;
    let tol = 1e-12_f64;

    let (row_ptr, col_idx, vals) = path_n6_csr();
    let k = SymmetricOperator::from_csr(n, &row_ptr, &col_idx, &vals, 1e-12_f64)
        .expect("G_MASSK_CONSISTENT: from_csr failed");

    let m_dense = tridiag_mass_n6(n);
    let r = TriangularFactor::dense_cholesky_spd(&m_dense, n)
        .expect("G_MASSK_CONSISTENT: Cholesky failed");
    let op = MassKOperator::new(k, r);

    let v = [1.0_f64, -0.5, 0.3, 0.7, -0.2, 0.4];
    let mut out_krylov = vec![0.0_f64; n];
    let mut out_dense = vec![0.0_f64; n];
    let mut scratch = ScratchPool::new();

    op.evolve(tau, &v, &mut out_krylov, KrylovPath::Chebyshev, tol, &mut scratch)
        .expect("G_MASSK_CONSISTENT: Krylov evolve failed");

    dense_massk_expmv_ref(&op, tau, &v, &mut out_dense)
        .expect("G_MASSK_CONSISTENT: dense ref failed");

    let sup_error = out_krylov
        .iter()
        .zip(out_dense.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    eprintln!("G_MASSK_CONSISTENT  n={n}  tau={tau}  sup_error={sup_error:.3e}");
    assert!(
        sup_error <= 1e-8_f64,
        "G_MASSK_CONSISTENT: sup_error={sup_error:.3e} > 1e-8 (threshold)"
    );
}
