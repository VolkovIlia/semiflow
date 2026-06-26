//! `G_MASSK_LUMPED` (`RELEASE_BLOCKING`, §55.3): `mass_lumped_evolve` vs dense
//! Padé-13 oracle on the congruence `Â = D^{−½} K D^{−½}`.
//!
//! N=6 path Laplacian K; masses m = [1.0, 2.0, 1.5, 0.8, 1.2, 0.9].
//! `sup_error ≤ 1e-8`.
//!
//! Non-vacuity: `sup_error` is printed; any result > 1e-15 confirms real
//! computation (non-unit masses break trivial orbits).

use semiflow::{
    dense_csr_expmv_ref, mass_lumped_evolve, scratch::ScratchPool, KrylovPath,
    SymmetricOperator,
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

/// `G_MASSK_LUMPED`: `mass_lumped_evolve` vs dense oracle on lumped congruence.
///
/// Reference: `D^{½} v → dense_csr_expmv_ref(Â, τ, ·) → D^{−½}`.
///
/// Expected: `sup_error ≤ 1e-8`.
#[test]
#[ignore = "slow-test: run with --features slow-tests --release -- --ignored"]
fn g_massk_lumped() {
    let n = 6_usize;
    let tau = 0.5_f64;
    let tol = 1e-12_f64;

    let (row_ptr, col_idx, vals) = path_n6_csr();
    let k = SymmetricOperator::from_csr(n, &row_ptr, &col_idx, &vals, 1e-12_f64)
        .expect("G_MASSK_LUMPED: from_csr failed");

    let masses = [1.0_f64, 2.0, 1.5, 0.8, 1.2, 0.9];
    let v = [1.0_f64, -0.5, 0.3, 0.7, -0.2, 0.4];

    let mut out_krylov = vec![0.0_f64; n];
    let mut scratch = ScratchPool::new();

    mass_lumped_evolve(&k, &masses, tau, &v, &mut out_krylov, KrylovPath::Chebyshev, tol, &mut scratch)
        .expect("G_MASSK_LUMPED: lumped_evolve failed");

    // Reference: build Â = D^{−½} K D^{−½}, pre-scale, dense-evolve, post-scale.
    let a_hat = k.lumped_congruence(&masses).expect("G_MASSK_LUMPED: congruence failed");
    let w0: Vec<f64> = v.iter().zip(masses.iter()).map(|(&vi, &mi)| vi * mi.sqrt()).collect();
    let mut w1 = vec![0.0_f64; n];
    dense_csr_expmv_ref(&a_hat, tau, &w0, &mut w1).expect("G_MASSK_LUMPED: dense ref failed");
    let out_ref: Vec<f64> =
        w1.iter().zip(masses.iter()).map(|(&wi, &mi)| wi / mi.sqrt()).collect();

    let sup_error = out_krylov
        .iter()
        .zip(out_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    eprintln!("G_MASSK_LUMPED  n={n}  tau={tau}  sup_error={sup_error:.3e}");
    assert!(
        sup_error <= 1e-8_f64,
        "G_MASSK_LUMPED: sup_error={sup_error:.3e} > 1e-8 (threshold)"
    );
}
