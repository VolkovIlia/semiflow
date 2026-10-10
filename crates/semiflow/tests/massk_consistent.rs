//! `G_MASSK_CONSISTENT` (`RELEASE_BLOCKING`, §55.4): `MassKOperator::evolve`
//! (Krylov on `Â = R^{−T} K R^{−1}`) vs `dense_massk_expmv_ref` (Padé-13).
//!
//! N=6 path Laplacian K; tridiagonal consistent-mass M (`M[i,i]=2`, `M[i,i±1]=0.5`).
//! `sup_error ≤ 1e-11`.
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
        0u32, 1, // row 0
        0, 1, 2, // row 1
        1, 2, 3, // row 2
        2, 3, 4, // row 3
        3, 4, 5, // row 4
        4, 5, // row 5
    ];
    let vals = vec![
        1.0, -1.0, // row 0
        -1.0, 2.0, -1.0, // row 1
        -1.0, 2.0, -1.0, // row 2
        -1.0, 2.0, -1.0, // row 3
        -1.0, 2.0, -1.0, // row 4
        -1.0, 1.0, // row 5
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
/// Expected: `sup_error ≤ 1e-11`.
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

    op.evolve(
        tau,
        &v,
        &mut out_krylov,
        KrylovPath::Chebyshev,
        tol,
        &mut scratch,
    )
    .expect("G_MASSK_CONSISTENT: Krylov evolve failed");

    dense_massk_expmv_ref(&op, tau, &v, &mut out_dense)
        .expect("G_MASSK_CONSISTENT: dense ref failed");

    let sup_error = out_krylov
        .iter()
        .zip(out_dense.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    eprintln!("G_MASSK_CONSISTENT  n={n}  tau={tau}  sup_error={sup_error:.3e}");
    // Tightened 1e-8 → 1e-11 once the Chebyshev scaling bound became rigorous
    // (measured 2.2e-13; it was 3.5e-10 with the under-estimated λ_max).
    assert!(
        sup_error <= 1e-11_f64,
        "G_MASSK_CONSISTENT: sup_error={sup_error:.3e} > 1e-11 (threshold)"
    );
}

/// P1 finite-element stiffness `K` (Dirichlet) and consistent mass `M` on `n`
/// interior nodes of `[0, 1]`: returns `(K, dense M, h)`.
fn p1_fem(n: usize) -> (SymmetricOperator<f64>, Vec<f64>, f64) {
    #[allow(clippy::cast_precision_loss)]
    let h = 1.0 / (n as f64 + 1.0);
    let (mut rp, mut ci, mut va) = (vec![0_usize], Vec::new(), Vec::new());
    let mut m = vec![0.0_f64; n * n];
    for i in 0..n {
        #[allow(clippy::cast_possible_truncation)]
        let iu = i as u32;
        if i > 0 {
            ci.push(iu - 1);
            va.push(-1.0 / h);
            m[i * n + i - 1] = h / 6.0;
        }
        ci.push(iu);
        va.push(2.0 / h);
        m[i * n + i] = 2.0 * h / 3.0;
        if i + 1 < n {
            ci.push(iu + 1);
            va.push(-1.0 / h);
            m[i * n + i + 1] = h / 6.0;
        }
        rp.push(ci.len());
    }
    let k = SymmetricOperator::from_csr(n, &rp, &ci, &va, 1e-12).expect("P1 stiffness");
    (k, m, h)
}

/// `G_MASSK_RIGOROUS_BOUND` (`RELEASE_BLOCKING`): the Chebyshev scaling bound of
/// `MassKOperator` is a true upper bound of `λ_max(M⁻¹K)`, and the action is accurate.
///
/// P1 consistent mass is the adversarial case for the old 5-step inverse-power
/// estimate: the constant start vector is the eigenvector of `M` with its LARGEST
/// eigenvalue, so `λ_min(M)` was over-estimated, `λ_max(Â)` under-estimated
/// (1450 vs the exact 1942 at `n = 12`) and the Chebyshev series diverged
/// (relative error 1.7e11). Closed-form spectrum of the generalized problem:
/// `λ_k = (6/h²)(1 − cos θ_k)/(2 + cos θ_k)`, `θ_k = kπ/(n+1)`.
#[test]
#[allow(clippy::many_single_char_names)] // n, k, m, h, r, v: FEM notation
fn g_massk_rigorous_bound() {
    for n in [4_usize, 12, 40] {
        let (k, m, h) = p1_fem(n);
        let r = TriangularFactor::dense_cholesky_spd(&m, n).expect("Cholesky");
        let op = MassKOperator::new(k, r);
        #[allow(clippy::cast_precision_loss)]
        let theta = n as f64 * core::f64::consts::PI / (n as f64 + 1.0);
        let lambda_exact = 6.0 / (h * h) * (1.0 - theta.cos()) / (2.0 + theta.cos());
        let bound = semiflow::SymmetricLinearOp::lambda_max_bound(&op);
        assert!(
            bound >= lambda_exact && bound <= 1.5 * lambda_exact,
            "G_MASSK_RIGOROUS_BOUND n={n}: bound {bound} vs exact lambda_max {lambda_exact}"
        );
        if n > 12 {
            continue; // dense Padé oracle is limited to n <= 12
        }
        // Highest-frequency datum: the component the under-estimate blew up.
        let v: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let tau = 0.05;
        let (mut out, mut refv) = (vec![0.0; n], vec![0.0; n]);
        let mut scratch = ScratchPool::new();
        op.evolve(
            tau,
            &v,
            &mut out,
            KrylovPath::Chebyshev,
            1e-12,
            &mut scratch,
        )
        .expect("Chebyshev evolve");
        dense_massk_expmv_ref(&op, tau, &v, &mut refv).expect("dense ref");
        let err = out
            .iter()
            .zip(&refv)
            .map(|(a, b)| (a - b).abs())
            .fold(
                0.0_f64,
                |acc, e| if e.is_nan() { f64::NAN } else { acc.max(e) },
            );
        let scale = v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(
            err <= 1e-11 * scale,
            "G_MASSK_RIGOROUS_BOUND n={n}: sup error {err:e} > 1e-11·‖v‖∞"
        );
    }
}
