//! Stiff-regime Chebyshev accuracy test (Issue #11 / #14 guard).
//!
//! Verifies that `graph_expmv_krylov` (Chebyshev path) handles arbitrarily-stiff
//! operators without silently producing NaN or garbage.
//!
//! **Pre-fix bug**: for the k=[1,100,1] conservative operator (n=9, Neumann) the
//! Gershgorin bound is `λ_max ≈ 25600`, so `z = τ·λ_max/2 ≈ 6400` at τ=0.5.
//! Both `exp(−6400)` (underflows to 0.0) and the Bessel series `I_k(6400)`
//! (overflows to ∞) are non-finite → `0·∞ = NaN` propagates into every
//! Chebyshev coefficient and the output vector is silently filled with NaN.
//! Because `f64::max(acc, NaN) = acc`, the subsequent `sup_error` fold produced
//! 0.0, masking the total failure.
//!
//! **Post-fix**: the Chebyshev path splits τ into `s = ⌈z / Z_SAFE⌉` substeps
//! (each with `z_sub = τ/s · λ_max/2 ≤ Z_SAFE = 200`), applies one Chebyshev
//! step per substep, and rejects any non-finite output with `SemiflowError`.
//!
//! `cheb_stiff_no_nan` runs in test-fast (no `#[ignore]`).

use semiflow::{
    assemble_conservative_csr_1d,
    boundary::BoundaryPolicy,
    dense_csr_expmv_ref,
    graph_krylov::{graph_expmv_krylov, KrylovPath},
    grid::Grid1D,
    scratch::ScratchPool,
    SymmetricOperator,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// k=[1,100,1] step profile: indices [lo, hi) get `k_inner`, rest get `k_outer`.
fn k_three_layer(n: usize, lo: usize, hi: usize, k_outer: f64, k_inner: f64) -> Vec<f64> {
    (0..n)
        .map(|i| if i < lo || i >= hi { k_outer } else { k_inner })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Stiff test: k=[1,100,1], n=9, τ=0.5
// ─────────────────────────────────────────────────────────────────────────────

/// `cheb_stiff_no_nan` (fast, no `#[ignore]`): conservative 1-D operator with
/// k=[1,100,1] (n=9 nodes, Neumann BCs).
///
/// Computed Gershgorin bound ≈ 25 600, giving `z = τ·λ_max/2 ≈ 6 400` at τ=0.5.
///
/// **Pre-fix**: `dst_krylov` is filled with NaN → the first `assert!` (finite guard)
/// fails.  `cargo test -- cheb_stiff_no_nan` shows:
/// ```text
/// thread 'cheb_stiff_no_nan' panicked at:
/// cheb_stiff_no_nan: dst_krylov contains NaN — Chebyshev underflow (z=6400)
/// ```
///
/// **Post-fix**: output is finite, agrees with Padé-13 within `1e-10`, and the
/// two-sided band `1e-16 < sup_error ≤ 1e-10` confirms a genuine independent
/// comparison (not shared-path or NaN masking).
///
/// Non-vacuity:
/// 1. Asserts z > 1 000 before running (ensures the stiff regime is truly exercised).
/// 2. NaN/Inf guard on `dst_krylov` (primary gate — catches the pre-fix failure).
/// 3. `dst_norm > 1e-14` (non-trivial action — guards against zero output).
/// 4. `sup_error > 1e-16` (genuinely independent algorithms; rules out NaN masking).
/// 5. `sup_error ≤ 1e-10` (accuracy gate).
#[test]
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn cheb_stiff_no_nan() {
    let n = 9_usize;
    let k_nodes = k_three_layer(n, 3, 6, 1.0, 100.0);

    let grid = Grid1D::new(0.0_f64, 1.0_f64, n).expect("cheb_stiff_no_nan: grid");
    let op: SymmetricOperator<f64> =
        assemble_conservative_csr_1d(grid, &k_nodes, None, BoundaryPolicy::Neumann)
            .expect("cheb_stiff_no_nan: assemble");

    let lambda_max = op.lambda_max_bound();
    let tau = 0.5_f64;
    let z = tau * lambda_max / 2.0;

    // Non-vacuity (1): confirm the operator is genuinely stiff.
    eprintln!(
        "cheb_stiff_no_nan  n={n}  k=[1,100,1]  lambda_max={lambda_max:.0}  \
         tau={tau}  z={z:.0}"
    );
    assert!(
        z > 1_000.0,
        "cheb_stiff_no_nan: z={z:.0} ≤ 1000 — operator not stiff enough to trigger bug"
    );

    // Gaussian test vector centred on node 4.
    let src: Vec<f64> = (0..n)
        .map(|i| { let x = i as f64 - 4.0; (-0.5 * x * x).exp() })
        .collect();

    let mut dst_krylov = vec![0.0_f64; n];
    let mut dst_dense  = vec![0.0_f64; n];
    let mut scratch = ScratchPool::new();

    // Use tol=1e-12 per substep so the accumulated error over s≈32 substeps
    // stays below 1e-10 total (worst-case: s × tol_per_step ≈ 32 × 1e-12 = 3.2e-11).
    graph_expmv_krylov(
        &op, tau, &src, &mut dst_krylov, KrylovPath::Chebyshev, 1e-12, &mut scratch,
    )
    .expect("cheb_stiff_no_nan: krylov returned Err");

    dense_csr_expmv_ref(&op, tau, &src, &mut dst_dense)
        .expect("cheb_stiff_no_nan: dense ref failed");

    // Non-vacuity (2): NaN/Inf guard — the pre-fix failure surface.
    assert!(
        !dst_krylov.iter().any(|v| !v.is_finite()),
        "cheb_stiff_no_nan: dst_krylov contains NaN/Inf — Chebyshev underflow? \
         (z={z:.0}, lambda_max={lambda_max:.0})"
    );

    let sup_error = dst_krylov
        .iter()
        .zip(dst_dense.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let dst_norm = dst_dense.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    eprintln!(
        "cheb_stiff_no_nan  dst_norm={dst_norm:.3e}  sup_error={sup_error:.3e}"
    );

    // Non-vacuity (3): non-trivial result.
    assert!(dst_norm > 1e-14, "cheb_stiff_no_nan: dst_norm={dst_norm:.3e} trivially zero");

    // Non-vacuity (4) + (5): two-sided accuracy band.
    assert!(
        sup_error > 1e-16,
        "cheb_stiff_no_nan: sup_error={sup_error:.3e} ≤ 1e-16 — \
         shared code path or NaN masking (expected genuine algorithmic error)"
    );
    assert!(
        sup_error <= 1e-10,
        "cheb_stiff_no_nan: sup_error={sup_error:.3e} > 1e-10"
    );
}
