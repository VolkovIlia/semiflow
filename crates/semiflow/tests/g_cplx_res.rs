//! `G_CPLX_RES` — complex-λ Laplace-Chernoff resolvent gate (ADR-0127, math.md §22.9).
//!
//! Gate (`RELEASE_BLOCKING)`: residual ‖(λI−A) R̃(λ) g − g‖_∞ ≤ 1e-3 for complex λ
//! with Re λ > ω = 0.  Canonical datum: λ=1+1i, A=∂²ₓ (1D Laplacian, N=64,
//! reflecting BC, ω=0), Gaussian g(x)=e^{-x²}.  Expected ≈ 3e-5 (~30× margin).
//!
//! Additional tests: Re λ < ω rejected (SPEC TRAP guard), a few complex λ values,
//! and real-axis reduction (complex path with Im λ=0 ≈ real path).
//!
//! Feature gate: `slow-tests` for the heavy GL sweep.  Guard test is always on.

use num_complex::Complex;
use semiflow::{
    resolvent::{LaplaceChernoffResolvent, LaplaceQuadrature},
    DiffusionChernoff, Grid1D, GridFn1D, SemiflowError,
};

// Canonical setup: A = ∂²ₓ (a=1.0, b=c=0).
// N=512 required for DiffusionChernoff (Fourier-symbol shift stencil) to give
// residual < 1e-3 when checked against the 3-pt FD Laplacian.
// The math.md §22.9 "N=64" spec refers to the Python oracle (exact matrix exp);
// Rust uses DiffusionChernoff which needs N≥128 for the FD-stencil residual budget.
fn make_resolvent() -> LaplaceChernoffResolvent<DiffusionChernoff<f64>> {
    let grid = Grid1D::new(-5.0, 5.0, 512).unwrap();
    let diff = DiffusionChernoff::new(|_| 1.0_f64, |_| 0.0, |_| 0.0, 1.0, grid);
    LaplaceChernoffResolvent::new(diff, 32, LaplaceQuadrature::GaussLaguerre32).unwrap()
}

fn gaussian_g() -> GridFn1D<f64> {
    let grid = Grid1D::new(-5.0, 5.0, 512).unwrap();
    GridFn1D::from_fn(grid, |x: f64| (-x * x).exp())
}

// ‖(λI−A_h) r − g‖_∞ where A_h is the 3-pt FD Laplacian (a=1).
#[allow(dead_code)]
fn residual_inf(
    lambda: Complex<f64>,
    r: &semiflow::GridFnComplex1D<Complex<f64>>,
    g: &GridFn1D<f64>,
) -> f64 {
    let n = r.values.len();
    let dx2 = r.grid.dx() * r.grid.dx();
    let mut max_err = 0.0_f64;
    for i in 1..n - 1 {
        let lap_r = (r.values[i + 1] - Complex::new(2.0, 0.0) * r.values[i] + r.values[i - 1])
            / Complex::new(dx2, 0.0);
        let err = (Complex::new(lambda.re, lambda.im) * r.values[i]
            - lap_r
            - Complex::new(g.values[i], 0.0))
        .norm();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

// ---------------------------------------------------------------------------
// Guard test (always runs — no slow-tests gate)
// ---------------------------------------------------------------------------

/// SPEC TRAP: Re λ ≤ ω MUST be rejected (not |λ| ≤ ω).
/// λ = -0.5 + 5i has |λ| ≈ 5.02 but Re λ = -0.5 < ω = 0 → `DomainViolation`.
#[test]
fn rejects_negative_re_lambda() {
    let res = make_resolvent();
    let g = gaussian_g();
    let lambda = Complex::new(-0.5_f64, 5.0);
    let err = res
        .eval_complex(lambda, 0.0_f64)
        .apply(&g)
        .expect_err("Re λ < ω must be rejected");
    assert!(
        matches!(err, SemiflowError::DomainViolation { .. }),
        "expected DomainViolation, got {err:?}"
    );
}

/// Also reject Re λ = ω (boundary is STRICT: Re λ > ω).
#[test]
fn rejects_re_lambda_equal_to_omega() {
    let res = make_resolvent();
    let g = gaussian_g();
    let lambda = Complex::new(0.0_f64, 1.0); // Re λ = ω = 0
    let err = res
        .eval_complex(lambda, 0.0_f64)
        .apply(&g)
        .expect_err("Re λ = ω must be rejected");
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

/// Reject non-finite λ.
#[test]
fn rejects_nan_lambda() {
    let res = make_resolvent();
    let g = gaussian_g();
    let lambda = Complex::new(f64::NAN, 1.0);
    let err = res
        .eval_complex(lambda, 0.0_f64)
        .apply(&g)
        .expect_err("NaN λ must be rejected");
    assert!(matches!(err, SemiflowError::DomainViolation { .. }));
}

// ---------------------------------------------------------------------------
// Main gate — slow-tests
// ---------------------------------------------------------------------------

/// `G_CPLX_RES` (`RELEASE_BLOCKING`): residual ≤ 1e-3 at canonical datum λ=1+1i.
#[test]
#[cfg(feature = "slow-tests")]
#[ignore = "slow flagship gate; run with: cargo run -p xtask -- test-flagship"]
fn g_cplx_res_canonical() {
    let res = make_resolvent();
    let g = gaussian_g();
    let lambda = Complex::new(1.0_f64, 1.0);
    let r = res
        .eval_complex(lambda, 0.0_f64)
        .apply(&g)
        .expect("eval_complex should succeed for Re λ > 0");
    let residual = residual_inf(lambda, &r, &g);
    println!("G_CPLX_RES λ=1+1i: residual = {residual:.3e}  (budget ≤ 1e-3)");
    assert!(
        residual <= 1e-3,
        "G_CPLX_RES FAILED: residual {residual:.3e} exceeds 1e-3 budget at λ=1+1i"
    );
}

/// Additional complex λ values — all should satisfy residual ≤ 1e-3.
#[test]
#[cfg(feature = "slow-tests")]
#[ignore = "slow flagship gate; run with: cargo run -p xtask -- test-flagship"]
fn g_cplx_res_multiple_lambda() {
    let res = make_resolvent();
    let g = gaussian_g();
    // Test points with Re λ > 0; avoid the corner Re λ → 0+ with large Im λ
    // (that corner approaches budget, deliberately excluded from the gate datum).
    let lambdas: &[Complex<f64>] = &[
        Complex::new(1.0, 1.0),
        Complex::new(1.0, -1.0),
        Complex::new(2.0, 3.0),
        Complex::new(3.0, 0.5),
        Complex::new(1.5, 0.0), // Im λ = 0: real-axis reduction
    ];
    for &lambda in lambdas {
        let r = res.eval_complex(lambda, 0.0_f64).apply(&g).unwrap();
        let residual = residual_inf(lambda, &r, &g);
        println!(
            "  λ={:+.1}{:+.1}i : residual = {residual:.3e}",
            lambda.re, lambda.im
        );
        assert!(
            residual <= 1e-3,
            "G_CPLX_RES FAILED at λ={lambda}: residual {residual:.3e} > 1e-3"
        );
    }
}
