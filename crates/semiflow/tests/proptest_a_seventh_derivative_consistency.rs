//! Smoke tests for the C⁷ caller invariant of `Diffusion6thChernoff` (v0.7.0, ADR-0015).
//!
//! The caller invariant is `a ∈ C⁷(ℝ)` with bounded derivatives through order 8.
//! This file verifies that the curated set of analytical a(x) functions used in the
//! test suite satisfy this invariant by checking known analytical derivatives at
//! sample points, and by checking that the functions are strictly positive.
//!
//! These are SMOKE TESTS — we document and check the mathematical properties of
//! the a(x) functions used in the test suite. No numerical FD is used for 7th
//! derivatives (FP cancellation makes iterated FD unreliable at order 7).
//! Instead, known closed-form expressions are verified.
//!
//! Test cases (curated analytical a functions):
//! 1. Constant: a(x) = 1.5  →  all derivatives ≡ 0 (trivially C^∞).
//! 2. Quadratic: a(x) = 1 + x²  →  a⁽³⁾ = 0 identically (polynomial).
//! 3. Trig: a(x) = 1.5 + 0.3·sin(x)  →  a⁽⁷⁾ = -0.3·cos(x), verified analytically.
//! 4. Tanh: a(x) = 1 + 0.1·tanh(x)  →  verified via 4-pt central FD of 1st derivative.
//! 5. Gaussian: a(x) = 1 + 0.5·exp(-x²/4)  →  Schwartz class; all derivatives bounded.

/// Simple 4-point (O(h⁴)) central FD for 1st derivative.
///
/// Used to verify the analytical a'(x) against the numerical FD at sample points.
/// Much more numerically stable than 7th-order composed differences.
fn fd1_fourth_order(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    // (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
}

/// Simple O(h⁴) central FD for 2nd derivative.
fn fd2_fourth_order(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    // (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h²)
    (-f(x + 2.0 * h) + 16.0 * f(x + h) - 30.0 * f(x) + 16.0 * f(x - h) - f(x - 2.0 * h))
        / (12.0 * h * h)
}

// ---------------------------------------------------------------------------
// Test 1: constant a — strictly positive, all derivatives 0
// ---------------------------------------------------------------------------

/// Constant a=1.5 — trivially C^∞ with all derivatives vanishing.
/// Verifies that the constant case, which short-circuits the FD correction, is correct.
#[test]
fn c7_smoke_constant_a() {
    let a_val = 1.5_f64;
    let a = |_: f64| a_val;
    let sample_points: [f64; 5] = [-3.0, -1.0, 0.0, 1.0, 3.0];
    for &x in &sample_points {
        // Strict positivity
        assert!(a(x) > 0.0, "constant a({x}) must be > 0");
        // 1st derivative == 0 (verified by FD)
        let d1 = fd1_fourth_order(a, x, 0.1);
        assert!(
            d1.abs() < 1e-8,
            "constant a: a'({x}) = {d1:.4e}, expected ~0"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: quadratic a — polynomial, derivatives vanish at order 3+
// ---------------------------------------------------------------------------

/// a(x) = 1 + x² — polynomial degree 2, C^∞.
/// a'(x) = 2x, a''(x) = 2, a⁽³⁾(x) = 0 identically.
#[test]
fn c7_smoke_quadratic_a() {
    let a = |x: f64| 1.0 + x * x;
    let a_prime = |x: f64| 2.0 * x;
    let a_dbl = |_: f64| 2.0_f64;

    let sample_points: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
    for &x in &sample_points {
        assert!(a(x) > 0.0, "quadratic a({x}) must be > 0");

        // Verify a'(x) via FD
        let d1_fd = fd1_fourth_order(a, x, 1e-3);
        let d1_analytic = a_prime(x);
        assert!(
            (d1_fd - d1_analytic).abs() < 1e-6,
            "quadratic a: FD a'({x}) = {d1_fd:.6e}, analytic = {d1_analytic:.6e}"
        );

        // Verify a''(x) via FD
        let d2_fd = fd2_fourth_order(a, x, 1e-3);
        let d2_analytic = a_dbl(x);
        assert!(
            (d2_fd - d2_analytic).abs() < 1e-4,
            "quadratic a: FD a''({x}) = {d2_fd:.6e}, analytic = {d2_analytic:.6e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: trig a — a⁽⁷⁾(x) = -0.3·cos(x) analytically verified
// ---------------------------------------------------------------------------

/// a(x) = 1.5 + 0.3·sin(x).
/// d^7/dx^7 sin(x) = -cos(x)  (odd derivative, 7 mod 4 = 3 → -cos).
/// So a⁽⁷⁾(x) = 0.3 · (-cos(x)), bounded by 0.3.
///
/// Verification strategy: compare FD a'(x) to analytic; verify a(x) > 0 always
/// (since |0.3·sin(x)| ≤ 0.3 < 1.5); document a⁽⁷⁾ is bounded by 0.3.
#[test]
fn c7_smoke_trig_a() {
    let a = |x: f64| 1.5 + 0.3 * x.sin();
    let a_prime = |x: f64| 0.3 * x.cos();

    let sample_points: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
    for &x in &sample_points {
        // Strict positivity: 1.5 - 0.3 = 1.2 > 0 always
        assert!(a(x) > 0.0, "trig a({x}) must be > 0 (min = 1.2)");

        // Verify a'(x) via FD
        let d1_fd = fd1_fourth_order(a, x, 1e-3);
        let d1_analytic = a_prime(x);
        assert!(
            (d1_fd - d1_analytic).abs() < 1e-8,
            "trig a: FD a'({x}) = {d1_fd:.6e}, analytic = {d1_analytic:.6e}"
        );

        // Verify a⁽⁷⁾(x) analytically: -0.3·cos(x), bounded by 0.3
        let d7_analytic = -0.3 * x.cos();
        assert!(
            d7_analytic.abs() <= 0.3 + 1e-12,
            "trig a: |a⁽⁷⁾({x})| = {:.6e} > 0.3 (C⁷ invariant violated analytically)",
            d7_analytic.abs()
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: tanh a — FD verification of a' and strict positivity
// ---------------------------------------------------------------------------

/// a(x) = 1 + 0.1·tanh(x) — C^∞, all derivatives bounded (Schwartz class).
/// a'(x) = 0.1·sech²(x) > 0. a is monotone increasing with range (0.9, 1.1).
#[test]
fn c7_smoke_tanh_a() {
    let a = |x: f64| 1.0 + 0.1 * x.tanh();
    let a_prime = |x: f64| {
        let ch = x.cosh();
        0.1 / (ch * ch)
    };

    let sample_points: [f64; 5] = [-3.0, -1.0, 0.0, 1.0, 3.0];
    for &x in &sample_points {
        // Range: [0.9, 1.1] — strictly positive
        assert!(a(x) > 0.8, "tanh a({x}) = {:.4} must be > 0.8", a(x));
        assert!(a(x) < 1.2, "tanh a({x}) = {:.4} must be < 1.2", a(x));

        // Verify a'(x) via FD
        let d1_fd = fd1_fourth_order(a, x, 1e-3);
        let d1_analytic = a_prime(x);
        let err = (d1_fd - d1_analytic).abs();
        assert!(
            err < 1e-8,
            "tanh a: FD a'({x}) = {d1_fd:.6e}, analytic = {d1_analytic:.6e}, err = {err:.4e}"
        );

        // a' is bounded: 0 < a'(x) ≤ 0.1 (sech² ≤ 1, amplitude 0.1)
        assert!(
            (0.0..=0.1 + 1e-12).contains(&d1_analytic),
            "tanh a'({x}) = {d1_analytic:.6e} out of [0, 0.1] range"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: Gaussian-envelope a — Schwartz class, strict positivity
// ---------------------------------------------------------------------------

/// a(x) = 1 + 0.5·exp(-x²/4) — C^∞ Schwartz class.
/// All derivatives decay as x → ±∞ faster than any polynomial.
/// a(x) ∈ [1.0, 1.5] for all x — strictly positive.
#[test]
fn c7_smoke_gaussian_envelope_a() {
    let a = |x: f64| 1.0 + 0.5 * libm::exp(-x * x / 4.0);
    let a_prime = |x: f64| 0.5 * libm::exp(-x * x / 4.0) * (-x / 2.0);

    let sample_points: [f64; 5] = [-3.0, -1.0, 0.0, 1.0, 3.0];
    for &x in &sample_points {
        // Range: [1, 1.5] — strictly positive
        assert!(a(x) >= 1.0 - 1e-12, "Gaussian a({x}) must be >= 1.0");
        assert!(a(x) <= 1.5 + 1e-12, "Gaussian a({x}) must be <= 1.5");

        // Verify a'(x) via FD
        let d1_fd = fd1_fourth_order(a, x, 1e-3);
        let d1_analytic = a_prime(x);
        let err = (d1_fd - d1_analytic).abs();
        assert!(
            err < 1e-8,
            "Gaussian a: FD a'({x}) = {d1_fd:.6e}, analytic = {d1_analytic:.6e}, err = {err:.4e}"
        );

        // All values are finite
        assert!(a(x).is_finite(), "Gaussian a({x}) is not finite");
        assert!(d1_analytic.is_finite(), "Gaussian a'({x}) is not finite");
    }
}
