//! G14 + G15 — `diffusion_chernoff_a_prime_consistency` and
//! `diffusion_chernoff_a_double_prime_consistency` (v0.3.0, ADR-0008).
//!
//! Soft caller-correctness checks that user-supplied analytic derivatives
//! `a_prime(x)` and `a_double_prime(x)` match central-FD approximations
//! within 1% relative tolerance.  These are **informational** (the soft
//! check uses `prop_assume!` to skip degenerate cases rather than abort).
//!
//! G14 — `a_prime` consistency:
//!   `|a'(x) - (a(x+h) - a(x-h))/(2h)| ≤ 0.01·|a'(x)| + 1e-6`, h = 1e-4.
//!   50 cases, x ∈ [-2, 2].
//!
//! G15 — `a_double_prime` consistency:
//!   `|a''(x) - (a(x+h) - 2·a(x) + a(x-h))/h²| ≤ 0.01·|a''(x)| + 1e-6`,
//!   h = 1e-3.
//!   50 cases, x ∈ [-2, 2].
//!
//! Three fixed `(a, a', a'')` triples (no closures needed — fn pointers):
//!   1. `a(x) = 0.5 + 0.1·x²`
//!   2. `a(x) = 1.0 + 0.05·sin(x)`   (approximated via libm)
//!   3. `a(x) = (1.0 + 0.1·x)²`
//!
//! Reference: `contracts/semiflow-core.properties.yaml`
//! `diffusion_chernoff_a_prime_consistency` and
//! `diffusion_chernoff_a_double_prime_consistency`.

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Triple 1: a(x) = 0.5 + 0.1·x²
// ---------------------------------------------------------------------------
fn a1(x: f64) -> f64 {
    0.5 + 0.1 * x * x
}
fn ap1(x: f64) -> f64 {
    0.2 * x
}
fn app1(_: f64) -> f64 {
    0.2
}

// ---------------------------------------------------------------------------
// Triple 2: a(x) = 1.0 + 0.05·sin(x)
// ---------------------------------------------------------------------------
fn a2(x: f64) -> f64 {
    1.0 + 0.05 * libm::sin(x)
}
fn ap2(x: f64) -> f64 {
    0.05 * libm::cos(x)
}
fn app2(x: f64) -> f64 {
    -0.05 * libm::sin(x)
}

// ---------------------------------------------------------------------------
// Triple 3: a(x) = (1 + 0.1·x)²
// ---------------------------------------------------------------------------
fn a3(x: f64) -> f64 {
    let v = 1.0 + 0.1 * x;
    v * v
}
fn ap3(x: f64) -> f64 {
    2.0 * 0.1 * (1.0 + 0.1 * x)
}
fn app3(_: f64) -> f64 {
    2.0 * 0.1 * 0.1
}

type Triple = (fn(f64) -> f64, fn(f64) -> f64, fn(f64) -> f64);

const TRIPLES: [Triple; 3] = [(a1, ap1, app1), (a2, ap2, app2), (a3, ap3, app3)];

// ---------------------------------------------------------------------------
// G14 — a_prime consistency
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 50, ..ProptestConfig::default() })]

    /// G14: analytic `a'(x)` matches central-FD first derivative within 1%.
    #[test]
    fn g14_a_prime_consistency(
        triple_idx in 0usize..3,
        x in -2.0f64..=2.0f64,
    ) {
        let (a, a_prime, _) = TRIPLES[triple_idx];
        let h = 1.0e-4_f64;
        let fd_aprime = (a(x + h) - a(x - h)) / (2.0 * h);
        let analytic = a_prime(x);
        let tol = 0.01 * analytic.abs() + 1.0e-6;
        // Soft check — use prop_assume! to skip numerically degenerate cases.
        prop_assume!(tol > 0.0);
        prop_assert!(
            (analytic - fd_aprime).abs() <= tol,
            "G14 a_prime mismatch at triple={triple_idx}, x={x:.6}: \
             analytic={analytic:.10e}, fd={fd_aprime:.10e}, \
             diff={:.4e}, tol={tol:.4e}",
            (analytic - fd_aprime).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// G15 — a_double_prime consistency
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 50, ..ProptestConfig::default() })]

    /// G15: analytic `a''(x)` matches central-FD second derivative within 1%.
    #[test]
    fn g15_a_double_prime_consistency(
        triple_idx in 0usize..3,
        x in -2.0f64..=2.0f64,
    ) {
        let (a, _, a_double_prime) = TRIPLES[triple_idx];
        let h = 1.0e-3_f64;
        let fd_app = (a(x + h) - 2.0 * a(x) + a(x - h)) / (h * h);
        let analytic = a_double_prime(x);
        let tol = 0.01 * analytic.abs() + 1.0e-6;
        prop_assume!(tol > 0.0);
        prop_assert!(
            (analytic - fd_app).abs() <= tol,
            "G15 a_double_prime mismatch at triple={triple_idx}, x={x:.6}: \
             analytic={analytic:.10e}, fd={fd_app:.10e}, \
             diff={:.4e}, tol={tol:.4e}",
            (analytic - fd_app).abs()
        );
    }
}
