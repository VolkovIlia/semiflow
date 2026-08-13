//! `G_THETA_M_TABLE` — every `(m, θ_m)` pair is a radius that degree `m` actually
//! honours (ADR-0197, §45.2).
//!
//! `θ_m` is the largest argument at which a degree-`m` truncated Taylor
//! exponential is accurate to double precision. `select_s_m` derives the substep
//! count from it: `s = ⌈τ‖A‖ / θ_m⌉`. A θ that is too large produces too few
//! substeps and a silently wrong answer — which is exactly what shipped, because
//! each degree had been paired with the radius of a degree two to three rows
//! further down Al-Mohy & Higham's Table 3.1 (`m = 18` carried `θ ≈ 8.84`, whose
//! true owner is `m ≈ 51`; its own radius is 1.09).
//!
//! Nothing in the previous test suite could see it. The accuracy gate
//! (`expmv_div_form_action_accuracy`) measures **absolute** sup-error on a
//! strongly decaying symmetric operator, where a relative error of `10⁴` at the
//! claimed radius hides inside an absolute error of `10⁻¹⁵`, and its own
//! reference runs at per-step argument ≤ 1.0 — inside the *correct* radius. This
//! gate tests the table directly instead, so a mis-paired entry fails at that
//! entry.

// Factorial/series arithmetic on small test constants.
#![allow(clippy::cast_precision_loss)]

/// Degree-`m` truncated Taylor exponential, evaluated the way `horner_step` does.
fn taylor(x: f64, m: u32) -> f64 {
    let mut sum = 0.0_f64;
    let mut term = 1.0_f64;
    for k in 0..=m {
        sum += term;
        term *= x / f64::from(k + 1);
    }
    sum
}

/// Both shipped tables, as `(m, θ_m)`.
///
/// Duplicated here on purpose: the gate must fail if the tables drift, and a
/// gate that imports the constant it is checking cannot detect a wrong value.
const TABLE: &[(u32, f64)] = &[
    (1, 2.220e-16),
    (2, 2.581e-8),
    (3, 1.386e-5),
    (4, 3.397e-4),
    (5, 2.401e-3),
    (6, 9.066e-3),
    (7, 2.384e-2),
    (8, 4.991e-2),
    (9, 8.958e-2),
    (10, 1.442e-1),
    (11, 2.142e-1),
    (12, 2.996e-1),
    (13, 3.998e-1),
    (14, 5.139e-1),
    (15, 6.411e-1),
    (16, 7.803e-1),
    (17, 9.305e-1),
    (18, 1.091),
    (19, 1.260),
    (20, 1.438),
    (25, 2.429),
    (30, 3.540),
];

/// `G_THETA_M_TABLE` — degree `m` reproduces `exp(±θ_m)` to double precision.
///
/// The tolerance is `1e-12`, not `1e-16`, because the *forward* error of a
/// monomial Horner evaluation at argument θ carries an `O(θ·u)` roundoff term on
/// top of the backward-error radius the table encodes; the corrected table's own
/// worst case is `3.8e-14`. It is still four orders tighter than the smallest
/// damaging mis-pairing (`1.4e-8` at the old `(5, 1.44e-1)`).
#[test]
fn g_theta_m_table() {
    for &(m, theta) in TABLE {
        for signed in [theta, -theta] {
            let got = taylor(signed, m);
            let want = libm::exp(signed);
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel <= 1e-12,
                "m={m} theta={theta:.4e} x={signed:+.4e}: \
                 T_m={got:.17e} exp={want:.17e} rel={rel:.3e} > 1e-12"
            );
        }
    }
}

/// Non-vacuity: the gate must reject the radii that actually shipped.
///
/// Without this, a future "simplification" that loosened the tolerance would
/// leave a green gate that no longer catches the original defect.
///
/// **What this gate cannot see.** The old `(4, 3.40e-3)` entry is ten times the
/// correct `θ_4 = 3.397e-4` in *backward*-error terms, but its forward error at
/// that argument is `3.8e-15` — below the corrected table's own worst case. A
/// forward-error gate is structurally blind to it, so it is excluded here rather
/// than papered over with a tolerance chosen to make it fail. That entry is
/// corrected on the strength of the recomputation (ADR-0197), not of this gate.
#[test]
fn g_theta_m_table_rejects_the_shipped_radii() {
    let wrong: &[(u32, f64)] = &[(5, 1.44e-1), (8, 1.44), (10, 2.74), (13, 4.74), (18, 8.84)];
    for &(m, theta) in wrong {
        let rel = (taylor(-theta, m) - libm::exp(-theta)).abs() / libm::exp(-theta);
        assert!(
            rel > 1e-12,
            "the pre-ADR-0197 pair (m={m}, theta={theta}) would pass the gate — \
             the gate has been weakened to the point of not detecting the defect \
             it exists for (measured rel={rel:.3e})"
        );
    }
}
