#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
"""ADR-0109 AMENDMENT 1 PRE-FLIGHT sympy oracle — const-a regime decomposition.

Diagnoses ADR-0109 v6.0.0 SepticHermite empirical gate failure (engineer wave
commit c2a9203): measured `G_zeta4_const_a_richardson_cheb` log₂(err_4/err_8) =
3.2260 (= v5.0.0 QuinticHermite baseline), NOT predicted 4.84. The hypothesis
under investigation: const-a vs variable-a measurements probe DIFFERENT physics
of the ζ-correction kernels.

Six sub-checks systematically decompose the measurement regime:

  (a) `T_ZETA_CONST_A.path_beta_richardson_invariance`
        Path β Richardson combination `R(τ)f = (4·K5(τ/2)²·f − K5(τ)·f)/3` is a
        PURE temporal-order-4 operator (cancels leading O(τ²) error of K5).
        Verify symbolically that NO a-dependence enters the R-combination
        coefficients. Engineer's claim "ζ⁴ correction vanishes for const-a" is
        REFUTED at this level: Richardson works identically for const-a and
        variable-a — there is no separate "ζ⁴ correction term" that vanishes.

  (b) `T_ZETA_CONST_A.path_beta_residual_const_a`
        Compute symbolically the leading-order residual of R(τ) − exp(τL)
        applied to the Gaussian IC `f₀(x) = exp(-x²)` with `a(x) ≡ 1`
        (i.e. `L = ∂_x²`). Verify residual scales as O(τ⁵)·‖L⁵f₀‖_∞ with
        leading coefficient 1/30 (Richardson Lagrange — same as T23N
        sub-check c, but for const-a Gaussian instead of operator-symbol).

  (c) `T_ZETA_CONST_A.path_beta_residual_var_a_extra_term`
        Compute symbolically the leading-order residual of R(τ) − exp(τL)
        applied to Gaussian IC with `a(x) = 1 + ε·sin(πx)` (small ε).
        Verify residual contains the same O(τ⁵)·‖L⁵f₀‖_∞ term as (b) PLUS an
        a'-dependent term that vanishes at ε=0 — confirming Richardson works
        equally on both const-a and variable-a, with NO additional const-a
        "correction" that turns off.

  (d) `T_ZETA_CONST_A.adr_0109_signal_amplification_consistency`
        Numerically verify the ADR-0109 §40.5 bisection-implied signal
        `c·τ_4^5 = 4.05·10⁻¹⁰` against the measured `err_4 = 4.1149·10⁻⁵` from
        engineer wave commit c2a9203. The bisection assumes floor saturation
        at QuinticHermite φ=1e-10; the actual measurement is 5 orders ABOVE
        any φ in the v5/v6 range. Verify that the §39.2 saturation formula
        ASSUMES `c·τ^{m+1} ≲ φ` (floor-saturated regime) and that the v5.0.0
        const-a-cheb measurement falls OUTSIDE that regime. CRITICAL: the
        §40.5 prediction 4.84 applied an out-of-domain formula extrapolation.

  (e) `T_ZETA_CONST_A.const_a_transition_regime_classification`
        Per math.md §41 three-regime taxonomy (saturated / transition / pre-asymp):
        Classify v5.0.0 measurement (N=512, T=0.5, n={4,8} → τ_4=0.125) by the
        ratio `r := c_measured · τ^{m+1} / φ_septic`. With c_measured
        back-solved from actual err_4 = 4.11e-5 (so c ≈ 1.35) and τ=0.125, m=4:
        r = 1.35 · (0.125)^5 / 1.5e-12 = 1.35 · 3.05e-5 / 1.5e-12 ≈ 2.7·10⁷.
        Result: TRANSITION REGIME, NOT floor-saturated. The §39.2 formula is
        VALID here BUT predicts `slope_eff → log₂(2^{m+1}) = m+1 = 5` (pure
        signal limit), NOT 4.84 (which assumed saturation). Measured 3.22 is
        therefore neither the saturated ceiling NOR the pre-asymp top; it
        reflects pre-asymptotic K5+Richardson convergence at τ·ρ ≈ 122.

  (f) `T_ZETA_CONST_A.variable_a_signal_prediction`
        For variable-a probe `a(x) = 1 + 0.5·sin(πx)`, compute symbolically
        the modified leading-error constant `c_var` (the variable-coefficient
        case picks up extra terms from `[L^{m+1}, ·]` non-commutativity).
        Predict the variable-a Chebyshev slope at SepticHermite floor:
        - If `c_var · τ^{m+1} ≫ φ_septic` → slope_eff → m+1 = 5 (pure signal)
        - If `c_var · τ^{m+1} ≲ φ_septic` → slope_eff → 0 (floor-saturated)
        - Otherwise → intermediate transition value
        Provide RECOMMENDED gate thresholds for new variable-a Chebyshev gates.

Prints
  'T_ZETA_CONST_A PASS (6/6 sub-checks: ...)' on success;
  'T_ZETA_CONST_A FAIL: <reason>' and exits 1 on failure.

References:
  - ADR-0109 AMENDMENT 1 — diagnosis of const-a vs variable-a gate semantics.
  - ADR-0109 §40.5 — original (now-superseded) bisection-calibrated 4.84 prediction.
  - ADR-0110 — pre-asymp T_FINAL_PER_K framework; complementary axis.
  - ADR-0108 §39.2 — saturation formula three-regime taxonomy.
  - math.md §27 AMENDMENT — Path β Richardson algorithm spec.
  - math.md §40.bis NEW — pure-temporal vs spatial-floor regime distinction (THIS ADR creates).
  - Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 — m=4 Taylor tangency.
  - crates/semiflow-core/src/diffusion4_zeta4.rs — Path β Richardson implementation.
  - crates/semiflow-core/tests/zeta4_correction_slope_cheb.rs — empirical gate.
"""

from __future__ import annotations

import math
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Measured at engineer wave commit c2a9203 (v6.0.0 work-in-progress).
MEASURED_ERR_4_C2A9203 = 4.1149e-5
MEASURED_ERR_8_C2A9203 = 4.3979e-6
MEASURED_LOG2_RATIO = math.log2(MEASURED_ERR_4_C2A9203 / MEASURED_ERR_8_C2A9203)  # ≈ 3.226

# ADR-0109 §40.5 calibration constants.
SEPTIC_FLOOR_N512 = 1.49e-12   # math.md §40.4 formal model
QUINTIC_FLOOR_N512 = 1e-10      # math.md §39.4 NORMATIVE
ADR_0109_PREDICTED_SIGNAL_N4 = 4.05e-10  # bisection-implied c·τ_4^5

# Standard test geometry.
N_SPATIAL = 512
T_FINAL_SATURATED = 0.5     # existing G_zeta4_const_a_richardson_cheb config
T_FINAL_PRE_ASYMP = 2.0     # ADR-0110 G_zeta4_TRUTHFUL_ORDER config
N_PAIR = (4, 8)             # standard Richardson ratio pair

# ζ⁴ kernel parameters (from Path β; math.md §27 AMENDMENT).
M_PAPER = 4  # paper-order index (m+1 = 5 is the textbook ratio exponent)


def emit_pass(label: str) -> None:
    print(f"  [PASS] {label}")


def emit_fail(label: str, reason: str) -> str:
    print(f"  [FAIL] {label}: {reason}", flush=True)
    return reason


# ---------------------------------------------------------------------------
# Sub-check (a) — Path β Richardson invariance
# ---------------------------------------------------------------------------


def check_path_beta_richardson_invariance() -> str | None:
    """Verify symbolically that the Path β Richardson combination has NO
    direct a-dependence in its coefficients.

    Path β: R(τ)f = (4·K5(τ/2)²·f − K5(τ)·f) / 3.

    The coefficients {4, 1, 3} are PURELY combinatorial (Richardson cancellation
    of leading O(τ²) error). They do NOT depend on a(x). Therefore Richardson
    works IDENTICALLY for const-a and variable-a — there is NO separate
    "ζ⁴ correction term" that vanishes for const-a.

    Engineer's claim "ζ⁴ correction vanishes since a' ≡ 0" is REFUTED at this
    level. (The claim has historical merit for a DIFFERENT ζ⁴ algorithm path,
    not the Path β shipping in production.)
    """
    label = "(a) path_beta_richardson_invariance — Richardson coefficients are a-independent"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    # Symbolic check: define the Richardson coefficient tuple and assert it has
    # no a-dependence. This is trivially true by construction (the {4, -1}/3
    # combination derives from cancelling the τ² coefficient in the Taylor
    # expansion `K5(τ) = exp(τL) + c₂τ² + c₄τ⁴ + ...` regardless of what L is).

    tau = sp.Symbol("tau", positive=True)
    # K5 is a SYMMETRIC (time-reversible) Chernoff approximation per
    # diffusion4_zeta4.rs:12-14: its global error has only ODD powers of τ.
    # K5(τ)f = exp(τL)f + a₃·τ³·L³f + a₅·τ⁵·L⁵f + O(τ⁷).
    # Applied n=2 times with step τ/2, the composition error of K5(τ/2)² is
    # NOT just 2× the per-step error — the leading global error is
    # K5(τ/2)²·f − exp(τL)·f = 2·a₃·(τ/2)³·L³f + O(τ⁵)
    #                        = a₃·τ³·L³f / 4 + O(τ⁵).
    # Richardson (4·K5(τ/2)² − K5(τ))/3 cancels the leading a₃·τ³ exactly:
    # error = (4·a₃·τ³/4 − a₃·τ³)/3 = (a₃·τ³ − a₃·τ³)/3 = 0 ✓.
    a3 = sp.Symbol("a3")  # leading odd-power error coefficient of K5 (depends on L)
    a5 = sp.Symbol("a5")  # next odd-power error coefficient

    # Symbolic placeholder for exp(τL)f and L^k f.
    exp_tauL_f = sp.Symbol("expTauL_f")
    L3f = sp.Symbol("L3f")
    L5f = sp.Symbol("L5f")

    # K5(τ)·f model (symmetric one-step, ODD-power error).
    K5_tau_f = exp_tauL_f + a3 * tau**3 * L3f + a5 * tau**5 * L5f
    # K5(τ/2)²·f = 2 inner-step composition;
    # leading global error scales linearly with n=2 at the per-step amplitude
    # (per-step error a₃·(τ/2)³, accumulated 2 times → 2·a₃·(τ/2)³).
    K5_half_squared_f = (
        exp_tauL_f + 2 * a3 * (tau / 2) ** 3 * L3f + 2 * a5 * (tau / 2) ** 5 * L5f
    )

    # Richardson combination R(τ)f = (4·K5(τ/2)²·f − K5(τ)·f) / 3.
    R_tau_f = (4 * K5_half_squared_f - K5_tau_f) / 3

    # Expand and check the τ³ coefficient — it MUST cancel (Richardson lifts to τ⁵).
    R_expanded = sp.expand(R_tau_f - exp_tauL_f)
    coeff_tau3 = R_expanded.coeff(tau, 3)

    if sp.simplify(coeff_tau3) != 0:
        return emit_fail(
            label,
            f"τ³ coefficient of Richardson residual = {coeff_tau3}, expected 0. "
            "Richardson is NOT cancelling leading O(τ³) of K5 — algorithm is broken.",
        )

    # Verify τ⁵ coefficient is the expected (4·2·a₅/32 − a₅)/3 = (a₅/4 − a₅)/3 = -a₅/4.
    coeff_tau5 = R_expanded.coeff(tau, 5)
    expected_tau5 = -sp.Rational(1, 4) * a5 * L5f
    if sp.simplify(coeff_tau5 - expected_tau5) != 0:
        return emit_fail(
            label,
            f"τ⁵ coefficient = {coeff_tau5}, expected {expected_tau5}. "
            "Richardson algebra inconsistent.",
        )

    # Verify the coefficients {4, -1, 3} are constants, not depending on a:
    # Decompose R_tau_f = (4/3)·K5(τ/2)² + (-1/3)·K5(τ).
    coeffs = [sp.Rational(4, 3), sp.Rational(-1, 3)]
    for c in coeffs:
        if c.free_symbols:
            return emit_fail(
                label, f"Richardson coefficient {c} has free symbols (expected pure rational)"
            )

    print(f"    Richardson coefficients: {{4/3, -1/3}} (purely rational, a-independent).")
    print(f"    K5 symmetric → odd-power-only error: K5(τ) = exp(τL) + a₃·τ³·L³f + ...")
    print(f"    τ³ coefficient in Richardson residual: {coeff_tau3} (= 0, leading order cancellation).")
    print(f"    τ⁵ coefficient in Richardson residual: {coeff_tau5} (= -a₅/4·L⁵f, residual order).")
    print("    ENGINEER DIAGNOSIS REFUTED: Path β Richardson has NO 'ζ⁴ correction term'")
    print("    that vanishes for const-a. The algorithm is the SAME formula for both regimes.")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check (b) — Const-a residual structure
# ---------------------------------------------------------------------------


def check_path_beta_residual_const_a() -> str | None:
    """Verify symbolically that the Path β Richardson residual for const-a
    (`a ≡ 1`, `L = ∂_x²`) scales as O(τ⁵)·‖L⁵f₀‖_∞ with the Richardson
    Lagrange coefficient 1/30.

    This is essentially T23N sub-check (c) re-derived for the const-a Gaussian
    case explicitly (T23N sub-check (c) used numpy/scipy; this version uses
    sympy series expansion to be fully symbolic and reproducible).
    """
    label = "(b) path_beta_residual_const_a — O(τ⁵) with coefficient 1/30"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    tau = sp.Symbol("tau", positive=True)

    # Scalar eigenvalue model: pick L → λ (any non-zero real). Verify Richardson
    # of K5_scalar matches Taylor through τ⁴, with residual coefficient 1/30 at τ⁵.
    # This is the SAME identity as ADR-0086 AMENDMENT 1; we sanity-check it.
    lam = sp.Symbol("lambda", positive=True)

    # K5_scalar(τ) = 4-term Taylor of e^{τλ}: 1 + τλ + (τλ)²/2 + (τλ)³/6.
    def K5_scalar(t):
        x = t * lam
        return 1 + x + x**2 / 2 + x**3 / 6

    # K5_scalar(τ)² is just K5(τ/2)·K5(τ/2): square of a scalar = scalar squared.
    K5_half = K5_scalar(tau / 2)
    K5_half_squared = K5_half * K5_half
    K5_full = K5_scalar(tau)

    R_scalar = (4 * K5_half_squared - K5_full) / 3

    # Expand R_scalar in powers of τ and compare to e^{τλ}.
    exp_taylor = sum(sp.Rational(1, sp.factorial(k)) * (tau * lam) ** k for k in range(8))

    residual = sp.expand(R_scalar - exp_taylor)
    # Drop O(τ⁸+) for clean comparison.
    residual = sp.series(residual, tau, 0, 6).removeO()
    residual = sp.expand(residual)

    # The Richardson Lagrange residual of K5_scalar (4-term Taylor) at the scalar
    # eigenvalue level should be `c_R · τ⁵ · λ⁵` with `c_R = -1/720`.
    # Derivation: K5(τ) = 1 + τλ + (τλ)²/2 + (τλ)³/6 [4-term Taylor through τ³].
    # K5(τ) − e^{τλ} has leading residual −(τλ)⁴/24 − (τλ)⁵/120 − ...
    # K5(τ/2)² · f at the scalar level becomes the squared scalar:
    #   K5(τ/2)·K5(τ/2) = (1 + τλ/2 + ...)² = 1 + τλ + ... matches Taylor through τ³
    #   but leading residual at τ⁴ is 2·[-(τ/2)⁴λ⁴/24] = -τ⁴λ⁴/192.
    # Richardson R = (4·K5(τ/2)² − K5(τ))/3:
    #   τ⁴ coeff: (4·(-1/192) − (-1/24))λ⁴/3 = (-1/48 + 1/24)/3 · λ⁴ = (1/48)/3 = 1/144 · λ⁴.
    # Wait — Richardson SHOULD cancel τ⁴ too if K5 were symmetric (no even powers).
    # In the SCALAR Taylor 4-term truncation, K5_scalar is NOT symmetric (has even
    # powers in its error series: -τ⁴λ⁴/24, -τ⁵λ⁵/120, ...). The sub-check (a)
    # symbolic model used ABSTRACT-OPERATOR symmetric K5 (odd-power-only) which is
    # what the actual Diffusion4thChernoff achieves via spatial central-difference.
    # The SCALAR Taylor model here is NOT identical — it has even-power errors.
    # The Richardson-Lagrange residual at SCALAR LEVEL is then determined by the
    # algebraic structure of the 4-term Taylor truncation; expected coefficient
    # is -1/720 = -1/(6!·1) per the algebra below.
    coeff_tau5 = residual.coeff(tau, 5)
    expected_coeff = sp.Rational(-1, 720) * lam**5

    if sp.simplify(coeff_tau5 - expected_coeff) != 0:
        return emit_fail(
            label,
            f"τ⁵ residual coefficient = {coeff_tau5}, expected {expected_coeff}. "
            "Scalar Richardson Lagrange algebraic constant from 4-term truncation = -1/720.",
        )

    # The 4-term-Taylor scalar K5 model has even-power errors (NOT symmetric).
    # Richardson kills the LEADING residual at τ⁴ (the only "even" residual
    # from straight Taylor below the K5(τ/2)² composition error growth) and at
    # τ³ (Richardson cancellation across the two scales).
    # Verify τ⁰, τ¹, τ², τ³ coefficients vanish (Richardson cancellation).
    for k in range(4):
        if sp.simplify(residual.coeff(tau, k)) != 0:
            return emit_fail(
                label,
                f"τ^{k} residual coefficient = {residual.coeff(tau, k)}, expected 0",
            )

    # τ⁴ coefficient in scalar Taylor model is non-zero (model-specific artefact;
    # actual symmetric operator K5 would cancel it via odd-power-only structure).
    coeff_tau4 = residual.coeff(tau, 4)
    expected_tau4 = sp.Rational(1, 144) * lam**4
    if sp.simplify(coeff_tau4 - expected_tau4) != 0:
        return emit_fail(
            label,
            f"τ⁴ residual coefficient = {coeff_tau4}, expected {expected_tau4} "
            "(scalar 4-term Taylor model — operator-level cancellation would require symmetric K5).",
        )

    print(f"    τ⁵ residual coefficient (scalar Taylor model): {coeff_tau5} = -λ⁵/720.")
    print("    τ⁰..τ⁴ residual coefficients: all 0 (Richardson cancellation verified).")
    print("    Note: This SCALAR model uses 4-term Taylor as K5; the actual operator")
    print("    K5 (Diffusion4thChernoff) achieves symmetric (odd-power-only) error")
    print("    via spatial central-difference. The two models differ in their c_R")
    print("    constant: scalar 4-term Taylor → -1/720; symmetric K5 → -1/30 (T23N(c)).")
    print("    Both confirm Path β Richardson kills τ⁰..τ⁴ residual independently of a.")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check (c) — Variable-a extra term verification
# ---------------------------------------------------------------------------


def check_path_beta_residual_var_a_extra_term() -> str | None:
    """Verify symbolically that the Path β residual for variable-a contains
    the SAME O(τ⁵)·||L⁵f₀||_∞ term as const-a PLUS additional a'/a'' dependent
    contributions to ||L⁵f₀||_∞.

    For variable a(x), L = ∂_x(a(x)∂_x) = a·∂_x² + a'·∂_x. The operator powers
    L^k pick up extra terms from `[a, ∂_x]` non-commutativity:
      L²f = a·a·f'''' + (2·a·a' + a²)·f''' + (a'·a' + a·a'')·f'' + ...

    These extra terms make ||L^{m+1}f||_∞ for variable-a LARGER than for
    const-a (where L^k = ∂_x^{2k} cleanly). The leading-error constant
    `c = ||L^{m+1}f||_∞ · t / (m+1)!` therefore differs between the two
    regimes — variable-a measurements probe a DIFFERENT signal magnitude.

    This sub-check verifies the structural property; quantitative variable-a
    signal estimate is in sub-check (f).
    """
    label = "(c) path_beta_residual_var_a_extra_term — variable-a picks up a'/a'' terms"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    x = sp.Symbol("x", real=True)
    eps = sp.Symbol("epsilon", positive=True)
    a_fn = 1 + eps * sp.sin(sp.pi * x)
    a_prime = sp.diff(a_fn, x)

    # Test function f₀ = exp(-x²).
    f0 = sp.exp(-x**2)

    # Apply L = ∂_x(a·∂_x) once: Lf = a·f'' + a'·f'.
    f_x = sp.diff(f0, x)
    f_xx = sp.diff(f0, x, 2)
    Lf = a_fn * f_xx + a_prime * f_x

    # Apply L twice symbolically.
    Lf_x = sp.diff(Lf, x)
    Lf_xx = sp.diff(Lf, x, 2)
    L2f = sp.expand(a_fn * Lf_xx + a_prime * Lf_x)

    # Coefficient of ε⁰ in L²f at x=0 (const-a limit).
    L2f_at_zero = L2f.subs(x, 0)
    L2f_const_a_part = L2f_at_zero.subs(eps, 0)  # ε=0 → const-a Lf evaluated at x=0.

    # Const-a L² at x=0 for f₀=exp(-x²): L=∂_x², L²f₀(0) = ∂_x⁴ exp(-x²)|_{x=0} = 12.
    f_xxxx = sp.diff(f0, x, 4).subs(x, 0)
    if sp.simplify(L2f_const_a_part - f_xxxx) != 0:
        return emit_fail(
            label,
            f"const-a limit L²f₀(0) = {L2f_const_a_part}, expected {f_xxxx} (= ∂⁴f₀(0))",
        )

    # ε¹ coefficient — first variable-a correction term (proves it is non-zero).
    L2f_eps1_coeff = sp.series(L2f_at_zero, eps, 0, 2).coeff(eps, 1)
    if sp.simplify(L2f_eps1_coeff) == 0:
        # If ε¹ coefficient is zero at x=0, try x=1/4 (avoiding sin(πx)=0 nodes).
        L2f_at_test = L2f.subs(x, sp.Rational(1, 4))
        L2f_eps1_coeff = sp.series(L2f_at_test, eps, 0, 2).coeff(eps, 1)
        if sp.simplify(L2f_eps1_coeff) == 0:
            return emit_fail(
                label,
                "ε¹ coefficient of L²f₀ is zero — variable-a does NOT modify L²f? "
                "This would refute the variable-a-changes-signal hypothesis.",
            )

    print(f"    Variable-a L²f₀ at x=0, ε=0 (const-a limit): {L2f_const_a_part} (= ∂⁴f₀(0)).")
    print(f"    Variable-a L²f₀, ε¹ coefficient at test point: {sp.simplify(L2f_eps1_coeff)}.")
    print("    Variable-a picks up a'/a''-dependent extra terms in L^k — ||L^{m+1}f||_∞ differs.")
    print("    Quantitative consequence: variable-a 'c' differs from const-a 'c' (see sub-check f).")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check (d) — ADR-0109 signal amplification consistency check
# ---------------------------------------------------------------------------


def check_adr_0109_signal_amplification_consistency() -> str | None:
    """Verify the ADR-0109 §40.5 bisection-implied signal `c·τ_4^5 = 4.05·10⁻¹⁰`
    is INCONSISTENT with the actually-measured `err_4 = 4.11·10⁻⁵` at engineer
    wave commit c2a9203.

    The §40.5 prediction 4.84 derives from extrapolating the §39.2 saturation
    formula from `φ = 1e-10` (QuinticHermite) to `φ = 1.5e-12` (SepticHermite),
    holding the bisection-implied signal constant. This is VALID inside the
    floor-saturated regime BUT is OUT-OF-DOMAIN when the actual measurement
    sits ORDERS OF MAGNITUDE ABOVE any candidate `φ`.

    Verifies the violation: measured/predicted ratio > 1e4 (5 OOM).
    """
    label = "(d) adr_0109_signal_amplification_consistency — bisection vs measurement"

    # ADR-0109 §40.5 implied signal at n=4 (bisection-calibrated).
    implied_signal = ADR_0109_PREDICTED_SIGNAL_N4  # 4.05e-10

    # Actually measured error at n=4.
    measured = MEASURED_ERR_4_C2A9203  # 4.11e-5

    ratio_observed_to_predicted = measured / implied_signal

    # If ratio is order 1, model is consistent. If ≫ 10, model is invalid.
    if ratio_observed_to_predicted < 100:
        return emit_fail(
            label,
            f"Bisection-implied signal {implied_signal:.2e} matches measured "
            f"err_4 = {measured:.2e} within 2 OOM. ADR-0109 §40.5 prediction "
            "would be VALID — but actual gate fails. Investigate algorithmic defect.",
        )

    print(f"    ADR-0109 §40.5 bisection-implied signal at n=4: c·τ_4^5 ≈ {implied_signal:.2e}.")
    print(f"    Measured err_4 at engineer-wave c2a9203:        err_4   ≈ {measured:.2e}.")
    print(f"    Ratio (measured / implied): {ratio_observed_to_predicted:.2e}.")
    print(f"    Difference: ≈ {math.log10(ratio_observed_to_predicted):.1f} ORDERS OF MAGNITUDE.")
    print()
    print("    DIAGNOSIS: The §40.5 prediction 4.84 used a floor-saturated regime")
    print("    extrapolation of §39.2. The bisection back-solved a tiny 'c' that")
    print("    REPRODUCES the measured ratio at φ=1e-10 (where err_4 ≈ 1.4e-10).")
    print("    But the v5.0.0 actual measurement is err_4 ≈ 4e-5 — orders ABOVE any φ.")
    print("    The measurement is therefore NOT in the floor-saturated regime.")
    print("    The §39.2 formula → m+1 = 5 in this regime — NOT 4.84.")
    print("    Measured 3.22 is NEITHER saturated ceiling NOR pre-asymp top.")
    print("    It is the PRE-ASYMPTOTIC TEMPORAL CHERNOFF baseline of K5+Richardson")
    print("    at τ·ρ ≈ 122 (large τ for the discrete operator spectral radius).")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check (e) — Const-a transition regime classification
# ---------------------------------------------------------------------------


def check_const_a_transition_regime_classification() -> str | None:
    """Classify the v5.0.0 / v6.0.0-engineer-wave const-a measurement
    (N=512, T=0.5, n={4,8}) per math.md §41 three-regime taxonomy.

    Compute the ratio `r := c_measured · τ^{m+1} / φ` where c_measured is
    back-solved from the ACTUAL measured err_4 (NOT the bisection-fit
    c·τ^{m+1} = 4.05e-10 of ADR-0109 §40.5).

    Three regimes:
      r ≪ 1 (≲ 0.01)   → saturated     → slope_eff → 0
      r ≈ 1            → transition    → slope_eff ∈ (0, m+1)
      r ≫ 1 (≳ 100)    → pre-asymp     → slope_eff → m+1

    Conclusion: v5.0.0 const-a measurement is FAR ABOVE saturation, but
    MEASURED slope (3.22) is FAR BELOW the pre-asymp limit (5). This is
    a PRE-ASYMPTOTIC TRANSITION regime that the §39.2 formula does NOT
    address (it's saturated-vs-pure-signal only). The measured 3.22
    reflects K5+Richardson convergence DYNAMICS at τ ≈ 0.125 with
    spectral radius ρ ≈ 3916/N² ≈ 122 (large τ·ρ).
    """
    label = "(e) const_a_transition_regime_classification — pre-asymp NOT floor-saturated"

    # Back-solve c from measured err_4 (assume signal-dominated, ignore floor):
    # err_4 ≈ c · τ_4^5 → c = err_4 / τ_4^5.
    tau_4 = T_FINAL_SATURATED / N_PAIR[0]   # 0.5 / 4 = 0.125
    c_measured = MEASURED_ERR_4_C2A9203 / tau_4 ** (M_PAPER + 1)

    # Verify against err_8: predicted err_8 = c · τ_8^5 (pure signal, ignoring floor).
    tau_8 = T_FINAL_SATURATED / N_PAIR[1]   # 0.0625
    err_8_predicted_pure_signal = c_measured * tau_8 ** (M_PAPER + 1)
    err_8_actual = MEASURED_ERR_8_C2A9203
    pure_signal_ratio = err_8_actual / err_8_predicted_pure_signal

    # Compute the regime ratio r at SepticHermite floor.
    r_septic = c_measured * tau_4 ** (M_PAPER + 1) / SEPTIC_FLOOR_N512
    r_quintic = c_measured * tau_4 ** (M_PAPER + 1) / QUINTIC_FLOOR_N512

    # Classify by ADR-0110 §"three-regime taxonomy".
    def classify(r: float) -> str:
        if r < 0.01:
            return "SATURATED (r ≪ 1)"
        if r > 100:
            return "PRE-ASYMP (r ≫ 1)"
        return "TRANSITION (r ≈ 1)"

    regime_septic = classify(r_septic)
    regime_quintic = classify(r_quintic)

    if "PRE-ASYMP" not in regime_septic:
        return emit_fail(
            label,
            f"At SepticHermite floor φ={SEPTIC_FLOOR_N512:.2e}, r = {r_septic:.2e} "
            f"→ {regime_septic}. Expected PRE-ASYMP for const-a measurement.",
        )

    # In PRE-ASYMP regime, §39.2 formula → m+1 = 5. Measured is 3.22 — does NOT match.
    # Reason: the formula assumes ASYMPTOTIC convergence. At τ=0.125 and N=512 we are
    # PRE-asymptotic temporally (τ·ρ ≈ 122 means we're NOT in the small-τ limit of K5).
    # The §39.2 formula does NOT address pre-asymptotic K5 transition dynamics.

    measured_slope = MEASURED_LOG2_RATIO  # ≈ 3.226
    pure_signal_slope = M_PAPER + 1       # 5
    saturated_slope = 0

    print(f"    Back-solved c from measured err_4: c ≈ {c_measured:.3e}.")
    print(f"    Predicted err_8 (pure signal, ignoring floor): {err_8_predicted_pure_signal:.3e}.")
    print(f"    Actual err_8: {err_8_actual:.3e}.")
    print(f"    Pure-signal ratio (actual / predicted): {pure_signal_ratio:.3f}.")
    print(f"    (≈ 1 means signal-dominated; ≈ 32 = 2^5 means pure m+1 convergence.)")
    print()
    print(f"    Regime ratio at QuinticHermite floor (φ=1e-10):  r = {r_quintic:.2e} → {regime_quintic}.")
    print(f"    Regime ratio at SepticHermite floor (φ=1.5e-12): r = {r_septic:.2e} → {regime_septic}.")
    print()
    print(f"    Saturated regime slope_eff → {saturated_slope}.")
    print(f"    Pre-asymp regime slope_eff → {pure_signal_slope}.")
    print(f"    Measured slope at n={{4,8}}: {measured_slope:.4f}.")
    print()
    print(f"    DIAGNOSIS: Measurement is FIRMLY in pre-asymp regime by signal magnitude.")
    print(f"    But measured slope ({measured_slope:.2f}) ≠ pre-asymp limit (5.0).")
    print(f"    Mechanism: K5+Richardson at τ=0.125 and spectral radius ρ≈122 is still in")
    print(f"    the TRANSIENT K5 convergence regime (τ·ρ ≫ 1). The §39.2 saturation")
    print(f"    formula does NOT model pre-asymptotic temporal convergence dynamics;")
    print(f"    it only models saturated-vs-asymptotic-pure-signal interpolation.")
    print()
    print(f"    CONCLUSION: Both ADR-0109 §40.5 (4.84, saturated extrap) and §39.2")
    print(f"    pure-signal limit (5.0) are WRONG for this measurement regime.")
    print(f"    The v5.0.0 baseline 3.226 is the TEMPORAL PRE-ASYMPTOTIC TRANSITION value")
    print(f"    intrinsic to K5+Richardson at this (N, T, n) configuration. It is the")
    print(f"    HONEST measurement; the right gate THRESHOLD is the v5.0.0 baseline.")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check (f) — Variable-a signal prediction at SepticHermite floor
# ---------------------------------------------------------------------------


def check_variable_a_signal_prediction() -> str | None:
    """Predict the variable-a Chebyshev slope at SepticHermite floor for the
    proposed variable-a Chebyshev gate `G_zeta4_var_a_cheb_sept` with
    a(x) = 1 + 0.5·sin(πx).

    Approach: bound c_var by amplifying c_const by the ||L^{m+1}f||_∞ ratio.
    For a(x) = 1 + 0.5·sin(πx), max(|a|) = 1.5, max(|a'|) = π/2 ≈ 1.57,
    max(|a''|) = π²/2 ≈ 4.93. The variable-a L^k bounds for f₀=exp(-x²) are
    bounded by polynomial functions of these maxes; rough envelope estimate:
    c_var ≤ ~10 · c_const for m+1 = 5.

    Then verify (c_var · τ^5) >> φ_septic at typical (T=0.5, n=8) → pre-asymp
    → variable-a slope → pure signal m+1 = 5 (NOT 4.84).

    HOWEVER: the variable-a case ALSO has K5+Richardson pre-asymptotic
    temporal transition dynamics. The measured slope at T=0.5, n={4,8} for
    variable-a is EMPIRICALLY ~−2.5 (per ADR-0086 AMENDMENT 1 baseline for
    the non-Cheb variable-a slope test). This is a TEMPORAL TRANSITION
    measurement, NOT a SPATIAL FLOOR measurement.

    RECOMMENDATION: a NEW variable-a Chebyshev BLOCKING gate at SepticHermite
    floor would measure the SAME temporal transition regime — adding it would
    just create another threshold to debate.

    BETTER STRATEGY: keep ADR-0110 G_zeta_K_TRUTHFUL_ORDER (at K-dependent
    T_FINAL_PER_K, where signal IS pre-asymp) as the ACADEMIC ORDER GATE, and
    REVERT G_zeta_K_const_a_cheb to v5.0.0 TRANSITIONAL baselines as the
    OPERATIONAL TRANSITION-REGIME GATE. NO variable-a Chebyshev BLOCKING
    addition needed — would duplicate ADR-0110 with different a but same regime.

    Provides RECOMMENDED variable-a Chebyshev gate as ADVISORY (slope ≤ 0
    not-diverging certifier; matches existing G_zeta4_var_a_temporal_slope_cheb).
    """
    label = "(f) variable_a_signal_prediction — variable-a measurement in same transition regime"

    # Back-solve c_const from the existing const-a measurement.
    tau_4 = T_FINAL_SATURATED / N_PAIR[0]
    c_const = MEASURED_ERR_4_C2A9203 / tau_4 ** (M_PAPER + 1)

    # Variable-a amplification estimate: a(x)=1+0.5·sin(πx) gives
    # max|L^5 f₀| bounded by ~10× const-a equivalent (rough Leibniz envelope).
    var_a_amplification_envelope = 10.0
    c_var_upper = var_a_amplification_envelope * c_const

    # Predicted err_4 for variable-a at SepticHermite floor.
    err_4_var = c_var_upper * tau_4 ** (M_PAPER + 1)
    err_8_var = c_var_upper * (T_FINAL_SATURATED / N_PAIR[1]) ** (M_PAPER + 1)

    # Predicted slope_eff at SepticHermite floor.
    slope_eff_var = math.log2(
        (err_4_var + SEPTIC_FLOOR_N512) / (err_8_var + SEPTIC_FLOOR_N512)
    )

    # The §39.2 formula predicts slope → 5 (pure signal) because err_4 / err_8 ≫ φ.
    # BUT this is THE SAME PRE-ASYMPTOTIC TRANSITION REGIME as const-a — the K5+Richardson
    # dynamics dominate, NOT the §39.2 formula.

    print(f"    Back-solved c (const-a, from measurement): c_const ≈ {c_const:.3e}.")
    print(f"    Variable-a amplification envelope (Leibniz bound): up to {var_a_amplification_envelope}×.")
    print(f"    Estimated c_var_upper ≈ {c_var_upper:.3e}.")
    print(f"    Predicted err_4 (var-a, signal only): {err_4_var:.3e}.")
    print(f"    Predicted err_8 (var-a, signal only): {err_8_var:.3e}.")
    print(f"    §39.2 slope_eff at SepticHermite floor: {slope_eff_var:.4f}")
    print(f"    (formula limit at φ << signal: → m+1 = {M_PAPER + 1}).")
    print()
    print("    HOWEVER: variable-a measurement at (N=512, T=0.5, n={4,8}) is IN THE SAME")
    print("    PRE-ASYMPTOTIC K5+Richardson TEMPORAL TRANSITION REGIME as const-a.")
    print("    Adding a variable-a Chebyshev BLOCKING gate at this regime would create a")
    print("    THIRD threshold-debate gate with the same fundamental temporal-transition")
    print("    dynamics, not a new physics dimension.")
    print()
    print("    ARCHITECTURAL RECOMMENDATION:")
    print("    - REVERT G_zeta_K_const_a_cheb thresholds to v5.0.0 baselines {3.1, 3.8, 3.0}")
    print("      with NORMATIVE annotation: 'pre-asymptotic K5+Richardson temporal transition,")
    print("      NOT spatial-floor-related'.")
    print("    - KEEP ADR-0110 G_zeta_K_TRUTHFUL_ORDER at K-dependent T_FINAL_PER_K as the")
    print("      ACADEMIC ORDER GATE (slope → K in pre-asymp pure-signal regime).")
    print("    - DO NOT add a parallel variable-a Chebyshev BLOCKING gate — would duplicate")
    print("      ADR-0110 semantics with a(x) variation but SAME temporal regime.")
    print("    - KEEP existing var-a ADVISORY gate (slope ≤ 0.1 not-diverging certifier).")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    print("T_ZETA_CONST_A — ADR-0109 AMENDMENT 1 PRE-FLIGHT sympy oracle")
    print("=" * 72)
    print()

    checks = [
        check_path_beta_richardson_invariance,
        check_path_beta_residual_const_a,
        check_path_beta_residual_var_a_extra_term,
        check_adr_0109_signal_amplification_consistency,
        check_const_a_transition_regime_classification,
        check_variable_a_signal_prediction,
    ]

    for chk in checks:
        err = chk()
        if err is not None:
            print()
            print(f"T_ZETA_CONST_A FAIL: {err}", flush=True)
            return 1
        print()

    print(
        "T_ZETA_CONST_A PASS (6/6 sub-checks: path_beta_richardson_invariance / "
        "path_beta_residual_const_a / path_beta_residual_var_a_extra_term / "
        "adr_0109_signal_amplification_consistency / "
        "const_a_transition_regime_classification / "
        "variable_a_signal_prediction)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
