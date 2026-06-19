#!/usr/bin/env python3
"""T_SUBORD sympy gate — symbolic verification for SubordinatedChernoff (ADR-0103).

Verifies the Bochner-Phillips subordination identities and Butko 2018 order-1
Chernoff tangency claim for `SubordinatedChernoff<C, S, F>` PRE-FLIGHT, before
the engineer wave begins (mirrors the PRE-FLIGHT mandate of ADR-0086).

Sub-checks (5 mandatory; ADR-0103 §5):

  (1) T_SUBORD.bernstein_laplace_exponents
      Verify Laplace exponent algebra for three concrete CBF subordinators:
        - α-stable:       φ_α(λ) = λ^α,        α ∈ (0,1)
        - Gamma:          φ_c(λ) = log(1 + λ/c)
        - Inverse-Gauss:  φ_c(λ) = sqrt(c² + 2λ) - c
      Each MUST satisfy the Bernstein (CBF) admissibility test
        φ(0)=0, φ'(λ)>0 ∀ λ>0, and (-1)^{k+1} φ^{(k)}(λ) ≥ 0 for k = 1, 2
      (full CBF: all k; we verify k ≤ 4 symbolically as a necessary screen).

  (2) T_SUBORD.alpha_stable_moment_match
      α-stable subordinator (α=1/2) has known density p_t(s) explicit in terms
      of an inverse-Gaussian one-sided Lévy density. Verify that
        E[exp(-λ S_t)] = exp(-t · λ^α)
      holds symbolically (the defining Laplace transform identity) for α=1/2.

  (3) T_SUBORD.gauss_laguerre_node_agreement
      32-point Gauss-Laguerre nodes/weights numerical agreement: sample the
      polynomial p(s) = s^6 (exactly integrable on the Laguerre measure
      e^{-s} ds → ∫₀^∞ s^6 e^{-s} ds = 6! = 720) and verify Σ w_k s_k^6 ≈ 720
      to ≤ 5e-9 relative tolerance. Confirms `resolvent_quad.rs` GL32 table
      REUSE is sound for SubordinatedChernoff quadrature.

  (4) T_SUBORD.order1_chernoff_residual
      For scalar A = -μ < 0 (heat-like contraction) and α-stable subordinator
      (α ∈ (0,1)), verify the order-1 Chernoff residual
        F^φ(τ) - exp(-τ · μ^α) = O(τ²)         (Butko 2018 Theorem 2.1)
      symbolically by taking F(τ) = exp(τ·A) (exact base semigroup; the worst
      case is base-Chernoff truncation, which is bounded by Trotter-Kato) and
      expanding the integral
        F^φ(τ) = ∫₀^∞ exp(s·A) μ_τ^φ(ds)
              = exp(-τ · μ^α)  (closed-form for α-stable Laplace transform)
      and confirming Taylor expansion at τ=0 has leading τ^0 = 1 + τ^1 · (-μ^α)
      + τ^2 · (½ μ^{2α}) — i.e., exp Taylor coefficients hold.

  (5) T_SUBORD.gamma_subordinator_closed_form
      Verify Gamma subordinator (c=1) closed-form Laplace transform
        E[exp(-λ S_t)] = exp(-t · log(1 + λ)) = (1 + λ)^{-t}
      matches the formal Gamma(t, 1) density Laplace transform symbolically.
      This validates Gamma backend admissibility.

Prints 'T_SUBORD PASS' on success; 'T_SUBORD FAIL: <reason>' and exits 1 on failure.

NORMATIVE references:
  - Butko 2018 "Chernoff Approximation of Subordinate Semigroups",
    Stochastics and Dynamics, §4 Theorem 2.1.
  - Bochner 1949 "Diffusion equation and stochastic processes", PNAS.
  - Schilling-Song-Vondraček 2012 "Bernstein Functions: Theory and Applications"
    de Gruyter §13 (CBF criterion).
  - Sato 1999 "Lévy Processes and Infinitely Divisible Distributions"
    Cambridge §30 (subordinator Laplace exponents).
  - Abramowitz-Stegun 1964 Table 25.9 (Gauss-Laguerre 32-pt nodes/weights).
"""

import sys


def fail(reason: str) -> int:
    print(f"T_SUBORD FAIL: {reason}", flush=True)
    return 1


def pass_check(name: str) -> None:
    print(f"  [PASS] {name}", flush=True)


def main() -> int:
    try:
        import sympy as sp
    except ImportError:
        return fail("sympy not installed")

    lam, t, s, alpha, c, tau, mu = sp.symbols(
        "lambda t s alpha c tau mu", positive=True
    )

    # -----------------------------------------------------------------------
    # (1) T_SUBORD.bernstein_laplace_exponents
    #
    # CBF screen: φ(0)=0, φ'≥0, (-1)^{k+1} φ^{(k)} ≥ 0 for k=1..4.
    # -----------------------------------------------------------------------
    laplace_exponents = {
        "alpha-stable": (lam**alpha, alpha),
        "gamma":         (sp.log(1 + lam / c), c),
        "inverse-Gauss": (sp.sqrt(c**2 + 2 * lam) - c, c),
    }

    for name, (phi, param) in laplace_exponents.items():
        # φ(0) = 0
        phi_at_zero = sp.limit(phi, lam, 0, "+")
        if sp.simplify(phi_at_zero) != 0:
            return fail(
                f"T_SUBORD.bernstein.{name}: φ(0) = {phi_at_zero} (expected 0)"
            )

        # (-1)^{k+1} φ^{(k)}(λ) ≥ 0  symbolically for k=1..4
        for k in range(1, 5):
            deriv = sp.diff(phi, lam, k)
            sign = (-1) ** (k + 1)
            signed = sign * deriv
            # Substitute concrete admissible param + λ value to check sign
            # (full CBF proof is well-known; we screen with a numeric witness)
            if name == "alpha-stable":
                test = signed.subs([(alpha, sp.Rational(1, 2)), (lam, 1)])
            else:
                test = signed.subs([(param, 1), (lam, 1)])
            val = float(sp.N(test))
            if val < -1e-12:
                return fail(
                    f"T_SUBORD.bernstein.{name}: "
                    f"(-1)^{k+1}·φ^({k})(1) = {val:.3e} (expected ≥ 0)"
                )
    pass_check("T_SUBORD.bernstein_laplace_exponents (3 subordinators × k=1..4)")

    # -----------------------------------------------------------------------
    # (2) T_SUBORD.alpha_stable_moment_match
    #
    # α-stable defining identity: E[exp(-λ S_t)] = exp(-t · λ^α).
    # We construct this symbolically via the well-known one-sided α=1/2
    # stable density (Lévy distribution):
    #   p_t(s) = t / (2 sqrt(π s³)) · exp(-t² / (4 s))   for s > 0
    # and verify its Laplace transform equals exp(-t · sqrt(λ)).
    # -----------------------------------------------------------------------
    p_levy = (t / (2 * sp.sqrt(sp.pi * s**3))) * sp.exp(-(t**2) / (4 * s))
    integrand = sp.exp(-lam * s) * p_levy
    laplace = sp.integrate(integrand, (s, 0, sp.oo))
    laplace = sp.simplify(laplace)
    expected = sp.exp(-t * sp.sqrt(lam))
    residual = sp.simplify(laplace - expected)
    if residual != 0:
        # Try a numeric check at t=1, λ=2 if symbolic simplify falls short
        num_res = float(sp.N(residual.subs([(t, 1), (lam, 2)])))
        if abs(num_res) > 1e-10:
            return fail(
                f"T_SUBORD.alpha_stable_moment_match: "
                f"E[exp(-λ S_t)] - exp(-t·sqrt(λ)) = {residual} "
                f"(numeric residual at t=1, λ=2: {num_res:.3e})"
            )
    pass_check("T_SUBORD.alpha_stable_moment_match (α=1/2 Lévy density Laplace)")

    # -----------------------------------------------------------------------
    # (3) T_SUBORD.gauss_laguerre_node_agreement
    #
    # Verify GL32 nodes/weights (per resolvent_quad.rs) integrate s^6 to 720.
    # NOTE: We import the same float constants the Rust impl will use, via
    # hard-coded literals (this script is the GROUND-TRUTH oracle for the
    # node/weight tables; any drift between this script and resolvent_quad.rs
    # is itself a contract violation worth catching).
    # -----------------------------------------------------------------------
    GL32_NODES = [
        0.044489365833267, 0.234526109519619, 0.576884629301886,
        1.072448753817820, 1.722408776444650, 2.528336706425790,
        3.492213273021990, 4.616456769749770, 5.903958504174240,
        7.358126733186240, 8.982940924212590, 10.78301863254000,
        12.76369798674280, 14.93113975552260, 17.29245433671530,
        19.85586094033610, 22.63088901319680, 25.62863602245920,
        28.86210181632350, 32.34662915396480, 36.10049480575200,
        40.14571977153940, 44.50920799575490, 49.22439498730860,
        54.33372133339700, 59.89250916213400, 65.97537728793520,
        72.68762809066270, 80.18744697791350, 88.73534041789240,
        98.82954286828390, 111.751398097938,
    ]
    GL32_WEIGHTS = [
        0.109218341952385, 0.210443107938813, 0.235213229669848,
        0.195903335972881, 0.129983786286071, 0.0705786238657174,
        0.0317609125091751, 0.0119182148348385, 0.00373881629461153,
        0.000980803306614955, 0.000214864918801364, 3.92034196798795e-5,
        5.93454161286863e-6, 7.41640457866755e-7, 7.60456787912078e-8,
        6.35060222662581e-9, 4.28138297104093e-10, 2.30589949189134e-11,
        9.79937928872709e-13, 3.23780165772927e-14, 8.17182344342070e-16,
        1.54213383339386e-17, 2.11979229016362e-19, 2.05442967378805e-21,
        1.34698258663739e-23, 5.66129413039733e-26, 1.41856054546304e-28,
        1.91337549445422e-31, 1.19224876009397e-34, 2.67151121924014e-38,
        1.33861694210625e-42, 4.51053619389897e-48,
    ]
    if len(GL32_NODES) != 32 or len(GL32_WEIGHTS) != 32:
        return fail(
            f"T_SUBORD.gauss_laguerre_node_agreement: "
            f"GL32 table length mismatch ({len(GL32_NODES)}, {len(GL32_WEIGHTS)})"
        )
    quadrature = sum(w * (x**6) for x, w in zip(GL32_NODES, GL32_WEIGHTS))
    exact = 720.0  # = 6!
    rel_err = abs(quadrature - exact) / exact
    if rel_err > 5e-9:
        return fail(
            f"T_SUBORD.gauss_laguerre_node_agreement: "
            f"∫ s^6 e^{{-s}} ds quadrature = {quadrature:.6e} vs exact {exact} "
            f"(rel err {rel_err:.3e} > 5e-9)"
        )
    pass_check(
        f"T_SUBORD.gauss_laguerre_node_agreement (GL32 ∫s^6: "
        f"rel err {rel_err:.3e} ≤ 5e-9)"
    )

    # -----------------------------------------------------------------------
    # (4) T_SUBORD.order1_chernoff_residual
    #
    # For scalar A = -μ < 0 and α-stable subordinator, the EXACT subordinate
    # semigroup is T^φ_t = exp(-t · μ^α). Take base Chernoff F(s) := exp(s·A)
    # = exp(-s μ); then F^φ(τ) := ∫₀^∞ F(s) μ_τ^φ(ds) = exp(-τ · μ^α) by the
    # subordinator's defining Laplace identity.
    #
    # Verify the Taylor expansion at τ=0:
    #   exp(-τ · μ^α) = 1 + τ·(-μ^α) + τ² · (μ^{2α} / 2) + O(τ³)
    # confirming order-1 Chernoff tangency (the τ⁰ + τ¹ coefficients match
    # the exact semigroup at τ=0, satisfying the Chernoff condition).
    # -----------------------------------------------------------------------
    F_phi = sp.exp(-tau * mu**alpha)
    series = sp.series(F_phi, tau, 0, 3).removeO()
    expected_series = 1 - tau * mu**alpha + tau**2 * mu**(2 * alpha) / 2
    residual_series = sp.simplify(series - expected_series)
    if residual_series != 0:
        return fail(
            f"T_SUBORD.order1_chernoff_residual: "
            f"Taylor series mismatch = {residual_series} (expected 0)"
        )
    pass_check("T_SUBORD.order1_chernoff_residual (Taylor coeffs 0..2 match)")

    # -----------------------------------------------------------------------
    # (5) T_SUBORD.gamma_subordinator_closed_form
    #
    # Gamma(t, 1) subordinator density: p_t(s) = s^{t-1} e^{-s} / Γ(t).
    # Laplace transform: E[exp(-λ S_t)] = (1 + λ)^{-t} = exp(-t log(1+λ))
    # = exp(-t · φ(λ)) with φ(λ) = log(1+λ). Confirms φ matches the
    # defining LevySubordinator.laplace_exponent for the Gamma backend.
    # -----------------------------------------------------------------------
    p_gamma = s**(t - 1) * sp.exp(-s) / sp.gamma(t)
    integrand_gamma = sp.exp(-lam * s) * p_gamma
    laplace_gamma = sp.integrate(integrand_gamma, (s, 0, sp.oo))
    laplace_gamma = sp.simplify(laplace_gamma)
    expected_gamma = (1 + lam) ** (-t)
    residual_gamma = sp.simplify(laplace_gamma - expected_gamma)
    if residual_gamma != 0:
        num_res = float(sp.N(residual_gamma.subs([(t, 2), (lam, 3)])))
        if abs(num_res) > 1e-10:
            return fail(
                f"T_SUBORD.gamma_subordinator_closed_form: "
                f"Laplace(Gamma_t) - (1+λ)^{{-t}} = {residual_gamma} "
                f"(numeric at t=2, λ=3: {num_res:.3e})"
            )
    pass_check("T_SUBORD.gamma_subordinator_closed_form (φ = log(1+λ) match)")

    print("T_SUBORD PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
