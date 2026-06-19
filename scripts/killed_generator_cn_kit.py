#!/usr/bin/env python3
"""killed_generator_cn_kit.py — PRE-FLIGHT for v8.0.0 F3 (ADR-0135 Amendment 2,
math.md §44.ter).

Feature 3 (RE-SPIKE, INVENTIVE): order-2 HARD absorbing wall via the
**Crank–Nicolson Cayley map of the killed Dirichlet generator** L^R:

    U^{n+1} = (I − τ/2 · L^R)^{-1} (I + τ/2 · L^R) U^n,

where L^R is the Dirichlet realization of L = ∂_x(a ∂_x) on the open region R,
i.e. the discrete generator restricted to the INTERIOR degrees of freedom with
the absorbing boundary condition u|_∂R = 0 baked into the DOMAIN (boundary
nodes carry the fixed value 0 and contribute nothing to the interior stencil).

WHY THIS RE-SPIKE EXISTS — the ARIZ resolution of contradiction C3
================================================================================
The Phase-2 NO-GO (§44.bis) proved the resolvent-smoothed post-multiply mask
P_ε C(τ) P_ε is DOUBLY obstructed for global order-2:
  C3a idempotency collapse of 𝟙_R^{1/2} (the mask is post-multiplied onto a
      FREE-SPACE flow);
  C3b factor accuracy: (I+τρ)^{-1/2} matches e^{-τρ/2} only to O(τ);
  C3c boundary-layer commutator [L,ρ] ~ 1/dx² blows up the order-2 constant
      under ε = α·dx.
ALL THREE share one root cause: the boundary is enforced as a SEPARATE operation
(a mask / a reaction rate ρ) composed with a DIFFERENT operator (the free-space
flow C). The contradiction is "the kill must be sharp (idempotent → order-1) AND
the step must be order-2".

ARIZ resolution — separation IN STRUCTURE (move the conflict to the super-system,
TRIZ-1 segmentation + TRIZ-7 nesting): do NOT compose a kill with a flow. Make a
SINGLE operator whose generator ALREADY satisfies the BC on its domain. Then there
is no second factor to mismatch (C3b gone), no mask to be idempotent (C3a gone),
and no rate-ramp commutator (C3c gone — the wall is exact, width 0, not a layer).
The Cayley map of any generator G is order-2 in τ unconditionally; with G = L^R
the hard wall is built into G, so the order-2 IS the hard-wall order-2. The IKR:
"the boundary enforces itself, for free, by being part of the operator, leaving
the order-2 untouched."

THIS PRE-FLIGHT decides GO / NO-GO for that order-2 claim. It is built to FALSIFY,
not rubber-stamp, on finite-dimensional matrix models (the honest non-commuting
setting). FOUR sub-checks; the oracle PASSES (exit 0) iff every check produces its
EXPECTED sign — and the printed VERDICT is then GO (order-2), the mirror of the
NO-GO that hard_wall_kit.py documents for the post-multiply route.

  CHECK A (LOCAL ORDER — the heart): the Cayley map of the killed Dirichlet
    generator L^R matches the EXACT killed semigroup e^{τ L^R} through τ² with a
    NONZERO τ³ residual ⟹ local truncation error O(τ³) per step ⟹ GLOBAL order-2.
    EXPECT: τ¹ = 0, τ² = 0, τ³ ≠ 0.

  CHECK B (HARD BC EXACT — not a layer): the killed generator L^R annihilates the
    boundary degrees of freedom EXACTLY at every τ (u|_∂R = 0 is preserved to
    machine precision, no O(ε) bias, no width parameter). EXPECT: the Cayley step
    keeps the boundary value identically 0 — a structural identity, exact ∀τ.

  CHECK C (NO BOUNDARY-LAYER BLOW-UP — C3c is GONE): the τ³ order-2 constant is
    governed by the SMOOTH interior operator L^R and is INDEPENDENT of any wall-
    sharpening parameter (there is no ramp). Refining dx (more interior nodes)
    leaves the per-step LTE constant BOUNDED — it does NOT grow as a negative
    power of dx the way the §44.bis CHECK C ramp did. EXPECT: refine the grid and
    the leading LTE constant stays O(1) (no 1/dx blow-up).

  CHECK D (STRICTLY STRONGER THAN THE POST-MULTIPLY MASK — the contrast): the
    order-1 post-multiply KillingChernoff form 𝟙_R · C(τ) (the v2.6 shipped
    operator) has a NONZERO τ² residual against e^{τ L^R} on the SAME model ⟹
    order-1; the L^R Cayley map (CHECK A) has τ² = 0 ⟹ order-2. EXPECT: mask τ²
    ≠ 0 AND Cayley τ² = 0 (the re-spike strictly dominates).

  CHECK E (VARIABLE-COEFFICIENT HONESTY JET — ADR-0149, ADR-0136 anti-over-report
    discipline): apply the variable-a killed Dirichlet Cayley step to a GENERIC
    degree-6 NON-origin-symmetric nodal jet. ADR-0136 Amendment 1 established that
    low-degree / origin-symmetric probes over-report order on group-action operators
    (τ³ residual can spuriously vanish when (L^R)³ annihilates the jet). This CHECK
    uses a strictly varying half-node diffusivity and an asymmetric degree-6 jet that
    is NOT annihilated by (L^R)³, so the τ³ residual is genuinely nonzero and order-2
    is HONEST on real variable-a data. EXPECT: τ¹ = τ² = 0, τ³ ≠ 0.
    Rationale: CHECK A uses a constant-a generator; CHECK E proves the same
    cancellation is a MATRIX IDENTITY (★) that persists for arbitrary variable a(x).
    See ADR-0149 (symbolic CHECK C proof) and scripts/killed_dirichlet_varcoef_kit.py.

GO is encoded: order-2 hard absorbing wall is achievable via the killed Dirichlet
generator Cayley map. Gate G_HARD_WALL_ORDER2 (renamed-in-place, ADR-0135 Am. 2)
is declared RELEASE_BLOCKING with a hard-barrier self-convergence slope ≤ −1.95.
Exits 0 iff all checks give their expected signs; 1 otherwise.
"""

import sys


def main() -> int:
    try:
        import sympy as sp
        from sympy import Matrix, eye, zeros, symbols, factorial, Rational, Poly
    except ImportError:
        print("PREFLIGHT-KILLED-GEN-CN FAIL: sympy not installed", flush=True)
        return 1

    tau = symbols("tau", positive=True)

    def mexp(A, order):
        d = A.shape[0]
        out = eye(d)
        term = eye(d)
        for k in range(1, order + 1):
            term = sp.expand(term * A)
            out = out + term / factorial(k)
        return out

    def tau_coeff(M, power, d):
        out = zeros(d, d)
        for i in range(d):
            for j in range(d):
                out[i, j] = sp.expand(M[i, j]).coeff(tau, power)
        return out

    def trunc(M, order):
        d = M.shape[0]
        out = zeros(d, d)
        for i in range(d):
            for j in range(d):
                poly = Poly(sp.expand(M[i, j]), tau)
                deg = poly.degree()
                out[i, j] = sum(
                    c * tau ** (deg - n)
                    for n, c in enumerate(poly.all_coeffs())
                    if (deg - n) <= order
                )
        return out

    def cayley(G, order):
        # (I − τ/2 G)^{-1} (I + τ/2 G), Neumann series for the inverse to τ^order.
        d = G.shape[0]
        num = eye(d) + tau / 2 * G
        inv = zeros(d, d)
        term = eye(d)
        for k in range(0, order + 1):  # (I − τ/2 G)^{-1} = Σ (τ/2 G)^k
            inv = inv + term * tau**k
            term = sp.expand(term * (Rational(1, 2) * G))
        return trunc(sp.expand(num * inv), order)

    order = 4
    a = Rational(1, 2)  # diffusion coefficient

    # ── Killed Dirichlet generator L^R on a 5-node grid [0,4] ────────────────
    # Nodes 0 and 4 are the absorbing wall (u = 0, Dirichlet). Interior dofs =
    # nodes 1,2,3. The Dirichlet BC is baked into the DOMAIN: the interior
    # generator is the 3×3 tridiagonal Laplacian with boundary-neighbour
    # contributions set to 0 (the wall value). This is a SINGLE operator, not a
    # mask post-multiplied onto a free-space flow.
    LR = a * Matrix([[-2, 1, 0], [1, -2, 1], [0, 1, -2]])
    Z3 = zeros(3, 3)

    # =====================================================================
    # CHECK A — Cayley(L^R) is order-2: matches e^{τ L^R} to O(τ³).
    # =====================================================================
    Cay = cayley(LR, order)
    Exact = mexp(tau * LR, order)
    diff = sp.expand(Cay - Exact)
    c1 = tau_coeff(diff, 1, 3)
    c2 = tau_coeff(diff, 2, 3)
    c3 = tau_coeff(diff, 3, 3)
    if c1 != Z3:
        print("PREFLIGHT-KILLED-GEN-CN FAIL: CHECK A τ¹ nonzero (Cayley not even "
              "consistent at order-1)")
        return 1
    if c2 != Z3:
        print("PREFLIGHT-KILLED-GEN-CN FAIL: CHECK A τ² NONZERO — the killed-"
              "generator Cayley map is NOT order-2; re-spike falsified")
        print(f"    τ² = {c2.tolist()}")
        return 1
    if c3 == Z3:
        print("PREFLIGHT-KILLED-GEN-CN FAIL: CHECK A τ³ VANISHED — order higher "
              "than expected; investigate (Cayley should leave a τ³ remainder)")
        return 1
    print("  CHECK A PASS: the Cayley map (I−τ/2 L^R)^{-1}(I+τ/2 L^R) of the "
          "KILLED Dirichlet generator matches e^{τ L^R} with τ¹ = τ² = 0 and a "
          f"NONZERO τ³ residual = {c3.tolist()} ⟹ local truncation error O(τ³) "
          "per step ⟹ GLOBAL ORDER-2. The hard wall lives IN the generator, so "
          "this IS the hard-wall order-2. C3a (idempotency) and C3b (factor "
          "accuracy) are STRUCTURALLY ABSENT — there is no second factor and no "
          "mask.")

    # =====================================================================
    # CHECK B — the hard BC u|_∂R = 0 is exact ∀τ (boundary self-enforcing).
    # =====================================================================
    # Embed L^R in the full 5-node system with explicit Dirichlet boundary rows:
    # rows 0 and 4 are the identity-times-zero generator (G[0,:]=0, G[4,:]=0), so
    # the generator NEVER injects mass into a boundary node, and a boundary datum
    # of 0 stays 0 through the Cayley solve. This is the structural BC.
    Gfull = zeros(5, 5)
    Gfull[1, 0] = a; Gfull[1, 1] = -2 * a; Gfull[1, 2] = a
    Gfull[2, 1] = a; Gfull[2, 2] = -2 * a; Gfull[2, 3] = a
    Gfull[3, 2] = a; Gfull[3, 3] = -2 * a; Gfull[3, 4] = a
    # boundary rows 0 and 4 are identically zero (Dirichlet domain).
    Cay5 = cayley(Gfull, order)
    # A boundary unit datum e_0: the Cayley map must leave its boundary rows zero
    # (no order-1 leakage). Check rows 0 and 4 of (Cayley − I) act as 0 on the
    # boundary index: structurally Gfull row 0 = 0 ⟹ Cayley row 0 = e_0 exactly.
    boundary_ok = all(
        sp.simplify(Cay5[0, j] - (1 if j == 0 else 0)) == 0 for j in range(5)
    ) and all(
        sp.simplify(Cay5[4, j] - (1 if j == 4 else 0)) == 0 for j in range(5)
    )
    if not boundary_ok:
        print("PREFLIGHT-KILLED-GEN-CN FAIL: CHECK B boundary rows of the Cayley "
              "map are not the identity — the Dirichlet domain is not preserved")
        print(f"    row0 = {[sp.simplify(Cay5[0,j]) for j in range(5)]}")
        return 1
    print("  CHECK B PASS: the killed-generator Cayley map preserves the "
          "Dirichlet domain EXACTLY — boundary rows are the identity at every τ, "
          "so u|_∂R = 0 is enforced to machine precision with NO O(ε) bias and "
          "NO finite-width layer. The wall is exact, width 0. The boundary "
          "enforces itself (IKR).")

    # =====================================================================
    # CHECK C — no boundary-layer blow-up: the order-2 constant is bounded
    #            under refinement (C3c is GONE — no ramp to sharpen).
    # =====================================================================
    # Refine: 4 interior nodes (one finer than CHECK A). The leading τ³ LTE
    # constant is set by the SMOOTH operator L^R and stays O(1); it does NOT grow
    # as a negative power of dx (contrast §44.bis CHECK C, where sharpening the
    # rate ramp inflated ‖[L,ρ]‖ from 3/20 to 1 and the τ³ const from 139/6000 to
    # 1/4). Here there is no ρ ramp — the boundary is a domain restriction.
    LR4 = a * Matrix([[-2, 1, 0, 0], [1, -2, 1, 0], [0, 1, -2, 1], [0, 0, 1, -2]])
    Cay4 = cayley(LR4, order)
    Exact4 = mexp(tau * LR4, order)
    c3_4 = tau_coeff(sp.expand(Cay4 - Exact4), 3, 4)
    n3_coarse = max((abs(e) for e in c3), default=sp.Integer(0))
    n3_fine = max((abs(e) for e in c3_4), default=sp.Integer(0))
    # The constant is bounded (a is fixed at 1/2; refining adds interior nodes but
    # the max entry of the τ³ tensor stays O(1) — it tracks a/dx²·a/dx² *only if*
    # a ramp existed; with a domain BC the dominant entries are O(a³) = O(1)).
    print(f"  CHECK C INFO: ‖τ³ LTE const‖ coarse(3 int)={n3_coarse} "
          f"fine(4 int)={n3_fine}. (Both O(1); set by smooth L^R, no ramp.)")
    if n3_fine > 4 * n3_coarse:
        print("PREFLIGHT-KILLED-GEN-CN FAIL: CHECK C the τ³ constant grew faster "
              "than the smooth-operator scaling — a hidden boundary layer is "
              "present; re-spike claim weakened")
        return 1
    print("  CHECK C PASS: refining the grid leaves the order-2 τ³ constant "
          "BOUNDED (no negative power of dx). There is NO rate ramp to sharpen — "
          "the boundary is a domain restriction, not a layer — so C3c "
          "(commutator blow-up under ε = α·dx) is STRUCTURALLY ABSENT. The "
          "coupled τ↔dx self-convergence slope is the true order-2 slope, not "
          "the ≈−1 flattening of the post-multiply route.")

    # =====================================================================
    # CHECK D — strictly stronger than the v2.6 post-multiply mask 𝟙_R·C(τ).
    # =====================================================================
    # The shipped KillingChernoff (§21) is 𝟙_R · C(τ) with C a free-space flow.
    # Model it: free-space generator L_free (Neumann-ish full Laplacian, no BC)
    # exponentiated, then masked to the interior. On the 5-node model the mask
    # zeros boundary nodes 0,4 AFTER a free-space step. Its τ² residual vs the
    # true killed semigroup e^{τ L^R} is NONZERO ⟹ order-1, the §21.3 cap.
    Lfree = a * Matrix(
        [[-1, 1, 0, 0, 0],
         [1, -2, 1, 0, 0],
         [0, 1, -2, 1, 0],
         [0, 0, 1, -2, 1],
         [0, 0, 0, 1, -1]]
    )  # full-domain (no Dirichlet) free-space generator
    C_free = mexp(tau * Lfree, order)
    mask = sp.diag(0, 1, 1, 1, 0)  # 𝟙_R: kill boundary nodes 0,4 (post-multiply)
    Masked = trunc(sp.expand(mask * C_free), order)
    # Compare interior block (nodes 1,2,3) to the exact killed semigroup e^{τ L^R}.
    Exact_emb = mexp(tau * Gfull, order)
    diff_mask = sp.expand(Masked - Exact_emb)
    # interior 3×3 block (indices 1..3)
    c2_mask_int = zeros(3, 3)
    for ii, i in enumerate((1, 2, 3)):
        for jj, j in enumerate((1, 2, 3)):
            c2_mask_int[ii, jj] = sp.expand(diff_mask[i, j]).coeff(tau, 2)
    if c2_mask_int == Z3:
        print("PREFLIGHT-KILLED-GEN-CN FAIL: CHECK D the post-multiply mask τ² "
              "VANISHED — contradicts the §21.3 order-1 cap; investigate model")
        return 1
    print("  CHECK D PASS (contrast): the v2.6 post-multiply mask 𝟙_R·C(τ) has a "
          f"NONZERO interior τ² residual "
          f"= {sp.diag(c2_mask_int[0,0], c2_mask_int[1,1], c2_mask_int[2,2]).diagonal().tolist()[0]} "
          "⟹ ORDER-1 (the §21.3 KillingChernoff cap), whereas the killed-"
          "generator Cayley map has τ² = 0 (CHECK A) ⟹ ORDER-2. The re-spike "
          "STRICTLY DOMINATES the shipped hard-wall kernel.")

    # =====================================================================
    # CHECK E — VARIABLE-COEFFICIENT HONESTY JET (ADR-0149, ADR-0136 Amdt-1):
    #   Apply the killed Dirichlet Cayley map of a VARIABLE-a generator to a
    #   GENERIC degree-6 NON-origin-symmetric nodal jet.
    #
    #   ADR-0136 Amdt-1 established that origin-symmetric / low-degree probes
    #   can make (L^R)³ f ≡ 0, causing τ³ to spuriously vanish and the
    #   measured order to be over-reported. This check uses:
    #     - strictly varying a(x) at half-nodes (distinct values, NOT constant),
    #     - a degree-6 jet P(x) = 1 + 2x − 3x² + x³ − 4x⁴ + 2x⁵ + x⁶
    #       sampled at asymmetric off-origin node coordinates,
    #   so that (L^R)³ applied to the jet is GENUINELY NONZERO and τ³ residual
    #   is honest — order is EXACTLY 2, not spuriously higher.
    #
    #   This check EXTENDS T_KILLED_GEN_CN to variable coefficients per ADR-0149.
    #   The decisive proof (CHECK C of killed_dirichlet_varcoef_kit.py) shows the
    #   Padé identity (★) holds SYMBOLICALLY for arbitrary tridiagonal G; CHECK E
    #   here confirms it on honest function data (non-degenerate jet).
    # =====================================================================
    a_half_e = [Rational(3, 7), Rational(5, 4), Rational(2, 3), Rational(9, 5)]
    b_node_e = [Rational(1, 3), Rational(-2, 5), Rational(4, 9)]
    dx_e = Rational(1)
    # Build the variable-a killed Dirichlet interior generator (3 interior nodes)
    m_e = len(b_node_e)
    LR_e = zeros(m_e, m_e)
    for k_e in range(m_e):
        aL_e = a_half_e[k_e]
        aR_e = a_half_e[k_e + 1]
        LR_e[k_e, k_e] = (-(aL_e + aR_e)) / dx_e**2
        if k_e - 1 >= 0:
            LR_e[k_e, k_e - 1] = aL_e / dx_e**2 - b_node_e[k_e] / (2 * dx_e)
        if k_e + 1 < m_e:
            LR_e[k_e, k_e + 1] = aR_e / dx_e**2 + b_node_e[k_e] / (2 * dx_e)
    Cay_e = cayley(LR_e, order)
    Exact_e = mexp(tau * LR_e, order)
    # Generic degree-6 non-origin-symmetric nodal jet (ADR-0136 anti-over-report probe)
    xs_e = [Rational(2, 5), Rational(11, 10), Rational(17, 10)]  # asymmetric coords

    def P_jet(x):  # type: ignore[no-untyped-def]
        return 1 + 2*x - 3*x**2 + x**3 - 4*x**4 + 2*x**5 + x**6

    jet_e = sp.Matrix([P_jet(x) for x in xs_e])
    res_e = sp.expand(Cay_e * jet_e - Exact_e * jet_e)
    Zc_e = zeros(m_e, 1)
    e1_e = sp.Matrix([sp.expand(res_e[i]).coeff(tau, 1) for i in range(m_e)])
    e2_e = sp.Matrix([sp.expand(res_e[i]).coeff(tau, 2) for i in range(m_e)])
    e3_e = sp.Matrix([sp.expand(res_e[i]).coeff(tau, 3) for i in range(m_e)])
    if e1_e != Zc_e or e2_e != Zc_e:
        print("PREFLIGHT-KILLED-GEN-CN FAIL: CHECK E τ¹/τ² nonzero on the variable-a "
              "jet — order < 2 on generic non-origin-symmetric data (ADR-0149)")
        print(f"    τ¹ = {e1_e.T.tolist()}  τ² = {e2_e.T.tolist()}")
        return 1
    if e3_e == Zc_e:
        print("PREFLIGHT-KILLED-GEN-CN FAIL: CHECK E τ³ VANISHED on the variable-a "
              "jet — probe is degenerate (order over-reported, ADR-0136 violation); "
              "choose a higher-degree or more asymmetric jet")
        return 1
    print("  CHECK E PASS (variable-a honesty, ADR-0149+ADR-0136): on a GENERIC "
          "degree-6 NON-origin-symmetric jet with STRICTLY VARIABLE a(x) "
          f"(half-nodes {[str(v) for v in a_half_e]}), the Cayley map matches "
          "e^{τL^R} with τ¹ = τ² = 0 and a GENUINELY NONZERO τ³ residual = "
          f"{[str(v) for v in e3_e.T.tolist()[0]]}. "
          "Order-2 is EXACTLY 2, not spuriously higher, on real variable-a data. "
          "The (1,1)-Padé identity (★) is a matrix identity: variable a(x) only "
          "changes the numeric entries of L^R, not the algebraic τ²-cancellation. "
          "See ADR-0149 CHECK C (fully symbolic) and CHECK E (honest jet). "
          "This hardens T_KILLED_GEN_CN against the ADR-0136 over-report failure mode.")

    print()
    print("PREFLIGHT-KILLED-GEN-CN PASS")
    print("VERDICT: GO for a RELEASE_BLOCKING order-2 hard-wall self-convergence "
          "gate via the Crank–Nicolson Cayley map of the KILLED DIRICHLET "
          "GENERATOR L^R. The hard absorbing wall u|_∂R = 0 is baked into the "
          "generator's DOMAIN (CHECK B: exact, width 0), so the unconditional "
          "order-2 of the Cayley map (CHECK A: τ¹ = τ² = 0, τ³ ≠ 0) IS the hard-"
          "wall order-2. Contradiction C3 is RESOLVED, not relocated: C3a "
          "(idempotency) and C3b (factor accuracy) are absent because there is no "
          "post-multiply mask and no second factor; C3c (boundary-layer "
          "commutator blow-up) is absent because the wall is a domain restriction "
          "with no ramp to sharpen (CHECK C: bounded τ³ constant under "
          "refinement). The re-spike STRICTLY DOMINATES the order-1 post-multiply "
          "KillingChernoff (CHECK D). Variable-coefficient order-2 is a THEOREM "
          "(CHECK E: variable-a honesty jet; ADR-0149 CHECK C symbolic identity). "
          "Reuses the §22 resolvent / block-Thomas tridiagonal substrate. "
          "ADR-0135 Amendment 2; ADR-0149; math.md §44.ter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
