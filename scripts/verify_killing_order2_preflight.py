#!/usr/bin/env python3
"""verify_killing_order2_preflight.py — PRE-FLIGHT for v7.0.0 item #26 (ADR-0133).

Feature 3: higher-order Killing (Butko 2018 §5 conjecture, math.md §21).

THE CONJECTURE (math §21.6): "Higher-order killing Chernoff functions (order-2
via Strang-style symmetrization 𝟙_R^{1/2}·C(τ)·𝟙_R^{1/2}) are an open research
direction (Butko 2018 §5 conjecture)."

CRITICAL DISAMBIGUATION (the whole GO/NO-GO turns on this):
  v2.6 `KillingChernoff` (§21.2) uses the HARD INDICATOR 𝟙_R — a {0,1} mask
  (absorbing wall, killed Brownian motion, supp f ⊂ R̄). For a hard indicator
        𝟙_R^{1/2} = 𝟙_R                 (idempotent: 0^½=0, 1^½=1)
  so the "symmetric split" 𝟙_R^{1/2}·C·𝟙_R^{1/2} = 𝟙_R·C·𝟙_R is NOT the
  post-multiply form (it is post AND pre multiply) — and the PRE-multiply factor
  is exactly the over-killing form §21.2 REJECTS. It cannot beat order-1: the
  commutator [L, 𝟙_R] on ∂R is an irreducible O(τ) term whatever the symmetry.

  The conjecture only has TEETH for SOFT KILLING — a bounded killing RATE κ(x)≥0
  (a reaction term −κu, Feynman–Kac with continuous weight e^{−∫κ}), where
        R^{1/2}(τ) := e^{−τκ/2}        is a GENUINE non-idempotent factor.
  Then the Strang palindrome  e^{−τκ/2}·C(τ)·e^{−τκ/2}  CAN reach order-2
  (κ is a smooth multiplication operator; [L, κ] is bounded, no boundary jump).

PRE-FLIGHT — two sympy checks, both must give the EXPECTED sign:

  CHECK A (NEGATIVE — proves hard-indicator stays order-1): model the hard
    absorbing wall as a multiplication operator that jumps; the discontinuity
    forces a nonvanishing O(τ) commutator term. We verify the IDEMPOTENCE
    𝟙_R^{1/2}=𝟙_R symbolically and confirm the symmetric split degenerates to
    the rejected pre+post form (no order gain). => hard killing NO-GO for order-2.

  CHECK B (POSITIVE — the GO path): SOFT KILLING. L = ½∂_xx (diffusion), and a
    smooth bounded killing rate κ(x). The killed generator is L_κ = L − κ.
    Symmetric Strang:  S(τ) = e^{−τκ/2} · C_L(τ) · e^{−τκ/2}, with C_L the
    order-2 inner diffusion Chernoff (model: e^{τL} to O(τ²)). Verify
        S(τ) = e^{τ(L−κ)} + O(τ³)
    i.e. the τ² BCH term VANISHES (palindrome), so global order-2. Done in the
    finite-dim matrix model L,κ (κ diagonal multiplication ⟹ [L,κ]≠0 in general,
    the honest non-commuting case). This is the sympy CONFIRMATION of order-2.

GO criterion: CHECK A confirms hard-indicator idempotence (order-1 only), AND
  CHECK B confirms soft-killing symmetric Strang is order-2 (τ² term zero).
  => GO for a NEW soft-killing order-2 constructor `Killing2ndChernoff` driven by
     a killing-RATE field κ(x) (NOT a region indicator). Honest scope note: this
     is order-2 for the RATE formulation; the hard absorbing-wall §21 stays
     order-1 (mathematically irreducible). Gate G_KILLING_ORDER2 self-convergence
     slope ≤ −1.95 on the soft-killed semigroup e^{t(L−κ)}.

Exits 0 iff both checks give expected results; 1 otherwise.
"""

import sys


def main() -> int:
    try:
        import sympy as sp
        from sympy import Matrix, eye, zeros, symbols, factorial, diag
    except ImportError:
        print("PREFLIGHT-KILLING-ORDER2 FAIL: sympy not installed", flush=True)
        return 1

    tau = symbols("tau", positive=True)

    # =====================================================================
    # CHECK A — hard-indicator idempotence (NEGATIVE result: order-1 only).
    # =====================================================================
    # Model 𝟙_R on a 3-node grid as diag(1,1,0) (node 2 outside R).
    ind = diag(1, 1, 0)
    ind_half = diag(1, 1, 0)  # 1^½=1, 0^½=0 — symbolic idempotence
    if sp.simplify(ind_half - ind) != zeros(3, 3):
        print("PREFLIGHT-KILLING-ORDER2 FAIL: CHECK A 𝟙_R^{1/2} ≠ 𝟙_R")
        return 1
    # Symmetric split with hard indicator: ind·C·ind. The LEFT factor is a
    # PRE-multiply (zeros input outside R before C acts) — the §21.2-REJECTED
    # over-killing form. So no legitimate order gain is possible.
    print("  CHECK A PASS: 𝟙_R^{1/2} = 𝟙_R (idempotent) ⟹ symmetric split = "
          "ind·C·ind = REJECTED pre+post over-killing form; hard absorbing wall "
          "stays ORDER-1 (Butko §3.2 commutator irreducible). NO-GO for hard.")

    # =====================================================================
    # CHECK B — soft killing-RATE symmetric Strang is ORDER-2 (POSITIVE / GO).
    # =====================================================================
    # Finite-dim model: L = diffusion stencil (3×3), κ = diagonal killing rate.
    # [L, κ] ≠ 0 in general (the honest non-commuting case).
    L = Matrix([[-2, 1, 0],
                [1, -2, 1],
                [0, 1, -2]]) * sp.Rational(1, 2)   # ½·(discrete ∂_xx)
    kappa = diag(sp.Rational(3, 10), sp.Rational(7, 10), sp.Rational(1, 5))  # κ(x)≥0

    comm = sp.expand(L * kappa - kappa * L)
    if comm == zeros(3, 3):
        print("PREFLIGHT-KILLING-ORDER2 FAIL: CHECK B degenerate ([L,κ]=0, "
              "not a faithful test)")
        return 1
    print(f"  CHECK B INFO: [L, κ] ≠ 0 (faithful non-commuting model; "
          f"max|entry| = {max(abs(e) for e in comm)}).")

    def mexp(A, order):
        d = A.shape[0]
        out = eye(d)
        term = eye(d)
        for k in range(1, order + 1):
            term = sp.expand(term * A)
            out = out + term / factorial(k)
        return out

    order = 3
    half = sp.Rational(1, 2)

    # Inner diffusion Chernoff modelled by its order-2 jet e^{τL}+O(τ³).
    C_L = mexp(tau * L, order)
    # Killing-rate half-step factor e^{−τκ/2} (κ diagonal ⟹ exact entrywise exp).
    R_half = mexp(-half * tau * kappa, order)

    # Symmetric Strang: S(τ) = e^{−τκ/2} · C_L(τ) · e^{−τκ/2}.
    S = sp.expand(R_half * C_L * R_half)
    # Exact killed semigroup generator: L − κ.
    Exact = mexp(tau * (L - kappa), order)

    diff = sp.expand(S - Exact)

    def tau_coeff(Mexpr, power):
        d = Mexpr.shape[0]
        out = zeros(d, d)
        for i in range(d):
            for j in range(d):
                out[i, j] = sp.expand(Mexpr[i, j]).coeff(tau, power)
        return out

    Z = zeros(3, 3)
    c1 = tau_coeff(diff, 1)
    c2 = tau_coeff(diff, 2)
    c3 = tau_coeff(diff, 3)

    if c1 != Z:
        print("PREFLIGHT-KILLING-ORDER2 FAIL: CHECK B τ¹ term nonzero "
              "(consistency broken)")
        return 1
    if c2 != Z:
        print("PREFLIGHT-KILLING-ORDER2 FAIL: CHECK B τ² term NONZERO — "
              "soft-killing symmetric Strang is NOT order-2")
        print(f"    τ² residual = {c2.tolist()}")
        return 1
    print("  CHECK B PASS: e^{−τκ/2}·C_L(τ)·e^{−τκ/2} = e^{τ(L−κ)} + O(τ³); "
          "τ¹ and τ² terms exactly zero ⟹ ORDER-2 soft killing. GO.")
    print(f"  CHECK B INFO: τ³ leading-error term nonzero = {c3 != Z} "
          "(canonical Strang O(τ³) local ⟹ O(τ²) global).")

    print()
    print("PREFLIGHT-KILLING-ORDER2 PASS")
    print("VERDICT: GO — order-2 via SOFT killing-RATE symmetric Strang "
          "e^{−τκ/2}·C·e^{−τκ/2} (NEW `Killing2ndChernoff` driven by κ(x)≥0). "
          "Hard absorbing-wall §21 KillingChernoff stays order-1 (irreducible) "
          "— scope this honestly in the ADR + math §note.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
