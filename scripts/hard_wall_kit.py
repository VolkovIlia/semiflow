#!/usr/bin/env python3
"""hard_wall_kit.py — PRE-FLIGHT for v8.0.0 F3 (ADR-0135, math.md §44.bis).

Feature 3: order-2 HARD absorbing wall via a resolvent-regularized projector
    P_ε = (I + ε κ 𝟙_R)^{-1},   ε = α·dx
applied symmetrically as the Strang palindrome  P_ε^{1/2} · C(τ) · P_ε^{1/2}.

THE CONTRADICTION (C3, math §21.8): the SHARP indicator 𝟙_R is idempotent
(𝟙_R^{1/2}=𝟙_R), so the symmetric split collapses to the §21.2-REJECTED
pre+post over-killing form and the hard absorbing wall is irreducibly ORDER-1
(the boundary jump makes [L, 𝟙_R] a non-removable O(τ) term). ADR-0135 proposes
to escape this by SMOOTHING the projector with a resolvent that is NON-idempotent
at finite ε, while letting ε = α·dx → 0 to recover the hard wall.

This PRE-FLIGHT decides GO / NARROW / NO-GO for the order-2 claim. It is built to
FALSIFY, not to rubber-stamp, on finite-dimensional matrix models (the honest
non-commuting setting). FOUR sub-checks; the oracle "passes" (exit 0) iff every
check produces its EXPECTED sign — the printed VERDICT is then NO-GO with a named,
documented obstruction.

  CHECK A (MECHANISM REAL — positive): P_ε = (I + εκ𝟙_R)^{-1} is NON-idempotent at
    finite ε (escapes §21.8 collapse); εκ→∞ recovers the hard projector. ✓

  CHECK B (PRIMARY OBSTRUCTION — the ADR operator as specified is NOT order-2):
    even with a perfectly SMOOTH, fixed-width rate ρ (no grid sharpness, no
    commutator pathology), the resolvent palindrome  (I+τρ)^{-1/2} C_L (I+τρ)^{-1/2}
    is NOT order-2: its τ² coefficient is NONZERO. Root cause is factor accuracy —
    (I+τρ)^{-1/2} matches e^{-τρ/2} only to O(τ); the τ² self-terms differ
    (3/8·ρ² vs 1/8·ρ²). The plain resolvent is an ORDER-1 rational approximant of
    the reaction exponential. EXPECT: τ² NONZERO ⟹ NO-GO for the ADR-0135 operator.

  CHECK B' (the only NARROW path — requires upgrading the factor): replace the
    plain resolvent half-step by the (1,1)-Padé / CAYLEY factor
    G(τ) = (I − τρ/4)(I + τρ/4)^{-1} = e^{-τρ/2} + O(τ³), and use a SMOOTH rate.
    Then G C_L G IS order-2 (τ¹,τ² vanish). EXPECT: order-2 — but this is no longer
    "(I+εκ𝟙)^{-1}", and it solves the SOFT collar problem, not the hard wall.

  CHECK C (CLOSES even the upgrade — boundary-layer blow-up under ε = α·dx): the
    order-2 τ³ CONSTANT is set by commutators of the rate ramp with L. Sharpen the
    ramp (1-cell jump, mimicking ε = α·dx as dx→0) and show ‖[L,ρ]‖ and the τ³
    error-constant STRICTLY GROW vs a wider ramp ⟹ under the coupled τ↔dx self-
    convergence limit the constant blows up in lockstep with refinement. EXPECT:
    sharp > wide ⟹ even the Cayley upgrade is τ-only at fixed dx, NOT global.

RULING encoded: NO-GO for a RELEASE_BLOCKING G_HARD_WALL_ORDER2 self-convergence
gate. The resolvent IS monotone+stable ⟹ Barles–Souganidis convergence to the
CORRECT hard-wall viscosity solution (math §44.bis; no wrong-limit / fraud risk),
but it does NOT deliver global order-2 — doubly obstructed (factor accuracy +
boundary layer). Honest-defer F3; ATTEMPT-ORIGINAL = Crank–Nicolson on the killed
Dirichlet generator (math §44.bis.5).

Exits 0 iff all checks give their expected signs; 1 otherwise.
"""

import sys


def main() -> int:
    try:
        import sympy as sp
        from sympy import Matrix, eye, zeros, symbols, factorial, diag
    except ImportError:
        print("PREFLIGHT-HARD-WALL FAIL: sympy not installed", flush=True)
        return 1

    tau = symbols("tau", positive=True)
    eps, kap = symbols("epsilon kappa", positive=True)

    def mexp(A, order):
        d = A.shape[0]
        out = eye(d)
        term = eye(d)
        for k in range(1, order + 1):
            term = sp.expand(term * A)
            out = out + term / factorial(k)
        return out

    def tau_coeff(Mexpr, power, d):
        out = zeros(d, d)
        for i in range(d):
            for j in range(d):
                out[i, j] = sp.expand(Mexpr[i, j]).coeff(tau, power)
        return out

    def resolvent_half_jet(rho, order):
        # (I + τρ)^{-1/2} = Σ_k binom(-1/2,k) (τρ)^k  — the ADR-0135 factor.
        d = rho.shape[0]
        out = zeros(d, d)
        term = eye(d)
        for k in range(0, order + 1):
            out = out + sp.binomial(sp.Rational(-1, 2), k) * term * tau**k
            term = sp.expand(term * rho)
        return out

    def cayley_half_jet(rho, order):
        # (I − τρ/4)(I + τρ/4)^{-1} = e^{-τρ/2} + O(τ³): order-2 reaction factor.
        d = rho.shape[0]
        num = eye(d) - tau * rho / 4
        inv = zeros(d, d)
        term = eye(d)
        for k in range(0, order + 1):  # (I+τρ/4)^{-1} = Σ (−τρ/4)^k
            inv = inv + term * tau**k
            term = sp.expand(term * (-rho / 4))
        prod = sp.expand(num * inv)
        # truncate to τ^order
        out = zeros(d, d)
        for i in range(d):
            for j in range(d):
                p = sp.Poly(sp.expand(prod[i, j]), tau)
                out[i, j] = sum(c * tau**(p.degree() - n)
                                for n, c in enumerate(p.all_coeffs())
                                if (p.degree() - n) <= order)
        return out

    order = 3
    Z3 = zeros(3, 3)
    L = Matrix([[-2, 1, 0], [1, -2, 1], [0, 1, -2]]) * sp.Rational(1, 2)
    C_L = mexp(tau * L, order)  # inner order-2 diffusion jet e^{τL}+O(τ³)

    # =====================================================================
    # CHECK A — resolvent non-idempotent at finite ε (mechanism real).
    # =====================================================================
    ind5 = diag(0, 0, 0, 1, 1)
    P_eps = (eye(5) + eps * kap * ind5).inv()
    if sp.simplify(P_eps * P_eps - P_eps) == zeros(5, 5):
        print("PREFLIGHT-HARD-WALL FAIL: CHECK A P_ε idempotent — no escape")
        return 1
    damped = sp.simplify(P_eps[3, 3])
    if sp.limit(damped, kap, sp.oo) != 0:
        print("PREFLIGHT-HARD-WALL FAIL: CHECK A εκ→∞ does not recover hard wall")
        return 1
    print(f"  CHECK A PASS: P_ε damped entry = {damped} ∈ (0,1) ⟹ NON-idempotent "
          "at finite ε (escapes §21.8 collapse); εκ→∞ ⟹ 0 (recovers hard "
          "projector). MECHANISM REAL.")

    # =====================================================================
    # CHECK B — ADR-0135 resolvent factor is NOT order-2 even for smooth ρ.
    # =====================================================================
    rho_smooth = diag(sp.Rational(2, 10), sp.Rational(5, 10), sp.Rational(8, 10))
    P_half = resolvent_half_jet(rho_smooth, order)
    S = sp.expand(P_half * C_L * P_half)
    Exact = mexp(tau * (L - rho_smooth), order)
    diff = sp.expand(S - Exact)
    c1 = tau_coeff(diff, 1, 3)
    c2 = tau_coeff(diff, 2, 3)
    if c1 != Z3:
        print("PREFLIGHT-HARD-WALL FAIL: CHECK B τ¹ nonzero (factor not even "
              "consistent at order-1)")
        return 1
    if c2 == Z3:
        print("PREFLIGHT-HARD-WALL FAIL: CHECK B τ² VANISHED — contradicts the "
              "factor-accuracy obstruction; investigate")
        return 1
    print("  CHECK B PASS (as a NO-GO): even for a SMOOTH rate, the resolvent "
          "palindrome (I+τρ)^{-1/2}·C_L·(I+τρ)^{-1/2} has NONZERO τ² term "
          f"= {sp.diag(c2[0,0], c2[1,1], c2[2,2]).diagonal().tolist()[0]} "
          "⟹ ORDER-1, not order-2. Root cause: (I+τρ)^{-1/2} matches e^{-τρ/2} "
          "only to O(τ) (τ² self-terms 3/8·ρ² vs 1/8·ρ²). The ADR-0135 operator "
          "AS SPECIFIED cannot reach order-2. PRIMARY OBSTRUCTION.")

    # =====================================================================
    # CHECK B' — Cayley/Padé factor upgrade IS order-2 for smooth ρ (narrow).
    # =====================================================================
    G_half = cayley_half_jet(rho_smooth, order)
    Sg = sp.expand(G_half * C_L * G_half)
    diffg = sp.expand(Sg - Exact)
    if tau_coeff(diffg, 1, 3) != Z3 or tau_coeff(diffg, 2, 3) != Z3:
        print("PREFLIGHT-HARD-WALL FAIL: CHECK B' Cayley upgrade not order-2 — "
              "narrow path does not even exist")
        print(f"    τ² = {tau_coeff(diffg, 2, 3).tolist()}")
        return 1
    c3_smooth = tau_coeff(diffg, 3, 3)
    print("  CHECK B' PASS: upgrading to the Cayley factor "
          "(I−τρ/4)(I+τρ/4)^{-1} = e^{-τρ/2}+O(τ³) restores ORDER-2 (τ¹,τ² zero) "
          "for a SMOOTH rate. NARROW PATH — but this is no longer (I+εκ𝟙)^{-1}, "
          "and it solves the SOFT collar, not the hard wall.")

    # =====================================================================
    # CHECK C — boundary-layer blow-up closes even the upgrade under ε=α·dx.
    # =====================================================================
    comm_smooth = sp.expand(L * rho_smooth - rho_smooth * L)
    rho_sharp = diag(0, 0, sp.Integer(2))  # 1-cell wall, magnitude ↑ ~ 1/w
    comm_sharp = sp.expand(L * rho_sharp - rho_sharp * L)
    n_smooth = max(abs(e) for e in comm_smooth)
    n_sharp = max(abs(e) for e in comm_sharp)
    G_sharp = cayley_half_jet(rho_sharp, order)
    Ss = sp.expand(G_sharp * C_L * G_sharp)
    Es = mexp(tau * (L - rho_sharp), order)
    c3_sharp = tau_coeff(sp.expand(Ss - Es), 3, 3)
    n3_smooth = max((abs(e) for e in c3_smooth), default=sp.Integer(0))
    n3_sharp = max((abs(e) for e in c3_sharp), default=sp.Integer(0))
    print(f"  CHECK C INFO: ‖[L,ρ]‖ smooth={n_smooth} sharp={n_sharp}; "
          f"‖τ³ coeff‖ smooth={n3_smooth} sharp={n3_sharp}.")
    if not (n_sharp > n_smooth and n3_sharp > n3_smooth):
        print("PREFLIGHT-HARD-WALL FAIL: CHECK C expected blow-up not exhibited")
        return 1
    print("  CHECK C PASS: sharpening the wall layer (ε = α·dx → 0) STRICTLY "
          "inflates ‖[L,ρ]‖ and the order-2 τ³ constant ⟹ under the coupled "
          "τ↔dx self-convergence limit the constant blows up in lockstep with "
          "refinement. Even the Cayley upgrade is τ-only at FIXED dx, NOT global. "
          "OBSTRUCTION CONFIRMED.")

    print()
    print("PREFLIGHT-HARD-WALL PASS")
    print("VERDICT: NO-GO for a RELEASE_BLOCKING order-2 hard-wall self-"
          "convergence gate. The ADR-0135 operator (I+εκ𝟙)^{-1} is DOUBLY "
          "obstructed: (1) factor accuracy — the plain resolvent is an order-1 "
          "rational approximant of the reaction exponential (CHECK B); (2) "
          "boundary layer — even a Cayley-upgraded factor loses global order-2 "
          "under ε = α·dx (CHECK C). It IS non-idempotent (CHECK A) and "
          "monotone+stable ⟹ Barles–Souganidis convergence to the CORRECT hard-"
          "wall viscosity solution (no wrong-limit risk). Honest-defer F3; "
          "ATTEMPT-ORIGINAL = Crank–Nicolson on the killed Dirichlet generator "
          "(math §44.bis.5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
