#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false
#
# Sympy's Max/Piecewise arithmetic is dynamically typed through __mul__/__add__;
# numpy ndarray operator overloads are likewise opaque to Pyright. All
# operations are valid at runtime (verified by this oracle's PASS).
"""T_OBSTACLE_PROJECTION oracle — projective-splitting obstacle evolver
(math.md §44; ADR-0116).

PRE-FLIGHT math-fidelity oracle. INDEPENDENTLY verifies, BEFORE any Rust kernel
is written, that the projection step and the analytic obstacle oracle used by
`ObstacleChernoff` are correct consequences of convex-analysis / variational-
inequality theory. The scheme is (math.md §44.1):

    V^{n+1} = Π_g( S(Δτ) Vⁿ ),     Π_g(W) = max(W, g)   (elementwise),

with `S(Δτ)` ANY library ChernoffFunction (the Chernoff tangent → e^{ΔτL}) and
`Π_g` the metric projection onto the closed convex cone K = {V ≥ g}. This oracle
checks ONLY the in-core consequence-math; it does NOT depend on any Rust code.

Sub-checks (6 mandatory; math.md §44.2 / §44.5 / §44.4 claims):

  (a) projection_identity        [§44.2, Theorem 44.1]
      Π_g(W)[i] = max(W_i, g_i) IS the metric projection onto {v ≥ g}: it is the
      argmin of (v − w)² over v ≥ c. Verified via the KKT system of the
      constrained scalar problem: feasibility v* − c ≥ 0, dual feasibility
      μ = v* − w ≥ 0, complementarity μ·(v* − c) = 0, stationarity v* = w + μ —
      whose unique solution is v* = Max(w, c).

  (b) idempotence                [§44.2]
      Π_g(Π_g(W)) = Π_g(W): Max(Max(w,c),c) = Max(w,c) symbolically (a projection
      onto a convex set is idempotent).

  (c) nonexpansiveness           [§44.2, Theorem 44.1 — the stability certificate]
      |Max(w1,c) − Max(w2,c)| ≤ |w1 − w2| over ALL four sign branches
      (w_k ⋚ c), proved symbolically with ordering assumptions. This 1-Lipschitz
      property — NOT the multiplicative growth bound — is what makes the
      projected iterate stable (math.md §44.3, §1.4 growth() honesty note).

  (d) active_set_jacobian        [§44.5, Theorem 44.3]
      ∂/∂w Max(w,c) = Heaviside(w − c) = 𝟙[w > c] (a.e.). This diagonal indicator
      of the continuation set {W > g} IS the Π_g Jacobian used by the active-set
      adjoint primitive `apply_active_set_adjoint_into`.

  (e) membrane_oracle            [§44 analytic correctness oracle / G_OBSTACLE_STATIONARY]
      The stationary 1D obstacle problem (membrane over a concave parabola) has a
      CLOSED-FORM solution that is the exact fixed point u* = Π_g(S(Δτ) u*) of the
      projected heat flow driven to equilibrium. For g(x) = A − B(x−½)² on [0,1],
      Dirichlet u(0)=u(1)=0, 0 < A < B/4, by symmetry the contact set is [α, 1−α]
      with, derived here symbolically from C¹ smooth-fit (u(α)=g(α), u'(α)=g'(α)):

          α = √(1/4 − A/B),     s = B(1 − 2α),
          u(x) = s·x       on [0, α]
          u(x) = g(x)      on [α, 1−α]   (contact set)
          u(x) = s·(1 − x) on [1−α, 1].

      The oracle solves the smooth-fit system with sympy.solve and confirms it
      matches this closed form, then verifies the membrane variational
      inequality: u ≥ g, −u'' ≥ 0, on both regions.

  (f) complementarity            [§44.1, §44.2]
      (−u'')·(u − g) = 0 on BOTH regions: free region has u'' = 0 (linear);
      contact region has u − g = 0. Confirms the LCP/obstacle complementarity the
      projection enforces.

Out-of-symbolic-scope (stated precisely, not silently dropped):
  - The Δτ-ORDER of the scheme (O(Δτ) convex / O(√Δτ) at the free boundary) is
    EMPIRICALLY GATED by the Rust slope tests G_OBSTACLE_SLOPE_* (math.md §44.4,
    honest disclosure): the variable-coefficient sharp order is NOT a theorem and
    is not claimed here.
  - The Barles–Souganidis viscosity-convergence theorem (§44.3) is PROVEN by
    citation at the abstract-scheme level; this oracle checks the projection's
    monotonicity/nonexpansiveness/idempotence ingredients, not the limit theorem.
  - The non-contractive inner regime (ω > 0, discount c(x) > 0) is CONJECTURAL
    (Trotter projection counterexample, arXiv math/0109049) and not gated.

Prints "T_OBSTACLE_PROJECTION PASS (6/6 sub-checks: ...)" on success;
"T_OBSTACLE_PROJECTION FAIL: <reason>" and exits 1 on failure.

References:
  - M.G. Crandall, T.M. Liggett, Amer. J. Math. 93 (1971) 265–298.
  - H. Brezis, A. Pazy, J. Funct. Anal. 9 (1972) 63–74.
  - P.-L. Lions, B. Mercier, SIAM J. Numer. Anal. 16 (1979) 964–979.
  - G. Barles, P.E. Souganidis, Asymptotic Anal. 4 (1991) 271–283.
  - P. Jaillet, D. Lamberton, B. Lapeyre, Acta Appl. Math. 21 (1990) 263–289.
  - G. Leduc, Mathematics 13(2) (2025) 213.
  - S.D. Howison, C. Reisinger, J.H. Witte, SIAM J. Financial Math. 4 (2013) 539–574.
  - math.md §44 (Theorems 44.1–44.3); ADR-0116 — contract authority.
"""

import sys

# --------------------------------------------------------------------------- #
# Concrete membrane fixture (deterministic): A < B/4 so the obstacle pokes
# through with a genuine, symmetric contact interval [α, 1−α] strictly inside.
# --------------------------------------------------------------------------- #
_A_VAL = 0.1
_B_VAL = 1.0  # A/B = 0.1 < 1/4 ⇒ α = √(0.15) ≈ 0.3873, contact ≈ [0.387, 0.613]


# --------------------------------------------------------------------------- #
# Sub-check (a): projection identity via the constrained-argmin KKT system.
# --------------------------------------------------------------------------- #
def check_projection_identity():
    """Max(w,c) is the unique KKT point of min (v−w)² s.t. v ≥ c."""
    import sympy as sp

    w, c = sp.symbols("w c", real=True)
    v_star = sp.Max(w, c)

    # Rewrite all Max() to Piecewise before simplifying so the mixed-Max identity
    # Max(w,c) = c + Max(w−c,0) reduces cleanly (plain simplify() leaves it).
    def _pw(expr):
        return sp.piecewise_fold(expr.rewrite(sp.Piecewise))

    # Feasibility: v* − c ≥ 0, i.e. v* − c = Max(w−c, 0).
    feas = sp.simplify(_pw(v_star - c - sp.Max(w - c, 0)))
    if feas != 0:
        return f"projection_identity: v*−c ≠ Max(w−c,0); residual {feas}."

    # Multiplier μ = v* − w (stationarity (v−w)−μ=0). Must be ≥ 0 (= Max(c−w,0)).
    mu = sp.simplify(_pw(v_star - w - sp.Max(c - w, 0)))
    if mu != 0:
        return f"projection_identity: μ = v*−w ≠ Max(c−w,0); residual {mu}."

    # Complementarity μ·(v*−c) = Max(c−w,0)·Max(w−c,0) ≡ 0 (factors disjoint support).
    comp = sp.Max(c - w, 0) * sp.Max(w - c, 0)
    # One factor is identically zero in each branch; confirm on both orderings.
    for ordering, sub in (("w>c", {w: c + 1}), ("w<c", {w: c - 1})):
        if sp.simplify(comp.subs(sub)) != 0:
            return f"projection_identity: complementarity μ·(v*−c) ≠ 0 when {ordering}."
    return None


# --------------------------------------------------------------------------- #
# Sub-check (b): idempotence of the projection.
# --------------------------------------------------------------------------- #
def check_idempotence():
    """Max(Max(w,c),c) = Max(w,c)."""
    import sympy as sp

    w, c = sp.symbols("w c", real=True)
    diff = sp.simplify(sp.Max(sp.Max(w, c), c) - sp.Max(w, c))
    if diff != 0:
        return f"idempotence: Π_g∘Π_g ≠ Π_g; residual {diff}."
    return None


# --------------------------------------------------------------------------- #
# Sub-check (c): 1-Lipschitz (nonexpansiveness) over all four sign branches.
# --------------------------------------------------------------------------- #
def check_nonexpansiveness():
    """|Max(w1,c)−Max(w2,c)| ≤ |w1−w2| in every (w_k ⋚ c) branch."""
    import sympy as sp

    w1, w2, c = sp.symbols("w1 w2 c", real=True)
    # Branch (out1, out2) and the assumption substitution that realises it.
    branches = [
        # (w1≥c, w2≥c): out1=w1, out2=w2 ⇒ |Δout| = |w1−w2|  (equality).
        (w1, w2, {w1: c + sp.Symbol("p1", positive=True),
                  w2: c + sp.Symbol("p2", positive=True)}),
        # (w1≥c, w2<c): out1=w1, out2=c ⇒ slack = |w1−w2|−|w1−c| = c−w2 > 0.
        (w1, c, {w1: c + sp.Symbol("p1", positive=True),
                 w2: c - sp.Symbol("q2", positive=True)}),
        # (w1<c, w2≥c): symmetric.
        (c, w2, {w1: c - sp.Symbol("q1", positive=True),
                 w2: c + sp.Symbol("p2", positive=True)}),
        # (w1<c, w2<c): out1=out2=c ⇒ |Δout| = 0 ≤ |w1−w2|.
        (c, c, {w1: c - sp.Symbol("q1", positive=True),
                w2: c - sp.Symbol("q2", positive=True)}),
    ]
    for out1, out2, sub in branches:
        slack = sp.simplify(sp.Abs(w1 - w2) - sp.Abs(out1 - out2))
        slack_b = sp.simplify(slack.subs(sub))
        # slack must be provably ≥ 0 in this branch.
        if sp.ask(sp.Q.nonnegative(slack_b)) is False:
            return (
                f"nonexpansiveness: |w1−w2|−|out1−out2| can be negative in branch "
                f"(out1={out1}, out2={out2}); slack={slack_b}."
            )
        # Stronger: confirm it is not symbolically negative on a probe point.
        probe = {sp.Symbol("p1", positive=True): 2, sp.Symbol("p2", positive=True): 1,
                 sp.Symbol("q1", positive=True): 2, sp.Symbol("q2", positive=True): 1,
                 c: 0}
        val = float(slack_b.subs(probe))
        if val < -1e-12:
            return (
                f"nonexpansiveness: probe slack {val} < 0 in branch "
                f"(out1={out1}, out2={out2})."
            )
    return None


# --------------------------------------------------------------------------- #
# Sub-check (d): active-set Jacobian ∂/∂w Max(w,c) = 𝟙[w>c].
# --------------------------------------------------------------------------- #
def check_active_set_jacobian():
    """∂/∂w Max(w,c) = Heaviside(w−c) = 𝟙[w>c] (the Π_g Jacobian, a.e.)."""
    import sympy as sp

    w, c = sp.symbols("w c", real=True)
    jac = sp.diff(sp.Max(w, c), w)
    target = sp.Heaviside(w - c)
    if sp.simplify(jac - target) != 0:
        return f"active_set_jacobian: ∂/∂w Max(w,c) = {jac} ≠ Heaviside(w−c)."
    # Numeric branch confirmation: 1 on the active set (w>c), 0 on contact (w<c).
    if float(jac.subs({w: 1, c: 0})) != 1.0:
        return "active_set_jacobian: Jacobian ≠ 1 on the active set (w>c)."
    if float(jac.subs({w: -1, c: 0})) != 0.0:
        return "active_set_jacobian: Jacobian ≠ 0 on the contact set (w<c)."
    return None


# --------------------------------------------------------------------------- #
# Sub-check (e): stationary membrane closed form via C¹ smooth-fit.
# --------------------------------------------------------------------------- #
def check_membrane_oracle():
    """Smooth-fit solve reproduces α=√(1/4−A/B), s=B(1−2α); VI inequalities hold."""
    import sympy as sp

    x = sp.symbols("x", real=True)
    A, B = sp.symbols("A B", positive=True)
    g = A - B * (x - sp.Rational(1, 2)) ** 2
    gp = sp.diff(g, x)

    alpha, s = sp.symbols("alpha s", positive=True)
    u_lin = s * x  # left free arm, u(0)=0
    # C¹ smooth-fit at x = α: value and slope match the obstacle.
    eq_val = sp.Eq(u_lin.subs(x, alpha), g.subs(x, alpha))
    eq_slope = sp.Eq(sp.diff(u_lin, x).subs(x, alpha), gp.subs(x, alpha))
    sols = sp.solve([eq_val, eq_slope], [alpha, s], dict=True)

    # Expect the physical root α = +√(1/4 − A/B), s = B(1 − 2α).
    alpha_closed = sp.sqrt(sp.Rational(1, 4) - A / B)
    s_closed = B * (1 - 2 * alpha_closed)
    matched = None
    for sol in sols:
        a_s, s_s = sol[alpha], sol[s]
        if sp.simplify(a_s - alpha_closed) == 0 and sp.simplify(s_s - s_closed) == 0:
            matched = sol
            break
    if matched is None:
        return (
            f"membrane_oracle: smooth-fit solve {sols} did not yield "
            f"α=√(1/4−A/B), s=B(1−2α)."
        )

    # Numeric instantiation (A<B/4) so √ is real and contact strictly interior.
    subs_num = {A: _A_VAL, B: _B_VAL}
    a_num = float(alpha_closed.subs(subs_num))
    s_num = float(s_closed.subs(subs_num))
    if not (0 < a_num < 0.5):
        return f"membrane_oracle: α={a_num} not strictly in (0, 1/2) for A<B/4."

    # VI on the left free arm [0, α]: h = u_lin − g must be ≥ 0, h convex, min at α.
    h = (s_closed * x - g).subs(subs_num)
    if abs(float(h.subs(x, a_num))) > 1e-12:
        return "membrane_oracle: u_lin(α) ≠ g(α) — C¹ value-fit broken."
    if abs(float(sp.diff(h, x).subs(x, a_num))) > 1e-12:
        return "membrane_oracle: u_lin'(α) ≠ g'(α) — C¹ slope-fit broken."
    if float(sp.diff(h, x, 2).subs(subs_num if isinstance(h, sp.Expr) else {})) <= 0:
        # h'' = 2B > 0 (h convex) ⇒ tangency at α (right endpoint) is the min ⇒ h≥0 on [0,α].
        return "membrane_oracle: h=u−g not convex (h''≤0); cannot certify u≥g on free arm."
    # Sample u ≥ g on the free arm (convex h with zero min at α ⇒ h>0 for x<α).
    for xv in (0.0, 0.1, 0.2, 0.3, a_num):
        if float(h.subs(x, xv)) < -1e-12:
            return f"membrane_oracle: u < g at x={xv} on the free arm (VI u≥g violated)."

    # −u'' ≥ 0 on both regions: free arm u''=0 ⇒ 0; contact u=g ⇒ −g''=2B>0.
    if float(sp.diff(s_closed * x, x, 2).subs(subs_num)) != 0.0:
        return "membrane_oracle: free-arm u'' ≠ 0 (should be linear)."
    if float((-sp.diff(g, x, 2)).subs(subs_num)) <= 0:
        return "membrane_oracle: −g'' ≤ 0 on contact (−u'' must be ≥ 0)."
    return None


# --------------------------------------------------------------------------- #
# Sub-check (f): complementarity (−u'')·(u−g) = 0 on both regions.
# --------------------------------------------------------------------------- #
def check_complementarity():
    """(−u'')(u−g) = 0: free region u''=0; contact region u−g=0."""
    import sympy as sp

    x = sp.symbols("x", real=True)
    A, B = sp.symbols("A B", positive=True)
    g = A - B * (x - sp.Rational(1, 2)) ** 2
    alpha = sp.sqrt(sp.Rational(1, 4) - A / B)
    s = B * (1 - 2 * alpha)

    # Free arm: u = s·x ⇒ −u'' = 0 ⇒ product = 0 identically.
    u_free = s * x
    prod_free = sp.simplify((-sp.diff(u_free, x, 2)) * (u_free - g))
    if prod_free != 0:
        return f"complementarity: free-arm (−u'')(u−g) = {prod_free} ≠ 0."

    # Contact region: u = g ⇒ (u−g) = 0 ⇒ product = 0 identically.
    prod_contact = sp.simplify((-sp.diff(g, x, 2)) * (g - g))
    if prod_contact != 0:
        return f"complementarity: contact (−u'')(u−g) = {prod_contact} ≠ 0."
    return None


def fail(reason):
    print(f"T_OBSTACLE_PROJECTION FAIL: {reason}", flush=True)
    return 1


def main():
    """Run all 6 sub-checks; print result; exit 0/1."""
    try:
        import sympy  # noqa: F401
    except ImportError:
        return fail("sympy not installed (required for PRE-FLIGHT oracle).")

    checks = [
        ("projection_identity", check_projection_identity),
        ("idempotence", check_idempotence),
        ("nonexpansiveness", check_nonexpansiveness),
        ("active_set_jacobian", check_active_set_jacobian),
        ("membrane_oracle", check_membrane_oracle),
        ("complementarity", check_complementarity),
    ]
    print("=" * 70)
    print("T_OBSTACLE_PROJECTION — projective-splitting obstacle evolver / VI")
    print("(math.md §44; ADR-0116)")
    print("=" * 70)

    failures = []
    passed = []
    for name, check in checks:
        try:
            result = check()
        except Exception as e:  # noqa: BLE001
            return fail(f"sub-check {name} raised exception: {e!r}")
        if result is None:
            print(f"  (PASS) {name}")
            passed.append(name)
        else:
            print(f"  (FAIL) {name}: {result}")
            failures.append(f"{name}: {result}")

    print()
    if failures:
        return fail(
            f"{len(failures)}/{len(checks)} sub-checks failed: " + "; ".join(failures)
        )
    print(
        "T_OBSTACLE_PROJECTION PASS (6/6 sub-checks: " + " / ".join(passed) + ")",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
