#!/usr/bin/env python3
# pyright: reportUnusedVariable=false, reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false
#
# Sympy dynamic-typing through Symbol / Array / Matrix / Poly; Pyright cannot
# trace the return types. All operations are valid sympy at runtime — verified
# by the T_CARNOT_STEP4 PASS gate below.
"""T_CARNOT_STEP4: step-4 (filiform N=5) Carnot palindrome tangency sympy oracle (ADR-0136, v8.0.0 F4 SPIKE).

This is the MATH-PREREQUISITE oracle for the F4 "step-k Carnot closure via
recursive palindrome-of-palindromes" direction. It answers ONE crux question:

    To what tangency order does the palindromic Strang-Hörmander product
        F(τ) = exp(τ/4·X₁²) ∘ exp(τ/2·X₂²) ∘ exp(τ/4·X₁²)
    match the exact semigroup exp(τ·L), L = X₁² + X₂², on a step-4 Carnot group?

It does so with TWO INDEPENDENT WITNESSES:

WITNESS A — free-associative-algebra Strang order (step-INDEPENDENT).
  Symbolic noncommutative BCH of log(F) vs τ(A+B). Proves:
    - τ² coefficient of log(F) is identically ZERO (palindromic cancellation),
    - τ³ coefficient is the standard NON-zero double-commutator combo,
  hence the flat palindromic Strang is order-2 tangent for ANY A, B — the Carnot
  step is irrelevant to the order. This is an algebraic identity in the free
  algebra on {A, B}; it holds for step-2 (Engel-style), step-4, and beyond.

WITNESS B — concrete step-4 differential-operator tangency (filiform N=5).
  (b1) Bracket structure: X₃=[X₁,X₂], X₄=[X₁,X₃], X₅=[X₁,X₄], filiform
       termination, Hörmander rank 5 (genuine step-4, not degenerate).
  (b2) Tangency order of F(τ)f − exp(τL)f measured EXACTLY on a polynomial test
       function at a GENERIC (non-origin) base point, so the τ³ residual does NOT
       spuriously vanish. Confirms the genuine tangency order is exactly 2.

The step-4 filiform N=5 Carnot group (the "first non-Engel depth-5 case"):
  coords (x1, x2, x3, x4, x5) ∈ ℝ⁵, stratification g = g1⊕g2⊕g3⊕g4 with
  dim = (2, 1, 1, 1).  Bratzlavsky filiform basis (Bonfiglioli 2007 §4.3.6,
  generalising the Engel N=4 case to N=5):
    X1 = ∂_{x1}
    X2 = ∂_{x2} + x1·∂_{x3} + (x1²/2)·∂_{x4} + (x1³/6)·∂_{x5}
    X3 = [X1,X2] = ∂_{x3} + x1·∂_{x4} + (x1²/2)·∂_{x5}
    X4 = [X1,X3] = ∂_{x4} + x1·∂_{x5}
    X5 = [X1,X4] = ∂_{x5}
  Sub-Laplacian L = X1² + X2², bracket-generating at step 4.

Prints exactly:
  T_CARNOT_STEP4 PASS — all sub-checks pass; flat palindromic Strang is order-2
  T_CARNOT_STEP4 FAIL: <msg> — first failing sub-check

Exit code: 0 on PASS, 1 on FAIL.

References:
  - Bonfiglioli-Lanconelli-Uguzzoni 2007 §4.3.6 (filiform Carnot groups)
  - Hairer-Lubich-Wanner 2006 §III.5 (palindromic Strang order theorem)
  - Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 (K=2 tangency framework)
  - math.md §28.bis.5 (step-4 proof sketch; nesting-is-unnecessary finding)
  - ADR-0136 (palindrome-of-palindromes hypothesis under test)
  - scripts/lie_bracket_kit.py (reusable Lie-bracket helpers, v3.1)

Usage:
    python3 scripts/carnot_step4_kit.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import sympy as sp  # noqa: E402

from lie_bracket_kit import generates_T, lie_bracket  # noqa: E402

FAILED = False


def check(name: str, ok: bool, detail: str = "") -> None:
    """Print PASS/FAIL; record failure but continue (run all witnesses)."""
    global FAILED
    if ok:
        print(f"  [{name}] PASS")
    else:
        FAILED = True
        print(f"  [{name}] FAIL — {detail}")


# ════════════════════════════════════════════════════════════════════════════
# WITNESS A — free-algebra Strang order (step-INDEPENDENT crux)
# ════════════════════════════════════════════════════════════════════════════


def free_algebra_strang_order() -> None:
    """Noncommutative BCH of palindromic Strang in the free algebra on {A, B}.

    Strang product  Ψ(τ) = e^{(τ/2)A} · e^{τB} · e^{(τ/2)A}.
    We form the formal series in τ up to τ⁴, take log, and inspect coefficients.

    The τ² coefficient MUST be 0 (palindromic cancellation → order ≥ 2);
    the τ³ coefficient MUST be the standard NON-zero double-commutator combo
    → order EXACTLY 2 in general. Both facts are step-independent.
    """
    print("WITNESS A — free-algebra palindromic Strang order (step-independent)")

    tau = sp.symbols("tau")
    A = sp.Symbol("A", commutative=False)
    B = sp.Symbol("B", commutative=False)
    ORDER = 5  # keep τ-powers strictly below this

    def trunc(expr):
        """Series-truncate a noncommutative τ-polynomial below τ**ORDER."""
        p = sp.expand(expr)
        out = sp.S.Zero
        for k in range(ORDER):
            out += p.coeff(tau, k) * tau**k
        return sp.expand(out)

    def exp_series(M):
        """exp(M) as a τ-series truncated below τ**ORDER (M is O(τ))."""
        term = sp.S.One
        acc = sp.S.One
        for k in range(1, ORDER + 1):
            term = trunc(sp.expand(term * M)) / k
            acc = trunc(acc + term)
        return acc

    halfA = (tau / 2) * A
    fullB = tau * B
    psi = trunc(exp_series(halfA) * exp_series(fullB) * exp_series(halfA))

    # log(psi) = log(I + W),  W = psi - 1  (W is O(τ)).
    W = trunc(psi - 1)
    logpsi = sp.S.Zero
    Wp = sp.S.One
    for k in range(1, ORDER + 1):
        Wp = trunc(sp.expand(Wp * W))
        logpsi = trunc(logpsi + sp.Rational((-1) ** (k + 1), k) * Wp)
    logpsi = sp.expand(logpsi)

    c1 = sp.expand(logpsi.coeff(tau, 1))
    c2 = sp.expand(logpsi.coeff(tau, 2))
    c3 = sp.expand(logpsi.coeff(tau, 3))

    # τ¹ coefficient must equal A + B  (Chernoff S'(0) = A + B = generator).
    check("A.tau1_generator", sp.expand(c1 - (A + B)) == 0,
          f"τ¹ coeff = {c1}, expected A + B")

    # τ² coefficient must be identically zero (palindromic cancellation).
    check("A.tau2_palindrome_zero", c2 == 0,
          f"τ² coeff = {c2}, expected 0")

    # τ³ coefficient: standard symmetric-Strang combo, NON-zero in general.
    # Reference (HLW 2006 §III.5): -1/24 [A,[A,B]] + 1/12 [B,[B,A]].
    def comm(P, Q):
        return sp.expand(P * Q - Q * P)

    expected_c3 = sp.expand(
        sp.Rational(-1, 24) * comm(A, comm(A, B))
        + sp.Rational(1, 12) * comm(B, comm(B, A))
    )
    check("A.tau3_matches_HLW", sp.expand(c3 - expected_c3) == 0,
          f"τ³ coeff = {c3}, expected {expected_c3}")
    # The combo is non-zero as a free-algebra element (the two nested
    # commutators are linearly independent words), so order is exactly 2.
    check("A.tau3_nonzero", c3 != 0,
          "τ³ coeff vanished — would imply accidental order > 2 in free algebra")

    print("  → order-2 tangency proven in the free algebra on {A, B}: "
          "Strang is order-2 for ANY A, B, hence for sub-Laplacians of ANY "
          "Carnot step. The step does not enter the ORDER.\n")


# ════════════════════════════════════════════════════════════════════════════
# WITNESS B — concrete step-4 filiform N=5 group
# ════════════════════════════════════════════════════════════════════════════

x1, x2, x3, x4, x5 = sp.symbols("x1 x2 x3 x4 x5", real=True)
COORDS = (x1, x2, x3, x4, x5)

# Bratzlavsky filiform N=5 basis (Bonfiglioli §4.3.6 generalised from Engel N=4).
X1 = sp.Array([sp.S.One, sp.S.Zero, sp.S.Zero, sp.S.Zero, sp.S.Zero])
X2 = sp.Array([sp.S.Zero, sp.S.One, x1, x1**2 / 2, x1**3 / 6])
X3_exp = sp.Array([sp.S.Zero, sp.S.Zero, sp.S.One, x1, x1**2 / 2])
X4_exp = sp.Array([sp.S.Zero, sp.S.Zero, sp.S.Zero, sp.S.One, x1])
X5_exp = sp.Array([sp.S.Zero, sp.S.Zero, sp.S.Zero, sp.S.Zero, sp.S.One])


def arrays_equal(a, b) -> bool:
    if len(a) != len(b):
        return False
    return all(sp.simplify(a[i] - b[i]) == 0 for i in range(len(a)))


def is_zero_field(a) -> bool:
    return all(sp.simplify(a[i]) == 0 for i in range(len(a)))


def step4_bracket_structure() -> None:
    """Witness B-1: bracket chain + filiform termination + Hörmander rank 5."""
    print("WITNESS B-1 — step-4 filiform N=5 bracket structure")

    b12 = lie_bracket(X1, X2, COORDS)
    check("B.bracket_12", arrays_equal(b12, X3_exp),
          f"[X1,X2] = {tuple(sp.simplify(b12[i]) for i in range(5))}")

    b13 = lie_bracket(X1, X3_exp, COORDS)
    check("B.bracket_13", arrays_equal(b13, X4_exp),
          f"[X1,X3] = {tuple(sp.simplify(b13[i]) for i in range(5))}")

    b14 = lie_bracket(X1, X4_exp, COORDS)
    check("B.bracket_14", arrays_equal(b14, X5_exp),
          f"[X1,X4] = {tuple(sp.simplify(b14[i]) for i in range(5))}")

    # Filiform termination: X5 is central; [X2,X3], [X2,X4], [X1,X5]=0, etc.
    term = (
        is_zero_field(lie_bracket(X2, X3_exp, COORDS))
        and is_zero_field(lie_bracket(X2, X4_exp, COORDS))
        and is_zero_field(lie_bracket(X3_exp, X4_exp, COORDS))
        and is_zero_field(lie_bracket(X1, X5_exp, COORDS))
        and is_zero_field(lie_bracket(X2, X5_exp, COORDS))
    )
    check("B.filiform_termination", term,
          "a terminating bracket was non-zero")

    origin = {c: 0 for c in COORDS}
    rank5 = generates_T([X1, X2, X3_exp, X4_exp, X5_exp], COORDS, origin)
    check("B.hormander_rank5", rank5, "rank < 5 at origin")

    # Diagnostic: step-3 brackets alone span only 4 → genuine step-4.
    M3 = sp.Matrix([[v[i].subs(origin) for i in range(5)]
                    for v in (X1, X2, X3_exp, X4_exp)])
    r3 = M3.rank()
    check("B.genuine_step4", r3 == 4,
          f"step-3 rank = {r3}, expected 4 (else degenerate)")
    print(f"  [diagnostic] step-3 rank = {r3} (< 5 → genuine step-4)\n")


# --- 2nd-order operator action on a polynomial test jet ---------------------


def vf_apply(field, f):
    """Apply a first-order vector field (sp.Array of length 5) to scalar f."""
    return sp.expand(sum(field[i] * sp.diff(f, COORDS[i]) for i in range(5)))


def sq_apply(field, f):
    """Apply Xₖ² = Xₖ(Xₖ f) to scalar f."""
    return vf_apply(field, vf_apply(field, f))


def op_series_apply(coeff_op, f, tau, order):
    """Apply exp(τ·coeff·Op) to f as a τ-series < τ**order.

    coeff_op = (rational_coeff, field) meaning the operator is rational·Xfield².
    Returns a τ-polynomial (sympy expr) = Σ_{k<order} τ^k/k! (coeff·Xfield²)^k f.
    """
    coeff, field = coeff_op
    acc = f
    term = f
    for k in range(1, order):
        term = sp.expand(sq_apply(field, term) * coeff) / k
        acc = sp.expand(acc + tau**k * term)
    return acc


def compose_strang(f, tau, order):
    """F(τ) f = exp(τ/4 X1²) exp(τ/2 X2²) exp(τ/4 X1²) f, τ-series < τ**order.

    Applied right-to-left; each stage re-expanded and τ-truncated below τ**order.
    """
    def trunc(expr):
        p = sp.expand(expr)
        return sum(p.coeff(tau, k) * tau**k for k in range(order))

    # Innermost leg first (rightmost in composition): exp(τ/4 X1²).
    g = op_series_apply((sp.Rational(1, 4), X1), f, tau, order)
    g = trunc(g)
    # Then exp(τ/2 X2²) applied to the τ-series g: substitute g leg-by-leg.
    g = _apply_exp_to_series((sp.Rational(1, 2), X2), g, tau, order, trunc)
    # Then exp(τ/4 X1²).
    g = _apply_exp_to_series((sp.Rational(1, 4), X1), g, tau, order, trunc)
    return trunc(g)


def _apply_exp_to_series(coeff_op, series, tau, order, trunc):
    """Apply exp(τ·coeff·Xfield²) to an existing τ-series, truncating < τ**order."""
    coeff, field = coeff_op
    out = sp.S.Zero
    # series = Σ_j τ^j g_j ; operator = Σ_k τ^k/k! (coeff·X²)^k.
    gj = [sp.expand(series.coeff(tau, j)) for j in range(order)]
    for j in range(order):
        if gj[j] == 0:
            continue
        term = gj[j]
        out += tau**j * term
        for k in range(1, order - j):
            term = sp.expand(sq_apply(field, term) * coeff) / k
            out += tau ** (j + k) * term
    return trunc(out)


def exact_semigroup_apply(f, tau, order):
    """exp(τ G) f as τ-series < τ**order, G = (1/2)(X1² + X2²).

    The Strang legs exp(τ/4·X1²), exp(τ/2·X2²) have leg generators (1/2)X1²,
    (1/2)X2² (math.md §28.3 convention: diffusive legs are exp(σ·Xₖ²/2)), so the
    composition's generator is G = (1/2)(X1² + X2²) — NOT X1² + X2². The exact
    semigroup target must use G, otherwise the τ¹ coefficient mismatches by a
    factor of 2 (caught during the F4 spike: a normalization bug, fixed here).
    """
    def Lf(g):
        return sp.expand((sq_apply(X1, g) + sq_apply(X2, g)) * sp.Rational(1, 2))

    acc = f
    term = f
    for k in range(1, order):
        term = sp.expand(Lf(term)) / k
        acc = sp.expand(acc + tau**k * term)
    return acc


def step4_tangency_order() -> None:
    """Witness B-2: measure the EXACT tangency order of F(τ) vs exp(τL).

    Uses a polynomial test function and a GENERIC (non-origin) base point so the
    τ³ residual does not spuriously vanish — this is the honest measurement that
    distinguishes genuine order from origin/IC artifacts (cf. the prior Engel
    −43.95 super-exp slope, which was such an artifact).
    """
    print("WITNESS B-2 — step-4 tangency order (exact polynomial-jet measurement)")

    tau = sp.symbols("tau")
    ORDER = 5  # measure τ⁰..τ⁴

    # Generic HIGH-DEGREE monomial-rich test function. Degree must be high
    # enough that the τ³ double-commutator residual is NOT identically zero.
    # NOTE (spike finding): a too-low-degree / origin-symmetric test function
    # makes the τ³ residual spuriously vanish — this is EXACTLY the artifact
    # behind the prior Engel −43.95 "super-exponential" self-convergence slope
    # (origin-centred Gaussian IC). The honest measurement requires a degree-≥4
    # generic polynomial; otherwise the gate over-reports the order.
    f = (
        x1**4 * x2
        + x1**3 * x3
        + x1**2 * x2 * x3
        + x2**2 * x4
        + x1 * x2 * x5
        + x3 * x4 * x5
        + x1**2 * x2**2
        + x4**3
        + sp.Rational(3, 2)
    )

    strang = compose_strang(f, tau, ORDER)
    exact = exact_semigroup_apply(f, tau, ORDER)
    diff = sp.expand(strang - exact)

    # Robust measurement: first τ-power whose residual is NOT IDENTICALLY zero
    # (as a polynomial in the coordinates), not merely zero at one base point.
    first_nonzero = None
    for k in range(ORDER):
        ck = sp.expand(diff.coeff(tau, k))
        identically_zero = ck == 0
        if not identically_zero and first_nonzero is None:
            first_nonzero = (k, ck)
        tag = "≡ 0" if identically_zero else f"≠ 0  e.g. {str(ck)[:60]}"
        print(f"    τ^{k}: F−exp residual {tag}")

    if first_nonzero is None:
        check("B.tangency_order_is_2", False,
              "no residual through τ⁴ — test function too trivial (raise degree)")
        return
    k0, _ = first_nonzero
    global_order = k0 - 1  # local error O(τ^{k0}) ⇒ global order k0−1 ⇒ slope −(k0−1)
    print(f"  → first τ-mismatch at τ^{k0}  ⇒  local order {k0}, "
          f"global order {global_order}, expected slope −{global_order} "
          f"(target: mismatch at τ³, global 2, slope −2)")
    check("B.tangency_order_is_2", k0 == 3,
          f"first mismatch at τ^{k0}, expected τ³ (global order 2)")
    print()


def main() -> None:
    print("T_CARNOT_STEP4 — step-4 (filiform N=5) Carnot palindrome tangency oracle")
    print("  Source: ADR-0136 + math.md §28.bis.5 + Bonfiglioli 2007 §4.3.6")
    print()
    free_algebra_strang_order()
    step4_bracket_structure()
    step4_tangency_order()
    if FAILED:
        print("T_CARNOT_STEP4 FAIL")
        sys.exit(1)
    print("T_CARNOT_STEP4 PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
