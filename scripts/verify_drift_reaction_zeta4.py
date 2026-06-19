#!/usr/bin/env python3
"""PRE-FLIGHT oracle for the order-2 ζ⁴ drift-reaction sibling (v7.0.0 item #18, ADR-0131).

§27.7 (v3.0 framing) called the ζ⁴ τ²-correction WITH drift an OPEN math problem:
"the τ²-coefficient with drift introduces additional monomials beyond the diffusion
case (Galkin-Remizov §3.3)". That framing belongs to the RETIRED 6-monomial P_2[A]
correction-operator approach.

CRITICAL RE-GROUNDING (math §27 AMENDMENT + AMENDMENT 1/2, NORMATIVE for v4.1+):
The §27 framework PIVOTED away from correction-monomials entirely. The current
NORMATIVE ζ⁴ algorithm is **Path β = Richardson extrapolation over the symmetric K5
order-2 base** (no P_2[A] polynomial, no monomial table at all):

    F_beta(tau) = (4/3) K5(tau/2)^2 - (1/3) K5(tau)

This promotes a SYMMETRIC (time-reversible) order-2 Chernoff base K5 to global order-4
by cancelling the leading tau^3 global-error term (only odd tau-powers survive in a
symmetric scheme's error expansion). The "extra monomials" obstruction therefore
DISSOLVES: Richardson is BLIND to which operator K5 approximates — it only needs K5 to
be (i) order-2 consistent and (ii) time-symmetric (palindromic), so that its global
error expansion is odd-in-tau.

This PRE-FLIGHT verifies the drift-reaction case reduces to "is there a SYMMETRIC
order-2 base for L = a u'' + b u' + c u?" — YES: the palindromic Strang split

    S(tau) = exp(tau R/2) . D(tau) . exp(tau R/2)         (R = b d_x + c, D = a d_xx)

is time-symmetric and order-2, so Richardson over S gives order-4 exactly as for pure
diffusion. The sibling is `DriftReactionZeta4Chernoff` = Richardson-over-(symmetric
Strang drift-reaction base). NO new monomials. The §27.7 OPEN is RESOLVED, not deferred.

Sub-checks:
  (1) symmetry: a palindromic Strang split S(tau)=e^{tau R/2} D(tau) e^{tau R/2} of the
      drift-reaction generator has an ODD-in-tau global error expansion (leading error
      is tau^3, no tau^2 term), the precondition Richardson needs. Verified on the
      scalar-symbol (Lie-algebra) model where R,D are non-commuting symbols.
  (2) Richardson cancels tau^3: (4/3) S(tau/2)^2 - (1/3) S(tau) has leading error tau^5
      => global order 4 (pair-slope ~ -4). Verified symbolically in the BCH expansion.
  (3) the "extra drift monomials" of the OLD framing are exactly the tau^2 BCH
      commutator [R,D] terms that the SYMMETRIC split already annihilates at tau^2 and
      Richardson removes at tau^3 — i.e. they are NOT an obstruction under Path β.

Gate G_DR_ZETA4_TRUTHFUL_ORDER: pair-slope <= -3.5 (numeric, Rust). This is the
SYMBOLIC PRE-FLIGHT proving the order-4 tangency of the construction. GO if all pass.

Run: python3 scripts/verify_drift_reaction_zeta4.py    Exit 0 = PASS (GO), 1 = FAIL.
"""
import sys

try:
    import sympy as sp
except ImportError:
    print("verify_drift_reaction_zeta4 SKIP (sympy not available)")
    sys.exit(0)


def bch_expand(expr, tau, order):
    """Series of a non-commutative product of matrix exponentials, truncated at tau^order."""
    return sp.expand(sp.series(expr, tau, 0, order + 1).removeO())


def make_noncommuting_model():
    """Faithful finite-dim model of non-commuting D (diffusion) and R (drift+reaction).

    We use small matrices with [D,R] != 0 so the BCH structure is exercised exactly.
    D ~ second-difference (diffusion), R ~ first-difference + diagonal (drift+reaction).
    """
    D = sp.Matrix([[-2, 1, 0], [1, -2, 1], [0, 1, -2]])          # symmetric (diffusion)
    R = sp.Matrix([[1, 1, 0], [-1, 1, 1], [0, -1, 1]])          # nonsym (drift) + diag (reaction)
    # ensure non-commuting
    assert (D * R - R * D) != sp.zeros(3, 3)
    return D, R


def matexp_series(Mat, t, order):
    """Truncated matrix exponential exp(t*Mat) as a polynomial matrix in t."""
    I = sp.eye(Mat.rows)
    acc = I.copy()
    term = I.copy()
    for k in range(1, order + 1):
        term = term * Mat * t / k
        acc = acc + term
    return acc


def check_symmetric_split_odd_error(_):
    """Sub-check 1: palindromic Strang S = e^{tR/2} e^{tD} e^{tR/2} has NO tau^2 error.

    The exact semigroup is exp(t(D+R)). The Strang split error E(t)=S(t)-exp(t(D+R))
    must start at t^3 (symmetric splits kill the t^2 BCH commutator).
    """
    t = sp.symbols("t")
    D, R = make_noncommuting_model()
    order = 4
    half = matexp_series(R, t / 2, order)
    mid = matexp_series(D, t, order)
    S = sp.expand(half * mid * half)
    exact = matexp_series(D + R, t, order)
    E = sp.expand(S - exact)
    # truncate each entry to a t-polynomial and read coefficients
    c2 = E.applyfunc(lambda e: sp.expand(e).coeff(t, 2))
    c3 = E.applyfunc(lambda e: sp.expand(e).coeff(t, 3))
    if c2 != sp.zeros(3, 3):
        return False, f"Strang split has nonzero tau^2 error: {c2}"
    if c3 == sp.zeros(3, 3):
        return False, "Strang split tau^3 error vanished too (model degenerate)"
    return True, "palindromic Strang(drift-reaction): tau^2 error = 0, tau^3 != 0 (order-2 symmetric)"


def check_richardson_promotes_to_order4(_):
    """Sub-check 2: (4/3) S(t/2)^2 - (1/3) S(t) has error starting at t^5 (order 4)."""
    t = sp.symbols("t")
    D, R = make_noncommuting_model()
    order = 6

    def strang(tt):
        half = matexp_series(R, tt / 2, order)
        mid = matexp_series(D, tt, order)
        return sp.expand(half * mid * half)

    S_half = strang(t / 2)
    S_full = strang(t)
    F = sp.expand(sp.Rational(4, 3) * (S_half * S_half) - sp.Rational(1, 3) * S_full)
    exact = matexp_series(D + R, t, order)
    E = sp.expand(F - exact)
    for k in (3, 4):
        ck = E.applyfunc(lambda e: sp.expand(e).coeff(t, k))
        if ck != sp.zeros(3, 3):
            return False, f"Richardson F has nonzero tau^{k} error (expected order-4): {ck}"
    c5 = E.applyfunc(lambda e: sp.expand(e).coeff(t, 5))
    if c5 == sp.zeros(3, 3):
        return False, "Richardson tau^5 error vanished too (model degenerate / order>4?)"
    return True, "Richardson over symmetric drift-reaction Strang: tau^3=tau^4=0, tau^5!=0 (order-4)"


def check_old_monomials_are_bch_terms(_):
    """Sub-check 3: the OLD-framing 'extra drift monomials' = the tau^2 BCH [R,D] terms.

    The retired §27 framing's obstruction (drift introduces extra τ²-monomials) is
    precisely the commutator content [R,D] that a NON-symmetric (Lie-Trotter) split
    leaves at tau^2. Verify: Lie-Trotter e^{tR} e^{tD} HAS a nonzero tau^2 term equal to
    (1/2)[R,D]+... , while the SYMMETRIC split (sub-check 1) does NOT — so the 'extra
    monomials' are an artifact of asymmetry, removed for free by Path β's symmetric base.
    """
    t = sp.symbols("t")
    D, R = make_noncommuting_model()
    order = 3
    LT = sp.expand(matexp_series(R, t, order) * matexp_series(D, t, order))
    exact = matexp_series(D + R, t, order)
    E = sp.expand(LT - exact)
    c2 = E.applyfunc(lambda e: sp.expand(e).coeff(t, 2))
    # Lie-Trotter tau^2 error must equal (1/2)(RD + DR ... ) - actually (1/2)[R,D] structure
    comm = sp.expand(R * D - D * R)
    # The LT tau^2 error is (1/2)(R^2 + 2RD + D^2) - (1/2)(R+D)^2 = (1/2)(RD - DR) = (1/2)[R,D]
    expected = sp.Rational(1, 2) * comm
    if sp.expand(c2 - expected) != sp.zeros(3, 3):
        return False, f"LT tau^2 error != (1/2)[R,D]: got {c2}"
    if comm == sp.zeros(3, 3):
        return False, "model commutator zero (degenerate)"
    return True, "OLD 'extra drift monomials' == (1/2)[R,D] BCH term — asymmetry artifact, killed by symmetric Path-β base"


def main():
    checks = [
        ("symmetric_split_odd_error", check_symmetric_split_odd_error),
        ("richardson_order4", check_richardson_promotes_to_order4),
        ("old_monomials_are_bch", check_old_monomials_are_bch_terms),
    ]
    names = []
    for name, fn in checks:
        ok, msg = fn(None)
        if not ok:
            print(f"G_DR_ZETA4_TRUTHFUL_ORDER FAIL [{name}]: {msg}")
            return 1
        names.append(name)
        print(f"  [{name}] {msg}")
    print(f"G_DR_ZETA4_TRUTHFUL_ORDER PASS ({len(names)}/{len(names)} sub-checks: {' / '.join(names)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
