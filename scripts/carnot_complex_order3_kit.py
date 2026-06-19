#!/usr/bin/env python3
# pyright: reportUnusedVariable=false, reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false
#
# Sympy dynamic-typing through Symbol / Array / Matrix / Poly; Pyright cannot
# trace the return types. All operations are valid sympy at runtime — verified
# by the T_CARNOT_CPLX3 PASS gate below.
"""T_CARNOT_CPLX3: complex-time order-3 splitting on step-k Carnot, feasibility oracle.

v8.0.0 F4 INVENTIVE re-spike (ADR-0136 Amendment 2 candidate). This is the
math-prerequisite oracle for the TRIZ-chosen direction:

    Escape the Sheng-Suzuki real-coefficient order barrier (any real splitting of
    order >= 3 must contain a NEGATIVE substep -> backward heat -> UNBOUNDED for
    a parabolic/hypoelliptic semigroup) by going to COMPLEX time: complex
    coefficients with POSITIVE real part give order >= 3 with BOUNDED substeps for
    ANALYTIC semigroups (Suzuki 1990; Castella-Chartier-Descombes-Vilmart 2009;
    Hansen-Ostermann 2009). The Carnot sub-Laplacian generates an analytic
    semigroup (Hormander hypoellipticity), so the escape applies.

It answers TWO crux questions with independent symbolic witnesses:

WITNESS A — free-associative-algebra order + substep boundedness (step-INDEPENDENT).
  For the symmetric 3-exponential complex Strang
        Psi(tau) = e^{a tau A} e^{tau B} e^{a' tau A}      (1 inner B leg)
  and the symmetric 5-exponential complex composition
        Psi(tau) = e^{a1 tau A} e^{b1 tau B} e^{a2 tau A} e^{b1 tau B} e^{a1 tau A}
  (Suzuki "triple-jump"-style, complexified), we form log(Psi) in the free
  algebra on {A,B} and SOLVE the order conditions through tau^3. We then check
  whether a solution exists with Re(a_j) > 0 AND Re(b_j) > 0 for ALL legs (the
  bounded-substep / analytic-admissibility condition). This is the decisive
  Sheng-Suzuki escape test.

WITNESS B — concrete step-4 filiform N=5 differential-operator tangency.
  Re-uses the §28.bis.7 filiform N=5 group. Substitutes the WITNESS-A complex
  coefficients into the actual operator composition exp(c·tau·X1^2)... and
  MEASURES the operator tangency order on a generic degree-5 polynomial jet at a
  non-origin base point (the honest probe per ADR-0136 Amendment 1 honesty note),
  confirming the free-algebra order survives on a genuine step-4 sub-Laplacian.

Prints exactly:
  T_CARNOT_CPLX3 PASS — all sub-checks pass; complex-time order-3 escape works
  T_CARNOT_CPLX3 FAIL: <msg> — first failing sub-check

Exit code: 0 on PASS, 1 on FAIL.

References:
  - M. Suzuki, "General theory of fractal path integrals...", Phys. Lett. A 146
    (1990) / J. Math. Phys. 32 (1991) — fractal/complex decompositions.
  - Castella, Chartier, Descombes, Vilmart, "Splitting methods with complex
    times for parabolic equations", BIT 49 (2009) 487-508 — order >= 3 with
    Re(coeff) > 0 for analytic semigroups.
  - Hansen, Ostermann, "High order splitting methods for analytic semigroups
    exist", BIT 49 (2009) 527-542 — existence of arbitrary-order complex splits.
  - Q. Sheng, "Solving linear PDEs by exponential splitting", IMA J. Numer.
    Anal. 9 (1989) — the order barrier (real, order>=3 => negative substep).
  - math.md §28.bis.7 (the real-coefficient order-2 obstruction this escapes).
  - scripts/carnot_step4_kit.py (the real-coefficient sibling; shares the
    filiform-N5 operator machinery).
  - scripts/lie_bracket_kit.py (reusable Lie-bracket helpers).

Usage:
    python3 scripts/carnot_complex_order3_kit.py
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
# Free-algebra BCH machinery (noncommutative τ-series in {A, B})
# ════════════════════════════════════════════════════════════════════════════

TAU = sp.symbols("tau")
A = sp.Symbol("A", commutative=False)
B = sp.Symbol("B", commutative=False)
ORDER = 6  # keep τ-powers strictly below this (measure through τ⁵)


def trunc(expr):
    """Series-truncate a noncommutative τ-polynomial below τ**ORDER."""
    p = sp.expand(expr)
    out = sp.S.Zero
    for k in range(ORDER):
        out += p.coeff(TAU, k) * TAU**k
    return sp.expand(out)


def exp_leg(coeff, op):
    """exp(coeff·τ·op) as a τ-series truncated below τ**ORDER.

    coeff may be a sympy expression (possibly complex / symbolic); op ∈ {A, B}.
    """
    m = coeff * TAU * op
    term = sp.S.One
    acc = sp.S.One
    for k in range(1, ORDER + 1):
        term = trunc(sp.expand(term * m)) / k
        acc = trunc(acc + term)
    return acc


def log_of(psi):
    """log(psi) as τ-series, psi = I + W with W = O(τ)."""
    w = trunc(psi - 1)
    out = sp.S.Zero
    wp = sp.S.One
    for k in range(1, ORDER + 1):
        wp = trunc(sp.expand(wp * w))
        out = trunc(out + sp.Rational((-1) ** (k + 1), k) * wp)
    return sp.expand(out)


def comm(p, q):
    return sp.expand(p * q - q * p)


def bch_coeffs(psi):
    """Return (c1, c2, c3) = τ¹,τ²,τ³ coefficients of log(psi) in free algebra."""
    lg = log_of(psi)
    return (
        sp.expand(lg.coeff(TAU, 1)),
        sp.expand(lg.coeff(TAU, 2)),
        sp.expand(lg.coeff(TAU, 3)),
    )


# The independent free-Lie words at degree 1..3 in {A,B}, used as a basis to
# read off order conditions (each condition = a coefficient of one word = 0).
WORDS = {
    "A": A,
    "B": B,
    "AAB": comm(A, comm(A, B)),   # [A,[A,B]]
    "BBA": comm(B, comm(B, A)),   # [B,[B,A]]
}


def word_coeff(expr, word_label):
    """Extract the scalar coefficient of a given free-Lie word in `expr`.

    `expr` is assumed to be a real/complex linear combination of A, B,
    [A,[A,B]], [B,[B,A]] (the only words that survive in log of a symmetric or
    near-symmetric product through τ³ when A,B are abstract). We project by
    matching monomials. To make this robust we substitute the abstract A,B with
    two generic non-commuting 2×2-like symbolic matrices is overkill; instead we
    use the fact that {AAB, BBA, ABA-type} words are linearly independent and
    read coefficients by pattern via sympy's noncommutative monomial collection.
    """
    # Expand into noncommutative monomials and collect coefficients of the
    # canonical monomials that appear in each target word.
    e = sp.expand(expr)
    # Represent each target word in its expanded monomial form, pick a SIGNATURE
    # monomial unique to that word, and read its coefficient.
    signatures = {
        "A": A,
        "B": B,
        # [A,[A,B]] = A·A·B − 2·A·B·A + B·A·A ; signature monomial A*A*B
        "AAB": A * A * B,
        # [B,[B,A]] = B·B·A − 2·B·A·B + A·B·B ; signature monomial B*B*A
        "BBA": B * B * A,
    }
    sig = signatures[word_label]
    # coefficient of the signature noncommutative monomial
    return e.coeff(sig)


# ════════════════════════════════════════════════════════════════════════════
# WITNESS A — free-algebra order + substep boundedness (the Sheng-Suzuki escape)
# ════════════════════════════════════════════════════════════════════════════


def symmetric_triple_complex():
    """Symmetric 3-exp complex Strang  e^{aτA} e^{τB} e^{aτA}.

    Order conditions through τ³ for matching e^{τ(A+B)}:
      τ¹ :  coeff(A) = 2a must = 1  →  a = 1/2  (then coeff(B)=1 automatically)
      τ² :  vanishes by symmetry for ANY a (palindrome) — no condition.
      τ³ :  the [A,[A,B]] and [B,[B,A]] coefficients are FIXED once a=1/2; this
            is exactly the real palindromic Strang. So the 3-exp symmetric form
            has NO free complex parameter to kill τ³ — it is order 2 only.
    This sub-check DOCUMENTS that the minimal symmetric triple cannot reach
    order 3 (confirming we genuinely need MORE stages), i.e. it reproduces the
    §28.bis.7 obstruction and shows the escape needs extra legs.
    """
    print("WITNESS A-1 — symmetric 3-exp complex Strang has NO free τ³ knob")
    a = sp.symbols("a")  # complex-allowed symbol
    psi = trunc(exp_leg(a, A) * exp_leg(sp.Integer(1), B) * exp_leg(a, A))
    c1, c2, c3 = bch_coeffs(psi)

    cA = word_coeff(c1, "A")
    # consistency: τ¹ must be A+B  ⇒  2a=1 and coeff(B)=1
    sol_a = sp.solve(sp.Eq(cA, 1), a)
    check("A1.tau1_fixes_a", sol_a == [sp.Rational(1, 2)],
          f"coeff(A)=2a; solving 2a=1 gave {sol_a}")
    a_val = sp.Rational(1, 2)
    # τ² palindrome-zero for any a
    check("A1.tau2_zero", sp.expand(c2.subs(a, a_val)) == 0,
          f"τ² = {sp.expand(c2.subs(a, a_val))}")
    # τ³ with a=1/2 is the fixed nonzero Strang combo (no free param) → order 2.
    c3v = sp.expand(c3.subs(a, a_val))
    waab = word_coeff(c3v, "AAB")
    wbba = word_coeff(c3v, "BBA")
    check("A1.tau3_is_fixed_strang",
          sp.simplify(waab - sp.Rational(-1, 24)) == 0
          and sp.simplify(wbba - sp.Rational(1, 12)) == 0,
          f"τ³ coeffs (AAB,BBA)=({waab},{wbba}) — expected (-1/24, 1/12)")
    print("  → minimal symmetric triple is order-2 only; need extra legs.\n")


def symmetric_five_complex():
    """Symmetric 5-exp complex composition (Suzuki 'triple-jump', complexified).

        Psi(τ) = e^{a1 τA} e^{b1 τB} e^{a2 τA} e^{b1 τB} e^{a1 τA}

    Free complex parameters: a1, a2, b1 (palindromic ⇒ b's mirror, a's mirror).
    Consistency (τ¹):  2a1 + a2 = 1   (coeff A),   2b1 = 1  (coeff B) ⇒ b1=1/2.
    τ² : vanishes by palindromic symmetry for ANY parameters.
    τ³ : TWO order-3 conditions — coeff[A,[A,B]] = 0 and coeff[B,[B,A]] = 0.

    With b1=1/2 fixed and a2 = 1 − 2a1, we have ONE remaining free complex
    parameter a1 but TWO τ³ conditions. The classical REAL Suzuki solution
    a1 = 1/(2 − 2^{1/3}) makes BOTH vanish — but a2 = 1 − 2a1 < 0 (the negative/
    backward substep: the Sheng-Suzuki barrier). The question: is there a COMPLEX
    a1 making both τ³ conditions vanish with Re(a1) > 0 AND Re(a2) > 0?

    Because the symmetric-5 palindrome's τ³ conditions are NOT independent for a
    single complex a1 (one complex equation), we instead solve the canonical
    pair of τ³ order conditions and report the resulting (a1, a2) and their real
    parts. This is the Sheng-Suzuki escape measurement.
    """
    print("WITNESS A-2 — symmetric 5-exp complex composition: solve τ³, check Re>0")
    a1, a2, b1 = sp.symbols("a1 a2 b1")
    psi = trunc(
        exp_leg(a1, A) * exp_leg(b1, B) * exp_leg(a2, A)
        * exp_leg(b1, B) * exp_leg(a1, A)
    )
    c1, c2, c3 = bch_coeffs(psi)

    # τ¹ consistency.
    # coeff(A) must be 2a1 + a2 = 1; coeff(B) must be 2b1 = 1.
    check("A2.tau1_A", sp.simplify(word_coeff(c1, "A") - (2 * a1 + a2)) == 0,
          f"coeff(A) = {word_coeff(c1,'A')}")
    check("A2.tau1_B", sp.simplify(word_coeff(c1, "B") - (2 * b1)) == 0,
          f"coeff(B) = {word_coeff(c1,'B')}")

    # τ² palindrome-zero.
    c2s = sp.expand(c2.subs(b1, sp.Rational(1, 2)))
    check("A2.tau2_zero", c2s == 0, f"τ² = {c2s}")

    # τ³ order conditions (the two nested-commutator words). Substitute
    # b1 = 1/2 and a2 = 1 − 2a1, leaving ONE complex unknown a1.
    subs0 = {b1: sp.Rational(1, 2), a2: 1 - 2 * a1}
    c3s = sp.expand(c3.subs(subs0))
    cond_AAB = sp.expand(word_coeff(c3s, "AAB"))
    cond_BBA = sp.expand(word_coeff(c3s, "BBA"))
    print(f"    τ³ condition coeff[A,[A,B]](a1) = {sp.nsimplify(cond_AAB)}")
    print(f"    τ³ condition coeff[B,[B,A]](a1) = {sp.nsimplify(cond_BBA)}")

    # Solve each τ³ condition for a1 (complex roots allowed).
    roots_AAB = sp.solve(sp.Eq(cond_AAB, 0), a1)
    roots_BBA = sp.solve(sp.Eq(cond_BBA, 0), a1)
    print(f"    roots of coeff[A,[A,B]]=0 : {roots_AAB}")
    print(f"    roots of coeff[B,[B,A]]=0 : {roots_BBA}")

    # KEY FINDING: the two τ³ words give INDEPENDENT conditions on the single
    # complex unknown a1 — there is NO common root.
    #     coeff[A,[A,B]] = a1²/2 − a1/2 + 1/12 = 0  ⇒  a1 = 1/2 ± √3/6
    #     coeff[B,[B,A]] = a1/4 − 1/24       = 0  ⇒  a1 = 1/6
    # These are inconsistent. So the SYMMETRIC-5 form with ONE free parameter
    # cannot reach order 3 at all (real OR complex) — it is structurally
    # order-2. This is the well-known fact that the symmetric-5 Yoshida split
    # needs the constrained TRIPLE-JUMP relation a2 = 1 − 2a1 PLUS the cubic
    # 2a1³ + a2³ = 0, which simultaneously satisfies BOTH τ³ words. We test that
    # below in WITNESS A-3 (the correct construction). Here we only DOCUMENT the
    # no-common-root obstruction.
    common = sp.solve([sp.Eq(cond_AAB, 0), sp.Eq(cond_BBA, 0)], a1)
    print(f"    common root of BOTH τ³ conditions: {common}")
    check("A2.symmetric5_one_param_has_no_order3_root", len(common) == 0,
          f"unexpected common root {common} — would contradict the known "
          f"structural order-2 of the 1-parameter symmetric-5 form")
    print("  → 1-parameter symmetric-5 is order-2; order-3 needs the\n"
          "    triple-jump relation (WITNESS A-3).\n")


def ccdv_two_stage_complex():
    """CCDV-style minimal complex order-3: 2 complex-conjugate B-legs.

    Castella-Chartier-Descombes-Vilmart 2009 show the MINIMAL order-3 splitting
    with Re(coeff)>0 uses complex coefficients. The simplest such on a 2-operator
    split A,B is the 'complex Strang triple-jump':

        Psi(τ) = e^{γ τ L} e^{(1-2γ) τ L} e^{γ τ L}   (single-operator triple)

    with COMPLEX γ = 1/2 + i/(2√3) (so that 2γ³ + (1-2γ)³ = 0 with Re(γ)>0 and
    Re(1-2γ)=0 ... ) — the canonical complex triple-jump that replaces the real
    Suzuki/Yoshida γ_real = 1/(2-2^{1/3}) (which forces a negative middle step).

    For a TWO-operator split we apply the complex triple-jump to the OUTER
    Strang map S(τ)=e^{τA/2}e^{τB}e^{τA/2} (itself order 2, symmetric):

        Psi(τ) = S(γτ) ∘ S((1-2γ)τ) ∘ S(γτ),    γ complex, Re(γ)>0.

    Because S is symmetric order-2, the triple-jump with the order-3 γ-condition
    2γ³+(1-2γ)³=0 lifts it to order 3. The Sheng-Suzuki escape: pick the COMPLEX
    root of 2γ³+(1-2γ)³=0 with Re(γ)>0 AND Re(1-2γ)>0, so NO sub-map runs
    backward in time. This sub-check verifies (i) the γ-cubic, (ii) a complex
    root with Re>0 on BOTH the γ and (1-2γ) maps exists, (iii) the resulting
    composition is order 3 in the free algebra.
    """
    print("WITNESS A-3 — complex triple-jump on the order-2 symmetric Strang map")

    gamma = sp.symbols("gamma")

    def strang(scale):
        """Symmetric order-2 Strang map S(scale·τ) = e^{sτA/2}e^{sτB}e^{sτA/2}."""
        return trunc(
            exp_leg(scale / 2, A) * exp_leg(scale, B) * exp_leg(scale / 2, A)
        )

    # Triple-jump composition (note: composition order is left-to-right product).
    psi = trunc(strang(gamma) * strang(1 - 2 * gamma) * strang(gamma))
    c1, c2, c3 = bch_coeffs(psi)

    # τ¹ consistency: total scale = γ + (1-2γ) + γ = 1.  coeff(A)=coeff(B)=1.
    check("A3.tau1_A", sp.simplify(word_coeff(c1, "A") - 1) == 0,
          f"coeff(A) = {word_coeff(c1,'A')}")
    check("A3.tau1_B", sp.simplify(word_coeff(c1, "B") - 1) == 0,
          f"coeff(B) = {word_coeff(c1,'B')}")

    # τ² vanishes (each S symmetric ⇒ whole map symmetric ⇒ palindrome).
    check("A3.tau2_zero", sp.expand(c2) == 0, f"τ² = {sp.expand(c2)}")

    # τ³ order-3 condition. For a triple-jump of a symmetric order-2 method the
    # τ³ error scales as (2γ³ + (1-2γ)³)·E₃ where E₃ is the (fixed, non-zero)
    # τ³ defect of S. So the order-3 condition is the scalar cubic:
    #     2γ³ + (1-2γ)³ = 0.
    cubic = sp.expand(2 * gamma**3 + (1 - 2 * gamma) ** 3)
    # Verify the τ³ free-algebra coefficients are proportional to this cubic.
    waab = sp.expand(word_coeff(c3, "AAB"))
    wbba = sp.expand(word_coeff(c3, "BBA"))
    # Each must be (cubic)·(constant). Check the ratio is γ-independent.
    prop_ok = True
    for w in (waab, wbba):
        if sp.simplify(w) == 0:
            continue
        ratio = sp.simplify(w / cubic)
        if ratio.has(gamma):
            prop_ok = False
    check("A3.tau3_proportional_to_cubic", prop_ok,
          f"τ³ words not ∝ (2γ³+(1-2γ)³): AAB={waab}, BBA={wbba}")

    roots = sp.solve(sp.Eq(cubic, 0), gamma)
    print(f"    γ-cubic 2γ³+(1-2γ)³=0 roots: {[complex(sp.N(r)) for r in roots]}")

    admissible = []
    for r in roots:
        gv = complex(sp.N(r))
        mv = complex(sp.N(1 - 2 * r))
        re_g, re_m = gv.real, mv.real
        ok = (re_g > 1e-12) and (re_m > 1e-12)
        tag = "ADMISSIBLE (Re>0 on γ AND 1-2γ maps)" if ok \
            else "rejected (a sub-map runs backward, Re<=0)"
        print(f"      γ={gv:.6g}  1-2γ={mv:.6g}  Re(γ)={re_g:.6g} "
              f"Re(1-2γ)={re_m:.6g}  → {tag}")
        if ok:
            admissible.append(r)

    check("A3.complex_admissible_root_exists", len(admissible) > 0,
          "no complex γ kills τ³ with Re>0 on both maps — escape FAILED")

    if admissible:
        g_star = admissible[0]
        # Verify order-3: substitute and check both τ³ words vanish.
        c3v = sp.expand(c3.subs(gamma, g_star))
        waabv = complex(sp.N(word_coeff(c3v, "AAB")))
        wbbav = complex(sp.N(word_coeff(c3v, "BBA")))
        order3 = abs(waabv) < 1e-9 and abs(wbbav) < 1e-9
        check("A3.tau3_cancels_order3", order3,
              f"τ³ residual: AAB={waabv:.3e}, BBA={wbbav:.3e}")

        # DECISIVE even-order bonus: the whole triple-jump is SYMMETRIC
        # (palindromic) because each S is symmetric and the scale sequence
        # (γ, 1−2γ, γ) is palindromic. A symmetric method of order 3 is
        # AUTOMATICALLY order 4 (the τ⁴ term — an even power — vanishes by the
        # log(Ψ(τ))=−log(Ψ(−τ)) odd-function identity, HLW 2006 §III.5). Verify
        # the FULL τ⁴ free-algebra coefficient vanishes at γ*.
        c4 = sp.expand(log_of(psi).coeff(TAU, 4))
        c4v = sp.expand(c4.subs(gamma, g_star))
        # numeric magnitude of every surviving monomial coefficient
        c4_zero = sp.simplify(c4v) == 0
        if not c4_zero:
            # fall back to numeric: substitute generic 2×2 matrices to test
            c4_zero = abs(complex(sp.N(
                c4v.subs({A: sp.Symbol("a"), B: sp.Symbol("b")}, simultaneous=True)
                if False else c4v.coeff(comm(A, comm(A, comm(A, B))))
            ))) < 1e-9 if c4v.coeff(comm(A, comm(A, comm(A, B)))) != 0 else True
        check("A3.tau4_vanishes_symmetric_order4", sp.simplify(c4v) == 0,
              f"τ⁴ free-algebra coeff ≠ 0 at γ*: {c4v}")

        globals()["_GAMMA_STAR"] = g_star
        print(f"  → chosen complex γ* = {complex(sp.N(g_star)):.10g} "
              f"(Re={complex(sp.N(g_star)).real:.6g} > 0); "
              f"1-2γ* = {complex(sp.N(1-2*g_star)):.10g} "
              f"(Re={complex(sp.N(1-2*g_star)).real:.6g} > 0)")
        print("  → SHENG-SUZUKI BARRIER ESCAPED: order-3 conditions met with ALL "
              "substeps forward-in-time (Re>0).")
        print("  → SYMMETRY BONUS: τ⁴ also vanishes ⇒ the symmetric complex "
              "triple-jump is order 4 (even-order theorem).\n")
    else:
        globals()["_GAMMA_STAR"] = None


# ════════════════════════════════════════════════════════════════════════════
# WITNESS B — concrete step-4 filiform N=5 operator tangency with complex legs
# ════════════════════════════════════════════════════════════════════════════

x1, x2, x3, x4, x5 = sp.symbols("x1 x2 x3 x4 x5", real=True)
COORDS = (x1, x2, x3, x4, x5)

X1 = sp.Array([sp.S.One, sp.S.Zero, sp.S.Zero, sp.S.Zero, sp.S.Zero])
X2 = sp.Array([sp.S.Zero, sp.S.One, x1, x1**2 / 2, x1**3 / 6])


def vf_apply(field, f):
    return sp.expand(sum(field[i] * sp.diff(f, COORDS[i]) for i in range(5)))


def sq_apply(field, f):
    return vf_apply(field, vf_apply(field, f))


def exp_op_to_series(coeff, field, series, tau, order):
    """Apply exp(coeff·τ·field²) to an existing τ-series, truncate < τ**order.

    coeff may be COMPLEX (sympy I allowed). field is X1 or X2.
    """
    gj = [sp.expand(series.coeff(tau, j)) for j in range(order)]
    out = sp.S.Zero
    for j in range(order):
        if gj[j] == 0:
            continue
        term = gj[j]
        out += tau**j * term
        for k in range(1, order - j):
            term = sp.expand(sq_apply(field, term) * coeff) / k
            out += tau ** (j + k) * term
    p = sp.expand(out)
    return sum(p.coeff(tau, k) * tau**k for k in range(order))


def exact_semigroup(f, tau, order, gen_scale):
    """exp(gen_scale·τ·(X1²+X2²)) f as τ-series < τ**order."""
    def Lf(g):
        return sp.expand((sq_apply(X1, g) + sq_apply(X2, g)) * gen_scale)
    acc, term = f, f
    for k in range(1, order):
        term = sp.expand(Lf(term)) / k
        acc = sp.expand(acc + tau**k * term)
    return acc


def filiform5_complex_triple_tangency():
    """Witness B: complex triple-jump on filiform-N5 Strang, measure order.

    Builds Psi(τ) = S(γτ)∘S((1-2γ)τ)∘S(γτ) with the WITNESS-A γ* substituted as
    a CONCRETE complex number, S(s·τ)=exp(sτX1²/4)∘exp(sτX2²/2)∘exp(sτX1²/4) the
    filiform-N5 horizontal Strang, and measures the operator tangency order of
    Psi(τ)f − exp((τ/2)L₅)f on a generic degree-5 jet at a non-origin base point.
    Expected: first mismatch at τ⁴ (local order 4 ⇒ global order 3 ⇒ slope −3).
    """
    print("WITNESS B — filiform N=5 complex triple-jump operator tangency")

    # First confirm genuine step-4 (reuse the real-kit structure check).
    X3 = lie_bracket(X1, X2, COORDS)
    X4 = lie_bracket(X1, X3, COORDS)
    X5 = lie_bracket(X1, X4, COORDS)
    origin = {c: 0 for c in COORDS}
    rank5 = generates_T([X1, X2, X3, X4, X5], COORDS, origin)
    check("B.hormander_rank5", rank5, "filiform N=5 not rank 5 at origin")

    g_star = globals().get("_GAMMA_STAR", None)
    if g_star is None:
        check("B.have_gamma", False, "WITNESS A-3 produced no admissible γ*")
        return
    gamma = sp.nsimplify(g_star)  # keep exact complex algebraic number
    one_m_2g = 1 - 2 * gamma

    tau = sp.symbols("tau_b")
    ORD = 6  # measure τ⁰..τ⁵ (need τ³ and τ⁴ both non-trivial)

    def strang_op(scale, f):
        """S(scale·τ) f = exp(sτX1²/4) exp(sτX2²/2) exp(sτX1²/4) f."""
        g = exp_op_to_series(scale * sp.Rational(1, 4), X1, f, tau, ORD)
        g = exp_op_to_series(scale * sp.Rational(1, 2), X2, g, tau, ORD)
        g = exp_op_to_series(scale * sp.Rational(1, 4), X1, g, tau, ORD)
        return g

    # HONEST PROBE (critical). L=X1²+X2² is 2nd-order ⇒ each L-power LOWERS
    # polynomial degree by 2. To make BOTH the τ³ and τ⁴ residual coefficients
    # non-trivial we need degree ≥ 2·5 = 10 so that L⁵f ≠ 0. A degree-5 jet
    # (used in the real order-2 kit) makes L³f=0 ⇒ residual ≡0 past τ³ for ANY
    # method — the exact "probe too trivial" artifact of ADR-0136 Amdt 1. Use a
    # generic, non-origin, anisotropic degree-10 polynomial mixing all coords.
    f = (
        x1**6 * x2**2 * x3 + x1**5 * x2 * x3 * x4 + x1**4 * x2**2 * x4**2
        + x1**3 * x2**3 * x5 + x2**4 * x3**2 + x1**2 * x3**2 * x4 * x5
        + x1**4 * x4 * x5 + x2**2 * x3 * x4 * x5 + x1 * x2 * x3 * x4 * x5
        + x1**3 * x2 + x4 * x5 + sp.Rational(7, 5)
    )

    def measure_order(label, compose_fn):
        """Compose, diff vs exact, report first non-trivial τ-mismatch."""
        psi = compose_fn(f)
        psi = sum(sp.expand(psi).coeff(tau, k) * tau**k for k in range(ORD))
        exact = exact_semigroup(f, tau, ORD, sp.Rational(1, 2))
        diff = sp.expand(psi - exact)
        first = None
        for k in range(ORD):
            ck = sp.simplify(sp.expand(diff.coeff(tau, k)))
            zero = ck == 0
            if not zero and first is None:
                first = k
            print(f"    [{label}] τ^{k}: residual {'≡ 0' if zero else '≠ 0'}")
        return first

    # Control: REAL flat palindromic Strang on the same degree-10 jet must give
    # first mismatch at τ³ (global order 2) — reproduces §28.bis.7 honestly.
    real_first = measure_order(
        "real-Strang",
        lambda ff: strang_op(sp.Integer(1), ff),
    )
    print(f"    [real-Strang] first mismatch at τ^{real_first} "
          f"(expect τ³ ⇒ order 2)")
    check("B.real_control_is_order2", real_first == 3,
          f"real Strang first mismatch τ^{real_first}, expected τ³ "
          f"(probe may be degenerate if ≠3)")

    # Complex triple-jump: innermost first (right), then middle, then outer.
    def triple(ff):
        g = strang_op(gamma, ff)
        g = strang_op(one_m_2g, g)
        g = strang_op(gamma, g)
        return g

    first_nonzero = measure_order("cplx-triple", triple)
    if first_nonzero is None:
        check("B.tangency_order_ge_3", False,
              "no residual through τ⁵ — probe STILL too trivial (raise degree)")
        return
    global_order = first_nonzero - 1
    print(f"  → complex triple-jump: first τ-mismatch at τ^{first_nonzero} ⇒ "
          f"local order {first_nonzero}, global order {global_order}, slope "
          f"−{global_order}")
    print("    (symmetric triple-jump is order 4 by the even-order theorem ⇒ "
          "first mismatch at τ⁵, global order 4, slope −4)")
    # STRONG result: order ≥ 3 is the open-problem threshold; the symmetric
    # construction over-achieves at order 4. Accept ≥ 4 (τ⁵ mismatch) as the
    # canonical outcome; ≥ τ⁴ mismatch (order ≥ 3) is the minimum success bar.
    check("B.tangency_order_ge_3", first_nonzero >= 4,
          f"first mismatch at τ^{first_nonzero}; need ≥ τ⁴ (global order ≥ 3)")
    check("B.tangency_order_is_4", first_nonzero == 5,
          f"first mismatch at τ^{first_nonzero}; symmetric triple-jump should "
          f"give τ⁵ (global order 4)")
    print()


def main() -> None:
    print("T_CARNOT_CPLX3 — complex-time order-3 splitting on step-k Carnot")
    print("  Source: ADR-0136 Amdt 2 candidate + math.md §28.bis.8 (proposed)")
    print("  TRIZ escape of the Sheng-Suzuki real-coefficient order barrier")
    print()
    symmetric_triple_complex()
    symmetric_five_complex()
    ccdv_two_stage_complex()
    filiform5_complex_triple_tangency()
    if FAILED:
        print("T_CARNOT_CPLX3 FAIL")
        sys.exit(1)
    print("T_CARNOT_CPLX3 PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
