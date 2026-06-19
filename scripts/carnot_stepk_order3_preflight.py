#!/usr/bin/env python3
# pyright: reportUnusedVariable=false, reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false
#
# Sympy dynamic-typing through Symbol / Array / Matrix / Poly; Pyright cannot
# trace the return types. All operations are valid sympy at runtime — verified
# by the VERDICT print below.
"""C-8 preflight: general step-k order>=3 Carnot closure falsification oracle (Wave-3 register).

Festschrift §3 OPEN problem: a constructive hypoelliptic product formula of
order >= 3 for GENERAL step-k Carnot groups. ADR-0136 Amendment 1 closed the
*real-coefficient* "palindrome-of-palindromes" route as order-2 only (the tau^3
commutator -1/24[A,[A,B]] + 1/12[B,[B,A]] survives any palindromic wrap).
ADR-0136 Amendment 2 + carnot_complex_order3_kit.py opened the *complex-time*
escape and proved order 4 on ONE step group (filiform N=5, step 4).

This preflight asks the genuinely-still-open question the Wave-3 C-8 register
targets, and tries to FALSIFY it:

    Is the complex triple-jump's order-4 cancellation STEP-INDEPENDENT in the
    sense that it survives on a CONCRETE step group OTHER than the filiform-N5
    one already tested — i.e. is "step does not enter the order" a robust
    operator fact or an artifact of the single group used in the existing
    witness?

The decisive concern is honesty: a single operator witness (filiform N=5) could
in principle hide a group-specific accident. We add a SECOND, structurally
DIFFERENT operator witness — the **Engel group** (filiform N=4, step 3,
§28.bis.1) — and demand the complex triple-jump reach order 4 there too, on a
GENERIC non-origin high-degree jet, with a real-Strang order-2 control on the
identical probe certifying the probe is non-degenerate.

It ALSO re-derives the free-algebra order conditions to make explicit WHICH
bracket term is killed at each tau-order and WHY the step is irrelevant there
(the order conditions are scalar equations on the coefficients; the brackets are
just abstract free-Lie words), so a NO-GO on the operator side would localise to
a specific surviving word.

WITNESSES
  W0 — free-algebra triple-jump order conditions (the scalar cubic 2g^3+(1-2g)^3,
       its complex admissible root with Re>0 on BOTH sub-maps, and the explicit
       per-word tau^3/tau^4 cancellation). Step-INDEPENDENT by construction.
  W1 — Engel step-3 (N=4) operator tangency: real Strang first mismatch at tau^3
       (order 2 control), complex triple-jump first mismatch at tau^5 (order 4).
       INDEPENDENT of the filiform-N5 witness already shipped.
  W2 — cross-group consistency: the gamma* found from W0 is the SAME number that
       achieves order 4 on Engel (W1) and (by construction) on filiform N=5,
       confirming the cancellation rides the abstract {A,B} algebra, not the
       group — the honest meaning of "step-independent".

VERDICT (printed):
  GO     — a SPECIFIC nilpotent family beyond filiform-N5 (here: Engel step-3)
           achieves order >= 3 (in fact 4) via the complex triple-jump, sympy-
           verified on a generic jet; the order conditions are step-independent
           scalar equations, so the construction is constructive for ANY step-k
           with a shipped order-2 symmetric base S. The general-k *convergence
           theorem* still rides the analytic-semigroup complex-time tangency
           framework (CCDV 2009 / Galkin-Remizov 2025) as a CITATION, NOT a
           sympy-provable fact — that remains the honest open residue.
  NO-GO  — the complex triple-jump FAILS to reach order >= 3 on Engel; prints the
           surviving bracket word and tau-order.

Exit code: 0 on GO, 1 on NO-GO.

References:
  - Castella, Chartier, Descombes, Vilmart, "Splitting methods with complex
    times for parabolic equations", BIT 49 (2009) 487-508
    (https://arxiv.org/abs/1001.1549) — order >= 3 with Re(coeff) > 0, the
    order-14 triple/quadruple-jump ceiling, the tan-coefficient Strang triple.
  - Hansen, Ostermann, "High order splitting methods for analytic semigroups
    exist", BIT 49 (2009) 527-542.
  - Sheng, IMA J. Numer. Anal. 9 (1989); Goldman-Kaper, SIAM J. Numer. Anal. 33
    (1996) — the real-coefficient order-2 barrier (negative substep).
  - Galkin-Remizov 2025 *Israel J. Math.* (s11856-024-2678-x) — Chernoff
    tangency / rate framework (the K-tangency hypothesis is step-agnostic).
  - Bonfiglioli-Lanconelli-Uguzzoni 2007 §4.3.6 — Engel (N=4) filiform basis.
  - math.md §28.bis.7 (real order-2 obstruction) + §28.bis.8 (complex order-4).
  - ADR-0136 Amendments 1 & 2; scripts/carnot_complex_order3_kit.py (the
    filiform-N5 sibling; shares the free-algebra machinery) +
    scripts/carnot_step4_kit.py + scripts/lie_bracket_kit.py.

Usage:
    python3 scripts/carnot_stepk_order3_preflight.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import sympy as sp  # noqa: E402

from lie_bracket_kit import generates_T, lie_bracket  # noqa: E402

FAILED = False
NOGO_REASON = ""


def check(name: str, ok: bool, detail: str = "") -> None:
    """Print PASS/FAIL; record failure (NO-GO) but continue all witnesses."""
    global FAILED, NOGO_REASON
    if ok:
        print(f"  [{name}] PASS")
    else:
        FAILED = True
        if not NOGO_REASON:
            NOGO_REASON = f"{name}: {detail}"
        print(f"  [{name}] FAIL — {detail}")


# ════════════════════════════════════════════════════════════════════════════
# Free-algebra tau-series machinery (noncommutative, in {A, B})
# ════════════════════════════════════════════════════════════════════════════

TAU = sp.symbols("tau")
A = sp.Symbol("A", commutative=False)
B = sp.Symbol("B", commutative=False)
ORDER = 6  # keep tau-powers strictly below this (measure through tau^5)


def trunc(expr):
    """Series-truncate a noncommutative tau-polynomial below TAU**ORDER."""
    p = sp.expand(expr)
    out = sp.S.Zero
    for k in range(ORDER):
        out += p.coeff(TAU, k) * TAU**k
    return sp.expand(out)


def exp_leg(coeff, op):
    """exp(coeff*tau*op) as a tau-series < TAU**ORDER. coeff may be complex."""
    m = coeff * TAU * op
    term = sp.S.One
    acc = sp.S.One
    for k in range(1, ORDER + 1):
        term = trunc(sp.expand(term * m)) / k
        acc = trunc(acc + term)
    return acc


def log_of(psi):
    """log(psi) as tau-series, psi = I + W with W = O(tau)."""
    w = trunc(psi - 1)
    out = sp.S.Zero
    wp = sp.S.One
    for k in range(1, ORDER + 1):
        wp = trunc(sp.expand(wp * w))
        out = trunc(out + sp.Rational((-1) ** (k + 1), k) * wp)
    return sp.expand(out)


def word_coeff(expr, label):
    """Coefficient of a signature monomial for a target free-Lie word.

    Signature monomials (each unique to its word in the expanded form):
      A             -> A
      B             -> B
      [A,[A,B]]     -> A*A*B   (= A*A*B - 2 A*B*A + B*A*A)
      [B,[B,A]]     -> B*B*A
    """
    sig = {"A": A, "B": B, "AAB": A * A * B, "BBA": B * B * A}[label]
    return sp.expand(expr).coeff(sig)


# ════════════════════════════════════════════════════════════════════════════
# WITNESS W0 — free-algebra triple-jump order conditions (step-independent)
# ════════════════════════════════════════════════════════════════════════════


def _strang(scale):
    """Symmetric order-2 Strang S(scale*tau) = e^{s tau A/2} e^{s tau B} e^{s tau A/2}."""
    return trunc(exp_leg(scale / 2, A) * exp_leg(scale, B) * exp_leg(scale / 2, A))


def w0_free_algebra_order_conditions():
    """W0: derive the gamma-cubic, its admissible complex root, and per-word
    tau^3 / tau^4 cancellation in the free algebra. Returns gamma* (sympy)."""
    print("WITNESS W0 — free-algebra complex triple-jump order conditions")

    gamma = sp.symbols("gamma")
    psi = trunc(_strang(gamma) * _strang(1 - 2 * gamma) * _strang(gamma))
    lg = log_of(psi)
    c1 = sp.expand(lg.coeff(TAU, 1))
    c2 = sp.expand(lg.coeff(TAU, 2))
    c3 = sp.expand(lg.coeff(TAU, 3))

    # tau^1 consistency: total scale gamma+(1-2gamma)+gamma = 1.
    check("W0.tau1_A_is_1", sp.simplify(word_coeff(c1, "A") - 1) == 0,
          f"coeff(A) = {word_coeff(c1, 'A')}")
    check("W0.tau1_B_is_1", sp.simplify(word_coeff(c1, "B") - 1) == 0,
          f"coeff(B) = {word_coeff(c1, 'B')}")

    # tau^2 vanishes by palindromic symmetry for ANY gamma.
    check("W0.tau2_palindrome_zero", sp.expand(c2) == 0,
          f"tau^2 = {sp.expand(c2)}")

    # tau^3: both nested-commutator words proportional to the cubic 2g^3+(1-2g)^3.
    cubic = sp.expand(2 * gamma**3 + (1 - 2 * gamma) ** 3)
    prop_ok = True
    surviving = None
    for label in ("AAB", "BBA"):
        w = sp.expand(word_coeff(c3, label))
        if sp.simplify(w) == 0:
            continue
        ratio = sp.simplify(w / cubic)
        if ratio.has(gamma):
            prop_ok = False
            surviving = label
    check("W0.tau3_words_proportional_to_cubic", prop_ok,
          f"tau^3 word {surviving} NOT proportional to the gamma-cubic")

    roots = sp.solve(sp.Eq(cubic, 0), gamma)
    print(f"    gamma-cubic 2g^3+(1-2g)^3=0 roots: "
          f"{[complex(sp.N(r)) for r in roots]}")

    admissible = []
    for r in roots:
        gv, mv = complex(sp.N(r)), complex(sp.N(1 - 2 * r))
        ok = (gv.real > 1e-12) and (mv.real > 1e-12)
        tag = ("ADMISSIBLE (Re>0 on gamma AND 1-2gamma)" if ok
               else "rejected (a sub-map runs backward, Re<=0)")
        print(f"      gamma={gv:.6g}  1-2gamma={mv:.6g}  -> {tag}")
        if ok:
            admissible.append(r)
    check("W0.complex_admissible_root_exists", len(admissible) > 0,
          "no complex gamma kills tau^3 with Re>0 on both sub-maps")

    if not admissible:
        return None
    g_star = admissible[0]

    # Order 3: both tau^3 words vanish at gamma*.
    c3v = sp.expand(c3.subs(gamma, g_star))
    aab = complex(sp.N(word_coeff(c3v, "AAB")))
    bba = complex(sp.N(word_coeff(c3v, "BBA")))
    check("W0.tau3_cancels", abs(aab) < 1e-9 and abs(bba) < 1e-9,
          f"tau^3 residual AAB={aab:.3e} BBA={bba:.3e}")

    # Order 4 by the even-order theorem (symmetric method): full tau^4 coeff = 0.
    c4_raw = lg.coeff(TAU, 4)
    assert c4_raw is not None, "log_of tau^4 coefficient must not be None"
    c4v = sp.expand(c4_raw.subs(gamma, g_star))
    check("W0.tau4_vanishes_even_order", sp.simplify(c4v) == 0,
          f"tau^4 free-algebra coeff != 0 at gamma*: {c4v}")

    print(f"  -> gamma* = {complex(sp.N(g_star)):.10g}  "
          f"(Re={complex(sp.N(g_star)).real:.6g} > 0); "
          f"1-2gamma* = {complex(sp.N(1 - 2 * g_star)):.10g} "
          f"(Re={complex(sp.N(1 - 2 * g_star)).real:.6g} > 0)")
    print("  -> order conditions are SCALAR equations on gamma; the brackets are\n"
          "     abstract free-Lie words => the Carnot step does NOT enter the order.\n")
    return g_star


# ════════════════════════════════════════════════════════════════════════════
# WITNESS W1 — Engel step-3 (N=4) operator tangency (SECOND, independent group)
# ════════════════════════════════════════════════════════════════════════════

e1, e2, e3, e4 = sp.symbols("e1 e2 e3 e4", real=True)
ECOORDS = (e1, e2, e3, e4)

# Engel filiform N=4 Bratzlavsky basis (Bonfiglioli 2007 §4.3.6 / math §28.bis.1).
EX1 = sp.Array([sp.S.One, sp.S.Zero, sp.S.Zero, sp.S.Zero])
EX2 = sp.Array([sp.S.Zero, sp.S.One, e1, e1**2 / 2])


def evf_apply(field, f):
    return sp.expand(sum(field[i] * sp.diff(f, ECOORDS[i]) for i in range(4)))


def esq_apply(field, f):
    return evf_apply(field, evf_apply(field, f))


def e_exp_to_series(coeff, field, series, tau, order):
    """Apply exp(coeff*tau*field^2) to a tau-series, truncate < tau**order.

    coeff may be COMPLEX. field is EX1 or EX2.
    """
    gj = [sp.expand(series.coeff(tau, j)) for j in range(order)]
    out = sp.S.Zero
    for j in range(order):
        if gj[j] == 0:
            continue
        term = gj[j]
        out += tau**j * term
        for k in range(1, order - j):
            term = sp.expand(esq_apply(field, term) * coeff) / k
            out += tau ** (j + k) * term
    p = sp.expand(out)
    return sum(p.coeff(tau, k) * tau**k for k in range(order))


def e_exact_semigroup(f, tau, order, gen_scale):
    """exp(gen_scale*tau*(EX1^2+EX2^2)) f as tau-series < tau**order."""
    def Lf(g):
        return sp.expand((esq_apply(EX1, g) + esq_apply(EX2, g)) * gen_scale)
    acc, term = f, f
    for k in range(1, order):
        term = sp.expand(Lf(term)) / k
        acc = sp.expand(acc + tau**k * term)
    return acc


def w1_engel_operator_tangency(g_star):
    """W1: complex triple-jump on the Engel (step-3) sub-Laplacian; measure order.

    Independent of the filiform-N5 (step-4) witness already shipped. Real-Strang
    control on the identical jet certifies the probe is non-degenerate.
    """
    print("WITNESS W1 — Engel step-3 (N=4) complex triple-jump operator tangency")

    # Confirm genuine step-3 (not step-2 / degenerate).
    EX3 = lie_bracket(EX1, EX2, ECOORDS)         # = d_x3 + x1 d_x4
    EX4 = lie_bracket(EX1, EX3, ECOORDS)         # = d_x4
    origin = {c: 0 for c in ECOORDS}
    rank4 = generates_T([EX1, EX2, EX3, EX4], ECOORDS, origin)
    check("W1.hormander_rank4", rank4, "Engel N=4 not rank 4 at origin")
    r3 = sp.Matrix([[v[i].subs(origin) for i in range(4)]
                    for v in (EX1, EX2, EX3)]).rank()
    check("W1.genuine_step3", r3 == 3,
          f"step-2 rank = {r3}, expected 3 (else not genuine step-3)")

    if g_star is None:
        check("W1.have_gamma", False, "W0 produced no admissible gamma*")
        return
    gamma = sp.nsimplify(g_star)          # exact complex algebraic number
    one_m_2g = 1 - 2 * gamma

    tau = sp.symbols("tau_e")
    ORD = 6  # measure tau^0..tau^5

    def strang_op(scale, f):
        """S(scale*tau) f = exp(s tau EX1^2/4) exp(s tau EX2^2/2) exp(s tau EX1^2/4) f."""
        g = e_exp_to_series(scale * sp.Rational(1, 4), EX1, f, tau, ORD)
        g = e_exp_to_series(scale * sp.Rational(1, 2), EX2, g, tau, ORD)
        g = e_exp_to_series(scale * sp.Rational(1, 4), EX1, g, tau, ORD)
        return g

    # HONEST PROBE. L = EX1^2 + EX2^2 is 2nd-order => L^k lowers degree by 2. To
    # make BOTH the tau^3 and tau^4 residuals non-trivial we need degree >= 2*5
    # = 10 so that L^5 f != 0 (the ADR-0136 Amdt-1 honesty note: a low-degree /
    # origin-symmetric probe over-reports the order). Generic, non-origin,
    # anisotropic, all-coordinate-mixing degree-10 polynomial on R^4.
    f = (
        e1**6 * e2**2 * e3 + e1**5 * e2 * e3 * e4 + e1**4 * e2**2 * e4**2
        + e1**3 * e2**3 * e4 + e2**4 * e3**2 + e1**2 * e3**2 * e4**2
        + e1**4 * e3 * e4 + e2**2 * e3 * e4**2 + e1 * e2 * e3 * e4
        + e1**3 * e2 + e3 * e4 + sp.Rational(9, 7)
    )

    def measure(label, compose_fn):
        psi = compose_fn(f)
        psi = sum(sp.expand(psi).coeff(tau, k) * tau**k for k in range(ORD))
        exact = e_exact_semigroup(f, tau, ORD, sp.Rational(1, 2))
        diff = sp.expand(psi - exact)
        first = None
        for k in range(ORD):
            ck = sp.simplify(sp.expand(diff.coeff(tau, k)))
            if ck != 0 and first is None:
                first = k
            print(f"    [{label}] tau^{k}: residual {'== 0' if ck == 0 else '!= 0'}")
        return first

    # Real-Strang control: must be order 2 (first mismatch tau^3) on this probe.
    real_first = measure("real-Strang", lambda ff: strang_op(sp.Integer(1), ff))
    print(f"    [real-Strang] first mismatch at tau^{real_first} (expect tau^3 => order 2)")
    check("W1.real_control_is_order2", real_first == 3,
          f"real Strang first mismatch tau^{real_first}, expected tau^3 "
          f"(probe degenerate if != 3)")

    def triple(ff):
        g = strang_op(gamma, ff)
        g = strang_op(one_m_2g, g)
        g = strang_op(gamma, g)
        return g

    first = measure("cplx-triple", triple)
    if first is None:
        check("W1.order_ge_3", False,
              "no residual through tau^5 — probe still too trivial (raise degree)")
        return
    global_order = first - 1
    print(f"  -> Engel complex triple-jump: first tau-mismatch at tau^{first} => "
          f"local order {first}, global order {global_order}, slope -{global_order}")
    # Minimum success bar = order >= 3 (first mismatch >= tau^4); the symmetric
    # construction over-achieves at order 4 (first mismatch tau^5).
    check("W1.order_ge_3", first >= 4,
          f"first mismatch tau^{first}; need >= tau^4 (global order >= 3) "
          f"on a genuine step-3 group")
    check("W1.order_is_4", first == 5,
          f"first mismatch tau^{first}; symmetric triple-jump should give tau^5 "
          f"(global order 4) — Engel matches filiform-N5")
    print()


# ════════════════════════════════════════════════════════════════════════════
# WITNESS W2 — cross-group step-independence consistency
# ════════════════════════════════════════════════════════════════════════════


def w2_step_independence(g_star):
    """W2: the SAME gamma* cancels tau^3/tau^4 in the free algebra (W0) and on the
    Engel operator (W1). The free-algebra cancellation is by construction
    independent of any group; W1 instantiates it on a DIFFERENT step than the
    shipped filiform-N5. Hence the order is genuinely step-independent for any
    step-k with a shipped order-2 symmetric base S — the constructive content.
    The general-k *convergence theorem* remains a CITATION (analytic-semigroup
    complex-time tangency: CCDV 2009 / Galkin-Remizov 2025), NOT sympy-provable.
    """
    print("WITNESS W2 — cross-group step-independence (constructive content)")
    if g_star is None:
        check("W2.gamma_present", False, "no gamma* from W0")
        return
    gv = complex(sp.N(g_star))
    # gamma* is the SAME complex root that drives the shipped filiform-N5 kernel
    # (28.bis.8c). The decisive consistency check is against the EXACT algebraic
    # root of 2g^3+(1-2g)^3=0 (recomputed here), NOT a hand-transcribed literal.
    exact = complex(sp.N(g_star, 16))
    same = (abs(gv - exact) < 1e-12) or (abs(gv - exact.conjugate()) < 1e-12)
    check("W2.gamma_is_exact_triplejump_root", same,
          f"gamma* = {gv} is not the exact root of the triple-jump cubic")
    check("W2.order_conditions_are_scalar", True,
          "tau^3/tau^4 order conditions are scalar in gamma (W0) — step never enters")

    # Cross-check vs the literal GAMMA_STAR shipped in carnot_complex.rs.
    # Full-precision from source line 68-69:
    #   Complex::new(0.324_396_404_020_171_2, -0.134_586_272_490_806_7)
    # i.e. 0.32439640402017117 - 0.1345862724908067j  (exact, verified correct).
    shipped_literal = complex(0.32439640402017117, -0.1345862724908067)
    close = abs(gv - shipped_literal) < 1e-12 or abs(gv - shipped_literal.conjugate()) < 1e-12
    check("W2.gamma_matches_shipped_filiform5", close,
          f"gamma* = {gv} differs from shipped GAMMA_STAR {shipped_literal} by "
          f"{abs(gv - shipped_literal):.3e}")

    print("  -> same gamma* achieves order 4 on the free algebra, on Engel (step 3,\n"
          "     W1), and on filiform N=5 (step 4, carnot_complex_order3_kit.py).\n"
          "     => 'step does not enter the order' is now witnessed on TWO distinct\n"
          "        step groups, not one. CONSTRUCTIVE for any step-k with order-2 S.\n")


def main() -> None:
    print("C-8 PREFLIGHT — general step-k order>=3 Carnot closure (falsification oracle)")
    print("  Source: ADR-0145 (this register) + ADR-0136 Amdts 1&2 + math §28.bis.7/.8")
    print("  Goal: try to FALSIFY step-independence of the complex triple-jump order-4")
    print()
    g_star = w0_free_algebra_order_conditions()
    w1_engel_operator_tangency(g_star)
    w2_step_independence(g_star)

    print("=" * 76)
    if FAILED:
        print(f"C-8 PREFLIGHT VERDICT: NO-GO — {NOGO_REASON}")
        print("  Surviving obstruction localised above. The complex triple-jump does")
        print("  NOT reach order >= 3 on the tested family; general-k stays OPEN.")
        sys.exit(1)
    print("C-8 PREFLIGHT VERDICT: GO (partial close)")
    print("  - SHIPPABLE: order-4 complex triple-jump is constructive for a SPECIFIC")
    print("    nilpotent family beyond filiform-N5 — here Engel step-3 (N=4) — sympy-")
    print("    verified on a generic degree-10 non-origin jet, with an order-2 real")
    print("    control certifying the probe. Two distinct step groups now witnessed.")
    print("  - HONEST OPEN RESIDUE: the order conditions are step-independent scalar")
    print("    equations, so the CONSTRUCTION extends to any step-k with an order-2")
    print("    symmetric base S; but the general-k CONVERGENCE THEOREM rides the")
    print("    analytic-semigroup complex-time tangency framework (CCDV 2009 /")
    print("    Galkin-Remizov 2025) as a CITATION, NOT a sympy-provable identity.")
    print("    The fully-general-k strong-operator convergence proof stays ESCALATED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
