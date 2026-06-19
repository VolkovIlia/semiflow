#!/usr/bin/env python3
"""ADR-0117 PRE-FLIGHT sympy oracle — OCTONIC-Hermite degree-9 virtual-node sampler.

The v6.0.0 SepticHermite sampler (degree-7, matches f, f', f'', f''' at 2 nodes)
floors at φ ≈ 1.49e-12 (math.md §40.4). ADR-0117 introduces OCTONIC-Hermite:
degree-9 interpolant matching f, f', f'', f''', f'''' (FIVE data per node) at the
two cell endpoints → 5 × 2 = 10 constraints → unique degree-9 polynomial with
remainder `R(x) = f^{(10)}(ξ)/10! · ∏(x−x_i)²(x−x_{i+1})²·(extra factor)` →
leading residue O(dx¹⁰). Predicted virtual-node floor at N=512: ~1e-16.

(Naming note: "OCTONIC" follows the v7.0.0 backlog freeze item #1 label. The
interpolant is degree-9 = matches up to the 4th derivative. The backlog table
also calls it `OcticHermite`; the NORMATIVE enum variant is
`InterpKind::OctonicHermite`.)

Sub-checks:
  (a) octonic-hermite-weight-derivation: derive the degree-9 weight a_0(s)
      (value-matching weight for node 0) and verify all 10 endpoint constraints
      (value + 4 derivatives at s=0 and s=1) hold symbolically with 0 residual,
      degree exactly 9.
  (b) tenth-order-remainder: build the FULL degree-9 Hermite interpolant of a
      Gaussian probe f(x)=exp(−x²) on [0,h], evaluate residual at s=1/2, verify
      leading term is h^10.
  (c) fornberg-fd-weights: derive the 10-point central Fornberg FD weights for
      f^{(k)}, k=0..4, and verify each reproduces the derivative to O(dx^{≥6})
      (symbolic residual leading power), so the ghost-data feeding the degree-9
      Hermite is accurate enough to preserve the O(dx¹⁰) interpolant floor.
  (d) condition-number-bound: weight 1-norm sup over s∈[0,1] is bounded (Lebesgue
      analogue) — stays O(1) so the 65-node barycentric amplification keeps φ_eff
      within a small constant × the local truncation.
  (e) predicted-floor: at N=512, dx≈0.0391 → dx^10 ≈ 8.3e-15; with ‖f^{(10)}‖_∞
      Gaussian prefactor, 10!, condition number and Lebesgue Λ_M=64 → predicted
      φ ∈ [1e-17, 5e-14] band, centred ~1e-16. RELEASE_BLOCKING target ≤ 5e-14.

If 5/5 PASS → ADR-0117 GO: degree-9 Hermite is sound (Birkhoff-Garabedian-Lorentz),
10-point Fornberg ghost data is accurate, predicted floor ≤ 5e-14 lifts SepticHermite.

ADR-0086 PRE-FLIGHT-first principle. NORMATIVE.
"""

from __future__ import annotations

import math
import sys

X_MIN = -10.0
X_MAX = 10.0
N_SPATIAL = 512
DX = (X_MAX - X_MIN) / N_SPATIAL  # ≈ 0.0391

# SepticHermite reference floor (math.md §40.4).
SEPTIC_FLOOR_N512 = 1.49e-12
# OCTONIC predicted-floor RELEASE_BLOCKING target (backlog item #1).
OCTONIC_FLOOR_TARGET = 5e-14


def emit_pass(label: str) -> None:
    print(f"  [PASS] {label}")


def emit_fail(label: str, reason: str) -> str:
    print(f"  [FAIL] {label}: {reason}", flush=True)
    return reason


# ---------------------------------------------------------------------------
# Sub-check (a) — degree-9 Hermite weight derivation
# ---------------------------------------------------------------------------


def check_octonic_weight_derivation() -> str | None:
    """Derive a_0(s) for degree-9 Hermite (5 data per node) and verify 10 constraints.

    a_0 must satisfy:
      a_0(0)=1; a_0^{(k)}(0)=0, k=1..4;
      a_0(1)=0; a_0^{(k)}(1)=0, k=1..4.

    Closed form (generalised Hermite value-weight):
      a_0(s) = (1−s)^5 · (1 + 5s + 15s² + 35s³ + 70s⁴)
    (the polynomial factor is the degree-4 truncation of (1−s)^{−5} binomial
    series, which forces the first 4 derivatives to vanish at s=1 while the
    (1−s)^5 factor forces value+4 derivatives to vanish at s=1; the design also
    yields a_0(0)=1, a_0^{(k)}(0)=0).
    """
    label = "(a) OctonicHermite weight a_0 derivation (degree-9, 10 endpoint constraints)"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    s = sp.Symbol("s", real=True)
    a0 = (1 - s) ** 5 * (1 + 5 * s + 15 * s**2 + 35 * s**3 + 70 * s**4)
    a0 = sp.expand(a0)

    # Constraints at s=0.
    if sp.simplify(a0.subs(s, 0) - 1) != 0:
        return emit_fail(label, f"a_0(0)={a0.subs(s,0)}, expected 1")
    for k in range(1, 5):
        if sp.simplify(sp.diff(a0, s, k).subs(s, 0)) != 0:
            return emit_fail(label, f"a_0^{{({k})}}(0) ≠ 0")
    # Constraints at s=1.
    if sp.simplify(a0.subs(s, 1)) != 0:
        return emit_fail(label, f"a_0(1)={a0.subs(s,1)}, expected 0")
    for k in range(1, 5):
        if sp.simplify(sp.diff(a0, s, k).subs(s, 1)) != 0:
            return emit_fail(label, f"a_0^{{({k})}}(1) ≠ 0")

    poly = sp.Poly(a0, s)
    if poly.degree() != 9:
        return emit_fail(label, f"a_0 degree {poly.degree()}, expected 9")

    print("    a_0(s) = (1−s)^5 · (1 + 5s + 15s² + 35s³ + 70s⁴) — degree 9")
    print("    All 10 endpoint constraints (value + 4 derivs at s=0,1) PASS.")
    print(f"    Expansion: a_0(s) = {sp.expand(a0)}")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check (b) — tenth-order remainder
# ---------------------------------------------------------------------------


def _octonic_basis(sp, s):
    """Return the 10 degree-9 Hermite basis weights (a0..a4, b0..b4) in s∈[0,1].

    Built by solving the 10×10 endpoint constraint system directly (robust,
    no guessing of closed forms for the derivative-matching weights).
    """
    # General degree-9 polynomial with 10 unknown coefficients.
    coeffs = sp.symbols("c0:10")
    poly = sp.Add(*[coeffs[i] * s**i for i in range(10)])

    def build(constraints):
        eqs = []
        for (node, order, rhs) in constraints:
            expr = sp.diff(poly, s, order).subs(s, node) if order > 0 else poly.subs(s, node)
            eqs.append(sp.Eq(expr, rhs))
        sol = sp.solve(eqs, coeffs, dict=True)[0]
        return sp.expand(poly.subs(sol))

    weights = {}
    # a_k owns node 0, derivative order k; b_k owns node 1, derivative order k.
    for k in range(5):
        cons = []
        for order in range(5):
            cons.append((0, order, 1 if order == k else 0))
            cons.append((1, order, 0))
        weights[f"a{k}"] = build(cons)
    for k in range(5):
        cons = []
        for order in range(5):
            cons.append((0, order, 0))
            cons.append((1, order, 1 if order == k else 0))
        weights[f"b{k}"] = build(cons)
    return weights


def check_tenth_order_remainder() -> str | None:
    label = "(b) OctonicHermite residual scales as O(h^10) on Gaussian probe"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    h = sp.Symbol("h", positive=True, real=True)
    s = sp.Symbol("s", real=True)
    x = sp.Symbol("x", real=True)

    w = _octonic_basis(sp, s)

    f = sp.exp(-(x**2))
    # Scaled nodal data: h^k · f^{(k)} at each endpoint.
    f0 = [sp.diff(f, x, k).subs(x, 0) for k in range(5)]
    f1 = [sp.diff(f, x, k).subs(x, h) for k in range(5)]

    p = sp.Add(*[w[f"a{k}"] * h**k * f0[k] for k in range(5)])
    p += sp.Add(*[w[f"b{k}"] * h**k * f1[k] for k in range(5)])

    p_half = p.subs(s, sp.Rational(1, 2))
    f_true_half = f.subs(x, h / 2)
    residual = sp.simplify(f_true_half - p_half)

    series = sp.series(residual, h, 0, 14).removeO()
    poly = sp.Poly(series, h)
    lowest = None
    for monom, coeff in poly.terms():
        if sp.simplify(coeff) != 0:
            p_pow = monom[0]
            if lowest is None or p_pow < lowest:
                lowest = p_pow
    if lowest is None:
        lowest = 14
    print(f"    Leading residual term: h^{lowest}")
    if lowest != 10:
        return emit_fail(label, f"leading residual power h^{lowest}, expected h^10")
    print("    Degree-9 Hermite achieves O(h^10) — algebraic identity verified.")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check (c) — 10-point Fornberg central FD weights for f^{(k)}, k=0..4
# ---------------------------------------------------------------------------


def _fornberg_weights(sp, order, offsets):
    """Fornberg central FD weights for derivative `order` on stencil `offsets`
    (list of integer node offsets, in units of dx). Returns the weights as a list
    of sympy Rationals (the FD formula is Σ w_j f(x + offsets[j]·dx) / dx^order).
    """
    import sympy as sp_
    n = len(offsets)
    # Vandermonde: row p (p=0..n-1) is offsets^p; rhs is order! at p=order else 0.
    M = sp_.Matrix(n, n, lambda r, col: sp_.Integer(offsets[col]) ** r)
    rhs = sp_.Matrix(n, 1, lambda r, _col: sp_.factorial(order) if r == order else 0)
    w = M.solve(rhs)
    return [sp_.nsimplify(wi) for wi in w]


def check_fornberg_fd_weights() -> str | None:
    label = "(c) 10-point Fornberg central FD weights for f^(k), k=0..4, accurate"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    dx = sp.Symbol("dx", positive=True)
    x = sp.Symbol("x", real=True)
    f = sp.Function("f")

    # 10-point central stencil offsets for derivatives at a NODE: {-5..-1, 1..5}
    # for odd derivatives (k=1,3) is anti-symmetric; for even (k=2,4) symmetric;
    # k=0 is identity (weight = value, trivially exact). Use {-5,...,-1,0,1,...,4}
    # is NOT symmetric → instead use symmetric {-4,-3,-2,-1,0,1,2,3,4} (9-point)
    # for even, and {-5..-1,1..5} (10-point) for odd. For uniform "10-point"
    # accounting we verify the standard symmetric central stencils.
    results = {}
    # k=0: identity — exact by construction.
    results[0] = ("identity", math.inf)

    # k=1: 10-point central {±1,±2,±3,±4,±5}.
    offs1 = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    w1 = _fornberg_weights(sp, 1, offs1)
    approx1 = sum(w1[j] * f(x + offs1[j] * dx) for j in range(len(offs1))) / dx
    res1 = sp.series(approx1 - sp.diff(f(x), x), dx, 0, 12).removeO()
    p1 = _lowest_power(sp, res1, dx)
    results[1] = (offs1, p1)

    # k=2: 9-point central {0,±1,±2,±3,±4}.
    offs2 = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    w2 = _fornberg_weights(sp, 2, offs2)
    approx2 = sum(w2[j] * f(x + offs2[j] * dx) for j in range(len(offs2))) / dx**2
    res2 = sp.series(approx2 - sp.diff(f(x), x, 2), dx, 0, 12).removeO()
    p2 = _lowest_power(sp, res2, dx)
    results[2] = (offs2, p2)

    # k=3: 10-point central {±1,±2,±3,±4,±5}.
    offs3 = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    w3 = _fornberg_weights(sp, 3, offs3)
    approx3 = sum(w3[j] * f(x + offs3[j] * dx) for j in range(len(offs3))) / dx**3
    res3 = sp.series(approx3 - sp.diff(f(x), x, 3), dx, 0, 12).removeO()
    p3 = _lowest_power(sp, res3, dx)
    results[3] = (offs3, p3)

    # k=4: 9-point central {0,±1,±2,±3,±4}.
    offs4 = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    w4 = _fornberg_weights(sp, 4, offs4)
    approx4 = sum(w4[j] * f(x + offs4[j] * dx) for j in range(len(offs4))) / dx**4
    res4 = sp.series(approx4 - sp.diff(f(x), x, 4), dx, 0, 12).removeO()
    p4 = _lowest_power(sp, res4, dx)
    results[4] = (offs4, p4)

    print("    Derivative   stencil-size   leading FD-error power (dx)")
    # Required absolute accuracy for f^{(k)} so that h^k·f^{(k)} ghost preserves
    # the O(dx^10) interpolant: need f^{(k)} accurate to O(dx^{10-k}) → since
    # h^k multiplies it. Symmetric central stencils above give well above that.
    min_required = {0: 0, 1: 9, 2: 8, 3: 7, 4: 6}
    for k in range(5):
        if k == 0:
            print(f"      f^(0)        identity        exact")
            continue
        offs, p = results[k]
        print(f"      f^({k})        {len(offs)}-point        dx^{p}   (need ≥ dx^{min_required[k]})")
        if p < min_required[k]:
            return emit_fail(
                label,
                f"f^({k}) FD accuracy dx^{p} < required dx^{min_required[k]} — "
                "ghost data too coarse to preserve O(dx^10) interpolant floor.",
            )
    print("    All ghost-derivative FD stencils accurate enough for O(dx^10) floor.")
    print(f"    f'  weights ×840-scale: {[str(wi) for wi in w1]}")
    emit_pass(label)
    return None


def _lowest_power(sp, expr, dx):
    expr = sp.expand(expr)
    poly = sp.Poly(expr, dx)
    lowest = None
    for monom, coeff in poly.terms():
        if sp.simplify(coeff) != 0:
            p = monom[0]
            if lowest is None or p < lowest:
                lowest = p
    return 12 if lowest is None else lowest


# ---------------------------------------------------------------------------
# Sub-check (d) — condition number bound
# ---------------------------------------------------------------------------


def check_condition_number_bound() -> str | None:
    label = "(d) OctonicHermite weight 1-norm bounded (Lebesgue-like constant)"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    s_sym = sp.Symbol("s", real=True)
    w = _octonic_basis(sp, s_sym)
    weight_fns = [sp.lambdify(s_sym, w[name], "math") for name in
                  ("a0", "a1", "a2", "a3", "a4", "b0", "b1", "b2", "b3", "b4")]

    sup_1norm = 0.0
    sup_s = 0.0
    n_probes = 1001
    for i in range(n_probes):
        s = i / (n_probes - 1)
        one_norm = sum(abs(fn(s)) for fn in weight_fns)
        if one_norm > sup_1norm:
            sup_1norm = one_norm
            sup_s = s

    bound = 10.0  # degree-9 has slightly larger Lebesgue than degree-7's ≤5
    print(f"    sup_{{s∈[0,1]}} Σ|weight|(s) = {sup_1norm:.4f} at s = {sup_s:.3f}")
    if sup_1norm > bound:
        return emit_fail(label, f"1-norm {sup_1norm:.4f} exceeds {bound}")
    print(f"    Benign bound (≤ {bound}) PASS — amplification stays O(1).")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check (e) — predicted floor at N=512
# ---------------------------------------------------------------------------


def check_predicted_floor() -> str | None:
    label = "(e) predicted OCTONIC floor at N=512 ≤ target 5e-14"
    dx = DX
    dx_to_10 = dx**10
    # ‖f^{(10)}‖_∞ for f(x)=exp(−x²): Hermite-polynomial recursion gives
    # f^{(10)}(x) = H_10(x)·exp(−x²) (physicists' Hermite). Peak magnitude ≈ 30240.
    # (He_10(0) = -945·(-1)^5·... ; |f^{(10)}|_max ≈ 3.0e4 — use 3.02e4.)
    f_10_max = 30240.0
    fact_10 = math.factorial(10)  # 3628800
    c_weights = 4.0  # generous bound from sub-check (d)
    lebesgue_m64 = 3.3
    phi_predicted = f_10_max * dx_to_10 / fact_10 * c_weights * lebesgue_m64

    band_lo, band_hi = 1e-18, 5e-14
    print(f"    dx = {dx:.4f}  →  dx^10 = {dx_to_10:.3e}")
    print(f"    ‖f^(10)‖_∞ ≈ {f_10_max} (Gaussian IC), 10! = {fact_10}")
    print(f"    C_weights ≤ {c_weights}, Λ_M=64 ≈ {lebesgue_m64}")
    print(f"    φ_predicted = {phi_predicted:.3e}  (target ≤ {OCTONIC_FLOOR_TARGET:.0e})")
    print(f"    Improvement vs SepticHermite: {SEPTIC_FLOOR_N512 / phi_predicted:.0f}× lower floor")
    if not (band_lo <= phi_predicted <= band_hi):
        return emit_fail(
            label,
            f"predicted φ = {phi_predicted:.3e} outside band [{band_lo:.0e}, {band_hi:.0e}]",
        )
    if phi_predicted > OCTONIC_FLOOR_TARGET:
        return emit_fail(
            label, f"predicted φ {phi_predicted:.3e} > RELEASE_BLOCKING target {OCTONIC_FLOOR_TARGET:.0e}"
        )
    emit_pass(label)
    return None


def main() -> int:
    print("=" * 76)
    print("T_OCTONIC_HERMITE — ADR-0117 PRE-FLIGHT sympy oracle (degree-9 sampler)")
    print("=" * 76)
    print()
    print(f"Configuration: N={N_SPATIAL}, [xmin,xmax]=[{X_MIN},{X_MAX}], dx={DX:.4e}")
    print(f"SepticHermite reference floor (§40.4): φ ≈ {SEPTIC_FLOOR_N512:.2e}")
    print(f"OCTONIC predicted-floor target:        φ ≤ {OCTONIC_FLOOR_TARGET:.0e}")
    print()
    print("Sub-checks:")

    checks = [
        ("a", check_octonic_weight_derivation),
        ("b", check_tenth_order_remainder),
        ("c", check_fornberg_fd_weights),
        ("d", check_condition_number_bound),
        ("e", check_predicted_floor),
    ]
    failures: list[str] = []
    for letter, fn in checks:
        print()
        result = fn()
        if result is not None:
            failures.append(f"({letter}) {result}")

    print()
    print("=" * 76)
    if failures:
        print(f"T_OCTONIC_HERMITE FAIL ({len(failures)}/5 sub-checks):")
        for fmsg in failures:
            print(f"  - {fmsg}")
        return 1
    print("T_OCTONIC_HERMITE PASS (5/5 sub-checks:")
    print(" weight_derivation / tenth_order_remainder / fornberg_fd_weights /")
    print(" condition_number_bound / predicted_floor)")
    print()
    print("ARCHITECTURAL CONCLUSION:")
    print("  - Degree-9 Hermite (5 data/node) is sound (Birkhoff-Garabedian-Lorentz).")
    print("  - 10-point Fornberg central FD ghost data preserves O(dx^10) floor.")
    print("  - Predicted virtual-node floor ≤ 5e-14 (~30× below SepticHermite).")
    print("  - InterpKind::OctonicHermite is ADDITIVE; default stays SepticHermite.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
