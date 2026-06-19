#!/usr/bin/env python3
"""ADR-0118 (6th-order) PRE-FLIGHT sympy oracle — 7-point 6th-order divergence stencil.

Sibling of `verify_div_stencil_4th.py`. The 4th-order stencil dropped the spatial
floor from `T·dx²` to `T·dx⁴`; ADR-0119's decisive oracle showed that even `T·dx⁴`
leaves the ζ⁸ pair-slope gate floor-limited at the gate horizon. A **6th-order**
`T·dx⁶` stencil is required to open a non-empty τ-window for BOTH ζ⁶ and ζ⁸ at
N=4096 / T=10 (see `verify_zeta_truthful_order_octonic.py`). This oracle proves the
6th-order divergence-form stencil is mathematically sound.

The design (NORMATIVE):

  6th-order flux-difference form. Write the operator as a flux divergence
  `(∂_x F)_i` where the half-node flux `F_{i+½} = a_{i+½} · (∂_x f)_{i+½}` and BOTH
  the half-node derivative AND the outer flux divergence are taken to 6th order.

  6th-order staggered half-node first derivative (6-point, offsets {∓5/2,∓3/2,∓1/2}
  in units of dx relative to the midpoint):
    (∂_x f)_{i+½} ≈ [ −9 f_{i-2} + 125 f_{i-1} − 2250 f_i
                      + 2250 f_{i+1} − 125 f_{i+2} + 9 f_{i+3} ] / (1920 dx)   (O(dx⁶))
  (derived from the staggered Vandermonde system; sub-check (1) proves O(dx⁶)).

  `a` is evaluated EXACTLY at the half-node x_{i+½} (known coefficient, no FD error),
  so the only truncation is in the f-derivative and the outer flux divergence.

  Outer divergence (6th order) applies the SAME staggered weights to the half-node
  fluxes F at the half-node centres {∓5/2,∓3/2,∓1/2}·dx around node i.

Composing two 6th-order staggered operators yields a globally 6th-order discretisation
of `∂_x(a ∂_x f)` for smooth a, f. This oracle proves it symbolically.

Sub-checks:
  (1) half-node-derivative-6th: the 6-point staggered first-derivative stencil
      reproduces the midpoint derivative to O(dx⁶); coefficients of dx²..dx⁵ are 0.
  (2) constant-a-composition: for a ≡ 1, the composed operator is O(dx⁶) (the
      6th-order accurate discrete Laplacian).
  (3) variable-a-truncation: for arbitrary smooth a, f the truncation
      `(∂_x F)_i − ∂_x(a ∂_x f)|_{x_i}` has dx², dx³, dx⁴ AND dx⁵ coefficients
      identically zero as symbolic functions of a, a', ..., f, f', ... → O(dx⁶).
  (4) conservation-form: the generating polynomial
      `−9 + 125z − 2250z² + 2250z³ − 125z⁴ + 9z⁵` is divisible by `(1−z)` → the
      operator is a discrete divergence (telescoping flux difference), so Neumann
      mass is preserved exactly (mirror 4th-order / 3-point forms).

If 4/4 PASS → ADR-0118 (6th-order) GO: the 6th-order stencil drops the spatial floor
from `T·dx⁴` to `T·dx⁶`. At N=4096, dx⁶/dx⁴ = dx² ≈ 2.4e-5 further reduction; this is
the floor lowering that opens the ζ⁶/ζ⁸ gate window (ADR-0119 GO).

ADR-0086 PRE-FLIGHT-first principle. NORMATIVE.
"""

from __future__ import annotations

import sys

# 6th-order staggered half-node first-derivative weights (×1920 scale) on
# offsets {−5/2, −3/2, −1/2, 1/2, 3/2, 5/2} (units of dx, midpoint-centred).
STENCIL_INT = (-9, 125, -2250, 2250, -125, 9)
STENCIL_DEN = 1920
HALF_OFFSETS = (-5, -3, -1, 1, 3, 5)  # in units of dx/2


def emit_pass(label: str) -> None:
    print(f"  [PASS] {label}")


def emit_fail(label: str, reason: str) -> str:
    print(f"  [FAIL] {label}: {reason}", flush=True)
    return reason


def _lowest_power(sp, expr, dx):
    expr = sp.expand(expr)
    poly = sp.Poly(expr, dx)
    lowest = None
    for monom, coeff in poly.terms():
        if sp.simplify(coeff) != 0:
            p = monom[0]
            if lowest is None or p < lowest:
                lowest = p
    return lowest


# ---------------------------------------------------------------------------
# Sub-check 1 — 6th-order staggered half-node first derivative
# ---------------------------------------------------------------------------


def check_half_node_derivative_6th() -> str | None:
    label = "(1) 6-point staggered half-node derivative is O(dx⁶)"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    dx = sp.Symbol("dx", positive=True)
    x = sp.Symbol("x", real=True)
    f = sp.Function("f")

    # Midpoint x; nodes at x + off·dx/2 for off in HALF_OFFSETS.
    stencil = sp.Add(
        *[STENCIL_INT[j] * f(x + sp.Rational(HALF_OFFSETS[j], 2) * dx) for j in range(6)]
    ) / (STENCIL_DEN * dx)

    true_deriv = sp.diff(f(x), x)
    residual = sp.series(stencil - true_deriv, dx, 0, 8).removeO()
    lowest = _lowest_power(sp, residual, dx)
    if lowest is None:
        lowest = 8

    print("    Stencil: (−9 f_{i-2} + 125 f_{i-1} − 2250 f_i")
    print("              + 2250 f_{i+1} − 125 f_{i+2} + 9 f_{i+3}) / (1920 dx)")
    print(f"    Leading truncation power of dx: {lowest}")
    if lowest < 6:
        return emit_fail(label, f"leading truncation is dx^{lowest}, expected ≥ dx^6")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check 2 — constant-a composition is O(dx⁶)
# ---------------------------------------------------------------------------


def _half_node_deriv(sp, dx, f, center):
    return sp.Add(
        *[STENCIL_INT[j] * f(center + sp.Rational(HALF_OFFSETS[j], 2) * dx) for j in range(6)]
    ) / (STENCIL_DEN * dx)


def check_constant_a_composition() -> str | None:
    label = "(2) const-a composed operator is O(dx⁶) (6th-order Laplacian)"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    dx = sp.Symbol("dx", positive=True)
    x = sp.Symbol("x", real=True)
    f = sp.Function("f")

    # For a ≡ 1 the flux F = ∂_x f. Outer 6th-order divergence of half-node fluxes:
    # apply the same staggered weights to F at half-node centres around x.
    composed = sp.Add(
        *[
            STENCIL_INT[j]
            * _half_node_deriv(sp, dx, f, x + sp.Rational(HALF_OFFSETS[j], 2) * dx)
            for j in range(6)
        ]
    ) / (STENCIL_DEN * dx)

    true_lap = sp.diff(f(x), x, 2)
    residual = sp.series(composed - true_lap, dx, 0, 8).removeO()
    lowest = _lowest_power(sp, residual, dx)
    if lowest is None:
        lowest = 8

    print(f"    Composed const-a operator leading truncation power: dx^{lowest}")
    if lowest < 6:
        return emit_fail(label, f"const-a composition is O(dx^{lowest}), expected O(dx^6)")
    print("    Conservative flux-difference form, 6th-order accurate for a ≡ 1.")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check 3 — variable-a truncation O(dx⁶)
# ---------------------------------------------------------------------------


def check_variable_a_truncation() -> str | None:
    label = "(3) variable-a truncation is O(dx⁶) (dx²,dx³,dx⁴,dx⁵ coeffs = 0)"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    dx = sp.Symbol("dx", positive=True)
    x = sp.Symbol("x", real=True)
    f = sp.Function("f")
    a = sp.Function("a")

    def half_node_flux(center):
        deriv = _half_node_deriv(sp, dx, f, center)
        return a(center) * deriv  # a EXACT at the half-node (no FD error).

    composed = sp.Add(
        *[
            STENCIL_INT[j] * half_node_flux(x + sp.Rational(HALF_OFFSETS[j], 2) * dx)
            for j in range(6)
        ]
    ) / (STENCIL_DEN * dx)

    true_op = sp.diff(a(x) * sp.diff(f(x), x), x)  # = a' f' + a f''.
    residual = sp.series(composed - true_op, dx, 0, 7).removeO()
    poly = sp.Poly(sp.expand(residual), dx)

    for p in (2, 3, 4, 5):
        coeff = sp.simplify(poly.coeff_monomial(dx**p))
        print(f"    dx^{p} coefficient of truncation: {coeff}")
        if coeff != 0:
            return emit_fail(label, f"dx^{p} coefficient ≠ 0: {coeff} → order < 6")

    lowest = _lowest_power(sp, residual, dx)
    if lowest is None:
        lowest = 7
    print(f"    Leading truncation power (variable a): dx^{lowest}")
    if lowest < 6:
        return emit_fail(label, f"variable-a truncation is O(dx^{lowest})")
    print("    Genuine 6th-order for ∂_x(a ∂_x f) with arbitrary smooth a, f. ✓")
    emit_pass(label)
    return None


# ---------------------------------------------------------------------------
# Sub-check 4 — conservation form (telescoping flux difference)
# ---------------------------------------------------------------------------


def check_conservation_form() -> str | None:
    label = "(4) flux-difference form is conservative (telescoping → Neumann mass)"
    try:
        import sympy as sp
    except ImportError:
        return emit_fail(label, "sympy not installed")

    z = sp.Symbol("z")
    gen = sp.Add(*[STENCIL_INT[j] * z**j for j in range(6)])
    quotient, remainder = sp.div(gen, 1 - z, z)
    remainder = sp.simplify(remainder)
    print("    Stencil generating polynomial:")
    print("      −9 + 125z − 2250z² + 2250z³ − 125z⁴ + 9z⁵")
    print(f"    Divided by (1 − z): quotient = {sp.expand(quotient)}, remainder = {remainder}")
    if remainder != 0:
        return emit_fail(
            label,
            f"generating polynomial NOT divisible by (1−z): remainder {remainder} "
            "→ operator is NOT in conservation form, Neumann mass not preserved.",
        )
    print("    Remainder = 0 → operator IS a discrete divergence (telescoping).")
    print("    Conservation form ⇒ Neumann boundary mass preserved (mirror 4th-order form).")
    emit_pass(label)
    return None


def main() -> int:
    print("=" * 78)
    print("T_DIV_STENCIL_6TH — ADR-0118 (6th-order) PRE-FLIGHT sympy oracle")
    print("=" * 78)
    print()
    print("Target: 7-point 6th-order flux-difference discretisation of ∂_x(a ∂_x f)")
    print("        → spatial floor T·dx⁴ → T·dx⁶ (opens ζ⁶/ζ⁸ gate window at N=4096).")
    print()
    print("Sub-checks:")

    checks = [
        ("1", check_half_node_derivative_6th),
        ("2", check_constant_a_composition),
        ("3", check_variable_a_truncation),
        ("4", check_conservation_form),
    ]
    failures: list[str] = []
    for letter, fn in checks:
        print()
        result = fn()
        if result is not None:
            failures.append(f"({letter}) {result}")

    print()
    print("=" * 78)
    if failures:
        print(f"T_DIV_STENCIL_6TH FAIL ({len(failures)}/4 sub-checks):")
        for fmsg in failures:
            print(f"  - {fmsg}")
        return 1
    print("T_DIV_STENCIL_6TH PASS (4/4 sub-checks:")
    print(" half_node_derivative_6th / constant_a_composition /")
    print(" variable_a_truncation / conservation_form)")
    print()
    print("ARCHITECTURAL CONCLUSION:")
    print("  - 6-point staggered 6th-order divergence stencil is mathematically sound.")
    print("  - Half-node derivative weights (−9,125,−2250,2250,−125,9)/(1920 dx) are O(dx⁶).")
    print("  - Composed operator ∂_x(a ∂_x f) truncation is O(dx⁶) for arbitrary a, f.")
    print("  - Conservation form preserved (generating poly divisible by 1−z).")
    print("  - Spatial floor T·dx⁴ → T·dx⁶ (further factor dx² ≈ 2.4e-5 at N=4096).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
