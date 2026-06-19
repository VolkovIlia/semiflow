#!/usr/bin/env python3
"""T22N sympy gate — image-method symbolic identity for half-line reflected heat.

Per properties.yaml (T22N, NORMATIVE, ADR-0072, math.md §25.5):

  K_N(x, y, t) := K(x, y, t) + K(x, -y, t)
  K(x, y, t)   := (4πt)^{-1/2} · exp(-(x - y)² / (4t))

Three MANDATORY sub-checks (all must pass for T22N PASS):

  (1) (T22N.heat_pde) Heat-PDE residual:
      ∂_t K_N == ∂_xx K_N  symbolically in (0,∞)×(0,∞).
      Expected: simplify(∂_t K_N - ∂_xx K_N) == 0.

  (2) (T22N.neumann_boundary) Neumann boundary at x=0:
      ∂_x K_N(x, y, t) |_{x=0} == 0  symbolically.
      The two derivatives cancel exactly (image method's defining property).

  (3) (T22N.initial_delta) Ghost contribution decays to zero as t → 0+:
      limit(K(x, -y, t), t, 0, '+') == 0  for x, y > 0.
      (The direct term K(x, y, t) recovers δ(x-y); the ghost K(x, -y, t)
       decays exponentially since (x+y)²/(4t) → ∞.)

BONUS sub-check (not gated, advisory only):

  (4) (T22N.mass_conservation) Mass conservation:
      ∫_0^∞ K_N(x, y, t) dy == 1  for x > 0, t > 0.
      May depend on sympy erf coverage; prints PASS/FAIL but does not block.

Prints exactly 'T22N PASS' on success (all 3 mandatory pass) or
'T22N FAIL: <reason>' on failure (any mandatory fails). Exits 1 on failure.
"""

import sys


def fail(msg: str) -> int:
    """Print failure message and return exit code 1."""
    print(f"T22N FAIL: {msg}", flush=True)
    return 1


def check_heat_pde(sp, x, t, K_N) -> bool:
    """Check (1): ∂_t K_N - ∂_xx K_N == 0 symbolically."""
    dt_KN = sp.diff(K_N, t)
    dxx_KN = sp.diff(K_N, x, 2)
    residual = sp.simplify(dt_KN - dxx_KN)
    if residual != 0:
        print(f"  T22N.heat_pde: residual = {residual} (expected 0)", flush=True)
        return False
    print("  T22N.heat_pde: PASS (∂_t K_N - ∂_xx K_N = 0)", flush=True)
    return True


def check_neumann_bc(sp, x, K_N) -> bool:
    """Check (2): ∂_x K_N |_{x=0} == 0 symbolically."""
    dK_dx = sp.diff(K_N, x)
    bc_val = sp.simplify(dK_dx.subs(x, 0))
    if bc_val != 0:
        print(f"  T22N.neumann_boundary: ∂_x K_N|_{{x=0}} = {bc_val} (expected 0)", flush=True)
        return False
    print("  T22N.neumann_boundary: PASS (∂_x K_N|_{x=0} = 0)", flush=True)
    return True


def check_initial_delta(sp, x, y, t) -> bool:
    """Check (3): limit(K(x, -y, t), t→0+) == 0 for x, y > 0.

    The ghost term K(x, -y, t) = (4πt)^{-1/2} exp(-(x+y)²/(4t)).
    For x, y > 0, (x+y)² > 0, so (x+y)²/(4t) → ∞ as t → 0+,
    meaning the exponential dominates (4πt)^{-1/2} → ∞.
    """
    # Free-space kernel for the ghost: K(x, -y, t) = (4πt)^{-1/2} exp(-(x+y)²/(4t))
    K_ghost = (4 * sp.pi * t) ** sp.Rational(-1, 2) * sp.exp(-(x + y) ** 2 / (4 * t))
    lim_val = sp.limit(K_ghost, t, 0, "+")
    if lim_val != 0:
        print(f"  T22N.initial_delta: limit K(x,-y,t) as t→0+ = {lim_val} (expected 0)", flush=True)
        return False
    print("  T22N.initial_delta: PASS (ghost contribution → 0 as t → 0+)", flush=True)
    return True


def check_mass_conservation(sp, y, K_N) -> None:
    """Bonus (4): ∫_0^∞ K_N dy == 1. Not gated — print PASS/FAIL advisory."""
    try:
        mass = sp.integrate(K_N, (y, 0, sp.oo))
        mass_simplified = sp.simplify(mass)
        if sp.simplify(mass_simplified - 1) == 0:
            print("  T22N.mass_conservation (BONUS): PASS (mass = 1)", flush=True)
        else:
            print(f"  T22N.mass_conservation (BONUS): FAIL (mass = {mass_simplified})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  T22N.mass_conservation (BONUS): SKIP (sympy error: {exc})", flush=True)


def main() -> int:
    """Run all T22N sub-checks. Return 0 on success, 1 on failure."""
    try:
        import sympy as sp
    except ImportError:
        return fail("sympy not installed — install via: pip install sympy")

    # Symbolic variables: x, y, t all strictly positive.
    x, y, t = sp.symbols("x y t", positive=True)

    # Free-space 1D heat kernel K(x, y, t) = (4πt)^{-1/2} exp(-(x-y)²/(4t)).
    K_free = (4 * sp.pi * t) ** sp.Rational(-1, 2) * sp.exp(-(x - y) ** 2 / (4 * t))

    # Image-method kernel K_N = K(x, y, t) + K(x, -y, t).
    # Note: for x, y > 0, the ghost term is K(x, -y, t) which evaluates
    # at the reflected point -y. Since -y < 0 for y > 0, this is the
    # contribution from outside the half-line [0, ∞).
    K_ghost_direct = (4 * sp.pi * t) ** sp.Rational(-1, 2) * sp.exp(-(x + y) ** 2 / (4 * t))
    K_N = K_free + K_ghost_direct

    print("T22N: running 3 mandatory + 1 bonus sub-check ...", flush=True)

    # Mandatory check (1): Heat PDE residual.
    if not check_heat_pde(sp, x, t, K_N):
        return fail("T22N.heat_pde: K_N does not satisfy ∂_t K = ∂_xx K")

    # Mandatory check (2): Neumann BC at x=0.
    if not check_neumann_bc(sp, x, K_N):
        return fail("T22N.neumann_boundary: ∂_x K_N|_{x=0} ≠ 0")

    # Mandatory check (3): Ghost decays to zero as t → 0+.
    if not check_initial_delta(sp, x, y, t):
        return fail("T22N.initial_delta: ghost K(x,-y,t) does not decay to 0")

    # Bonus check (4): Mass conservation (not gated).
    check_mass_conservation(sp, y, K_N)

    print("T22N PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
