#!/usr/bin/env python3
"""T_DIRICHLET_ORDER2 sympy gate — odd-image symbolic identity for Dirichlet BC.

Per ADR-0176 / math.md §21.9 (NORMATIVE):

  K^D(x, y, t) := K(x, y, t) − K(x, −y, t)        [MINUS sign vs T22N's PLUS]
  K(x, y, t)   := (4πt)^{-1/2} · exp(−(x − y)² / (4t))

Two mandatory sub-checks (both must pass for T_DIRICHLET_ORDER2 PASS):

  (1) (T_DIRICHLET_ORDER2.heat_pde) Heat-PDE residual:
      ∂_t K^D == ∂_xx K^D  symbolically in (0,∞)×(0,∞).
      Expected: simplify(∂_t K^D − ∂_xx K^D) == 0.

  (2) (T_DIRICHLET_ORDER2.dirichlet_boundary) Dirichlet BC at x=0:
      K^D(x=0, y, t) == 0  symbolically (for y > 0, t > 0).
      The two terms K(0, y, t) and K(0, −y, t) are equal (both use (0±y)²),
      so they cancel exactly. Mirror of T22N.neumann_boundary which checks
      ∂_x K^N|_{x=0} = 0; here the VALUE (not derivative) vanishes.

Prints exactly 'T_DIRICHLET_ORDER2 PASS' on success or
'T_DIRICHLET_ORDER2 FAIL: <reason>' on failure; exits 1 on failure.
Pure symbolic — no Rust runtime required.
"""

import sys


def fail(msg: str) -> int:
    """Print failure message and return exit code 1."""
    print(f"T_DIRICHLET_ORDER2 FAIL: {msg}", flush=True)
    return 1


def check_heat_pde(sp, x, t, K_D) -> bool:
    """Check (1): ∂_t K^D − ∂_xx K^D == 0 symbolically."""
    dt_KD = sp.diff(K_D, t)
    dxx_KD = sp.diff(K_D, x, 2)
    residual = sp.simplify(dt_KD - dxx_KD)
    if residual != 0:
        print(f"  T_DIRICHLET_ORDER2.heat_pde: residual = {residual} (expected 0)", flush=True)
        return False
    print("  T_DIRICHLET_ORDER2.heat_pde: PASS (∂_t K^D − ∂_xx K^D = 0)", flush=True)
    return True


def check_dirichlet_bc(sp, x, y, t, K_D) -> bool:
    """Check (2): K^D(x=0, y, t) == 0 symbolically.

    At x=0: K(0,y,t) = (4πt)^{-1/2} exp(−y²/(4t))
             K(0,−y,t) = (4πt)^{-1/2} exp(−(0−(−y))²/(4t)) = same as K(0,y,t)
    So K^D(0,y,t) = K(0,y,t) − K(0,y,t) = 0. The odd extension vanishes at the wall.
    """
    bc_val = sp.simplify(K_D.subs(x, 0))
    if bc_val != 0:
        print(
            f"  T_DIRICHLET_ORDER2.dirichlet_boundary: K^D(0,y,t) = {bc_val} (expected 0)",
            flush=True,
        )
        return False
    print("  T_DIRICHLET_ORDER2.dirichlet_boundary: PASS (K^D(0,y,t) = 0)", flush=True)
    return True


def main() -> int:
    """Run all T_DIRICHLET_ORDER2 sub-checks. Return 0 on success, 1 on failure."""
    try:
        import sympy as sp
    except ImportError:
        return fail("sympy not installed — install via: pip install sympy")

    # Symbolic variables: x, y, t all strictly positive.
    x, y, t = sp.symbols("x y t", positive=True)

    # Free-space 1D heat kernel K(x, y, t) = (4πt)^{-1/2} exp(−(x−y)²/(4t)).
    K_free = (4 * sp.pi * t) ** sp.Rational(-1, 2) * sp.exp(-(x - y) ** 2 / (4 * t))

    # Odd-image kernel K^D = K(x, y, t) − K(x, −y, t)  [MINUS sign — Dirichlet].
    # Ghost term: K(x, −y, t) = (4πt)^{-1/2} exp(−(x+y)²/(4t)).
    K_ghost = (4 * sp.pi * t) ** sp.Rational(-1, 2) * sp.exp(-(x + y) ** 2 / (4 * t))
    K_D = K_free - K_ghost  # odd (antisymmetric) combination

    print("T_DIRICHLET_ORDER2: running 2 mandatory sub-checks ...", flush=True)

    # Mandatory check (1): Heat PDE residual.
    if not check_heat_pde(sp, x, t, K_D):
        return fail("T_DIRICHLET_ORDER2.heat_pde: K^D does not satisfy ∂_t K = ∂_xx K")

    # Mandatory check (2): Dirichlet BC at x=0.
    if not check_dirichlet_bc(sp, x, y, t, K_D):
        return fail("T_DIRICHLET_ORDER2.dirichlet_boundary: K^D(0,y,t) ≠ 0")

    print("T_DIRICHLET_ORDER2 PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
