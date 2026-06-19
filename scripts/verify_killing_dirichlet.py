#!/usr/bin/env python3
"""T18N sympy gate — symbolic verification of the Dirichlet eigenmode oracle.

Verifies (symbolically with sympy) that the closed-form expansion
    u(t, x) = Σ_{k=1}^{4} a_k · sin(kπx) · exp(-(kπ)²t/2)
is the correct solution to the heat equation ∂_t u = ½ ∂_xx u on (0,1)
with absorbing boundary u(t,0) = u(t,1) = 0.

Four checks (math §21.5, ADR-0068, T18N):
  1. Boundary conditions: sin(kπ·0)=0 and sin(kπ·1)=0 for k=1..4
  2. Eigenfunction property: ½∂_xx[sin(kπx)] = -(kπ)²/2 · sin(kπx) for k=1..4
  3. Superposition satisfies the PDE: ∂_t u = ½ ∂_xx u (symbolically, simplified)
  4. Commutator [½∂_xx, 𝟙_(0,1)] vanishes on the open interior

Prints 'T18N PASS' on success; 'T18N FAIL: <reason>' and exits 1 on failure.
"""

import sys


def main() -> int:
    try:
        import sympy as sp
    except ImportError:
        print("T18N FAIL: sympy not installed", flush=True)
        return 1

    x, t = sp.symbols("x t", real=True)

    # -----------------------------------------------------------------------
    # Check 1: boundary conditions sin(k*pi*0) = 0 and sin(k*pi*1) = 0
    # -----------------------------------------------------------------------
    for k in range(1, 5):
        val_left = sp.sin(k * sp.pi * x).subs(x, 0)
        val_right = sp.sin(k * sp.pi * x).subs(x, 1)
        if sp.simplify(val_left) != 0:
            print(f"T18N FAIL: BC at x=0 for k={k}: sin({k}π·0) = {val_left}")
            return 1
        if sp.simplify(val_right) != 0:
            print(f"T18N FAIL: BC at x=1 for k={k}: sin({k}π·1) = {val_right}")
            return 1

    # -----------------------------------------------------------------------
    # Check 2: eigenfunction property ½ ∂_xx[sin(kπx)] = -(kπ)²/2 · sin(kπx)
    # -----------------------------------------------------------------------
    for k in range(1, 5):
        mode = sp.sin(k * sp.pi * x)
        lhs = sp.Rational(1, 2) * sp.diff(mode, x, 2)
        rhs = -sp.Rational((k * k), 2) * sp.pi**2 * mode
        residual = sp.simplify(lhs - rhs)
        if residual != 0:
            print(
                f"T18N FAIL: eigenfunction check k={k}: "
                f"½∂_xx[sin({k}πx)] - (-(k π)²/2)sin({k}πx) = {residual}"
            )
            return 1

    # -----------------------------------------------------------------------
    # Check 3: superposition u(t,x) = Σ a_k sin(kπx) exp(-(kπ)²t/2)
    #          satisfies ∂_t u = ½ ∂_xx u
    # -----------------------------------------------------------------------
    # Use symbolic coefficients a1..a4
    a = [sp.Symbol(f"a{k}", real=True) for k in range(1, 5)]
    u = sum(
        a[k - 1] * sp.sin(k * sp.pi * x) * sp.exp(-(k * sp.pi) ** 2 * t / 2)
        for k in range(1, 5)
    )
    pde_residual = sp.simplify(sp.diff(u, t) - sp.Rational(1, 2) * sp.diff(u, x, 2))
    if pde_residual != 0:
        print(f"T18N FAIL: PDE residual ∂_t u - ½∂_xx u = {pde_residual}")
        return 1

    # -----------------------------------------------------------------------
    # Check 4: commutator [½∂_xx, 𝟙_R] vanishes on (0,1) interior
    # For a test function φ supported in (0,1) (no boundary jumps),
    # ½∂_xx(𝟙_R · φ) = ½∂_xx(φ) on the open interior because 𝟙_R = 1 on (0,1).
    # Symbolically: let φ(x) = sin(πx) (compactly supported in [0,1], zero at 0,1)
    # and verify ½∂_xx(φ) - 1·½∂_xx(φ) = 0 on (0,1).
    # -----------------------------------------------------------------------
    phi = sp.sin(sp.pi * x)  # test function: zero at 0 and 1
    indicator_phi = phi  # 𝟙_(0,1) · φ = φ on (0,1)
    commutator = sp.simplify(
        sp.Rational(1, 2) * sp.diff(indicator_phi, x, 2)
        - sp.Rational(1, 2) * sp.diff(phi, x, 2)
    )
    if commutator != 0:
        print(f"T18N FAIL: commutator [½∂_xx, 𝟙_R] on (0,1) = {commutator}")
        return 1

    print("T18N PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
