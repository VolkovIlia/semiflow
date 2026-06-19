#!/usr/bin/env python3
# pyright: reportArgumentType=false
#
# sympy.Matrix.inv() return type is dynamically typed (Unknown | SparseMatrix);
# Pyright cannot resolve it to Matrix. All calls are valid at runtime.
"""T_MANIFOLD_CURVATURE: symbolic verification of scalar curvature for all 3
closed-form backends (v2.8, ADR-0071, math.md §24.4).

Verifies that the Riemann scalar curvature computed from the metric tensor
matches the values hard-coded in the Rust implementations:
  1. Torus<F, 2>  — flat metric  → R ≡ 0
  2. Sphere2<F>   — sphere metric → R ≡ 2/r²  (R=2 for unit sphere)
  3. Hyperbolic2<F> — Poincaré disk metric → R ≡ -2/s²  (R=-2 for unit disk)

Prints MANIFOLD_CURVATURE PASS on success.
Prints MANIFOLD_CURVATURE FAIL: <backend>: <reason> and exits 1 on failure.

Usage:
    python3 scripts/verify_manifold_curvature.py
"""

import os
import sys

# Allow import of manifold_curvature_kit from the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sympy as sp
from manifold_curvature_kit import (  # pyright: ignore[reportMissingImports]
    christoffel_symbols,
    ricci_tensor,
    riemann_curvature,
    scalar_curvature,
)


def check_torus() -> tuple[bool, str]:
    """Flat T² metric → scalar curvature must be identically zero.

    Metric: g = diag(1, 1)  (standard flat metric in (x, y) coordinates).
    """
    x, y = sp.symbols("x y", real=True)
    g = sp.Matrix([[1, 0], [0, 1]])
    gamma = christoffel_symbols(g, (x, y))
    riem = riemann_curvature(gamma, (x, y))
    ric = ricci_tensor(riem, (x, y))
    r_scalar = scalar_curvature(ric, g.inv())
    if sp.simplify(r_scalar) != 0:
        return False, f"scalar curvature = {r_scalar}, expected 0"
    return True, ""


def check_sphere2() -> tuple[bool, str]:
    """S² with radius r: scalar curvature must equal 2/r².

    Metric in (θ, φ) spherical coordinates:
        g = diag(r², r²·sin²(θ))
    The R = 2/r² result follows from the standard sphere curvature formula.
    """
    r, theta, phi = sp.symbols("r theta phi", positive=True)
    g = sp.Matrix([
        [r**2, 0],
        [0, r**2 * sp.sin(theta)**2],
    ])
    gamma = christoffel_symbols(g, (theta, phi))
    riem = riemann_curvature(gamma, (theta, phi))
    ric = ricci_tensor(riem, (theta, phi))
    r_scalar = scalar_curvature(ric, g.inv())
    expected = sp.Integer(2) / r**2
    diff = sp.simplify(r_scalar - expected)
    if diff != 0:
        return False, f"scalar curvature = {r_scalar}, expected {expected}, diff = {diff}"
    return True, ""


def check_hyperbolic2() -> tuple[bool, str]:
    """Poincaré disk H² with scale s: scalar curvature must equal -2/s².

    Poincaré disk metric in (u, v) coordinates (|z| < 1):
        g = (4·s²)/(1 − u² − v²)² · I₂
    The R = -2/s² result follows from the standard hyperbolic plane formula.
    """
    s, u, v = sp.symbols("s u v", positive=True)
    factor = 4 * s**2 / (1 - u**2 - v**2)**2
    g = sp.Matrix([[factor, 0], [0, factor]])
    gamma = christoffel_symbols(g, (u, v))
    riem = riemann_curvature(gamma, (u, v))
    ric = ricci_tensor(riem, (u, v))
    r_scalar = scalar_curvature(ric, g.inv())
    expected = sp.Integer(-2) / s**2
    diff = sp.simplify(r_scalar - expected)
    if diff != 0:
        return False, f"scalar curvature = {r_scalar}, expected {expected}, diff = {diff}"
    return True, ""


def main() -> int:
    """Run all three curvature checks and report results."""
    checks = [
        ("torus", check_torus),
        ("sphere2", check_sphere2),
        ("hyperbolic2", check_hyperbolic2),
    ]
    for name, check in checks:
        ok, reason = check()
        if not ok:
            print(f"MANIFOLD_CURVATURE FAIL: {name}: {reason}", flush=True)
            return 1
    print("MANIFOLD_CURVATURE PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
