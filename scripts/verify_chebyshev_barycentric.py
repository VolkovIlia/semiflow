#!/usr/bin/env python3
"""T_CHEB — Chebyshev barycentric Lagrange verification oracle (ADR-0090, AC7).

4 sub-checks (NORMATIVE):

  (a) Node formula: x_k = cos(k·π/M) on [-1,1] matches Chebyshev-Lobatto definition.
  (b) Weight formula: w_k = (-1)^k · δ_k (δ_0=δ_M=1/2, δ_k=1 else) is correct.
  (c) Barycentric formula recovers polynomial exactly for degree-M polynomial on [-1,1].
  (d) Spectral floor: for f(x)=exp(-x²) on [-5,5], M=64 interpolation error ≤ 1e-10.

References:
  Berrut & Trefethen 2004, SIAM Review 46:501 — barycentric formula (5.1).
  Trefethen 2000 Spectral Methods in MATLAB — Lobatto nodes + weights.
  ADR-0090 §"Algorithm" — normative Chebyshev spectral collocation spec.
  math.md §9.2.7 — normative section.
"""

import math
import sys

# ---------------------------------------------------------------------------
# Sub-check (a): Chebyshev-Lobatto node formula
# ---------------------------------------------------------------------------

def check_a_node_formula(m: int) -> bool:
    """Verify x_k = cos(k*pi/M) for k=0..M on [-1,1]."""
    nodes = [math.cos(k * math.pi / m) for k in range(m + 1)]
    # Check endpoints
    if abs(nodes[0] - 1.0) > 1e-15:
        print(f"  FAIL (a): nodes[0] = {nodes[0]:.6e}, expected 1.0", file=sys.stderr)
        return False
    if abs(nodes[m] - (-1.0)) > 1e-15:
        print(f"  FAIL (a): nodes[{m}] = {nodes[m]:.6e}, expected -1.0", file=sys.stderr)
        return False
    # Check midpoint (M even): cos(M/2 * pi/M) = cos(pi/2) = 0.
    if m % 2 == 0:
        mid_idx = m // 2
        if abs(nodes[mid_idx]) > 1e-15:
            print(f"  FAIL (a): nodes[{mid_idx}] = {nodes[mid_idx]:.6e}, expected 0.0", file=sys.stderr)
            return False
    # Check all nodes in [-1, 1]
    for k, xk in enumerate(nodes):
        if xk < -1.0 - 1e-14 or xk > 1.0 + 1e-14:
            print(f"  FAIL (a): nodes[{k}] = {xk:.6e} out of [-1,1]", file=sys.stderr)
            return False
    # Verify sorted strictly decreasing (cos is decreasing on [0,pi])
    for k in range(m):
        if nodes[k] <= nodes[k + 1]:
            print(f"  FAIL (a): nodes not strictly decreasing at k={k}", file=sys.stderr)
            return False
    print(f"  PASS (a): node formula correct for M={m} (endpoints ±1, midpoint 0, decreasing)")
    return True


# ---------------------------------------------------------------------------
# Sub-check (b): Barycentric weight formula
# ---------------------------------------------------------------------------

def chebyshev_weights(m: int) -> list:
    """Compute barycentric weights: w_k = (-1)^k * delta_k."""
    weights = []
    for k in range(m + 1):
        sign = (-1) ** k
        delta = 0.5 if (k == 0 or k == m) else 1.0
        weights.append(sign * delta)
    return weights


def check_b_weight_formula(m: int) -> bool:
    """Verify w_0 = 0.5, w_M = ±0.5, w_k = ±1 for 0<k<M."""
    weights = chebyshev_weights(m)
    # Check endpoints have magnitude 0.5
    if abs(abs(weights[0]) - 0.5) > 1e-15:
        print(f"  FAIL (b): |w_0| = {abs(weights[0]):.6e}, expected 0.5", file=sys.stderr)
        return False
    if abs(abs(weights[m]) - 0.5) > 1e-15:
        print(f"  FAIL (b): |w_M| = {abs(weights[m]):.6e}, expected 0.5", file=sys.stderr)
        return False
    # Check interior weights have magnitude 1.0
    for k in range(1, m):
        if abs(abs(weights[k]) - 1.0) > 1e-15:
            print(f"  FAIL (b): |w_{k}| = {abs(weights[k]):.6e}, expected 1.0", file=sys.stderr)
            return False
    # Check alternating sign: w_k * w_{k+1} < 0 for k=0..M-1
    for k in range(m):
        if weights[k] * weights[k + 1] >= 0:
            print(f"  FAIL (b): weights[{k}], weights[{k+1}] don't alternate in sign", file=sys.stderr)
            return False
    print(f"  PASS (b): weight formula correct for M={m} (endpoints ±0.5, interior ±1, alternating)")
    return True


# ---------------------------------------------------------------------------
# Barycentric evaluation
# ---------------------------------------------------------------------------

def barycentric_eval(nodes: list, weights: list, f_vals: list, x: float) -> float:
    """Evaluate barycentric Lagrange interpolant at x.

    Returns f_k directly when |x - x_k| < EPSILON_GUARD (removable-singularity).
    """
    epsilon_guard = 8.0 * 2.220446049250313e-16  # 8 * machine epsilon
    num = 0.0
    den = 0.0
    for _, (xk, wk, fk) in enumerate(zip(nodes, weights, f_vals)):
        diff = x - xk
        if abs(diff) < epsilon_guard:
            return fk
        term = wk / diff
        num += term * fk
        den += term
    return num / den


# ---------------------------------------------------------------------------
# Sub-check (c): Polynomial exactness
# ---------------------------------------------------------------------------

def check_c_polynomial_exactness(m: int) -> bool:
    """Verify that barycentric formula interpolates a degree-M polynomial exactly.

    Uses p(x) = x^M + x^(M-1) + ... + x + 1 (Horner evaluation).
    Tests at 5 interior points not coinciding with Lobatto nodes.
    """
    nodes = [math.cos(k * math.pi / m) for k in range(m + 1)]
    weights = chebyshev_weights(m)

    def poly(x: float) -> float:
        # p(x) = sum_{j=0}^{M} x^j via Horner
        result = 0.0
        for _ in range(m, -1, -1):
            result = result * x + 1.0
        return result

    f_vals = [poly(xk) for xk in nodes]

    # Test at 5 interior probe points (not Lobatto nodes)
    probes = [-0.73, -0.31, 0.0, 0.45, 0.81]
    max_err = 0.0
    for xp in probes:
        approx = barycentric_eval(nodes, weights, f_vals, xp)
        exact = poly(xp)
        err = abs(approx - exact)
        max_err = max(max_err, err)

    tol = 1e-10
    if max_err > tol:
        print(f"  FAIL (c): degree-{m} polynomial exactness: max_err = {max_err:.4e} > {tol:.0e}", file=sys.stderr)
        return False
    print(f"  PASS (c): polynomial exactness (M={m}): max_err = {max_err:.4e} ≤ {tol:.0e}")
    return True


# ---------------------------------------------------------------------------
# Sub-check (d): Spectral floor for Gaussian
# ---------------------------------------------------------------------------

def check_d_spectral_floor(m: int = 64) -> bool:
    """Verify Chebyshev M=64 achieves spectral floor ≤ 1e-10 for Gaussian on [-5,5].

    Uses exact analytic values at Chebyshev-Lobatto nodes to isolate the barycentric
    formula accuracy from virtual-node sampling noise. This tests pure spectral
    convergence of the barycentric interpolant (ADR-0090 §"Spectral floor").

    The exp(-x²) Gaussian is entire (analytic everywhere), so its Chebyshev
    interpolation error is exp(-M·γ) with γ = acosh(1 + gap) where gap is the
    distance from [-5,5] to the nearest singularity. At M=64, the Chebyshev
    tail truncation error is ≈ exp(-64·1.4) ≈ 1e-39 — well below 1e-10.
    In practice at M=64 the floating-point noise floor (~1e-15) dominates.
    """
    xmin, xmax = -5.0, 5.0
    mid = (xmax + xmin) * 0.5
    half = (xmax - xmin) * 0.5

    # Chebyshev-Lobatto nodes mapped to [xmin, xmax]
    nodes_01 = [math.cos(k * math.pi / m) for k in range(m + 1)]
    nodes_phys = [mid + half * yk for yk in nodes_01]
    weights = chebyshev_weights(m)

    # Exact analytic values at Chebyshev nodes (no virtual-sampling error).
    f_exact_at_nodes = [math.exp(-xk * xk) for xk in nodes_phys]

    # Evaluate at 50 probe points between nodes (cell midpoints in [-1,1]).
    n_probes = 50
    probes_01 = [math.cos((k + 0.5) * math.pi / n_probes) for k in range(n_probes)]
    probes_phys = [mid + half * yp for yp in probes_01]

    max_err = 0.0
    for xp in probes_phys:
        approx = barycentric_eval(nodes_phys, weights, f_exact_at_nodes, xp)
        exact = math.exp(-xp * xp)
        err = abs(approx - exact)
        max_err = max(max_err, err)

    # At M=64, spectral accuracy for entire Gaussian: floor ≈ machine epsilon (~1e-15).
    # Gate: ≤ 1e-10 (conservative, includes floating-point accumulation budget).
    tol = 1e-10
    if max_err > tol:
        print(
            f"  FAIL (d): spectral floor M={m}: max_err = {max_err:.4e} > {tol:.0e}",
            file=sys.stderr,
        )
        print(
            "  Note: sub-check (d) tests pure barycentric accuracy with exact node values.",
            file=sys.stderr,
        )
        return False
    print(f"  PASS (d): spectral floor (M={m}): max_err = {max_err:.4e} ≤ {tol:.0e}")
    return True


# ---------------------------------------------------------------------------
# Main: run all 4 sub-checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("T_CHEB — Chebyshev barycentric Lagrange verification (ADR-0090, math.md §9.2.7)")
    print()

    passed = 0
    total = 4

    print("Sub-check (a): Chebyshev-Lobatto node formula [M=64]")
    if check_a_node_formula(64):
        passed += 1
    print()

    print("Sub-check (b): Barycentric weight formula [M=64]")
    if check_b_weight_formula(64):
        passed += 1
    print()

    print("Sub-check (c): Polynomial exactness (degree-M) [M=16]")
    if check_c_polynomial_exactness(16):
        passed += 1
    print()

    print("Sub-check (d): Spectral floor for Gaussian exp(-x²) on [-5,5] [M=64, exact node values]")
    if check_d_spectral_floor(64):
        passed += 1
    print()

    print(f"T_CHEB result: {passed}/{total} sub-checks passed")
    if passed < total:
        print(f"FAIL: {total - passed} sub-check(s) failed", file=sys.stderr)
        sys.exit(1)
    else:
        print("T_CHEB: ALL PASS")
        sys.exit(0)
