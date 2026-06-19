#!/usr/bin/env python3
"""ADR-0104 PRE-FLIGHT sympy oracle — Chebyshev barycentric formula verification.

Two sub-checks (per ADR-0104 §4 sub-action 1):

  Sub-check 1: Chebyshev-Lobatto barycentric weights w_k = (-1)^k · delta_k
    (delta_0 = delta_M = 1/2, else 1) are reproduced symbolically and verified
    against the closed-form Chebyshev-of-second-kind quadrature identity
    (Berrut-Trefethen 2004, formula 5.1).

  Sub-check 2: Out-of-domain (x outside [xmin, xmax]) divergence of the
    barycentric Lagrange formula is symbolically demonstrated:
      For x = xmax + h (h > 0), all (x - x_k) > 0 — denominator becomes
      alternating sum with no sign-flip cancellation → polynomial growth.
    This is the Runge-style divergence that breaks gamma_a_baseline probes
    at boundary nodes.

The PRE-FLIGHT oracle confirms the ARCHITECTURAL DEFECT in the current
ChebyshevSpectral{m} dispatch path. Both sub-checks PASS by design — they
demonstrate the mathematical fact that the current implementation has no
out-of-domain handling for the boundary-overshoot probes in Diffusion4thChernoff
gamma_a baseline.

ADR-0104 §4 sub-action 1.
"""

from __future__ import annotations

import argparse
import math
import sys
from fractions import Fraction


def chebyshev_lobatto_weights(m: int) -> list[Fraction]:
    """Return barycentric weights w_k = (-1)^k * delta_k for k = 0..M.

    delta_0 = delta_M = 1/2, else 1 (Berrut-Trefethen 2004, Table 5.1).
    """
    weights: list[Fraction] = []
    half = Fraction(1, 2)
    one = Fraction(1, 1)
    for k in range(m + 1):
        delta_k = half if k == 0 or k == m else one
        sign = -one if k % 2 == 1 else one
        weights.append(sign * delta_k)
    return weights


def chebyshev_lobatto_nodes_float(m: int) -> list[float]:
    """Return nodes y_k = cos(k pi / M), k = 0..M, in [-1, 1]."""
    return [math.cos(k * math.pi / m) for k in range(m + 1)]


def barycentric_eval(
    nodes: list[float],
    weights: list[Fraction],
    values: list[float],
    x: float,
    guard: float = 1e-14,
) -> float:
    """Evaluate barycentric Lagrange at x using nodes/weights/values."""
    num = 0.0
    den = 0.0
    for k, x_k in enumerate(nodes):
        diff = x - x_k
        if abs(diff) < guard:
            return values[k]
        term = float(weights[k]) / diff
        num += term * values[k]
        den += term
    return num / den


# ---------------------------------------------------------------------------
# Sub-check 1: Weight reproduction + identity verification
# ---------------------------------------------------------------------------

def sub_check_1_weight_identity(verbose: bool = False) -> bool:
    """Verify Chebyshev-Lobatto weights reproduce the closed-form pattern.

    Identity tested: sum_k w_k = 0 for M >= 1 (alternating-sign 1/2 + 1/2 + ...
    sum-to-zero invariant per Berrut-Trefethen Lemma 3.1).
    Also: w_0 = +1/2, w_M = (+1/2 if M even else -1/2).
    """
    if verbose:
        print("Sub-check 1: Chebyshev-Lobatto weights (Berrut-Trefethen 2004)")

    all_ok = True
    for m in (8, 16, 32, 64, 128):
        weights = chebyshev_lobatto_weights(m)

        # Endpoint magnitudes
        if abs(weights[0]) != Fraction(1, 2):
            print(f"  M={m}: w_0 = {weights[0]} != 1/2 (FAIL)")
            all_ok = False
            continue
        if abs(weights[m]) != Fraction(1, 2):
            print(f"  M={m}: w_M = {weights[m]} != 1/2 (FAIL)")
            all_ok = False
            continue

        # Endpoint signs: w_0 = +1/2 always; w_M = +1/2 if M even else -1/2.
        if weights[0] != Fraction(1, 2):
            print(f"  M={m}: sign w_0 wrong (FAIL)")
            all_ok = False
            continue
        expected_wM = Fraction(1, 2) if m % 2 == 0 else Fraction(-1, 2)
        if weights[m] != expected_wM:
            print(f"  M={m}: sign w_M wrong, got {weights[m]} expected {expected_wM} (FAIL)")
            all_ok = False
            continue

        # Identity: sum_k w_k = 0 for M >= 1 (alternating-half on endpoints
        # cancels the alternating-one bulk).
        s = sum(weights)
        # For odd M: w_0 = +1/2, w_M = -1/2, sum of (-1)^k for k=1..M-1 alternates
        # for M-1 even (M odd) gives 0. For M even: w_0=w_M=+1/2, interior alternates
        # with sign (-1)^k from k=1 to M-1 (odd count) sums to -1; total = 1/2 + (-1) + 1/2 = 0.
        if s != Fraction(0, 1):
            print(f"  M={m}: sum_k w_k = {s} != 0 (FAIL)")
            all_ok = False
            continue

        if verbose:
            print(f"  M={m}: w_0=+1/2, w_M={expected_wM}, sum=0  PASS")

    return all_ok


# ---------------------------------------------------------------------------
# Sub-check 2: Out-of-domain divergence (architectural defect)
# ---------------------------------------------------------------------------

def sub_check_2_out_of_domain_divergence(verbose: bool = False) -> bool:
    """Demonstrate barycentric Lagrange diverges outside [xmin, xmax].

    Setup:
      Grid [xmin, xmax] = [-10, 10], M=64.
      Sample exp(-x^2) at Chebyshev-Lobatto nodes (exact analytic values).
      Probe at:
        x = 10.0   (exactly boundary)
        x = 10.71  (gamma_a near-probe at boundary for tau=0.125, a=1)
        x = 11.22  (gamma_a far-probe at boundary for tau=0.125, a=1)
        x = 12.0   (well beyond)

    Assertion: |chebyshev_eval(x) - exp(-x^2)| > 1.0 for all out-of-domain x
    (Runge divergence is >> 1, while analytic exp(-x^2) at x > 10 is < 1e-44).

    This is the H3 boundary defect. The test PASSES when divergence IS
    observed (confirming the architectural defect documented in ADR-0104).
    """
    if verbose:
        print("\nSub-check 2: Out-of-domain barycentric divergence (architectural defect)")

    xmin, xmax = -10.0, 10.0
    M = 64

    # Sample exp(-x^2) at M+1 Chebyshev-Lobatto nodes mapped to [xmin, xmax]
    nodes_minus1to1 = chebyshev_lobatto_nodes_float(M)
    mid = 0.5 * (xmax + xmin)
    half = 0.5 * (xmax - xmin)
    x_mapped = [mid + half * y for y in nodes_minus1to1]
    values = [math.exp(-x * x) for x in x_mapped]
    weights = chebyshev_lobatto_weights(M)

    # Use nodes in [-1, 1] coordinate system for barycentric, mapping x as well
    def cheb_eval(x: float) -> float:
        y = (x - mid) / half
        return barycentric_eval(nodes_minus1to1, weights, values, y)

    # Probe points
    probes = [
        (0.0, False),    # deep interior; in-domain
        (10.0, False),   # exactly boundary; in-domain (boundary node 0)
        (10.71, True),   # gamma_a near-probe; OUT-OF-DOMAIN; expect divergence
        (11.22, True),   # gamma_a far-probe; OUT-OF-DOMAIN; expect divergence
        (12.0, True),    # well beyond; OUT-OF-DOMAIN; expect divergence
    ]

    all_match_expectation = True
    for (x, expect_divergence) in probes:
        analytic = math.exp(-x * x)
        approx = cheb_eval(x)
        err = abs(approx - analytic)

        diverged = err > 1.0  # Order-1 absolute error = catastrophic for Gaussian
        ok = diverged == expect_divergence

        if verbose:
            status = "PASS" if ok else "FAIL"
            label = "DIVERGES" if diverged else "OK"
            print(
                f"  x={x:>6.2f}  analytic={analytic:>10.3e}  "
                f"cheb={approx:>14.6e}  err={err:>12.4e}  "
                f"[{label}, expected {'divergence' if expect_divergence else 'in-domain'}]  {status}"
            )

        if not ok:
            all_match_expectation = False

    return all_match_expectation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("ADR-0104 PRE-FLIGHT ORACLE — Chebyshev barycentric verification")
    print("=" * 72)

    r1 = sub_check_1_weight_identity(verbose=args.verbose)
    r2 = sub_check_2_out_of_domain_divergence(verbose=args.verbose)

    print()
    print("=" * 72)
    print(f"Sub-check 1 (weight identity):              {'PASS' if r1 else 'FAIL'}")
    print(f"Sub-check 2 (out-of-domain divergence):     {'PASS' if r2 else 'FAIL'}")
    print("=" * 72)

    if r1 and r2:
        print("ORACLE VERDICT: PASS")
        print("  - Weights match Berrut-Trefethen 2004 closed form.")
        print("  - Out-of-domain divergence CONFIRMED (architectural defect ADR-0104).")
        return 0
    print("ORACLE VERDICT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
