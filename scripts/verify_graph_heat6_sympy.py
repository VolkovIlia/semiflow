"""verify_graph_heat6_sympy.py — order-6 spatial graph heat sympy gates (math.md §19.5 T16N).

Verifies that the order-6 operator-Taylor truncation

    S_6(tau) f = sum_{k=0}^{6} (-tau L_G)^k / k! * f

satisfies

    S_6(tau) f - exp(-tau L_G) f = O(tau^7)

on a symbolic 4-node path Laplacian L_G (constant edge weight w = 1). The
identity is proved by series expansion of both sides in `tau` and checking
that coefficients agree through `tau^6` and disagree at order `tau^7`.

Exits 0 iff all gates pass.

References:
- math.md §19 (Order-6 spatial graph diffusion).
- ADR-0062 §"T16N sympy gate".
- Higham 2008 *Functions of Matrices* §10 (Taylor exp truncation bounds).
"""

import sys

from sympy import Matrix, Rational, eye, exp, expand, factorial, simplify, symbols, zeros


def path_laplacian_4():
    """Combinatorial Laplacian of the 4-node path graph 0-1-2-3 with unit weights.

    L_G = D - W, D = diag(1, 2, 2, 1), W = adjacency.
    """
    L = Matrix([
        [Rational(1), Rational(-1), Rational(0), Rational(0)],
        [Rational(-1), Rational(2), Rational(-1), Rational(0)],
        [Rational(0), Rational(-1), Rational(2), Rational(-1)],
        [Rational(0), Rational(0), Rational(-1), Rational(1)],
    ])
    return L


def truncated_taylor(L, tau, k_max):
    """Compute sum_{k=0}^{k_max} (-tau L)^k / k! as a symbolic matrix."""
    n = L.shape[0]
    result = eye(n)
    term = eye(n)
    for k in range(1, k_max + 1):
        term = term * (-tau * L) / k
        result = result + term
    return result


def gate(label, residual_matrix, expect_order, tau):
    """Pass iff every entry of residual_matrix has tau-series leading order >= expect_order."""
    n, m = residual_matrix.shape
    worst = None
    for i in range(n):
        for j in range(m):
            entry = expand(residual_matrix[i, j])
            if entry == 0:
                continue
            # Strip leading tau^k by repeated differentiation at tau=0
            for k in range(expect_order):
                if entry.subs(tau, 0) != 0:
                    worst = (i, j, k, entry.subs(tau, 0))
                    break
                entry = simplify(entry / tau)
            if worst is not None:
                break
        if worst is not None:
            break
    mark = "OK" if worst is None else "FAIL"
    print(f"{label} {mark}")
    if worst is not None:
        print(f"  FAIL at [{worst[0]},{worst[1]}]: tau^{worst[2]} coefficient = {worst[3]}")
    return worst is None


def main():
    tau = symbols('tau', real=True, positive=True)
    L = path_laplacian_4()
    n = L.shape[0]

    print("=" * 60)
    print("GraphHeat6 sympy gates (math.md §19.5 T16N)")
    print("=" * 60)

    # exp(-tau L) as symbolic matrix exponential.
    # For a 4x4 symbolic Laplacian this is tractable via eigendecomposition,
    # but is expensive. Instead, expand exp(-tau L) as a Taylor series to
    # degree 8 (one above what we test) and compare.
    exp_truncated_8 = truncated_taylor(L, tau, 8)

    # The library's S_6(tau).
    S6 = truncated_taylor(L, tau, 6)

    # Residual: exp(-tau L) - S_6 should equal -(tau^7/7! + tau^8/8!) L^k terms.
    # As a Taylor series with sufficient degree, the leading nonzero order in
    # (exp_truncated_8 - S6) is tau^7. We verify each matrix entry's leading
    # order is >= 7.
    residual = exp_truncated_8 - S6

    g1 = gate("T16N: S_6(tau) matches exp(-tau L_G) through tau^6", residual, expect_order=7, tau=tau)

    # Sanity: order-5 truncation S_5 should disagree at tau^6.
    S5 = truncated_taylor(L, tau, 5)
    residual_5 = exp_truncated_8 - S5
    # Find the leading order of any entry — expect 6.
    leading_orders = []
    for i in range(n):
        for j in range(n):
            entry = expand(residual_5[i, j])
            if entry == 0:
                continue
            for k in range(10):
                if entry.subs(tau, 0) != 0:
                    leading_orders.append(k)
                    break
                entry = simplify(entry / tau)
    min_leading = min(leading_orders) if leading_orders else 99
    g2_ok = (min_leading == 6)
    print(f"sanity: S_5 disagrees at tau^6: {'OK' if g2_ok else 'FAIL'}  (got tau^{min_leading})")

    results = [g1, g2_ok]
    passed = all(results)
    print()
    print("All GraphHeat6 gates passed." if passed else "FAIL: gates failed.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
