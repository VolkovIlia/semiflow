"""verify_magnus_sympy.py — Magnus K=4 sympy gates (v0.4.0, math.md §9.2.3.C).

Verifies M(tau) f = sum_{k=0}^{4} (tau^k/k!) A_self^k f satisfies:
  M_tau^0: M(0)f = f
  M_tau^1: d/dtau M(tau)f|_{tau=0} = A_self f
  M_tau^2: (1/2) d^2/dtau^2 M(tau)f|_{tau=0} = A_self^2 f / 2

Exits 0 iff all gates pass.
"""

import sys
from sympy import symbols, factorial, expand, diff, Poly, Integer

x, tau = symbols('x tau', real=True)
a0, a1, a2 = symbols('a0 a1 a2', real=True)
f_syms = symbols('f0 f1 f2 f3 f4 f5 f6', real=True)

# Polynomial models at x=0: a(x), f(x)
a_poly = a0 + a1 * x + a2 * x**2 / 2
f_poly = sum((f_syms[k] * x**k / factorial(k) for k in range(7)), Integer(0))


def A_self(a, f):
    """A_self f = a f'' + a' f' (divergence form)."""
    return diff(a, x) * diff(f, x) + a * diff(f, x, 2)


def magnus_series(k_max=4):
    """Compute sum_{k=0}^{k_max} (tau^k / k!) A_self^k f, evaluated at x=0."""
    total, g = Integer(0), f_poly
    for k in range(k_max + 1):
        total += (tau**k / factorial(k)) * g.subs(x, 0)
        g = A_self(a_poly, g)
    return expand(total)


def nth_coeff(expr, n):
    """Return coefficient of tau^n in expr (polynomial in tau)."""
    return expand(Poly(expr, tau).nth(n))


def gate(label, actual, expected):
    ok = expand(actual - expected) == 0
    mark = "✓" if ok else "✗"
    print(f"{label} {mark}")
    if not ok:
        print(f"  FAIL: residual = {expand(actual - expected)}")
    return ok


def main():
    f0 = f_syms[0]
    A1 = expand(A_self(a_poly, f_poly).subs(x, 0))
    A2 = expand(A_self(a_poly, A_self(a_poly, f_poly)).subs(x, 0))

    M = magnus_series()
    c0, c1, c2 = nth_coeff(M, 0), nth_coeff(M, 1), nth_coeff(M, 2)

    print("=" * 50)
    print("Magnus K=4 sympy gates (math.md §9.2.3.C)")
    print("=" * 50)
    results = [
        gate("M_τ⁰", c0, f0),
        gate("M_τ¹", c1, A1),
        gate("M_τ²", c2, expand(A2 / 2)),
    ]
    print()
    passed = all(results)
    print("All Magnus gates passed." if passed else "FAIL: gates failed.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
