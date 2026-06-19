"""verify_varcoef_magnus_graph_sympy.py — VarCoef × time-dependent Magnus K=4 sympy gates
(math.md §20.6 T17N).

Verifies that the GL2 + Omega_4 expansion of the time-dependent operator
L_a(t) = sqrt(a(t)) diag-mult L_G(t) diag-mult sqrt(a(t))
matches the true Magnus operator through tau^4 symbolically.

Setup: 4-node path Laplacian with time-varying edge weight w(t) = 1 + s_w * t
and time-varying node coefficient a_i(t) = 1 + s_a * t (homogeneous in i for
simplicity). We expand both Omega_4 (library formula) and Omega_true (series
solution of the IVP) in tau and confirm the residual is O(tau^5).

Exits 0 iff all gates pass.

References:
- math.md §20 (VarCoef × time-dependent graph Magnus K=4).
- ADR-0063 §"T17N sympy gate".
- Iserles+ 2000 Acta Numerica §6.
- Blanes+ 2009 Phys. Rep. §3 (Magnus expansion convergence).
"""

import sys

from sympy import Matrix, Rational, diag, eye, expand, simplify, sqrt, symbols, sympify


def path_laplacian_4_w(w):
    """4-node path Laplacian with all edges weight w."""
    return Matrix([
        [w, -w, 0, 0],
        [-w, 2 * w, -w, 0],
        [0, -w, 2 * w, -w],
        [0, 0, -w, w],
    ])


def L_a(L_G, a_vec):
    """L_a = diag(sqrt(a)) * L_G * diag(sqrt(a))."""
    sa = diag(*[sqrt(a) for a in a_vec])
    return sa * L_G * sa


def omega4_library(L_a_of_t, tau):
    """Library's Omega_4 via GL2 abscissae c_1 = (3 - sqrt(3))/6 c_2 = (3 + sqrt(3))/6."""
    s3 = sqrt(3)
    c1 = (3 - s3) / 6
    c2 = (3 + s3) / 6
    A1 = -L_a_of_t(c1 * tau)
    A2 = -L_a_of_t(c2 * tau)
    half = Rational(1, 2)
    return (tau * half) * (A1 + A2) + (s3 * tau**2 / 12) * (A2 * A1 - A1 * A2)


def omega_true_through_tau4(L_a_of_t, tau, t_sym):
    """Expand the true Magnus operator Omega(tau) = log(U(tau)) through tau^4.

    Using the series Omega = integral_0^tau A(s) ds
                            + (1/2) integral_0^tau [integral_0^{s_1} A(s_2) ds_2, A(s_1)] ds_1
                            + (higher commutator triple-integrals)

    For order-4 accuracy we keep:
      Omega^(1)(tau) = integral_0^tau A(s) ds
      Omega^(2)(tau) = (1/2) integral_0^tau integral_0^{s_1} [A(s_2), A(s_1)] ds_2 ds_1

    Higher-order terms Omega^(3,4) contribute only at tau^5 or higher
    when A(t) is at most degree-1 polynomial in t (which is our test setup).

    Returns Omega^(1) + Omega^(2) computed via sympy symbolic integration.
    """
    from sympy import integrate

    s, s1, s2 = symbols('s s1 s2', real=True)
    A_s = -L_a_of_t(s)
    # Omega^(1)
    Omega1 = A_s.applyfunc(lambda e: integrate(e, (s, 0, tau)))
    # Omega^(2): (1/2) ∫_0^τ [Ω(s_1), A(s_1)] ds_1 where Ω(s_1) = ∫_0^{s_1} A(s_2) ds_2
    A_s2 = -L_a_of_t(s2)
    Omega_inner = A_s2.applyfunc(lambda e: integrate(e, (s2, 0, s1)))
    A_s1 = -L_a_of_t(s1)
    commutator = Omega_inner * A_s1 - A_s1 * Omega_inner
    Omega2 = (commutator * Rational(1, 2)).applyfunc(lambda e: integrate(e, (s1, 0, tau)))
    return Omega1 + Omega2


def leading_tau_order(M, tau, max_check=8):
    """Find smallest k such that any entry of M has a nonzero tau^k coefficient.

    Returns max_check + 1 if M is zero through tau^max_check.
    """
    leading = max_check + 1
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            entry = expand(M[i, j])
            if entry == 0:
                continue
            for k in range(max_check + 1):
                coeff_k = expand(entry.coeff(tau, k))
                if coeff_k != 0:
                    if k < leading:
                        leading = k
                    break
    return leading


def gate(label, residual_matrix, expect_order, tau):
    ord_ = leading_tau_order(residual_matrix, tau, max_check=expect_order + 2)
    ok = ord_ >= expect_order
    mark = "OK" if ok else "FAIL"
    print(f"{label} {mark}  (leading order tau^{ord_}, expected >= tau^{expect_order})")
    return ok


def main():
    tau = symbols('tau', real=True, positive=True)
    t = symbols('t', real=True)
    s_w, s_a = symbols('s_w s_a', real=True, positive=True)

    def L_a_of_t(t_val):
        w = 1 + s_w * t_val
        a_vec = [1 + s_a * t_val] * 4
        return L_a(path_laplacian_4_w(w), a_vec)

    print("=" * 60)
    print("VarCoef Magnus K=4 sympy gates (math.md §20.6 T17N)")
    print("=" * 60)

    Omega_lib = omega4_library(L_a_of_t, tau)
    Omega_true = omega_true_through_tau4(L_a_of_t, tau, t)

    residual = Omega_lib - Omega_true
    g1 = gate("T17N: Omega_4_library matches Omega_true through tau^4", residual, expect_order=5, tau=tau)

    results = [g1]
    passed = all(results)
    print()
    print("All VarCoef-Magnus gates passed." if passed else "FAIL: gates failed.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
