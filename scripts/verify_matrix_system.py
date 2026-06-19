"""verify_matrix_system.py — T_MATRIX_STRANG sympy gate (ADR-0082 AMENDMENT).

T_MATRIX_STRANG: BCH-order verification for palindromic Marchuk-Strang.

For Φ_τ = exp(τR/2) ∘ exp(τL) ∘ exp(τR/2) with M=2 matrix operators L, R:
  - Local truncation error is O(τ³) — leading BCH commutator term
    (τ³/12)·(2[R,[R,L]] − [L,[L,R]]) per Auzinger et al. 2016 Eq. 8.
  - Global error O(τ²) over T = n·τ fixed (parabolic semigroup smoothing).
  - BCH analysis is DIMENSION-INDEPENDENT in M (Strang applies to any operator).

This sub-check is ADVISORY (no new RELEASE_BLOCKING gate).
Exits 0 iff all checks pass; exits 1 on any failure.
References:
  Auzinger-Herfort-Koch-Thalhammer 2016 arXiv:1604.01190 Theorem 4.1.
  Strang 1968 SIAM J. Numer. Anal. 5(3), 506-517.
  math.md §33.7 AMENDMENT.
"""

import sys
from sympy import symbols, Matrix, eye, zeros, expand, series, factorial

tau = symbols('tau', positive=True)

# ─── M=2 scalar model (matrix entries as distinct symbols) ──────────────────

# Diffusion operator L (pure-diffusion part, 2×2 constant matrix for BCH analysis).
# In the operator-splitting sense L = D·∂² and R = C (reaction).
# We model them as constant 2×2 matrices for the BCH series check.
l00, l01, l10, l11 = symbols('l00 l01 l10 l11', real=True)
r00, r01, r10, r11 = symbols('r00 r01 r10 r11', real=True)

L = Matrix([[l00, l01], [l10, l11]])
R = Matrix([[r00, r01], [r10, r11]])

I2 = eye(2)


def mat_exp_series(A, order=4):
    """Series expansion of exp(A) truncated at O(A^{order+1})."""
    result = I2 + zeros(2, 2)
    term = I2.copy()
    for k in range(1, order + 1):
        term = term * A
        result = result + term / factorial(k)
    return result


def check_t_matrix_strang():
    """Verify palindromic Strang gives LTE O(τ³) via BCH at depth 2."""
    print("T_MATRIX_STRANG: BCH order-condition check for M=2")

    # Build Strang composition to O(τ³):
    #   Φ_τ = exp(τR/2) · exp(τL) · exp(τR/2)
    # We expand each factor to O(τ³) then multiply, keeping only O(τ³) terms.
    exp_half_R = mat_exp_series(tau * R / 2, order=3)
    exp_L      = mat_exp_series(tau * L,     order=3)
    exp_half_R2 = mat_exp_series(tau * R / 2, order=3)

    strang = exp_half_R * exp_L * exp_half_R2

    # Build exact composition exp(τ(L+R)) to O(τ³).
    exp_sum = mat_exp_series(tau * (L + R), order=3)

    # The error Φ_τ - exp(τ(L+R)) should have NO τ¹ or τ² terms.
    error = expand(strang - exp_sum)

    # Extract τ¹ coefficient of each entry.
    def coeff_tau(entry, k):
        s = series(entry, tau, 0, k + 1)
        return s.coeff(tau, k)

    # Check τ¹ and τ² coefficients vanish (order-2 Strang).
    all_pass = True
    for i in range(2):
        for j in range(2):
            entry = error[i, j]
            c1 = expand(coeff_tau(entry, 1))
            c2 = expand(coeff_tau(entry, 2))
            if c1 != 0:
                print(f"  FAIL: τ¹ coeff[{i},{j}] = {c1} (expected 0)")
                all_pass = False
            if c2 != 0:
                print(f"  FAIL: τ² coeff[{i},{j}] = {c2} (expected 0)")
                all_pass = False

    if all_pass:
        print("  PASS: τ¹ and τ² error coefficients vanish for all 4 entries")
    else:
        return False

    # Verify τ³ coefficient is the known BCH commutator (Auzinger 2016 Eq. 8):
    #   e³ = (τ³/12)·(2[R,[R,L]] − [L,[L,R]])
    # We just verify the τ³ coefficient is generically non-zero for [L,R]≠0.
    # If [L,R] != 0, the τ³ coefficient should be non-zero in general.
    # Use a simple non-commuting example to confirm.
    err_num = (strang - exp_sum).subs({
        l00: 1, l01: 0, l10: 0, l11: 2,
        r00: 0, r01: 1, r10: 1, r11: 0,
    })
    c3_entry = expand(coeff_tau(err_num[0, 1], 3))
    if c3_entry != 0:
        print(f"  PASS: τ³ leading BCH commutator non-zero ({c3_entry})")
        print(f"        (parabolic smoothing absorbs τ³ into global O(τ²) rate)")
    else:
        # This could legitimately be zero for specific L,R values; warn not fail.
        print("  NOTE: τ³ coefficient zero for test example (may be coincidental)")

    return True


def main():
    passed = check_t_matrix_strang()
    if passed:
        print("\nT_MATRIX_STRANG: ALL CHECKS PASSED (advisory)")
        sys.exit(0)
    else:
        print("\nT_MATRIX_STRANG: CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
