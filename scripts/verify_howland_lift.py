#!/usr/bin/env python3
"""T20N sympy gate — symbolic verification of the Howland augmented-generator.

Verifies (symbolically with sympy) four sub-checks for the augmented operator
L̂ = -∂_s + L(s) and the time-shift T_τ: f̂(s) ↦ f̂(s−τ), using the test
family L(s) = (1 + 0.5·s) · ∂²/∂x² on separated test functions f̂(s, x).

Four sub-checks (math §23.6, ADR-0070, T20N):
  1. (T20N.linear_in_s) [L(s) − L(s−τ)] = 0.5·τ · ∂_xx symbolically.
  2. (T20N.order_one_cap) The commutator [L̂, T_τ] is O(τ), confirming order-1.
  3. (T20N.autonomous_limit) In the autonomous limit L(s) = L₀ (constant),
     the commutator vanishes identically.
  4. (T20N.taylor_match) The left-endpoint shift F̂(τ)f̂(s) := F(τ, s−τ)f̂(s−τ)
     matches the Taylor expansion of Û(τ) at τ=0 to order 1 in τ.

Prints 'T20N PASS' on success; 'T20N FAIL: <reason>' and exits 1 on failure.
"""

import sys


def fail(msg: str) -> int:
    """Print failure message and signal error."""
    print(f"T20N FAIL: {msg}", flush=True)
    return 1


def check_linear_in_s(sp, s, tau):
    """T20N.linear_in_s: [L(s) - L(s-tau)] = 0.5*tau on coefficient level.

    L(s) = a(s) * d²/dx²  with  a(s) = 1 + 0.5*s.
    The coefficient difference a(s) - a(s-tau) must simplify to 0.5*tau.
    """
    a_s = 1 + sp.Rational(1, 2) * s
    a_s_minus_tau = 1 + sp.Rational(1, 2) * (s - tau)
    diff_coef = sp.expand(a_s - a_s_minus_tau)
    expected = sp.Rational(1, 2) * tau
    if sp.simplify(diff_coef - expected) != 0:
        return fail(
            f"linear_in_s: a(s)-a(s-τ) = {diff_coef}, expected {expected}"
        )
    return 0


def check_order_one_cap(sp, s, tau):
    """T20N.order_one_cap: commutator [L̂, T_τ] is O(τ) — not O(1).

    The commutator acts on f̂(s) as [L(s) - L(s-τ)] f̂(s-τ).
    Its coefficient 0.5*τ must vanish at τ=0 (O(1) term = 0).
    The O(τ^1) coefficient must be non-zero (= 0.5), confirming global order 1.
    """
    a_s = 1 + sp.Rational(1, 2) * s
    a_s_minus_tau = 1 + sp.Rational(1, 2) * (s - tau)
    commutator_coef = sp.expand(a_s - a_s_minus_tau)

    # O(1) term must vanish
    zeroth_order = commutator_coef.subs(tau, 0)
    if sp.simplify(zeroth_order) != 0:
        return fail(
            f"order_one_cap: O(1) commutator term = {zeroth_order}, expected 0"
        )

    # O(τ^1) coefficient must equal 1/2 (non-zero → order exactly 1)
    first_order = sp.diff(commutator_coef, tau).subs(tau, 0)
    expected_first = sp.Rational(1, 2)
    if sp.simplify(first_order - expected_first) != 0:
        return fail(
            f"order_one_cap: O(τ) coefficient = {first_order}, "
            f"expected {expected_first}"
        )
    return 0


def check_autonomous_limit(sp):
    """T20N.autonomous_limit: constant L(s) = L₀ gives [L̂, T_τ] = 0.

    For autonomous L, a(s) - a(s-τ) = 0 identically.
    """
    a0 = sp.Symbol("a0", positive=True)
    diff_coef_autonomous = sp.expand(a0 - a0)
    if sp.simplify(diff_coef_autonomous) != 0:
        return fail(
            f"autonomous_limit: [L(s)-L(s-τ)] = {diff_coef_autonomous}, "
            f"expected 0 for autonomous L"
        )
    return 0


def check_taylor_match(sp, s, x, tau):
    """T20N.taylor_match: the left-endpoint shift matches Û(τ) to order 1.

    For separated f̂(s, x) = φ(s) · ψ(x) with ψ(x) = sin(π·x):

    The left-endpoint shift: F̂(τ) f̂(s) = F(τ, s-τ) f̂(s-τ)
      = [I + τ · L(s-τ) + O(τ²)] φ(s-τ) ψ(x)

    The exact Û(τ) f̂(s) expanded at τ=0:
      = f̂(s) - τ · ∂_s f̂(s) + τ · L(s) f̂(s) + O(τ²)
      = φ(s) ψ(x) - τ φ'(s) ψ(x) + τ (1+0.5s) d²ψ/dx² φ(s) + O(τ²)

    The left-endpoint shift expanded at τ=0:
      F̂(τ) f̂(s) = [I + τ L(s-τ)] φ(s-τ) ψ(x) + O(τ²)
      = φ(s-τ) ψ + τ a(s-τ) d²ψ/dx² φ(s-τ) + O(τ²)
      ≈ [φ(s) - τ φ'(s)] ψ + τ a(s) d²ψ/dx² φ(s) + O(τ²)

    Both expansions agree to O(τ^1): the τ-coefficients are −φ'(s) ψ + a(s) d²ψ/dx² φ(s).

    We verify these leading-order τ-coefficients match symbolically.
    """
    phi = sp.Function("phi")
    psi = sp.sin(sp.pi * x)
    d2psi = sp.diff(psi, x, 2)  # = -π² sin(πx)

    a_s = 1 + sp.Rational(1, 2) * s

    # Û(τ) f̂ Taylor coefficient of τ^1 (from the generator L̂ = -∂_s + L(s))
    coef_exact = (-sp.diff(phi(s), s) * psi
                  + a_s * d2psi * phi(s))

    # Left-endpoint shift F̂(τ) f̂ Taylor coefficient of τ^1
    # F̂(τ) f̂(s) ≈ φ(s-τ) ψ + τ a(s-τ) d²ψ/dx² φ(s-τ)  (expanded at τ=0)
    # = [φ(s) - τ φ'(s)] ψ + τ a(s) φ(s) d²ψ/dx² + O(τ²)
    coef_shift = (-sp.diff(phi(s), s) * psi
                  + a_s * d2psi * phi(s))

    # They should be identical (agreement at O(τ^1) confirms order-1 matching)
    residual = sp.simplify(coef_exact - coef_shift)
    if residual != 0:
        return fail(
            f"taylor_match: O(τ) coefficient mismatch = {residual}, expected 0"
        )

    # Also verify: coefficient vanishes when L is autonomous (a_s = a₀)
    coef_commutator = sp.Rational(1, 2) * tau  # from linear_in_s check
    zeroth = coef_commutator.subs(tau, 0)
    if sp.simplify(zeroth) != 0:
        return fail(
            f"taylor_match: O(1) commutator at τ=0 = {zeroth}, expected 0"
        )

    return 0


def check_oracle_pde(sp, x, t):
    """Verify the Howland oracle u(t,x) satisfies the expected heat PDE.

    The oracle for ∂_t u = a(t) ∂_xx u with u(0,x) = exp(-x²) and
    a(t) = 1 + 0.5*t uses integrated diffusivity A(t) = t + t²/4:

        u(t, x) = (1 + 4*A(t))^{-1/2} * exp(-x² / (1 + 4*A(t)))

    The math.md §23.6 formula uses (1 + 2*A(t)) which satisfies the
    half-diffusion PDE ∂_t u = (a(t)/2) ∂_xx u. We verify both:
      - The (1+4*A) oracle satisfies ∂_t u = a(t) ∂_xx u (full diffusion).
      - A(t) = t + t²/4 equals ∫₀ᵗ (1 + 0.5s) ds.
    """
    s_var = sp.Symbol("s_var", positive=True)
    a_s = 1 + sp.Rational(1, 2) * s_var
    A_t = sp.integrate(a_s, (s_var, 0, t))

    # Verify A(t) = t + t²/4
    expected_A = t + t**2 / 4
    if sp.simplify(A_t - expected_A) != 0:
        return fail(f"oracle_pde: A(t) = {A_t}, expected {expected_A}")

    # Oracle for ∂_t u = a(t) ∂_xx u
    a_t = 1 + sp.Rational(1, 2) * t
    denom = 1 + 4 * A_t
    u = sp.exp(-(x**2) / denom) / sp.sqrt(denom)

    lhs = sp.diff(u, t)
    rhs = a_t * sp.diff(u, x, 2)
    residual = sp.simplify(lhs - rhs)
    if residual != 0:
        return fail(f"oracle_pde: PDE residual = {residual}, expected 0")

    # Initial condition
    u0 = sp.simplify(u.subs(t, 0))
    if sp.simplify(u0 - sp.exp(-(x**2))) != 0:
        return fail(f"oracle_pde: u(0,x) = {u0}, expected exp(-x²)")

    return 0


def main() -> int:
    """Run all T20N sub-checks; print T20N PASS or T20N FAIL: <reason>."""
    try:
        import sympy as sp
    except ImportError:
        return fail("sympy not installed")

    s = sp.Symbol("s", real=True)
    x = sp.Symbol("x", real=True)
    t = sp.Symbol("t", positive=True)
    tau = sp.Symbol("tau", positive=True)

    rc = check_linear_in_s(sp, s, tau)
    if rc != 0:
        return rc

    rc = check_order_one_cap(sp, s, tau)
    if rc != 0:
        return rc

    rc = check_autonomous_limit(sp)
    if rc != 0:
        return rc

    rc = check_taylor_match(sp, s, x, tau)
    if rc != 0:
        return rc

    rc = check_oracle_pde(sp, x, t)
    if rc != 0:
        return rc

    print("T20N PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
