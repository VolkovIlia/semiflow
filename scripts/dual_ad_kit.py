#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
#
# Sympy's symbolic expressions are dynamically typed through __add__/__mul__/etc.;
# Pyright cannot trace operator return types. All operations in this module are
# valid sympy at runtime (verified by the T_DUAL oracle below).
"""T_DUAL sympy PRE-FLIGHT oracle — forward-mode dual-number AD (v8.0.0, ADR-0133).

Verifies, BEFORE the engineer writes `crates/semiflow-core/src/dual.rs`, that the
forward-mode dual-number arithmetic specified in `contracts/semiflow-core.math.md`
§46 is mathematically exact. A dual number is `Dual(value, tangent) = a + ε·b`
with the nilpotent unit `ε² = 0`. Carrying `b = dF/dθ` in the tangent slot makes
every arithmetic op propagate the exact first derivative by the chain rule — at
ZERO new allocation, because the pair rides in the same registers as the scalar.

Sub-checks (all symbolic; an op is "correct" iff its dual tangent equals the
sympy-exact derivative of its value component as a function of θ):

  (a) T_DUAL.arithmetic
        +, −, ×, ÷ on Dual(u, u') and Dual(v, v') reproduce the exact
        derivative rules:
          (u+v)'  = u' + v'
          (u−v)'  = u' − v'
          (u·v)'  = u'·v + u·v'                      (product rule)
          (u/v)'  = (u'·v − u·v') / v²               (quotient rule)
        Verified against sympy.diff over a symbolic parameter θ.

  (b) T_DUAL.transcendental
        The SemiflowFloat / num_traits::Float transcendental ops that kernels
        actually call — exp, ln, sqrt, recip, sin, cos, powi(n), abs — carry the
        chain rule:  g(Dual(u,u')) = Dual(g(u), g'(u)·u'). Each is checked
        against sympy.diff(g(u(θ)), θ).

  (c) T_DUAL.hyperdual_second_deriv
        Dual(Dual(F)) (hyper-dual) recovers the exact SECOND derivative Γ.
        For g = exp on the value u(θ), the (ε₁ε₂) cross-component must equal
        d²/dθ² g(u(θ)). Confirms the Γ-path of ADR-0133 with one blanket reuse.

  (d) T_DUAL.kernel_forward_grad
        A representative scalar reduction of one Chernoff product step —
        the ζ-A symbol factor  k(θ) = exp(−θ·s²)·(1 + θ·s²)  with θ the
        diffusivity parameter and s a fixed Fourier mode — is evaluated in
        dual arithmetic; the forward tangent must match sympy d k/dθ exactly,
        AND a central-difference probe must agree to ≤ 1e-10 (mirrors the
        G_DUAL_AD_GRADIENT acceptance gate's forward-vs-central comparison).

Prints 'T_DUAL PASS (4/4 sub-checks: arithmetic / transcendental /
hyperdual_second_deriv / kernel_forward_grad)' on success; 'T_DUAL FAIL: <reason>'
and exits 1 on failure.

References:
  - ADR-0133 §"Decision" — Dual<F>: SemiflowFloat + blanket ChernoffFunction<Dual<F>>.
  - contracts/semiflow-core.math.md §46 (NORMATIVE — dual-number forward AD).
  - crates/semiflow-core/src/float.rs (SemiflowFloat: Float + … bound surface).
  - num_traits::Float (the transcendental ops Dual<F> must forward).
"""

import sys


def fail(reason: str) -> int:
    print(f"T_DUAL FAIL: {reason}", flush=True)
    return 1


# ---------------------------------------------------------------------------
# A minimal symbolic Dual: a + ε·b with ε² = 0. The arithmetic rules below are
# EXACTLY what the Rust `impl` must reproduce; the oracle proves they equal the
# sympy-exact derivative of the value component.
# ---------------------------------------------------------------------------


class Dual:
    """Symbolic dual number value + ε·tangent (ε² = 0)."""

    def __init__(self, value, tangent):
        self.value = value
        self.tangent = tangent

    def __add__(self, other):
        return Dual(self.value + other.value, self.tangent + other.tangent)

    def __sub__(self, other):
        return Dual(self.value - other.value, self.tangent - other.tangent)

    def __mul__(self, other):
        # Product rule: (u·v)' = u'·v + u·v'  (the ε² term vanishes).
        return Dual(
            self.value * other.value,
            self.tangent * other.value + self.value * other.tangent,
        )

    def __truediv__(self, other):
        # Quotient rule: (u/v)' = (u'·v − u·v') / v².
        v = other.value
        return Dual(
            self.value / v,
            (self.tangent * v - self.value * other.tangent) / (v * v),
        )


def check_arithmetic() -> str | None:
    """T_DUAL sub-check (a): +, −, ×, ÷ reproduce exact derivative rules."""
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    theta = sp.Symbol("theta", real=True)
    # Two non-trivial functions of θ with known derivatives.
    u = sp.sin(theta) + theta**2
    v = sp.exp(theta) + 1
    du = sp.diff(u, theta)
    dv = sp.diff(v, theta)

    U = Dual(u, du)
    V = Dual(v, dv)

    cases = {
        "add": (U + V, sp.diff(u + v, theta)),
        "sub": (U - V, sp.diff(u - v, theta)),
        "mul": (U * V, sp.diff(u * v, theta)),
        "div": (U / V, sp.diff(u / v, theta)),
    }
    for op_name, (dual_result, exact_deriv) in cases.items():
        if sp.simplify(dual_result.tangent - exact_deriv) != 0:
            return (
                f"arithmetic.{op_name}: dual tangent {dual_result.tangent} != "
                f"exact derivative {exact_deriv}"
            )
    return None


def check_transcendental() -> str | None:
    """T_DUAL sub-check (b): exp/ln/sqrt/recip/sin/cos/powi/abs chain rule."""
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    theta = sp.Symbol("theta", positive=True)
    u = sp.Rational(1, 2) + theta**2  # strictly positive on θ>0 (ln/sqrt safe)
    du = sp.diff(u, theta)
    # Each entry: (name, dual-op result, sympy-exact value function g(u)).
    n_pow = 3
    ops = {
        "exp": (Dual(sp.exp(u), sp.exp(u) * du), sp.exp(u)),
        "ln": (Dual(sp.log(u), du / u), sp.log(u)),
        "sqrt": (Dual(sp.sqrt(u), du / (2 * sp.sqrt(u))), sp.sqrt(u)),
        "recip": (Dual(1 / u, -du / u**2), 1 / u),
        "sin": (Dual(sp.sin(u), sp.cos(u) * du), sp.sin(u)),
        "cos": (Dual(sp.cos(u), -sp.sin(u) * du), sp.cos(u)),
        "powi": (Dual(u**n_pow, n_pow * u ** (n_pow - 1) * du), u**n_pow),
        # abs on a positive argument: d|u|/dθ = sign(u)·u' = u' here.
        "abs": (Dual(sp.Abs(u), du), sp.Abs(u)),
    }
    for name, (dual_result, value_fn) in ops.items():
        exact_deriv = sp.diff(value_fn, theta)
        if sp.simplify(dual_result.value - value_fn) != 0:
            return f"transcendental.{name}: value {dual_result.value} != {value_fn}"
        if sp.simplify(dual_result.tangent - exact_deriv) != 0:
            return (
                f"transcendental.{name}: tangent {dual_result.tangent} != "
                f"exact derivative {exact_deriv}"
            )
    return None


def check_hyperdual_second_deriv() -> str | None:
    """T_DUAL sub-check (c): Dual<Dual<F>> recovers the exact second derivative."""
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    theta = sp.Symbol("theta", real=True)
    u = theta**3 + sp.sin(theta)
    du = sp.diff(u, theta)
    d2u = sp.diff(u, theta, 2)

    # Hyper-dual seed: outer dual carries (value, d/dθ); inner dual carries the
    # same for both slots so the cross ε₁ε₂ component picks up the 2nd derivative.
    # Represent the hyper-dual of u as nested Dual:
    #   U2 = Dual( Dual(u, du), Dual(du, d2u) )
    U2 = Dual(Dual(u, du), Dual(du, d2u))

    # Apply g = exp via the chain rule at BOTH nesting levels:
    #   g(Dual(a, b)) = Dual(g(a), g'(a)·b), recursively on the inner duals.
    def dual_exp(x):
        if isinstance(x.value, Dual):
            # outer level: components are themselves duals → recurse
            ga = dual_exp(x.value)
            # g'(a)·b where g'(a) = ga (exp), b = x.tangent (a dual)
            gp_b = dual_mul(ga, x.tangent)
            return Dual(ga, gp_b)
        ev = sp.exp(x.value)
        return Dual(ev, ev * x.tangent)

    def dual_mul(p, q):
        return Dual(
            p.value * q.value,
            p.tangent * q.value + p.value * q.tangent,
        )

    G = dual_exp(U2)
    # The exact second derivative of exp(u(θ)) w.r.t. θ:
    exact_d2 = sp.diff(sp.exp(u), theta, 2)
    # Cross component: outer.tangent.tangent  (ε₁ε₂ slot).
    cross = G.tangent.tangent
    if sp.simplify(cross - exact_d2) != 0:
        return (
            f"hyperdual_second_deriv: ε₁ε₂ component {sp.simplify(cross)} != "
            f"exact d²/dθ² exp(u) {sp.simplify(exact_d2)}"
        )
    return None


def check_kernel_forward_grad() -> str | None:
    """T_DUAL sub-check (d): forward grad of a ζ-A symbol factor vs exact + FD."""
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    theta = sp.Symbol("theta", positive=True)
    s = sp.Rational(7, 5)  # fixed Fourier mode

    # Representative single-step Chernoff symbol factor (ζ-A flavour):
    #   k(θ) = exp(−θ·s²) · (1 + θ·s²)
    # θ is the diffusivity parameter we differentiate against.
    s2 = s**2
    exp_part = Dual(sp.exp(-theta * s2), -s2 * sp.exp(-theta * s2))  # d/dθ
    poly_part = Dual(1 + theta * s2, s2)  # d/dθ (1 + θ s²) = s²
    k_dual = exp_part * poly_part

    k_value = sp.exp(-theta * s2) * (1 + theta * s2)
    exact_grad = sp.diff(k_value, theta)

    # Forward (dual) tangent must equal the symbolic exact derivative.
    if sp.simplify(k_dual.tangent - exact_grad) != 0:
        return (
            f"kernel_forward_grad: dual tangent {sp.simplify(k_dual.tangent)} != "
            f"exact grad {sp.simplify(exact_grad)}"
        )

    # Central-difference probe at θ₀ (mirrors G_DUAL_AD_GRADIENT acceptance).
    theta0 = sp.Rational(1, 2)
    h = sp.Rational(1, 10**6)
    k_fn = sp.lambdify(theta, k_value, "mpmath")
    fwd = float(k_dual.tangent.subs(theta, theta0))
    central = float((k_fn(theta0 + h) - k_fn(theta0 - h)) / (2 * h))
    if abs(fwd - central) > 1e-10:
        return (
            f"kernel_forward_grad: forward {fwd:.12e} vs central {central:.12e}, "
            f"|diff|={abs(fwd - central):.3e} > 1e-10"
        )
    return None


def main() -> int:
    err = check_arithmetic()
    if err is not None:
        return fail(f"T_DUAL.arithmetic: {err}")

    err = check_transcendental()
    if err is not None:
        return fail(f"T_DUAL.transcendental: {err}")

    err = check_hyperdual_second_deriv()
    if err is not None:
        return fail(f"T_DUAL.hyperdual_second_deriv: {err}")

    err = check_kernel_forward_grad()
    if err is not None:
        return fail(f"T_DUAL.kernel_forward_grad: {err}")

    print(
        "T_DUAL PASS (4/4 sub-checks: arithmetic / transcendental / "
        "hyperdual_second_deriv / kernel_forward_grad)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
