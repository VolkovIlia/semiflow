#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false
"""T_RESOLVENT_JUMP oracle: resolvent time-jump amortization (F2, ADR-0134).

v8.0.0 Phase-2 SPIKE oracle for direction F2. Verifies the numerical-Laplace-
inversion *time-jump* scheme of math.md §35 (NEW): a single large time-step
``e^{tA} g`` evaluated as a weighted contour-quadrature sum of bounded resolvent
evaluations ``(λ_k I − A)⁻¹ g``, at a CONSTANT number of contour nodes M that is
INDEPENDENT of t — decoupling large-T cost from the n ∝ T·‖A‖ Chernoff ladder.

────────────────────────────────────────────────────────────────────────────
Mathematics (math.md §35 — Cauchy / numerical Laplace inversion)
────────────────────────────────────────────────────────────────────────────
For a sectorial generator A (here self-adjoint, negative-(semi)definite, ω = 0),

    e^{tA} g = (1/2πi) ∮_Γ e^{λt} (λI − A)⁻¹ g dλ                       (35.1)

where Γ is a Bromwich contour deformed to wrap the spectrum σ(A) ⊂ (−∞, 0].
Parametrise Γ by the Trefethen–Weideman–Schmelzer (2006) optimised PARABOLIC
contour, SCALED by M/t so the node count needed for a target accuracy is
t-independent:

    λ(θ) = (M/t) · ( a0 − a1 θ² + i a2 θ ),   θ ∈ (−π, π),
    a0 = 0.1309,  a1 = 0.1194,  a2 = 0.2500   (TWS-2006 parabolic optimum)

Midpoint rule with M equispaced θ_k = −π + (k+½)(2π/M):

    e^{tA} g ≈ (1/2πi) Σ_{k=0}^{M−1} e^{λ_k t} (λ_k I − A)⁻¹ g · λ'(θ_k) · (2π/M)  (35.2)

KEY STRUCTURAL FACT (the reuse + the restriction):
  • the per-node operation is ONE resolvent application (λ_k I − A)⁻¹ g — exactly
    the object math.md §22 (ADR-0069) ships;
  • BUT the optimal contour places MOST nodes in the LEFT half-plane Re λ_k < 0
    (that is how it wraps the negative-real spectrum and earns geometric
    convergence). The shipped Gauss-Laguerre Laplace-Chernoff *quadrature*
    (R̃_n(λ) = (1/λ)Σ w_k (C(s_k/(λn)))^n g, §22.6 + §22.9) is valid ONLY for
    Re λ > ω and DIVERGES for Re λ ≤ ω. Therefore the time-jump REUSES the
    resolvent *abstraction* but must evaluate it with a LEFT-half-plane-capable
    backend (direct complex tridiagonal / Thomas solve, O(N) for the 3-pt
    Laplacian), NOT the GL₃₂ Laplace integral. This is the NARROW boundary.

This oracle uses a direct complex solve for (λI − A)⁻¹ to mirror the LHP-capable
backend the engineer must add; it isolates the OUTER contour-quadrature question
(which is what G_RESOLVENT_JUMP_ORDER gates).

────────────────────────────────────────────────────────────────────────────
Gate T_RESOLVENT_JUMP — 4 sub-checks (all mandatory)
────────────────────────────────────────────────────────────────────────────
  (1) geometric_decay  : err(M) decays geometrically (super-algebraically) in M
                         — log10(err) vs M is ~linear with rate ≤ −0.3 dec/node.
  (2) t_independence   : the M needed for a fixed tolerance does NOT grow with t
                         (err-vs-M curves at t ∈ {1, 20, 100} coincide within 10×)
                         — this is the cost-decoupling claim of ADR-0134.
  (3) order_slope      : G24-convention OLS slope d log(err)/d log(1/M) ≥ 1.95
                         (G_RESOLVENT_JUMP_ORDER: at least order-2; here ≫ 2 because
                         the contour is geometrically convergent).
  (4) lhp_guard        : the optimal contour DOES place nodes in Re λ < 0; the
                         shipped GL₃₂ eval_complex would DIVERGE there. Confirms
                         the documented NARROW restriction (engineer must NOT route
                         the contour through eval_complex; needs an LHP solve).
"""

import numpy as np
from scipy.linalg import expm

# TWS-2006 optimised parabolic-contour coefficients.
A0, A1, A2 = 0.1309, 0.1194, 0.2500


def laplacian_1d(n: int, dx: float) -> np.ndarray:
    """3-point Laplacian A = ∂²ₓ, reflecting (Neumann) BC. Symmetric, σ(A) ≤ 0."""
    a = np.zeros((n, n))
    inv = 1.0 / (dx * dx)
    for i in range(n):
        a[i, i] = -2.0 * inv
        if i > 0:
            a[i, i - 1] = inv
        if i < n - 1:
            a[i, i + 1] = inv
    a[0, 0] = -1.0 * inv
    a[n - 1, n - 1] = -1.0 * inv
    return a


def resolvent_solve(a: np.ndarray, lam: complex, g: np.ndarray) -> np.ndarray:
    """(λI − A)⁻¹ g by a direct complex solve.

    Mirrors the LEFT-half-plane-capable resolvent backend the engineer adds in
    resolvent_jump.rs (a complex tridiagonal Thomas solve, O(N) for the 3-pt
    Laplacian). VALID for any λ ∉ σ(A) — including Re λ < 0 off (−∞, 0].
    """
    n = a.shape[0]
    return np.linalg.solve(lam * np.eye(n, dtype=complex) - a, g.astype(complex))


def time_jump(a: np.ndarray, t: float, g: np.ndarray, m_nodes: int) -> np.ndarray:
    """e^{tA} g via TWS-2006 parabolic-contour midpoint quadrature, M nodes."""
    n = a.shape[0]
    acc = np.zeros(n, dtype=complex)
    for k in range(m_nodes):
        th = -np.pi + (k + 0.5) * (2.0 * np.pi / m_nodes)
        lam = (m_nodes / t) * (A0 - A1 * th * th + 1j * A2 * th)
        dlam = (m_nodes / t) * (-2.0 * A1 * th + 1j * A2)
        r = resolvent_solve(a, lam, g)
        acc += np.exp(lam * t) * r * dlam
    return (1.0 / (2j * np.pi)) * acc * (2.0 * np.pi / m_nodes)


def contour_re_signs(t: float, m_nodes: int) -> np.ndarray:
    """Re(λ_k) of all contour nodes (for the LHP-guard sub-check)."""
    res = []
    for k in range(m_nodes):
        th = -np.pi + (k + 0.5) * (2.0 * np.pi / m_nodes)
        lam = (m_nodes / t) * (A0 - A1 * th * th + 1j * A2 * th)
        res.append(lam.real)
    return np.array(res)


def main() -> int:
    n = 64
    dx = 10.0 / (n - 1)
    a = laplacian_1d(n, dx)
    xs = np.linspace(-5.0, 5.0, n)
    g = np.exp(-(xs**2))  # Gaussian datum (matches G24 / G_CPLX_RES setup)

    print("resolvent time-jump amortization PRE-FLIGHT (F2, ADR-0134)")
    print(f"  model: A = ∂²ₓ (1D Laplacian, N={n}), ω = 0, σ(A) ⊂ (−∞, 0]")
    print("  contour: TWS-2006 parabolic, λ(θ)=(M/t)(0.1309−0.1194θ²+0.25iθ)")
    print()
    ok = True

    # ---- sub-check 1: geometric decay in M (at a LARGE t) -----------------
    t = 20.0
    ref = expm(t * a) @ g
    ms = [8, 12, 16, 20, 24, 28]
    errs = [float(np.max(np.abs(np.real(time_jump(a, t, g, m)) - ref))) for m in ms]
    rate = float(np.polyfit(ms, np.log10(errs), 1)[0])
    print(f"(1) geometric_decay (t={t}):")
    for m, e in zip(ms, errs):
        print(f"    M={m:3d}  err_inf={e:.3e}  log10={np.log10(e):+.2f}")
    print(f"    decay rate = {rate:+.3f} decades/node  (geometric ⇒ ≤ −0.3)")
    ok &= rate <= -0.3

    # ---- sub-check 2: t-independence of the node count --------------------
    print("(2) t_independence (err-vs-M coincides across t):")
    ts = [1.0, 20.0, 100.0]
    m_probe = 16
    e_by_t = []
    for tt in ts:
        rr = expm(tt * a) @ g
        e = float(np.max(np.abs(np.real(time_jump(a, tt, g, m_probe)) - rr)))
        e_by_t.append(e)
        print(f"    t={tt:6.1f}  err(M={m_probe})={e:.3e}")
    spread = max(e_by_t) / min(e_by_t)
    print(f"    spread max/min = {spread:.2f}×  (t-independent ⇒ ≤ 10×)")
    ok &= spread <= 10.0

    # ---- sub-check 3: G_RESOLVENT_JUMP_ORDER slope ------------------------
    t = 100.0  # large-T target
    ref = expm(t * a) @ g
    ms = [6, 8, 10, 12, 14]
    errs = [float(np.max(np.abs(np.real(time_jump(a, t, g, m)) - ref))) for m in ms]
    inv = np.log([1.0 / m for m in ms])
    slope = float(np.polyfit(inv, np.log(errs), 1)[0])
    print(f"(3) order_slope (G24-convention, t={t}):")
    for m, e in zip(ms, errs):
        print(f"    M={m:3d}  err_inf={e:.3e}")
    print(f"    slope d log(err)/d log(1/M) = {slope:+.3f}  (gate ≥ 1.95)")
    ok &= slope >= 1.95

    # ---- sub-check 4: LHP guard (NARROW boundary) -------------------------
    print("(4) lhp_guard (NARROW boundary: contour enters Re λ < 0):")
    res = contour_re_signs(20.0, 24)
    n_lhp = int(np.sum(res <= 0.0))
    print(f"    Re(λ_k) range [{res.min():+.3f}, {res.max():+.3f}], "
          f"#(Re≤0)={n_lhp}/24")
    print("    ⇒ shipped GL₃₂ eval_complex (valid Re λ > ω) would DIVERGE here;")
    print("    engineer MUST use an LHP-capable resolvent solve (NOT eval_complex).")
    ok &= n_lhp > 0  # the optimal contour necessarily dips into the LHP

    print()
    if ok:
        print("T_RESOLVENT_JUMP PASS")
        return 0
    print("T_RESOLVENT_JUMP FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
