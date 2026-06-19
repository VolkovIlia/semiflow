#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false
"""PRE-FLIGHT + T_CPLX_RES oracle: complex-λ Laplace-Chernoff resolvent.

Feature 3 (v7.0.0 backlog #14, ADR-0129 in this campaign's free range).

math.md §22 ships the resolvent (λI − A)⁻¹ g via the Hille-Yosida Laplace
representation, GATED for real λ > ω only (§22.7 limitation #1: "Complex
resolvent values R(λ; A) g for λ ∈ ℂ require complex-valued semigroup
support — deferred to v4.0 B6 SemiflowComplex"). SemiflowComplex now exists
(src/complex.rs). This PRE-FLIGHT verifies the SAME Gauss-Laguerre quadrature
extends to complex λ with Re λ > ω.

────────────────────────────────────────────────────────────────────────────
Mathematics (Hille-Yosida, Pazy 1983 §1.5; Engel-Nagel 2000 §II.1)
────────────────────────────────────────────────────────────────────────────
For a generator A of a C₀-semigroup S(t) with growth bound ω (‖S(t)‖ ≤ M e^{ωt}),
the resolvent set ρ(A) ⊇ {λ : Re λ > ω} and on that half-plane

    (λI − A)⁻¹ g = ∫₀^∞ e^{−λ t} S(t) g dt              (Laplace representation)

The integral converges absolutely because |e^{−λt}| = e^{−(Re λ) t} and
‖S(t) g‖ ≤ M e^{ω t}‖g‖, so the integrand is bounded by M‖g‖ e^{−(Re λ − ω) t},
which is integrable iff Re λ > ω. The IMAGINARY part of λ contributes only a
bounded oscillation e^{−i (Im λ) t} (modulus 1) and does NOT affect convergence.

Laplace-Chernoff quadrature (math.md §22.2, Remizov 2025 Vladikavkaz Thm 3):
substitute s = λ t (now a COMPLEX change of variable along the ray arg λ):

    (λI − A)⁻¹ g ≈ (1/λ) Σ_k w_k · (C(s_k/(λn)))^n g       (Gauss-Laguerre 32-pt)

where s_k, w_k are the (real, positive) GL nodes/weights for ∫₀^∞ e^{−s} f(s) ds.
The contour deformation t ∈ [0,∞) → s = λt is valid (Cauchy's theorem) because
the integrand e^{−λt}S(t)g is analytic in t in the sector |arg t| < π/2 − |arg λ|
and decays on the arc; this is the standard Laplace-transform contour rotation.
For C the exact semigroup C(τ)=S(τ), the quadrature is EXACT up to GL truncation.

This script tests the quadrature numerically on a canonical SELF-ADJOINT model
operator A = ∂²ₓ (1D Laplacian, reflecting BC) discretised on a grid. Its
spectrum is real ≤ 0, so ω = 0 and the formula holds for ANY λ with Re λ > 0,
INCLUDING complex λ. The exact resolvent is computed by eigendecomposition;
the Laplace-Chernoff quadrature uses the EXACT matrix exponential as C (so the
only error is GL truncation — isolating the quadrature-convergence question).

────────────────────────────────────────────────────────────────────────────
Gate T_CPLX_RES — 4 sub-checks (all mandatory)
────────────────────────────────────────────────────────────────────────────
  (1) convergence_real_axis : complex-λ formula reduces to real-λ shipped result
                              when Im λ = 0 (byte-comparable residual).
  (2) resolvent_identity     : ‖(λI − A) R̃(λ) g − g‖_∞ ≤ 1e-3 for a set of
                              complex λ with Re λ > 0 (the backlog exit
                              criterion G_CPLX_RES residual ≤ 1e-3).
  (3) growth_bound_guard     : for Re λ ≤ ω the Laplace integrand does NOT
                              decay → residual blows up; confirms the guard
                              boundary is Re λ > ω (NOT |λ| > ω — a spec trap).
  (4) analyticity_cauchy     : R̃(λ) is analytic in the right half-plane:
                              numerically check the Cauchy-Riemann residual of
                              the quadrature R̃(λ) on a small λ-grid is ~0.
"""

import numpy as np
from scipy.linalg import expm

# Gauss-Laguerre 32-pt nodes/weights (the table shipped in resolvent_quad.rs).
GL_NODES, GL_WEIGHTS = np.polynomial.laguerre.laggauss(32)


def laplacian_1d(n: int, dx: float) -> np.ndarray:
    """3-point Laplacian A = ∂²ₓ with reflecting (Neumann) BCs. Symmetric, ≤ 0."""
    a = np.zeros((n, n))
    inv = 1.0 / (dx * dx)
    for i in range(n):
        a[i, i] = -2.0 * inv
        if i > 0:
            a[i, i - 1] = inv
        if i < n - 1:
            a[i, i + 1] = inv
    # reflecting BC: rows 0 and n-1 keep only one neighbour (already so above),
    # symmetrise the boundary diagonal so A stays symmetric negative-semidefinite.
    a[0, 0] = -1.0 * inv
    a[n - 1, n - 1] = -1.0 * inv
    return a


def exact_resolvent(a: np.ndarray, lam: complex, g: np.ndarray) -> np.ndarray:
    """(λI − A)⁻¹ g by direct complex solve (reference)."""
    n = a.shape[0]
    return np.linalg.solve(lam * np.eye(n, dtype=complex) - a, g.astype(complex))


def laplace_chernoff_resolvent(
    a: np.ndarray, lam: complex, g: np.ndarray, n_chern: int
) -> np.ndarray:
    """Laplace-Chernoff GL32 quadrature with the EXACT semigroup C(τ)=exp(τA).

    R̃(λ) g = (1/λ) Σ_k w_k · (exp((s_k/(λ n)) A))^n g
            = (1/λ) Σ_k w_k · exp((s_k/λ) A) g     (since (exp(τA))^n = exp(nτA))
    The n-power telescopes for the EXACT semigroup; we keep n explicit to mirror
    the Rust kernel's (C(τ))^n structure (where C ≠ exact for a real Chernoff fn).
    """
    g_c = g.astype(complex)
    acc = np.zeros_like(g_c)
    for s_k, w_k in zip(GL_NODES, GL_WEIGHTS):
        tau = s_k / (lam * n_chern)  # COMPLEX step
        # (exp(τA))^n g  via exact matrix exp; complex τ.
        step = expm(tau * a.astype(complex))
        v = g_c.copy()
        for _ in range(n_chern):
            v = step @ v
        acc += (w_k / lam) * v
    return acc


def residual_inf(a: np.ndarray, lam: complex, g: np.ndarray, r: np.ndarray) -> float:
    """‖(λI − A) r − g‖_∞ (the resolvent-identity residual used by G_CPLX_RES)."""
    return float(np.max(np.abs((lam * r - a @ r) - g.astype(complex))))


def main() -> int:
    n = 64
    dx = 10.0 / (n - 1)
    a = laplacian_1d(n, dx)
    xs = np.linspace(-5.0, 5.0, n)
    g = np.exp(-(xs**2))  # Gaussian datum (matches G_RES_RES setup)
    n_chern = 8

    print("complex-λ Laplace-Chernoff resolvent PRE-FLIGHT")
    print(f"  model: A = ∂²ₓ (1D Laplacian, N={n}), ω = 0 (spectrum ≤ 0)")
    print(f"  quadrature: GL32, n_chern = {n_chern}, exact semigroup C = exp(τA)")
    print()

    ok = True

    # ---- sub-check 1: real-axis reduction ---------------------------------
    lam_real = 1.0 + 0.0j
    r_q = laplace_chernoff_resolvent(a, lam_real, g, n_chern)
    r_ex = exact_resolvent(a, lam_real, g)
    real_axis_err = float(np.max(np.abs(r_q - r_ex)))
    print(f"(1) convergence_real_axis : λ=1.0,  ‖R̃−R_exact‖_∞ = {real_axis_err:.3e}")
    ok &= real_axis_err <= 1e-3

    # ---- sub-check 2: complex-λ resolvent identity ------------------------
    print("(2) resolvent_identity (complex λ, Re λ > 0):")
    complex_lams = [
        1.0 + 1.0j,
        1.0 - 1.0j,
        2.0 + 3.0j,
        0.5 + 2.0j,
        3.0 + 0.5j,
    ]
    worst = 0.0
    for lam in complex_lams:
        r_q = laplace_chernoff_resolvent(a, lam, g, n_chern)
        res = residual_inf(a, lam, g, r_q)
        # also compare to exact solve for a convergence read
        r_ex = exact_resolvent(a, lam, g)
        conv = float(np.max(np.abs(r_q - r_ex)))
        worst = max(worst, res)
        flag = "OK " if res <= 1e-3 else "XX "
        print(
            f"    {flag}λ={lam.real:+.1f}{lam.imag:+.1f}i : "
            f"residual ‖(λI−A)R̃−g‖_∞ = {res:.3e}   ‖R̃−R_exact‖_∞ = {conv:.3e}"
        )
    print(f"    worst residual = {worst:.3e}   (gate ≤ 1e-3)")
    ok &= worst <= 1e-3

    # ---- sub-check 3: growth-bound guard (Re λ > ω, NOT |λ| > ω) ----------
    # ω = 0 here. Take λ with |λ| large but Re λ < 0  → integral must DIVERGE.
    print("(3) growth_bound_guard (spec trap: boundary is Re λ > ω, not |λ| > ω):")
    lam_bad = -0.5 + 5.0j  # |λ| = 5.02 large, but Re λ = -0.5 < ω = 0
    try:
        r_bad = laplace_chernoff_resolvent(a, lam_bad, g, n_chern)
        res_bad = residual_inf(a, lam_bad, g, r_bad)
    except Exception:  # noqa: BLE001
        res_bad = float("inf")
    diverges = (not np.isfinite(res_bad)) or res_bad > 1e-1
    print(
        f"    λ={lam_bad.real:+.1f}{lam_bad.imag:+.1f}i (|λ|={abs(lam_bad):.2f}, "
        f"Re λ < ω): residual = {res_bad:.3e}  → "
        f"{'DIVERGES (guard correct)' if diverges else 'CONVERGES (trap!)'}"
    )
    ok &= diverges  # the guard must reject this; library MUST check Re λ > ω

    # ---- sub-check 4: analyticity (Cauchy-Riemann of the quadrature) ------
    # R̃(λ) entrywise must be holomorphic. Check CR on the first component via
    # finite-difference ∂R/∂(Re λ) + i? No: CR is ∂u/∂x = ∂v/∂y, ∂u/∂y=−∂v/∂x.
    print("(4) analyticity_cauchy (CR residual of R̃ at λ₀ = 2 + 1i):")
    lam0 = 2.0 + 1.0j
    h = 1e-5
    comp = 0  # test the first grid component
    f = lambda l: laplace_chernoff_resolvent(a, l, g, n_chern)[comp]  # noqa: E731
    dfx = (f(lam0 + h) - f(lam0 - h)) / (2 * h)  # ∂/∂(Re λ)
    dfy = (f(lam0 + 1j * h) - f(lam0 - 1j * h)) / (2j * h)  # (1/i)∂/∂(Im λ)
    cr_res = float(abs(dfx - dfy))  # holomorphic ⇔ dfx == dfy
    print(f"    |∂R/∂x − (1/i)∂R/∂y| = {cr_res:.3e}   (holomorphic ⇒ ≈ 0)")
    ok &= cr_res <= 1e-4

    print()
    if ok:
        print("T_CPLX_RES PASS")
        return 0
    print("T_CPLX_RES FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
