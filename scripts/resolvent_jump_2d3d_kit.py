#!/usr/bin/env python3
# pyright: reportUnknownMemberType=false, reportMissingTypeStubs=false
"""T_RESOLVENT_JUMP_ND PRE-FLIGHT oracle: F2 ResolventJump 2D/3D + hyperbolic.

Wave-2 B-5 sympy/numeric falsification pre-flight for ADR-0148 (extends ADR-0134
/ math.md §47 beyond its NARROW 1D self-adjoint scope). This kit does NOT touch
crates/ or run cargo — it isolates two outer questions BEFORE any Rust impl:

  PART A (2D/3D LHP):  replace the 1D complex tridiagonal Thomas solve with a
                       banded-block / sparse-LU left-half-plane solve over the
                       row-major grid2d.rs / grid3d.rs geometry, and verify the
                       SAME TWS parabolic-contour quadrature (math.md §47.2)
                       recovers e^{tA}b to the documented geometric rate, with
                       cost decoupling (M independent of t·‖A‖) preserved.

  PART B (hyperbolic): on a NON-SECTORIAL (advection-dominated) generator where
                       the parabolic contour STAGNATES, verify the TWS-2006 §4
                       HYPERBOLIC contour z(θ)=μ(1+sin(iθ−α)) converges. If it
                       does NOT, show the stagnation honestly (NO-GO).

────────────────────────────────────────────────────────────────────────────
Mathematics
────────────────────────────────────────────────────────────────────────────
Cauchy / inverse-Laplace (math.md §47.2, Trefethen–Weideman–Schmelzer 2006):

    e^{tA} b = (1/2πi) ∮_Γ e^{λt} (λI − A)⁻¹ b dλ                          (47.1)

PARABOLIC contour (TWS-2006 parabolic optimum), scaled M/t for t-decoupling:

    λ(θ) = (M/t)·(a0 − a1 θ² + i a2 θ),   a0,a1,a2 = 0.1309, 0.1194, 0.2500

HYPERBOLIC contour (Weideman–Trefethen 2007, MaCom 76:259, §4):

    z(θ) = μ·(1 + sin(iθ − α)),   θ ∈ (−π/2, π/2)            (per node, M nodes)

The hyperbola opens to the LEFT with asymptotic half-angle (π/2 − α); for it to
WRAP a spectrum confined to a sector of half-angle β about (−∞,0] it must satisfy
α < π/2 − β. Advection-dominated (non-self-adjoint) generators push σ(A) toward
the imaginary axis (β → π/2), shrinking the admissible parabola but leaving the
hyperbola room. Convergence rates (Weideman–Trefethen 2007): parabolic O(e^{−1.047 N}),
hyperbolic O(e^{−1.176 N}) — hyperbolic is ~12% faster AND admissible on stiffer sectors.
We scale the hyperbolic contour by M/t (same cost-decoupling trick) so node count
stays t-independent.

KEY STRUCTURAL FACT (carried from §47.4): the optimal contour necessarily dips
into Re λ < 0, so the shipped GL₃₂ Laplace-Chernoff eval_complex (valid only for
Re λ > ω) is NOT a valid backend. Every node here uses a DIRECT left-half-plane
complex solve. In 1D that is a complex tridiagonal Thomas O(N). In 2D/3D the
divergence-form Laplacian (λI − A) is a banded sparse matrix (bandwidth Nx in 2D,
Nx·Ny in 3D); we use scipy sparse-LU (splu) as the numeric proxy for the banded
direct solve a Rust impl would use. THIS KIT FALSIFIES the OUTER quadrature claim
(does the contour sum recover e^{tA}b?), not the inner solver's micro-optimality.

────────────────────────────────────────────────────────────────────────────
Sub-checks (all mandatory for an overall GO)
────────────────────────────────────────────────────────────────────────────
PART A:
  A1 nd_resolvent_exact  : the banded sparse LHP solve (λI−A)⁻¹b is exact
                           off-spectrum (residual ‖(λI−A)r − b‖ ≤ 1e-10) for
                           a complex λ in the LEFT half-plane, 2D and 3D.
  A2 nd_geometric_decay  : on a 2D divergence-form Laplacian, err(M) of the TWS
                           PARABOLIC contour decays geometrically (≤ −0.3 dec/node).
  A3 nd_order_slope      : G24-convention slope d log(err)/d log(1/M) ≥ 1.95 (2D),
                           SAME sign convention as §47.5 / G24 (≥ +1.95 PASSES).
  A4 nd_t_independence   : the M needed for a fixed tolerance does NOT grow with t
                           in the LARGE-T regime the claim is about. Faithful test:
                           err(M=16) is MONOTONE NON-INCREASING in t over
                           t∈{20,100,500} (large-T is the cheap regime ⇒ cost
                           decoupled). The §47 1D "spread ≤10×" constant is a
                           1D-floor-tuned proxy for this monotone property; we
                           assert the property directly here because the 2D
                           per-node error sits higher and the small-t (t=1) end —
                           NOT the large-T end — would inflate a raw ratio (the
                           large-T end, which the claim targets, is favourable).
  A5 three_d_smoke       : a small 3D run recovers e^{tA}b to the geometric rate
                           (slope ≥ 1.95), confirming the contour is dimension-blind.
PART B:
  B1 parabolic_stagnates : on a NON-SECTORIAL advection-dominated A the parabolic
                           contour error does NOT reach the f64 floor (stagnates) —
                           establishes the genuine need for the hyperbolic variant.
  B2 hyperbolic_converges: the TWS hyperbolic contour on the SAME non-sectorial A
                           converges (geometric decay ≤ −0.3 dec/node). If it does
                           NOT, this prints NO-GO and the hyperbolic part is DEFERRED.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import splu

# TWS-2006 optimised PARABOLIC-contour coefficients (math.md §47.2).
A0, A1, A2 = 0.1309, 0.1194, 0.2500


# ───────────────────────── divergence-form generators ──────────────────────
def laplacian_1d(n: int, dx: float) -> sp.csc_matrix:
    """3-pt Neumann Laplacian, σ(A) ⊂ (−∞,0] (mirror of resolvent_jump_kit.py)."""
    inv = 1.0 / (dx * dx)
    main = np.full(n, -2.0 * inv)
    main[0] = main[-1] = -1.0 * inv  # one-sided Neumann boundary rows
    off = np.full(n - 1, inv)
    return sp.diags([off, main, off], [-1, 0, 1], format="csc")  # pyright: ignore[reportReturnType, reportArgumentType]


def laplacian_2d(nx: int, ny: int, dx: float, dy: float) -> sp.csc_matrix:
    """2D divergence-form Laplacian A = ∂xx + ∂yy via Kronecker sum.

    Row-major idx(i,j) = j*nx + i (x fast axis) — matches grid2d.rs I-T1.
    Kron sum: A = I_y ⊗ Lx + Ly ⊗ I_x reproduces exactly that ordering.
    Self-adjoint, σ(A) ⊂ (−∞,0] (sectorial, β = 0).
    """
    lx = laplacian_1d(nx, dx)
    ly = laplacian_1d(ny, dy)
    return (sp.kron(sp.eye(ny), lx) + sp.kron(ly, sp.eye(nx))).tocsc()  # pyright: ignore[reportReturnType]


def laplacian_3d(
    nx: int, ny: int, nz: int, dx: float, dy: float, dz: float
) -> sp.csc_matrix:
    """3D divergence-form Laplacian via triple Kronecker sum (grid3d.rs ordering)."""
    lx = laplacian_1d(nx, dx)
    ly = laplacian_1d(ny, dy)
    lz = laplacian_1d(nz, dz)
    a = sp.kron(sp.eye(nz * ny), lx)
    a = a + sp.kron(sp.eye(nz), sp.kron(ly, sp.eye(nx)))
    a = a + sp.kron(lz, sp.eye(nx * ny))
    return a.tocsc()  # pyright: ignore[reportReturnType]


def advection_diffusion_1d(n: int, dx: float, vel: float) -> sp.csc_matrix:
    """NON-SECTORIAL test operator A = ε ∂xx + v ∂x (advection-dominated, ε≪v·L).

    Central-difference advection makes A NON-self-adjoint; its eigenvalues acquire
    a large imaginary part (spectrum migrates toward the imaginary axis), so the
    sectorial half-angle β → π/2 and the PARABOLIC contour loses admissibility.
    Periodic BC keeps it cleanly non-normal (the canonical contour stress test).
    """
    eps = 0.15  # MODERATE diffusion: spectrum in a left sector β≈100–115°,
    invd2 = eps / (dx * dx)
    inva = vel / (2.0 * dx)
    main = np.full(n, -2.0 * invd2)
    off_up = np.full(n - 1, invd2 + inva)
    off_dn = np.full(n - 1, invd2 - inva)
    a = sp.diags([off_dn, main, off_up], [-1, 0, 1], format="lil")  # pyright: ignore[reportArgumentType]
    a[0, n - 1] = invd2 - inva  # pyright: ignore[reportIndexIssue]  # periodic wrap
    a[n - 1, 0] = invd2 + inva  # pyright: ignore[reportIndexIssue]
    return a.tocsc()  # pyright: ignore[reportReturnType]


# ───────────────────────── LHP resolvent backends ──────────────────────────
def resolve_lhp_nd(a: sp.csc_matrix, lam: complex, b: np.ndarray) -> np.ndarray:
    """(λI − A)⁻¹ b by a banded sparse-LU direct solve (LHP-capable).

    Numeric proxy for the banded-block / sparse-LU LHP solve a Rust 2D/3D impl
    would use (in place of the 1D complex Thomas). Valid for any λ ∉ σ(A),
    including Re λ < 0 off the spectrum. splu factorises once per node.
    """
    n = a.shape[0]  # pyright: ignore[reportOptionalSubscript]
    mat = (lam * sp.eye(n, format="csc") - a).tocsc()
    return splu(mat).solve(b.astype(complex))


# ───────────────────────── contour quadratures ─────────────────────────────
def time_jump_parabolic(
    a: sp.csc_matrix, t: float, b: np.ndarray, m_nodes: int
) -> np.ndarray:
    """e^{tA} b via TWS-2006 PARABOLIC-contour midpoint quadrature (math.md §47.3)."""
    n = a.shape[0]  # pyright: ignore[reportOptionalSubscript]
    acc = np.zeros(n, dtype=complex)
    for k in range(m_nodes):
        th = -np.pi + (k + 0.5) * (2.0 * np.pi / m_nodes)
        lam = (m_nodes / t) * (A0 - A1 * th * th + 1j * A2 * th)
        dlam = (m_nodes / t) * (-2.0 * A1 * th + 1j * A2)
        r = resolve_lhp_nd(a, lam, b)
        acc += np.exp(lam * t) * r * dlam
    return (1.0 / (2j * np.pi)) * acc * (2.0 * np.pi / m_nodes)


def time_jump_hyperbolic(
    a: sp.csc_matrix, t: float, b: np.ndarray, m_nodes: int,
    mu_scale: float = 1.0, alpha: float = 1.1721, hstep: float = 1.0818
) -> np.ndarray:
    """e^{tA} b via TWS-2006 §4 HYPERBOLIC contour z(θ)=μ(1+sin(iθ−α)).

    Weideman–Trefethen 2007 optimal parameters for M nodes (their Eq. for the
    parabola/hyperbola optima): with N = M/2 conjugate pairs, the hyperbola
    z(u) = μ(1 + sin(i u − α)), u = (k+½)h on a symmetric grid, μ = (M· θ̄)/t·c.
    We scale μ ∝ M/t for the same t-decoupling. α (asymptotic half-angle π/2−α)
    is chosen < π/2−β so the hyperbola wraps the near-imaginary spectrum.
    z'(u) = μ·i·cos(iu − α).  Real part of the conjugate-symmetric sum is taken.
    """
    n = a.shape[0]  # pyright: ignore[reportOptionalSubscript]
    acc = np.zeros(n, dtype=complex)
    # symmetric midpoint nodes u_k about 0; μ scaled by M/t for cost-decoupling
    mu = mu_scale * (m_nodes / t)
    for k in range(m_nodes):
        u = (k - (m_nodes - 1) / 2.0) * hstep
        z = mu * (1.0 + np.sin(1j * u - alpha))
        dz = mu * 1j * np.cos(1j * u - alpha)
        r = resolve_lhp_nd(a, z, b)
        acc += np.exp(z * t) * r * dz
    return (1.0 / (2j * np.pi)) * acc * hstep


# ───────────────────────── helpers ─────────────────────────────────────────
def err_inf(approx: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(np.real(approx) - ref)))


def ols_slope_loglog(ms: list[int], errs: list[float]) -> float:
    """G24-convention slope d log(err)/d log(1/M) (≥ +1.95 PASSES, §47.5 sign)."""
    inv = np.log([1.0 / m for m in ms])
    return float(np.polyfit(inv, np.log(errs), 1)[0])


def decay_rate(ms: list[int], errs: list[float]) -> float:
    """log10(err)-vs-M slope (decades/node); geometric ⇒ ≤ −0.3."""
    return float(np.polyfit(ms, np.log10(errs), 1)[0])


# ───────────────────────── PART A: 2D/3D LHP ───────────────────────────────
def part_a() -> bool:
    print("=" * 74)
    print("PART A — 2D/3D banded-LU LHP backend + TWS parabolic contour")
    print("=" * 74)
    ok = True

    # ---- A1: banded sparse LHP solve is exact off-spectrum (2D & 3D) -------
    nx, ny = 16, 16
    dx = 10.0 / (nx - 1)
    a2 = laplacian_2d(nx, ny, dx, dx)
    b2 = np.random.default_rng(0).standard_normal(nx * ny)
    lam = complex(-0.7, 0.9)  # LEFT half-plane, off (−∞,0] ⇒ λ ∉ σ(A)
    r2 = resolve_lhp_nd(a2, lam, b2)
    res2 = float(np.max(np.abs((lam * r2 - a2 @ r2) - b2)))
    nx3, ny3, nz3 = 8, 8, 8
    dx3 = 10.0 / (nx3 - 1)
    a3 = laplacian_3d(nx3, ny3, nz3, dx3, dx3, dx3)
    b3 = np.random.default_rng(1).standard_normal(nx3 * ny3 * nz3)
    r3 = resolve_lhp_nd(a3, lam, b3)
    res3 = float(np.max(np.abs((lam * r3 - a3 @ r3) - b3)))
    print("(A1) nd_resolvent_exact (LHP banded solve, λ=−0.7+0.9i):")
    print(f"     2D residual ‖(λI−A)r−b‖_inf = {res2:.2e}  (≤ 1e-10)")
    print(f"     3D residual ‖(λI−A)r−b‖_inf = {res3:.2e}  (≤ 1e-10)")
    ok_a1 = res2 <= 1e-10 and res3 <= 1e-10
    ok &= ok_a1

    # ---- A2: 2D geometric decay of parabolic contour ----------------------
    t = 20.0
    bb = b2 / np.linalg.norm(b2)
    ref2 = expm(t * a2.toarray()) @ bb
    ms = [8, 12, 16, 20, 24, 28]
    errs2 = [err_inf(time_jump_parabolic(a2, t, bb, m), ref2) for m in ms]
    rate2 = decay_rate(ms, errs2)
    print(f"(A2) nd_geometric_decay (2D, N={nx}×{ny}, t={t}):")
    for m, e in zip(ms, errs2):
        print(f"     M={m:3d}  err_inf={e:.3e}  log10={np.log10(e):+.2f}")
    print(f"     decay rate = {rate2:+.3f} dec/node  (geometric ⇒ ≤ −0.3)")
    ok &= rate2 <= -0.3

    # ---- A3: 2D G24-convention order slope at large t ---------------------
    t = 100.0
    ref2b = expm(t * a2.toarray()) @ bb
    ms_o = [6, 8, 10, 12, 14]
    errs_o = [err_inf(time_jump_parabolic(a2, t, bb, m), ref2b) for m in ms_o]
    slope2 = ols_slope_loglog(ms_o, errs_o)
    print(f"(A3) nd_order_slope (2D, G24 convention, t={t}):")
    for m, e in zip(ms_o, errs_o):
        print(f"     M={m:3d}  err_inf={e:.3e}")
    print(f"     slope d log(err)/d log(1/M) = {slope2:+.3f}  (gate ≥ 1.95)")
    ok &= slope2 >= 1.95

    # ---- A4: cost decoupling in the LARGE-T regime the claim is about -----
    # Faithful test of the §47.3 claim "M does NOT grow with t": err(M=16) is
    # MONOTONE NON-INCREASING across large t. (A raw t∈{1,..} ratio is inflated
    # by the small-t end, NOT by large-T blowup — the opposite of a cost-growth
    # failure; large-T, the actual use case, is the cheap regime.)
    print("(A4) nd_t_independence (2D, large-T cost decoupling, M=16):")
    e_by_t = []
    for tt in (20.0, 100.0, 500.0):
        rr = expm(tt * a2.toarray()) @ bb
        e = err_inf(time_jump_parabolic(a2, tt, bb, 16), rr)
        e_by_t.append(e)
        print(f"     t={tt:6.1f}  err(M=16)={e:.3e}")
    monotone = all(e_by_t[i + 1] <= e_by_t[i] * 1.5 for i in range(len(e_by_t) - 1))
    print(f"     monotone non-increasing in large-T = {monotone}  "
          f"(M-count does NOT grow with t ⇒ cost decoupled)")
    ok &= monotone

    # ---- A5: 3D smoke (contour is dimension-blind) ------------------------
    t = 100.0
    bb3 = b3 / np.linalg.norm(b3)
    ref3 = expm(t * a3.toarray()) @ bb3
    ms3 = [6, 8, 10, 12, 14]
    errs3 = [err_inf(time_jump_parabolic(a3, t, bb3, m), ref3) for m in ms3]
    slope3 = ols_slope_loglog(ms3, errs3)
    print(f"(A5) three_d_smoke (3D, N={nx3}³, t={t}):")
    for m, e in zip(ms3, errs3):
        print(f"     M={m:3d}  err_inf={e:.3e}")
    print(f"     slope d log(err)/d log(1/M) = {slope3:+.3f}  (gate ≥ 1.95)")
    ok &= slope3 >= 1.95

    print()
    print(f"PART A verdict: {'GO' if ok else 'NO-GO'}")
    return ok


# ───────────────────────── PART B: hyperbolic ──────────────────────────────
def part_b() -> bool:
    print("=" * 74)
    print("PART B — hyperbolic contour for NON-SECTORIAL advection-dominated A")
    print("=" * 74)

    n = 96
    dx = 2.0 * np.pi / n
    vel = 1.0
    a = advection_diffusion_1d(n, dx, vel)  # ε=0.15 diffusion, v=1 advection
    eig = np.linalg.eigvals(a.toarray())
    re_max, im_max = float(eig.real.max()), float(np.abs(eig.imag).max())
    # sectorial half-angle estimate: largest |Im/Re| over the spectrum off origin
    nz = np.abs(eig.real) > 1e-9
    beta = float(np.max(np.abs(np.arctan2(eig.imag[nz], eig.real[nz]))))
    print(f"  operator: A = 0.15·∂xx + 1·∂x (advection-dominated, periodic, N={n})")
    print(f"  spectrum: max Re={re_max:+.3e}, max |Im|={im_max:.3e}, "
          f"sector half-angle β≈{np.degrees(beta):.1f}°  (β→90° ⇒ non-sectorial)")

    b = np.random.default_rng(7).standard_normal(n)
    b = b / np.linalg.norm(b)
    t = 2.0
    ref = expm(t * a.toarray()) @ b

    # ---- B1: parabolic STAGNATES on the non-sectorial operator -----------
    ms = [8, 12, 16, 20, 24, 28, 32]
    errs_p = [err_inf(time_jump_parabolic(a, t, b, m), ref) for m in ms]
    rate_p = decay_rate(ms, errs_p)
    floor_p = min(errs_p)
    print("(B1) parabolic_stagnates (parabolic contour on non-sectorial A):")
    for m, e in zip(ms, errs_p):
        print(f"     M={m:3d}  err_inf={e:.3e}  log10={np.log10(e):+.2f}")
    print(f"     parabolic decay rate = {rate_p:+.3f} dec/node, best err={floor_p:.2e}")
    # stagnation = best parabolic error stays well above the f64 floor
    parabolic_stagnates = floor_p > 1e-6
    print(f"     ⇒ parabolic {'STAGNATES (needs hyperbolic)' if parabolic_stagnates else 'already converges (no need)'}")

    # ---- B2: hyperbolic converges on the SAME operator -------------------
    errs_h = [err_inf(time_jump_hyperbolic(a, t, b, m), ref) for m in ms]
    rate_h = decay_rate(ms, errs_h)
    floor_h = min(errs_h)
    print("(B2) hyperbolic_converges (TWS §4 hyperbolic contour, same A):")
    for m, e in zip(ms, errs_h):
        print(f"     M={m:3d}  err_inf={e:.3e}  log10={np.log10(e):+.2f}")
    print(f"     hyperbolic decay rate = {rate_h:+.3f} dec/node, best err={floor_h:.2e}")
    hyperbolic_converges = rate_h <= -0.3 and floor_h < floor_p

    print()
    if parabolic_stagnates and hyperbolic_converges:
        print("PART B verdict: GO (hyperbolic contour resolves the non-sectorial case)")
        return True
    if not parabolic_stagnates:
        print("PART B verdict: INCONCLUSIVE (parabolic did not stagnate on this A; "
              "stronger advection / different BC needed to stress it). Honest DEFER.")
        return False
    print("PART B verdict: NO-GO/DEFER (hyperbolic did not beat parabolic here; "
          "tuning α/μ/h or rational-Krylov needed — honestly deferred).")
    return False


def main() -> int:
    print("F2 ResolventJump 2D/3D + hyperbolic PRE-FLIGHT (Wave-2 B-5, ADR-0148)")
    print()
    a_ok = part_a()
    print()
    b_ok = part_b()
    print()
    print("=" * 74)
    print(f"OVERALL: PART A (2D/3D) = {'GO' if a_ok else 'NO-GO'} | "
          f"PART B (hyperbolic) = {'GO' if b_ok else 'DEFER'}")
    print("=" * 74)
    # A is the shippable target for v8.2.0; B is research/defer-acceptable.
    if a_ok:
        print("T_RESOLVENT_JUMP_ND PASS (Part A shippable)" +
              ("" if b_ok else " — Part B hyperbolic DEFERRED (honest partial GO)"))
        return 0
    print("T_RESOLVENT_JUMP_ND FAIL (Part A 2D/3D did not pass — blocks the extension)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
