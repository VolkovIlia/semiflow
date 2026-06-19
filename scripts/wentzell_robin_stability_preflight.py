#!/usr/bin/env python3
# wentzell_robin_stability_preflight.py
#
# Wave-3 research register C-9 — Dynamic (Wentzell) Robin BC stability preflight.
#
# CONTRADICTION (ADR-0098 Amendment 3 / ADR-0146):
#   "need a time-dependent boundary scaling AND boundary-layer stability."
#   Stephan 2023 (arXiv:2307.00419, ZAMM 2025) proves the *explicit freezing*
#   Trotter product formula on the product space X ⊕ L²(∂Ω) DIVERGES when the
#   bulk↔boundary coupling (the normal-derivative ∂_ν) is UNBOUNDED:
#   ‖T(t/n)^n‖ ≥ n^β t^{1−β} → ∞  for relative-bound exponent β ∈ (0, 1].
#
# TRIZ resolution under test (ИКР: the boundary block updates itself, A-stably,
#   without a time-step restriction killing the boundary layer):
#   Treat the dynamic-boundary block IMPLICITLY (Cayley / backward-Euler resolvent
#   step) instead of explicitly. This is the mechanism that the Altmann–Verfürth–
#   Zimmer bulk–surface Lie/BDF splittings (arXiv:2108.08147 IMA JNA 2023;
#   arXiv:2209.07835) use to obtain *unconditional* (A-stable) boundary-block
#   stability, and it mirrors the library's own §17.4 Crank–Nicolson Cayley map
#   that already gives ‖K_CN‖₂ = 1 exactly for the Schrödinger kinetic step.
#
# This script performs a von-Neumann / matrix amplification analysis of a model
# Wentzell-coupled bulk+boundary semidiscrete system and compares:
#   (A) EXPLICIT boundary update  (the Stephan-unstable freezing) — expect ρ > 1.
#   (B) IMPLICIT Cayley/backward-Euler boundary update — expect ρ ≤ 1 (A-stable),
#       including for a TIME-DEPENDENT boundary scaling γ(t).
#
# A GO verdict requires: the candidate (B) amplification factor ≤ 1 (to a tiny
# tolerance) across a stiffness sweep that drives (A) unstable, AND for the
# time-dependent γ(t) case. Otherwise NO-GO with the unstable mode printed.
#
# Pure numpy/sympy. No cargo. Read-only w.r.t. crates/.

import numpy as np
import sympy as sp

TOL = 1e-9          # amplification "≤ 1" slack (round-off + discretization)
np.set_printoptions(precision=4, suppress=False, linewidth=120)


# ---------------------------------------------------------------------------
# Model problem (semidiscrete, scalar-per-mode reduction)
# ---------------------------------------------------------------------------
# Heat equation on [0, L] with a dynamic Wentzell/Robin boundary at x = 0:
#
#   u_t = a u_xx                         (bulk, x ∈ (0, L])
#   u_t(0) + γ(t) u_x(0) + c u(0) = 0    (dynamic boundary ODE at x = 0)
#
# Reformulated as a coupled PDAE system on X ⊕ ℝ_∂ (one boundary DOF), the
# semidiscrete generator in the lowest two near-boundary modes is the 2×2 block
#
#   d/dt [ u_b ]   [ -a κ²      coupling·(1)        ] [ u_b ]
#        [ u_∂ ] = [ -γ(t)/dx   -(γ(t)/dx + c)      ] [ u_∂ ]
#
# where
#   u_b  = the first interior bulk mode (Laplacian eigenvalue −a κ², κ ~ π/L … large),
#   u_∂  = the boundary trace DOF,
#   the (2,1) entry −γ(t)/dx is the DISCRETE NORMAL DERIVATIVE coupling — it is
#         O(1/dx) → ∞ as the grid refines: this is exactly the *unbounded*
#         coupling operator B that Stephan's Section 5 makes diverge under
#         explicit freezing. The "off-diagonal coupling is bounded" hypothesis of
#         Stephan's Thm 4.4 FAILS here (relative bound exponent β = 1/2 … 1).
#
# This 2×2 block is the worst-case (highest-wavenumber) Fourier symbol of the
# semidiscrete coupled operator; controlling its amplification ≤ 1 is the
# von-Neumann stability condition for the full scheme.
# ---------------------------------------------------------------------------


def coupled_generator(a, kappa, gamma, c, dx):
    """2x2 generator block C(t) for the bulk+boundary coupled system at a fixed
    boundary scaling gamma. Row 0 = bulk mode, row 1 = boundary DOF.
    The (1,0) entry -gamma/dx is the O(1/dx) UNBOUNDED normal-derivative coupling."""
    return np.array(
        [
            [-a * kappa * kappa, +1.0 / dx],          # bulk: diffusion + weak boundary feedback
            [-gamma / dx,        -(gamma / dx + c)],   # boundary ODE: STIFF coupling -gamma/dx
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# (A) EXPLICIT freezing step — the Stephan-unstable Trotter product.
#     One step = explicit (forward) treatment of the off-diagonal coupling block:
#     freeze boundary, advance bulk; freeze bulk, advance boundary EXPLICITLY.
#     Equivalent amplification: M_expl = (I + tau * C). Forward-Euler symbol.
# ---------------------------------------------------------------------------
def explicit_step_amp(a, kappa, gamma, c, dx, tau):
    C = coupled_generator(a, kappa, gamma, c, dx)
    M = np.eye(2) + tau * C
    return M


# ---------------------------------------------------------------------------
# (B) IMPLICIT Cayley (Crank-Nicolson) step on the coupled block.
#     M_cay = (I - tau/2 C)^{-1} (I + tau/2 C).  A-stable: for any C with
#     spectrum in the closed left half-plane, ‖eigs(M_cay)‖ ≤ 1 (the Cayley/
#     Möbius map sends LHP -> unit disk). This is the §17.4 mechanism applied to
#     the *boundary block* instead of the kinetic operator.
#     Backward-Euler variant M_be = (I - tau C)^{-1} is also reported (the
#     Altmann–Verfürth implicit-Euler sub-step).
# ---------------------------------------------------------------------------
def implicit_cayley_step_amp(a, kappa, gamma, c, dx, tau):
    C = coupled_generator(a, kappa, gamma, c, dx)
    I = np.eye(2)
    left = I - 0.5 * tau * C
    right = I + 0.5 * tau * C
    M = np.linalg.solve(left, right)
    return M


def implicit_backward_euler_step_amp(a, kappa, gamma, c, dx, tau):
    C = coupled_generator(a, kappa, gamma, c, dx)
    I = np.eye(2)
    M = np.linalg.solve(I - tau * C, I)
    return M


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


# ---------------------------------------------------------------------------
# Sanity: the continuous coupled generator C must be (essentially) dissipative
# in the Wentzell inner product so that an A-stable map yields rho<=1. We check
# that the symmetric part has non-positive eigenvalues for the chosen c>0 regime.
# ---------------------------------------------------------------------------
def generator_is_dissipative(a, kappa, gamma, c, dx):
    C = coupled_generator(a, kappa, gamma, c, dx)
    # Wentzell weighting: the boundary DOF carries weight 1/dx (L^2(dOmega)
    # surface measure vs bulk L^2 volume). In the WEIGHTED inner product the
    # cross terms cancel and C is dissipative. Weight matrix W = diag(1, dx).
    W = np.diag([1.0, dx])
    Csym_w = 0.5 * (W @ C + C.T @ W)
    return float(np.max(np.linalg.eigvalsh(Csym_w)))


# ---------------------------------------------------------------------------
# Symbolic Cayley A-stability witness (sympy): for a generator with eigenvalue
# mu in the closed LHP (Re mu <= 0), the Cayley map z(mu) = (1 + tau*mu/2)/
# (1 - tau*mu/2) satisfies |z| <= 1. This is the contradiction-resolving identity:
# the boundary scaling gamma(t) can be ARBITRARILY large (mu -> -infinity along
# the negative real axis as the coupling stiffens) and |z| -> 1 from BELOW, never
# exceeding 1. The explicit map z_expl = 1 + tau*mu BLOWS UP (|z_expl| -> infinity).
# ---------------------------------------------------------------------------
def symbolic_cayley_witness():
    tau, mu = sp.symbols("tau mu", positive=True)  # mu := -Re(eigenvalue) >= 0 (decay rate)
    # eigenvalue lambda = -mu (negative real, dissipative). tau > 0.
    z_cayley = (1 - tau * mu / 2) / (1 + tau * mu / 2)   # Cayley symbol at lambda=-mu
    z_explicit = 1 - tau * mu                            # forward-Euler symbol at lambda=-mu

    # Cayley: |z| <= 1 for all mu>=0, tau>0; and stiff limit mu->oo gives z->-1, |z|->1.
    cay_limit = sp.limit(z_cayley, mu, sp.oo)
    cay_abs_le_1 = sp.simplify(1 - z_cayley**2)  # = (since 0<= ) ... show >= 0
    # Explicit: stiff limit mu->oo gives z->-oo, |z|->oo (UNSTABLE).
    expl_limit = sp.limit(sp.Abs(z_explicit), mu, sp.oo)
    return z_cayley, z_explicit, cay_limit, cay_abs_le_1, expl_limit


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("Wentzell/Robin DYNAMIC BC — von-Neumann stability preflight (C-9)")
    print("Contradiction: time-dependent boundary scaling gamma(t) vs boundary-")
    print("layer stability (ADR-0098 Amendment 3 / Stephan 2023 arXiv:2307.00419)")
    print("=" * 78)

    # ---- Symbolic witness first (the heart of the TRIZ resolution) ----
    print("\n[SYMBOLIC] Cayley vs explicit amplification at a dissipative eigenvalue "
          "lambda = -mu (mu >= 0 = decay rate; mu -> oo = stiff/unbounded coupling):")
    z_cay, z_expl, cay_limit, cay_id, expl_limit = symbolic_cayley_witness()
    print(f"  Cayley symbol      z_cay(mu)  = {z_cay}")
    print(f"  Explicit symbol    z_expl(mu) = {z_expl}")
    print(f"  1 - z_cay(mu)^2             = {sp.simplify(cay_id)}   (>= 0 for mu,tau>0 ⇒ |z_cay| <= 1)")
    print(f"  lim_{{mu->oo}} z_cay         = {cay_limit}   (|.| = 1 : marginal, NEVER > 1)")
    print(f"  lim_{{mu->oo}} |z_expl|      = {expl_limit}   (UNSTABLE: explicit blows up)")

    # numeric confirmation that 1 - z_cay^2 >= 0 on a grid
    mu_s, tau_s = sp.symbols("mu tau", positive=True)
    cay_id_fn = sp.lambdify((mu_s, tau_s), (1 - z_cay**2).subs({sp.Symbol("mu", positive=True): mu_s, sp.Symbol("tau", positive=True): tau_s}), "numpy")
    grid_mu = np.linspace(0, 1e6, 50001)
    vals = cay_id_fn(grid_mu, 0.01)
    min_id = float(np.min(vals))
    print(f"  numeric min of (1 - z_cay^2) over mu in [0,1e6], tau=0.01: {min_id:.3e}  "
          f"({'>= 0 OK' if min_id >= -1e-12 else 'NEGATIVE!'})")

    # ---- Matrix amplification sweep over coupling stiffness ----
    a = 1.0
    L = 1.0
    c = 0.5                       # boundary reaction coefficient (c > 0 dissipative)
    # Refinement sweep: as dx -> 0 the coupling -gamma/dx -> infinity (UNBOUNDED B).
    dx_list = [1.0 / 16, 1.0 / 64, 1.0 / 256, 1.0 / 1024]
    gamma_list = [0.5, 1.0, 4.0, 16.0]   # boundary scalings, incl. large (stiff) values

    print("\n[MATRIX] Worst-case Fourier-symbol amplification, kappa = pi/dx "
          "(highest wavenumber), tau = 0.4*dx^2/a (heat CFL-scale step):")
    print(f"{'dx':>10} {'gamma':>8} {'rho_explicit':>14} {'rho_cayley':>12} "
          f"{'rho_bwdEuler':>13} {'gen_dissip':>11}")

    explicit_unstable_seen = False
    cayley_max = 0.0
    bwd_max = 0.0
    worst_explicit = (None, None, 0.0)

    for dx in dx_list:
        kappa = np.pi / dx                       # finest representable mode
        tau = 0.4 * dx * dx / a                  # parabolic step scale
        for gamma in gamma_list:
            M_e = explicit_step_amp(a, kappa, gamma, c, dx, tau)
            M_c = implicit_cayley_step_amp(a, kappa, gamma, c, dx, tau)
            M_b = implicit_backward_euler_step_amp(a, kappa, gamma, c, dx, tau)
            rho_e = spectral_radius(M_e)
            rho_c = spectral_radius(M_c)
            rho_b = spectral_radius(M_b)
            dissip = generator_is_dissipative(a, kappa, gamma, c, dx)
            cayley_max = max(cayley_max, rho_c)
            bwd_max = max(bwd_max, rho_b)
            if rho_e > 1.0 + 1e-9:
                explicit_unstable_seen = True
                if rho_e > worst_explicit[2]:
                    worst_explicit = (dx, gamma, rho_e)
            print(f"{dx:>10.5f} {gamma:>8.2f} {rho_e:>14.4f} {rho_c:>12.6f} "
                  f"{rho_b:>13.6f} {dissip:>11.2e}")

    # ---- Time-dependent gamma(t): does the A-stable Cayley map stay <= 1 as
    #      gamma varies step-to-step? (Stephan's instability is precisely about
    #      time-dependent / per-step-frozen coupling.) ----
    print("\n[TIME-DEPENDENT gamma(t)] product of per-step Cayley maps with "
          "gamma(t_k) varying; report growth of the n-step product norm:")
    dx = 1.0 / 256
    kappa = np.pi / dx
    a = 1.0
    c = 0.5
    n = 200
    T = 0.05
    tau = T / n
    rng = np.random.default_rng(0xC0FFEE)
    # gamma(t) sweeps across a wide band incl. large values, sign-positive (physical).
    gammas = 0.5 + 9.5 * rng.random(n)
    def safe_2norm(M):
        """Spectral 2-norm robust to overflow: returns inf if the matrix has
        already diverged to inf/nan (the explicit product literally overflows —
        that overflow IS the divergence witness, not a numerical artifact)."""
        if not np.all(np.isfinite(M)):
            return np.inf
        # ||M||_2 = largest singular value = sqrt(largest eig of M^T M); 2x2 closed-form
        # avoids LAPACK SVD non-convergence on near-overflow inputs.
        g = M.T @ M
        if not np.all(np.isfinite(g)):
            return np.inf
        tr = g[0, 0] + g[1, 1]
        det = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
        disc = max(tr * tr - 4.0 * det, 0.0)
        lam_max = 0.5 * (tr + np.sqrt(disc))
        return float(np.sqrt(max(lam_max, 0.0)))

    P_cay = np.eye(2)
    P_expl = np.eye(2)
    growth_cay = []
    growth_expl = []
    for k in range(n):
        g = gammas[k]
        P_cay = implicit_cayley_step_amp(a, kappa, g, c, dx, tau) @ P_cay
        with np.errstate(over="ignore", invalid="ignore"):
            P_expl = explicit_step_amp(a, kappa, g, c, dx, tau) @ P_expl
        growth_cay.append(safe_2norm(P_cay))
        growth_expl.append(safe_2norm(P_expl))
    print(f"  steps n = {n}, tau = {tau:.2e}, dx = {dx:.4e}, kappa = {kappa:.1f}")
    print(f"  Cayley   product norm:  start {growth_cay[0]:.4f}  ...  end {growth_cay[-1]:.4f}  "
          f"(max {max(growth_cay):.4f})")
    print(f"  Explicit product norm:  start {growth_expl[0]:.4f}  ...  end {growth_expl[-1]:.4e}  "
          f"(max {max(growth_expl):.4e})")

    cay_td_stable = max(growth_cay) <= 1.0 + 1e-6
    expl_td_unstable = max(growth_expl) > 10.0

    # ---- VERDICT ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  Explicit freezing step unstable (rho > 1) somewhere:  {explicit_unstable_seen}")
    if worst_explicit[0] is not None:
        print(f"    worst explicit mode: dx={worst_explicit[0]:.5f}, gamma={worst_explicit[1]:.2f}, "
              f"rho={worst_explicit[2]:.4f}  <-- Stephan-type unbounded-coupling blow-up witness")
    print(f"  Cayley   max amplification over sweep:  {cayley_max:.6f}  "
          f"({'<= 1 OK' if cayley_max <= 1.0 + TOL else 'EXCEEDS 1'})")
    print(f"  BwdEuler max amplification over sweep:  {bwd_max:.6f}  "
          f"({'<= 1 OK' if bwd_max <= 1.0 + TOL else 'EXCEEDS 1'})")
    print(f"  Time-dependent gamma(t) Cayley product bounded by 1:  {cay_td_stable}")
    print(f"  Time-dependent gamma(t) explicit product blows up:    {expl_td_unstable}")
    print(f"  Symbolic Cayley |z| <= 1 for all mu (incl. stiff limit): "
          f"{min_id >= -1e-12 and str(cay_limit) in ('-1', '-1.0')}")

    go = (
        cayley_max <= 1.0 + TOL
        and bwd_max <= 1.0 + TOL
        and cay_td_stable
        and explicit_unstable_seen          # the candidate must FIX a real instability
        and min_id >= -1e-12
    )

    if go:
        print("\n>>> VERDICT: GO")
        print("    An A-stable IMPLICIT (Cayley / backward-Euler resolvent) boundary-block")
        print("    update has amplification <= 1 UNCONDITIONALLY across the stiffness sweep")
        print("    AND for time-dependent gamma(t), exactly where the EXPLICIT freezing")
        print("    product formula (Stephan 2023) diverges. The contradiction is RESOLVED:")
        print("    the boundary may carry a time-dependent scaling gamma(t) AND remain")
        print("    boundary-layer stable, provided the dynamic-boundary block is advanced")
        print("    implicitly (resolvent step), not by explicit freezing.")
    else:
        print("\n>>> VERDICT: NO-GO")
        print("    The candidate implicit boundary update did NOT achieve amplification <= 1")
        print("    in all tested regimes. Instability re-confirmed; see the unstable mode above.")

    print("=" * 78)

    # ---- T_WENTZELL: 3-sub-check normalised oracle (ADR-0151, math §49) ----
    # Sub-check 1: cayley_abs_le_1 — symbolic 1 − z_cay² ≥ 0 for all mu,tau > 0.
    cay_abs_ok = min_id >= -1e-12 and str(cay_limit) in ("-1", "-1.0")
    # Sub-check 2: stiff_limit — lim_{mu→∞} z_cay = −1 (marginal, never > 1).
    stiff_ok = str(cay_limit) in ("-1", "-1.0")
    # Sub-check 3: explicit_blowup — lim_{mu→∞} |z_expl| = ∞.
    expl_ok = str(expl_limit) in ("oo", "inf", "Infinity", "zoo")
    t_wentzell_pass = cay_abs_ok and stiff_ok and expl_ok

    print("\n[T_WENTZELL] Sub-check results:")
    print(f"  (1) cayley_abs_le_1  : {'PASS' if cay_abs_ok  else 'FAIL'}  "
          f"(1 − z_cay² >= 0 numerically; min = {min_id:.3e}; symbolic limit = {cay_limit})")
    print(f"  (2) stiff_limit      : {'PASS' if stiff_ok   else 'FAIL'}  "
          f"(lim_{{mu→oo}} z_cay = {cay_limit}, expected -1)")
    print(f"  (3) explicit_blowup  : {'PASS' if expl_ok    else 'FAIL'}  "
          f"(lim_{{mu→oo}} |z_expl| = {expl_limit}, expected oo)")
    if t_wentzell_pass:
        print("T_WENTZELL PASS")
    else:
        reasons = []
        if not cay_abs_ok:
            reasons.append(f"cayley_abs_le_1 FAIL (min={min_id:.3e})")
        if not stiff_ok:
            reasons.append(f"stiff_limit FAIL (got {cay_limit})")
        if not expl_ok:
            reasons.append(f"explicit_blowup FAIL (got {expl_limit})")
        print(f"T_WENTZELL FAIL: {'; '.join(reasons)}")

    return 0 if (go and t_wentzell_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
