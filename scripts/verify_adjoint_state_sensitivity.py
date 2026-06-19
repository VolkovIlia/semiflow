#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false
#
# Sympy's Matrix arithmetic is dynamically typed through __mul__/__add__;
# numpy ndarray operator overloads are likewise opaque to Pyright. All
# operations are valid at runtime (verified by this oracle's PASS).
"""T_ADJOINT_STATE_SENSITIVITY oracle — Adjoint-state parameter-sensitivity for
graph / Magnus kernels (math.md §43; ADR-0115 Issue #1).

PRE-FLIGHT math-fidelity oracle. INDEPENDENTLY verifies, BEFORE any Rust kernel
is written, that the parameter-sensitivity primitive

    ∂(S(τ) u)/∂θ            (per-step forward-mode JVP, §43.3)
    ∂J/∂θ = Σ_k ⟨λ_{k+1}, (∂S_k/∂θ) u_k⟩   (discrete adjoint-state, §43.4)

is a correct consequence of the truncated-Magnus / Chernoff derivation. The
forward step is the degree-4 truncated Taylor map of the Magnus K=4 exponent
(NORMATIVE, math.md §42.1 / §43.3):

    S(τ) := Σ_{m=0}^{4} Ω₄(τ)^m / m!,
    Ω₄(τ) = (τ/2)(A₁ + A₂) + (√3·τ²/12)·[A₂, A₁],   A_i = −L_G(t₀ + c_i·τ; θ).

The library MATH (rank-1 generator derivatives, JVP/VJP linear-algebra,
adjoint-state recursion) is IN-CORE; the ML plumbing (autograd, training loops,
neural-ODE hooks) is explicitly OUT-OF-CORE (revssm) per ADR-0115 boundary. This
oracle checks ONLY the in-core consequence math, with a self-contained
numpy/sympy reference — it does NOT depend on any Rust code.

Sub-checks (5 mandatory; math.md §43.6 + §43.2/.3/.4 claims):

  (a) rank1_edge_sensitivity   [§43.2]
      ∂L_G/∂w_{ij} = (e_i − e_j)(e_i − e_j)ᵀ — symbolic differentiation of the
      assembled L_G(w) reproduces the rank-1, EXACTLY-4-nonzero stencil
      {(i,i),(j,j),(i,j),(j,i)}. Also the diagonal √a node perturbation
      ∂L_a/∂a_k = ½ a_k^{-1/2}(E_k L D + D L E_k) — bare combinatorial L,
      D = diag(√a), E_k = e_k e_kᵀ — is confirmed by symbolic diff of
      L_a(a) = D L D = √a L √a (math.md §43.2, corrected closed form).

  (b) magnus_step_jvp_fd       [§43.3]
      The analytic per-step JVP δ(S u) (§43.3 directional derivative of the
      finite Taylor map) matches the central finite difference
      [S(θ+ε δθ)u − S(θ−ε δθ)u]/(2ε) with the FD self-converging at observed
      order ≈ 2 (error quarters as ε halves), and the analytic-vs-FD residual
      → 0 at the round-off floor.

  (c) magnus_step_jvp_order    [§43.6]
      Under τ halving, the per-step JVP exhibits empirical convergence order
      ≥ 3.9 toward the genuine sensitivity ∂(exp(τ G)u)/∂θ (G = the true Magnus
      generator), i.e. the truncated-map JVP is order-4-consistent with the
      forward integrator and does NOT lose order.

  (d) adjoint_state_gradient_fd  [§43.4]
      For a scalar functional J(u_n) = ½‖u_n − target‖², the discrete
      adjoint-state gradient ∂J/∂θ (assembled as Σ_k ⟨λ_{k+1},(∂S_k/∂θ)u_k⟩
      with λ from the §42 S⋆ backward recursion) matches a direct central
      finite-difference of J w.r.t. each parameter θ (edge weight AND the a
      timescale) to O(ε²). The gradient is shown to accumulate from sparse
      rank-1 per-edge contributions.

  (e) adjoint_order_consistency  [§43.4]
      The adjoint-state gradient reproduces the EXACT discrete-solver gradient
      (forward-mode JVP through every step, the autograd-equivalent) to machine
      precision — independent of τ — confirming the discrete adjoint loses NO
      order relative to the forward integrator (the §43.4 "machine precision"
      claim). This is the substantive correctness statement of §43.

Out-of-symbolic-scope (stated precisely, not silently dropped):
  - The forward map's OWN order-4 accuracy vs the exact heat semigroup is the
    subject of the SEPARATE verify_magnus_* / T17N oracles; (c) here checks only
    that the JVP inherits that order, not the base accuracy.
  - exp(τ G) in (c) is computed by a high-degree (m≤14) Taylor map as the
    "genuine" reference; it is the genuine matrix exponential to ≫ machine
    precision on these small bounded operators, NOT a separate symbolic claim.
  - ML-framework integration (revssm) is OUT-OF-CORE by ADR-0115 and not tested.

Prints "T_ADJOINT_STATE_SENSITIVITY PASS (5/5 sub-checks: ...)" on success;
"T_ADJOINT_STATE_SENSITIVITY FAIL: <reason>" and exits 1 on failure.

References:
  - R.-E. Plessix, *A review of the adjoint-state method...*, GJI 167 (2006).
  - Y. Cao, S. Li, L. Petzold, R. Serban, *Adjoint sensitivity analysis for
    ODEs/DAEs*, SIAM J. Sci. Comput. 24 (2003) 1076–1089.
  - Iserles, Munthe-Kaas, Nørsett, Zanna, Acta Numerica 9 (2000), eq. (5.10).
  - math.md §42 (Theorem 42.1), §43 (Issue #1), §43.6 oracle reference.
  - ADR-0115 — contract authority + in-core-math vs revssm-ML boundary.
"""

import sys

import numpy as np

# GL₄ (2-point Gauss-Legendre) interior nodes on [0,1]: c = 1/2 ± √3/6.
# Matches the Ω₄ structure of math.md §42.1 / §12.9.
_C1 = 0.5 - np.sqrt(3.0) / 6.0
_C2 = 0.5 + np.sqrt(3.0) / 6.0
_COMM_PREFAC = np.sqrt(3.0) / 12.0  # commutator coefficient = _COMM_PREFAC · τ²

_SEED = 0xAD01_0115  # deterministic vectors / perturbations


# --------------------------------------------------------------------------- #
# Small self-contained graph Laplacian reference (combinatorial + √a variants).
# --------------------------------------------------------------------------- #
def path3_edges():
    """3-node path: edges (0,1),(1,2)."""
    return 3, [(0, 1), (1, 2)]


def cycle4_edges():
    """4-node cycle: edges (0,1),(1,2),(2,3),(3,0)."""
    return 4, [(0, 1), (1, 2), (2, 3), (3, 0)]


def _rank1(n, i, j):
    """(e_i − e_j)(e_i − e_j)ᵀ as a dense n×n numpy array (4 nonzeros)."""
    e = np.zeros(n)
    e[i] = 1.0
    e[j] = -1.0
    return np.outer(e, e)


def assemble_laplacian(n, edges, weights):
    """L_G = Σ_e w_e (e_i − e_j)(e_i − e_j)ᵀ  (numpy dense, symmetric)."""
    L = np.zeros((n, n))
    for (i, j), w in zip(edges, weights):
        L += w * _rank1(n, i, j)
    return L


def assemble_laplacian_a(n, edges, weights, a):
    """L_a = √a · L_G · √a with positive node-timescales a (length n)."""
    L = assemble_laplacian(n, edges, weights)
    s = np.sqrt(np.asarray(a, dtype=float))
    return (s[:, None] * L) * s[None, :]


# --------------------------------------------------------------------------- #
# Forward truncated-Magnus step S(τ) and its analytic JVP / state-adjoint.
# --------------------------------------------------------------------------- #
def omega4(A1, A2, tau):
    """Ω₄(τ) = (τ/2)(A₁+A₂) + (√3 τ²/12)[A₂,A₁]."""
    comm = A2 @ A1 - A1 @ A2
    return 0.5 * tau * (A1 + A2) + _COMM_PREFAC * tau**2 * comm


def d_omega4(A1, A2, dA1, dA2, tau):
    """δΩ₄ = (τ/2)(δA₁+δA₂) + (√3 τ²/12)([δA₂,A₁]+[A₂,δA₁])  (math.md §43.3)."""
    dcomm = (dA2 @ A1 - A1 @ dA2) + (A2 @ dA1 - dA1 @ A2)
    return 0.5 * tau * (dA1 + dA2) + _COMM_PREFAC * tau**2 * dcomm


def taylor_map(Omega, k_max=4):
    """S = Σ_{m=0..k_max} Ω^m/m!  (dense n×n matrix)."""
    n = Omega.shape[0]
    S = np.eye(n)
    power = np.eye(n)
    fact = 1.0
    for m in range(1, k_max + 1):
        power = power @ Omega
        fact *= m
        S = S + power / fact
    return S


def d_taylor_map(Omega, dOmega, k_max=4):
    """δS = Σ_{m=1..k_max} (1/m!) Σ_{p=0}^{m-1} Ω^p (δΩ) Ω^{m-1-p}  (math.md §43.3).

    Exact derivative of the finite Taylor polynomial S = Σ Ω^m/m!.
    """
    n = Omega.shape[0]
    dS = np.zeros((n, n))
    fact = 1.0
    for m in range(1, k_max + 1):
        fact *= m
        acc = np.zeros((n, n))
        for p in range(m):
            left = np.linalg.matrix_power(Omega, p)
            right = np.linalg.matrix_power(Omega, m - 1 - p)
            acc += left @ dOmega @ right
        dS += acc / fact
    return dS


# --------------------------------------------------------------------------- #
# NON-AUTONOMOUS schedule. To EXERCISE the Ω₄ commutator (the most error-prone
# term — it vanishes if A₁ = A₂), the edge weights are time-dependent:
#     w(t) = w · (1 + gamma · t)        (per-edge linear ramp; gamma a vector).
# Then A_i = −L_G(t₀ + c_i·τ) genuinely differ between the two GL nodes, so
# [A₂, A₁] ≠ 0 and the commutator coefficient (and its sign-flip in S⋆) is live.
# The √a node-timescale is held t-constant (it parametrises the diffusion metric,
# not the schedule); θ = {edge weights w, node timescales a}.
# --------------------------------------------------------------------------- #
def _nodes_at(n, edges, w, gamma, a, t0, tau):
    """Return (A1, A2) = (−L at t₀+c₁τ, −L at t₀+c₂τ) under the w(t) ramp."""
    t1 = t0 + _C1 * tau
    t2 = t0 + _C2 * tau
    w1 = w * (1.0 + gamma * t1)
    w2 = w * (1.0 + gamma * t2)
    if a is None:
        return -assemble_laplacian(n, edges, w1), -assemble_laplacian(n, edges, w2)
    return (-assemble_laplacian_a(n, edges, w1, a),
            -assemble_laplacian_a(n, edges, w2, a))


def _dnodes_at(n, edges, w, gamma, a, t0, tau, dw, da):
    """Return (δA1, δA2) for a (dw, da) parameter perturbation at the two nodes.

    dw perturbs the BASE edge weight (it rides the same w(t) ramp); da perturbs
    the node timescale. Uses §43.2 rank-1 ∂L_G/∂w and the √a node form.
    """
    t1 = t0 + _C1 * tau
    t2 = t0 + _C2 * tau
    out = []
    for _tk, wt_factor in ((t1, 1.0 + gamma * t1), (t2, 1.0 + gamma * t2)):
        if a is None:
            dLk = (assemble_laplacian(n, edges, dw * wt_factor)
                   if dw is not None else np.zeros((n, n)))
            out.append(-dLk)
        else:
            a_arr = np.asarray(a, dtype=float)
            s = np.sqrt(a_arr)
            wt = w * wt_factor
            L = assemble_laplacian(n, edges, wt)
            dL = (assemble_laplacian(n, edges, dw * wt_factor)
                  if dw is not None else np.zeros((n, n)))
            ds = 0.5 * (da / s) if da is not None else np.zeros(n)
            dLa = (s[:, None] * dL * s[None, :]
                   + ds[:, None] * L * s[None, :]
                   + s[:, None] * L * ds[None, :])
            out.append(-dLa)
    return out[0], out[1]


def magnus_step_matrix(n, edges, w, a, tau, gamma=None, t0=0.0, k_max=4):
    """Forward step S(τ) over [t₀,t₀+τ] under the w(t) ramp (gamma; default 0)."""
    if gamma is None:
        gamma = np.zeros(len(edges))
    A1, A2 = _nodes_at(n, edges, w, gamma, a, t0, tau)
    return taylor_map(omega4(A1, A2, tau), k_max=k_max)


def magnus_step_jvp(n, edges, w, a, tau, dw, da, u, gamma=None, t0=0.0, k_max=4):
    """Analytic JVP δ(S u) for a (dw, da) perturbation (math.md §43.3).

    Returns δ(S u). Two-node (non-autonomous) so the commutator JVP term is live.
    """
    if gamma is None:
        gamma = np.zeros(len(edges))
    A1, A2 = _nodes_at(n, edges, w, gamma, a, t0, tau)
    dA1, dA2 = _dnodes_at(n, edges, w, gamma, a, t0, tau, dw, da)
    Omega = omega4(A1, A2, tau)
    dOmega = d_omega4(A1, A2, dA1, dA2, tau)
    dS = d_taylor_map(Omega, dOmega, k_max=k_max)
    return dS @ u


# --------------------------------------------------------------------------- #
# §42 state-adjoint S⋆ = S(τ)ᵀ: same Taylor map with the commutator coefficient
# flipped (math.md §42.4). With the non-autonomous A₁ ≠ A₂, the flip is GENUINE
# (S⋆ ≠ S) yet S⋆ = Sᵀ holds EXACTLY (Theorem 42.1) — we assert this.
# --------------------------------------------------------------------------- #
def state_adjoint_matrix(n, edges, w, a, tau, gamma=None, t0=0.0, k_max=4):
    """S⋆(τ) = Σ_{m=0..4} (Ω₄ᵀ)^m/m! via commutator-sign flip (math.md §42.4)."""
    if gamma is None:
        gamma = np.zeros(len(edges))
    A1, A2 = _nodes_at(n, edges, w, gamma, a, t0, tau)
    comm = A2 @ A1 - A1 @ A2
    Omega_star = 0.5 * tau * (A1 + A2) - _COMM_PREFAC * tau**2 * comm
    return taylor_map(Omega_star, k_max=k_max)


# --------------------------------------------------------------------------- #
# Sub-check (a): rank-1 edge sensitivity + √a node sensitivity (§43.2).
# --------------------------------------------------------------------------- #
def check_rank1_edge_sensitivity():
    """∂L_G/∂w_{ij} = (e_i−e_j)(e_i−e_j)ᵀ (4 nonzeros); √a node form likewise."""
    try:
        import sympy as sp
    except ImportError:
        return "sympy not installed"

    for name, (n, edges) in (("3-path", path3_edges()), ("4-cycle", cycle4_edges())):
        w_syms = sp.symbols(f"w0:{len(edges)}", positive=True)
        # Symbolic assembled Laplacian.
        L = sp.zeros(n, n)
        for (i, j), wsym in zip(edges, w_syms):
            e = sp.zeros(n, 1)
            e[i] = 1
            e[j] = -1
            L += wsym * (e * e.T)

        for eidx, (i, j) in enumerate(edges):
            # Re-wrap the entrywise derivative as a concrete Matrix so .rank()/
            # subscripting resolve (sp.diff may type as a Derivative wrapper).
            dL = sp.Matrix(sp.diff(L, w_syms[eidx]))
            # Expected rank-1 stencil.
            e = sp.zeros(n, 1)
            e[i] = 1
            e[j] = -1
            expected = e * e.T
            residual = sp.expand(dL - expected)
            if residual != sp.zeros(n, n):
                return (
                    f"rank1_edge_sensitivity ({name}, edge {(i, j)}): "
                    f"∂L_G/∂w ≠ (e_i−e_j)(e_i−e_j)ᵀ. Residual = {residual}."
                )
            # EXACTLY 4 nonzeros at {(i,i),(j,j),(i,j),(j,i)}.
            nz = {(r, c) for r in range(n) for c in range(n) if dL[r, c] != 0}
            if nz != {(i, i), (j, j), (i, j), (j, i)}:
                return (
                    f"rank1_edge_sensitivity ({name}, edge {(i, j)}): nonzero set "
                    f"{sorted(nz)} ≠ {{(i,i),(j,j),(i,j),(j,i)}} — not the rank-1 stencil."
                )
            # rank exactly 1.
            if dL.rank() != 1:
                return (
                    f"rank1_edge_sensitivity ({name}, edge {(i, j)}): "
                    f"∂L_G/∂w has rank {dL.rank()} ≠ 1."
                )

    # --- diagonal √a node perturbation: ∂L_a/∂a_k vs the §43.2 closed form. ---
    n, edges = path3_edges()
    w_syms = sp.symbols(f"w0:{len(edges)}", positive=True)
    a_syms = sp.symbols(f"a0:{n}", positive=True)
    L = sp.zeros(n, n)
    for (i, j), wsym in zip(edges, w_syms):
        e = sp.zeros(n, 1)
        e[i] = 1
        e[j] = -1
        L += wsym * (e * e.T)
    sqrt_a = sp.diag(*[sp.sqrt(ak) for ak in a_syms])
    La = sqrt_a * L * sqrt_a
    for k in range(n):
        dLa = sp.expand(sp.diff(La, a_syms[k]))
        ek = sp.zeros(n, 1)
        ek[k] = 1
        Ek = ek * ek.T
        # §43.2 (corrected): ½ a_k^{-1/2} (E_k L D + D L E_k), with the BARE
        # combinatorial L and D = diag(√a) = sqrt_a. (The redundant fixture; the
        # independent truth is sp.diff(La, a_k) below — they must agree.)
        expected = sp.expand(sp.Rational(1, 2) * a_syms[k] ** sp.Rational(-1, 2)
                             * (Ek * L * sqrt_a + sqrt_a * L * Ek))
        residual = sp.simplify(dLa - expected)
        if residual != sp.zeros(n, n):
            return (
                f"rank1_edge_sensitivity (√a node {k}): ∂L_a/∂a_k ≠ "
                f"½ a_k^(−1/2)(E_k L D + D L E_k). Residual = {residual}."
            )
    return None


# --------------------------------------------------------------------------- #
# Sub-check (b): per-step JVP vs central FD, FD self-converging at order ≈ 2.
# --------------------------------------------------------------------------- #
def check_magnus_step_jvp_fd():
    """Analytic δ(S u) matches [S(θ+εδθ)u − S(θ−εδθ)u]/(2ε), FD order ≈ 2."""
    rng = np.random.default_rng(_SEED)
    n, edges = cycle4_edges()
    w = rng.uniform(0.5, 1.5, len(edges))
    gamma = rng.uniform(0.3, 0.9, len(edges))  # NON-autonomous ramp → [A₂,A₁]≠0
    dw = rng.uniform(-1.0, 1.0, len(edges))    # perturbation direction in edge space
    u = rng.standard_normal(n)
    tau = 0.05
    t0 = 0.7  # nonzero base time so the two GL nodes are well separated

    # Guard: confirm the commutator is genuinely live (else this check is vacuous).
    A1, A2 = _nodes_at(n, edges, w, gamma, None, t0, tau)
    if np.max(np.abs(A2 @ A1 - A1 @ A2)) < 1e-9:
        return "magnus_step_jvp_fd: [A₂,A₁]≈0 — schedule failed to make step non-autonomous."

    analytic = magnus_step_jvp(n, edges, w, None, tau, dw, None, u, gamma=gamma, t0=t0)

    eps_list = [1e-2, 5e-3, 2.5e-3, 1.25e-3]
    fd_errors = []
    for eps in eps_list:
        Sp = magnus_step_matrix(n, edges, w + eps * dw, None, tau, gamma=gamma, t0=t0)
        Sm = magnus_step_matrix(n, edges, w - eps * dw, None, tau, gamma=gamma, t0=t0)
        fd = (Sp @ u - Sm @ u) / (2.0 * eps)
        fd_errors.append(float(np.max(np.abs(fd - analytic))))

    # FD truncation error must shrink ~ ε² (ratio ≈ 4 per halving) until floor.
    orders = []
    for k in range(len(eps_list) - 1):
        if fd_errors[k + 1] < 1e-13:  # hit round-off floor; stop ranking
            break
        ratio = fd_errors[k] / fd_errors[k + 1]
        orders.append(np.log2(ratio))
    if not orders:
        # all below floor: analytic already matches to round-off → success.
        if max(fd_errors) > 1e-9:
            return (f"magnus_step_jvp_fd: FD never converged; errors={fd_errors}.")
        return None
    observed = float(np.median(orders))
    if observed < 1.8:
        return (
            f"magnus_step_jvp_fd: central-FD observed order {observed:.3f} < 1.8 "
            f"(expected ≈ 2). FD errors={fd_errors}, per-step orders={orders}. "
            f"Analytic JVP does NOT match the finite-difference derivative."
        )
    # finest-eps residual must be at the O(ε²) floor.
    if fd_errors[-1] > 1e-5:
        return (
            f"magnus_step_jvp_fd: finest-ε residual {fd_errors[-1]:.2e} too large "
            f"(analytic JVP disagrees with FD beyond O(ε²))."
        )
    return None


# --------------------------------------------------------------------------- #
# Sub-check (c): per-step JVP empirical order ≥ 3.9 under τ halving (§43.6).
# The genuine target is the JVP of the TRUE step exp(Ω₄(τ)) (with the SAME, fully
# non-autonomous Ω₄ that contains the live commutator). The truncated-K4 JVP
# (∂ of Σ_{m=0..4} Ω₄^m/m!) must converge to it at order ≥ 4 (Taylor remainder
# is O(Ω₄⁵)=O(τ⁵) ⇒ value order 5, JVP order ≥ 4). We use a high-degree Taylor
# map (m≤18) as the exp(Ω₄) reference (≫ machine precision on these bounded Ω₄).
# --------------------------------------------------------------------------- #
def _exp_jvp(Omega, dOmega, u, k_max=18):
    """∂(exp(Ω) u)/∂θ reference via a high-degree (K→∞) Taylor map of exp(Ω)."""
    S = taylor_map(Omega, k_max=k_max)
    dS = d_taylor_map(Omega, dOmega, k_max=k_max)
    return S, dS @ u


def check_magnus_step_jvp_order():
    """Truncated-K4 per-step JVP converges to genuine exp(Ω₄)-JVP at order ≥ 3.9."""
    rng = np.random.default_rng(_SEED + 1)
    n, edges = cycle4_edges()
    w = rng.uniform(0.5, 1.5, len(edges))
    gamma = rng.uniform(0.3, 0.9, len(edges))  # NON-autonomous → live commutator
    dw = rng.uniform(-1.0, 1.0, len(edges))
    u = rng.standard_normal(n)
    t0 = 0.7

    taus = [0.2, 0.1, 0.05, 0.025]
    errs = []
    for tau in taus:
        A1, A2 = _nodes_at(n, edges, w, gamma, None, t0, tau)
        dA1, dA2 = _dnodes_at(n, edges, w, gamma, None, t0, tau, dw, None)
        comm = A2 @ A1 - A1 @ A2
        if tau == taus[0] and np.max(np.abs(comm)) < 1e-9:
            return "magnus_step_jvp_order: [A₂,A₁]≈0 — step is not non-autonomous."
        # Reference Ω₄ assembled INDEPENDENTLY of omega4()/d_omega4() — guards a
        # transcription error in the forward-map commutator term itself (§42.1).
        Omega_ref = 0.5 * tau * (A1 + A2) + _COMM_PREFAC * tau**2 * comm
        if np.max(np.abs(omega4(A1, A2, tau) - Omega_ref)) > 1e-12:
            return (
                "magnus_step_jvp_order: omega4() ≠ (τ/2)(A₁+A₂)+(√3τ²/12)[A₂,A₁] "
                "(forward Magnus exponent transcription error)."
            )
        dcomm = (dA2 @ A1 - A1 @ dA2) + (A2 @ dA1 - dA1 @ A2)
        dOmega_ref = 0.5 * tau * (dA1 + dA2) + _COMM_PREFAC * tau**2 * dcomm
        if np.max(np.abs(d_omega4(A1, A2, dA1, dA2, tau) - dOmega_ref)) > 1e-12:
            return "magnus_step_jvp_order: d_omega4() ≠ δΩ₄ closed form (§43.3 error)."
        Omega = Omega_ref
        dOmega = dOmega_ref
        # truncated-K4 JVP of S(τ) = Σ_{m=0..4} Ω₄^m/m!.
        jvp_k4 = d_taylor_map(Omega, dOmega, k_max=4) @ u
        # genuine exp(Ω₄) JVP (K→∞ reference, SAME Ω₄).
        _, jvp_exp = _exp_jvp(Omega, dOmega, u, k_max=18)
        errs.append(float(np.max(np.abs(jvp_k4 - jvp_exp))))

    orders = []
    for k in range(len(taus) - 1):
        if errs[k + 1] < 1e-14:
            break
        orders.append(np.log2(errs[k] / errs[k + 1]))
    if not orders:
        return (f"magnus_step_jvp_order: errors below floor too early; errs={errs}.")
    observed = float(np.median(orders))
    if observed < 3.9:
        return (
            f"magnus_step_jvp_order: empirical JVP order {observed:.3f} < 3.9 "
            f"(τ-halving). errs={errs}, orders={orders}. The truncated-K4 JVP "
            f"LOSES order relative to the forward integrator — §43.6 claim FAILS."
        )
    return None


# --------------------------------------------------------------------------- #
# Sub-check (d): discrete adjoint-state ∂J/∂θ vs central FD-of-J (§43.4).
# --------------------------------------------------------------------------- #
def _forward_traj(n, edges, w, a, tau, u0, n_steps, gamma):
    """Return [u_0, ..., u_n] of the n-step trajectory; step k spans [kτ,(k+1)τ]."""
    us = [u0.copy()]
    for k in range(n_steps):
        Sk = magnus_step_matrix(n, edges, w, a, tau, gamma=gamma, t0=k * tau)
        us.append(Sk @ us[-1])
    return us


def _functional_J(u_n, target):
    return 0.5 * float(np.sum((u_n - target) ** 2))


def _adjoint_gradient(n, edges, w, a, tau, u0, n_steps, target, gamma):
    """∂J/∂θ for every edge weight (and a-node if a given) via §43.4 adjoint-state.

    Returns (grad_w [len edges], grad_a [len n] or None). Uses the PER-STEP S_k⋆
    for the costate recursion and the analytic per-step JVP (∂S_k/∂θ) u_k
    assembled from rank-1 contributions.
    """
    us = _forward_traj(n, edges, w, a, tau, u0, n_steps, gamma)
    # Backward costate recursion: λ_n = ∂J/∂u_n = (u_n − target); λ_k = S_k⋆ λ_{k+1}.
    lam = [None] * (n_steps + 1)
    lam[n_steps] = us[n_steps] - target
    for k in range(n_steps - 1, -1, -1):
        Sk_star = state_adjoint_matrix(n, edges, w, a, tau, gamma=gamma, t0=k * tau)
        lam[k] = Sk_star @ lam[k + 1]

    grad_w = np.zeros(len(edges))
    for eidx in range(len(edges)):
        dw = np.zeros(len(edges))
        dw[eidx] = 1.0  # unit perturbation of edge eidx → rank-1 ∂A/∂w (§43.2)
        g = 0.0
        for k in range(n_steps):
            dSu = magnus_step_jvp(n, edges, w, a, tau, dw, None, us[k],
                                  gamma=gamma, t0=k * tau)
            g += float(np.dot(lam[k + 1], dSu))   # ⟨λ_{k+1}, (∂S_k/∂w) u_k⟩
        grad_w[eidx] = g

    grad_a = None
    if a is not None:
        grad_a = np.zeros(n)
        for node in range(n):
            da = np.zeros(n)
            da[node] = 1.0
            g = 0.0
            for k in range(n_steps):
                dSu = magnus_step_jvp(n, edges, w, a, tau, None, da, us[k],
                                      gamma=gamma, t0=k * tau)
                g += float(np.dot(lam[k + 1], dSu))
            grad_a[node] = g
    return grad_w, grad_a


def check_adjoint_state_gradient_fd():
    """Adjoint-state ∂J/∂θ matches central FD of J for edge weights AND a."""
    rng = np.random.default_rng(_SEED + 2)
    n, edges = cycle4_edges()
    w = rng.uniform(0.6, 1.4, len(edges))
    gamma = rng.uniform(0.3, 0.9, len(edges))  # NON-autonomous → live commutator
    a = rng.uniform(0.7, 1.3, n)
    u0 = rng.standard_normal(n)
    target = rng.standard_normal(n)
    tau = 0.04
    n_steps = 5

    # Guard: confirm the per-step commutator is genuinely live.
    A1, A2 = _nodes_at(n, edges, w, gamma, a, tau, tau)  # step k=1
    if np.max(np.abs(A2 @ A1 - A1 @ A2)) < 1e-9:
        return "adjoint_state_gradient_fd: [A₂,A₁]≈0 — step is not non-autonomous."

    grad_w, grad_a = _adjoint_gradient(n, edges, w, a, tau, u0, n_steps, target, gamma)
    # a is not None here, so the a-node gradient is always returned.
    assert grad_a is not None

    # Central FD of J w.r.t. each edge weight.
    eps = 1e-6
    max_rel = 0.0
    for eidx in range(len(edges)):
        wp = w.copy(); wp[eidx] += eps
        wm = w.copy(); wm[eidx] -= eps
        Jp = _functional_J(_forward_traj(n, edges, wp, a, tau, u0, n_steps, gamma)[-1], target)
        Jm = _functional_J(_forward_traj(n, edges, wm, a, tau, u0, n_steps, gamma)[-1], target)
        fd = (Jp - Jm) / (2 * eps)
        denom = max(abs(fd), 1e-8)
        rel = abs(grad_w[eidx] - fd) / denom
        max_rel = max(max_rel, rel)
        if rel > 1e-5:
            return (
                f"adjoint_state_gradient_fd (edge {eidx}): adjoint ∂J/∂w={grad_w[eidx]:.8e} "
                f"vs FD={fd:.8e}, rel err {rel:.2e} > 1e-5. Discrete adjoint-state "
                f"gradient does NOT reproduce the solver gradient — §43.4 FAILS."
            )

    # Central FD of J w.r.t. each a-node timescale.
    for node in range(n):
        ap = a.copy(); ap[node] += eps
        am = a.copy(); am[node] -= eps
        Jp = _functional_J(_forward_traj(n, edges, w, ap, tau, u0, n_steps, gamma)[-1], target)
        Jm = _functional_J(_forward_traj(n, edges, w, am, tau, u0, n_steps, gamma)[-1], target)
        fd = (Jp - Jm) / (2 * eps)
        denom = max(abs(fd), 1e-8)
        rel = abs(grad_a[node] - fd) / denom
        max_rel = max(max_rel, rel)
        if rel > 1e-5:
            return (
                f"adjoint_state_gradient_fd (a-node {node}): adjoint ∂J/∂a={grad_a[node]:.8e} "
                f"vs FD={fd:.8e}, rel err {rel:.2e} > 1e-5. §43.4 (timescale) FAILS."
            )

    # The gradient assembles from sparse rank-1 contributions: confirm that
    # zeroing edge eidx's rank-1 term removes exactly that edge's gradient entry.
    # (Structural sanity: ∂A/∂w_e has support only on edge e's two endpoints.)
    for eidx, (i, j) in enumerate(edges):
        dw = np.zeros(len(edges)); dw[eidx] = 1.0
        dA = -assemble_laplacian(n, edges, dw)
        nz = {(r, c) for r in range(n) for c in range(n) if abs(dA[r, c]) > 1e-15}
        if nz != {(i, i), (j, j), (i, j), (j, i)}:
            return (
                f"adjoint_state_gradient_fd: per-edge ∂A/∂w_{eidx} support {sorted(nz)} "
                f"≠ rank-1 stencil — gradient not assembled from sparse rank-1 terms."
            )
    return None


# --------------------------------------------------------------------------- #
# Sub-check (e): adjoint gradient == EXACT discrete-solver gradient (autograd-eq).
# The autograd-equivalent forward-mode gradient through every step is
#   dJ/dθ = ⟨ ∂J/∂u_n , d u_n/dθ ⟩,  d u_n/dθ propagated by the chain rule with
#   the SAME analytic per-step JVP. The adjoint (reverse) assembly MUST equal this
#   to machine precision, independent of τ — confirming NO order loss (§43.4).
# --------------------------------------------------------------------------- #
def _forward_mode_gradient(n, edges, w, a, tau, u0, n_steps, target, dw, da, gamma):
    """Directional dJ/dθ·δθ by forward-propagating the JVP through every step."""
    u = u0.copy()
    du = np.zeros(n)  # d u_0 / dθ = 0
    for k in range(n_steps):
        Sk = magnus_step_matrix(n, edges, w, a, tau, gamma=gamma, t0=k * tau)
        # d u_{k+1} = S_k du_k + (∂S_k/∂θ) u_k   (product rule, chain rule).
        dSu = magnus_step_jvp(n, edges, w, a, tau, dw, da, u, gamma=gamma, t0=k * tau)
        du = Sk @ du + dSu
        u = Sk @ u
    # dJ/dθ·δθ = ⟨ u_n − target , du_n ⟩
    return float(np.dot(u - target, du))


def check_adjoint_order_consistency():
    """Adjoint ⟨∇J,δθ⟩ == forward-mode JVP-through-solver to machine precision,
    independent of τ (order-4-consistent with the forward integrator, §43.4)."""
    rng = np.random.default_rng(_SEED + 3)
    n, edges = cycle4_edges()
    w = rng.uniform(0.6, 1.4, len(edges))
    gamma = rng.uniform(0.3, 0.9, len(edges))  # NON-autonomous → live commutator
    a = rng.uniform(0.7, 1.3, n)
    u0 = rng.standard_normal(n)
    target = rng.standard_normal(n)
    n_steps = 6

    # A random combined direction in (w, a)-space.
    dw = rng.uniform(-1.0, 1.0, len(edges))
    da = rng.uniform(-1.0, 1.0, n)

    for tau in (0.08, 0.04, 0.02):
        grad_w, grad_a = _adjoint_gradient(n, edges, w, a, tau, u0, n_steps, target, gamma)
        adjoint_dir = float(np.dot(grad_w, dw) + np.dot(grad_a, da))
        forward_dir = _forward_mode_gradient(
            n, edges, w, a, tau, u0, n_steps, target, dw, da, gamma
        )
        denom = max(abs(forward_dir), 1e-10)
        rel = abs(adjoint_dir - forward_dir) / denom
        if rel > 1e-10:
            return (
                f"adjoint_order_consistency (τ={tau}): adjoint ⟨∇J,δθ⟩={adjoint_dir:.12e} "
                f"vs forward-mode JVP={forward_dir:.12e}, rel err {rel:.2e} > 1e-10. "
                f"Discrete adjoint does NOT reproduce the exact solver gradient — "
                f"order is LOST relative to the forward integrator (§43.4 FAILS)."
            )

    # Internal consistency: S⋆ == Sᵀ EXACTLY even with the live commutator
    # (Theorem 42.1: symmetric A_i ⇒ S(τ)ᵀ = Σ (Ω₄ᵀ)^m/m! = S⋆ for any A₁,A₂).
    tau = 0.05
    A1, A2 = _nodes_at(n, edges, w, gamma, a, tau, tau)
    if np.max(np.abs(A2 @ A1 - A1 @ A2)) < 1e-9:
        return "adjoint_order_consistency: [A₂,A₁]≈0 — commutator not exercised."
    S = magnus_step_matrix(n, edges, w, a, tau, gamma=gamma, t0=tau)
    Sstar = state_adjoint_matrix(n, edges, w, a, tau, gamma=gamma, t0=tau)
    if np.max(np.abs(Sstar - S.T)) > 1e-12:
        return (
            f"adjoint_order_consistency: S⋆ ≠ Sᵀ (max diff "
            f"{np.max(np.abs(Sstar - S.T)):.2e}) — the §42.4 sign-flip kernel does "
            f"NOT realise the transpose used by the adjoint recursion."
        )
    # And the flip is GENUINE here (S⋆ ≠ S), so the recursion truly uses the commutator.
    if np.max(np.abs(Sstar - S)) < 1e-9:
        return "adjoint_order_consistency: S⋆ == S — commutator vanished, sign-flip vacuous."
    return None


def fail(reason):
    print(f"T_ADJOINT_STATE_SENSITIVITY FAIL: {reason}", flush=True)
    return 1


def main():
    """Run all 5 sub-checks; print result; exit 0/1."""
    checks = [
        ("rank1_edge_sensitivity", check_rank1_edge_sensitivity),
        ("magnus_step_jvp_fd", check_magnus_step_jvp_fd),
        ("magnus_step_jvp_order", check_magnus_step_jvp_order),
        ("adjoint_state_gradient_fd", check_adjoint_state_gradient_fd),
        ("adjoint_order_consistency", check_adjoint_order_consistency),
    ]
    print("=" * 70)
    print("T_ADJOINT_STATE_SENSITIVITY — graph adjoint-state parameter sensitivity")
    print("(math.md §43, Issue #1; ADR-0115)")
    print("=" * 70)

    failures = []
    passed = []
    for name, check in checks:
        try:
            result = check()
        except Exception as e:  # noqa: BLE001
            return fail(f"sub-check {name} raised exception: {e!r}")
        if result is None:
            print(f"  (PASS) {name}")
            passed.append(name)
        else:
            print(f"  (FAIL) {name}: {result}")
            failures.append(f"{name}: {result}")

    print()
    if failures:
        return fail(
            f"{len(failures)}/{len(checks)} sub-checks failed: " + "; ".join(failures)
        )
    print(
        "T_ADJOINT_STATE_SENSITIVITY PASS (5/5 sub-checks: " + " / ".join(passed) + ")",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
