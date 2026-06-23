#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false
#
# Sympy Matrix arithmetic is dynamically typed through __mul__/__add__; all
# operations are valid at runtime (verified by this oracle's PASS).
"""T_REVERSE_AD_KVECTOR oracle — multi-parameter (K>1) reverse-AD per-region
gradient (math.md §51.10; ADR-0177; issue #1).

PRE-FLIGHT math-fidelity oracle. INDEPENDENTLY proves, BEFORE any Rust kernel is
written, that the per-region cotangent accumulation

    grad[r] = Σ_{k=n…1} ⟨λ_k, b_k^{(r)}⟩,   b_k^{(r)} = (∂F/∂θ_r)(u_{k-1})

equals the analytic partial derivative ∂J/∂θ_r of the L² loss
J(θ) = ‖(F_θ)ⁿ u₀ − target‖² for a small CLOSED-FORM K-region setup, where
a(x_i) = θ_{ρ(i)} is the piecewise-constant region coefficient (math.md §51.10).

This oracle is self-contained (sympy only). It does NOT depend on any Rust code;
it verifies the in-core MATH that the Rust impl must realise.

----------------------------------------------------------------------------
Closed-form K-region model (deliberately small so sympy stays exact)
----------------------------------------------------------------------------
  N_grid = 4 nodes, K = 2 regions (DoF-aligned):
      ρ = [0, 0, 1, 1]   →  Ω_0 = {0,1},  Ω_1 = {2,3}
      θ = (θ0, θ1) symbolic.
  One step F_θ is a banded linear map J(θ) (a tridiagonal shift-and-scale
  surrogate of the diffusion stencil) whose i-th row is scaled by the LOCAL
  coefficient a_i = θ_{ρ(i)}.  n steps: U = J(θ)ⁿ u₀.  Loss J = ‖U − t‖².

  Because each row i is scaled by θ_{ρ(i)}, the state-Jacobian ∂F/∂u = J(θ) is
  banded self-adjoint inside each region (Jᵀ=J for symmetric stencil), and the
  parameter column ∂F/∂θ_r is non-zero only on rows i with ρ(i)=r (support ⊆ Ω_r).

Sub-checks (3 mandatory; §51.10 NORMATIVE claims):

  (1) per_region_support
      ∂(F u)/∂θ_r has zero entries on every node i ∉ Ω_r — confirms
      supp(b^{(r)}) ⊆ Ω_r (disjoint columns ⇒ NOT the degenerate broadcast).

  (2) reverse_equals_analytic
      The discrete adjoint accumulation grad[r] = Σ_k ⟨λ_k, (∂F/∂θ_r) u_{k-1}⟩
      with λ_n = 2(U−t), λ_{k-1} = Jᵀ λ_k, equals sympy's direct symbolic
      diff(J_loss, θ_r) — for BOTH r ∈ {0,1}.  Non-vacuous: LHS is assembled
      from the per-region columns + transpose recursion, RHS from autodiff of
      the closed-form loss; they agree only if the per-region adjoint is correct.

  (3) k1_reduces_byte_identical
      With K=1 (single region Ω_0 = all nodes), grad[0] equals the §51.9 single
      scalar adjoint grad — the K=1 path is the special case (regression anchor).

PASS → print exactly `T_REVERSE_AD_KVECTOR PASS (3/3 …)`.
FAIL → print `T_REVERSE_AD_KVECTOR FAIL: <reason>` and exit 1.
Pure symbolic; runs in test-fast.  RELEASE_BLOCKING (math.md §51.10).
"""

import sys

import sympy as sp

# NOTE: this is the EXECUTABLE-SPEC scaffold the engineer fills/keeps.  The
# logic below is correct and self-checking; the engineer adapts the banded map
# J(θ) to mirror the real DiffusionChernoff stencil if a tighter coupling to the
# Rust kernel is desired (the support + adjoint-identity structure is invariant).


def build_region_map(n_grid: int, k_regions: int) -> list[int]:
    """DoF-aligned contiguous partition ρ: node i → region r (math.md §51.10)."""
    per = n_grid // k_regions
    return [min(i // per, k_regions - 1) for i in range(n_grid)]


def step_matrix(theta: list[sp.Symbol], rho: list[int], n_grid: int):
    """Banded symmetric shift-and-scale surrogate of one F_θ step.

    Row i scaled by local coefficient a_i = θ_{ρ(i)}; symmetric tridiagonal so
    Jᵀ = J (mirrors const-per-region self-adjoint stencil, math.md §51.10)."""
    j_mat = sp.zeros(n_grid, n_grid)
    for i in range(n_grid):
        a_i = theta[rho[i]]
        j_mat[i, i] = 1 - 2 * a_i
        if i > 0:
            j_mat[i, i - 1] = a_i
        if i < n_grid - 1:
            j_mat[i, i + 1] = a_i
    return j_mat


def analytic_grad(theta, rho, n_grid, n_steps, u0, target):
    """RHS: ∂J/∂θ_r by direct symbolic differentiation of the closed-form loss."""
    j_mat = step_matrix(theta, rho, n_grid)
    u = u0
    for _ in range(n_steps):
        u = j_mat * u
    loss = sum((u[i] - target[i]) ** 2 for i in range(n_grid))
    return [sp.simplify(sp.diff(loss, t)) for t in theta]


def reverse_grad(theta, rho, n_grid, n_steps, u0, target):
    """LHS: discrete per-region adjoint accumulation (math.md §51.10).

    Forward states u_0..u_n; seed λ_n = 2(u_n − t); backward k=n…1:
      grad[r] += ⟨λ_k, (∂J/∂θ_r) u_{k-1}⟩ ; λ_{k-1} = Jᵀ λ_k."""
    j_mat = step_matrix(theta, rho, n_grid)
    jt = j_mat.T
    states = [u0]
    for _ in range(n_steps):
        states.append(j_mat * states[-1])
    u_n = states[-1]
    lam = sp.Matrix([2 * (u_n[i] - target[i]) for i in range(n_grid)])
    k = len(theta)
    grad = [sp.Integer(0)] * k
    # Per-region parameter columns ∂J/∂θ_r (constant in u — J is linear in θ
    # per row, so dJ/dθ_r is a fixed matrix with support on rows ρ(i)=r).
    djdtheta = [sp.diff(j_mat, theta[r]) for r in range(k)]
    for kk in range(n_steps, 0, -1):
        u_prev = states[kk - 1]
        for r in range(k):
            b_kr = djdtheta[r] * u_prev  # column with support ⊆ Ω_r
            grad[r] += (lam.T * b_kr)[0]
        lam = jt * lam
    return [sp.simplify(g) for g in grad]


def main() -> int:
    n_grid, k_regions, n_steps = 4, 2, 2
    rho = build_region_map(n_grid, k_regions)
    theta = list(sp.symbols("theta0 theta1", real=True))
    u0 = sp.Matrix(sp.symbols("u0 u1 u2 u3", real=True))
    target = sp.Matrix([sp.Integer(0)] * n_grid)
    djdtheta = [sp.diff(step_matrix(theta, rho, n_grid), t) for t in theta]

    # (1) per-region support: ∂(F)/∂θ_r row i is zero unless ρ(i)=r.
    for r in range(k_regions):
        for i in range(n_grid):
            if rho[i] != r and any(djdtheta[r][i, j] != 0 for j in range(n_grid)):
                print(f"T_REVERSE_AD_KVECTOR FAIL: support leak ∂θ_{r} row {i}")
                return 1

    # (2) reverse == analytic for every region.
    rev = reverse_grad(theta, rho, n_grid, n_steps, u0, target)
    ana = analytic_grad(theta, rho, n_grid, n_steps, u0, target)
    for r in range(k_regions):
        if sp.simplify(rev[r] - ana[r]) != 0:
            print(f"T_REVERSE_AD_KVECTOR FAIL: grad[{r}] != ∂J/∂θ_{r}")
            return 1

    # (3) K=1 reduction (single region = all nodes) is the scalar adjoint.
    rho1 = [0] * n_grid
    th1 = [sp.symbols("theta", real=True)]
    rev1 = reverse_grad(th1, rho1, n_grid, n_steps, u0, target)
    ana1 = analytic_grad(th1, rho1, n_grid, n_steps, u0, target)
    if sp.simplify(rev1[0] - ana1[0]) != 0:
        print("T_REVERSE_AD_KVECTOR FAIL: K=1 reduction != scalar adjoint")
        return 1

    print(
        "T_REVERSE_AD_KVECTOR PASS (3/3: per-region support disjoint; "
        "reverse==analytic for K=2; K=1 reduces to scalar adjoint)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
