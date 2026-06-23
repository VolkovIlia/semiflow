#!/usr/bin/env python3
# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false, reportIndexIssue=false
#
# numpy ndarray operator overloads are opaque to Pyright; all ops are valid at
# runtime (verified by this oracle's PASS line).
"""G_GRAPH_ADJOINT_SAMPLED_PARITY oracle (ADR-0180).

PRE-FLIGHT, language-independent numeric oracle. Proves — BEFORE any Rust
`from_presampled` kernel is written — that replaying a PRE-SAMPLED Laplacian
weight sequence over the KNOWN GL4 time grid computes the SAME backward
state-adjoint costate as evaluating the time-dependent closure `L_G(t)` per step.

It encodes, in pure numpy, the exact discrete map the core implements
(`magnus_graph_adjoint.rs`, math.md §42.1):

    A_i(t_start) = −L_G(t_start + c_i·τ),   c1=(3−√3)/6, c2=(3+√3)/6
    Ω₄ᵀ(τ)      = (τ/2)(A₁+A₂) − (√3·τ²/12)·[A₂,A₁]     # adjoint: comm sign −1
    S⋆(τ)       = Σ_{m=0..4} (Ω₄ᵀ)^m / m!               # degree-4 Taylor
    λ_0 = S⋆_0 · S⋆_1 ⋯ S⋆_{n−1} · λ_n,  step k uses t_start=(n−1−k)·τ

THE CLAIM (ADR-0180): because the sample times T = {jτ+c1τ, jτ+c2τ} are known at
construction and the topology is FIXED (only edge weights vary with t), the
pre-sampled path that stores L_G at those 2·n_steps abscissae and the closure
path that calls L_G(t) per step feed BIT-IDENTICAL operators into S⋆ — hence
λ_0 is identical to 0 ULP. This oracle asserts that equality.

The "Magnus needs SUB-STEP samples" honesty point (ADR-0180 §scope-narrowing) is
made operational here: the grid has 2·n_steps points (one per (step, abscissa)),
NOT n_steps. A deliberate `WRONG` variant that samples only on the step grid is
shown to FAIL, proving the GL4-aware layout is necessary.

Run: python3 scripts/verify_graphadjoint_sampled.py
"""

import math

import numpy as np

# GL4 abscissae (NORMATIVE, magnus_graph.rs GL4_C1_F64 / GL4_C2_F64).
C1 = (3.0 - math.sqrt(3.0)) / 6.0
C2 = (3.0 + math.sqrt(3.0)) / 6.0
SQRT3_OVER_12 = math.sqrt(3.0) / 12.0


def laplacian_path(n: int, w: np.ndarray) -> np.ndarray:
    """Combinatorial L = D − W for a PATH graph on n nodes with edge weights
    w[k] on edge (k, k+1). Fixed topology; only `w` varies with t."""
    lap = np.zeros((n, n))
    for k in range(n - 1):
        wk = w[k]
        lap[k, k] += wk
        lap[k + 1, k + 1] += wk
        lap[k, k + 1] -= wk
        lap[k + 1, k] -= wk
    return lap


def edge_weights_at(t: float, n_edges: int) -> np.ndarray:
    """Time-varying-but-fixed-topology weights: w_k(t) = 1 + 0.5·sin(t + 0.1k)."""
    return np.array([1.0 + 0.5 * math.sin(t + 0.1 * k) for k in range(n_edges)])


def s_star(lap1: np.ndarray, lap2: np.ndarray, tau: float) -> np.ndarray:
    """Adjoint degree-4 Taylor map S⋆(τ) = Σ_{m=0..4} (Ω₄ᵀ)^m/m! (comm sign −1)."""
    a1 = -lap1
    a2 = -lap2
    comm = a2 @ a1 - a1 @ a2          # [A2, A1]
    omega_t = 0.5 * tau * (a1 + a2) - SQRT3_OVER_12 * tau * tau * comm
    n = lap1.shape[0]
    acc = np.eye(n)
    term = np.eye(n)
    for m in range(1, 5):
        term = term @ omega_t / m
        acc = acc + term
    return acc


def sweep_closure(n: int, n_steps: int, tau: float, lam_n: np.ndarray) -> np.ndarray:
    """Backward costate sweep evaluating L_G(t) PER STEP (the existing path)."""
    n_edges = n - 1
    lam = lam_n.copy()
    for k in range(n_steps):
        t_start = (n_steps - 1 - k) * tau
        lap1 = laplacian_path(n, edge_weights_at(t_start + C1 * tau, n_edges))
        lap2 = laplacian_path(n, edge_weights_at(t_start + C2 * tau, n_edges))
        lam = s_star(lap1, lap2, tau) @ lam
    return lam


def presample_vals_seq(n: int, n_steps: int, tau: float) -> np.ndarray:
    """Host-side: sample edge weights ONCE on the 2·n_steps GL4 grid, in
    schedule order [(0,c1),(0,c2),(1,c1),(1,c2),…]. This is `vals_seq`."""
    n_edges = n - 1
    seq = np.empty((2 * n_steps, n_edges))
    for k in range(n_steps):
        t_start = (n_steps - 1 - k) * tau   # SAME schedule the sweep uses
        seq[2 * k] = edge_weights_at(t_start + C1 * tau, n_edges)
        seq[2 * k + 1] = edge_weights_at(t_start + C2 * tau, n_edges)
    return seq


def sweep_presampled(n: int, n_steps: int, tau: float,
                     lam_n: np.ndarray, vals_seq: np.ndarray) -> np.ndarray:
    """Backward sweep REPLAYING the pre-sampled sequence (no per-step sampling)."""
    lam = lam_n.copy()
    for k in range(n_steps):
        lap1 = laplacian_path(n, vals_seq[2 * k])
        lap2 = laplacian_path(n, vals_seq[2 * k + 1])
        lam = s_star(lap1, lap2, tau) @ lam
    return lam


def sweep_presampled_WRONG(n: int, n_steps: int, tau: float,
                           lam_n: np.ndarray) -> np.ndarray:
    """COUNTER-EXAMPLE: sample only on the n_steps STEP grid (one Laplacian per
    step, both abscissae use the step-boundary weights). ADR-0180 warns this is
    silently wrong at O(τ²) via the commutator. Must DIFFER from the closure."""
    n_edges = n - 1
    lam = lam_n.copy()
    for k in range(n_steps):
        t_start = (n_steps - 1 - k) * tau
        lap = laplacian_path(n, edge_weights_at(t_start, n_edges))  # WRONG: no c_i
        lam = s_star(lap, lap, tau) @ lam
    return lam


def main() -> int:
    n = 8
    n_steps = 64
    t_horizon = 0.5
    tau = t_horizon / n_steps
    rng = np.random.default_rng(0xC0FFEE)
    lam_n = rng.standard_normal(n)

    lam0_closure = sweep_closure(n, n_steps, tau, lam_n)

    vals_seq = presample_vals_seq(n, n_steps, tau)
    assert vals_seq.size == 2 * n_steps * (n - 1), "GL4-aware length check"
    lam0_sampled = sweep_presampled(n, n_steps, tau, lam_n, vals_seq)

    lam0_wrong = sweep_presampled_WRONG(n, n_steps, tau, lam_n)

    # (1) Pre-sampled GL4-aware path == closure path to 0 ULP (bit-exact).
    bitexact = np.array_equal(lam0_sampled, lam0_closure)
    max_ulp_err = float(np.max(np.abs(lam0_sampled - lam0_closure)))

    # (2) Step-grid-only path DIFFERS (proves sub-step sampling is necessary).
    wrong_diff = float(np.max(np.abs(lam0_wrong - lam0_closure)))

    print(f"  n={n} n_steps={n_steps} t_horizon={t_horizon} grid_pts={2*n_steps}")
    print(f"  presampled(GL4-aware) vs closure : max|Δ| = {max_ulp_err:.3e}  bit_exact={bitexact}")
    print(f"  step-grid-only (WRONG) vs closure: max|Δ| = {wrong_diff:.3e}  (must be > 0)")

    ok = bitexact and (wrong_diff > 1e-6)
    if ok:
        print("PASS: G_GRAPH_ADJOINT_SAMPLED_PARITY oracle — "
              "pre-sampled GL4-aware sequence reproduces closure costate to 0 ULP; "
              "step-grid-only sampling correctly fails.")
        return 0
    print("FAIL: parity broken OR wrong-variant did not diverge.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
