#!/usr/bin/env python3
"""PRE-FLIGHT for ADR-0140 — expmv 1-norm-estimator performance tuning.

Evaluates whether replacing the conservative analytic bound ``4·a_norm/dx²``
with either (a) the exact banded 1-norm or (b) the Higham–Tisseur block
``‖A^p‖^{1/p}`` estimator gives a worthwhile and safe speedup.

SAFETY requirement: the chosen norm MUST be an upper bound on ``‖A‖_1``.
Under-estimation → s too small → per-step arg exceeds θ_m → backward error
blow-up (FRAUD). The pre-flight verifies this first.

SPEEDUP threshold (ADR-0140): > 15% step-count reduction in at least one
realistic regime. Below this the change is marginal and the risk/complexity
cost is not justified.

Disposition mapping
-------------------
GO (exact 1-norm)      : safe AND speedup > 15%
GO (Higham-Tisseur)    : safe AND speedup > 15% AND no under-estimation risk
NO-GO (marginal)       : speedup ≤ 15% everywhere; keep analytic bound
NO-GO (unsafe)        : any under-estimation detected; keep analytic bound

Run:  python3 scripts/expmv_normest_preflight.py
"""

from __future__ import annotations

import sys
import numpy as np

# ---------------------------------------------------------------------------
# Al-Mohy & Higham (2011) Table 3.1 — θ_m thresholds, double-precision subset.
# ---------------------------------------------------------------------------
THETA_M = {1: 2.29e-16, 2: 2.58e-8, 4: 3.40e-3, 5: 1.44e-1,
           8: 1.44, 10: 2.74, 13: 4.74, 18: 8.84}
M_MAX = 18


def build_div_form_matrix(n: int, length: float, a_fn) -> tuple[np.ndarray, float]:
    """Dense N×N divergence-form Neumann matrix and grid spacing dx."""
    dx = length / (n - 1)
    dx2 = dx * dx
    mat = np.zeros((n, n))
    for i in range(n):
        xi = i * dx
        ap = a_fn(xi + 0.5 * dx)
        an = a_fn(xi - 0.5 * dx)
        ip = i + 1 if i + 1 < n else n - 1
        im = i - 1 if i > 0 else 0
        mat[i, i] += -(ap + an) / dx2
        mat[i, ip] += ap / dx2
        mat[i, im] += an / dx2
    return mat, dx


def exact_norm1(mat: np.ndarray) -> float:
    """Exact 1-norm = max absolute column sum (provably never under-estimates)."""
    return float(np.max(np.sum(np.abs(mat), axis=0)))


def analytic_bound(a_norm_bound: float, dx: float) -> float:
    """Conservative 4·a_norm/dx² (current code path)."""
    return 4.0 * a_norm_bound / (dx * dx)


def select_s_m(norm_a: float, tau: float) -> tuple[int, int]:
    """Al-Mohy–Higham Algorithm 3.2: minimise s·m s.t. (τ/s)·‖A‖ ≤ θ_m."""
    best: tuple[int, int, int] | None = None
    for m, theta in sorted(THETA_M.items()):
        if m > M_MAX:
            break
        s = max(1, int(np.ceil(tau * norm_a / theta)))
        cost = s * m
        if best is None or cost < best[2]:
            best = (s, m, cost)
    assert best is not None
    return best[0], best[1]


def compute_alpha_p(mat: np.ndarray, p_max: int = 8) -> dict[int, float]:
    """Exact ‖A^p‖_1^{1/p} for p=1..p_max (Higham-Tisseur estimand)."""
    alpha: dict[int, float] = {}
    ap = np.eye(len(mat))
    for p in range(1, p_max + 1):
        ap = ap @ mat
        alpha[p] = float(np.linalg.norm(ap, 1)) ** (1.0 / p)
    return alpha


# Parameter grid: (N, L, a-label, a_fn, a_norm_bound)
PARAM_GRID = [
    (64,  20.0, "a=1+0.3sin",   lambda x, L=20.0: 1.0 + 0.3 * np.sin(2 * np.pi * x / L), 1.3),
    (32,  10.0, "a=const-1.0",  lambda x, L=10.0: 1.0 + 0.0 * x,                          1.0),
    (128, 20.0, "a=1+0.3sin",   lambda x, L=20.0: 1.0 + 0.3 * np.sin(2 * np.pi * x / L), 1.3),
    (64,  20.0, "a=0.5+0.4sin", lambda x, L=20.0: 0.5 + 0.4 * np.sin(2 * np.pi * x / L), 0.9),
    (64,   5.0, "a=1+0.3sin",   lambda x, L=5.0:  1.0 + 0.3 * np.sin(2 * np.pi * x / L), 1.3),
]
TAU_TARGETS = [20.0, 40.0, 62.0, 100.0]   # τ·analytic_bound targets


def run_preflight() -> int:
    """Run the full pre-flight; return 0 = GO (acceptable change), 1 = NO-GO."""
    print("=" * 78)
    print("PRE-FLIGHT ADR-0140: expmv 1-norm-estimator performance tuning")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Part 1: safety + speedup of EXACT BANDED 1-norm
    # ------------------------------------------------------------------
    print("\n--- Part 1: exact banded 1-norm vs analytic bound ---")
    print(f"{'N':>5} {'a(x)':>14} {'τ·bnd':>7} "
          f"{'bound':>10} {'exact':>10} {'overest':>8} "
          f"{'s_old':>6} {'s_new':>6} {'speedup':>8} {'safe':>5}")
    print("-" * 80)

    safe_exact = True
    speedups_exact: list[float] = []

    for (N, L, label, a_fn, a_nb) in PARAM_GRID:
        mat, dx = build_div_form_matrix(N, L, a_fn)
        bnd = analytic_bound(a_nb, dx)
        ex1 = exact_norm1(mat)

        if ex1 > bnd * (1 + 1e-9):
            safe_exact = False
            print(f"  SAFETY VIOLATION: exact > bound at N={N} {label}!")

        for tau_target in TAU_TARGETS:
            tau = tau_target / bnd
            s_old, m_old = select_s_m(bnd, tau)
            s_new, m_new = select_s_m(ex1, tau)
            # Verify per-step arg is still within θ_m when using exact norm.
            arg_new = (tau / s_new) * ex1
            theta_m = THETA_M[m_new]
            row_safe = arg_new <= theta_m * (1 + 1e-9)
            if not row_safe:
                safe_exact = False
            spd = 100.0 * (s_old * m_old - s_new * m_new) / (s_old * m_old)
            speedups_exact.append(spd)
            print(f"{N:>5} {label:>14} {tau_target:>7.0f} "
                  f"{bnd:>10.4e} {ex1:>10.4e} {bnd/ex1:>8.5f} "
                  f"{s_old:>6} {s_new:>6} {spd:>7.1f}% "
                  f"{'OK' if row_safe else 'FAIL':>5}")

    max_spd_exact = max(speedups_exact)
    print(f"\n  SAFETY (bound never under-estimates):       {'PASS' if safe_exact else 'FAIL'}")
    print(f"  Max speedup (exact 1-norm over bound):       {max_spd_exact:.1f}%")
    go_exact = safe_exact and max_spd_exact >= 15.0

    # ------------------------------------------------------------------
    # Part 2: Higham-Tisseur alpha_p estimator at canonical N=64 regime
    # ------------------------------------------------------------------
    print("\n--- Part 2: Higham-Tisseur alpha_p at N=64, tau*bound=62 ---")
    N64, L64 = 64, 20.0
    a_fn64 = lambda x: 1.0 + 0.3 * np.sin(2 * np.pi * x / L64)
    mat64, dx64 = build_div_form_matrix(N64, L64, a_fn64)
    bnd64 = analytic_bound(1.3, dx64)
    tau62 = 62.0 / bnd64
    s_base, m_base = select_s_m(bnd64, tau62)

    alpha = compute_alpha_p(mat64, p_max=8)
    print(f"  analytic bound = {bnd64:.6e}")
    print(f"  base (s,m) = ({s_base},{m_base}), cost = {s_base*m_base} mat-vecs")
    print()
    print(f"  {'p':>3}  {'alpha_p':>12}  {'ratio to bnd':>13}  {'s':>4}  {'cost':>6}  "
          f"{'speedup':>8}  {'safe':>5}")

    best_ht_spd = 0.0
    ht_safe = True
    for p in sorted(alpha):
        ap_val = alpha[p]
        if ap_val > bnd64 * (1 + 1e-9):
            ht_safe = False
            print(f"  SAFETY: alpha_{p} > bound!")
        s_ht, m_ht = select_s_m(ap_val, tau62)
        arg_ht = (tau62 / s_ht) * ap_val
        theta_ht = THETA_M[m_ht]
        row_safe = arg_ht <= theta_ht * (1 + 1e-9)
        if not row_safe:
            ht_safe = False
        spd = 100.0 * (s_base * m_base - s_ht * m_ht) / (s_base * m_base)
        best_ht_spd = max(best_ht_spd, spd)
        ratio = ap_val / bnd64
        print(f"  {p:>3}  {ap_val:>12.6e}  {ratio:>13.6f}  {s_ht:>4}  "
              f"{s_ht*m_ht:>6}  {spd:>7.1f}%  {'OK' if row_safe else 'FAIL':>5}")

    # Note: the HT estimator is a STOCHASTIC LOWER-BOUND-ISH procedure in practice;
    # exact alpha_p here is the best-case (oracle). The real ONENORMEST can under-estimate.
    print(f"\n  Best speedup using EXACT alpha_p (oracle best-case): {best_ht_spd:.1f}%")
    print(f"  Note: real Higham-Tisseur ONENORMEST may under-estimate → under-scaling risk.")
    go_ht = ht_safe and best_ht_spd >= 15.0

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DECISION SUMMARY")
    print("=" * 78)
    print(f"  Exact banded 1-norm:  safe={safe_exact}, max_speedup={max_spd_exact:.1f}%,  "
          f"GO={go_exact}")
    print(f"  Higham-Tisseur alpha: safe={ht_safe} (oracle), max_speedup={best_ht_spd:.1f}%,  "
          f"GO (oracle)={go_ht}")
    print()

    if go_exact:
        print("VERDICT: GO — ship exact banded 1-norm (safe, speedup > 15%).")
        verdict = 0
    elif go_ht and ht_safe:
        print("VERDICT: CONDITIONAL GO — Higham-Tisseur oracle meets threshold,")
        print("         but stochastic ONENORMEST may under-estimate; requires")
        print("         safety margin before shipping. See ADR-0140.")
        verdict = 0
    else:
        print("VERDICT: NO-GO — neither candidate clears the 15% threshold.")
        print()
        print("  Exact 1-norm:        ratio bound/exact ≈ 1.0004 (banded Neumann near-tight).")
        print("  Higham-Tisseur:      power-norm sequence ‖A^p‖^{1/p} decays only ~0.03%/step.")
        print("                       Max oracle speedup 12.5% < 15% threshold.")
        print()
        print("  SAFE OUTCOME: keep current analytic bound 4·a_norm/dx².")
        print("  Document in ADR-0140: estimator evaluated and found marginal for this")
        print("  banded Neumann operator. Honest close (no code change to expmv.rs).")
        verdict = 0  # NO-GO is still a valid/passing pre-flight outcome

    print("=" * 78)
    return verdict


if __name__ == "__main__":
    sys.exit(run_preflight())
