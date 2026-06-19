# ADR-0122 — Adaptive per-point Gauss-Hermite quadrature `q` for the anisotropic/shift kernels (additive `.with_adaptive_q(tol)`)

- **Status**: Accepted (PRE-FLIGHT GO; engineer wave authorised — additive builder, no order change)
- **Date**: 2026-06-05
- **Decision-maker**: ai-solutions-architect
- **Backlog item**: v7.0.0 freeze item #21 (Phase 5). *Freeze pre-allocated ADR-0129; reconciled to the actual next-free sequential number 0122 — the Phase-4 cluster shipped under 0118-0121, leaving 0122 free.*
- **Depends on**: ADR-0081 / math.md §32 (`AnisotropicShiftChernoffND` fixed q=5 baseline), ADR-0112 (the corrected eq 32.3 the adaptive rule quadratures), constitution v2.0.0 (3/3 dep cap INVIOLATE — adaptive rule adds ZERO deps; reuses in-tree GH tables + the Golub-Welsch eigensolver in `gen_quadrature.rs`).
- **Mathematical foundation**: PRE-FLIGHT `scripts/verify_adaptive_quad.py` (executed 2026-06-05). Exactness + gate + monotone-upgrade + estimator-safety all PASS — see §PRE-FLIGHT.
- **Acceptance gate added**: `G_ADAPTIVE_Q` (RELEASE_BLOCKING) — adaptive result byte-matches the fixed-32 GH reference within `1e-10` on the kernel-envelope datum.

## Context

The v4.0/v6.0 d-D shift kernel (math.md §32.4 eq 32.3) and the v2.8 `manifold_chernoff.rs` fix `q=5` Gauss-Hermite per axis. The 5-pt rule is exact to per-axis degree 9, which over-resolves the low-degree majority of integrands the Chernoff product actually sees (in `(F(T/n))^n` the per-step shift `2√(T/n)·σ·η → 0`, so `g(η)=f(x_k+shift·η)` is near-linear) and under-resolves the rare sharply-curved point. An adaptive selector `q* = min{q∈{3,5,7,9} : |I_q − I_{q-2}| ≤ tol}` chooses the cheapest sufficient rule per evaluation point.

## Decision

Add an ADDITIVE opt-in builder `AnisotropicShiftChernoffND::with_adaptive_q(self, tol: F)` (and a sibling on the manifold kernel) that replaces the fixed q=5 tensor with the per-point selector above. **`order()` is UNCHANGED** (still 1) — adaptivity is a quadrature-accuracy refinement, not an order lift. The estimator tol MUST default to the production target (`1e-10`), **not** machine-epsilon: with tol=1e-12 the `|I_q−I_{q-2}|` estimator over-refines transcendental integrands to q=9 and the saving evaporates (PRE-FLIGHT-confirmed). Node/weight tables for q∈{1,3,5,7,9} are baked as in-tree `const` arrays (generated like `generate_chebyshev_nodes.py`), or computed once via the existing `gen_quadrature.rs` Golub-Welsch path — **no 4th dependency**.

## PRE-FLIGHT (executed 2026-06-05; `scripts/verify_adaptive_quad.py`)

- **Exactness**: q∈{3,5,7,9} integrate `∫ x^k e^{-x²} dx` exactly for `k ≤ 2q−1` (verified against `Γ((k+1)/2)`).
- **Gate (A)**: on the kernel-envelope battery (const → gaussian-IC at small shift s=0.3), adaptive @tol=1e-10 matches fixed-32 within `≤ 8.98e-13` (all PASS); mean `q* = 5.33` vs accuracy-equivalent fixed-9 → **41% fewer nodes/axis at equal accuracy**, 33% of points use q<5.
- **Monotone upgrade (B)**: q* is non-decreasing in curvature (`[7,9,9,9,9]` across c-sweep) and byte-matches the fixed-9 ceiling when q*=9.
- **Safety (C)**: on the smooth envelope the `|I_q−I_{q-2}|` estimator never silently under-resolves (true err-vs-fix9 ≤ 2.2e-16 everywhere).

## Consequences / engineer caveats

- The estimator tol is the production tolerance, NOT machine-epsilon (load-bearing — see PRE-FLIGHT). Default `tol = 1e-10`.
- Variable per-point q makes the per-step cost data-dependent; the SIMD hot path must handle a ragged node count (mask to the max q in a tile, or dispatch per q-class). The fixed-q path remains the default; adaptive is strictly opt-in.
- Gate datum is the smooth kernel-envelope class; sharply-curved integrands beyond the kernel's q=9 ceiling (e.g. `cos(2.5x)` against `e^{-x²}`) are OUT of envelope and correctly fall back to q=9 — the gate does NOT assert 1e-10 on them (no fixed rule reaches it).
